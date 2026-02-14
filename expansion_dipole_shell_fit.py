#!/usr/bin/env python3
"""
Measure the expansion dipole in radial shells, and decompose it into
components parallel and perpendicular to the structural dipole.

Inputs expected in current directory:
  - dipole_vectors.npy        (N_snap, 3)
  - bulk_flow_vectors.npy     (N_snap, 3)

TNG300 groupcats are expected under:
  /mnt/g/TNG300-1/groupcat_XXX

Outputs:
  - expansion_shell_dipole_results.npz
"""

import os
import glob
import math
import numpy as np
import h5py

# -----------------------------
# Configuration
# -----------------------------

BASE_PATH = "/mnt/g/TNG300-1"
BOX_SIZE = 205.0  # Mpc/h, TNG300 box size

# Cosmology (same as earlier scripts)
H0 = 67.74          # km/s/Mpc
h_param = 0.6774
Omega_m = 0.3089
Omega_L = 0.6911

# Same snapshots as in dipole_time_series / bulk_flow_time_series
SNAPSHOTS = [33, 40, 50, 59, 67, 72, 78, 91]

# Approximate redshifts for those snapshots
SNAP_TO_Z = {
    33: 2.0,
    40: 1.5,
    50: 1.0,
    59: 0.8,
    67: 0.6,
    72: 0.5,
    78: 0.4,
    91: 0.2,
}

# Radial shells in comoving Mpc/h (relative to box centre)
# Feel free to tweak these
RADIAL_BINS = [
    (20.0, 80.0),
    (80.0, 140.0),
    (140.0, 205.0),
]

# Mass cut in 1e10 Msun/h units (None means use all subhalos)
MASS_CUT = None  # e.g. set to 1.0 for >1e10 Msun/h


# -----------------------------
# Utilities
# -----------------------------

def frw_H_comov(z):
    """
    Hubble rate in km/s/(Mpc/h) for comoving units,
    consistent with your previous expansion_dipole_time_series.py.
    """
    Ez = math.sqrt(Omega_m * (1.0 + z)**3 + Omega_L)
    H_phys = H0 * Ez                  # km/s/Mpc
    H_comov = H_phys / ((1.0 + z) * h_param)  # km/s/(Mpc/h)
    return H_comov


def find_groupcat_folder(base_path, snap):
    """
    Try a few sensible folder names for a given snapshot number.
    """
    candidates = [
        os.path.join(base_path, f"groupcat_{snap:03d}"),
        os.path.join(base_path, f"groupcat_{snap}"),
        os.path.join(base_path, f"groups_{snap:03d}"),
        os.path.join(base_path, f"groups_{snap}"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    raise RuntimeError(
        f"No groupcat folder found for snapshot {snap}. "
        f"Tried:\n" + "\n".join(candidates)
    )


def load_subhalos(snap, mass_cut=MASS_CUT):
    """
    Load SubhaloPos, SubhaloVel, SubhaloMass from all fof_subhalo_tab files
    for a given snapshot. Returns pos (Mpc/h), vel (km/s), mass (1e10 Msun/h).
    """
    folder = find_groupcat_folder(BASE_PATH, snap)
    print(f"  Using folder: {folder}")

    pattern = os.path.join(folder, f"fof_subhalo_tab_{snap:03d}.*.hdf5")
    files = sorted(glob.glob(pattern))
    if not files:
        raise RuntimeError(f"No HDF5 files found in {folder}")

    pos_list = []
    vel_list = []
    mass_list = []

    for fname in files:
        with h5py.File(fname, "r") as f:
            sub = f["Subhalo"]
            pos_list.append(sub["SubhaloPos"][:])   # ckpc/h
            vel_list.append(sub["SubhaloVel"][:])   # km/s peculiar
            mass_list.append(sub["SubhaloMass"][:]) # 1e10 Msun/h

    pos = np.concatenate(pos_list, axis=0)  # ckpc/h
    vel = np.concatenate(vel_list, axis=0)  # km/s
    mass = np.concatenate(mass_list, axis=0)  # 1e10 Msun/h

    # Convert positions to comoving Mpc/h
    pos = pos / 1000.0  # now in cMpc/h

    if mass_cut is not None:
        mask = mass >= mass_cut
        pos = pos[mask]
        vel = vel[mask]
        mass = mass[mask]

    print(f"  Total halos used: {pos.shape[0]}")
    return pos, vel, mass


def fit_dipole(deltaH, n_hat, weight=None):
    """
    Fit deltaH(n) ~ a + b · n using weighted least squares.

    deltaH: (N,)
    n_hat: (N,3) unit vectors
    weight: (N,) or None
    Returns: a (scalar), b (3-vector)
    """
    deltaH = np.asarray(deltaH)
    n_hat = np.asarray(n_hat)
    N = deltaH.shape[0]
    if N < 4:
        return np.nan, np.array([np.nan, np.nan, np.nan])

    if weight is None:
        w = np.ones(N, dtype=np.float64)
    else:
        w = np.asarray(weight, dtype=np.float64)

    # Design matrix: [1, n_x, n_y, n_z]
    X = np.column_stack([np.ones(N, dtype=np.float64), n_hat])
    WX = X * w[:, None]

    M = X.T @ WX              # 4x4
    rhs = WX.T @ deltaH       # 4

    try:
        beta = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        return np.nan, np.array([np.nan, np.nan, np.nan])

    a = beta[0]
    b = beta[1:]
    return a, b


# -----------------------------
# Main per-snapshot computation
# -----------------------------

def compute_shell_dipoles_for_snapshot(
    snap,
    dipole_vecs,
    bulk_flow_vecs,
    radial_bins=RADIAL_BINS,
    mass_cut=MASS_CUT,
):
    """
    For one snapshot:
      - load subhalos
      - subtract bulk flow
      - compute deltaH_i
      - fit dipole in each radial shell

    Returns:
      dict with arrays per shell.
    """
    if snap not in SNAP_TO_Z:
        raise RuntimeError(f"No redshift entry for snapshot {snap}")

    z = SNAP_TO_Z[snap]
    H_frw = frw_H_comov(z)

    # Find index for this snapshot in the dipole/bulk arrays
    idx = SNAPSHOTS.index(snap)
    struct_vec = dipole_vecs[idx]
    bulk_vec = bulk_flow_vecs[idx]

    # Structural dipole unit vector
    norm_struct = np.linalg.norm(struct_vec)
    if norm_struct == 0.0:
        raise RuntimeError(f"Structural dipole vector is zero for snap {snap}")
    d_hat = struct_vec / norm_struct

    pos, vel, mass = load_subhalos(snap, mass_cut=mass_cut)

    # Positions relative to box center, in Mpc/h
    center = np.array([BOX_SIZE / 2.0] * 3, dtype=np.float64)
    r_vec = pos - center
    r_mag = np.linalg.norm(r_vec, axis=1)

    # Unit directions
    eps = 1e-12
    r_mag_safe = np.where(r_mag == 0.0, eps, r_mag)
    n_hat = r_vec / r_mag_safe[:, None]

    # Subtract bulk flow
    vel_corr = vel - bulk_vec[None, :]

    # Radial velocity
    v_r = np.sum(vel_corr * n_hat, axis=1)  # km/s

    # H estimator per halo
    H_i = v_r / r_mag_safe  # km/s/(Mpc/h)

    # deltaH per halo
    deltaH = H_i - H_frw

    # Containers
    n_bins = len(radial_bins)
    mono = np.full(n_bins, np.nan, dtype=np.float64)
    mono_over_H = np.full(n_bins, np.nan, dtype=np.float64)
    dip_amp = np.full(n_bins, np.nan, dtype=np.float64)
    dip_amp_over_H = np.full(n_bins, np.nan, dtype=np.float64)
    dip_par = np.full(n_bins, np.nan, dtype=np.float64)
    dip_par_over_H = np.full(n_bins, np.nan, dtype=np.float64)
    dip_perp = np.full(n_bins, np.nan, dtype=np.float64)
    dip_perp_over_H = np.full(n_bins, np.nan, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)

    print(f"\nProcessing snapshot {snap} (z ~ {z:.2f})")
    print(f"  H_FRW_comov ~ {H_frw:.3f} km/s/(Mpc/h)")

    for ibin, (rmin, rmax) in enumerate(radial_bins):
        shell_mask = (r_mag >= rmin) & (r_mag < rmax)
        N_shell = np.count_nonzero(shell_mask)
        counts[ibin] = N_shell

        print(f"  Shell {ibin}: r in [{rmin:.1f}, {rmax:.1f}] Mpc/h, N = {N_shell}")

        if N_shell < 1000:
            print("    Too few halos, skipping fit.")
            continue

        deltaH_shell = deltaH[shell_mask]
        n_hat_shell = n_hat[shell_mask]
        mass_shell = mass[shell_mask]

        a, b = fit_dipole(deltaH_shell, n_hat_shell, weight=mass_shell)
        if np.isnan(a):
            print("    Fit failed, skipping.")
            continue

        b_amp = np.linalg.norm(b)
        b_par = np.dot(b, d_hat)
        b_perp = math.sqrt(max(b_amp**2 - b_par**2, 0.0))

        mono[ibin] = a
        mono_over_H[ibin] = a / H_frw
        dip_amp[ibin] = b_amp
        dip_amp_over_H[ibin] = b_amp / H_frw
        dip_par[ibin] = b_par
        dip_par_over_H[ibin] = b_par / H_frw
        dip_perp[ibin] = b_perp
        dip_perp_over_H[ibin] = b_perp / H_frw

        print(f"    Monopole shift a/H = {a / H_frw:+.3e}")
        print(f"    |b|/H             = {b_amp / H_frw:+.3e}")
        print(f"    b_parallel/H      = {b_par / H_frw:+.3e}")
        print(f"    b_perp/H          = {b_perp / H_frw:+.3e}")

    return {
        "snapshot": snap,
        "z": z,
        "H_frw": H_frw,
        "counts": counts,
        "mono": mono,
        "mono_over_H": mono_over_H,
        "dip_amp": dip_amp,
        "dip_amp_over_H": dip_amp_over_H,
        "dip_par": dip_par,
        "dip_par_over_H": dip_par_over_H,
        "dip_perp": dip_perp,
        "dip_perp_over_H": dip_perp_over_H,
    }


# -----------------------------
# Main
# -----------------------------

def main():
    print(f"BASE_PATH = {BASE_PATH}")

    # Load structural dipole vectors
    if not os.path.isfile("dipole_vectors.npy"):
        raise RuntimeError("dipole_vectors.npy not found in current directory.")
    dipole_vecs = np.load("dipole_vectors.npy")
    if dipole_vecs.shape != (len(SNAPSHOTS), 3):
        raise RuntimeError(
            f"dipole_vectors.npy has shape {dipole_vecs.shape}, "
            f"expected ({len(SNAPSHOTS)}, 3)."
        )

    # Load bulk flow vectors
    if os.path.isfile("bulk_flow_vectors.npy"):
        print("Found bulk_flow_vectors.npy, will subtract bulk flow.")
        bulk_flow_vecs = np.load("bulk_flow_vectors.npy")
        if bulk_flow_vecs.shape != (len(SNAPSHOTS), 3):
            raise RuntimeError(
                f"bulk_flow_vectors.npy has shape {bulk_flow_vecs.shape}, "
                f"expected ({len(SNAPSHOTS)}, 3)."
            )
    else:
        print("bulk_flow_vectors.npy not found, no bulk flow subtraction.")
        bulk_flow_vecs = np.zeros_like(dipole_vecs)

    n_snap = len(SNAPSHOTS)
    n_bins = len(RADIAL_BINS)

    snaps_arr = np.zeros(n_snap, dtype=int)
    z_arr = np.zeros(n_snap, dtype=float)

    mono_all = np.zeros((n_snap, n_bins))
    monoH_all = np.zeros((n_snap, n_bins))
    amp_all = np.zeros((n_snap, n_bins))
    ampH_all = np.zeros((n_snap, n_bins))
    par_all = np.zeros((n_snap, n_bins))
    parH_all = np.zeros((n_snap, n_bins))
    perp_all = np.zeros((n_snap, n_bins))
    perpH_all = np.zeros((n_snap, n_bins))
    counts_all = np.zeros((n_snap, n_bins), dtype=int)

    for i, snap in enumerate(SNAPSHOTS):
        res = compute_shell_dipoles_for_snapshot(
            snap,
            dipole_vecs=dipole_vecs,
            bulk_flow_vecs=bulk_flow_vecs,
            radial_bins=RADIAL_BINS,
            mass_cut=MASS_CUT,
        )

        snaps_arr[i] = res["snapshot"]
        z_arr[i] = res["z"]
        counts_all[i, :] = res["counts"]
        mono_all[i, :] = res["mono"]
        monoH_all[i, :] = res["mono_over_H"]
        amp_all[i, :] = res["dip_amp"]
        ampH_all[i, :] = res["dip_amp_over_H"]
        par_all[i, :] = res["dip_par"]
        parH_all[i, :] = res["dip_par_over_H"]
        perp_all[i, :] = res["dip_perp"]
        perpH_all[i, :] = res["dip_perp_over_H"]

    out_file = "expansion_shell_dipole_results.npz"
    np.savez(
        out_file,
        snapshots=snaps_arr,
        redshifts=z_arr,
        radial_bins=np.array(RADIAL_BINS),
        counts=counts_all,
        mono=mono_all,
        mono_over_H=monoH_all,
        dip_amp=amp_all,
        dip_amp_over_H=ampH_all,
        dip_par=par_all,
        dip_par_over_H=parH_all,
        dip_perp=perp_all,
        dip_perp_over_H=perpH_all,
    )

    print("\nSaved shell dipole results to:", out_file)
    print("\nSummary (outermost shell only, rough glance):")
    last_bin = n_bins - 1
    for i, snap in enumerate(SNAPSHOTS):
        print(
            f"  snap {snap:3d}, z={z_arr[i]:.2f} : "
            f"|b|/H = {ampH_all[i, last_bin]:+7.3e}, "
            f"b_parallel/H = {parH_all[i, last_bin]:+7.3e}, "
            f"N_shell = {counts_all[i, last_bin]}"
        )


if __name__ == "__main__":
    main()
