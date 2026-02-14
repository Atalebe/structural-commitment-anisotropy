#!/usr/bin/env python
"""
expansion_dipole_time_series.py

Directional expansion proxy along the structural dipole in TNG300.

For each snapshot in SNAPSHOTS, this script:
  - loads subhalo positions, velocities, masses from TNG300-1 groupcats,
  - subtracts the mass-weighted bulk flow (from bulk_flow_vectors.npy),
  - computes radial velocities with respect to the box centre,
  - fits a global slope deltaH_all from v_rad = deltaH * r,
  - defines hemispheres aligned with the structural dipole direction,
  - fits deltaH_N and deltaH_S for the two hemispheres,
  - builds a dipole-like expansion amplitude DeltaH = 0.5 (deltaH_N - deltaH_S),
  - normalises everything by an approximate FRW H_comov(z).

Outputs:
  - expansion_dipole_results.npz
  - prints a table of results to stdout.

Units:
  - Positions: comoving kpc/h from group catalogues.
  - Velocities: km/s.
  - r is converted to comoving Mpc/h.
  - deltaH_* are in km/s per (Mpc/h).
  - H_comov(z) is in km/s per (Mpc/h), so deltaH/H is dimensionless.
"""

import os
import numpy as np
import h5py

# -------------------------
# Configuration
# -------------------------

# TNG300-1 location (adjust if needed)
BASE_PATH = "/mnt/g/TNG300-1"

# Simulation box size in comoving Mpc/h
BOX_SIZE = 205.0

# Snapshots we have already used for the structural dipole
SNAPSHOTS = [33, 40, 50, 59, 67, 72, 78, 91]

# Approximate redshifts for these snapshots
# (coarse but good enough for H_comov normalisation)
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

# Cosmology roughly matching TNG (Planck-like)
H0 = 67.74      # km/s/Mpc
h  = 0.6774
OMEGA_M = 0.3089
OMEGA_L = 0.6911

# Minimum halo mass in 1e10 Msun/h units (set >0 to cut low-mass halos)
MASS_MIN = 0.0

# Minimum radius from box centre in kpc/h to avoid r~0
R_MIN_KPC_H = 1.0


# -------------------------
# Helpers
# -------------------------

def H_comov(z):
    """
    FRW Hubble factor in units km/s per (Mpc/h) for comoving coordinates.

    H_phys(a) = H0 * sqrt(Om (1+z)^3 + Ol)
    H_comov   = H_phys(a) * a / h
              = (H0/h) * sqrt(Om (1+z)^3 + Ol) / (1+z)

    This matches our use of r in comoving Mpc/h.
    """
    Ez = np.sqrt(OMEGA_M * (1.0 + z)**3 + OMEGA_L)
    return (H0 / h) * Ez / (1.0 + z)


def find_groupcat_folder(base_path, snap):
    """
    Find the group catalog folder for a given snapshot.
    Tries a few sensible naming patterns.
    """
    candidates = [
        os.path.join(base_path, f"groupcat_{snap:03d}"),
        os.path.join(base_path, f"groupcat_{snap}"),
        os.path.join(base_path, f"groups_{snap:03d}"),
        os.path.join(base_path, f"groups_{snap}"),
    ]
    for folder in candidates:
        if os.path.isdir(folder):
            return folder
    raise RuntimeError(
        f"No groupcat folder found for snapshot {snap}. Tried:\n" +
        "\n".join(candidates)
    )


def load_subhalos(folder, mass_min=MASS_MIN):
    """
    Load SubhaloPos, SubhaloVel, SubhaloMass from all fof_subhalo_tab_*.hdf5 files
    in the given folder.

    Returns:
      pos  : (N,3) comoving kpc/h
      vel  : (N,3) km/s
      mass : (N,)  in 1e10 Msun/h
    """
    files = sorted(
        f for f in os.listdir(folder)
        if f.startswith("fof_subhalo_tab") and f.endswith(".hdf5")
    )
    if not files:
        raise RuntimeError(f"No HDF5 files found in {folder}")

    pos_list = []
    vel_list = []
    mass_list = []

    for fname in files:
        path = os.path.join(folder, fname)
        with h5py.File(path, "r") as h:
            sub = h["Subhalo"]
            pos = sub["SubhaloPos"][:]   # kpc/h
            vel = sub["SubhaloVel"][:]   # km/s
            mass = sub["SubhaloMass"][:] # 1e10 Msun/h

            if mass_min > 0.0:
                mask = mass > mass_min
                if not np.any(mask):
                    continue
                pos = pos[mask]
                vel = vel[mask]
                mass = mass[mask]

            pos_list.append(pos)
            vel_list.append(vel)
            mass_list.append(mass)

    pos = np.concatenate(pos_list, axis=0)
    vel = np.concatenate(vel_list, axis=0)
    mass = np.concatenate(mass_list, axis=0)

    return pos, vel, mass


def compute_radial_data(pos, vel, v_bulk=None):
    """
    Compute radial distances and radial velocities relative to the box centre.

    Inputs:
      pos   : (N,3) kpc/h
      vel   : (N,3) km/s
      v_bulk: (3,) km/s, bulk flow vector to subtract (optional)

    Returns:
      r_mpc_h : (N,) comoving Mpc/h
      v_rad   : (N,) radial velocity in km/s (after bulk subtraction if given)
      n_hat   : (N,3) unit vectors from centre to halo
    """
    center_kpc_h = np.array([BOX_SIZE / 2.0] * 3) * 1000.0  # kpc/h
    x = pos - center_kpc_h

    r_kpc_h = np.linalg.norm(x, axis=1)
    mask = r_kpc_h > R_MIN_KPC_H

    x = x[mask]
    vel = vel[mask]
    r_kpc_h = r_kpc_h[mask]

    r_mpc_h = r_kpc_h / 1000.0

    n_hat = x / r_kpc_h[:, None]

    if v_bulk is not None:
        vel_corr = vel - v_bulk[None, :]
    else:
        vel_corr = vel

    v_rad = np.sum(vel_corr * n_hat, axis=1)

    return r_mpc_h, v_rad, n_hat, mask


def fit_deltaH(r_mpc_h, v_rad, mass=None):
    """
    Fit v_rad = deltaH * r_mpc_h via weighted least squares with zero intercept.

    Inputs:
      r_mpc_h : (N,)
      v_rad   : (N,)
      mass    : (N,) or None, weights (defaults to 1)

    Returns:
      deltaH in km/s per (Mpc/h).
    """
    if mass is None:
        w = np.ones_like(r_mpc_h)
    else:
        w = mass

    num = np.sum(w * r_mpc_h * v_rad)
    den = np.sum(w * r_mpc_h**2)

    if den <= 0.0:
        return np.nan

    return num / den


def hemisphere_masks(n_hat, axis_hat):
    """
    Given unit vectors n_hat and an axis axis_hat, return boolean masks for
    'north' (n·axis >= 0) and 'south' (n·axis < 0) hemispheres.
    """
    proj = np.dot(n_hat, axis_hat)
    north = proj >= 0.0
    south = proj < 0.0
    return north, south


# -------------------------
# Main
# -------------------------

if __name__ == "__main__":
    print(f"BASE_PATH = {BASE_PATH}")
    print("\n===== EXPANSION DIPOLE TIME SERIES (HEMISPHERES ALONG STRUCTURAL DIPOLE) =====\n")

    # Load structural dipole vectors and bulk flow vectors
    if not os.path.exists("dipole_vectors.npy"):
        raise FileNotFoundError("dipole_vectors.npy not found. Run dipole_time_series.py first.")

    dipole_vectors = np.load("dipole_vectors.npy")  # shape (N_snap, 3)
    if dipole_vectors.shape[0] != len(SNAPSHOTS):
        raise RuntimeError(
            f"dipole_vectors.npy has {dipole_vectors.shape[0]} entries, "
            f"but SNAPSHOTS has {len(SNAPSHOTS)}. Make sure the order matches."
        )

    bulk_vectors = None
    if os.path.exists("bulk_flow_vectors.npy"):
        bulk_vectors = np.load("bulk_flow_vectors.npy")
        if bulk_vectors.shape[0] != len(SNAPSHOTS):
            raise RuntimeError(
                f"bulk_flow_vectors.npy has {bulk_vectors.shape[0]} entries, "
                f"but SNAPSHOTS has {len(SNAPSHOTS)}."
            )
        print("Found bulk_flow_vectors.npy, will subtract bulk flow before fitting deltaH.")
    else:
        print("bulk_flow_vectors.npy not found, fitting deltaH without bulk-flow subtraction.")
        print("You probably want to run bulk_flow_time_series.py first.\n")

    # Storage
    snapshots_out = []
    z_out         = []
    H_frw_out     = []

    deltaH_all        = []
    deltaH_north      = []
    deltaH_south      = []
    deltaH_dipole     = []   # 0.5*(north - south)

    deltaH_all_over_H    = []
    deltaH_north_over_H  = []
    deltaH_south_over_H  = []
    deltaH_dipole_over_H = []

    # Loop over snapshots
    for i, snap in enumerate(SNAPSHOTS):
        print(f"Processing snapshot {snap}...")

        # Structural dipole axis
        d_vec = dipole_vectors[i]
        d_norm = np.linalg.norm(d_vec)
        if d_norm == 0.0 or not np.isfinite(d_norm):
            raise RuntimeError(f"Non-finite structural dipole vector at snapshot {snap}.")
        d_hat = d_vec / d_norm

        # Bulk flow vector (if available)
        v_bulk = None
        if bulk_vectors is not None:
            v_bulk = bulk_vectors[i]
        # else: we leave v_bulk as None

        # Find folder and load halos
        folder = find_groupcat_folder(BASE_PATH, snap)
        print(f"  Using folder: {folder}")
        pos, vel, mass = load_subhalos(folder, mass_min=MASS_MIN)
        print(f"  Total halos used (after mass cut): {pos.shape[0]}")

        # Radial data (with bulk flow subtraction)
        r_mpc_h, v_rad, n_hat, mask = compute_radial_data(pos, vel, v_bulk=v_bulk)
        mass_used = mass[mask]

        # Global deltaH
        dh_all = fit_deltaH(r_mpc_h, v_rad, mass=mass_used)

        # Hemispheres along structural dipole
        north_mask, south_mask = hemisphere_masks(n_hat, d_hat)

        if not np.any(north_mask) or not np.any(south_mask):
            raise RuntimeError(f"Hemisphere split failed at snapshot {snap}: empty hemisphere.")

        dh_north = fit_deltaH(
            r_mpc_h[north_mask],
            v_rad[north_mask],
            mass=mass_used[north_mask],
        )
        dh_south = fit_deltaH(
            r_mpc_h[south_mask],
            v_rad[south_mask],
            mass=mass_used[south_mask],
        )

        dh_dipole = 0.5 * (dh_north - dh_south)

        # FRW normalisation
        if snap not in SNAP_TO_Z:
            raise RuntimeError(f"No redshift entry for snapshot {snap} in SNAP_TO_Z.")
        z = SNAP_TO_Z[snap]
        H_frw = H_comov(z)

        snapshots_out.append(snap)
        z_out.append(z)
        H_frw_out.append(H_frw)

        deltaH_all.append(dh_all)
        deltaH_north.append(dh_north)
        deltaH_south.append(dh_south)
        deltaH_dipole.append(dh_dipole)

        deltaH_all_over_H.append(dh_all / H_frw)
        deltaH_north_over_H.append(dh_north / H_frw)
        deltaH_south_over_H.append(dh_south / H_frw)
        deltaH_dipole_over_H.append(dh_dipole / H_frw)

        print(f"  z ~ {z:.2f}, H_FRW_comov ~ {H_frw:.3f} km/s/(Mpc/h)")
        print(f"  deltaH_all     = {dh_all:.6e}  ->  deltaH_all/H = {dh_all / H_frw:.3e}")
        print(f"  deltaH_north   = {dh_north:.6e}  ->  deltaH_north/H = {dh_north / H_frw:.3e}")
        print(f"  deltaH_south   = {dh_south:.6e}  ->  deltaH_south/H = {dh_south / H_frw:.3e}")
        print(f"  deltaH_dipole  = {dh_dipole:.6e}  ->  (north-south)/2H = {dh_dipole / H_frw:.3e}")
        print("")

    # Convert to arrays and save
    snapshots_out = np.array(snapshots_out, dtype=int)
    z_out         = np.array(z_out, dtype=float)
    H_frw_out     = np.array(H_frw_out, dtype=float)

    deltaH_all        = np.array(deltaH_all, dtype=float)
    deltaH_north      = np.array(deltaH_north, dtype=float)
    deltaH_south      = np.array(deltaH_south, dtype=float)
    deltaH_dipole     = np.array(deltaH_dipole, dtype=float)

    deltaH_all_over_H    = np.array(deltaH_all_over_H, dtype=float)
    deltaH_north_over_H  = np.array(deltaH_north_over_H, dtype=float)
    deltaH_south_over_H  = np.array(deltaH_south_over_H, dtype=float)
    deltaH_dipole_over_H = np.array(deltaH_dipole_over_H, dtype=float)

    np.savez(
        "expansion_dipole_results.npz",
        snapshots=snapshots_out,
        z=z_out,
        H_frw_comov=H_frw_out,
        deltaH_all=deltaH_all,
        deltaH_north=deltaH_north,
        deltaH_south=deltaH_south,
        deltaH_dipole=deltaH_dipole,
        deltaH_all_over_H=deltaH_all_over_H,
        deltaH_north_over_H=deltaH_north_over_H,
        deltaH_south_over_H=deltaH_south_over_H,
        deltaH_dipole_over_H=deltaH_dipole_over_H,
    )

    print("Saved expansion dipole results to expansion_dipole_results.npz\n")
    print("Summary (per snapshot):")
    print("snap  z    deltaH_all/H   deltaH_dipole/H")
    for s, z, r_all, r_dip in zip(
        snapshots_out, z_out, deltaH_all_over_H, deltaH_dipole_over_H
    ):
        print(f"{s:3d}  {z:3.1f}  {r_all: .3e}       {r_dip: .3e}")

    print("\nDone.")
