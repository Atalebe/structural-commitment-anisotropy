#!/usr/bin/env python
"""
Expansion dipole time series for TNG50-1 using hemisphere fits along
the structural dipole axis, with bulk flow subtraction.

Outputs:
  tng50_expansion_dipole_results.npz
    - snapshots
    - redshifts
    - deltaH_all
    - deltaH_all_over_H
    - deltaH_north
    - deltaH_north_over_H
    - deltaH_south
    - deltaH_south_over_H
    - deltaH_dipole
    - deltaH_dipole_over_H
"""

import os
import glob
import numpy as np
import h5py

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

BASE_PATH = "/mnt/g/TNG50-1"
BOX_SIZE = 35.0  # Mpc/h
SNAPSHOTS = [40, 50, 59, 67, 72, 78, 91]

SNAP_TO_Z = {
    40: 1.5,
    50: 1.0,
    59: 0.8,
    67: 0.6,
    72: 0.5,
    78: 0.4,
    91: 0.2,
}

MASS_CUT = 1.0  # in TNG units -> 1e10 Msun/h


# ----------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------

def find_groupcat_folder(snapshot: int) -> str:
    candidates = [
        os.path.join(BASE_PATH, f"groupcat_{snapshot:03d}"),
        os.path.join(BASE_PATH, f"groups_{snapshot:03d}"),
    ]
    for folder in candidates:
        if os.path.isdir(folder):
            return folder
    raise RuntimeError(
        f"No groupcat folder found for snapshot {snapshot}. Tried:\n"
        + "\n".join(candidates)
    )


def load_subhalos(snapshot: int, mass_cut: float | None = None):
    """
    Load subhalo positions (Mpc/h), velocities (km/s) and masses.

    Handles missing SubhaloPos/SubhaloCM/SubhaloVel/SubhaloMass by
    skipping those chunks.
    """
    folder = find_groupcat_folder(snapshot)
    print(f"  Using folder: {folder}")

    pattern = os.path.join(folder, f"fof_subhalo_tab_{snapshot:03d}.*.hdf5")
    files = sorted(glob.glob(pattern))
    if not files:
        raise RuntimeError(f"No groupcat files found matching {pattern}")

    pos_list = []
    vel_list = []
    mass_list = []
    used_files = 0
    skipped_files = 0

    for path in files:
        with h5py.File(path, "r") as f:
            if "Subhalo" not in f:
                skipped_files += 1
                print(f"  [WARN] No 'Subhalo' group in {os.path.basename(path)}, skipping.")
                continue
            sub = f["Subhalo"]

            # Position field
            if "SubhaloPos" in sub:
                pos_field = "SubhaloPos"
            elif "SubhaloCM" in sub:
                pos_field = "SubhaloCM"
            else:
                skipped_files += 1
                print(
                    f"  [WARN] No SubhaloPos or SubhaloCM in "
                    f"{os.path.basename(path)}, skipping."
                )
                continue

            if "SubhaloVel" not in sub or "SubhaloMass" not in sub:
                skipped_files += 1
                print(
                    f"  [WARN] Missing SubhaloVel or SubhaloMass in "
                    f"{os.path.basename(path)}, skipping."
                )
                continue

            pos_ckpc = sub[pos_field][:]
            vel = sub["SubhaloVel"][:]
            mass = sub["SubhaloMass"][:]

            if mass_cut is not None:
                mask = mass >= mass_cut
                if not np.any(mask):
                    skipped_files += 1
                    print(
                        f"  [WARN] All subhalos below mass_cut in "
                        f"{os.path.basename(path)}, skipping."
                    )
                    continue
                pos_ckpc = pos_ckpc[mask]
                vel = vel[mask]
                mass = mass[mask]

            if pos_ckpc.size == 0:
                skipped_files += 1
                print(
                    f"  [WARN] No subhalos left after cuts in "
                    f"{os.path.basename(path)}, skipping."
                )
                continue

            pos_list.append(pos_ckpc)
            vel_list.append(vel)
            mass_list.append(mass)
            used_files += 1

    if not pos_list:
        raise RuntimeError(f"No usable subhalo data for snapshot {snapshot}")

    pos_ckpc = np.concatenate(pos_list, axis=0)
    vel = np.concatenate(vel_list, axis=0)
    mass = np.concatenate(mass_list, axis=0)

    pos_mpc = pos_ckpc.astype(np.float64) / 1000.0
    vel = vel.astype(np.float64)
    mass = mass.astype(np.float64)

    print(f"  Files used: {used_files}, skipped: {skipped_files}")
    print(f"  Total halos used (after mass cut): {pos_mpc.shape[0]}")

    return pos_mpc, vel, mass


# ----------------------------------------------------------------------
# Cosmology
# ----------------------------------------------------------------------

def H_comoving(z: float) -> float:
    """
    Comoving H(z) in km/s/(Mpc/h); simple flat LCDM approximation.
    """
    H0 = 67.74
    Omega_m = 0.3089
    Omega_L = 1.0 - Omega_m
    return H0 * np.sqrt(Omega_m * (1.0 + z) ** 3 + Omega_L)


# ----------------------------------------------------------------------
# Expansion dipole along structural axis
# ----------------------------------------------------------------------

def compute_expansion_dipole(snapshot: int,
                             struct_dir: np.ndarray,
                             bulk_flow_vec: np.ndarray):
    """
    For a given snapshot, fit effective H along and opposite the
    structural dipole axis using hemispheres and return deltaH
    (all, north, south, dipole).
    """
    pos, vel, mass = load_subhalos(snapshot, mass_cut=MASS_CUT)

    # subtract bulk flow
    vel_corr = vel - bulk_flow_vec[None, :]

    center = np.array([BOX_SIZE / 2.0] * 3, dtype=np.float64)
    r_vec = pos - center[None, :]
    r = np.linalg.norm(r_vec, axis=1)
    mask = r > 0.0
    r_vec = r_vec[mask]
    vel_corr = vel_corr[mask]
    r = r[mask]

    r_hat = r_vec / r[:, None]

    # radial velocities
    v_rad = np.sum(vel_corr * r_hat, axis=1)

    z = SNAP_TO_Z[snapshot]
    H_FRW = H_comoving(z)

    # global fit
    num = np.sum(r * v_rad, dtype=np.float64)
    den = np.sum(r * r, dtype=np.float64)
    H_eff_all = num / den
    deltaH_all = H_eff_all - H_FRW

    # hemispheres along structural axis
    d_hat = struct_dir / np.linalg.norm(struct_dir)
    cos_theta = np.sum(r_hat * d_hat[None, :], axis=1)

    north = cos_theta > 0.0
    south = cos_theta < 0.0

    def fit_H(mask_region: np.ndarray) -> float:
        if not np.any(mask_region):
            return H_FRW
        r_loc = r[mask_region]
        v_loc = v_rad[mask_region]
        num_loc = np.sum(r_loc * v_loc, dtype=np.float64)
        den_loc = np.sum(r_loc * r_loc, dtype=np.float64)
        return num_loc / den_loc

    H_eff_north = fit_H(north)
    H_eff_south = fit_H(south)

    deltaH_north = H_eff_north - H_FRW
    deltaH_south = H_eff_south - H_FRW
    deltaH_dipole = 0.5 * (H_eff_north - H_eff_south)

    return H_FRW, deltaH_all, deltaH_north, deltaH_south, deltaH_dipole


def main():
    print(f"BASE_PATH = {BASE_PATH}")
    print(f"BOX_SIZE  = {BOX_SIZE} Mpc/h")
    print(f"SNAPSHOTS = {SNAPSHOTS}")
    print(f"MASS_CUT  = {MASS_CUT} (1e10 Msun/h)\n")

    struct_vecs = np.load("tng50_dipole_vectors.npy")
    if struct_vecs.shape[0] != len(SNAPSHOTS):
        raise RuntimeError(
            f"tng50_dipole_vectors.npy has {struct_vecs.shape[0]} entries, "
            f"but SNAPSHOTS has {len(SNAPSHOTS)}."
        )

    bulk_vecs = np.load("tng50_bulk_flow_vectors.npy")
    if bulk_vecs.shape[0] != len(SNAPSHOTS):
        raise RuntimeError(
            f"tng50_bulk_flow_vectors.npy has {bulk_vecs.shape[0]} entries, "
            f"but SNAPSHOTS has {len(SNAPSHOTS)}."
        )

    n_snap = len(SNAPSHOTS)

    deltaH_all = np.zeros(n_snap)
    deltaH_all_over_H = np.zeros(n_snap)
    deltaH_north = np.zeros(n_snap)
    deltaH_north_over_H = np.zeros(n_snap)
    deltaH_south = np.zeros(n_snap)
    deltaH_south_over_H = np.zeros(n_snap)
    deltaH_dipole = np.zeros(n_snap)
    deltaH_dipole_over_H = np.zeros(n_snap)

    redshifts = np.array([SNAP_TO_Z[s] for s in SNAPSHOTS], dtype=float)

    print("===== EXPANSION DIPOLE TIME SERIES (TNG50, hemispheres) =====\n")
    print("Found tng50_bulk_flow_vectors.npy, will subtract bulk flow.\n")

    for i, snap in enumerate(SNAPSHOTS):
        print(f"Processing snapshot {snap}...")
        struct_dir = struct_vecs[i, :]
        bulk_flow_vec = bulk_vecs[i, :]

        H_FRW, d_all, d_north, d_south, d_dip = compute_expansion_dipole(
            snap, struct_dir, bulk_flow_vec
        )

        deltaH_all[i] = d_all
        deltaH_north[i] = d_north
        deltaH_south[i] = d_south
        deltaH_dipole[i] = d_dip

        deltaH_all_over_H[i] = d_all / H_FRW
        deltaH_north_over_H[i] = d_north / H_FRW
        deltaH_south_over_H[i] = d_south / H_FRW
        deltaH_dipole_over_H[i] = d_dip / H_FRW

        print(f"  z ~ {SNAP_TO_Z[snap]:.2f}, H_FRW_comov ~ {H_FRW:.3f} km/s/(Mpc/h)")
        print(f"  deltaH_all     = {d_all:+.6e}  ->  deltaH_all/H = {d_all / H_FRW:+.3e}")
        print(f"  deltaH_north   = {d_north:+.6e}  ->  deltaH_north/H = {d_north / H_FRW:+.3e}")
        print(f"  deltaH_south   = {d_south:+.6e}  ->  deltaH_south/H = {d_south / H_FRW:+.3e}")
        print(f"  deltaH_dipole  = {d_dip:+.6e}  ->  (north-south)/2H = {d_dip / H_FRW:+.3e}\n")

    np.savez(
        "tng50_expansion_dipole_results.npz",
        snapshots=np.array(SNAPSHOTS, dtype=int),
        redshifts=redshifts,
        deltaH_all=deltaH_all,
        deltaH_all_over_H=deltaH_all_over_H,
        deltaH_north=deltaH_north,
        deltaH_north_over_H=deltaH_north_over_H,
        deltaH_south=deltaH_south,
        deltaH_south_over_H=deltaH_south_over_H,
        deltaH_dipole=deltaH_dipole,
        deltaH_dipole_over_H=deltaH_dipole_over_H,
    )

    print("Saved: tng50_expansion_dipole_results.npz\n")
    print("Summary (per snapshot):")
    for i, snap in enumerate(SNAPSHOTS):
        print(
            f"  snap {snap:3d}, z={redshifts[i]:.1f}  "
            f"deltaH_all/H = {deltaH_all_over_H[i]:+.3e},  "
            f"deltaH_dipole/H = {deltaH_dipole_over_H[i]:+.3e}"
        )
    print("\nDone.")


if __name__ == "__main__":
    main()
