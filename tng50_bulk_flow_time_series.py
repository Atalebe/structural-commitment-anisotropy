#!/usr/bin/env python
"""
Compute bulk flow time series for TNG50-1 and its alignment with the
structural dipole direction.

Outputs:
  - tng50_bulk_flow_vectors.npy           (Nsnap, 3)
  - tng50_bulk_flow_amplitudes.npy        (Nsnap,)
  - tng50_bulk_flow_alignment_cos.npy     (Nsnap,)
  - tng50_bulk_flow_alignment_angle_deg.npy (Nsnap,)
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

# approximate redshifts (not actually used in this script, but kept for reference)
SNAP_TO_Z = {
    40: 1.5,
    50: 1.0,
    59: 0.8,
    67: 0.6,
    72: 0.5,
    78: 0.4,
    91: 0.2,
}


# ----------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------

def find_groupcat_folder(snapshot: int) -> str:
    """
    Find the group catalog folder for a given snapshot.
    Tries groupcat_XXX and groups_XXX.
    """
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
    Load subhalo positions, velocities, and masses for a given snapshot.

    - Positions: SubhaloPos (ckpc/h) or SubhaloCM (ckpc/h), converted to Mpc/h
    - Velocities: SubhaloVel (km/s)
    - Masses: SubhaloMass (1e10 Msun/h)

    Some TNG50 groupcat chunks do not contain Subhalo fields; those are skipped.
    Some chunks may lack one of the required datasets; they are also skipped.
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

            # Position field: prefer SubhaloPos, fall back to SubhaloCM
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

            pos_ckpc = sub[pos_field][:]      # (N, 3)
            vel = sub["SubhaloVel"][:]        # (N, 3), km/s
            mass = sub["SubhaloMass"][:]      # (N,)

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

    pos_mpc = pos_ckpc.astype(np.float64) / 1000.0  # ckpc/h -> Mpc/h
    vel = vel.astype(np.float64)
    mass = mass.astype(np.float64)

    print(f"  Files used: {used_files}, skipped: {skipped_files}")
    print(f"  Total subhalos loaded: {pos_mpc.shape[0]}")

    return pos_mpc, vel, mass


# ----------------------------------------------------------------------
# Bulk flow
# ----------------------------------------------------------------------

def compute_bulk_flow(snapshot: int, mass_cut: float | None = None):
    """
    Compute mass-weighted bulk flow vector and its amplitude.
    """
    pos, vel, mass = load_subhalos(snapshot, mass_cut=mass_cut)

    m_tot = mass.sum(dtype=np.float64)
    v_bulk = np.sum(vel * mass[:, None], axis=0, dtype=np.float64) / m_tot
    amp = np.linalg.norm(v_bulk)

    return v_bulk, amp


def main():
    print(f"BASE_PATH = {BASE_PATH}")
    print(f"BOX_SIZE  = {BOX_SIZE} Mpc/h")
    print(f"SNAPSHOTS = {SNAPSHOTS}\n")

    # structural dipole vectors from tng50_dipole_time_series.py
    struct_vecs = np.load("tng50_dipole_vectors.npy")
    if struct_vecs.shape[0] != len(SNAPSHOTS):
        raise RuntimeError(
            f"tng50_dipole_vectors.npy has {struct_vecs.shape[0]} entries, "
            f"but SNAPSHOTS has {len(SNAPSHOTS)}."
        )

    n_snap = len(SNAPSHOTS)
    bulk_vecs = np.zeros((n_snap, 3), dtype=np.float64)
    bulk_amp = np.zeros(n_snap, dtype=np.float64)
    align_cos = np.zeros(n_snap, dtype=np.float64)
    align_angle = np.zeros(n_snap, dtype=np.float64)

    print("===== BULK FLOW TIME SERIES (TNG50) =====\n")

    for i, snap in enumerate(SNAPSHOTS):
        print(f"Loading snapshot {snap}...")
        v_bulk, amp = compute_bulk_flow(snap, mass_cut=None)

        bulk_vecs[i, :] = v_bulk
        bulk_amp[i] = amp

        d_vec = struct_vecs[i, :]
        d_hat = d_vec / np.linalg.norm(d_vec)

        if amp > 0:
            v_hat = v_bulk / amp
            cos_theta = float(np.clip(np.dot(d_hat, v_hat), -1.0, 1.0))
            theta_deg = float(np.degrees(np.arccos(cos_theta)))
        else:
            cos_theta = np.nan
            theta_deg = np.nan

        align_cos[i] = cos_theta
        align_angle[i] = theta_deg

        print(f"  Bulk flow vector (km/s): {v_bulk}")
        print(f"  |v_bulk| = {amp:.3f} km/s")
        print("  Alignment with structural dipole:")
        print(f"    cos(theta) = {cos_theta:+.3f},  theta = {theta_deg:6.1f} deg\n")

    np.save("tng50_bulk_flow_vectors.npy", bulk_vecs)
    np.save("tng50_bulk_flow_amplitudes.npy", bulk_amp)
    np.save("tng50_bulk_flow_alignment_cos.npy", align_cos)
    np.save("tng50_bulk_flow_alignment_angle_deg.npy", align_angle)

    print("Saved:")
    print("  tng50_bulk_flow_vectors.npy")
    print("  tng50_bulk_flow_amplitudes.npy")
    print("  tng50_bulk_flow_alignment_cos.npy")
    print("  tng50_bulk_flow_alignment_angle_deg.npy")
    print("\nDone.")


if __name__ == "__main__":
    main()
