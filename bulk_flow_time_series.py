#!/usr/bin/env python
"""
bulk_flow_time_series.py

Compute the mass-weighted bulk flow vector in TNG300-1 for the
same snapshots used in the structural dipole time series, and
compare the direction to the structural dipole vectors stored
in dipole_vectors.npy.
"""

import os
import glob
import numpy as np
import h5py

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

# Direct TNG300-1 path where the groupcats live
BASE_PATH = "/mnt/g/TNG300-1"

# Same snapshot list as used for dipole_time_series.py
SNAPSHOTS = [33, 40, 50, 59, 67, 72, 78, 91]

# Name pattern of groupcat files
FILE_PATTERN = "fof_subhalo_tab_{snap:03d}.*.hdf5"


# ----------------------------------------------------------------------
# Helpers to find and load groupcat files
# ----------------------------------------------------------------------

def find_groupcat_folder(base_path, snapshot):
    """
    Find the group catalog folder for a given snapshot.

    Tries several common naming conventions:
      groupcat_033, groupcat_33, groups_033, groups_33
    and returns the first directory that exists and contains
    at least one fof_subhalo_tab_XXX.*.hdf5 file.
    """
    snap3 = f"{snapshot:03d}"
    candidates = [
        os.path.join(base_path, f"groupcat_{snap3}"),
        os.path.join(base_path, f"groupcat_{snapshot}"),
        os.path.join(base_path, f"groups_{snap3}"),
        os.path.join(base_path, f"groups_{snapshot}"),
    ]

    for folder in candidates:
        if os.path.isdir(folder):
            pattern = os.path.join(folder, FILE_PATTERN.format(snap=snapshot))
            files = glob.glob(pattern)
            if files:
                return folder

    msg = [
        f"No groupcat folder found for snapshot {snapshot}. Tried:"
    ] + candidates
    raise RuntimeError("\n".join(msg))


def load_subhalos(snapshot, mass_cut=None):
    """
    Load SubhaloPos, SubhaloVel, SubhaloMass for a given snapshot
    by reading all fof_subhalo_tab_XXX.*.hdf5 files in the groupcat folder.

    mass_cut is applied in the native units of SubhaloMass
    (1e10 Msun/h), so mass_cut=1.0 corresponds to 1e10 Msun/h,
    10.0 to 1e11 Msun/h, etc.
    """
    folder = find_groupcat_folder(BASE_PATH, snapshot)
    print(f"Using folder: {folder}")

    pattern = os.path.join(folder, FILE_PATTERN.format(snap=snapshot))
    files = sorted(glob.glob(pattern))
    if not files:
        raise RuntimeError(f"No HDF5 files found in {folder}")

    pos_list = []
    vel_list = []
    mass_list = []

    for fname in files:
        with h5py.File(fname, "r") as f:
            sub = f["Subhalo"]

            if "SubhaloPos" not in sub or "SubhaloVel" not in sub or "SubhaloMass" not in sub:
                raise RuntimeError(
                    f"Missing required fields in {fname}. "
                    "Expected SubhaloPos, SubhaloVel, SubhaloMass."
                )

            pos = sub["SubhaloPos"][:]      # (N, 3)
            vel = sub["SubhaloVel"][:]      # (N, 3)
            mass = sub["SubhaloMass"][:]    # (N,)

        pos_list.append(pos)
        vel_list.append(vel)
        mass_list.append(mass)

    pos = np.vstack(pos_list).astype(np.float64)
    vel = np.vstack(vel_list).astype(np.float64)
    mass = np.concatenate(mass_list).astype(np.float64)

    if mass_cut is not None:
        sel = mass >= mass_cut
        pos = pos[sel]
        vel = vel[sel]
        mass = mass[sel]
        print(
            f"After mass cut >= {mass_cut:.2e} (in 1e10 Msun/h units): "
            f"{mass.size} halos retained"
        )
    else:
        print(f"Total halos used: {mass.size}")

    return pos, vel, mass


# ----------------------------------------------------------------------
# Bulk flow computation
# ----------------------------------------------------------------------

def compute_bulk_flow(snapshot, mass_cut=None):
    """
    Compute the mass-weighted bulk flow vector for a given snapshot.

    Returns:
      v_bulk : np.ndarray shape (3,)
      amp    : float, |v_bulk|
    """
    pos, vel, mass = load_subhalos(snapshot, mass_cut=mass_cut)

    total_mass = mass.sum()
    if total_mass <= 0.0:
        raise RuntimeError(f"Total mass is non-positive for snapshot {snapshot}")

    # Mass-weighted mean velocity
    v_bulk = (mass[:, None] * vel).sum(axis=0) / total_mass
    amp = float(np.linalg.norm(v_bulk))

    return v_bulk, amp


# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------

def main():
    print(f"BASE_PATH = {BASE_PATH}")
    print("\n===== BULK FLOW TIME SERIES =====\n")

    # Try to load structural dipole vectors for alignment test
    dipole_vecs = None
    if os.path.exists("dipole_vectors.npy"):
        try:
            dipole_vecs = np.load("dipole_vectors.npy")
            print("Loaded structural dipole vectors from dipole_vectors.npy")
        except Exception as e:
            print(f"Warning: failed to load dipole_vectors.npy: {e}")
            dipole_vecs = None
    else:
        print("dipole_vectors.npy not found, will skip alignment diagnostics.")

    bulk_vecs = []
    bulk_amps = []
    cos_thetas = []
    angles_deg = []

    for i, snap in enumerate(SNAPSHOTS):
        print(f"\nLoading snapshot {snap}...")
        v_bulk, amp = compute_bulk_flow(snap, mass_cut=None)
        bulk_vecs.append(v_bulk)
        bulk_amps.append(amp)

        print(f"Bulk flow vector (km/s): {v_bulk}")
        print(f"|v_bulk| = {amp:.3f} km/s")

        # Alignment with structural dipole, if available
        if dipole_vecs is not None:
            if i >= len(dipole_vecs):
                print("No matching dipole vector for this snapshot index, skipping alignment.")
                cos_thetas.append(np.nan)
                angles_deg.append(np.nan)
            else:
                d_vec = dipole_vecs[i]
                d_norm = np.linalg.norm(d_vec)

                if d_norm == 0.0 or amp == 0.0:
                    print("Zero-length vector encountered, skipping alignment.")
                    cos_thetas.append(np.nan)
                    angles_deg.append(np.nan)
                else:
                    d_hat = d_vec / d_norm
                    v_hat = v_bulk / amp
                    cos_theta = float(np.clip(np.dot(d_hat, v_hat), -1.0, 1.0))
                    angle = float(np.degrees(np.arccos(cos_theta)))

                    cos_thetas.append(cos_theta)
                    angles_deg.append(angle)

                    print(f"Alignment with structural dipole:")
                    print(f"  cos(theta) = {cos_theta:.3f},  theta = {angle:.1f} deg")

    bulk_vecs = np.array(bulk_vecs)
    bulk_amps = np.array(bulk_amps)

    np.save("bulk_flow_vectors.npy", bulk_vecs)
    np.save("bulk_flow_amplitudes.npy", bulk_amps)
    print("\nSaved:")
    print("  bulk_flow_vectors.npy")
    print("  bulk_flow_amplitudes.npy")

    if dipole_vecs is not None:
        cos_thetas = np.array(cos_thetas)
        angles_deg = np.array(angles_deg)
        np.save("bulk_flow_dipole_alignment_cos.npy", cos_thetas)
        np.save("bulk_flow_dipole_alignment_angle_deg.npy", angles_deg)

        print("  bulk_flow_dipole_alignment_cos.npy")
        print("  bulk_flow_dipole_alignment_angle_deg.npy")

        print("\n===== ALIGNMENT SUMMARY =====")
        for snap, c, ang in zip(SNAPSHOTS, cos_thetas, angles_deg):
            print(
                f"Snapshot {snap:3d}: "
                f"cos(theta) = {c:6.3f}, angle = {ang:6.1f} deg"
                if np.isfinite(c) else
                f"Snapshot {snap:3d}: alignment not available"
            )

    print("\nDone.\n")


if __name__ == "__main__":
    main()
