#!/usr/bin/env python3
"""
tng50_dipole_time_series.py

Structural dipole in TNG50:
- Mass threshold stability test at snapshot 72
- Time series over selected snapshots
- Directional coherence matrix

Robust against missing SubhaloPos: falls back to SubhaloCM,
and skips files that do not contain usable subhalo positions.
"""

import os
import glob
import h5py
import numpy as np

# -------------------------
# CONFIG
# -------------------------

BASE_PATH   = "/mnt/g/TNG50-1"
BOX_SIZE    = 35.0          # Mpc/h for TNG50-1
MASS_UNIT   = 1e10          # Msun/h; SubhaloMass is in 1e10 Msun/h units

# Snapshots to analyse
SNAPSHOTS   = [40, 50, 59, 67, 72, 78, 91]

# Physical mass thresholds (Msun/h) for stability test
THRESHOLDS  = [None, 1e10, 1e11, 1e12]


# -------------------------
# I/O HELPERS
# -------------------------

def find_groupcat_folder(base_path: str, snapshot: int) -> str:
    """
    Find the group catalog folder for a given snapshot.
    Tries groupcat_XXX and groups_XXX.
    """
    candidates = [
        os.path.join(base_path, f"groupcat_{snapshot:03d}"),
        os.path.join(base_path, f"groups_{snapshot:03d}"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise RuntimeError(
        f"No groupcat folder found for snapshot {snapshot}. "
        f"Tried: {candidates}"
    )


def load_subhalos(snapshot: int, mass_cut_internal: float | None = None):
    """
    Load subhalo positions and masses for a given snapshot.

    Positions are returned in Mpc/h.
    Masses are returned in native units of SubhaloMass (1e10 Msun/h).

    If mass_cut_internal is not None, keep only halos with
    mass >= mass_cut_internal (same native units).
    """
    folder = find_groupcat_folder(BASE_PATH, snapshot)
    print(f"Using folder: {folder}")

    pattern = os.path.join(folder, f"fof_subhalo_tab_{snapshot:03d}.*.hdf5")
    files = sorted(glob.glob(pattern))
    if not files:
        raise RuntimeError(f"No HDF5 files found in {folder}")

    pos_list  = []
    mass_list = []
    used_files = 0
    skipped_files = 0

    for fname in files:
        with h5py.File(fname, "r") as f:
            if "Subhalo" not in f:
                print(f"  [WARN] No 'Subhalo' group in {os.path.basename(fname)}, skipping.")
                skipped_files += 1
                continue

            sub = f["Subhalo"]

            # Choose a position dataset
            if "SubhaloPos" in sub:
                pos_ckpc = sub["SubhaloPos"][:]       # ckpc/h
            elif "SubhaloCM" in sub:
                pos_ckpc = sub["SubhaloCM"][:]        # ckpc/h, centre-of-mass
                print(f"  [INFO] Using SubhaloCM as position in {os.path.basename(fname)}")
            else:
                print(f"  [WARN] No SubhaloPos or SubhaloCM in {os.path.basename(fname)}, skipping.")
                skipped_files += 1
                continue

            if "SubhaloMass" not in sub:
                print(f"  [WARN] No SubhaloMass in {os.path.basename(fname)}, skipping.")
                skipped_files += 1
                continue

            mass = sub["SubhaloMass"][:]             # 1e10 Msun/h

        # Convert positions to Mpc/h
        pos_mpc = pos_ckpc / 1000.0

        pos_list.append(pos_mpc)
        mass_list.append(mass)
        used_files += 1

    if not pos_list:
        raise RuntimeError(
            f"No usable Subhalo position+mass data found for snapshot {snapshot} "
            f"in folder {folder}. Used_files={used_files}, skipped_files={skipped_files}"
        )

    pos  = np.vstack(pos_list)
    mass = np.concatenate(mass_list)

    print(f"  Files used: {used_files}, skipped: {skipped_files}")
    print(f"  Total subhalos loaded: {pos.shape[0]:,d}")

    if mass_cut_internal is not None:
        mask = mass >= mass_cut_internal
        pos  = pos[mask]
        mass = mass[mask]
        print(f"  After mass cut: {pos.shape[0]:,d} subhalos")

    return pos, mass


# -------------------------
# STRUCTURAL DIPOLE
# -------------------------

def compute_structural_dipole(snapshot: int, mass_cut_internal: float | None = None):
    """
    Compute the mass-weighted structural dipole vector and amplitude
    for a given snapshot and optional internal mass cut.
    """
    pos, mass = load_subhalos(snapshot, mass_cut_internal=mass_cut_internal)

    # Center of the box
    center = np.array([BOX_SIZE / 2.0] * 3)
    rel    = pos - center

    r = np.linalg.norm(rel, axis=1)
    mask = r > 0.0
    rel  = rel[mask]
    mass = mass[mask]
    r    = r[mask]

    # Unit vectors
    n_hat = rel / r[:, None]

    # Mass-weighted dipole vector
    vec = np.sum(mass[:, None] * n_hat, axis=0) / np.sum(mass)
    amp = np.linalg.norm(vec)

    return vec, amp


# -------------------------
# MASS THRESHOLD STABILITY
# -------------------------

def run_mass_threshold_test():
    """
    At snapshot 72, test structural dipole stability under mass cuts:
    NO CUT, >1e10, >1e11, >1e12 Msun/h.
    """
    snapshot = 72
    print("\n===== MASS THRESHOLD STABILITY TEST (TNG50, snap 72) =====\n")

    # No cut
    pos, mass = load_subhalos(snapshot, mass_cut_internal=None)
    vec, amp  = compute_structural_dipole(snapshot, mass_cut_internal=None)
    print(f"Total halos used (NO CUT): {pos.shape[0]:,d}")
    print(f"NO CUT → Dipole amplitude: {amp:.5f}\n")

    # With physical mass thresholds
    for th in THRESHOLDS[1:]:
        mass_cut_internal = th / MASS_UNIT  # convert Msun/h → native units
        pos, mass = load_subhalos(snapshot, mass_cut_internal=mass_cut_internal)
        vec, amp  = compute_structural_dipole(snapshot, mass_cut_internal=mass_cut_internal)

        print(f"After cut > {th:.1e} Msun/h:")
        print(f"Halos retained: {pos.shape[0]:,d}")
        print(f">{th:.0e} → Dipole amplitude: {amp:.5f}\n")

    print("===== SUMMARY (TNG50, snap 72) =====")
    print("See logs above for NO CUT and thresholds 1e10, 1e11, 1e12 Msun/h.\n")


# -------------------------
# TIME SERIES + COHERENCE
# -------------------------

def run_time_series():
    """
    Compute structural dipole time series for TNG50 over SNAPSHOTS.
    Saves vectors and amplitudes, and prints the direction cosine matrix.
    """
    print("\n===== COMPUTING STRUCTURAL DIPOLE TIME SERIES (TNG50) =====\n")

    vecs = []
    amps = []

    for snap in SNAPSHOTS:
        print(f"Loading snapshot {snap}...")
        vec, amp = compute_structural_dipole(snap, mass_cut_internal=None)
        vecs.append(vec)
        amps.append(amp)
        print(f"Dipole amplitude: {amp:.5f}\n")

    vecs = np.array(vecs)  # shape (Nsnap, 3)
    amps = np.array(amps)  # shape (Nsnap,)

    # Direction cosine matrix
    norms    = np.linalg.norm(vecs, axis=1, keepdims=True)
    hat_vecs = vecs / norms
    cos_mat  = hat_vecs @ hat_vecs.T

    print("===== DIRECTIONAL COHERENCE (TNG50) =====\n")
    np.set_printoptions(precision=3, suppress=True)
    print(cos_mat)
    print()

    # Save results
    np.save("tng50_dipole_vectors.npy", vecs)
    np.save("tng50_dipole_amplitudes.npy", amps)

    print("Saved:")
    print("  tng50_dipole_vectors.npy")
    print("  tng50_dipole_amplitudes.npy")
    print()


# -------------------------
# MAIN
# -------------------------

if __name__ == "__main__":
    print(f"BASE_PATH = {BASE_PATH}")
    print(f"BOX_SIZE  = {BOX_SIZE} Mpc/h")
    print(f"SNAPSHOTS = {SNAPSHOTS}")
    print()

    run_mass_threshold_test()
    run_time_series()
