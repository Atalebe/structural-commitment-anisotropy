#!/usr/bin/env python
"""
tng300_dipole_time_series.py

Compute the structural dipole time series for TNG300 and save:
  - tng300_dipole_vectors.npy  (shape: (Nsnap, 3))
  - tng300_dipole_amplitudes.npy  (shape: (Nsnap,))
"""

import os
import glob
import numpy as np
import h5py

# --- CONFIG ---

BASE_PATH = "/mnt/g/TNG300-1"
BOX_SIZE = 205.0  # Mpc/h
SNAPSHOTS = [33, 40, 50, 59, 67, 72, 78, 91]

# --- HELPERS ---

def find_groupcat_folder(base_path, snap):
    s = f"{snap:03d}"
    candidates = [
        os.path.join(base_path, f"groupcat_{s}"),
        os.path.join(base_path, f"groups_{s}"),
        os.path.join(base_path, f"groupcat_{snap}"),
        os.path.join(base_path, f"groups_{snap}"),
    ]
    for folder in candidates:
        if os.path.isdir(folder):
            return folder
    raise RuntimeError(
        f"[TNG300] No groupcat folder found for snapshot {snap}. Tried:\n"
        + "\n".join(candidates)
    )

def load_subhalos(snapshot):
    """
    Load subhalo positions for a given snapshot.
    Returns positions in Mpc/h (numpy array of shape (N, 3)).
    """
    folder = find_groupcat_folder(BASE_PATH, snapshot)
    print(f"  Using folder: {folder}")

    pattern = os.path.join(folder, f"fof_subhalo_tab_{snapshot:03d}.*.hdf5")
    files = sorted(glob.glob(pattern))
    if not files:
        raise RuntimeError(f"[TNG300] No HDF5 groupcat files found for snapshot {snapshot} in {folder}")

    pos_list = []
    files_used = 0
    files_skipped = 0

    for fname in files:
        with h5py.File(fname, "r") as f:
            if "Subhalo" not in f:
                files_skipped += 1
                continue
            sub = f["Subhalo"]
            # Prefer SubhaloPos; fall back to SubhaloCM if needed
            if "SubhaloPos" in sub:
                pos_ckpc = sub["SubhaloPos"][:]  # ckpc/h
            elif "SubhaloCM" in sub:
                pos_ckpc = sub["SubhaloCM"][:]   # ckpc/h
            else:
                print(f"  [WARN] No SubhaloPos or SubhaloCM in {os.path.basename(fname)}, skipping.")
                files_skipped += 1
                continue

            pos_list.append(pos_ckpc)
            files_used += 1

    if not pos_list:
        raise RuntimeError(f"[TNG300] No usable subhalo positions found for snapshot {snapshot}.")

    pos_ckpc_all = np.vstack(pos_list)  # shape (N,3)
    pos_mpc_all = pos_ckpc_all / 1000.0  # ckpc/h -> Mpc/h

    print(f"  Files used: {files_used}, skipped: {files_skipped}")
    print(f"  Total subhalos loaded: {pos_mpc_all.shape[0]:,d}")
    return pos_mpc_all

def compute_structural_dipole(snapshot):
    """
    Compute structural dipole for a single snapshot:
      - project positions to unit sphere around box center
      - compute weighted COM on the sphere (equal weights)
    Returns:
      (dipole_vector[3], amplitude)
    """
    pos_mpc = load_subhalos(snapshot)

    # Shift to box-centred coordinates
    center = 0.5 * BOX_SIZE
    r = pos_mpc - center  # shape (N,3)
    r_norm = np.linalg.norm(r, axis=1)

    # Avoid division by zero (very unlikely)
    mask = r_norm > 0
    r = r[mask]
    r_norm = r_norm[mask]

    n_hat = r / r_norm[:, None]  # unit vectors on the sphere

    # Equal weights (number-weighted dipole)
    w = np.ones(n_hat.shape[0], dtype=np.float64)
    w_sum = np.sum(w)
    v = np.sum(w[:, None] * n_hat, axis=0) / w_sum  # COM on sphere

    amp = np.linalg.norm(v)
    if amp > 0:
        v_hat = v / amp
    else:
        v_hat = np.array([0.0, 0.0, 1.0])

    # Convert direction to RA/DEC for sanity
    x, y, z = v_hat
    ra = np.degrees(np.arctan2(y, x)) % 360.0
    dec = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))

    print(f"  Dipole amplitude: {amp:.5f}")
    print(f"  Dipole direction: RA = {ra:.1f} deg, DEC = {dec:.1f} deg")

    return v_hat, amp

# --- MAIN ---

def main():
    print(f"BASE_PATH = {BASE_PATH}")
    print(f"BOX_SIZE  = {BOX_SIZE} Mpc/h")
    print(f"SNAPSHOTS = {SNAPSHOTS}")
    print()

    dipole_vectors = []
    dipole_amplitudes = []

    print("===== TNG300 STRUCTURAL DIPOLE TIME SERIES =====\n")
    for snap in SNAPSHOTS:
        print(f"Snapshot {snap} ...")
        v_hat, amp = compute_structural_dipole(snap)
        dipole_vectors.append(v_hat)
        dipole_amplitudes.append(amp)
        print()

    dipole_vectors = np.array(dipole_vectors)      # (Nsnap, 3)
    dipole_amplitudes = np.array(dipole_amplitudes)  # (Nsnap,)

    print("===== DIRECTIONAL COHERENCE (TNG300) =====\n")
    # directional cosine matrix between all snapshot dipole directions
    dot_mat = dipole_vectors @ dipole_vectors.T
    print(np.round(dot_mat, 3))

    np.save("tng300_dipole_vectors.npy", dipole_vectors)
    np.save("tng300_dipole_amplitudes.npy", dipole_amplitudes)

    print("\nSaved:")
    print("  tng300_dipole_vectors.npy")
    print("  tng300_dipole_amplitudes.npy")
    print("\nDone.")

if __name__ == "__main__":
    main()

