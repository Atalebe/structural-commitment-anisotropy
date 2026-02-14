import numpy as np
import h5py
import glob
import os

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

BASE_PATH = "/mnt/g/TNG300-1"
SNAPSHOTS = [33, 40, 50, 59, 67, 72, 78, 91]

# TNG300 box size
BOX_SIZE = 205000.0   # ckpc/h

# Mass thresholds (Msun/h)
THRESHOLDS = [
    None,
    1e10,
    1e11,
    1e12
]

print("BASE_PATH =", BASE_PATH)

# --------------------------------------------------
# FILE LOADER
# --------------------------------------------------

from glob import glob
import os
import h5py
import numpy as np

def load_subhalos(snapshot):

    folder = f"/mnt/g/TNG300-1/groupcat_{snapshot:03d}"
    print("Using folder:", folder)

    files = sorted(glob(os.path.join(folder, "fof_subhalo_tab_*.hdf5")))

    if len(files) == 0:
        raise RuntimeError(f"No HDF5 files found in {folder}")

    pos_list = []
    mass_list = []

    for fpath in files:
        with h5py.File(fpath, "r") as h:

            # skip empty chunks safely
            if "Subhalo" not in h:
                continue
            if "SubhaloPos" not in h["Subhalo"]:
                continue

            pos_list.append(h["Subhalo"]["SubhaloPos"][:])
            mass_list.append(h["Subhalo"]["SubhaloMass"][:])

    pos = np.concatenate(pos_list)
    mass = np.concatenate(mass_list)

    return pos, mass


# --------------------------------------------------
# DIPOLE ENGINE
# --------------------------------------------------

def compute_structural_dipole(snapshot, mass_threshold=None):

    pos, mass = load_subhalos(snapshot)

    # Convert to Msun/h
    mass = mass * 1e10

    if mass_threshold is not None:

        mask = mass > mass_threshold

        pos = pos[mask]
        mass = mass[mask]

        print(f"After cut > {mass_threshold:.1e} Msun/h:")
        print(f"Halos retained: {len(mass):,}")

    print(f"Total halos used: {len(mass):,}")

    center = np.array([BOX_SIZE/2]*3)

    vec = pos - center
    r = np.linalg.norm(vec, axis=1)

    valid = r > 0

    vec = vec[valid]
    mass = mass[valid]
    r = r[valid]

    unit = vec / r[:, None]

    dipole_vector = np.sum(unit * mass[:, None], axis=0) / np.sum(mass)

    amplitude = np.linalg.norm(dipole_vector)

    return dipole_vector, amplitude


# --------------------------------------------------
# MASS STABILITY TEST (SNAPSHOT 72)
# --------------------------------------------------

def run_mass_threshold_test():

    print("\n===== MASS THRESHOLD STABILITY TEST =====\n")

    snapshot = 72

    results = {}

    for t in THRESHOLDS:

        label = "NO CUT" if t is None else f">{t:.0e}"

        vec, amp = compute_structural_dipole(snapshot, t)

        results[label] = amp

        print(f"{label} → Dipole amplitude: {amp:.5f}\n")

    print("===== SUMMARY =====")

    for k,v in results.items():
        print(f"{k:<10} : {v:.5f}")


# --------------------------------------------------
# TIME SERIES (NO THRESHOLD)
# --------------------------------------------------

def run_time_series():

    print("\n===== COMPUTING STRUCTURAL DIPOLE TIME SERIES =====\n")

    vectors = []
    amplitudes = []

    for snap in SNAPSHOTS:

        print(f"\nLoading snapshot {snap}...")

        vec, amp = compute_structural_dipole(snap)

        print(f"Dipole amplitude: {amp:.5f}")

        vectors.append(vec)
        amplitudes.append(amp)

    vectors = np.array(vectors)
    amplitudes = np.array(amplitudes)

    # Directional coherence
    hats = vectors / np.linalg.norm(vectors, axis=1)[:,None]
    cosine_matrix = hats @ hats.T

    print("\n===== DIRECTIONAL COHERENCE =====\n")
    print(np.round(cosine_matrix, 3))

    np.save("dipole_vectors.npy", vectors)
    np.save("dipole_amplitudes.npy", amplitudes)

    print("\nSaved:")
    print("dipole_vectors.npy")
    print("dipole_amplitudes.npy")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    # FIRST — diagnose structure driver
    run_mass_threshold_test()

    # THEN — full evolution
    run_time_series()
