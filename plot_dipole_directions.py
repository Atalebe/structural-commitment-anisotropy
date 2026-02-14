#!/usr/bin/env python
"""
plot_dipole_directions.py

Plot the cosine of the angles between dipole direction vectors across snapshots.
"""

import numpy as np
import matplotlib.pyplot as plt

SNAPSHOTS = np.array([33, 40, 50, 59, 67, 72, 78, 91])

vecs = np.load("dipole_vectors.npy")  # shape (N_snap, 3)
if vecs.shape[0] != SNAPSHOTS.size:
    raise RuntimeError(
        f"dipole_vectors.npy has {vecs.shape[0]} rows, "
        f"but SNAPSHOTS has {SNAPSHOTS.size} entries."
    )

# Normalise to unit vectors
norms = np.linalg.norm(vecs, axis=1, keepdims=True)
hat = vecs / norms

# Cosine similarity matrix
cos_mat = hat @ hat.T  # (N_snap, N_snap)

print("Directional cosine matrix:\n")
print(np.round(cos_mat, 3))

plt.figure(figsize=(5,4))
im = plt.imshow(cos_mat, vmin=-1.0, vmax=1.0, origin="lower")
plt.colorbar(im, label="cosine of angle")

plt.xticks(range(len(SNAPSHOTS)), SNAPSHOTS)
plt.yticks(range(len(SNAPSHOTS)), SNAPSHOTS)
plt.xlabel("Snapshot")
plt.ylabel("Snapshot")
plt.title("Directional coherence of structural dipole")

plt.tight_layout()
plt.savefig("fig_dipole_directions.png", dpi=200)
plt.close()

print("Saved figure: fig_dipole_directions.png")
