#!/usr/bin/env python
"""
plot_dipole_time_series.py

Plot structural dipole amplitude vs snapshot, overlaid on the null mean and
one-sigma band from the random-orientation isotropy baseline.
"""

import numpy as np
import matplotlib.pyplot as plt

# Snapshots, in the same order as used in dipole_time_series.py
SNAPSHOTS = np.array([33, 40, 50, 59, 67, 72, 78, 91])

# Files produced by your other scripts
amps = np.load("dipole_amplitudes.npy")  # shape (8,)
null_dist = np.load("structural_dipole_null_distribution.npy")

mu = float(null_dist.mean())
sigma = float(null_dist.std())

print("Time series amplitudes:", amps)
print(f"Null mean  : {mu:.6f}")
print(f"Null sigma : {sigma:.6f}")

plt.figure(figsize=(6,4))

plt.plot(SNAPSHOTS, amps, marker="o", linewidth=2, label="Structural dipole")

# Null baseline and one sigma band
plt.axhline(mu, linestyle="--", label=f"Null mean ({mu:.3f})")
plt.fill_between(SNAPSHOTS, mu - sigma, mu + sigma, alpha=0.2,
                 label=r"Null $\pm 1\sigma$")

plt.xlabel("Snapshot")
plt.ylabel("Dipole amplitude")
plt.title("Structural dipole time series vs null baseline")
plt.legend()
plt.tight_layout()
plt.savefig("fig_dipole_time_series.png", dpi=200)
plt.close()

print("Saved figure: fig_dipole_time_series.png")
