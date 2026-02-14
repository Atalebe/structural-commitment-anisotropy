#!/usr/bin/env python
"""
plot_mass_thresholds.py

Simple bar chart of dipole amplitude vs subhalo mass cut at snapshot 72.
Values are taken from the current mass-threshold test output.
"""

import numpy as np
import matplotlib.pyplot as plt

labels = ["No cut", r"$>10^{10}$", r"$>10^{11}$", r"$>10^{12}$"]
amps = np.array([0.07444, 0.08247, 0.09116, 0.11979])

x = np.arange(len(labels))

plt.figure(figsize=(5,4))
plt.bar(x, amps)

plt.xticks(x, labels)
plt.ylabel("Dipole amplitude")
plt.title("Structural dipole vs mass threshold (snapshot 72)")

for i, a in enumerate(amps):
    plt.text(i, a + 0.002, f"{a:.3f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig("fig_mass_thresholds.png", dpi=200)
plt.close()

print("Saved figure: fig_mass_thresholds.png")
