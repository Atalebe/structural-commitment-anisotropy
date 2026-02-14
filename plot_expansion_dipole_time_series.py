#!/usr/bin/env python
"""
plot_expansion_dipole_time_series.py

Quick look at the expansion dipole time series produced by
expansion_dipole_time_series.py.
"""

import numpy as np
import matplotlib.pyplot as plt

fname = "expansion_dipole_results.npz"
data = np.load(fname)

snap = data["snapshots"]
z    = data["z"]

d_all   = data["deltaH_all_over_H"]
d_dip   = data["deltaH_dipole_over_H"]

print("Snapshots:", snap)
print("Redshifts:", z)
print("deltaH_all/H:", d_all)
print("deltaH_dipole/H:", d_dip)

# Sort by redshift (just in case)
idx = np.argsort(z)
z_plot    = z[idx]
d_all_p   = d_all[idx]
d_dip_p   = d_dip[idx]

plt.figure()
plt.plot(z_plot, d_all_p, marker="o", label=r"$\delta H_{\rm all}/H$")
plt.plot(z_plot, d_dip_p, marker="s", label=r"$\Delta H_{\rm dipole}/H$")

plt.gca().invert_xaxis()  # high z on left, low z on right
plt.xlabel("Redshift z")
plt.ylabel(r"Expansion anomaly (dimensionless)")
plt.title("Directional expansion along structural dipole")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("fig_expansion_dipole_time_series.png", dpi=200)
print("Saved figure: fig_expansion_dipole_time_series.png")
