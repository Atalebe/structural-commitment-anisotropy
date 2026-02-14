#!/usr/bin/env python3
"""
tng50_plot_expansion_dipole_time_series.py

Plot the TNG50 expansion monopole and hemispheric dipole (deltaH/H)
as a function of redshift, using the results from
`tng50_expansion_dipole_time_series.py`.
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    fname = "tng50_expansion_dipole_results.npz"
    data = np.load(fname)

    print(f"Loaded {fname}")
    print("Keys:", list(data.keys()))

    snapshots = data["snapshots"]             # shape (Nsnap,)
    redshifts = data["redshifts"]             # shape (Nsnap,)
    deltaH_all_over_H = data["deltaH_all_over_H"]       # (Nsnap,)
    deltaH_dipole_over_H = data["deltaH_dipole_over_H"] # (Nsnap,)

    print("Snapshots:", snapshots)
    print("Redshifts:", redshifts)
    print("deltaH_all/H:", deltaH_all_over_H)
    print("deltaH_dipole/H:", deltaH_dipole_over_H)

    # Monopole + dipole vs z
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(
        redshifts,
        deltaH_all_over_H,
        marker="o",
        linestyle="-",
        label=r"TNG50 $\,\Delta H_{\rm all}/H$",
    )
    ax.plot(
        redshifts,
        deltaH_dipole_over_H,
        marker="s",
        linestyle="--",
        label=r"TNG50 $\,\Delta H_{\rm dip}/H$",
    )

    ax.set_xlabel("Redshift $z$")
    ax.set_ylabel(r"$\Delta H / H$")
    ax.set_title("TNG50 expansion monopole and dipole vs redshift")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.invert_xaxis()  # high z on the left, like the TNG300 plot

    outname = "tng50_fig_expansion_dipole_time_series.png"
    fig.tight_layout()
    fig.savefig(outname, dpi=200)
    print(f"Saved figure: {outname}")


if __name__ == "__main__":
    main()
