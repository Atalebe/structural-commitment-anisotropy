#!/usr/bin/env python3
"""
compare_expansion_dipole_TNG300_TNG50.py

Overlay TNG300 and TNG50 expansion monopole and dipole (deltaH/H)
as a function of redshift, using:

- expansion_dipole_results.npz          (TNG300)
- tng50_expansion_dipole_results.npz    (TNG50)
"""

import numpy as np
import matplotlib.pyplot as plt


def load_tng300(fname="expansion_dipole_results.npz"):
    data = np.load(fname)
    print(f"[TNG300] Loaded {fname}")
    print("  keys:", list(data.keys()))

    snaps = data["snapshots"]
    # Handle both "z" (current file) and "redshifts" (if ever renamed)
    if "redshifts" in data.files:
        z = data["redshifts"]
    else:
        z = data["z"]

    d_all = data["deltaH_all_over_H"]
    d_dip = data["deltaH_dipole_over_H"]

    print("  snapshots:", snaps)
    print("  redshifts:", z)
    print("  deltaH_all/H:", d_all)
    print("  deltaH_dipole/H:", d_dip)

    return snaps, z, d_all, d_dip


def load_tng50(fname="tng50_expansion_dipole_results.npz"):
    data = np.load(fname)
    print(f"[TNG50] Loaded {fname}")
    print("  keys:", list(data.keys()))

    snaps = data["snapshots"]
    z = data["redshifts"]
    d_all = data["deltaH_all_over_H"]
    d_dip = data["deltaH_dipole_over_H"]

    print("  snapshots:", snaps)
    print("  redshifts:", z)
    print("  deltaH_all/H:", d_all)
    print("  deltaH_dipole/H:", d_dip)

    return snaps, z, d_all, d_dip


def main():
    s300, z300, d300_all, d300_dip = load_tng300()
    s50, z50, d50_all, d50_dip = load_tng50()

    # Two-panel figure: left monopole, right dipole
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

    # --- Monopole ---
    ax1.plot(
        z300,
        d300_all,
        marker="o",
        linestyle="-",
        label="TNG300",
    )
    ax1.plot(
        z50,
        d50_all,
        marker="s",
        linestyle="--",
        label="TNG50",
    )
    ax1.set_xlabel("Redshift z")
    ax1.set_ylabel("Delta H_all / H")
    ax1.set_title("Monopole shift of H_eff")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)
    ax1.invert_xaxis()

    # --- Dipole ---
    ax2.plot(
        z300,
        d300_dip,
        marker="o",
        linestyle="-",
        label="TNG300",
    )
    ax2.plot(
        z50,
        d50_dip,
        marker="s",
        linestyle="--",
        label="TNG50",
    )
    ax2.set_xlabel("Redshift z")
    ax2.set_ylabel("Delta H_dipole / H")
    ax2.set_title("Hemispheric dipole along structural axis")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    ax2.invert_xaxis()

    fig.suptitle("Expansion monopole and dipole: TNG300 vs TNG50", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    outname = "fig_compare_expansion_dipole_TNG300_TNG50.png"
    fig.savefig(outname, dpi=200)
    print(f"Saved {outname}")


if __name__ == "__main__":
    main()
