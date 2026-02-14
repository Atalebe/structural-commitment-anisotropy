#!/usr/bin/env python3
"""
tng50_plot_expansion_shell_dipole.py

Plot the radial-shell dipole fit for TNG50, using the results from
`tng50_expansion_shell_fit.py`.

We show:
- Outer shell (largest radii) b_parallel/H vs redshift
- All shells b_parallel/H vs redshift (for a quick sanity check)
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    fname = "tng50_expansion_shell_dipole_results.npz"
    data = np.load(fname)

    print(f"Loaded {fname}")
    print("Keys:", list(data.keys()))

    snapshots = data["snapshots"]            # (Nsnap,)
    redshifts = data["redshifts"]            # (Nsnap,)
    radial_bins = data["radial_bins"]        # (Nshell, 2)  [r_min, r_max]
    counts = data["counts"]                  # (Nsnap, Nshell)
    dip_par_over_H = data["dip_par_over_H"]  # (Nsnap, Nshell)
    dip_amp_over_H = data["dip_amp_over_H"]  # (Nsnap, Nshell)

    Nsnap = snapshots.shape[0]
    Nshell = radial_bins.shape[0]

    print(f"Nsnap  = {Nsnap}")
    print(f"Nshell = {Nshell}")
    print("Radial bins (Mpc/h):")
    for i in range(Nshell):
        rmin, rmax = radial_bins[i]
        print(f"  shell {i}: [{rmin:.1f}, {rmax:.1f}]")

    # Choose outer shell (largest radii)
    outer_idx = Nshell - 1
    rmin_outer, rmax_outer = radial_bins[outer_idx]
    print(
        f"Using outer shell index {outer_idx}: "
        f"r in [{rmin_outer:.1f}, {rmax_outer:.1f}] Mpc/h"
    )

    dip_par_outer = dip_par_over_H[:, outer_idx]
    dip_amp_outer = dip_amp_over_H[:, outer_idx]
    counts_outer = counts[:, outer_idx]

    print("Outer shell quick summary:")
    for i in range(Nsnap):
        print(
            f"  snap {snapshots[i]:3d}, z={redshifts[i]:.1f} : "
            f"|b|/H = {dip_amp_outer[i]:+.3e}, "
            f"b_parallel/H = {dip_par_outer[i]:+.3e}, "
            f"N_shell = {int(counts_outer[i])}"
        )

    # --- Figure 1: outer shell only, parallel component and amplitude ---
    fig1, ax1 = plt.subplots(figsize=(6, 4))

    ax1.plot(
        redshifts,
        dip_par_outer,
        marker="o",
        linestyle="-",
        label="b_parallel/H (outer shell)",
    )
    ax1.plot(
        redshifts,
        dip_amp_outer,
        marker="s",
        linestyle="--",
        label="|b|/H (outer shell)",
    )

    ax1.set_xlabel("Redshift z")
    ax1.set_ylabel("Shell dipole / H")
    ax1.set_title(
        f"TNG50 expansion dipole (outer shell)\n"
        f"r in [{rmin_outer:.0f}, {rmax_outer:.0f}] Mpc/h"
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.invert_xaxis()

    out1 = "tng50_fig_expansion_shell_dipole_outer.png"
    fig1.tight_layout()
    fig1.savefig(out1, dpi=200)
    print(f"Saved {out1}")

    # --- Figure 2: all shells, b_parallel/H vs z ---
    fig2, ax2 = plt.subplots(figsize=(6, 4))

    for i in range(Nshell):
        rmin, rmax = radial_bins[i]
        ax2.plot(
            redshifts,
            dip_par_over_H[:, i],
            marker="o",
            linestyle="-",
            label=f"shell {i}: [{rmin:.0f},{rmax:.0f}] Mpc/h",
        )

    ax2.set_xlabel("Redshift z")
    ax2.set_ylabel("b_parallel/H")
    ax2.set_title("TNG50 expansion dipole: all radial shells")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=7)
    ax2.invert_xaxis()

    out2 = "tng50_fig_expansion_shell_dipole_all.png"
    fig2.tight_layout()
    fig2.savefig(out2, dpi=200)
    print(f"Saved {out2}")

    print("Done.")


if __name__ == "__main__":
    main()
