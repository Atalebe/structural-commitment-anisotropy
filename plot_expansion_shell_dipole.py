#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# Load shell dipole results
# ----------------------------------------------------------
data = np.load("expansion_shell_dipole_results.npz")

print("Loaded expansion_shell_dipole_results.npz")
print("Keys:", data.files)

snapshots       = data["snapshots"]        # (Nsnap,)
redshifts       = data["redshifts"]        # (Nsnap,)
radial_bins     = data["radial_bins"]      # (Nshell, 2) → [r_min, r_max] per shell
counts          = data["counts"]           # (Nsnap, Nshell)
dip_amp_over_H  = data["dip_amp_over_H"]   # (Nsnap, Nshell)
dip_par_over_H  = data["dip_par_over_H"]   # (Nsnap, Nshell)
dip_perp_over_H = data["dip_perp_over_H"]  # (Nsnap, Nshell)

Nsnap, Nshell = dip_amp_over_H.shape
print(f"Nsnap  = {Nsnap}")
print(f"Nshell = {Nshell}")
print("Radial bins (Mpc/h):", radial_bins)

z = np.array(redshifts)

# ----------------------------------------------------------
# Outer shell only (last row of radial_bins)
# ----------------------------------------------------------
outer_idx         = Nshell - 1
r_min_outer, r_max_outer = radial_bins[outer_idx]
print(f"Using outer shell index {outer_idx}: r in [{r_min_outer}, {r_max_outer}] Mpc/h")

babs_outer = dip_amp_over_H[:, outer_idx]
bpar_outer = dip_par_over_H[:, outer_idx]

fig1, ax1 = plt.subplots(figsize=(6, 4))

ax1.plot(z, babs_outer, marker="o", label=r"$|b|/H$ (outer shell)")
ax1.plot(z, bpar_outer, marker="s", linestyle="--",
         label=r"$b_{\parallel}/H$ (outer shell)")

ax1.set_xlabel(r"$z$")
ax1.set_ylabel(r"Dipole amplitude / $H_{\rm FRW}$")
ax1.set_title(
    rf"Expansion dipole in outer shell "
    rf"({r_min_outer:.0f}–{r_max_outer:.0f} $h^{{-1}}$ Mpc)"
)

# high z on the left
ax1.invert_xaxis()
ax1.grid(True, alpha=0.3)
ax1.legend()
fig1.tight_layout()

outname1 = "fig_expansion_shell_dipole_outer.png"
fig1.savefig(outname1, dpi=200)
print(f"Saved {outname1}")

# ----------------------------------------------------------
# All shells: |b|/H and b_parallel/H vs z
# ----------------------------------------------------------
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

for s in range(Nshell):
    rmin, rmax = radial_bins[s]
    label = rf"{rmin:.0f}–{rmax:.0f} $h^{{-1}}$ Mpc"

    ax2a.plot(z, dip_amp_over_H[:, s], marker="o", linestyle="-", label=label)
    ax2b.plot(z, dip_par_over_H[:, s], marker="s", linestyle="--", label=label)

ax2a.set_xlabel(r"$z$")
ax2b.set_xlabel(r"$z$")

ax2a.set_ylabel(r"$|b|/H_{\rm FRW}$")
ax2b.set_ylabel(r"$b_{\parallel}/H_{\rm FRW}$")

ax2a.set_title(r"Shell dipole amplitude $|b|/H$")
ax2b.set_title(r"Aligned component $b_{\parallel}/H$")

ax2a.invert_xaxis()
ax2b.invert_xaxis()

ax2a.grid(True, alpha=0.3)
ax2b.grid(True, alpha=0.3)

ax2a.legend(fontsize=8)
ax2b.legend(fontsize=8)

fig2.tight_layout()
outname2 = "fig_expansion_shell_dipole_all.png"
fig2.savefig(outname2, dpi=200)
print(f"Saved {outname2}")

print("Done.")
