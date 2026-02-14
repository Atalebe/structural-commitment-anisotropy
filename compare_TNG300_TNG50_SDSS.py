#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

TNG300_EXP = "expansion_dipole_results.npz"
TNG50_EXP  = "tng50_expansion_dipole_results.npz"

# Structural dipole time series (your saved arrays)
TNG300_DSIGMA = "tng300_dipole_amplitudes.npy"
TNG50_DSIGMA  = "tng50_dipole_amplitudes.npy"

# SDSS: use the normalized mask-aware output (NOT the raw isotropic-null one)
SDSS_MASK_NORMED = "sdss_structural_dipole_mask_null_results_normed.npz"

OUTFIG = "fig_TNG300_TNG50_SDSS.png"

def load_npz(path):
    data = np.load(path, allow_pickle=True)
    return data

def load_tng_expansion(path, is_tng300=False):
    data = load_npz(path)
    keys = list(data.keys())
    print(f"[{'TNG300' if is_tng300 else 'TNG50'}] {path} loaded")
    print(f"  keys: {keys}")

    if is_tng300:
        snaps = data["snapshots"]
        z = data["z"]
        dip = data["deltaH_dipole_over_H"]
    else:
        snaps = data["snapshots"]
        z = data["redshifts"]
        dip = data["deltaH_dipole_over_H"]

    return snaps.astype(int), z.astype(float), dip.astype(float)

def load_dsigma_series(path, snapshots_expected=None):
    arr = np.load(path)
    arr = np.asarray(arr, dtype=float).ravel()
    if snapshots_expected is not None and len(arr) != len(snapshots_expected):
        raise RuntimeError(f"Expected {len(snapshots_expected)} amplitudes, got {len(arr)} in {path}")
    return arr

def pick_dsigma_at_snap(snapshots, dsigma, target_snap=72):
    snapshots = np.asarray(snapshots, dtype=int)
    dsigma = np.asarray(dsigma, dtype=float)
    if target_snap in snapshots:
        i = int(np.where(snapshots == target_snap)[0][0])
        return float(dsigma[i])
    # fallback: closest snapshot
    j = int(np.argmin(np.abs(snapshots - target_snap)))
    return float(dsigma[j])

def load_sdss_mask_normed(path):
    data = load_npz(path)
    keys = list(data.keys())
    print(f"[SDSS] {path} loaded")
    print(f"  keys: {keys}")

    D_geo_vec = data["global_D_geo_vec"].astype(float)
    D_obs_vec = data["global_D_obs_vec"].astype(float)
    D_geo = float(data["global_D_geo"])
    D_obs = float(data["global_D_obs"])
    angle = float(data["global_angle_deg"])

    D_phys_vec = D_obs_vec - D_geo_vec
    D_phys = float(np.linalg.norm(D_phys_vec))

    return dict(D_geo=D_geo, D_obs=D_obs, D_phys=D_phys, angle_deg=angle)

def main():
    # Expansion dipoles
    s300, z300, dH300 = load_tng_expansion(TNG300_EXP, is_tng300=True)
    s50,  z50,  dH50  = load_tng_expansion(TNG50_EXP,  is_tng300=False)

    print(f"  [TNG300] snapshots: {s300}")
    print(f"  [TNG300] redshifts: {z300}")
    print(f"  [TNG50 ] snapshots: {s50}")
    print(f"  [TNG50 ] redshifts: {z50}")

    # Structural dipoles: ensure the arrays match your snapshot lists
    ds300_series = load_dsigma_series(TNG300_DSIGMA, snapshots_expected=s300)
    ds50_series  = load_dsigma_series(TNG50_DSIGMA,  snapshots_expected=s50)

    ds300 = pick_dsigma_at_snap(s300, ds300_series, target_snap=72)
    ds50  = pick_dsigma_at_snap(s50,  ds50_series,  target_snap=72)

    print("\n[SUMMARY: structural dipoles used for panel (b)]")
    print(f"  TNG300 (z~0.5, number-weighted) : d_Sigma = {ds300:.4f}")
    print(f"  TNG50  (z~0.5, number-weighted) : d_Sigma = {ds50:.4f}")

    # SDSS mask-aware
    sdss = load_sdss_mask_normed(SDSS_MASK_NORMED)
    print(f"  SDSS mask-aware:")
    print(f"    D_geo  = {sdss['D_geo']:.4f}")
    print(f"    D_obs  = {sdss['D_obs']:.4f}")
    print(f"    D_phys = {sdss['D_phys']:.4f}")
    print(f"    angle(D_geo,D_obs) = {sdss['angle_deg']:.2f} deg")

    # Figure
    fig = plt.figure(figsize=(10, 4))

    # Panel A: expansion dipole vs z
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(z300, np.abs(dH300), marker="o", label="TNG300 |dH/H|")
    ax1.plot(z50,  np.abs(dH50),  marker="o", label="TNG50  |dH/H|")
    ax1.set_xlabel("Redshift z")
    ax1.set_ylabel("|dH/H|")
    ax1.set_title("Expansion dipole vs redshift")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Panel B: structural dipoles comparison
    ax2 = fig.add_subplot(1, 2, 2)
    labels = ["TNG300\n(z~0.5)\n(number)", "TNG50\n(z~0.5)\n(number)", "SDSS\n(raw obs)\n(mass)", "SDSS\n(mask geo)\n(unweighted)", "SDSS\n(excess)\n(D_phys)"]
    vals = [ds300, ds50, sdss["D_obs"], sdss["D_geo"], sdss["D_phys"]]
    ax2.bar(range(len(vals)), vals)
    ax2.set_xticks(range(len(vals)))
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("Dipole amplitude")
    ax2.set_title("Structural dipoles: sims vs SDSS (mask-aware)")
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTFIG, dpi=200)
    print(f"\nSaved combined figure: {OUTFIG}")
    print("Done.")

if __name__ == "__main__":
    main()

