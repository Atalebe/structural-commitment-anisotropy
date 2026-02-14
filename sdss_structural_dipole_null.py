#!/usr/bin/env python
"""
sdss_structural_dipole_null.py

Constructs a null distribution for the SDSS DR8 structural dipole
by randomizing galaxy directions isotropically on the sky, while
preserving the number of galaxies, redshift distribution, and
mass weights.

This is an *isotropic-sky* null, i.e. it answers:
"How big would |d_Σ| be if the same galaxies (and weights)
were distributed isotropically rather than in the SDSS footprint?"
"""

import os
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

# Path to the SDSS DR8 base analysis CSV (same file you used above)
DATA_PATH = os.path.join(
    "data",
    "sdss_dr8",
    "sdss_dr8_analysis_base_v1.csv"
)

# Redshift range and binning
Z_MIN = 0.020
Z_MAX = 0.200
NBINS = 6   # same as your earlier run

# Column names
COL_RELIABLE   = "RELIABLE"
COL_RA_DEG     = "RA"
COL_DEC_DEG    = "DEC"
COL_Z          = "Z"
COL_LGM_TOT    = "LGM_TOT_P50"

# Null distribution parameters
N_NULL = 1000   # you can increase to 2000 if you want
MIN_PER_BIN = 1000  # minimum galaxies in bin to bother with a null

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def random_unit_vectors(N):
    """
    Draw N random unit vectors isotropically on the sphere.
    """
    v = np.random.normal(size=(N, 3))
    norm = np.linalg.norm(v, axis=1)
    v /= norm[:, None]
    return v


def compute_weighted_dipole(vecs, weights):
    """
    vecs: (N, 3) array of unit vectors.
    weights: (N,) array of weights (e.g. stellar mass).
    Returns: (d_vec, d_amp)
    """
    wsum = np.sum(weights)
    if wsum == 0.0 or vecs.shape[0] == 0:
        return np.zeros(3), 0.0
    d_vec = np.sum(vecs * weights[:, None], axis=0) / wsum
    d_amp = np.linalg.norm(d_vec)
    return d_vec, d_amp


def vec_to_radec(vec):
    """
    Convert a 3D unit vector to (RA_deg, DEC_deg).
    """
    x, y, z = vec
    # DEC = arcsin(z)
    dec = np.degrees(np.arcsin(z))
    # RA = atan2(y, x) in [0, 360)
    ra = np.degrees(np.arctan2(y, x))
    ra = ra % 360.0
    return ra, dec


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    # -----------------------------
    # Load and select SDSS sample
    # -----------------------------
    print("Loading SDSS DR8 catalog from:")
    print(" ", os.path.abspath(DATA_PATH))

    df = pd.read_csv(DATA_PATH)

    print(f"Initial N (raw): {len(df)}")

    # Reliability
    if COL_RELIABLE in df.columns:
        df = df[df[COL_RELIABLE] == 1]
        print(f"After {COL_RELIABLE} == 1 cut: N = {len(df)}")
    else:
        print(f"[WARN] Column {COL_RELIABLE} not found, skipping reliability cut.")

    # Drop NaNs in RA/DEC/Z and LGM
    df = df.dropna(subset=[COL_RA_DEG, COL_DEC_DEG, COL_Z, COL_LGM_TOT])
    print(f"After dropping NaN RA/DEC/Z/{COL_LGM_TOT}: N = {len(df)}")

    # Redshift cut
    z = df[COL_Z].to_numpy()
    mask_z = (z >= Z_MIN) & (z <= Z_MAX)
    df = df[mask_z].copy()
    z = df[COL_Z].to_numpy()
    print(f"After {Z_MIN:.3f} <= z <= {Z_MAX:.3f} cut: N = {len(df)}")

    if len(df) == 0:
        print("[ERROR] No galaxies survive cuts. Aborting.")
        return

    # Positions and weights
    ra_deg  = df[COL_RA_DEG].to_numpy()
    dec_deg = df[COL_DEC_DEG].to_numpy()

    # Stellar mass weights: w ~ 10^{LGM_TOT_P50}
    lgm = df[COL_LGM_TOT].to_numpy()
    weights = 10.0**lgm

    # Convert observed RA/DEC to unit vectors
    ra_rad  = np.radians(ra_deg)
    dec_rad = np.radians(dec_deg)
    cos_dec = np.cos(dec_rad)
    x = cos_dec * np.cos(ra_rad)
    y = cos_dec * np.sin(ra_rad)
    z3 = np.sin(dec_rad)
    vec_obs = np.column_stack([x, y, z3])

    print("\n===== SDSS STRUCTURAL DIPOLE NULL TEST (ISOTROPIC SKY) =====\n")

    N_total = len(df)
    print(f"Total galaxies used: N = {N_total}")

    # --------------------------------------------------------
    # Compute observed global dipole
    # --------------------------------------------------------
    d_vec_global, d_amp_global = compute_weighted_dipole(vec_obs, weights)
    ra_g, dec_g = vec_to_radec(d_vec_global)
    print("Observed GLOBAL dipole:")
    print(f"  |d_Σ|_obs = {d_amp_global:.4f}")
    print(f"  RA, DEC   = ({ra_g:.1f} deg, {dec_g:.1f} deg)")

    # --------------------------------------------------------
    # Define redshift bins and compute observed per-bin dipoles
    # --------------------------------------------------------
    z_edges = np.linspace(Z_MIN, Z_MAX, NBINS + 1)
    bin_labels = [f"[{z_edges[i]:.3f}, {z_edges[i+1]:.3f})"
                  for i in range(NBINS)]

    print("\nObserved dipole per redshift bin:")
    d_amp_bins_obs = np.full(NBINS, np.nan)
    ra_bins_obs    = np.full(NBINS, np.nan)
    dec_bins_obs   = np.full(NBINS, np.nan)
    N_bins         = np.zeros(NBINS, dtype=int)
    idx_bins       = []

    for i in range(NBINS):
        zmin = z_edges[i]
        zmax = z_edges[i+1]
        mask_bin = (z >= zmin) & (z < zmax)
        idx = np.where(mask_bin)[0]
        idx_bins.append(idx)
        N_bins[i] = len(idx)

        if len(idx) == 0:
            print(f"  Bin {i} {bin_labels[i]}: N = 0 -> skipped.")
            continue

        d_vec_i, d_amp_i = compute_weighted_dipole(vec_obs[idx], weights[idx])
        ra_i, dec_i = vec_to_radec(d_vec_i)

        d_amp_bins_obs[i] = d_amp_i
        ra_bins_obs[i]    = ra_i
        dec_bins_obs[i]   = dec_i

        print(
            f"  Bin {i} {bin_labels[i]}: N = {N_bins[i]:6d}, "
            f"|d_Σ|_obs = {d_amp_i:.4f}, (RA,DEC) = ({ra_i:.1f}, {dec_i:.1f})"
        )

    # --------------------------------------------------------
    # Build isotropic null distribution
    # --------------------------------------------------------
    print("\nBuilding isotropic null distribution...")
    print(f"  N_NULL = {N_NULL}")

    # Global null
    null_global = np.empty(N_NULL, dtype=float)

    for r in range(N_NULL):
        if (r+1) % max(1, N_NULL // 10) == 0:
            print(f"  Global null realisation {r+1}/{N_NULL}")
        v_rand = random_unit_vectors(N_total)   # (N_total, 3)
        _, d_amp_r = compute_weighted_dipole(v_rand, weights)
        null_global[r] = d_amp_r

    mu_global = np.mean(null_global)
    sig_global = np.std(null_global, ddof=1)
    zscore_global = (d_amp_global - mu_global) / sig_global

    print("\n===== GLOBAL NULL RESULTS =====")
    print(f"  <|d_Σ|>_null   = {mu_global:.4f}")
    print(f"  sigma_null     = {sig_global:.4f}")
    print(f"  |d_Σ|_obs      = {d_amp_global:.4f}")
    print(f"  z-score        = {zscore_global:.2f} sigma")

    # Per-bin nulls
    null_bins = np.full((NBINS, N_NULL), np.nan)
    mu_bins   = np.full(NBINS, np.nan)
    sig_bins  = np.full(NBINS, np.nan)
    z_bins    = np.full(NBINS, np.nan)

    print("\n===== BINNED NULL RESULTS =====")
    for i in range(NBINS):
        N_i = N_bins[i]
        if N_i < MIN_PER_BIN:
            if N_i == 0:
                print(f"  Bin {i} {bin_labels[i]}: N = 0 -> skipped.")
            else:
                print(
                    f"  Bin {i} {bin_labels[i]}: N = {N_i} < {MIN_PER_BIN}, "
                    "skipping null."
                )
            continue

        w_i = weights[idx_bins[i]]

        print(
            f"  Bin {i} {bin_labels[i]}: N = {N_i}, building null "
            f"({N_NULL} realisations)..."
        )

        for r in range(N_NULL):
            v_rand = random_unit_vectors(N_i)
            _, d_amp_r = compute_weighted_dipole(v_rand, w_i)
            null_bins[i, r] = d_amp_r

        mu_i = np.nanmean(null_bins[i])
        sig_i = np.nanstd(null_bins[i], ddof=1)
        mu_bins[i] = mu_i
        sig_bins[i] = sig_i

        d_obs_i = d_amp_bins_obs[i]
        if np.isfinite(d_obs_i):
            z_i = (d_obs_i - mu_i) / sig_i
            z_bins[i] = z_i
            print(
                f"    <|d_Σ|>_null = {mu_i:.4f}, "
                f"sigma_null = {sig_i:.4f}, "
                f"|d_Σ|_obs = {d_obs_i:.4f}, "
                f"z = {z_i:.2f} sigma"
            )
        else:
            print("    [WARN] No observed dipole for this bin (NaN).")

    # --------------------------------------------------------
    # Save results to NPZ
    # --------------------------------------------------------
    out_path = "sdss_structural_dipole_null_results.npz"
    np.savez(
        out_path,
        # global
        dSigma_obs_global=d_amp_global,
        dSigma_null_global=null_global,
        mu_global=mu_global,
        sigma_global=sig_global,
        zscore_global=zscore_global,
        # binning
        z_edges=z_edges,
        N_bins=N_bins,
        dSigma_obs_bins=d_amp_bins_obs,
        ra_bins_obs=ra_bins_obs,
        dec_bins_obs=dec_bins_obs,
        mu_bins=mu_bins,
        sigma_bins=sig_bins,
        zscore_bins=z_bins,
    )

    print(f"\nSaved null results to: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
