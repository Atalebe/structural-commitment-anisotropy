#!/usr/bin/env python
"""
Compute the structural dipole in SDSS DR8 (sky dipole on the sphere)
using sdss_dr8_analysis_base_v1.csv as input.

We do NOT try to reconstruct 3D comoving positions here; we treat each
galaxy as a unit vector on the sphere n_hat(RA, DEC) and compute a
mass-weighted mean direction per redshift bin.

Output:
  - sdss_structural_dipole_results.npz
  - sdss_dipole_vectors.npy      (Nbins x 3)
  - sdss_dipole_amplitudes.npy   (Nbins)
"""

import numpy as np
import pandas as pd
import os

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

DATA_FILE = "data/sdss_dr8/sdss_dr8_analysis_base_v1.csv"

# Redshift range and number of bins for the time series
Z_MIN = 0.02
Z_MAX = 0.20
NBINS = 6

# Reliability / quality cut
RELIABLE_COL = "RELIABLE"
RELIABLE_VALUE = 1

# Columns for sky position and redshift
RA_COL = "RA"     # degrees
DEC_COL = "DEC"   # degrees
Z_COL = "Z"

# Mass weighting: use LGM_TOT_P50 (log10 M*) if available
USE_MASS_WEIGHT = True
MASS_COL = "LGM_TOT_P50"

# Minimum galaxies per bin to report something sensible
MIN_BIN_COUNT = 1000


# ----------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------

def sph_to_cart(ra_deg, dec_deg):
    """
    Convert RA, DEC in degrees to Cartesian unit vectors (x, y, z).
    RA:  0 .. 360 deg
    DEC: -90 .. +90 deg
    """
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)

    cos_dec = np.cos(dec)
    x = cos_dec * np.cos(ra)
    y = cos_dec * np.sin(ra)
    z = np.sin(dec)
    return np.vstack((x, y, z)).T  # shape (N, 3)


def vec_to_radec(v):
    """
    Convert a 3D vector to (RA, DEC) in degrees.
    If v is zero, returns (NaN, NaN).
    """
    norm = np.linalg.norm(v)
    if norm == 0.0:
        return np.nan, np.nan

    x, y, z = v / norm
    ra = np.arctan2(y, x)  # range [-pi, pi]
    if ra < 0:
        ra += 2.0 * np.pi
    dec = np.arcsin(z)
    return np.rad2deg(ra), np.rad2deg(dec)


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    print(f"Loading SDSS DR8 catalog from:\n  {os.path.abspath(DATA_FILE)}")
    df = pd.read_csv(DATA_FILE)

    # Basic quality / reliability cut
    if RELIABLE_COL in df.columns:
        mask_rel = (df[RELIABLE_COL] == RELIABLE_VALUE)
        df = df.loc[mask_rel].copy()
        print(f"After RELIABLE == {RELIABLE_VALUE} cut: N = {len(df)}")
    else:
        print(f"[WARN] Column '{RELIABLE_COL}' not found, using all rows.")

    # Drop rows with missing RA, DEC, Z
    df = df.dropna(subset=[RA_COL, DEC_COL, Z_COL])
    print(f"After dropping NaN RA/DEC/Z: N = {len(df)}")

    # Redshift cut
    z = df[Z_COL].to_numpy()
    mask_z = (z >= Z_MIN) & (z <= Z_MAX)
    df = df.loc[mask_z].copy()
    z = df[Z_COL].to_numpy()
    print(f"After {Z_MIN:.3f} <= z <= {Z_MAX:.3f} cut: N = {len(df)}")

    # Sky positions
    ra = df[RA_COL].to_numpy()
    dec = df[DEC_COL].to_numpy()
    n_hat = sph_to_cart(ra, dec)  # shape (N, 3)

    # Weights
    if USE_MASS_WEIGHT and (MASS_COL in df.columns):
        logM = df[MASS_COL].to_numpy()
        # convert log10 M* -> M*
        w = np.power(10.0, logM)
        print(f"Using stellar mass weighting from '{MASS_COL}'.")
    else:
        w = np.ones_like(z)
        if USE_MASS_WEIGHT:
            print(f"[WARN] '{MASS_COL}' not found, falling back to uniform weights.")
        else:
            print("Using uniform weights for all galaxies.")

    # Normalise weights so absolute scale doesn't matter
    w = w.astype(float)
    w_sum_total = np.sum(w)
    if w_sum_total <= 0:
        raise RuntimeError("Total weight <= 0, something is wrong.")
    w /= w_sum_total

    # Redshift bins
    z_edges = np.linspace(Z_MIN, Z_MAX, NBINS + 1)
    z_mids = 0.5 * (z_edges[:-1] + z_edges[1:])

    dip_vecs = np.zeros((NBINS, 3), dtype=float)
    dip_amps = np.zeros(NBINS, dtype=float)
    dip_ra = np.zeros(NBINS, dtype=float)
    dip_dec = np.zeros(NBINS, dtype=float)
    counts = np.zeros(NBINS, dtype=int)

    print("\n===== SDSS STRUCTURAL DIPOLE IN REDSHIFT BINS =====\n")

    for i in range(NBINS):
        z_lo = z_edges[i]
        z_hi = z_edges[i + 1]
        mask_bin = (z >= z_lo) & (z < z_hi)
        N_bin = np.count_nonzero(mask_bin)
        counts[i] = N_bin

        print(f"Bin {i}: z in [{z_lo:.3f}, {z_hi:.3f})  N = {N_bin}")

        if N_bin < MIN_BIN_COUNT:
            print("  [WARN] Too few galaxies in this bin, skipping dipole.")
            dip_vecs[i, :] = 0.0
            dip_amps[i] = np.nan
            dip_ra[i] = np.nan
            dip_dec[i] = np.nan
            continue

        nh_bin = n_hat[mask_bin]
        w_bin = w[mask_bin]

        # Renormalise weights within bin (just to be safe)
        w_bin = w_bin / np.sum(w_bin)

        # Weighted structural dipole vector
        vec = np.sum(w_bin[:, None] * nh_bin, axis=0)
        amp = np.linalg.norm(vec)
        ra_dir, dec_dir = vec_to_radec(vec)

        dip_vecs[i, :] = vec
        dip_amps[i] = amp
        dip_ra[i] = ra_dir
        dip_dec[i] = dec_dir

        print(f"  Dipole amplitude |d_Σ| = {amp:.4f}")
        print(f"  Dipole direction  RA = {ra_dir:.1f} deg, DEC = {dec_dir:.1f} deg")

    # Global dipole over the full z-range
    print("\n===== GLOBAL DIPOLE OVER FULL z-RANGE =====\n")
    nh_all = n_hat
    w_all = w / np.sum(w)  # re-normalise
    vec_all = np.sum(w_all[:, None] * nh_all, axis=0)
    amp_all = np.linalg.norm(vec_all)
    ra_all, dec_all = vec_to_radec(vec_all)
    print(f"Global N = {len(nh_all)}")
    print(f"Global |d_Σ| = {amp_all:.4f}")
    print(f"Global direction  RA = {ra_all:.1f} deg, DEC = {dec_all:.1f} deg")

    # Save results
    out_npz = "sdss_structural_dipole_results.npz"
    np.save("sdss_dipole_vectors.npy", dip_vecs)
    np.save("sdss_dipole_amplitudes.npy", dip_amps)

    np.savez(
        out_npz,
        z_edges=z_edges,
        z_mids=z_mids,
        counts=counts,
        dipole_vectors=dip_vecs,
        dipole_amplitudes=dip_amps,
        dipole_ra=dip_ra,
        dipole_dec=dip_dec,
        global_vector=vec_all,
        global_amplitude=amp_all,
        global_ra=ra_all,
        global_dec=dec_all,
        config=dict(
            Z_MIN=Z_MIN,
            Z_MAX=Z_MAX,
            NBINS=NBINS,
            USE_MASS_WEIGHT=USE_MASS_WEIGHT,
            MASS_COL=MASS_COL,
        ),
    )

    print(f"\nSaved:")
    print(f"  {out_npz}")
    print(f"  sdss_dipole_vectors.npy")
    print(f"  sdss_dipole_amplitudes.npy")
    print("\nDone.")


if __name__ == "__main__":
    main()
