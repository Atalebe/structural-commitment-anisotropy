#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd

# -----------------------------
# Config
# -----------------------------
CATALOG = "/home/satal/structural-commitment-anisotropy/data/sdss_dr8/sdss_dr8_analysis_base_v1.csv"
Z_MIN, Z_MAX = 0.020, 0.100
Z_EDGES = np.array([0.02, 0.05, 0.08])  # bins: [0.02,0.05), [0.05,0.08)
N_NULL = 1000
SEED = 12345
WEIGHT_COL = "LGM_TOT_P50"  # log10 stellar mass
OUT_NPZ = "sdss_structural_dipole_mask_null_results_normed.npz"

# -----------------------------
# Helpers
# -----------------------------
def radec_to_unitvec(ra_deg, dec_deg):
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.stack([x, y, z], axis=1)  # (N,3)

def unitvec_to_radec(v):
    v = np.asarray(v, dtype=float)
    v = v / (np.linalg.norm(v) + 1e-30)
    x, y, z = v
    ra = np.rad2deg(np.arctan2(y, x)) % 360.0
    dec = np.rad2deg(np.arcsin(np.clip(z, -1.0, 1.0)))
    return ra, dec

def normalized_dipole(nhat, weights):
    """
    D = sum_i w_i * n_i / sum_i w_i
    """
    w = np.asarray(weights, dtype=float)
    s = np.sum(w)
    if not np.isfinite(s) or s <= 0:
        raise RuntimeError("Sum of weights is non-positive or invalid.")
    v = np.sum(nhat * w[:, None], axis=0) / s
    amp = float(np.linalg.norm(v))
    return v, amp

def shuffled_null_amps(nhat, weights, n_null, rng):
    """
    Keep positions fixed (mask preserved), shuffle weights across objects.
    Return distribution of |D| under shuffling.
    """
    w = np.asarray(weights, dtype=float)
    amps = np.zeros(n_null, dtype=float)
    for k in range(n_null):
        w_shuf = rng.permutation(w)
        _, a = normalized_dipole(nhat, w_shuf)
        amps[k] = a
        if (k + 1) % 200 == 0:
            print(f"  Realisation {k+1}/{n_null}")
    return amps

def zscore(obs, dist):
    mu = float(np.mean(dist))
    sig = float(np.std(dist, ddof=1))
    if sig <= 0:
        return mu, sig, np.inf
    return mu, sig, (obs - mu) / sig

def load_and_filter():
    print("Loading SDSS DR8 catalog from:")
    print(f"  {CATALOG}")
    df = pd.read_csv(CATALOG)
    print(f"Initial N (raw): {len(df)}")

    if "RELIABLE" in df.columns:
        df = df[df["RELIABLE"] == 1].copy()
        print(f"After RELIABLE == 1 cut: N = {len(df)}")

    need = ["RA", "DEC", "Z", WEIGHT_COL]
    df = df.dropna(subset=need).copy()
    print(f"After dropping NaN RA/DEC/Z/{WEIGHT_COL}: N = {len(df)}")

    df = df[(df["Z"] >= Z_MIN) & (df["Z"] <= Z_MAX)].copy()
    print(f"After {Z_MIN:.3f} <= z <= {Z_MAX:.3f} cut: N = {len(df)}")

    return df

# -----------------------------
# Main
# -----------------------------
def main():
    rng = np.random.default_rng(SEED)

    df = load_and_filter()
    ra = df["RA"].to_numpy()
    dec = df["DEC"].to_numpy()
    z = df["Z"].to_numpy()
    lgm = df[WEIGHT_COL].to_numpy()

    nhat = radec_to_unitvec(ra, dec)

    # weights: w = 10**LGM (stellar mass proxy)
    w_mass = np.power(10.0, lgm)

    print("\n===== SDSS STRUCTURAL DIPOLE: MASK-AWARE NULL TEST (NORMALIZED) =====\n")
    print(f"Total galaxies used: N = {len(df)}")
    print(f"Redshift range in sample: z_min = {z.min():.3f}, z_max = {z.max():.3f}\n")

    # Global dipoles
    print("---- Global dipoles (full mask) ----")
    D_geo_vec, D_geo = normalized_dipole(nhat, np.ones(len(df)))
    geo_ra, geo_dec = unitvec_to_radec(D_geo_vec)
    print("Geometric (unweighted) dipole:")
    print(f"  |D_geo|     = {D_geo:.4f}")
    print(f"  (RA, DEC)   = ({geo_ra:.1f} deg, {geo_dec:.1f} deg)\n")

    D_obs_vec, D_obs = normalized_dipole(nhat, w_mass)
    obs_ra, obs_dec = unitvec_to_radec(D_obs_vec)
    angle = np.rad2deg(np.arccos(np.clip(
        np.dot(D_geo_vec, D_obs_vec) / ((np.linalg.norm(D_geo_vec) * np.linalg.norm(D_obs_vec)) + 1e-30),
        -1.0, 1.0
    )))
    print(f"Mass-weighted dipole (w = 10**{WEIGHT_COL.split('_')[0]}):")
    print(f"  |D_obs|     = {D_obs:.4f}")
    print(f"  (RA, DEC)   = ({obs_ra:.1f} deg, {obs_dec:.1f} deg)")
    print(f"  Angle(D_geo, D_obs) = {angle:.2f} deg\n")

    # Global shuffled null
    print(f"Building shuffled-weight null for GLOBAL sample (N_NULL = {N_NULL}) ...")
    null_global = shuffled_null_amps(nhat, w_mass, N_NULL, rng)
    mu_g, sig_g, z_g = zscore(D_obs, null_global)

    print("\n===== GLOBAL SHUFFLED NULL RESULTS (mask preserved) =====")
    print(f"  <|D|>_null   = {mu_g:.4f}")
    print(f"  sigma_null   = {sig_g:.4f}")
    print(f"  |D_obs|      = {D_obs:.4f}")
    print(f"  z-score      = {z_g:.2f} sigma\n")

    # Bins
    print("===== BINNED MASK-AWARE RESULTS =====")
    print(f"Redshift bins: {Z_EDGES}\n")

    bin_results = []
    null_bins = []
    for i in range(len(Z_EDGES) - 1):
        z0, z1 = Z_EDGES[i], Z_EDGES[i + 1]
        sel = (z >= z0) & (z < z1)
        nbin = int(np.sum(sel))
        print(f"Bin {i} [{z0:.3f}, {z1:.3f}): N = {nbin}")
        if nbin < 1000:
            print("  [WARN] Too few galaxies in this bin, skipping.\n")
            bin_results.append(None)
            null_bins.append(None)
            continue

        nh = nhat[sel]
        w = w_mass[sel]

        Dg_vec, Dg = normalized_dipole(nh, np.ones(nbin))
        Dg_ra, Dg_dec = unitvec_to_radec(Dg_vec)

        Do_vec, Do = normalized_dipole(nh, w)
        Do_ra, Do_dec = unitvec_to_radec(Do_vec)

        ang = np.rad2deg(np.arccos(np.clip(
            np.dot(Dg_vec, Do_vec) / ((np.linalg.norm(Dg_vec) * np.linalg.norm(Do_vec)) + 1e-30),
            -1.0, 1.0
        )))

        print(f"  Geometric dipole: |D_geo| = {Dg:.4f}, (RA,DEC) = ({Dg_ra:.1f}, {Dg_dec:.1f})")
        print(f"  Weighted dipole : |D_obs| = {Do:.4f}, (RA,DEC) = ({Do_ra:.1f}, {Do_dec:.1f})")
        print(f"  Angle(D_geo, D_obs) = {ang:.2f} deg")

        print(f"  Building shuffled-weight null for bin {i} (N_NULL = {N_NULL}) ...")
        null_i = shuffled_null_amps(nh, w, N_NULL, rng)
        mu_i, sig_i, z_i = zscore(Do, null_i)
        print(f"  <|D|>_null = {mu_i:.4f}, sigma_null = {sig_i:.4f}, |D_obs| = {Do:.4f}, z = {z_i:.2f} sigma\n")

        bin_results.append({
            "z0": z0, "z1": z1, "N": nbin,
            "D_geo_vec": Dg_vec, "D_geo": Dg, "D_geo_ra": Dg_ra, "D_geo_dec": Dg_dec,
            "D_obs_vec": Do_vec, "D_obs": Do, "D_obs_ra": Do_ra, "D_obs_dec": Do_dec,
            "angle_deg": ang,
            "null_mean": mu_i, "null_sigma": sig_i, "null_z": z_i
        })
        null_bins.append(null_i)

    # Save
    np.savez(
        OUT_NPZ,
        config=dict(
            catalog=CATALOG, z_min=Z_MIN, z_max=Z_MAX,
            z_edges=Z_EDGES.tolist(), N_NULL=N_NULL, seed=SEED,
            weight_col=WEIGHT_COL, weight_def="w=10**LGM"
        ),
        global_N=len(df),
        global_z_min=float(z.min()),
        global_z_max=float(z.max()),
        global_D_geo_vec=D_geo_vec,
        global_D_geo=D_geo,
        global_D_geo_ra=geo_ra,
        global_D_geo_dec=geo_dec,
        global_D_obs_vec=D_obs_vec,
        global_D_obs=D_obs,
        global_D_obs_ra=obs_ra,
        global_D_obs_dec=obs_dec,
        global_angle_deg=angle,
        null_global_amps=null_global,
        null_global_mean=mu_g,
        null_global_sigma=sig_g,
        null_global_z=z_g,
        bin_results=bin_results,
        null_bins=null_bins
    )

    print(f"Saved normalized mask-aware null results to: {OUT_NPZ}")
    print("Done.")

if __name__ == "__main__":
    main()
