#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd

# -----------------------------
# Geometry helpers
# -----------------------------
def rhat_from_radec(ra_deg, dec_deg):
    ra = np.deg2rad(ra_deg.astype(float))
    dec = np.deg2rad(dec_deg.astype(float))
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.vstack([x, y, z]).T

def radec_from_vec(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n <= 0:
        return np.nan, np.nan
    x, y, z = v / n
    ra = np.rad2deg(np.arctan2(y, x)) % 360.0
    dec = np.rad2deg(np.arcsin(np.clip(z, -1, 1)))
    return ra, dec

def angle_deg(v1, v2):
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 <= 0 or n2 <= 0:
        return np.nan
    c = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
    return float(np.rad2deg(np.arccos(c)))

# -----------------------------
# Weights
# -----------------------------
def compute_weights(df, mode, col_lgm):
    """
    col_lgm is treated as:
      - log10(M*) when mode in {mass, logmass, rankmass, clippedmass}
    """
    x = df[col_lgm].astype(float).to_numpy()

    if mode == "mass":
        # interpret x as log10(M*), return linear mass weights
        w = np.power(10.0, x)
    elif mode == "logmass":
        # use shifted log-mass as a positive weight
        x0 = np.nanmin(x)
        w = (x - x0) + 1e-6
    elif mode == "rankmass":
        # rank-based weights (robust to outliers)
        order = np.argsort(x)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(x) + 1, dtype=float)
        w = ranks / np.mean(ranks)
    elif mode == "clippedmass":
        # convert log10(M*) -> mass, clip extreme tails, normalize
        m = np.power(10.0, x)
        lo = np.percentile(m, 1.0)
        hi = np.percentile(m, 99.0)
        m = np.clip(m, lo, hi)
        w = m / np.mean(m)
    else:
        raise ValueError(f"Unknown weight-mode: {mode}")

    # final sanity
    w = np.asarray(w, dtype=float)
    bad = ~np.isfinite(w)
    if np.any(bad):
        w[bad] = 0.0
    w[w < 0] = 0.0
    return w

# -----------------------------
# Dipole estimator (normalized, bounded by 1)
# -----------------------------
def dipole_vector(ra, dec, weights=None):
    rhat = rhat_from_radec(ra, dec)
    if weights is None:
        w = np.ones(rhat.shape[0], dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
    sw = np.sum(w)
    if sw <= 0:
        v = np.zeros(3, dtype=float)
    else:
        v = np.sum(rhat * w[:, None], axis=0) / sw
    amp = float(np.linalg.norm(v))
    ra_v, dec_v = radec_from_vec(v)
    return v, amp, ra_v, dec_v

# -----------------------------
# IO + filtering
# -----------------------------
def require_cols(df, cols, context=""):
    missing = [c for c in cols if c and c not in df.columns]
    if missing:
        print("\n[ERROR] Missing required columns:", missing)
        print("[INFO] Available columns:", list(df.columns))
        if context:
            print("[INFO] Context:", context)
        raise SystemExit(2)

def parse_bins(zmin, zmax, z_bins_str):
    # user may pass: "0.02,0.05,0.08"
    if z_bins_str is None or str(z_bins_str).strip() == "":
        return np.array([zmin, zmax], dtype=float)
    parts = [p.strip() for p in str(z_bins_str).split(",") if p.strip() != ""]
    inner = np.array([float(p) for p in parts], dtype=float)
    edges = [float(zmin)]
    for v in inner:
        if v > zmin and v < zmax:
            edges.append(float(v))
    edges.append(float(zmax))
    edges = np.array(sorted(set(edges)), dtype=float)
    return edges

def load_and_filter(cfg, rng):
    df = pd.read_csv(cfg.input)
    print("Loading SDSS DR8 catalog from:\n ", cfg.input)
    print(f"Initial N (raw): {len(df)}")

    # reliability cut is optional
    col_rel = cfg.col_reliable
    if col_rel is not None:
        col_rel = str(col_rel)
        if col_rel.strip() == "":
            col_rel = None

    need = [cfg.col_ra, cfg.col_dec, cfg.col_z, cfg.col_lgm]
    if col_rel is not None:
        need.append(col_rel)

    require_cols(df, need, context="load_and_filter()")

    # drop NaNs
    df = df.dropna(subset=need)

    # reliability cut
    if col_rel is not None:
        df = df[df[col_rel].astype(int) == 1]
        print(f"After {col_rel} == 1 cut: N = {len(df)}")

    # z cut
    z = df[cfg.col_z].astype(float).to_numpy()
    m = (z >= cfg.zmin) & (z <= cfg.zmax)
    df = df.loc[m].copy()
    print(f"After {cfg.zmin:.3f} <= z <= {cfg.zmax:.3f} cut: N = {len(df)}")

    # optional shuffle of z assignments (robustness)
    if cfg.shuffle_z:
        zvals = df[cfg.col_z].astype(float).to_numpy()
        df[cfg.col_z] = rng.permutation(zvals)
        print("Applied --shuffle-z (permuted Z within filtered sample).")

    # optional downsample
    if cfg.n_max is not None and cfg.n_max > 0 and len(df) > cfg.n_max:
        idx = rng.choice(len(df), size=cfg.n_max, replace=False)
        df = df.iloc[idx].copy()
        print(f"Applied --n-max {cfg.n_max}: N = {len(df)}")

    return df

# -----------------------------
# Null test
# -----------------------------
def shuffled_null_amplitudes(ra, dec, weights, rng, n_null, progress_every=0):
    amps = np.empty(n_null, dtype=float)
    n = len(weights)
    for i in range(n_null):
        perm = rng.permutation(n)
        _, a, _, _ = dipole_vector(ra, dec, weights=weights[perm])
        amps[i] = a
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  Realisation {i+1}/{n_null}")
    return amps

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/sdss_dr8/sdss_dr8_analysis_base_v1.csv")
    ap.add_argument("--out", default="sdss_structural_dipole_mask_null_results_normed.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-null", type=int, default=1000)
    ap.add_argument("--zmin", type=float, default=0.020)
    ap.add_argument("--zmax", type=float, default=0.100)
    ap.add_argument("--z-bins", default="0.02,0.05,0.08")
    ap.add_argument("--n-max", type=int, default=None)

    ap.add_argument("--weight-mode", choices=["mass", "logmass", "rankmass", "clippedmass"], default="mass")
    ap.add_argument("--shuffle-z", action="store_true")
    ap.add_argument("--jackknife-ra", type=int, default=0)

    ap.add_argument("--col-ra", default="RA")
    ap.add_argument("--col-dec", default="DEC")
    ap.add_argument("--col-z", default="Z")
    ap.add_argument("--col-lgm", default="LGM_TOT_P50")
    ap.add_argument("--col-reliable", default="RELIABLE")

    cfg = ap.parse_args()
    rng = np.random.default_rng(cfg.seed)

    df = load_and_filter(cfg, rng)

    ra = df[cfg.col_ra].astype(float).to_numpy()
    dec = df[cfg.col_dec].astype(float).to_numpy()
    z = df[cfg.col_z].astype(float).to_numpy()

    print("\n===== SDSS STRUCTURAL DIPOLE: MASK-AWARE NULL TEST (NORMALIZED) =====\n")
    print(f"Total galaxies used: N = {len(df)}")
    print(f"Redshift range in sample: z_min = {z.min():.4f}, z_max = {z.max():.4f}")

    # geometric
    v_geo, a_geo, ra_geo, dec_geo = dipole_vector(ra, dec, weights=None)

    # weights
    w = compute_weights(df, cfg.weight_mode, cfg.col_lgm)
    v_obs, a_obs, ra_obs, dec_obs = dipole_vector(ra, dec, weights=w)

    print("\n---- Global dipoles (full mask) ----")
    print("Geometric (unweighted) dipole:")
    print(f"  |D_geo|     = {a_geo:.4f}")
    print(f"  (RA, DEC)   = ({ra_geo:.1f} deg, {dec_geo:.1f} deg)\n")

    print(f"{cfg.weight_mode}-weighted dipole:")
    print(f"  |D_obs|     = {a_obs:.4f}")
    print(f"  (RA, DEC)   = ({ra_obs:.1f} deg, {dec_obs:.1f} deg)")
    print(f"  Angle(D_geo, D_obs) = {angle_deg(v_geo, v_obs):.2f} deg\n")

    print(f"Args: seed={cfg.seed}, N_NULL={cfg.n_null}")
    print(f"Building shuffled-weight null for GLOBAL sample (N_NULL = {cfg.n_null}) .")

    prog = max(1, cfg.n_null // 5)
    amps_null = shuffled_null_amplitudes(ra, dec, w, rng, cfg.n_null, progress_every=prog)

    mu = float(np.mean(amps_null))
    sig = float(np.std(amps_null, ddof=1)) if cfg.n_null > 1 else np.nan
    zscore = (a_obs - mu) / sig if sig > 0 else np.nan

    print("\n===== GLOBAL SHUFFLED NULL RESULTS (mask preserved) =====")
    print(f"  <|D|>_null   = {mu:.4f}")
    print(f"  sigma_null   = {sig:.4f}")
    print(f"  |D_obs|      = {a_obs:.4f}")
    print(f"  z-score      = {zscore:.2f} sigma")

    # binned
    edges = parse_bins(cfg.zmin, cfg.zmax, cfg.z_bins)
    print("\n===== BINNED MASK-AWARE RESULTS =====")
    # print only inner edges like your previous output style
    inner = edges[1:-1]
    if len(inner) > 0:
        print("Redshift bins:", inner)
    else:
        print("Redshift bins: (none)")

    bin_results = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (z >= lo) & (z < hi) if i < len(edges) - 2 else (z >= lo) & (z <= hi)
        dfb = df.loc[m]
        if len(dfb) == 0:
            continue
        ra_b = dfb[cfg.col_ra].astype(float).to_numpy()
        dec_b = dfb[cfg.col_dec].astype(float).to_numpy()
        w_b = compute_weights(dfb, cfg.weight_mode, cfg.col_lgm)

        v_geo_b, a_geo_b, ra_geo_b, dec_geo_b = dipole_vector(ra_b, dec_b, None)
        v_obs_b, a_obs_b, ra_obs_b, dec_obs_b = dipole_vector(ra_b, dec_b, w_b)

        print(f"\nBin {i} [{lo:.3f}, {hi:.3f}): N = {len(dfb)}")
        print(f"  Geometric dipole: |D_geo| = {a_geo_b:.4f}, (RA,DEC) = ({ra_geo_b:.1f}, {dec_geo_b:.1f})")
        print(f"  Weighted dipole : |D_obs| = {a_obs_b:.4f}, (RA,DEC) = ({ra_obs_b:.1f}, {dec_obs_b:.1f})")
        print(f"  Angle(D_geo, D_obs) = {angle_deg(v_geo_b, v_obs_b):.2f} deg")
        print(f"  Building shuffled-weight null for bin {i} (N_NULL = {cfg.n_null}) .")

        rng_b = np.random.default_rng(cfg.seed + 1000 + i)
        prog_b = max(1, cfg.n_null // 5)
        amps_b = shuffled_null_amplitudes(ra_b, dec_b, w_b, rng_b, cfg.n_null, progress_every=prog_b)
        mu_b = float(np.mean(amps_b))
        sig_b = float(np.std(amps_b, ddof=1)) if cfg.n_null > 1 else np.nan
        z_b = (a_obs_b - mu_b) / sig_b if sig_b > 0 else np.nan
        print(f"  <|D|>_null = {mu_b:.4f}, sigma_null = {sig_b:.4f}, |D_obs| = {a_obs_b:.4f}, z = {z_b:.2f} sigma")

        bin_results.append((lo, hi, len(dfb), a_geo_b, a_obs_b, mu_b, sig_b, z_b, ra_obs_b, dec_obs_b))

    # jackknife in RA
    jk = []
    if cfg.jackknife_ra and cfg.jackknife_ra > 1:
        K = int(cfg.jackknife_ra)
        print(f"\n===== JACKKNIFE (leave-one-out RA sectors: K={K}) =====")
        ra_all = ra
        for k in range(K):
            lo = 360.0 * k / K
            hi = 360.0 * (k + 1) / K
            drop = (ra_all >= lo) & (ra_all < hi)
            keep = ~drop
            ra_k = ra[keep]
            dec_k = dec[keep]
            w_k = w[keep]

            v_geo_k, a_geo_k, _, _ = dipole_vector(ra_k, dec_k, None)
            v_obs_k, a_obs_k, _, _ = dipole_vector(ra_k, dec_k, w_k)
            ang = angle_deg(v_geo_k, v_obs_k)
            print(f"  drop sector {k} [{lo:.1f},{hi:.1f}): |D_geo|={a_geo_k:.4f}, |D_obs|={a_obs_k:.4f}, angle={ang:.2f} deg")
            jk.append((k, lo, hi, a_geo_k, a_obs_k, ang))

    # save
    np.savez(
        cfg.out,
        config=vars(cfg),
        N=len(df),
        global_geo_vector=v_geo,
        global_geo_amplitude=a_geo,
        global_geo_ra=ra_geo,
        global_geo_dec=dec_geo,
        global_obs_vector=v_obs,
        global_obs_amplitude=a_obs,
        global_obs_ra=ra_obs,
        global_obs_dec=dec_obs,
        global_angle_deg=angle_deg(v_geo, v_obs),
        null_amplitudes=amps_null,
        null_mu=mu,
        null_sigma=sig,
        null_zscore=zscore,
        bin_results=np.array(bin_results, dtype=float) if len(bin_results) else np.zeros((0, 11)),
        jackknife=np.array(jk, dtype=float) if len(jk) else np.zeros((0, 6)),
    )
    print(f"\nSaved normalized mask-aware null results to: {cfg.out}")
    print("Done.")

if __name__ == "__main__":
    main()
