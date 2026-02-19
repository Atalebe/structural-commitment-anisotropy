#!/usr/bin/env python3
"""
tng_structural_dipole_mask_null.py

Mask-aware dipole excess test on a "sky catalog" CSV (RA,DEC,Z,MSTAR).
Designed for TNG sky catalogs made by make_tng_sky_catalog.py, but will
work for any similar file.

Computes:
  - Geometric (unweighted) dipole: D_geo
  - Weighted dipole: D_obs (weights from MSTAR, or rankmass, or clippedmass)
  - Shuffled-weight null preserving sky geometry (RA/DEC positions fixed)

Optional:
  - Apply a crude SDSS-like footprint mask to the sky catalog (for apples-to-apples)
  - Redshift binning
  - Jackknife RA sectors

Notes:
  - Dipole vector uses unit vectors on the sphere:
      rhat_i = (cos dec cos ra, cos dec sin ra, sin dec)
    Then:
      D = sum(w_i rhat_i) / sum(w_i)
    For unweighted, w_i = 1.

  - With an incomplete footprint, |D_geo| can be large even for uniform weights.
    The physically relevant quantity is:
      Delta|D| = |D_obs| - <|D|>_null
    where null shuffles weights among galaxies, preserving the mask and positions.
"""

import argparse
import numpy as np
import pandas as pd


def rhat_from_radec(ra_deg, dec_deg):
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.vstack([x, y, z]).T


def vec_to_radec(v):
    v = np.asarray(v, dtype=float)
    v = v / np.linalg.norm(v)
    ra = np.degrees(np.arctan2(v[1], v[0])) % 360.0
    dec = np.degrees(np.arcsin(np.clip(v[2], -1.0, 1.0)))
    return ra, dec


def dipole_vector(rhat, w):
    w = np.asarray(w, dtype=float)
    W = np.sum(w)
    if not np.isfinite(W) or W <= 0:
        return np.array([np.nan, np.nan, np.nan])
    return np.sum(rhat * w[:, None], axis=0) / W


def dipole_amp(v):
    return float(np.linalg.norm(v))


def angle_deg(v1, v2):
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 <= 0 or n2 <= 0:
        return np.nan
    c = np.dot(v1, v2) / (n1 * n2)
    c = float(np.clip(c, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def build_weights(mstar, mode="mass", clip_hi=0.995):
    mstar = np.asarray(mstar, dtype=float)

    if mode == "mass":
        w = mstar.copy()
        w[~np.isfinite(w)] = np.nan
        w[w <= 0] = np.nan
        return w

    if mode == "rankmass":
        # ranks 1..N mapped to (0,1], robust to outliers
        good = np.isfinite(mstar)
        idx = np.where(good)[0]
        ranks = np.empty_like(mstar, dtype=float)
        ranks[:] = np.nan
        order = np.argsort(mstar[idx])
        r = np.empty_like(order, dtype=float)
        r[order] = np.linspace(1.0, float(len(idx)), len(idx))
        ranks[idx] = r / float(len(idx))
        return ranks

    if mode == "clippedmass":
        good = np.isfinite(mstar) & (mstar > 0)
        w = mstar.copy()
        w[~good] = np.nan
        if np.any(good):
            hi = np.quantile(w[good], clip_hi)
            w[good] = np.minimum(w[good], hi)
        return w

    raise ValueError(f"Unknown weight mode: {mode}")


def apply_sdss_like_mask(df, kind="none"):
    """
    Crude SDSS-like footprint masks (not the real angular mask).
    You can replace this later with an exact mangle polygon mask if you want.

    kind:
      - none: no mask
      - northcap: DEC > 0 (toy)
      - stripe: a rectangular RA/DEC window (toy)
    """
    if kind == "none":
        return df

    if kind == "northcap":
        return df[df["DEC"] > 0].copy()

    if kind == "stripe":
        # toy rectangle roughly near your SDSS dipole direction, tweak freely
        ra_min, ra_max = 110.0, 260.0
        dec_min, dec_max = -5.0, 70.0
        m = (df["RA"] >= ra_min) & (df["RA"] <= ra_max) & (df["DEC"] >= dec_min) & (df["DEC"] <= dec_max)
        return df[m].copy()

    raise ValueError(f"Unknown sdss-mask kind: {kind}")


def shuffled_null(rhat, w, n_null, seed, progress_every=200):
    rng = np.random.default_rng(seed)
    amps = np.empty(n_null, dtype=float)
    n = len(w)
    for i in range(n_null):
        perm = rng.permutation(n)
        v = dipole_vector(rhat, w[perm])
        amps[i] = dipole_amp(v)
        if progress_every and ((i + 1) % progress_every == 0):
            print(f"  Realisation {i+1}/{n_null}")
    mu = float(np.nanmean(amps))
    sig = float(np.nanstd(amps, ddof=1))
    return amps, mu, sig


def run_one(df, weight_mode, n_null, seed, z_bins=None, jackknife_ra=None, verbose=False):
    # geometry
    rhat = rhat_from_radec(df["RA"].to_numpy(), df["DEC"].to_numpy())

    # weights
    w = build_weights(df["MSTAR"].to_numpy(), mode=weight_mode)

    # drop NaNs in weights (keep geometry consistent)
    good = np.isfinite(w)
    df = df.iloc[np.where(good)[0]].copy()
    rhat = rhat[good]
    w = w[good]

    # global dipoles
    v_geo = dipole_vector(rhat, np.ones(len(w)))
    v_obs = dipole_vector(rhat, w)

    a_geo = dipole_amp(v_geo)
    a_obs = dipole_amp(v_obs)
    ra_geo, dec_geo = vec_to_radec(v_geo)
    ra_obs, dec_obs = vec_to_radec(v_obs)
    ang = angle_deg(v_geo, v_obs)

    print("\n---- Global dipoles (current mask) ----")
    print("Geometric (unweighted) dipole:")
    print(f"  |D_geo|     = {a_geo:.4f}")
    print(f"  (RA, DEC)   = ({ra_geo:.1f} deg, {dec_geo:.1f} deg)\n")

    print(f"{weight_mode}-weighted dipole:")
    print(f"  |D_obs|     = {a_obs:.4f}")
    print(f"  (RA, DEC)   = ({ra_obs:.1f} deg, {dec_obs:.1f} deg)")
    print(f"  Angle(D_geo, D_obs) = {ang:.2f} deg\n")

    print(f"Args: seed={seed}, N_NULL={n_null}")
    print(f"Building shuffled-weight null for GLOBAL sample (N_NULL = {n_null}) ...")
    amps, mu, sig = shuffled_null(rhat, w, n_null=n_null, seed=seed, progress_every=200)

    zscore = (a_obs - mu) / sig if sig > 0 else np.nan

    print("\n===== GLOBAL SHUFFLED NULL RESULTS (mask preserved) =====")
    print(f"  <|D|>_null   = {mu:.4f}")
    print(f"  sigma_null   = {sig:.4f}")
    print(f"  |D_obs|      = {a_obs:.4f}")
    print(f"  z-score      = {zscore:.2f} sigma\n")

    # LaTeX-ready snippet
    delta = a_obs - mu
    print("LaTeX-ready (global):")
    print(rf"\[|D_{{\rm geo}}| = {a_geo:.4f},\quad |D_{{\rm obs}}| = {a_obs:.4f},\quad \langle |D| \rangle_{{\rm null}} = {mu:.4f},\quad \sigma_{{\rm null}} = {sig:.4f}\]")
    print(rf"\[\Delta |D| \equiv |D_{{\rm obs}}|-\langle |D| \rangle_{{\rm null}} = {delta:.4f}\]")
    print(rf"\[\Delta |D|/\sigma_{{\rm null}} = {zscore:.2f}\]\n")

    # binned in z (optional)
    if z_bins is not None and len(z_bins) >= 2:
        print("===== BINNED MASK-AWARE RESULTS =====")
        print(f"Redshift bins: {np.array(z_bins)}\n")
        z = df["Z"].to_numpy()

        for bi in range(len(z_bins) - 1):
            z0, z1 = z_bins[bi], z_bins[bi + 1]
            m = (z >= z0) & (z < z1)
            nbin = int(np.sum(m))
            if nbin < 50:
                print(f"Bin {bi} [{z0:.3f},{z1:.3f}): N = {nbin} (skip)")
                continue

            rhat_b = rhat[m]
            w_b = w[m]

            v_geo_b = dipole_vector(rhat_b, np.ones(nbin))
            v_obs_b = dipole_vector(rhat_b, w_b)

            a_geo_b = dipole_amp(v_geo_b)
            a_obs_b = dipole_amp(v_obs_b)
            ra_geo_b, dec_geo_b = vec_to_radec(v_geo_b)
            ra_obs_b, dec_obs_b = vec_to_radec(v_obs_b)
            ang_b = angle_deg(v_geo_b, v_obs_b)

            print(f"Bin {bi} [{z0:.3f}, {z1:.3f}): N = {nbin}")
            print(f"  Geometric dipole: |D_geo| = {a_geo_b:.4f}, (RA,DEC) = ({ra_geo_b:.1f}, {dec_geo_b:.1f})")
            print(f"  Weighted dipole : |D_obs| = {a_obs_b:.4f}, (RA,DEC) = ({ra_obs_b:.1f}, {dec_obs_b:.1f})")
            print(f"  Angle(D_geo, D_obs) = {ang_b:.2f} deg")
            print(f"  Building shuffled-weight null for bin {bi} (N_NULL = {n_null}) ...")
            amps_b, mu_b, sig_b = shuffled_null(rhat_b, w_b, n_null=n_null, seed=seed + 1000 + bi, progress_every=0)
            z_b = (a_obs_b - mu_b) / sig_b if sig_b > 0 else np.nan
            print(f"  <|D|>_null = {mu_b:.4f}, sigma_null = {sig_b:.4f}, |D_obs| = {a_obs_b:.4f}, z = {z_b:.2f} sigma\n")

    # jackknife RA sectors (optional)
    if jackknife_ra is not None and jackknife_ra >= 2:
        K = int(jackknife_ra)
        print(f"===== JACKKNIFE (leave-one-out RA sectors: K={K}) =====")
        ra = df["RA"].to_numpy()
        edges = np.linspace(0.0, 360.0, K + 1)

        for k in range(K):
            lo, hi = edges[k], edges[k + 1]
            drop = (ra >= lo) & (ra < hi)
            keep = ~drop
            if np.sum(keep) < 50:
                print(f"  drop sector {k} [{lo:.1f},{hi:.1f}): too few kept, skip")
                continue

            rhat_k = rhat[keep]
            w_k = w[keep]

            v_geo_k = dipole_vector(rhat_k, np.ones(np.sum(keep)))
            v_obs_k = dipole_vector(rhat_k, w_k)
            a_geo_k = dipole_amp(v_geo_k)
            a_obs_k = dipole_amp(v_obs_k)
            ang_k = angle_deg(v_geo_k, v_obs_k)

            print(f"  drop sector {k} [{lo:.1f},{hi:.1f}): |D_geo|={a_geo_k:.4f}, |D_obs|={a_obs_k:.4f}, angle={ang_k:.2f} deg")

    return {
        "N": len(df),
        "D_geo_amp": a_geo,
        "D_obs_amp": a_obs,
        "D_null_mu": mu,
        "D_null_sigma": sig,
        "zscore": zscore,
        "ra_geo": ra_geo,
        "dec_geo": dec_geo,
        "ra_obs": ra_obs,
        "dec_obs": dec_obs,
        "angle_deg": ang,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True, help="CSV with RA,DEC,Z,MSTAR")
    ap.add_argument("--weight-mode", default="mass", choices=["mass", "rankmass", "clippedmass"])
    ap.add_argument("--n-null", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--zmin", type=float, default=None)
    ap.add_argument("--zmax", type=float, default=None)
    ap.add_argument("--z-bins", type=str, default=None, help="comma list, e.g. 0.0,0.003,0.006,0.01")

    ap.add_argument("--sdss-mask", default="none", choices=["none", "northcap", "stripe"])
    ap.add_argument("--jackknife-ra", type=int, default=None)

    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.catalog)
    need = {"RA", "DEC", "Z", "MSTAR"}
    if not need.issubset(set(df.columns)):
        raise SystemExit(f"Catalog must have columns {sorted(need)}, got: {list(df.columns)}")

    # optional z cut
    if args.zmin is not None:
        df = df[df["Z"] >= args.zmin].copy()
    if args.zmax is not None:
        df = df[df["Z"] <= args.zmax].copy()

    # optional sdss-like mask
    df = apply_sdss_like_mask(df, kind=args.sdss_mask)

    print("===== TNG SKY-CATALOG DIPOLE: MASK-AWARE NULL TEST (NORMALIZED) =====\n")
    print(f"Catalog: {args.catalog}")
    print(f"Total galaxies used: N = {len(df)}")
    if len(df) == 0:
        raise SystemExit("No rows left after cuts/mask.")
    print(f"Z range in sample: z_min = {df['Z'].min():.4g}, z_max = {df['Z'].max():.4g}")
    print(f"Mask mode: {args.sdss_mask}")
    print(f"Weight mode: {args.weight_mode}\n")

    z_bins = None
    if args.z_bins is not None:
        z_bins = [float(x) for x in args.z_bins.split(",") if x.strip() != ""]
        if len(z_bins) < 2:
            z_bins = None

    run_one(
        df=df,
        weight_mode=args.weight_mode,
        n_null=args.n_null,
        seed=args.seed,
        z_bins=z_bins,
        jackknife_ra=args.jackknife_ra,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
