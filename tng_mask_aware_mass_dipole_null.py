#!/usr/bin/env python3
"""
Mask-aware dipole excess test on simulations (TNG300 / TNG50).

Goal: reproduce the SDSS DR8 mask-preserving shuffled-weight null logic:
- Compute geometric (unweighted) dipole inside a mask
- Compute mass-weighted dipole inside the same mask
- Shuffle weights to get <|D|>_null and sigma_null (mask preserved)
- Report excess Delta|D| and Z-score
- Optional: binned by redshift, and RA-jackknife sectors

Input expectations (CSV):
Must contain at least: RA [deg], DEC [deg], Z, and stellar mass proxy.
Column names are configurable via CLI.

This script is deliberately survey-agnostic: it works for full-sky and artificial masks.
"""

import argparse
import math
import numpy as np
import pandas as pd


def radec_to_unit(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.vstack([x, y, z]).T


def unit_to_radec(v: np.ndarray) -> tuple[float, float]:
    x, y, z = v
    ra = math.degrees(math.atan2(y, x)) % 360.0
    hyp = math.hypot(x, y)
    dec = math.degrees(math.atan2(z, hyp))
    return ra, dec


def dipole_vector(nhat: np.ndarray, weights: np.ndarray) -> np.ndarray:
    wsum = np.sum(weights)
    if not np.isfinite(wsum) or wsum <= 0:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    v = (nhat * weights[:, None]).sum(axis=0) / wsum
    return v


def angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    uu = np.linalg.norm(u)
    vv = np.linalg.norm(v)
    if not np.isfinite(uu) or not np.isfinite(vv) or uu == 0 or vv == 0:
        return np.nan
    c = np.clip(np.dot(u, v) / (uu * vv), -1.0, 1.0)
    return math.degrees(math.acos(c))


def make_weights(mass: np.ndarray, mode: str, clip_q: float = 0.995) -> np.ndarray:
    mass = np.asarray(mass, dtype=float)
    mass = np.where(np.isfinite(mass), mass, np.nan)
    if mode == "mass":
        w = mass.copy()
    elif mode == "rankmass":
        # monotone mapping: ranks in [0,1], then shift to positive weights
        # preserves ordering but compresses dynamic range
        valid = np.isfinite(mass)
        w = np.full_like(mass, np.nan, dtype=float)
        ranks = pd.Series(mass[valid]).rank(method="average").to_numpy()
        ranks = (ranks - 1.0) / max(len(ranks) - 1.0, 1.0)  # [0,1]
        w[valid] = 0.5 + ranks  # strictly positive
    elif mode == "clippedmass":
        valid = np.isfinite(mass)
        w = mass.copy()
        if valid.sum() > 10:
            hi = np.nanquantile(w[valid], clip_q)
            w = np.minimum(w, hi)
    else:
        raise ValueError(f"Unknown weight mode: {mode}")
    # enforce positive weights
    w = np.where(np.isfinite(w) & (w > 0), w, np.nan)
    return w


def apply_mask(df: pd.DataFrame, mask_mode: str, ra_col: str, dec_col: str,
               ra_min: float, ra_max: float, dec_min: float, dec_max: float) -> pd.DataFrame:
    if mask_mode == "none":
        return df
    if mask_mode == "box":
        ra = df[ra_col].to_numpy()
        dec = df[dec_col].to_numpy()
        # RA wrap handling: allow ra_min > ra_max to indicate wrap-around
        if ra_min <= ra_max:
            m_ra = (ra >= ra_min) & (ra <= ra_max)
        else:
            m_ra = (ra >= ra_min) | (ra <= ra_max)
        m_dec = (dec >= dec_min) & (dec <= dec_max)
        return df[m_ra & m_dec]
    raise ValueError(f"Unknown mask mode: {mask_mode}")


def shuffled_null(nhat: np.ndarray, weights: np.ndarray, n_null: int, seed: int, progress: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    amps = np.empty(n_null, dtype=float)
    idx = np.arange(len(weights))
    for r in range(n_null):
        perm = rng.permutation(idx)
        v = dipole_vector(nhat, weights[perm])
        amps[r] = np.linalg.norm(v)
        if progress and (r + 1) % progress == 0:
            print(f"  Realisation {r+1}/{n_null}")
    mu = float(np.nanmean(amps))
    sig = float(np.nanstd(amps, ddof=1))
    return mu, sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV catalog with RA/DEC/Z/mass.")
    ap.add_argument("--ra-col", default="RA", help="RA column name (deg).")
    ap.add_argument("--dec-col", default="DEC", help="DEC column name (deg).")
    ap.add_argument("--z-col", default="Z", help="Redshift column name.")
    ap.add_argument("--mass-col", default="MSTAR", help="Stellar mass column name (linear).")

    ap.add_argument("--z-min", type=float, default=None)
    ap.add_argument("--z-max", type=float, default=None)

    ap.add_argument("--mask-mode", choices=["none", "box"], default="none")
    ap.add_argument("--ra-min", type=float, default=110.0)
    ap.add_argument("--ra-max", type=float, default=270.0)
    ap.add_argument("--dec-min", type=float, default=-10.0)
    ap.add_argument("--dec-max", type=float, default=70.0)

    ap.add_argument("--weight-mode", choices=["mass", "rankmass", "clippedmass"], default="mass")
    ap.add_argument("--clip-q", type=float, default=0.995)

    ap.add_argument("--n-null", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--progress", type=int, default=200)

    ap.add_argument("--bins", type=str, default="0.02,0.05,0.08,0.10",
                    help="Comma-separated z bin edges (optional).")
    ap.add_argument("--no-bins", action="store_true", help="Disable binned analysis.")
    ap.add_argument("--jackknife-ra", type=int, default=0, help="If >0, do RA sector jackknife with K sectors.")

    args = ap.parse_args()

    print("Loading catalog:", args.input)
    df = pd.read_csv(args.input)

    # Basic cleaning
    for c in [args.ra_col, args.dec_col, args.z_col, args.mass_col]:
        if c not in df.columns:
            raise KeyError(f"Missing column '{c}' in {args.input}")

    df = df.dropna(subset=[args.ra_col, args.dec_col, args.z_col, args.mass_col]).copy()

    if args.z_min is not None:
        df = df[df[args.z_col] >= args.z_min]
    if args.z_max is not None:
        df = df[df[args.z_col] <= args.z_max]

    df = apply_mask(df, args.mask_mode, args.ra_col, args.dec_col,
                    args.ra_min, args.ra_max, args.dec_min, args.dec_max)

    N = len(df)
    if N < 1000:
        raise RuntimeError(f"Too few objects after cuts: N={N}")

    z_min = float(df[args.z_col].min())
    z_max = float(df[args.z_col].max())
    print("\n===== MASK-AWARE DIPLOLE NULL TEST (SIMULATION) =====")
    print(f"Total objects used: N = {N}")
    print(f"Redshift range in sample: z_min = {z_min:.3f}, z_max = {z_max:.3f}")
    print(f"Mask mode: {args.mask_mode}")
    if args.mask_mode == "box":
        print(f"  RA in [{args.ra_min:.1f}, {args.ra_max:.1f}] deg (wrap allowed)")
        print(f"  DEC in [{args.dec_min:.1f}, {args.dec_max:.1f}] deg")
    print(f"Weight mode: {args.weight_mode}")

    ra = df[args.ra_col].to_numpy(dtype=float)
    dec = df[args.dec_col].to_numpy(dtype=float)
    z = df[args.z_col].to_numpy(dtype=float)
    mass = df[args.mass_col].to_numpy(dtype=float)

    nhat = radec_to_unit(ra, dec)

    w_obs = make_weights(mass, args.weight_mode, clip_q=args.clip_q)
    good = np.isfinite(w_obs)
    nhat = nhat[good]
    ra = ra[good]
    dec = dec[good]
    z = z[good]
    w_obs = w_obs[good]
    N = len(w_obs)

    # Global dipoles
    D_geo = dipole_vector(nhat, np.ones(N, dtype=float))
    D_obs = dipole_vector(nhat, w_obs)

    amp_geo = float(np.linalg.norm(D_geo))
    amp_obs = float(np.linalg.norm(D_obs))
    ra_geo, dec_geo = unit_to_radec(D_geo)
    ra_obs, dec_obs = unit_to_radec(D_obs)

    print("\n---- Global dipoles (mask applied) ----")
    print(f"Geometric dipole:")
    print(f"  |D_geo|     = {amp_geo:.4f}")
    print(f"  (RA, DEC)   = ({ra_geo:.1f} deg, {dec_geo:.1f} deg)")
    print(f"{args.weight_mode}-weighted dipole:")
    print(f"  |D_obs|     = {amp_obs:.4f}")
    print(f"  (RA, DEC)   = ({ra_obs:.1f} deg, {dec_obs:.1f} deg)")
    print(f"  Angle(D_geo, D_obs) = {angle_deg(D_geo, D_obs):.2f} deg")

    print(f"\nArgs: seed={args.seed}, N_NULL={args.n_null}")
    print(f"Building shuffled-weight null for GLOBAL sample (N_NULL = {args.n_null}) ...")
    mu, sig = shuffled_null(nhat, w_obs, args.n_null, args.seed, args.progress)

    delta = amp_obs - mu
    zsig = delta / sig if sig > 0 else np.nan

    print("\n===== GLOBAL SHUFFLED NULL RESULTS (mask preserved) =====")
    print(f"  <|D|>_null   = {mu:.4f}")
    print(f"  sigma_null   = {sig:.4f}")
    print(f"  |D_obs|      = {amp_obs:.4f}")
    print(f"  Delta|D|     = {delta:.4f}")
    print(f"  z-score      = {zsig:.2f} sigma")

    # Bins
    if not args.no_bins and args.bins.strip():
        edges = np.array([float(x) for x in args.bins.split(",")], dtype=float)
        if len(edges) >= 3:
            print("\n===== BINNED MASK-AWARE RESULTS =====")
            print("Redshift bins:", edges)
            for bi in range(len(edges) - 1):
                lo, hi = edges[bi], edges[bi + 1]
                m = (z >= lo) & (z < hi)
                Nb = int(m.sum())
                if Nb < 2000:
                    print(f"\nBin {bi} [{lo:.3f}, {hi:.3f}): N = {Nb} (skipping, too small)")
                    continue
                nh = nhat[m]
                wb = w_obs[m]
                Dg = dipole_vector(nh, np.ones(Nb, dtype=float))
                Do = dipole_vector(nh, wb)
                ag = float(np.linalg.norm(Dg))
                ao = float(np.linalg.norm(Do))
                ang = angle_deg(Dg, Do)
                print(f"\nBin {bi} [{lo:.3f}, {hi:.3f}): N = {Nb}")
                print(f"  Geometric dipole: |D_geo| = {ag:.4f}")
                print(f"  Weighted dipole : |D_obs| = {ao:.4f}")
                print(f"  Angle(D_geo, D_obs) = {ang:.2f} deg")
                print(f"  Building shuffled-weight null for bin {bi} (N_NULL = {args.n_null}) ...")
                mu_b, sig_b = shuffled_null(nh, wb, args.n_null, args.seed, max(args.progress, 0))
                delta_b = ao - mu_b
                z_b = delta_b / sig_b if sig_b > 0 else np.nan
                print(f"  <|D|>_null = {mu_b:.4f}, sigma_null = {sig_b:.4f}, |D_obs| = {ao:.4f}, Delta|D| = {delta_b:.4f}, z = {z_b:.2f} sigma")

    # RA jackknife
    if args.jackknife_ra and args.jackknife_ra > 1:
        K = int(args.jackknife_ra)
        print(f"\n===== JACKKNIFE (leave-one-out RA sectors: K={K}) =====")
        # Define sectors in RA
        bounds = np.linspace(0.0, 360.0, K + 1)
        for k in range(K):
            lo, hi = bounds[k], bounds[k + 1]
            # drop sector k
            if lo <= hi:
                drop = (ra >= lo) & (ra < hi)
            else:
                drop = (ra >= lo) | (ra < hi)
            keep = ~drop
            Nk = int(keep.sum())
            if Nk < 2000:
                print(f"  drop sector {k} [{lo:.1f},{hi:.1f}): N={Nk} (too small)")
                continue
            nh = nhat[keep]
            wb = w_obs[keep]
            Dg = dipole_vector(nh, np.ones(Nk, dtype=float))
            Do = dipole_vector(nh, wb)
            ag = float(np.linalg.norm(Dg))
            ao = float(np.linalg.norm(Do))
            ang = angle_deg(Dg, Do)
            print(f"  drop sector {k} [{lo:.1f},{hi:.1f}): |D_geo|={ag:.4f}, |D_obs|={ao:.4f}, angle={ang:.2f} deg")

    print("\nDone.")


if __name__ == "__main__":
    main()
