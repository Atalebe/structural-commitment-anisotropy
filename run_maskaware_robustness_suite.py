#!/usr/bin/env python3
"""
run_maskaware_robustness_suite.py

Mask-aware dipole robustness suite for a single catalog:
- main shuffled-weight null
- vector-level excess |D_phys|
- hemisphere split about observed axis
- jackknife RA sectors with full null per jackknife sample
- alternative nulls (shuffle RA, shuffle DEC, shuffle mass within z-bins)
- optional trimming of the highest-mass objects

Outputs:
  results/maskaware_robustness/<label>/
    summary_global.csv
    summary_global_table.tex
    hemisphere_split.csv
    hemisphere_split_table.tex
    jackknife.csv
    jackknife_table.tex
    alt_nulls.csv
    alt_nulls_table.tex
    trimmed.csv
    trimmed_table.tex
"""

import argparse
import os
import math
import numpy as np
import pandas as pd


# -------------------------
# Geometry helpers
# -------------------------
def radec_to_unitvec(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(ra_deg.astype(float))
    dec = np.deg2rad(dec_deg.astype(float))
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.vstack([x, y, z]).T  # (N,3)


def safe_unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n <= 0:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    return v / n


def dipole_vector(unit_vecs: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    """
    Normalized "mean direction" vector.
    If weights is None: mean(unit_vecs).
    If weights provided: mean weighted direction sum / sum(weights).
    """
    if unit_vecs.size == 0:
        return np.array([np.nan, np.nan, np.nan], dtype=float)

    if weights is None:
        v = unit_vecs.mean(axis=0)
    else:
        w = np.asarray(weights, dtype=float)
        wsum = float(np.sum(w))
        if not np.isfinite(wsum) or wsum <= 0:
            return np.array([np.nan, np.nan, np.nan], dtype=float)
        v = (unit_vecs * w[:, None]).sum(axis=0) / wsum

    return v


def amp_and_dir(unit_vecs: np.ndarray, weights: np.ndarray | None):
    v = dipole_vector(unit_vecs, weights)
    amp = float(np.linalg.norm(v)) if np.all(np.isfinite(v)) else float("nan")
    vhat = safe_unit(v) if np.isfinite(amp) else np.array([np.nan, np.nan, np.nan], dtype=float)
    return v, vhat, amp


def angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v))):
        return float("nan")
    uu = safe_unit(u)
    vv = safe_unit(v)
    c = float(np.clip(np.dot(uu, vv), -1.0, 1.0))
    return float(np.rad2deg(np.arccos(c)))


def vec_to_radec(vhat: np.ndarray):
    if not np.all(np.isfinite(vhat)):
        return float("nan"), float("nan")
    x, y, z = vhat
    ra = math.degrees(math.atan2(y, x)) % 360.0
    dec = math.degrees(math.asin(np.clip(z, -1.0, 1.0)))
    return float(ra), float(dec)


# -------------------------
# Weights
# -------------------------
def build_weights(df: pd.DataFrame, col_lgm: str, mode: str, rng: np.random.Generator,
                  clip_lo: float = 1.0, clip_hi: float = 99.0) -> np.ndarray:
    """
    mode in {"mass","logmass","rankmass","clippedmass"}.
    col_lgm is expected to be log10(M) in solar masses (like SDSS LGM_TOT_P50 or TNG log10(MSTAR)).
    """
    lgm = df[col_lgm].astype(float).to_numpy()

    if mode == "logmass":
        w = lgm.copy()

    elif mode == "mass":
        w = 10.0 ** lgm

    elif mode == "clippedmass":
        lo = np.nanpercentile(lgm, clip_lo)
        hi = np.nanpercentile(lgm, clip_hi)
        lgm2 = np.clip(lgm, lo, hi)
        w = 10.0 ** lgm2

    elif mode == "rankmass":
        # deterministic rank mapping to (0,1]
        order = np.argsort(lgm, kind="mergesort")
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(lgm) + 1, dtype=float)
        w = ranks / float(len(lgm))

    else:
        raise ValueError(f"Unknown weight mode: {mode}")

    # Ensure strictly positive weights for stability
    w = np.asarray(w, dtype=float)
    w = np.clip(w, 1e-300, np.inf)
    return w


# -------------------------
# Null models
# -------------------------
def null_shuffle_weights(unit_vecs: np.ndarray, weights: np.ndarray, n_null: int,
                         rng: np.random.Generator, progress_every: int = 0) -> np.ndarray:
    amps = np.empty(n_null, dtype=float)
    n = len(weights)
    for i in range(n_null):
        perm = rng.permutation(n)
        w_shuf = weights[perm]
        _, _, a = amp_and_dir(unit_vecs, w_shuf)
        amps[i] = a
        if progress_every and ((i + 1) % progress_every == 0):
            print(f"  Realisation {i+1}/{n_null}")
    return amps


def null_shuffle_ra(df: pd.DataFrame, unit_vecs: np.ndarray, weights: np.ndarray, col_ra: str,
                    n_null: int, rng: np.random.Generator) -> np.ndarray:
    """
    Alternative null: permute RA values among galaxies, leaving DEC and weights fixed.
    This preserves the 1D RA distribution but destroys coherent RA structure.
    """
    amps = np.empty(n_null, dtype=float)
    ra = df[col_ra].astype(float).to_numpy()
    dec = df["_DEC_TMP_"].astype(float).to_numpy()
    for i in range(n_null):
        ra2 = ra[rng.permutation(len(ra))]
        u2 = radec_to_unitvec(ra2, dec)
        _, _, a = amp_and_dir(u2, weights)
        amps[i] = a
    return amps


def null_shuffle_dec(df: pd.DataFrame, unit_vecs: np.ndarray, weights: np.ndarray, col_dec: str,
                     n_null: int, rng: np.random.Generator) -> np.ndarray:
    """
    Alternative null: permute DEC values among galaxies, leaving RA and weights fixed.
    """
    amps = np.empty(n_null, dtype=float)
    ra = df["_RA_TMP_"].astype(float).to_numpy()
    dec = df[col_dec].astype(float).to_numpy()
    for i in range(n_null):
        dec2 = dec[rng.permutation(len(dec))]
        u2 = radec_to_unitvec(ra, dec2)
        _, _, a = amp_and_dir(u2, weights)
        amps[i] = a
    return amps


def null_shuffle_mass_within_zbins(df: pd.DataFrame, unit_vecs: np.ndarray, weights: np.ndarray,
                                  col_z: str, z_edges: np.ndarray,
                                  n_null: int, rng: np.random.Generator) -> np.ndarray:
    """
    Alternative null: scramble weights *within* z-bins.
    Controls mass--z coupling while testing directional correlation.
    """
    amps = np.empty(n_null, dtype=float)
    z = df[col_z].astype(float).to_numpy()
    n = len(z)
    # Assign bin indices
    b = np.digitize(z, z_edges) - 1
    bins = {}
    for i in range(n):
        bi = int(b[i])
        if bi < 0 or bi >= (len(z_edges) - 1):
            continue
        bins.setdefault(bi, []).append(i)
    bins = {k: np.array(v, dtype=int) for k, v in bins.items()}

    for t in range(n_null):
        w2 = weights.copy()
        for bi, idx in bins.items():
            if len(idx) < 2:
                continue
            perm = rng.permutation(len(idx))
            w2[idx] = weights[idx[perm]]
        _, _, a = amp_and_dir(unit_vecs, w2)
        amps[t] = a

    return amps


def summarize_null(obs_amp: float, null_amps: np.ndarray):
    mu = float(np.nanmean(null_amps))
    sig = float(np.nanstd(null_amps, ddof=1))
    z = float((obs_amp - mu) / sig) if (np.isfinite(sig) and sig > 0) else float("nan")
    return mu, sig, z


# -------------------------
# Load & filter
# -------------------------
def load_and_filter(path: str,
                    col_ra: str, col_dec: str, col_z: str,
                    col_lgm: str,
                    col_reliable: str | None,
                    reliable_value: int,
                    zmin: float, zmax: float,
                    n_max: int | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Trim columns of whitespace
    df.columns = [c.strip() for c in df.columns]

    needed = [col_ra, col_dec, col_z, col_lgm]
    for c in needed:
        if c not in df.columns:
            raise KeyError(f"Missing required column '{c}' in {path}. Found: {list(df.columns)[:20]} ...")

    if col_reliable and col_reliable.strip():
        if col_reliable not in df.columns:
            raise KeyError(f"Missing reliable column '{col_reliable}' in {path}.")
        df = df[df[col_reliable].astype(int) == int(reliable_value)].copy()

    # Drop NaNs in required fields
    df = df.dropna(subset=needed).copy()

    # z cut
    z = df[col_z].astype(float)
    df = df[(z >= float(zmin)) & (z <= float(zmax))].copy()

    # Optional max N for quick tests
    if n_max is not None and len(df) > int(n_max):
        df = df.sample(n=int(n_max), random_state=0).copy()

    # Cache RA/DEC arrays for alt nulls (avoid repeated column lookups)
    df["_RA_TMP_"] = df[col_ra].astype(float)
    df["_DEC_TMP_"] = df[col_dec].astype(float)

    return df


# -------------------------
# Robustness pieces
# -------------------------
def compute_core(df: pd.DataFrame, col_ra: str, col_dec: str, col_z: str, col_lgm: str,
                 weight_mode: str, n_null: int, rng: np.random.Generator,
                 progress: bool) -> dict:
    ra = df[col_ra].astype(float).to_numpy()
    dec = df[col_dec].astype(float).to_numpy()
    unit_vecs = radec_to_unitvec(ra, dec)

    # Geo
    Dgeo_vec, Dgeo_hat, Dgeo_amp = amp_and_dir(unit_vecs, None)

    # Weights + Obs
    weights = build_weights(df, col_lgm, weight_mode, rng)
    Dobs_vec, Dobs_hat, Dobs_amp = amp_and_dir(unit_vecs, weights)

    # Null shuffle weights
    prog = 200 if progress else 0
    null_amps = null_shuffle_weights(unit_vecs, weights, n_null=n_null, rng=rng, progress_every=prog)
    mu, sig, z = summarize_null(Dobs_amp, null_amps)

    # Vector excess
    Dphys_vec = Dobs_vec - Dgeo_vec
    Dphys_amp = float(np.linalg.norm(Dphys_vec)) if np.all(np.isfinite(Dphys_vec)) else float("nan")

    # Angles + directions
    ang_geo_obs = angle_deg(Dgeo_hat, Dobs_hat)
    ra_obs, dec_obs = vec_to_radec(Dobs_hat)

    out = dict(
        N=int(len(df)),
        zmin=float(df[col_z].min()),
        zmax=float(df[col_z].max()),
        weight_mode=str(weight_mode),

        D_geo=float(Dgeo_amp),
        D_obs=float(Dobs_amp),
        mu_null=float(mu),
        sigma_null=float(sig),
        z_score=float(z),

        D_phys=float(Dphys_amp),
        angle_geo_obs_deg=float(ang_geo_obs),
        ra_obs_deg=float(ra_obs),
        dec_obs_deg=float(dec_obs),
    )
    return out, unit_vecs, weights, Dgeo_vec, Dobs_hat


def hemisphere_split(df: pd.DataFrame, unit_vecs: np.ndarray, weights: np.ndarray,
                     axis_hat: np.ndarray,
                     n_null: int, rng: np.random.Generator,
                     progress: bool) -> pd.DataFrame:
    """
    Split sample into hemispheres by sign of dot(u, axis_hat).
    For each hemisphere: compute D_geo, D_obs, and shuffled-weight null z.
    """
    dots = unit_vecs @ axis_hat
    idx_plus = np.where(dots >= 0)[0]
    idx_minus = np.where(dots < 0)[0]

    rows = []
    for name, idx in [("plus", idx_plus), ("minus", idx_minus)]:
        u = unit_vecs[idx]
        w = weights[idx]

        Dgeo_vec, Dgeo_hat, Dgeo_amp = amp_and_dir(u, None)
        Dobs_vec, Dobs_hat, Dobs_amp = amp_and_dir(u, w)

        prog = 200 if progress else 0
        null_amps = null_shuffle_weights(u, w, n_null=n_null, rng=rng, progress_every=prog)
        mu, sig, z = summarize_null(Dobs_amp, null_amps)

        Dphys_vec = Dobs_vec - Dgeo_vec
        Dphys_amp = float(np.linalg.norm(Dphys_vec)) if np.all(np.isfinite(Dphys_vec)) else float("nan")
        ang = angle_deg(Dgeo_hat, Dobs_hat)

        rows.append(dict(
            hemi=name,
            N=int(len(idx)),
            D_geo=float(Dgeo_amp),
            D_obs=float(Dobs_amp),
            mu_null=float(mu),
            sigma_null=float(sig),
            z_score=float(z),
            D_phys=float(Dphys_amp),
            angle_geo_obs_deg=float(ang),
        ))

    return pd.DataFrame(rows)


def jackknife_ra(df: pd.DataFrame, unit_vecs: np.ndarray, weights: np.ndarray,
                 col_ra: str, K: int,
                 n_null: int, rng: np.random.Generator,
                 progress: bool) -> pd.DataFrame:
    """
    Leave-one-out RA sectors. For each drop-sector:
      compute D_geo, D_obs, and shuffled-weight null z on the remaining sample.
    """
    ra = df[col_ra].astype(float).to_numpy()
    edges = np.linspace(0.0, 360.0, K + 1)
    sector = np.digitize(ra, edges) - 1
    sector = np.clip(sector, 0, K - 1)

    rows = []
    for k in range(K):
        keep = np.where(sector != k)[0]
        u = unit_vecs[keep]
        w = weights[keep]

        Dgeo_vec, Dgeo_hat, Dgeo_amp = amp_and_dir(u, None)
        Dobs_vec, Dobs_hat, Dobs_amp = amp_and_dir(u, w)

        prog = 0
        if progress:
            # jackknife can be slow; print only per-sector summary
            prog = 0

        null_amps = null_shuffle_weights(u, w, n_null=n_null, rng=rng, progress_every=prog)
        mu, sig, z = summarize_null(Dobs_amp, null_amps)

        Dphys_vec = Dobs_vec - Dgeo_vec
        Dphys_amp = float(np.linalg.norm(Dphys_vec)) if np.all(np.isfinite(Dphys_vec)) else float("nan")
        ang = angle_deg(Dgeo_hat, Dobs_hat)

        rows.append(dict(
            drop_sector=int(k),
            ra_lo=float(edges[k]),
            ra_hi=float(edges[k + 1]),
            N=int(len(keep)),
            D_geo=float(Dgeo_amp),
            D_obs=float(Dobs_amp),
            mu_null=float(mu),
            sigma_null=float(sig),
            z_score=float(z),
            D_phys=float(Dphys_amp),
            angle_geo_obs_deg=float(ang),
        ))
        if progress:
            print(f"[JK] drop {k:02d} [{edges[k]:.1f},{edges[k+1]:.1f}): z={z:.2f}, N={len(keep)}")

    return pd.DataFrame(rows)


def alt_nulls(df: pd.DataFrame, unit_vecs: np.ndarray, weights: np.ndarray,
              col_ra: str, col_dec: str, col_z: str,
              Dobs_amp: float,
              n_null: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Run alternative null distributions and return their mu, sigma, and z.
    """
    rows = []

    a_ra = null_shuffle_ra(df, unit_vecs, weights, col_ra=col_ra, n_null=n_null, rng=rng)
    mu, sig, z = summarize_null(Dobs_amp, a_ra)
    rows.append(dict(null_model="shuffle_ra", mu_null=mu, sigma_null=sig, z_score=z))

    a_dec = null_shuffle_dec(df, unit_vecs, weights, col_dec=col_dec, n_null=n_null, rng=rng)
    mu, sig, z = summarize_null(Dobs_amp, a_dec)
    rows.append(dict(null_model="shuffle_dec", mu_null=mu, sigma_null=sig, z_score=z))

    # z-binned weight shuffle
    zmin = float(df[col_z].min())
    zmax = float(df[col_z].max())
    # default edges: 6 bins, inclusive-ish
    edges = np.linspace(zmin, zmax, 7)
    a_zbin = null_shuffle_mass_within_zbins(df, unit_vecs, weights, col_z=col_z, z_edges=edges, n_null=n_null, rng=rng)
    mu, sig, z = summarize_null(Dobs_amp, a_zbin)
    rows.append(dict(null_model="shuffle_mass_within_zbins", mu_null=mu, sigma_null=sig, z_score=z))

    return pd.DataFrame(rows)


def trimmed_tests(df: pd.DataFrame, col_lgm: str,
                  col_ra: str, col_dec: str, col_z: str,
                  weight_mode: str, trims: list[float],
                  n_null: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Drop top X fraction by log-mass and recompute z-score.
    trims: list of fractions (e.g. 0.01 means drop top 1%).
    """
    rows = []
    lgm = df[col_lgm].astype(float).to_numpy()
    n = len(lgm)
    order = np.argsort(lgm)  # ascending
    for frac in trims:
        frac = float(frac)
        if frac <= 0:
            df2 = df.copy()
            tag = "none"
        else:
            drop_n = int(math.floor(frac * n))
            keep_idx = order[: max(0, n - drop_n)]
            df2 = df.iloc[keep_idx].copy()
            tag = f"drop_top_{100.0*frac:.2f}pct"

        ra = df2[col_ra].astype(float).to_numpy()
        dec = df2[col_dec].astype(float).to_numpy()
        unit_vecs = radec_to_unitvec(ra, dec)

        Dgeo_vec, _, Dgeo_amp = amp_and_dir(unit_vecs, None)
        weights = build_weights(df2, col_lgm, weight_mode, rng)
        Dobs_vec, _, Dobs_amp = amp_and_dir(unit_vecs, weights)

        null_amps = null_shuffle_weights(unit_vecs, weights, n_null=n_null, rng=rng, progress_every=0)
        mu, sig, z = summarize_null(Dobs_amp, null_amps)

        Dphys_amp = float(np.linalg.norm(Dobs_vec - Dgeo_vec))
        rows.append(dict(trim=tag, N=int(len(df2)), D_geo=float(Dgeo_amp), D_obs=float(Dobs_amp),
                         mu_null=float(mu), sigma_null=float(sig), z_score=float(z),
                         D_phys=float(Dphys_amp)))
    return pd.DataFrame(rows)


# -------------------------
# LaTeX table writer
# -------------------------
def df_to_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """
    Simple booktabs table, no fancy formatting.
    """
    cols = list(df.columns)
    header = " & ".join([str(c) for c in cols]) + r" \\"
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{" + "l" * len(cols) + "}")
    lines.append(r"\toprule")
    lines.append(header)
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                if np.isnan(v):
                    vals.append(r"$\mathrm{nan}$")
                else:
                    # default compact formatting
                    vals.append(f"{v:.4g}")
            else:
                vals.append(str(v))
        lines.append(" & ".join(vals) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{" + label + r"}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input CSV catalog")
    ap.add_argument("--label", default="catalog", help="Label for output folder")
    ap.add_argument("--outdir", default="results/maskaware_robustness", help="Root output directory")

    ap.add_argument("--col-ra", default="RA")
    ap.add_argument("--col-dec", default="DEC")
    ap.add_argument("--col-z", default="Z")
    ap.add_argument("--col-lgm", default="LGM_TOT_P50", help="log10 stellar mass column")

    ap.add_argument("--col-reliable", default="RELIABLE", help="Reliability column name (set empty string to disable)")
    ap.add_argument("--reliable-value", type=int, default=1)

    ap.add_argument("--zmin", type=float, default=0.02)
    ap.add_argument("--zmax", type=float, default=0.10)

    ap.add_argument("--weight-mode", default="mass", choices=["mass", "logmass", "rankmass", "clippedmass"])
    ap.add_argument("--clip-lo", type=float, default=1.0)
    ap.add_argument("--clip-hi", type=float, default=99.0)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-null", type=int, default=2000)

    ap.add_argument("--do-hemis", action="store_true")
    ap.add_argument("--do-jackknife", action="store_true")
    ap.add_argument("--jk-sectors", type=int, default=8)
    ap.add_argument("--jk-n-null", type=int, default=None, help="If set, overrides --n-null for jackknife only")

    ap.add_argument("--do-alt-nulls", action="store_true")
    ap.add_argument("--alt-n-null", type=int, default=None, help="If set, overrides --n-null for alt nulls")

    ap.add_argument("--do-trim", action="store_true")
    ap.add_argument("--trim-fracs", default="0,0.01,0.005,0.001",
                    help="Comma list of fractions to drop from TOP mass (e.g. 0.01 drops top 1%)")
    ap.add_argument("--trim-n-null", type=int, default=None, help="If set, overrides --n-null for trim tests")

    ap.add_argument("--n-max", type=int, default=None)
    ap.add_argument("--progress", action="store_true")

    args = ap.parse_args()
    rng = np.random.default_rng(int(args.seed))

    # Output dir
    out_dir = os.path.join(args.outdir, str(args.label))
    os.makedirs(out_dir, exist_ok=True)

    col_reliable = args.col_reliable if (args.col_reliable and args.col_reliable.strip()) else None

    df = load_and_filter(
        path=args.input,
        col_ra=args.col_ra, col_dec=args.col_dec, col_z=args.col_z,
        col_lgm=args.col_lgm,
        col_reliable=col_reliable,
        reliable_value=args.reliable_value,
        zmin=args.zmin, zmax=args.zmax,
        n_max=args.n_max,
    )

    # Core measurement (includes shuffled-weight null)
    core, unit_vecs, weights, Dgeo_vec, Dobs_hat = compute_core(
        df=df,
        col_ra=args.col_ra, col_dec=args.col_dec, col_z=args.col_z, col_lgm=args.col_lgm,
        weight_mode=args.weight_mode,
        n_null=int(args.n_null),
        rng=rng,
        progress=args.progress,
    )

    print(f"[CORE] {args.label} mode={args.weight_mode} N={core['N']} "
          f"|D_geo|={core['D_geo']:.4f} |D_obs|={core['D_obs']:.4f} "
          f"mu={core['mu_null']:.4f} sigma={core['sigma_null']:.4g} z={core['z_score']:.2f} "
          f"|D_phys|={core['D_phys']:.4g} angle={core['angle_geo_obs_deg']:.2f} deg")

    df_core = pd.DataFrame([core])
    df_core.to_csv(os.path.join(out_dir, "summary_global.csv"), index=False)
    tex = df_to_latex_table(
        df_core,
        caption=r"Global mask-aware dipole summary, including the vector-level excess "
                r"$\mathbf{D}_{\rm phys}\equiv \mathbf{D}_{\rm obs}-\mathbf{D}_{\rm geo}$.",
        label=f"tab:maskaware_global_{args.label}",
    )
    with open(os.path.join(out_dir, "summary_global_table.tex"), "w") as f:
        f.write(tex)

    # Hemisphere split
    if args.do_hemis:
        df_hemi = hemisphere_split(
            df=df,
            unit_vecs=unit_vecs,
            weights=weights,
            axis_hat=Dobs_hat,
            n_null=int(args.n_null),
            rng=rng,
            progress=args.progress,
        )
        df_hemi.to_csv(os.path.join(out_dir, "hemisphere_split.csv"), index=False)
        tex = df_to_latex_table(
            df_hemi,
            caption=r"Hemisphere split about the observed dipole axis "
                    r"($\hat{\mathbf{D}}_{\rm obs}$). Each hemisphere is evaluated with its own "
                    r"shuffled-weight null preserving the same footprint.",
            label=f"tab:maskaware_hemi_{args.label}",
        )
        with open(os.path.join(out_dir, "hemisphere_split_table.tex"), "w") as f:
            f.write(tex)
        print(f"[OK] Hemisphere split saved: {out_dir}/hemisphere_split.csv")

    # Jackknife stability
    if args.do_jackknife:
        jk_nnull = int(args.jk_n_null) if args.jk_n_null is not None else int(args.n_null)
        df_jk = jackknife_ra(
            df=df,
            unit_vecs=unit_vecs,
            weights=weights,
            col_ra=args.col_ra,
            K=int(args.jk_sectors),
            n_null=jk_nnull,
            rng=rng,
            progress=args.progress,
        )
        df_jk.to_csv(os.path.join(out_dir, "jackknife.csv"), index=False)
        tex = df_to_latex_table(
            df_jk,
            caption=r"Leave-one-out RA-sector jackknife. Each row recomputes the full mask-aware "
                    r"measurement and shuffled-weight null on the remaining sample, yielding a "
                    r"stability check on $\Delta|D|$ and its significance.",
            label=f"tab:maskaware_jackknife_{args.label}",
        )
        with open(os.path.join(out_dir, "jackknife_table.tex"), "w") as f:
            f.write(tex)
        print(f"[OK] Jackknife saved: {out_dir}/jackknife.csv")

    # Alternative nulls
    if args.do_alt_nulls:
        alt_nnull = int(args.alt_n_null) if args.alt_n_null is not None else int(args.n_null)
        Dobs_amp = float(core["D_obs"])
        df_alt = alt_nulls(
            df=df, unit_vecs=unit_vecs, weights=weights,
            col_ra=args.col_ra, col_dec=args.col_dec, col_z=args.col_z,
            Dobs_amp=Dobs_amp,
            n_null=alt_nnull, rng=rng,
        )
        df_alt.to_csv(os.path.join(out_dir, "alt_nulls.csv"), index=False)
        tex = df_to_latex_table(
            df_alt,
            caption=r"Alternative null models. These tests stress the directional correlation "
                    r"by destroying sky structure (shuffle RA/DEC) or controlling mass--redshift "
                    r"coupling (shuffle weights within $z$ bins).",
            label=f"tab:maskaware_alt_nulls_{args.label}",
        )
        with open(os.path.join(out_dir, "alt_nulls_table.tex"), "w") as f:
            f.write(tex)
        print(f"[OK] Alternative nulls saved: {out_dir}/alt_nulls.csv")

    # Trim tests
    if args.do_trim:
        trims = [float(x.strip()) for x in str(args.trim_fracs).split(",") if x.strip() != ""]
        trim_nnull = int(args.trim_n_null) if args.trim_n_null is not None else int(args.n_null)
        df_trim = trimmed_tests(
            df=df,
            col_lgm=args.col_lgm,
            col_ra=args.col_ra, col_dec=args.col_dec, col_z=args.col_z,
            weight_mode=args.weight_mode,
            trims=trims,
            n_null=trim_nnull,
            rng=rng,
        )
        df_trim.to_csv(os.path.join(out_dir, "trimmed.csv"), index=False)
        tex = df_to_latex_table(
            df_trim,
            caption=r"Robustness to removing the most massive objects. Each row drops the top "
                    r"fraction of galaxies by $\log_{10} M_\star$ and recomputes the mask-aware "
                    r"dipole and null significance.",
            label=f"tab:maskaware_trimmed_{args.label}",
        )
        with open(os.path.join(out_dir, "trimmed_table.tex"), "w") as f:
            f.write(tex)
        print(f"[OK] Trim tests saved: {out_dir}/trimmed.csv")


if __name__ == "__main__":
    main()
