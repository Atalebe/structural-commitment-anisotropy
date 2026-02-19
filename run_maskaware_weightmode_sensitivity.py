#!/usr/bin/env python3
"""
run_maskaware_weightmode_sensitivity.py

Weight-mode sensitivity test for mask-aware dipole estimator.
Runs the same pipeline across SDSS / TNG50 / TNG300 catalogs for multiple weight modes:
  mass, logmass, clippedmass, rankmass

Outputs:
  results/maskaware_weightmodes/<label>/weightmode_summary.csv
  results/maskaware_weightmodes/<label>/weightmode_summary_table.tex
  results/maskaware_weightmodes/<label>/fig_weightmode_summary.png
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Core math helpers
# ----------------------------

def radec_to_unitvec(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """Convert RA/DEC [deg] to unit vectors in Cartesian coordinates."""
    ra = np.deg2rad(ra_deg.astype(np.float64))
    dec = np.deg2rad(dec_deg.astype(np.float64))
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    u = np.vstack([x, y, z]).T
    # numerical safety
    n = np.linalg.norm(u, axis=1)
    n[n == 0] = 1.0
    return (u.T / n).T


def dipole_unweighted(u: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return vector mean and magnitude for unweighted dipole."""
    v = u.mean(axis=0)
    return v, float(np.linalg.norm(v))


def dipole_weighted(u: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return weighted vector mean and magnitude. w must be non-negative."""
    w = w.astype(np.float64)
    ws = float(w.sum())
    if ws <= 0:
        v = np.zeros(3, dtype=np.float64)
        return v, 0.0
    # 3xN dot N -> 3 (fast BLAS)
    v = (u.T @ w) / ws
    return v, float(np.linalg.norm(v))


def vec_to_radec(v: np.ndarray) -> Tuple[float, float]:
    """Vector -> (RA, DEC) in degrees."""
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    r = np.sqrt(vx * vx + vy * vy + vz * vz)
    if r == 0:
        return np.nan, np.nan
    vx, vy, vz = vx / r, vy / r, vz / r
    ra = np.rad2deg(np.arctan2(vy, vx)) % 360.0
    dec = np.rad2deg(np.arcsin(vz))
    return float(ra), float(dec)


def angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle between vectors in degrees."""
    a = np.linalg.norm(v1)
    b = np.linalg.norm(v2)
    if a == 0 or b == 0:
        return np.nan
    c = float(np.clip(np.dot(v1, v2) / (a * b), -1.0, 1.0))
    return float(np.rad2deg(np.arccos(c)))


# ----------------------------
# Weight modes
# ----------------------------

def build_weights(
    lgm: np.ndarray,
    mode: str,
    clip_quantile: float = 0.99,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    lgm is log10(M*) (or any monotonic mass proxy in log10 space).
    Returns non-negative weights.
    """
    lgm = lgm.astype(np.float64)

    if mode == "mass":
        w = 10.0 ** lgm
        return w

    if mode == "logmass":
        # shift to be positive
        m = np.nanmin(lgm)
        w = (lgm - m) + eps
        # guard: if all equal
        w[~np.isfinite(w)] = eps
        return w

    if mode == "rankmass":
        # ranks 1..N (ties get average rank)
        s = pd.Series(lgm)
        r = s.rank(method="average").to_numpy(dtype=np.float64)
        # shift to positive
        r = r - np.nanmin(r) + 1.0
        r[~np.isfinite(r)] = 1.0
        return r

    if mode == "clippedmass":
        m = 10.0 ** lgm
        q = float(np.nanquantile(m, clip_quantile))
        if not np.isfinite(q) or q <= 0:
            q = np.nanmax(m[np.isfinite(m)])
        w = np.clip(m, 0.0, q)
        w[~np.isfinite(w)] = 0.0
        return w

    raise ValueError(f"Unknown weight mode: {mode}")


# ----------------------------
# Data loading
# ----------------------------

@dataclass
class CatalogCfg:
    name: str
    path: str
    col_ra: str
    col_dec: str
    col_z: str
    col_lgm: str
    col_reliable: str
    zmin: float
    zmax: float


def infer_lgm_column(df: pd.DataFrame) -> str:
    """
    Prefer SDSS-style logmass column, else LGM, else MSTAR -> log10(MSTAR).
    Returns name of a column containing log10 mass proxy.
    """
    for c in ["LGM_TOT_P50", "LGM", "lgm", "logMstar", "LOGMSTAR"]:
        if c in df.columns:
            return c
    if "MSTAR" in df.columns:
        df["LGM_FROM_MSTAR"] = np.log10(df["MSTAR"].astype(float).clip(lower=1e-30))
        return "LGM_FROM_MSTAR"
    raise KeyError("Could not infer a log-mass column. Need one of LGM_TOT_P50 / LGM / MSTAR.")


def load_catalog(cfg: CatalogCfg) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      u: unit vectors (N,3)
      z: redshift proxy (N,)
      lgm: log10 mass proxy (N,)
    Applies:
      reliable == 1 (if cfg.col_reliable != "")
      drop NaN in required columns
      z range cut
    """
    print(f"Loading {cfg.name} from:\n  {cfg.path}")
    df = pd.read_csv(cfg.path)
    print(f"Initial N (raw): {len(df)}")

    # reliability cut (optional)
    if cfg.col_reliable and cfg.col_reliable in df.columns:
        df = df[df[cfg.col_reliable].astype(int) == 1]
        print(f"After {cfg.col_reliable} == 1 cut: N = {len(df)}")

    # infer lgm if user gave placeholder
    col_lgm = cfg.col_lgm
    if col_lgm == "__AUTO__":
        col_lgm = infer_lgm_column(df)

    need = [cfg.col_ra, cfg.col_dec, cfg.col_z, col_lgm]
    for c in need:
        if c not in df.columns:
            raise KeyError(f"{cfg.name}: missing column '{c}' in {cfg.path}. Columns={list(df.columns)}")

    df = df.dropna(subset=need)
    print(f"After dropping NaN RA/DEC/Z/LGM: N = {len(df)}")

    z = df[cfg.col_z].astype(np.float64).to_numpy()
    m = (z >= cfg.zmin) & (z <= cfg.zmax)
    df = df.loc[m].copy()
    print(f"After {cfg.zmin:.3f} <= z <= {cfg.zmax:.3f} cut: N = {len(df)}")

    ra = df[cfg.col_ra].astype(np.float64).to_numpy()
    dec = df[cfg.col_dec].astype(np.float64).to_numpy()
    z = df[cfg.col_z].astype(np.float64).to_numpy()
    lgm = df[col_lgm].astype(np.float64).to_numpy()

    u = radec_to_unitvec(ra, dec)
    return u, z, lgm


# ----------------------------
# Null construction
# ----------------------------

def shuffled_weight_null(
    u: np.ndarray,
    w: np.ndarray,
    n_null: int,
    rng: np.random.Generator,
    progress_every: int = 200,
) -> Tuple[float, float]:
    """
    Mask-preserved shuffled-weight null:
      positions fixed, weights permuted across objects.
    Returns mean and std of |D| over realizations.
    """
    N = len(w)
    mags = np.empty(n_null, dtype=np.float64)

    # Use index permutation to avoid repeatedly copying w into new arrays with shuffle in place
    for i in range(n_null):
        perm = rng.permutation(N)
        _, mag = dipole_weighted(u, w[perm])
        mags[i] = mag
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  Realisation {i+1}/{n_null}")

    mu = float(mags.mean())
    sig = float(mags.std(ddof=1)) if n_null > 1 else 0.0
    return mu, sig


# ----------------------------
# Main runner
# ----------------------------

def run_one(
    cfg: CatalogCfg,
    weight_mode: str,
    n_null: int,
    seed: int,
    clip_quantile: float,
    progress: bool,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)

    u, z, lgm = load_catalog(cfg)
    N = len(lgm)

    v_geo, d_geo = dipole_unweighted(u)

    w = build_weights(lgm, weight_mode, clip_quantile=clip_quantile)
    v_obs, d_obs = dipole_weighted(u, w)

    print(f"\n===== {cfg.name} WEIGHT MODE: {weight_mode} =====")
    print(f"Total galaxies used: N = {N}")
    print(f"z range: z_min={z.min():.4f}, z_max={z.max():.4f}")
    ra_geo, dec_geo = vec_to_radec(v_geo)
    ra_obs, dec_obs = vec_to_radec(v_obs)
    print(f"|D_geo|={d_geo:.4f}  (RA,DEC)=({ra_geo:.1f},{dec_geo:.1f})")
    print(f"|D_obs|={d_obs:.4f}  (RA,DEC)=({ra_obs:.1f},{dec_obs:.1f})")
    ang = angle_deg(v_geo, v_obs)
    print(f"Angle(D_geo, D_obs) = {ang:.2f} deg")
    print(f"Args: seed={seed}, N_NULL={n_null}")

    print("Building shuffled-weight null (mask preserved) ...")
    mu, sig = shuffled_weight_null(
        u=u,
        w=w,
        n_null=n_null,
        rng=rng,
        progress_every=200 if progress else 0,
    )

    zscore = (d_obs - mu) / sig if sig > 0 else np.nan
    d_phys = d_obs - mu

    print("===== SHUFFLED NULL RESULTS =====")
    print(f"  <|D|>_null = {mu:.4f}")
    print(f"  sigma_null = {sig:.6f}")
    print(f"  |D_obs|    = {d_obs:.4f}")
    print(f"  z-score    = {zscore:.2f} sigma")
    print(f"  |D_phys|   = {d_phys:.5f}\n")

    return dict(
        dataset=cfg.name,
        weight=weight_mode,
        N=float(N),
        zmin=float(z.min()) if N > 0 else np.nan,
        zmax=float(z.max()) if N > 0 else np.nan,
        D_geo=d_geo,
        D_obs=d_obs,
        D_null=mu,
        sigma_null=sig,
        zscore=zscore,
        D_phys=d_phys,
        angle_deg=ang,
    )


def to_tex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    cols = [
        "dataset", "weight", "N", "D_geo", "D_obs", "D_null", "sigma_null", "zscore", "D_phys", "angle_deg"
    ]
    d = df[cols].copy()
    d["N"] = d["N"].round(0).astype(int)

    # formatting
    fmt = {
        "D_geo": "{:.4f}".format,
        "D_obs": "{:.4f}".format,
        "D_null": "{:.4f}".format,
        "sigma_null": "{:.6f}".format,
        "zscore": "{:.2f}".format,
        "D_phys": "{:.5f}".format,
        "angle_deg": "{:.2f}".format,
    }

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabular}{l l r r r r r r r r}")
    lines.append("\\hline")
    lines.append("Dataset & Weight & $N$ & $|D_{\\rm geo}|$ & $|D_{\\rm obs}|$ & $\\langle |D| \\rangle_{\\rm null}$ & $\\sigma_{\\rm null}$ & $z$ & $|D_{\\rm phys}|$ & Angle [deg]\\\\")
    lines.append("\\hline")

    for _, r in d.iterrows():
        row = []
        row.append(str(r["dataset"]))
        row.append(str(r["weight"]))
        row.append(str(int(r["N"])))
        for c in ["D_geo", "D_obs", "D_null", "sigma_null", "zscore", "D_phys", "angle_deg"]:
            row.append(fmt[c](float(r[c])))
        lines.append(" & ".join(row) + "\\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def plot_summary(df: pd.DataFrame, out_png: str) -> None:
    """
    Simple plot: z-score vs weight mode for each dataset.
    """
    # stable order
    mode_order = ["mass", "logmass", "clippedmass", "rankmass"]
    datasets = list(dict.fromkeys(df["dataset"].tolist()))  # preserve appearance order

    # map mode to x
    x = np.arange(len(mode_order), dtype=float)

    plt.figure()
    for ds in datasets:
        sub = df[df["dataset"] == ds].set_index("weight")
        y = [float(sub.loc[m, "zscore"]) if m in sub.index else np.nan for m in mode_order]
        plt.plot(x, y, marker="o", label=ds)

    plt.xticks(x, mode_order, rotation=0)
    plt.xlabel("weight mode")
    plt.ylabel("z-score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdss", default="", help="Path to SDSS CSV (optional).")
    ap.add_argument("--tng50", default="", help="Path to TNG50 CSV (optional).")
    ap.add_argument("--tng300", default="", help="Path to TNG300 CSV (optional).")

    ap.add_argument("--label", default="weightmode_suite", help="Label for output folder.")
    ap.add_argument("--n-null", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clip-quantile", type=float, default=0.99, help="Quantile for clippedmass mode.")

    # columns (global defaults, can be used for all)
    ap.add_argument("--col-ra", default="RA")
    ap.add_argument("--col-dec", default="DEC")
    ap.add_argument("--col-z", default="Z")
    ap.add_argument("--col-lgm", default="__AUTO__", help="Log-mass column. Use __AUTO__ to infer.")
    ap.add_argument("--col-reliable", default="RELIABLE", help="Set to empty string to disable reliability cut.")

    # z cuts per dataset
    ap.add_argument("--sdss-zmin", type=float, default=0.02)
    ap.add_argument("--sdss-zmax", type=float, default=0.10)
    ap.add_argument("--tng50-zmin", type=float, default=0.00)
    ap.add_argument("--tng50-zmax", type=float, default=0.06)
    ap.add_argument("--tng300-zmin", type=float, default=0.00)
    ap.add_argument("--tng300-zmax", type=float, default=0.06)

    ap.add_argument("--modes", default="mass,logmass,clippedmass,rankmass")
    ap.add_argument("--progress", action="store_true", help="Print null progress.")

    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if not modes:
        raise SystemExit("No weight modes provided.")

    out_dir = os.path.join("results", "maskaware_weightmodes", args.label)
    os.makedirs(out_dir, exist_ok=True)

    cfgs: List[CatalogCfg] = []
    if args.sdss:
        cfgs.append(CatalogCfg(
            name="SDSS",
            path=args.sdss,
            col_ra=args.col_ra, col_dec=args.col_dec, col_z=args.col_z,
            col_lgm=args.col_lgm, col_reliable=args.col_reliable,
            zmin=args.sdss_zmin, zmax=args.sdss_zmax
        ))
    if args.tng50:
        cfgs.append(CatalogCfg(
            name="TNG50",
            path=args.tng50,
            col_ra=args.col_ra, col_dec=args.col_dec, col_z=args.col_z,
            col_lgm=args.col_lgm, col_reliable=args.col_reliable,
            zmin=args.tng50_zmin, zmax=args.tng50_zmax
        ))
    if args.tng300:
        cfgs.append(CatalogCfg(
            name="TNG300",
            path=args.tng300,
            col_ra=args.col_ra, col_dec=args.col_dec, col_z=args.col_z,
            col_lgm=args.col_lgm, col_reliable=args.col_reliable,
            zmin=args.tng300_zmin, zmax=args.tng300_zmax
        ))

    if not cfgs:
        raise SystemExit("Provide at least one of --sdss, --tng50, --tng300")

    rows: List[Dict[str, float]] = []
    for cfg in cfgs:
        for mode in modes:
            res = run_one(
                cfg=cfg,
                weight_mode=mode,
                n_null=args.n_null,
                seed=args.seed,
                clip_quantile=args.clip_quantile,
                progress=args.progress,
            )
            rows.append(res)

    df = pd.DataFrame(rows)

    out_csv = os.path.join(out_dir, "weightmode_summary.csv")
    df.to_csv(out_csv, index=False)

    out_tex = os.path.join(out_dir, "weightmode_summary_table.tex")
    tex = to_tex_table(
        df,
        caption="Weight-mode sensitivity test for the mask-aware dipole estimator across SDSS and TNG mock skies.",
        label="tab:maskaware_weightmode_sensitivity",
    )
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(tex + "\n")

    out_png = os.path.join(out_dir, "fig_weightmode_summary.png")
    plot_summary(df, out_png)

    print("\nSaved outputs:")
    print(f"  {out_csv}")
    print(f"  {out_tex}")
    print(f"  {out_png}")


if __name__ == "__main__":
    main()
