#!/usr/bin/env python3
import os
import re
import csv
import math
import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def to_sdsscols_catalog(in_csv: str, out_csv: str, mode: str) -> str:
    """
    Convert a sky catalog to SDSS-like columns expected by sdss_structural_dipole_mask_null.py:
      RA, DEC, Z, LGM_TOT_P50, RELIABLE
    If input already contains LGM_TOT_P50, keep it.
    If input contains MSTAR, compute log10(MSTAR).
    """
    df = pd.read_csv(in_csv)
    cols = set(df.columns)

    # Normalize column names if needed
    # We assume sky catalogs are exactly RA,DEC,Z plus either MSTAR or LGM_TOT_P50.
    need = ["RA", "DEC", "Z"]
    for c in need:
        if c not in cols:
            raise ValueError(f"[{mode}] Missing required column '{c}' in {in_csv}. Found: {list(df.columns)}")

    if "LGM_TOT_P50" not in cols:
        if "MSTAR" not in cols:
            raise ValueError(
                f"[{mode}] Need either 'LGM_TOT_P50' (log10 mass) or 'MSTAR' (linear) in {in_csv}. "
                f"Found: {list(df.columns)}"
            )
        m = df["MSTAR"].astype(float).clip(lower=1e-30)
        df["LGM_TOT_P50"] = np.log10(m)

    if "RELIABLE" not in cols:
        df["RELIABLE"] = 1

    df_out = df[["RA", "DEC", "Z", "LGM_TOT_P50", "RELIABLE"]].copy()
    df_out.to_csv(out_csv, index=False)
    return out_csv


def run_maskaware(
    script: str,
    in_csv: str,
    out_npz: str,
    weight_mode: str,
    n_null: int,
    seed: int,
    zmin: float,
    zmax: float,
    col_ra: str,
    col_dec: str,
    col_z: str,
    col_lgm: str,
    col_reliable: str,
    extra_args: list[str],
    log_path: str,
) -> str:
    """
    Run sdss_structural_dipole_mask_null.py and return stdout as text.
    """
    cmd = [
        "python", script,
        "--input", in_csv,
        "--out", out_npz,
        "--seed", str(seed),
        "--n-null", str(n_null),
        "--zmin", str(zmin),
        "--zmax", str(zmax),
        "--weight-mode", weight_mode,
        "--col-ra", col_ra,
        "--col-dec", col_dec,
        "--col-z", col_z,
        "--col-lgm", col_lgm,
        "--col-reliable", col_reliable,
    ] + extra_args

    proc = subprocess.run(cmd, capture_output=True, text=True)
    txt = (proc.stdout or "") + "\n" + (proc.stderr or "")

    # Save log no matter what
    Path(log_path).write_text(txt)

    if proc.returncode != 0:
        raise RuntimeError(
            f"Run failed for input={in_csv}, weight_mode={weight_mode}. "
            f"See log: {log_path}\n\nLast ~30 lines:\n" + "\n".join(txt.splitlines()[-30:])
        )

    return txt


def parse_output(txt: str) -> dict:
    """
    Parse key numbers from your script output.
    We parse by regex so we don't depend on the internal .npz schema.
    """
    out = {}

    mN = re.search(r"Total galaxies used:\s*N\s*=\s*([0-9]+)", txt)
    if mN:
        out["N"] = int(mN.group(1))

    mzmin = re.search(r"Redshift range in sample:\s*z_min\s*=\s*([0-9.]+)", txt)
    mzmax = re.search(r"Redshift range in sample:\s*z_min\s*=\s*[0-9.]+,\s*z_max\s*=\s*([0-9.]+)", txt)
    if mzmin:
        out["z_min"] = float(mzmin.group(1))
    if mzmax:
        out["z_max"] = float(mzmax.group(1))

    # Dipole amplitudes
    mDgeo = re.search(r"\|D_geo\|\s*=\s*([0-9.]+)", txt)
    mDobs = re.search(r"\|D_obs\|\s*=\s*([0-9.]+)", txt)
    if mDgeo:
        out["D_geo"] = float(mDgeo.group(1))
    if mDobs:
        out["D_obs"] = float(mDobs.group(1))

    # Directions (first occurrence for geo, second for obs in the global block)
    dirs = re.findall(r"\(RA,\s*DEC\)\s*=\s*\(([-0-9.]+)\s*deg,\s*([-0-9.]+)\s*deg\)", txt)
    if len(dirs) >= 1:
        out["RA_geo"], out["DEC_geo"] = float(dirs[0][0]), float(dirs[0][1])
    if len(dirs) >= 2:
        out["RA_obs"], out["DEC_obs"] = float(dirs[1][0]), float(dirs[1][1])

    mang = re.search(r"Angle\(D_geo,\s*D_obs\)\s*=\s*([0-9.]+)\s*deg", txt)
    if mang:
        out["angle_deg"] = float(mang.group(1))

    # Null stats
    mmu = re.search(r"<\|D\|>_null\s*=\s*([0-9.]+)", txt)
    msig = re.search(r"sigma_null\s*=\s*([0-9.]+)", txt)
    mz = re.search(r"z-score\s*=\s*([0-9.]+)\s*sigma", txt)
    if mmu:
        out["mu_null"] = float(mmu.group(1))
    if msig:
        out["sigma_null"] = float(msig.group(1))
    if mz:
        out["z_score"] = float(mz.group(1))

    # Derived
    if "D_obs" in out and "mu_null" in out:
        out["Delta"] = out["D_obs"] - out["mu_null"]
    if "Delta" in out and "sigma_null" in out and out["sigma_null"] > 0:
        out["Delta_over_sigma"] = out["Delta"] / out["sigma_null"]

    return out


def write_latex_table(rows: list[dict], out_tex: str) -> None:
    """
    Emit a simple LaTeX table snippet (no fancy packages required).
    """
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Mask-aware shuffled-weight null comparison across SDSS DR8 and synthetic-sky catalogs from TNG50 and TNG300. "
                 r"Columns report the geometric dipole $|D_{\rm geo}|$, the weighted dipole $|D_{\rm obs}|$, the null mean $\langle|D|\rangle_{\rm null}$, "
                 r"the null scatter $\sigma_{\rm null}$, the excess $\Delta|D|$, and the corresponding significance.}")
    lines.append(r"\label{tab:maskaware_compare}")
    lines.append(r"\begin{tabular}{l l r c c c c c c}")
    lines.append(r"\hline")
    lines.append(r"Dataset & Weight & $N$ & $|D_{\rm geo}|$ & $|D_{\rm obs}|$ & $\langle|D|\rangle_{\rm null}$ & $\sigma_{\rm null}$ & $\Delta|D|$ & $z$--score \\")
    lines.append(r"\hline")

    for r in rows:
        ds = r["dataset"]
        w  = r["weight_mode"]
        N  = r.get("N", None)
        Dg = r.get("D_geo", None)
        Do = r.get("D_obs", None)
        mu = r.get("mu_null", None)
        sg = r.get("sigma_null", None)
        De = r.get("Delta", None)
        zs = r.get("z_score", None)

        def fmt(x, n=4):
            if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
                return r"--"
            return f"{x:.{n}f}"

        line = rf"{ds} & {w} & {N if N is not None else '--'} & {fmt(Dg)} & {fmt(Do)} & {fmt(mu)} & {fmt(sg)} & {fmt(De)} & {zs:.2f} \\"
        lines.append(line)

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    Path(out_tex).write_text("\n".join(lines) + "\n")


def make_plot(rows: list[dict], out_png: str) -> None:
    """
    Plot:
      Panel A: |D_geo| and |D_obs|
      Panel B: Delta|D| with sigma_null error bars, annotate z-score
    """
    # Keep deterministic ordering
    order = ["SDSS", "TNG50", "TNG300"]
    weights = ["mass", "rankmass"]

    # Build lookup
    by = {(r["dataset"], r["weight_mode"]): r for r in rows}

    xlabels = []
    D_geo = []
    D_obs = []
    Delta = []
    Sigma = []
    Zs = []

    for ds in order:
        for w in weights:
            r = by.get((ds, w))
            xlabels.append(f"{ds}\n{w}")
            if r is None:
                D_geo.append(np.nan); D_obs.append(np.nan)
                Delta.append(np.nan); Sigma.append(np.nan); Zs.append(np.nan)
                continue
            D_geo.append(r.get("D_geo", np.nan))
            D_obs.append(r.get("D_obs", np.nan))
            Delta.append(r.get("Delta", np.nan))
            Sigma.append(r.get("sigma_null", np.nan))
            Zs.append(r.get("z_score", np.nan))

    x = np.arange(len(xlabels))

    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    # Panel A
    ax1.plot(x, D_geo, marker="o", linestyle="-", label=r"$|D_{\rm geo}|$")
    ax1.plot(x, D_obs, marker="s", linestyle="-", label=r"$|D_{\rm obs}|$")
    ax1.set_ylabel(r"Dipole amplitude")
    ax1.set_xticks(x)
    ax1.set_xticklabels(xlabels)
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    # Panel B
    ax2.errorbar(x, Delta, yerr=Sigma, fmt="o", linestyle="none", label=r"$\Delta|D|$")
    ax2.set_ylabel(r"$\Delta|D| = |D_{\rm obs}|-\langle|D|\rangle_{\rm null}$")
    ax2.set_xticks(x)
    ax2.set_xticklabels(xlabels)
    ax2.grid(True, alpha=0.3)

    # Annotate z-scores
    for xi, d, z in zip(x, Delta, Zs):
        if np.isfinite(d) and np.isfinite(z):
            ax2.annotate(f"{z:.2f}$\\sigma$", (xi, d), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdss", default="data/sdss_dr8/sdss_dr8_analysis_base_v1.csv")
    ap.add_argument("--tng50", default="data/tng50/tng50_sky_catalog_snap072.csv")
    ap.add_argument("--tng300", default="data/tng300/tng300_sky_catalog_snap072.csv")
    ap.add_argument("--n-null", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zmin", type=float, default=0.0)
    ap.add_argument("--zmax", type=float, default=1.0)
    ap.add_argument("--sdss-script", default="sdss_structural_dipole_mask_null.py")
    ap.add_argument("--outdir", default="results/maskaware_compare")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    logsdir = outdir / "logs"
    ensure_dir(outdir)
    ensure_dir(logsdir)

    # Prepare catalogs
    sdss_csv = args.sdss
    tng50_sdsscols = str(outdir / "tng50_sdsscols.csv")
    tng300_sdsscols = str(outdir / "tng300_sdsscols.csv")

    to_sdsscols_catalog(args.tng50, tng50_sdsscols, "TNG50")
    to_sdsscols_catalog(args.tng300, tng300_sdsscols, "TNG300")

    datasets = [
        ("SDSS", sdss_csv),
        ("TNG50", tng50_sdsscols),
        ("TNG300", tng300_sdsscols),
    ]

    weight_modes = ["mass", "rankmass"]

    rows = []
    for ds_name, ds_csv in datasets:
        for wm in weight_modes:
            out_npz = str(outdir / f"{ds_name.lower()}_{wm}_null{args.n_null}_seed{args.seed}.npz")
            log_path = str(logsdir / f"{ds_name.lower()}_{wm}_null{args.n_null}_seed{args.seed}.log")

            # NOTE:
            # We point the script at SDSS-like column names, even for TNG.
            # The script expects a log-mass column (LGM_TOT_P50) for weight construction.
            txt = run_maskaware(
                script=args.sdss_script,
                in_csv=ds_csv,
                out_npz=out_npz,
                weight_mode=wm,
                n_null=args.n_null,
                seed=args.seed,
                zmin=args.zmin,
                zmax=args.zmax,
                col_ra="RA",
                col_dec="DEC",
                col_z="Z",
                col_lgm="LGM_TOT_P50",
                col_reliable="RELIABLE",
                extra_args=[],
                log_path=log_path,
            )

            parsed = parse_output(txt)
            parsed["dataset"] = ds_name
            parsed["weight_mode"] = wm
            parsed["out_npz"] = out_npz
            parsed["log"] = log_path
            rows.append(parsed)

            print(f"[OK] {ds_name} {wm}: "
                  f"|D_geo|={parsed.get('D_geo','--')} "
                  f"|D_obs|={parsed.get('D_obs','--')} "
                  f"mu={parsed.get('mu_null','--')} "
                  f"sigma={parsed.get('sigma_null','--')} "
                  f"z={parsed.get('z_score','--')}")

    # Save CSV summary
    out_csv = outdir / "maskaware_summary.csv"
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Save LaTeX table snippet
    out_tex = outdir / "maskaware_summary_table.tex"
    write_latex_table(rows, str(out_tex))

    # Make plot
    out_png = outdir / "fig_maskaware_sdss_tng_compare.png"
    make_plot(rows, str(out_png))

    print("\nSaved outputs:")
    print(f"  {out_csv}")
    print(f"  {out_tex}")
    print(f"  {out_png}")
    print(f"  logs: {logsdir}/")


if __name__ == "__main__":
    main()

