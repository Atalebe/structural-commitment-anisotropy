#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def radec_to_unit(ra_deg, dec_deg):
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.vstack([x, y, z]).T  # (N,3)

def vec_to_radec(v):
    v = np.asarray(v, dtype=float)
    v = v / (np.linalg.norm(v) + 1e-30)
    x, y, z = v
    ra = (np.rad2deg(np.arctan2(y, x)) + 360.0) % 360.0
    dec = np.rad2deg(np.arcsin(np.clip(z, -1.0, 1.0)))
    return ra, dec

def dipole_vector(unit_vecs, weights=None):
    if weights is None:
        v = unit_vecs.mean(axis=0)
    else:
        w = np.asarray(weights, dtype=float)
        w = np.clip(w, 0.0, np.inf)
        sw = w.sum()
        if sw <= 0:
            v = unit_vecs.mean(axis=0)
        else:
            v = (unit_vecs * w[:, None]).sum(axis=0) / sw
    amp = float(np.linalg.norm(v))
    ra, dec = vec_to_radec(v)
    return v, amp, ra, dec

def build_weights(lgm, mode):
    lgm = np.asarray(lgm, dtype=float)

    if mode == "mass":
        return np.power(10.0, lgm)
    if mode == "logmass":
        # still nonnegative, preserves ordering
        return np.clip(lgm - np.nanmin(lgm), 0.0, np.inf)
    if mode == "rankmass":
        # ranks in [1..N], normalize to [0..1]
        order = np.argsort(lgm)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(lgm) + 1, dtype=float)
        return ranks / float(len(lgm))
    if mode == "clippedmass":
        w = np.power(10.0, lgm)
        hi = np.nanpercentile(w, 99.0)
        return np.clip(w, 0.0, hi)

    raise ValueError(f"Unknown weight mode: {mode}")

def shuffled_null_amplitudes(unit_vecs, weights, n_null, rng):
    n = len(weights)
    amps = np.empty(n_null, dtype=float)
    for i in range(n_null):
        perm = rng.permutation(n)
        _, amp, _, _ = dipole_vector(unit_vecs, weights=weights[perm])
        amps[i] = amp
    return amps

def load_catalog(path, col_ra, col_dec, col_z, col_mass, col_reliable,
                 zmin, zmax, n_max, rng, name="CAT"):
    df = pd.read_csv(path)
    need = [col_ra, col_dec, col_z, col_mass]
    if col_reliable and col_reliable.strip():
        need.append(col_reliable)

    # strict: drop rows missing required cols
    df = df.dropna(subset=need)

    if col_reliable and col_reliable.strip():
        df = df[df[col_reliable].astype(int) == 1]

    df = df[(df[col_z] >= zmin) & (df[col_z] <= zmax)]

    if n_max is not None and len(df) > n_max:
        # sample deterministically under rng for reproducibility
        idx = rng.choice(df.index.values, size=int(n_max), replace=False)
        df = df.loc[idx]

    df = df.reset_index(drop=True)

    return df

def analyze_one(label, df, col_ra, col_dec, col_mass, weight_mode,
                n_null, seed):
    rng = np.random.default_rng(seed)

    ra = df[col_ra].to_numpy(dtype=float)
    dec = df[col_dec].to_numpy(dtype=float)
    unit = radec_to_unit(ra, dec)

    # geometric
    v_geo, d_geo, ra_geo, dec_geo = dipole_vector(unit, weights=None)

    # weights
    lgm = df[col_mass].to_numpy(dtype=float)  # assumed log10(M*)
    w = build_weights(lgm, weight_mode)

    # observed
    v_obs, d_obs, ra_obs, dec_obs = dipole_vector(unit, weights=w)

    # angle between vectors
    u_geo = v_geo / (np.linalg.norm(v_geo) + 1e-30)
    u_obs = v_obs / (np.linalg.norm(v_obs) + 1e-30)
    cosang = float(np.clip(np.dot(u_geo, u_obs), -1.0, 1.0))
    ang_deg = float(np.rad2deg(np.arccos(cosang)))

    # null
    amps = shuffled_null_amplitudes(unit, w, n_null, rng)
    mu = float(np.mean(amps))
    sig = float(np.std(amps, ddof=1)) if n_null > 1 else float("nan")
    z = (d_obs - mu) / (sig + 1e-30)

    return dict(
        label=label,
        weight_mode=weight_mode,
        N=int(len(df)),
        d_geo=d_geo, ra_geo=ra_geo, dec_geo=dec_geo,
        d_obs=d_obs, ra_obs=ra_obs, dec_obs=dec_obs,
        angle_deg=ang_deg,
        mu_null=mu, sigma_null=sig,
        delta=d_obs - mu,
        z=z,
    )

def to_tex_table(rows, outpath):
    df = pd.DataFrame(rows)
    # keep display-friendly columns
    keep = ["label","weight_mode","N","d_geo","d_obs","mu_null","sigma_null","delta","z"]
    d = df[keep].copy()

    # format
    def f(x, nd=4): return f"{x:.{nd}f}"
    def fz(x): return f"{x:.2f}"

    lines = []
    lines.append(r"\begin{tabular}{llrcccccc}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Weighting & $N$ & $|D_{\rm geo}|$ & $|D_{\rm obs}|$ & $\langle|D|\rangle_{\rm null}$ & $\sigma_{\rm null}$ & $\Delta|D|$ & $z$ \\")
    lines.append(r"\midrule")
    for _, r in d.iterrows():
        lines.append(
            f"{r['label']} & {r['weight_mode']} & {int(r['N'])} & "
            f"{f(r['d_geo'])} & {f(r['d_obs'])} & {f(r['mu_null'])} & {f(r['sigma_null'])} & {f(r['delta'])} & {fz(r['z'])} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    with open(outpath, "w") as f:
        f.write("\n".join(lines) + "\n")

def plot_compare(rows, outpath):
    labels = [f"{r['label']}\n{r['weight_mode']}" for r in rows]
    x = np.arange(len(rows))

    d_geo = [r["d_geo"] for r in rows]
    d_obs = [r["d_obs"] for r in rows]

    delta = [r["delta"] for r in rows]
    sig = [r["sigma_null"] for r in rows]
    ztxt = [r["z"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax1.plot(x, d_geo, marker="o", label=r"$|D_{\rm geo}|$")
    ax1.plot(x, d_obs, marker="s", label=r"$|D_{\rm obs}|$")
    ax1.set_ylabel("Dipole amplitude")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2.errorbar(x, delta, yerr=sig, fmt="o")
    ax2.axhline(0.0, linewidth=1.0)
    ax2.set_ylabel(r"$\Delta|D| = |D_{\rm obs}| - \langle|D|\rangle_{\rm null}$")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=0)
    ax2.grid(True, alpha=0.3)

    for xi, yi, zi in zip(x, delta, ztxt):
        ax2.text(xi, yi + (max(sig) if len(sig) else 0.0)*0.15, f"{zi:.2f}$\\sigma$", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdss", required=True)
    ap.add_argument("--tng50", required=True)
    ap.add_argument("--tng300", required=True)

    ap.add_argument("--outdir", default="results/maskaware_compare_zcuts")
    ap.add_argument("--n-null", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-max", type=int, default=None)

    # per-dataset z cuts
    ap.add_argument("--sdss-zmin", type=float, default=0.02)
    ap.add_argument("--sdss-zmax", type=float, default=0.10)
    ap.add_argument("--tng50-zmin", type=float, default=0.00)
    ap.add_argument("--tng50-zmax", type=float, default=0.06)
    ap.add_argument("--tng300-zmin", type=float, default=0.00)
    ap.add_argument("--tng300-zmax", type=float, default=0.06)

    # column mapping
    ap.add_argument("--sdss-cols", default="RA,DEC,Z,LGM_TOT_P50,RELIABLE")
    ap.add_argument("--tng-cols", default="RA,DEC,Z,MSTAR")  # for sky catalogs
    ap.add_argument("--sdss-reliable-col", default="RELIABLE")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # SDSS columns
    sdss_ra, sdss_dec, sdss_z, sdss_lgm, sdss_rel = [s.strip() for s in args.sdss_cols.split(",")]

    # TNG columns (sky catalogs are RA,DEC,Z,MSTAR)
    tng_ra, tng_dec, tng_z, tng_mstar = [s.strip() for s in args.tng_cols.split(",")]

    # load SDSS
    df_sdss = load_catalog(
        args.sdss, sdss_ra, sdss_dec, sdss_z, sdss_lgm, sdss_rel,
        args.sdss_zmin, args.sdss_zmax, args.n_max, rng, name="SDSS"
    )

    # load TNG50
    df_50 = load_catalog(
        args.tng50, tng_ra, tng_dec, tng_z, tng_mstar, "",
        args.tng50_zmin, args.tng50_zmax, args.n_max, rng, name="TNG50"
    )
    # convert MSTAR -> log10(MSTAR) for unified weight builder
    df_50 = df_50.rename(columns={tng_mstar: "LGM"})
    df_50["LGM"] = np.log10(df_50["LGM"].astype(float).clip(lower=1e-30))

    # load TNG300
    df_300 = load_catalog(
        args.tng300, tng_ra, tng_dec, tng_z, tng_mstar, "",
        args.tng300_zmin, args.tng300_zmax, args.n_max, rng, name="TNG300"
    )
    df_300 = df_300.rename(columns={tng_mstar: "LGM"})
    df_300["LGM"] = np.log10(df_300["LGM"].astype(float).clip(lower=1e-30))

    rows = []
    # SDSS rows (mass + rankmass)
    rows.append(analyze_one("SDSS", df_sdss, sdss_ra, sdss_dec, sdss_lgm, "mass", args.n_null, args.seed))
    rows.append(analyze_one("SDSS", df_sdss, sdss_ra, sdss_dec, sdss_lgm, "rankmass", args.n_null, args.seed))

    # TNG rows
    rows.append(analyze_one("TNG50", df_50, tng_ra, tng_dec, "LGM", "mass", args.n_null, args.seed))
    rows.append(analyze_one("TNG50", df_50, tng_ra, tng_dec, "LGM", "rankmass", args.n_null, args.seed))

    rows.append(analyze_one("TNG300", df_300, tng_ra, tng_dec, "LGM", "mass", args.n_null, args.seed))
    rows.append(analyze_one("TNG300", df_300, tng_ra, tng_dec, "LGM", "rankmass", args.n_null, args.seed))

    # print summary
    for r in rows:
        print(f"[OK] {r['label']} {r['weight_mode']}: "
              f"|D_geo|={r['d_geo']:.4f} |D_obs|={r['d_obs']:.4f} "
              f"mu={r['mu_null']:.4f} sigma={r['sigma_null']:.4f} z={r['z']:.2f}")

    # write outputs
    out_csv = os.path.join(args.outdir, "maskaware_summary.csv")
    out_tex = os.path.join(args.outdir, "maskaware_summary_table.tex")
    out_fig = os.path.join(args.outdir, "fig_maskaware_sdss_tng_compare.png")

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    to_tex_table(rows, out_tex)
    plot_compare(rows, out_fig)

    print("\nSaved outputs:")
    print(f"  {out_csv}")
    print(f"  {out_tex}")
    print(f"  {out_fig}")

if __name__ == "__main__":
    main()
