#!/usr/bin/env python3
"""
run_tng_random_observer_suite.py

Random observer robustness suite for TNG50 and TNG300.

For each observer seed:
- call make_tng_sky_catalog.py to generate RA,DEC,Z,MSTAR,LGM_TOT_P50,RELIABLE
- apply z cuts
- compute D_geo, D_obs, mu_null, sigma_null, zscore, and delta amplitude Δ|D|
- repeat for chosen weight modes: mass, rankmass, logmass, clippedmass (mass and rankmass by default)

Outputs:
  results/tng_random_observer/<label>/summary.csv
  results/tng_random_observer/<label>/by_observer.csv
"""

import argparse
import os
import subprocess
import numpy as np
import pandas as pd


def radec_to_unit(ra_deg, dec_deg):
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.vstack([x, y, z]).T


def weights_from_mode(df, mode):
    if mode == "mass":
        return df["MSTAR"].to_numpy(dtype=float)
    if mode == "logmass":
        return df["LGM_TOT_P50"].to_numpy(dtype=float)
    if mode == "clippedmass":
        w = df["MSTAR"].to_numpy(dtype=float)
        if len(w) == 0:
            return w
        cap = np.nanpercentile(w, 99.0)
        return np.clip(w, None, cap)
    if mode == "rankmass":
        # ranks from 1..N, normalized to mean 1
        w = df["MSTAR"].to_numpy(dtype=float)
        order = np.argsort(w)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(w) + 1, dtype=float)
        ranks /= np.mean(ranks)
        return ranks
    raise ValueError(f"Unknown weight mode: {mode}")


def dipole_geo(u):
    v = u.mean(axis=0)
    return float(np.linalg.norm(v)), v


def dipole_weighted(u, w):
    w = np.asarray(w, dtype=float)
    if np.any(~np.isfinite(w)):
        w = np.where(np.isfinite(w), w, 0.0)
    s = w.sum()
    if s <= 0:
        v = u.mean(axis=0)
    else:
        v = (u * w[:, None]).sum(axis=0) / s
    return float(np.linalg.norm(v)), v


def shuffled_null(u, w, n_null, rng, progress=False):
    mags = np.empty(n_null, dtype=float)
    for i in range(n_null):
        wp = rng.permutation(w)
        mags[i] = dipole_weighted(u, wp)[0]
        if progress and (i + 1) % 200 == 0:
            print(f"  Realisation {i+1}/{n_null}")
    mu = float(mags.mean())
    sig = float(mags.std(ddof=1))
    return mu, sig


def run_make_catalog(make_script, tng_root, snap, out_csv, seed, mstar_min, verbose=False):
    cmd = [
        "python", make_script,
        "--tng-root", tng_root,
        "--snap", str(int(snap)),
        "--out", out_csv,
        "--observer", "random",
        "--seed", str(int(seed)),
        "--mstar-min", str(float(mstar_min)),
    ]
    if verbose:
        cmd.append("--verbose")
    print("[MAKE]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def analyze_catalog(csv_path, zmin, zmax, weight_modes, n_null, seed, progress=False):
    df = pd.read_csv(csv_path)
    if "RELIABLE" in df.columns:
        df = df[df["RELIABLE"].astype(int) == 1].copy()

    df = df.dropna(subset=["RA", "DEC", "Z", "MSTAR", "LGM_TOT_P50"]).copy()
    df = df[(df["Z"] >= zmin) & (df["Z"] <= zmax)].copy()

    u = radec_to_unit(df["RA"].to_numpy(), df["DEC"].to_numpy())
    d_geo_mag, _ = dipole_geo(u)

    out_rows = []
    for mode in weight_modes:
        rng = np.random.default_rng(seed + 1000 * abs(hash(mode)) % 100000)
        w = weights_from_mode(df, mode)
        d_obs_mag, _ = dipole_weighted(u, w)
        mu, sig = shuffled_null(u, w, n_null=n_null, rng=rng, progress=progress)
        z = (d_obs_mag - mu) / sig if sig > 0 else np.nan
        delta = d_obs_mag - mu
        out_rows.append(
            dict(
                mode=mode,
                N=int(len(df)),
                zmin=float(zmin),
                zmax=float(zmax),
                D_geo=float(d_geo_mag),
                D_obs=float(d_obs_mag),
                mu_null=float(mu),
                sigma_null=float(sig),
                zscore=float(z),
                delta=float(delta),
            )
        )
    return out_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-script", required=True, help="Path to make_tng_sky_catalog.py")
    ap.add_argument("--tng50-root", required=True)
    ap.add_argument("--tng300-root", required=True)
    ap.add_argument("--snap", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--n-observers", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--mstar-min", type=float, default=1e9)
    ap.add_argument("--n-null", type=int, default=2000)
    ap.add_argument("--weight-modes", default="mass,rankmass")
    ap.add_argument("--tng50-zmin", type=float, default=0.0)
    ap.add_argument("--tng50-zmax", type=float, default=0.01)
    ap.add_argument("--tng300-zmin", type=float, default=0.0)
    ap.add_argument("--tng300-zmax", type=float, default=0.06)
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--verbose-make", action="store_true")
    args = ap.parse_args()

    weight_modes = [x.strip() for x in args.weight_modes.split(",") if x.strip()]
    outdir = os.path.join("results", "tng_random_observer", args.label)
    catdir = os.path.join(outdir, "catalogs")
    os.makedirs(catdir, exist_ok=True)

    all_rows = []
    by_obs_rows = []

    def do_sim(tag, root, zmin, zmax):
        nonlocal all_rows, by_obs_rows
        print(f"\n===== {tag} random observer suite =====")
        for i in range(args.n_observers):
            seed = args.seed0 + i
            out_csv = os.path.join(catdir, f"{tag}_snap{args.snap:03d}_observer_random_seed{seed:04d}.csv")
            run_make_catalog(
                make_script=args.make_script,
                tng_root=root,
                snap=args.snap,
                out_csv=out_csv,
                seed=seed,
                mstar_min=args.mstar_min,
                verbose=args.verbose_make,
            )
            rows = analyze_catalog(
                csv_path=out_csv,
                zmin=zmin,
                zmax=zmax,
                weight_modes=weight_modes,
                n_null=args.n_null,
                seed=seed,
                progress=args.progress,
            )
            for r in rows:
                r["dataset"] = tag
                r["observer_seed"] = seed
                by_obs_rows.append(r)

        # Aggregate summary across observers, per weight mode
        df = pd.DataFrame(by_obs_rows)
        df_sim = df[df["dataset"] == tag].copy()
        for mode in weight_modes:
            d = df_sim[df_sim["mode"] == mode]
            if len(d) == 0:
                continue
            all_rows.append(
                dict(
                    dataset=tag,
                    mode=mode,
                    n_observers=int(len(d)),
                    N_mean=float(d["N"].mean()),
                    zmin=float(zmin),
                    zmax=float(zmax),
                    D_geo_mean=float(d["D_geo"].mean()),
                    D_obs_mean=float(d["D_obs"].mean()),
                    mu_null_mean=float(d["mu_null"].mean()),
                    sigma_null_mean=float(d["sigma_null"].mean()),
                    zscore_mean=float(d["zscore"].mean()),
                    zscore_std=float(d["zscore"].std(ddof=1)) if len(d) > 1 else 0.0,
                    delta_mean=float(d["delta"].mean()),
                    delta_std=float(d["delta"].std(ddof=1)) if len(d) > 1 else 0.0,
                )
            )

    do_sim("TNG50", args.tng50_root, args.tng50_zmin, args.tng50_zmax)
    do_sim("TNG300", args.tng300_root, args.tng300_zmin, args.tng300_zmax)

    df_by = pd.DataFrame(by_obs_rows)
    df_sum = pd.DataFrame(all_rows)

    df_by.to_csv(os.path.join(outdir, "by_observer.csv"), index=False)
    df_sum.to_csv(os.path.join(outdir, "summary.csv"), index=False)

    print("\n[OK] Saved outputs:")
    print(" ", os.path.join(outdir, "by_observer.csv"))
    print(" ", os.path.join(outdir, "summary.csv"))


if __name__ == "__main__":
    main()
