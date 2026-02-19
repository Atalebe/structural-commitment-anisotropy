#!/usr/bin/env python3
"""
make_tng_sky_catalog.py (v2)

Robust builder for "observer sky catalogs" from IllustrisTNG group catalogs.

Outputs CSV columns:
  RA (deg), DEC (deg), Z, MSTAR (Msun), LGM_TOT_P50 (log10 Msun), RELIABLE (=1)

Fix vs v1:
- Auto-detect Subhalo position dataset (some dumps don't name it SubhaloPos).
- Auto-detect Subhalo mass-type dataset if SubhaloMassType is not present.
"""

import argparse
import glob
import os
import sys
import numpy as np
import pandas as pd
import h5py

C_KMS = 299792.458


def _snap3(snap: int) -> str:
    return f"{int(snap):03d}"


def find_groupcat_files(tng_root: str, snap: int):
    s = _snap3(snap)
    root = os.path.abspath(tng_root)

    candidates = [
        os.path.join(root, f"groupcat_{s}"),
        os.path.join(root, f"groupcat-{s}"),
        os.path.join(root, f"groups_{s}"),
        os.path.join(root, f"groups-{s}"),
        os.path.join(root, "output", f"groupcat_{s}"),
        os.path.join(root, "output", f"groupcat-{s}"),
        os.path.join(root, "output", f"groups_{s}"),
        os.path.join(root, "output", f"groups-{s}"),
    ]

    patterns = [
        f"fof_subhalo_tab_{s}.*.hdf5",
        f"fof_subhalo_tab_{s}.hdf5",
        f"subhalo_tab_{s}.*.hdf5",
        f"subhalo_tab_{s}.hdf5",
    ]

    hits = []
    for d in candidates:
        if not os.path.isdir(d):
            continue
        for pat in patterns:
            hits.extend(glob.glob(os.path.join(d, pat)))

    hits = sorted(set(hits))
    if hits:
        return hits

    # last resort recursive
    rec = []
    rec.extend(glob.glob(os.path.join(root, "**", f"fof_subhalo_tab_{s}.*.hdf5"), recursive=True))
    rec.extend(glob.glob(os.path.join(root, "**", f"subhalo_tab_{s}.*.hdf5"), recursive=True))
    hits = sorted(set(rec))
    if not hits:
        raise FileNotFoundError(f"No matching groupcat files found under '{root}' for snap {s}.")
    return hits


def _pick_pos_dataset(subgrp: h5py.Group):
    """
    Pick a position dataset inside Subhalo group.

    Priority:
    1) "SubhaloPos"
    2) any dataset containing "Pos" with shape (N,3)
    3) any dataset with shape (N,3)
    """
    keys = list(subgrp.keys())
    if "SubhaloPos" in keys:
        ds = subgrp["SubhaloPos"]
        if len(ds.shape) == 2 and ds.shape[1] == 3:
            return "SubhaloPos"

    # pos-like
    pos_like = []
    any_3 = []
    for k in keys:
        obj = subgrp[k]
        if not hasattr(obj, "shape"):
            continue
        sh = obj.shape
        if len(sh) == 2 and sh[1] == 3 and sh[0] > 0:
            any_3.append(k)
            if "pos" in k.lower():
                pos_like.append(k)

    if pos_like:
        # choose the most "pos-ish" shortest name
        pos_like = sorted(pos_like, key=lambda x: (len(x), x))
        return pos_like[0]
    if any_3:
        any_3 = sorted(any_3, key=lambda x: (len(x), x))
        return any_3[0]

    raise KeyError("Could not find any Subhalo position dataset with shape (N,3).")


def _pick_masstype_dataset(subgrp: h5py.Group):
    """
    Pick a mass-type dataset inside Subhalo group.

    Priority:
    1) "SubhaloMassType"
    2) any dataset containing "MassType" with shape (N,>=6)
    3) any dataset containing "Mass" with shape (N,>=6)
    """
    keys = list(subgrp.keys())
    if "SubhaloMassType" in keys:
        ds = subgrp["SubhaloMassType"]
        if len(ds.shape) == 2 and ds.shape[0] > 0 and ds.shape[1] >= 6:
            return "SubhaloMassType"

    cands = []
    for k in keys:
        obj = subgrp[k]
        if not hasattr(obj, "shape"):
            continue
        sh = obj.shape
        if len(sh) == 2 and sh[0] > 0 and sh[1] >= 6:
            kl = k.lower()
            if "masstype" in kl:
                cands.append((0, k))
            elif "mass" in kl:
                cands.append((1, k))

    if cands:
        cands.sort(key=lambda t: (t[0], len(t[1]), t[1]))
        return cands[0][1]

    raise KeyError("Could not find any Subhalo mass-type dataset with shape (N,>=6).")


def read_subhalos(groupcat_files, verbose=False):
    all_pos = []
    all_mstar = []
    header = None
    pos_key = None
    masstype_key = None

    # detect keys from first file that actually contains data
    for fp in groupcat_files:
        with h5py.File(fp, "r") as f:
            if "Subhalo" not in f:
                continue
            g = f["Subhalo"]
            if header is None:
                hdr = f["Header"].attrs
                header = {
                    "BoxSize": float(hdr.get("BoxSize")),
                    "HubbleParam": float(hdr.get("HubbleParam")),
                }
            if pos_key is None:
                pos_key = _pick_pos_dataset(g)
            if masstype_key is None:
                masstype_key = _pick_masstype_dataset(g)

            # sanity: do we have non-empty arrays?
            try:
                n = g[pos_key].shape[0]
            except Exception:
                n = 0
            if n > 0:
                break

    if header is None or pos_key is None or masstype_key is None:
        raise RuntimeError("Could not detect required datasets from groupcat files.")

    if verbose:
        print(f"[INFO] Using Subhalo position field: {pos_key}")
        print(f"[INFO] Using Subhalo mass-type field: {masstype_key}")

    h = header["HubbleParam"]

    # now read across all chunks
    for fp in groupcat_files:
        with h5py.File(fp, "r") as f:
            if "Subhalo" not in f:
                continue
            g = f["Subhalo"]
            if pos_key not in g or masstype_key not in g:
                # skip weird chunks
                continue

            pos = np.array(g[pos_key], dtype=np.float64)
            mtype = np.array(g[masstype_key], dtype=np.float64)

            if pos.ndim != 2 or pos.shape[1] != 3:
                continue
            if mtype.ndim != 2 or mtype.shape[1] < 5:
                continue

            mstar_1e10_msun_over_h = mtype[:, 4]
            mstar_msun = mstar_1e10_msun_over_h * 1e10 / h

            all_pos.append(pos)
            all_mstar.append(mstar_msun)

    if not all_pos:
        raise RuntimeError("No usable subhalo chunks read. Check your HDF5 structure.")

    pos_ckpch = np.vstack(all_pos)
    mstar_msun = np.concatenate(all_mstar)
    return pos_ckpch, mstar_msun, header


def periodic_delta(d, box):
    return d - box * np.round(d / box)


def build_catalog(pos_ckpch, mstar_msun, header, observer_mode, seed, mstar_min_msun):
    box = header["BoxSize"]  # ckpc/h
    h = header["HubbleParam"]
    rng = np.random.default_rng(seed)

    if observer_mode == "random":
        obs = rng.uniform(0.0, box, size=3)
    elif observer_mode == "center":
        obs = np.array([0.5 * box, 0.5 * box, 0.5 * box], dtype=np.float64)
    elif observer_mode == "origin":
        obs = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    else:
        raise ValueError("observer_mode must be one of: random, center, origin")

    rel = pos_ckpch - obs[None, :]
    rel = periodic_delta(rel, box)

    r_ckpch = np.sqrt((rel ** 2).sum(axis=1))
    r_mpc = (r_ckpch / 1000.0) / h
    z = (100.0 * h * r_mpc) / C_KMS

    with np.errstate(invalid="ignore", divide="ignore"):
        ux = rel[:, 0] / r_ckpch
        uy = rel[:, 1] / r_ckpch
        uz = rel[:, 2] / r_ckpch

    ok = np.isfinite(ux) & np.isfinite(uy) & np.isfinite(uz) & (r_ckpch > 0)
    ux, uy, uz, z, mstar_msun = ux[ok], uy[ok], uz[ok], z[ok], mstar_msun[ok]

    sel = mstar_msun >= float(mstar_min_msun)
    ux, uy, uz, z, mstar_msun = ux[sel], uy[sel], uz[sel], z[sel], mstar_msun[sel]

    ra = (np.degrees(np.arctan2(uy, ux)) + 360.0) % 360.0
    dec = np.degrees(np.arcsin(np.clip(uz, -1.0, 1.0)))
    lgm = np.log10(np.clip(mstar_msun, 1e-30, None))

    df = pd.DataFrame(
        {
            "RA": ra.astype(np.float64),
            "DEC": dec.astype(np.float64),
            "Z": z.astype(np.float64),
            "MSTAR": mstar_msun.astype(np.float64),
            "LGM_TOT_P50": lgm.astype(np.float64),
            "RELIABLE": np.ones_like(lgm, dtype=np.int64),
        }
    )
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tng-root", required=True)
    ap.add_argument("--snap", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--observer", default="random", choices=["random", "center", "origin"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mstar-min", type=float, default=1e9)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    files = find_groupcat_files(args.tng_root, args.snap)
    if args.verbose:
        print(f"[INFO] Found {len(files)} groupcat files. Example: {files[0]}")

    pos, mstar, header = read_subhalos(files, verbose=args.verbose)

    if args.verbose:
        print(f"[INFO] Loaded subhalos: N={pos.shape[0]}  BoxSize={header['BoxSize']} ckpc/h  h={header['HubbleParam']}")

    df = build_catalog(
        pos_ckpch=pos,
        mstar_msun=mstar,
        header=header,
        observer_mode=args.observer,
        seed=args.seed,
        mstar_min_msun=args.mstar_min,
    )

    outdir = os.path.dirname(os.path.abspath(args.out))
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)

    df.to_csv(args.out, index=False)
    print(f"[OK] Wrote {args.out}")
    print(f"[OK] N={len(df)} z range: {df['Z'].min():.4g} .. {df['Z'].max():.4g}")


if __name__ == "__main__":
    main()
