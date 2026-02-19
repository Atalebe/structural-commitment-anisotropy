#!/usr/bin/env python3
"""
Rotate the stellar-mass field relative to a fixed angular footprint (mask-aware control).

Key idea:
- The SDSS footprint induces a large geometric dipole even for uniform weights.
- The mask-preserving shuffled-weight null (mu_null, sigma_null) is computed ONCE.
- We then rotate the mass field (weights) relative to the fixed footprint and
  compute z_rot = (|D_rot| - mu_null)/sigma_null for many random rotations.

This avoids the catastrophic cost of nesting a full shuffled null inside each rotation.

Outputs:
  results/maskaware_rotatefield/<label>/
    - obs_summary.txt
    - null_amplitudes.npy
    - rotations.csv
    - fig_rotatefield.png (optional)
"""

import os
import math
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------
# Geometry helpers
# ----------------------------
def radec_to_unit(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(ra_deg.astype(float))
    dec = np.deg2rad(dec_deg.astype(float))
    cosd = np.cos(dec)
    x = cosd * np.cos(ra)
    y = cosd * np.sin(ra)
    z = np.sin(dec)
    return np.vstack([x, y, z]).T


def unit_to_radec(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = u[:, 0], u[:, 1], u[:, 2]
    ra = np.rad2deg(np.arctan2(y, x)) % 360.0
    dec = np.rad2deg(np.arcsin(np.clip(z, -1.0, 1.0)))
    return ra, dec


def amp_from_vec(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def weighted_dipole_vec(U: np.ndarray, w: np.ndarray) -> np.ndarray:
    # U: (N,3) unit vectors
    # w: (N,) weights (non-negative)
    sw = float(np.sum(w))
    if sw <= 0:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    # Vector dipole = sum_i w_i U_i / sum_i w_i
    return (U.T @ w) / sw


def geometric_dipole_vec(U: np.ndarray) -> np.ndarray:
    # Unweighted mean unit vector
    return np.mean(U, axis=0)


# ----------------------------
# Random rotations
# ----------------------------
def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """
    Uniform random rotation matrix in SO(3) using quaternion method.
    """
    u1 = rng.random()
    u2 = rng.random()
    u3 = rng.random()

    q1 = math.sqrt(1 - u1) * math.sin(2 * math.pi * u2)
    q2 = math.sqrt(1 - u1) * math.cos(2 * math.pi * u2)
    q3 = math.sqrt(u1) * math.sin(2 * math.pi * u3)
    q4 = math.sqrt(u1) * math.cos(2 * math.pi * u3)

    # quaternion -> rotation matrix
    R = np.array([
        [1 - 2*(q3*q3 + q4*q4),     2*(q2*q3 - q1*q4),       2*(q2*q4 + q1*q3)],
        [2*(q2*q3 + q1*q4),         1 - 2*(q2*q2 + q4*q4),   2*(q3*q4 - q1*q2)],
        [2*(q2*q4 - q1*q3),         2*(q3*q4 + q1*q2),       1 - 2*(q2*q2 + q3*q3)],
    ], dtype=float)
    return R


def build_nearest_neighbor_index(U: np.ndarray):
    """
    Build a nearest-neighbor search structure on the unit vectors.
    Prefer scipy.spatial.cKDTree; fall back to sklearn if available.
    """
    try:
        from scipy.spatial import cKDTree  # type: ignore
        tree = cKDTree(U)
        def query_nn(X):
            d, idx = tree.query(X, k=1, workers=-1)
            return idx
        return query_nn, "scipy.cKDTree"
    except Exception:
        pass

    try:
        from sklearn.neighbors import NearestNeighbors  # type: ignore
        nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nn.fit(U)
        def query_nn(X):
            d, idx = nn.kneighbors(X, return_distance=True)
            return idx[:, 0]
        return query_nn, "sklearn.NearestNeighbors"
    except Exception as e:
        raise RuntimeError(
            "Need scipy or sklearn for nearest-neighbor mapping.\n"
            "Install one of them, or run this on a machine where it exists.\n"
            f"Underlying import error: {e}"
        )


def rotate_massfield_to_mask(U: np.ndarray, w: np.ndarray, rng: np.random.Generator, query_nn) -> tuple[np.ndarray, float]:
    """
    Rotate the mass field by a random rotation:
      - each weight w_i is moved from direction U_i to rotated direction U_i' = R U_i
      - then re-assigned onto the fixed footprint using nearest-neighbor mapping
    Returns:
      w_new: weights on the original footprint directions (length N)
      ang_deg: rotation angle proxy is not returned; instead return mean mapping distortion is ignored.
    """
    R = random_rotation_matrix(rng)
    U_rot = (U @ R.T).astype(float)

    idx = query_nn(U_rot)  # nearest neighbor indices in original footprint
    w_new = np.zeros_like(w, dtype=float)
    # many-to-one mapping allowed; accumulate
    np.add.at(w_new, idx, w)

    # No single “rotation angle” is meaningful here due to NN mapping; return NaN placeholder.
    return w_new, float("nan")


# ----------------------------
# Data + weights
# ----------------------------
def make_weights(df: pd.DataFrame, weight_mode: str, col_lgm: str) -> np.ndarray:
    lgm = df[col_lgm].to_numpy(dtype=float)

    if weight_mode == "mass":
        # interpret column as log10(M*)
        w = np.power(10.0, lgm)
        return w.astype(float)

    if weight_mode == "logmass":
        # linear in log-mass, clipped to be non-negative
        w = np.clip(lgm, a_min=0.0, a_max=None)
        return w.astype(float)

    if weight_mode == "rankmass":
        # ranks from 1..N
        order = np.argsort(lgm)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(lgm) + 1, dtype=float)
        return ranks

    if weight_mode == "clippedmass":
        # cap extremes to reduce leverage
        # clip in log-space at (median +/- 3*sigma_robust) using MAD
        med = float(np.median(lgm))
        mad = float(np.median(np.abs(lgm - med))) + 1e-12
        sig = 1.4826 * mad
        lo = med - 3.0 * sig
        hi = med + 3.0 * sig
        lgm_c = np.clip(lgm, lo, hi)
        w = np.power(10.0, lgm_c)
        return w.astype(float)

    raise ValueError(f"Unknown weight_mode={weight_mode}")


def load_and_filter(
    path: str,
    col_ra: str, col_dec: str, col_z: str,
    col_lgm: str,
    col_reliable: str,
    zmin: float, zmax: float,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    # reliable cut if column given and non-empty
    if col_reliable is not None and str(col_reliable).strip() != "":
        if col_reliable not in df.columns:
            raise KeyError(f"Reliable column '{col_reliable}' not found in CSV.")
        df = df[df[col_reliable] == 1].copy()

    need = [col_ra, col_dec, col_z, col_lgm]
    for c in need:
        if c not in df.columns:
            raise KeyError(f"Required column '{c}' not found in CSV. Found columns: {list(df.columns)}")

    df = df.dropna(subset=need).copy()
    df = df[(df[col_z] >= zmin) & (df[col_z] <= zmax)].copy()

    return df


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--outdir", default="results/maskaware_rotatefield")

    ap.add_argument("--col-ra", default="RA")
    ap.add_argument("--col-dec", default="DEC")
    ap.add_argument("--col-z", default="Z")
    ap.add_argument("--col-lgm", default="LGM_TOT_P50")
    ap.add_argument("--col-reliable", default="RELIABLE")

    ap.add_argument("--zmin", type=float, default=0.02)
    ap.add_argument("--zmax", type=float, default=0.10)

    ap.add_argument("--weight-mode", choices=["mass", "logmass", "rankmass", "clippedmass"], default="mass")

    ap.add_argument("--n-null", type=int, default=2000)
    ap.add_argument("--n-rot", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--progress-every", type=int, default=200)
    ap.add_argument("--make-fig", action="store_true")
    ap.add_argument("--float32", action="store_true", help="Use float32 unit vectors for speed/memory.")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    out = os.path.join(args.outdir, args.label)
    os.makedirs(out, exist_ok=True)

    # Load
    df = load_and_filter(
        args.input,
        args.col_ra, args.col_dec, args.col_z,
        args.col_lgm,
        args.col_reliable,
        args.zmin, args.zmax,
    )
    N = len(df)
    ra = df[args.col_ra].to_numpy(dtype=float)
    dec = df[args.col_dec].to_numpy(dtype=float)

    U = radec_to_unit(ra, dec)
    if args.float32:
        U = U.astype(np.float32)

    w = make_weights(df, args.weight_mode, args.col_lgm).astype(np.float64)

    # Observed + geometric
    D_geo_vec = geometric_dipole_vec(U.astype(np.float64))
    D_obs_vec = weighted_dipole_vec(U.astype(np.float64), w)

    D_geo = amp_from_vec(D_geo_vec)
    D_obs = amp_from_vec(D_obs_vec)

    # Baseline shuffled null ONCE
    null_amps = np.empty(args.n_null, dtype=np.float64)
    print(f"[OBS] N={N} |D_geo|={D_geo:.4f} |D_obs|={D_obs:.4f}", flush=True)
    print(f"[NULL] Building shuffled-weight null once: N_NULL={args.n_null}", flush=True)

    # Precompute for speed
    Ut = U.astype(np.float64).T
    sw = float(np.sum(w))

    for k in range(args.n_null):
        perm = rng.permutation(N)
        wperm = w[perm]
        v = (Ut @ wperm) / sw
        null_amps[k] = np.linalg.norm(v)
        if (k + 1) % args.progress_every == 0:
            print(f"  Realisation {k+1}/{args.n_null}", flush=True)

    mu = float(np.mean(null_amps))
    sig = float(np.std(null_amps, ddof=1))
    z = (D_obs - mu) / sig if sig > 0 else float("nan")
    delta = D_obs - mu

    # Save baseline
    np.save(os.path.join(out, "null_amplitudes.npy"), null_amps)
    with open(os.path.join(out, "obs_summary.txt"), "w") as f:
        f.write(f"N={N}\n")
        f.write(f"|D_geo|={D_geo:.6f}\n")
        f.write(f"|D_obs|={D_obs:.6f}\n")
        f.write(f"mu_null={mu:.6f}\n")
        f.write(f"sigma_null={sig:.6f}\n")
        f.write(f"z={(z):.6f}\n")
        f.write(f"Delta=(|D_obs|-mu_null)={delta:.6f}\n")

    print(f"[NULL] mu={mu:.4f} sigma={sig:.6f} z={z:.2f} Δ={delta:.4f}", flush=True)

    # Build NN mapper
    query_nn, backend = build_nearest_neighbor_index(U.astype(np.float64))
    print(f"[ROT] Nearest-neighbor backend: {backend}", flush=True)
    print(f"[ROT] Running N_ROT={args.n_rot} random rotations (mass field rotated relative to fixed footprint) ...", flush=True)

    rows = []
    for r in range(args.n_rot):
        w_new, _ = rotate_massfield_to_mask(U.astype(np.float64), w, rng, query_nn)
        v_rot = (Ut @ w_new) / sw
        D_rot = float(np.linalg.norm(v_rot))
        z_rot = (D_rot - mu) / sig if sig > 0 else float("nan")
        rows.append((r, D_rot, z_rot))

        if (r + 1) % max(1, args.n_rot // 10) == 0:
            print(f"  Rotation {r+1}/{args.n_rot} : |D_rot|={D_rot:.4f} z_rot={z_rot:.2f}", flush=True)

    rot_df = pd.DataFrame(rows, columns=["rot_id", "D_rot", "z_rot"])
    rot_df.to_csv(os.path.join(out, "rotations.csv"), index=False)
    print(f"[OK] Saved rotations: {os.path.join(out, 'rotations.csv')}", flush=True)

    # Optional figure
    if args.make_fig:
        fig_path = os.path.join(out, "fig_rotatefield.png")

        plt.figure()
        plt.hist(rot_df["z_rot"].to_numpy(), bins=30)
        plt.axvline(z, linewidth=2)
        plt.xlabel(r"$z_{\rm rot}=(|D_{\rm rot}|-\mu_{\rm null})/\sigma_{\rm null}$")
        plt.ylabel("count")
        plt.title("Rotation control: mass field rotated relative to footprint")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()
        print(f"[OK] Saved figure: {fig_path}", flush=True)


if __name__ == "__main__":
    main()
