#!/usr/bin/env python
"""
Radial shell decomposition of the expansion dipole for TNG50-1.

We fit, in each radial shell and snapshot,
  y_i = v_rad / (H_FRW * r_i) - 1 = a + b · n_hat_i
where n_hat_i is the unit position vector.

Outputs:
  tng50_expansion_shell_dipole_results.npz
    - snapshots       (Nsnap,)
    - redshifts       (Nsnap,)
    - radial_bins     (Nshell, 2)
    - counts          (Nsnap, Nshell)
    - mono            (Nsnap, Nshell)         # a
    - mono_over_H     (Nsnap, Nshell)         # same as a
    - dip_amp         (Nsnap, Nshell)         # |b|
    - dip_amp_over_H  (Nsnap, Nshell)         # |b|
    - dip_par         (Nsnap, Nshell)         # b · d_hat
    - dip_par_over_H  (Nsnap, Nshell)         # b · d_hat
    - dip_perp        (Nsnap, Nshell)         # sqrt(|b|^2 - b_par^2)
    - dip_perp_over_H (Nsnap, Nshell)
"""

import os
import glob
import numpy as np
import h5py

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

BASE_PATH = "/mnt/g/TNG50-1"
BOX_SIZE = 35.0  # Mpc/h
SNAPSHOTS = [40, 50, 59, 67, 72, 78, 91]

SNAP_TO_Z = {
    40: 1.5,
    50: 1.0,
    59: 0.8,
    67: 0.6,
    72: 0.5,
    78: 0.4,
    91: 0.2,
}

MASS_CUT = 1.0  # 1e10 Msun/h

# Radial shells in Mpc/h (for a 35 Mpc/h box)
RADIAL_BINS = np.array([
    [5.0, 15.0],
    [15.0, 25.0],
    [25.0, 30.0],
], dtype=float)


# ----------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------

def find_groupcat_folder(snapshot: int) -> str:
    candidates = [
        os.path.join(BASE_PATH, f"groupcat_{snapshot:03d}"),
        os.path.join(BASE_PATH, f"groups_{snapshot:03d}"),
    ]
    for folder in candidates:
        if os.path.isdir(folder):
            return folder
    raise RuntimeError(
        f"No groupcat folder found for snapshot {snapshot}. Tried:\n"
        + "\n".join(candidates)
    )


def load_subhalos(snapshot: int, mass_cut: float | None = None):
    """
    Load subhalo positions (Mpc/h), velocities (km/s) and masses.

    Same robust loader as in the other TNG50 scripts.
    """
    folder = find_groupcat_folder(snapshot)
    print(f"  Using folder: {folder}")

    pattern = os.path.join(folder, f"fof_subhalo_tab_{snapshot:03d}.*.hdf5")
    files = sorted(glob.glob(pattern))
    if not files:
        raise RuntimeError(f"No groupcat files found matching {pattern}")

    pos_list = []
    vel_list = []
    mass_list = []
    used_files = 0
    skipped_files = 0

    for path in files:
        with h5py.File(path, "r") as f:
            if "Subhalo" not in f:
                skipped_files += 1
                print(f"  [WARN] No 'Subhalo' group in {os.path.basename(path)}, skipping.")
                continue
            sub = f["Subhalo"]

            if "SubhaloPos" in sub:
                pos_field = "SubhaloPos"
            elif "SubhaloCM" in sub:
                pos_field = "SubhaloCM"
            else:
                skipped_files += 1
                print(
                    f"  [WARN] No SubhaloPos or SubhaloCM in "
                    f"{os.path.basename(path)}, skipping."
                )
                continue

            if "SubhaloVel" not in sub or "SubhaloMass" not in sub:
                skipped_files += 1
                print(
                    f"  [WARN] Missing SubhaloVel or SubhaloMass in "
                    f"{os.path.basename(path)}, skipping."
                )
                continue

            pos_ckpc = sub[pos_field][:]
            vel = sub["SubhaloVel"][:]
            mass = sub["SubhaloMass"][:]

            if mass_cut is not None:
                mask = mass >= mass_cut
                if not np.any(mask):
                    skipped_files += 1
                    print(
                        f"  [WARN] All subhalos below mass_cut in "
                        f"{os.path.basename(path)}, skipping."
                    )
                    continue
                pos_ckpc = pos_ckpc[mask]
                vel = vel[mask]
                mass = mass[mask]

            if pos_ckpc.size == 0:
                skipped_files += 1
                print(
                    f"  [WARN] No subhalos left after cuts in "
                    f"{os.path.basename(path)}, skipping."
                )
                continue

            pos_list.append(pos_ckpc)
            vel_list.append(vel)
            mass_list.append(mass)
            used_files += 1

    if not pos_list:
        raise RuntimeError(f"No usable subhalo data for snapshot {snapshot}")

    pos_ckpc = np.concatenate(pos_list, axis=0)
    vel = np.concatenate(vel_list, axis=0)
    mass = np.concatenate(mass_list, axis=0)

    pos_mpc = pos_ckpc.astype(np.float64) / 1000.0
    vel = vel.astype(np.float64)
    mass = mass.astype(np.float64)

    print(f"  Files used: {used_files}, skipped: {skipped_files}")
    print(f"  Total halos used (after mass cut): {pos_mpc.shape[0]}")

    return pos_mpc, vel, mass


# ----------------------------------------------------------------------
# Cosmology
# ----------------------------------------------------------------------

def H_comoving(z: float) -> float:
    H0 = 67.74
    Omega_m = 0.3089
    Omega_L = 1.0 - Omega_m
    return H0 * np.sqrt(Omega_m * (1.0 + z) ** 3 + Omega_L)


# ----------------------------------------------------------------------
# Shell dipole fit
# ----------------------------------------------------------------------

def main():
    print(f"BASE_PATH   = {BASE_PATH}")
    print(f"BOX_SIZE    = {BOX_SIZE} Mpc/h")
    print(f"SNAPSHOTS   = {SNAPSHOTS}")
    print(f"MASS_CUT    = {MASS_CUT} (1e10 Msun/h)")
    print(f"RADIAL_BINS = {RADIAL_BINS.tolist()}\n")

    struct_vecs = np.load("tng50_dipole_vectors.npy")
    if struct_vecs.shape[0] != len(SNAPSHOTS):
        raise RuntimeError(
            f"tng50_dipole_vectors.npy has {struct_vecs.shape[0]} entries, "
            f"but SNAPSHOTS has {len(SNAPSHOTS)}."
        )

    bulk_vecs = np.load("tng50_bulk_flow_vectors.npy")
    if bulk_vecs.shape[0] != len(SNAPSHOTS):
        raise RuntimeError(
            f"tng50_bulk_flow_vectors.npy has {bulk_vecs.shape[0]} entries, "
            f"but SNAPSHOTS has {len(SNAPSHOTS)}."
        )

    n_snap = len(SNAPSHOTS)
    n_shell = RADIAL_BINS.shape[0]

    redshifts = np.array([SNAP_TO_Z[s] for s in SNAPSHOTS], dtype=float)

    counts = np.zeros((n_snap, n_shell), dtype=int)
    mono = np.zeros((n_snap, n_shell))
    mono_over_H = np.zeros((n_snap, n_shell))
    dip_amp = np.zeros((n_snap, n_shell))
    dip_amp_over_H = np.zeros((n_snap, n_shell))
    dip_par = np.zeros((n_snap, n_shell))
    dip_par_over_H = np.zeros((n_snap, n_shell))
    dip_perp = np.zeros((n_snap, n_shell))
    dip_perp_over_H = np.zeros((n_snap, n_shell))

    print("===== SHELL EXPANSION DIPOLE FIT (TNG50) =====\n")
    print("Found tng50_bulk_flow_vectors.npy, will subtract bulk flow.\n")

    for i, snap in enumerate(SNAPSHOTS):
        z = redshifts[i]
        H_FRW = H_comoving(z)

        print(f"Processing snapshot {snap} (z ~ {z:.2f})")
        pos, vel, mass = load_subhalos(snap, mass_cut=MASS_CUT)

        center = np.array([BOX_SIZE / 2.0] * 3, dtype=np.float64)
        r_vec = pos - center[None, :]
        r = np.linalg.norm(r_vec, axis=1)
        mask_r = r > 0.0
        r_vec = r_vec[mask_r]
        vel = vel[mask_r]
        r = r[mask_r]

        r_hat = r_vec / r[:, None]

        # subtract bulk flow
        v_bulk = bulk_vecs[i, :]
        vel_corr = vel - v_bulk[None, :]

        v_rad = np.sum(vel_corr * r_hat, axis=1)

        d_vec = struct_vecs[i, :]
        d_hat = d_vec / np.linalg.norm(d_vec)

        for j, (r_min, r_max) in enumerate(RADIAL_BINS):
            shell_mask = (r >= r_min) & (r < r_max)
            n_shell_halos = int(np.count_nonzero(shell_mask))
            counts[i, j] = n_shell_halos

            if n_shell_halos < 100:
                # too few halos to fit robustly; leave zeros
                print(
                    f"  Shell {j}: r in [{r_min:.1f}, {r_max:.1f}] Mpc/h, "
                    f"N = {n_shell_halos} (too few, skipping fit)"
                )
                continue

            r_s = r[shell_mask]
            r_hat_s = r_hat[shell_mask, :]
            v_rad_s = v_rad[shell_mask]

            # dimensionless quantity to fit: y = v_rad/(H_FRW r) - 1
            y = v_rad_s / (H_FRW * r_s) - 1.0

            X = np.column_stack([
                np.ones_like(r_s),
                r_hat_s[:, 0],
                r_hat_s[:, 1],
                r_hat_s[:, 2],
            ])

            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            a = beta[0]
            b_vec = beta[1:]

            mono[i, j] = a
            mono_over_H[i, j] = a  # already dimensionless

            b_amp = float(np.linalg.norm(b_vec))
            dip_amp[i, j] = b_amp
            dip_amp_over_H[i, j] = b_amp

            b_par = float(np.dot(b_vec, d_hat))
            b_perp_sq = max(b_amp ** 2 - b_par ** 2, 0.0)
            b_perp = float(np.sqrt(b_perp_sq))

            dip_par[i, j] = b_par
            dip_par_over_H[i, j] = b_par
            dip_perp[i, j] = b_perp
            dip_perp_over_H[i, j] = b_perp

            print(
                f"  Shell {j}: r in [{r_min:.1f}, {r_max:.1f}] Mpc/h, "
                f"N = {n_shell_halos}"
            )
            print(f"    Monopole shift a/H = {a:+.3e}")
            print(f"    |b|/H             = {b_amp:+.3e}")
            print(f"    b_parallel/H      = {b_par:+.3e}")
            print(f"    b_perp/H          = {b_perp:+.3e}")

        print("")

    np.savez(
        "tng50_expansion_shell_dipole_results.npz",
        snapshots=np.array(SNAPSHOTS, dtype=int),
        redshifts=redshifts,
        radial_bins=RADIAL_BINS,
        counts=counts,
        mono=mono,
        mono_over_H=mono_over_H,
        dip_amp=dip_amp,
        dip_amp_over_H=dip_amp_over_H,
        dip_par=dip_par,
        dip_par_over_H=dip_par_over_H,
        dip_perp=dip_perp,
        dip_perp_over_H=dip_perp_over_H,
    )

    print("\nSaved: tng50_expansion_shell_dipole_results.npz")
    print("Outer shell quick summary:")
    outer_idx = RADIAL_BINS.shape[0] - 1
    r_min, r_max = RADIAL_BINS[outer_idx]
    for i, snap in enumerate(SNAPSHOTS):
        print(
            f"  snap {snap:3d}, z={redshifts[i]:.1f} : "
            f"|b|/H = {dip_amp_over_H[i, outer_idx]:+.3e}, "
            f"b_parallel/H = {dip_par_over_H[i, outer_idx]:+.3e}, "
            f"N_shell = {counts[i, outer_idx]:d}"
        )
    print("\nDone.")


if __name__ == "__main__":
    main()
