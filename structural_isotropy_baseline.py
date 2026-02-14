#!/usr/bin/env python
"""
structural_isotropy_baseline.py

Compute the null distribution of the structural dipole amplitude in TNG300-1
using random hemisphere splits at z=0 (snapshot 99), then save and plot.

Uses the illustris_python groupcat loader, assuming:
    BASE_PATH = "data/TNG300"
and that
    data/TNG300/groups_099 -> /mnt/g/TNG300-1/groupcat_99
exists as a symlink.
"""

import numpy as np
import matplotlib.pyplot as plt
import illustris_python as il

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
BASE_PATH = "data/TNG300"   # uses the symlinked groups_099
SNAP = 99                   # z ~ 0

BOX_SIZE = 205000.0         # kpc/h, TNG300 box size
N_RANDOM = 2000             # number of random orientations

NPY_OUT = "structural_dipole_null_distribution.npy"
FIG_OUT = "fig_structural_dipole_null.png"


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def wrap_positions(pos, box_size=BOX_SIZE):
    """Shift to box centre and apply periodic wrapping."""
    center = np.array([box_size / 2.0] * 3, dtype=np.float32)
    x = pos - center
    x = (x + box_size / 2.0) % box_size - box_size / 2.0
    return x


def unit_vectors_from_positions(pos_wrapped):
    """Convert wrapped positions to unit vectors."""
    r = np.linalg.norm(pos_wrapped, axis=1)
    r[r == 0.0] = 1.0
    return (pos_wrapped.T / r).T  # shape (N, 3)


def hemisphere_dipole(mass, hat_pos, n_hat):
    """
    Mass-weighted hemisphere dipole along direction n_hat.

    D = (M_plus - M_minus) / (M_plus + M_minus)
    with plus/minus defined by sign of hat_pos · n_hat.
    """
    cos_theta = hat_pos @ n_hat
    mask_plus = cos_theta >= 0.0
    mask_minus = ~mask_plus

    M_plus = mass[mask_plus].sum()
    M_minus = mass[mask_minus].sum()

    if M_plus + M_minus == 0.0:
        return 0.0

    D = (M_plus - M_minus) / (M_plus + M_minus)
    return float(abs(D))


def random_unit_vectors(n):
    """Draw n random unit vectors uniformly on the sphere."""
    phi = 2.0 * np.pi * np.random.rand(n)
    cos_theta = 2.0 * np.random.rand(n) - 1.0
    sin_theta = np.sqrt(1.0 - cos_theta**2)

    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)
    z = cos_theta
    return np.vstack([x, y, z]).T  # (n, 3)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)

    print(f"BASE_PATH = {BASE_PATH}")
    print(f"Snapshot  = {SNAP}")

    # Load subhalos via illustris_python
    halos = il.groupcat.loadSubhalos(
        BASE_PATH,
        SNAP,
        fields=["SubhaloMass", "SubhaloPos"],
    )

    pos = halos["SubhaloPos"].astype(np.float32)
    mass = halos["SubhaloMass"].astype(np.float32)

    print(f"Total subhalos loaded: {pos.shape[0]}")

    # Re-centre and wrap
    pos_wrapped = wrap_positions(pos)
    hat_pos = unit_vectors_from_positions(pos_wrapped)

    print(f"\nRunning isotropy null test with {N_RANDOM} random orientations...\n")

    dirs = random_unit_vectors(N_RANDOM)
    amps = np.empty(N_RANDOM, dtype=np.float32)

    for i, nhat in enumerate(dirs):
        amps[i] = hemisphere_dipole(mass, hat_pos, nhat)
        if (i + 1) % 200 == 0:
            print(f"{i+1}/{N_RANDOM} orientations processed")

    # Stats
    mu = float(amps.mean())
    sigma = float(amps.std())
    amin = float(amps.min())
    amax = float(amps.max())

    print("\n===== STRUCTURAL ISOTROPY BASELINE =====\n")
    print(f"Mean dipole : {mu:.6f}")
    print(f"Std        : {sigma:.6f}")
    print(f"Max        : {amax:.6f}")
    print(f"Min        : {amin:.6f}")

    # Save raw distribution
    np.save(NPY_OUT, amps)
    print(f"\nSaved null distribution to: {NPY_OUT}")

    # Plot histogram
    plt.figure(figsize=(6, 4))
    plt.hist(amps, bins=40, density=True, alpha=0.7)
    plt.axvline(mu, linestyle="--", label=f"mean = {mu:.3f}")
    plt.axvline(mu + sigma, linestyle=":", label=r"mean $\pm 1\sigma$")
    plt.axvline(mu - sigma, linestyle=":")

    plt.xlabel("Structural dipole amplitude")
    plt.ylabel("Probability density")
    plt.title("Null distribution of structural dipole (random hemispheres, z=0)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_OUT, dpi=200)
    plt.close()

    print(f"Saved histogram figure: {FIG_OUT}")
    print("\nDone.\n")
