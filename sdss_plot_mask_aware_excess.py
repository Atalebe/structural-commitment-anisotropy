#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

INPZ = "sdss_structural_dipole_mask_null_results_normed.npz"
OUTFIG = "sdss_fig_mask_geometry_vs_excess.png"

def main():
    data = np.load(INPZ, allow_pickle=True)

    D_geo = float(data["global_D_geo"])
    D_obs = float(data["global_D_obs"])
    ang   = float(data["global_angle_deg"])

    D_geo_vec = data["global_D_geo_vec"].astype(float)
    D_obs_vec = data["global_D_obs_vec"].astype(float)
    D_phys_vec = D_obs_vec - D_geo_vec
    D_phys = float(np.linalg.norm(D_phys_vec))

    null_amps = data["null_global_amps"].astype(float)
    mu = float(np.mean(null_amps))
    sig = float(np.std(null_amps, ddof=1))

    fig = plt.figure(figsize=(8, 4))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.hist(null_amps, bins=40, alpha=0.8, density=True)
    ax1.axvline(D_geo, linestyle="--", label="D_geo")
    ax1.axvline(D_obs, linestyle="-",  label="D_obs")
    ax1.set_xlabel("|D| (normalized)")
    ax1.set_ylabel("Density")
    ax1.set_title("Mask-preserving shuffled null")
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    bars = ["D_geo", "D_obs", "D_phys"]
    vals = [D_geo, D_obs, D_phys]
    ax2.bar(range(3), vals)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(bars)
    ax2.set_ylabel("Amplitude")
    ax2.set_title(f"Geometry vs excess (angle={ang:.2f} deg)")
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTFIG, dpi=200)

    print("SDSS mask-aware summary:")
    print(f"  D_geo  = {D_geo:.4f}")
    print(f"  D_obs  = {D_obs:.4f}")
    print(f"  D_phys = {D_phys:.4f}")
    print(f"  angle(D_geo,D_obs) = {ang:.2f} deg")
    print(f"  null mean = {mu:.4f}, null sigma = {sig:.4f}")
    print(f"Saved: {OUTFIG}")

if __name__ == "__main__":
    main()
