import numpy as np
import matplotlib.pyplot as plt

# Time grid
t = np.linspace(0, 10, 2000)
dt = t[1] - t[0]

# Memory timescale
tau_m = 1.5

def kernel(dt):
    return (1/tau_m) * np.exp(-dt/tau_m)

# Structural histories
def sigma(t, tc, w=0.5):
    return 1/(1 + np.exp(-(t-tc)/w))

sigma_A = sigma(t, tc=4.0)
sigma_B = sigma(t, tc=7.0)

# Convolution (causal)
def convolve(sig):
    deltaH = np.zeros_like(sig)
    for i in range(len(t)):
        dtau = t[i] - t[:i+1]
        deltaH[i] = np.sum(kernel(dtau) * sig[:i+1]) * dt
    return deltaH

deltaH_A = convolve(sigma_A)
deltaH_B = convolve(sigma_B)

# Effective expansion
alpha = 0.05
H0 = np.ones_like(t)

H_A = H0 - alpha * deltaH_A
H_B = H0 - alpha * deltaH_B

dipole = (H_A - H_B) / ((H_A + H_B)/2)

# ---- Plotting ----
fig, axs = plt.subplots(1,3, figsize=(15,4))

axs[0].plot(t, sigma_A, label='Region A')
axs[0].plot(t, sigma_B, linestyle='--', label='Region B')
axs[0].set_title("Uneven Structural Commitment")
axs[0].legend()

axs[1].plot(t, deltaH_A)
axs[1].plot(t, deltaH_B, linestyle='--')
axs[1].set_title("Memory Response")

axs[2].plot(t, dipole)
axs[2].set_title("Emergent Late-Time Dipole")

plt.tight_layout()
plt.savefig("results/figures/toy_dipole.png", dpi=300)
plt.show()

taus = np.linspace(0.5, 4.0, 15)
max_dipoles = []

for tau in taus:
    tau_m = tau
    
    def kernel_local(dt):
        return (1/tau_m) * np.exp(-dt/tau_m)
    
    def convolve_local(sig):
        deltaH = np.zeros_like(sig)
        for i in range(len(t)):
            dtau = t[i] - t[:i+1]
            deltaH[i] = np.sum(kernel_local(dtau) * sig[:i+1]) * dt
        return deltaH
    
    deltaH_A = convolve_local(sigma_A)
    deltaH_B = convolve_local(sigma_B)
    
    H_A = H0 - alpha * deltaH_A
    H_B = H0 - alpha * deltaH_B
    
    dip = (H_A - H_B) / ((H_A + H_B)/2)
    max_dipoles.append(np.max(np.abs(dip)))

plt.figure()
plt.plot(taus, max_dipoles)
plt.xlabel("Memory Timescale τ_m")
plt.ylabel("Max Dipole Amplitude")
plt.title("Dipole Strength vs Memory Horizon")
plt.savefig("results/figures/dipole_vs_memory.png", dpi=300)
plt.show()

# -----------------------------
# Power-law kernel experiment
# -----------------------------

def power_kernel(dt, beta=2.0, t0=0.1):
    return (beta-1)*(t0**(beta-1)) / ((dt + t0)**beta)

def convolve_power(sig, beta):
    deltaH = np.zeros_like(sig)
    for i in range(len(t)):
        dtau = t[i] - t[:i+1]
        deltaH[i] = np.sum(power_kernel(dtau, beta=beta) * sig[:i+1]) * dt
    return deltaH

betas = [1.5, 2.0, 3.0]
plt.figure(figsize=(6,4))

for b in betas:
    deltaH_A = convolve_power(sigma_A, b)
    deltaH_B = convolve_power(sigma_B, b)
    
    H_A = H0 - alpha * deltaH_A
    H_B = H0 - alpha * deltaH_B
    
    dip = (H_A - H_B) / ((H_A + H_B)/2)
    plt.plot(t, dip, label=f"β = {b}")

plt.title("Dipole Evolution: Power-Law Memory")
plt.xlabel("Time")
plt.ylabel("Dipole")
plt.legend()
plt.tight_layout()
plt.savefig("results/figures/dipole_powerlaw.png", dpi=300)
plt.show()

# --------------------------------
# Max Dipole vs Beta
# --------------------------------

betas = np.linspace(1.2, 4.0, 20)
max_dipoles = []

for b in betas:
    deltaH_A = convolve_power(sigma_A, b)
    deltaH_B = convolve_power(sigma_B, b)
    
    H_A = H0 - alpha * deltaH_A
    H_B = H0 - alpha * deltaH_B
    
    dip = (H_A - H_B) / ((H_A + H_B)/2)
    max_dipoles.append(np.max(np.abs(dip)))

plt.figure(figsize=(6,4))
plt.plot(betas, max_dipoles)
plt.xlabel(r"Power-law exponent $\beta$")
plt.ylabel("Maximum Dipole Amplitude")
plt.title("Dipole Strength vs Memory Decay Rate")
plt.tight_layout()
plt.savefig("results/figures/max_dipole_vs_beta.png", dpi=300)
plt.show()
