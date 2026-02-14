import numpy as np
from src.load_tng import load_subhalos

coords, masses = load_subhalos("data/TNG300")


# Observer at box center
observer = np.array([75000,75000,75000])  # adjust after reading BoxSize

vecs = coords - observer
norms = np.linalg.norm(vecs, axis=1)
unit = vecs / norms[:,None]

def dipole_for_axis(axis):
    axis = axis/np.linalg.norm(axis)
    dot = np.dot(unit, axis)

    f = masses[dot>0].sum()
    b = masses[dot<0].sum()

    return (f-b)/(0.5*(f+b))

# scan axes
dipoles = []

for _ in range(300):
    axis = np.random.randn(3)
    dipoles.append(dipole_for_axis(axis))

dipoles = np.array(dipoles)

print("Mean |dipole|:", np.mean(np.abs(dipoles)))
print("Max |dipole|:", np.max(np.abs(dipoles)))

import matplotlib.pyplot as plt

plt.hist(np.abs(dipoles), bins=30)
plt.xlabel("|Dipole|")
plt.ylabel("Count")
plt.title("Structural Dipole Distribution")
plt.savefig("results/figures/structure_dipole_hist.png", dpi=300)
plt.show()
