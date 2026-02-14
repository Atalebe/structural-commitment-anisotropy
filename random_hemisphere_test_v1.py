import numpy as np
import illustris_python as il

basePath = "data/TNG300"
snap = 99

halos = il.groupcat.loadSubhalos(
    basePath,
    snap,
    fields=['SubhaloPos','SubhaloMass']
)

pos = halos['SubhaloPos']
center = pos.mean(axis=0)
vec = pos - center

norm = np.linalg.norm(vec, axis=1)
vec = vec[norm > 0]
unit = vec / norm[:,None]

Ntests = 200
dipoles = []

for _ in range(200):

    axis = np.random.normal(size=3)
    axis /= np.linalg.norm(axis)

    proj = unit @ axis
    dipoles.append(np.abs(proj.mean()))


print("Mean dipole:", dipoles.mean())
print("Std:", dipoles.std())
print("Max:", dipoles.max())
