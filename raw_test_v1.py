import numpy as np
import illustris_python as il

basePath = "data/TNG300"
snap = 99

halos = il.groupcat.loadSubhalos(
    basePath,
    snap,
    fields=['SubhaloMass','SubhaloPos']
)

pos = halos['SubhaloPos']

# center the box
center = pos.mean(axis=0)
vec = pos - center

# normalize to unit vectors
norm = np.linalg.norm(vec, axis=1)
vec = vec[norm > 0]
norm = norm[norm > 0]
unit = vec / norm[:,None]

# average direction
dipole_vector = unit.mean(axis=0)
dipole_amp = np.linalg.norm(dipole_vector)

print("Raw structural dipole amplitude:", dipole_amp)
print("Dipole vector:", dipole_vector)
