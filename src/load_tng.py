import illustris_python as il

def load_subhalos(basePath, snapNum=99, mass_cut=1e12):

    halos = il.groupcat.loadSubhalos(basePath, snapNum,
                                     fields=['SubhaloMass','SubhaloPos'])

    masses = halos['SubhaloMass'] * 1e10  # Msun/h
    coords = halos['SubhaloPos']

    mask = masses > mass_cut
    
    return coords[mask], masses[mask]
