import h5py
import numpy as np

def load_subhalos(path, mass_cut=1e12):

    with h5py.File(path,'r') as f:
        h = f['Header'].attrs['HubbleParam']
        
        coords = f['Subhalo']['SubhaloPos'][:] / h
        
        # SubhaloMass is in 1e10 Msun/h
        masses = f['Subhalo']['SubhaloMass'][:] * 1e10 / h

    mask = masses > mass_cut
    
    return coords[mask], masses[mask]
