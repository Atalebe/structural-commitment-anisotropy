import h5py

f = h5py.File("data/TNG300/groups_099/fof_subhalo_tab_099.0.hdf5",'r')

print(f['Header'].attrs['NumFiles'])
