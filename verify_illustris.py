import illustris_python as il

basePath = "data/TNG300"
snap = 99


halos = il.groupcat.loadSubhalos(
    basePath,
    snap,
    fields=['SubhaloMass','SubhaloPos']
)

print(len(halos['SubhaloMass']))

print(halos['SubhaloMass'].dtype)
print(halos['SubhaloPos'].dtype)

print(halos['SubhaloMass'].nbytes/1e9, "GB")
print(halos['SubhaloPos'].nbytes/1e9, "GB")
