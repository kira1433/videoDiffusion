import h5py

# Open the HDF5 file
p=""
with h5py.File(
    "/mnt/MIG_Store/Datasets/faceforensicspp/Originalface.hdf5", "r"
) as hdf5_file:
    def print_keys(name):
        if name.endswith(".jpg"):
            print(name.split("/")[-2])

    hdf5_file.visit(print_keys)


"/mnt/MIG_Store/Datasets/faceforensicspp/Originalface.hdf5"