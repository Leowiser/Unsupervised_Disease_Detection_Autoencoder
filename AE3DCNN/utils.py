import numpy as np
import h5py
from PIL import Image

def LoadHSI(path_to_hdf5, return_wlens=False, print_info=False):
    import h5py
    import numpy as np

    filetype = path_to_hdf5.split('.')[-1].lower()

    with h5py.File(path_to_hdf5, 'r') as f:
        if print_info:
            print("Datasets in the file:")
            for name in f.keys():
                print(name)
                data = f[name]
                print("Attributes of the dataset:")
                for key in data.attrs.keys():
                    print(f"{key}: {data.attrs[key]}")

        # Smart dataset name resolution for .h5 if the dataset is not called Hypercube than fallback to the first dataset
        if filetype == 'h5':
            hypercube_dataset = 'Hypercube' if 'Hypercube' in f else list(f.keys())[0]
        else:
            hypercube_dataset = 'hypercube' if 'hypercube' in f else list(f.keys())[0]

        dataset = f[hypercube_dataset]
        hcube = dataset[:]

        if return_wlens:
            wlens = np.array(dataset.attrs.get('Wavelengths', dataset.attrs.get('wavelength_nm', [])))
            return hcube, wlens
        else:
            return hcube

def read_mask(file_path):
    mask = np.array(Image.open(file_path), dtype=np.float32)  
    if len(mask.shape)>2 and mask.shape[2] == 4:
        # normalize the values, ensure they are integers for segmentation labels, and then restrict range
        mask = mask[:, :, 0]
        mask = ((mask / 255.0) * 2).round().astype(int)
    
    return mask