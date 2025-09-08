import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.ndimage import binary_erosion, binary_dilation
from PIL import Image
from torchvision.transforms import v2
from utils import *
from preprocessing import *


# Custom dataset class for RGB images
class RgbDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, apply_mask=False, mask_method=1):
        super().__init__()
        self.img_dir = img_dir
        self.transform = transform
        self.apply_mask = apply_mask
        self.mask_method = mask_method
        self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
        self.mask_paths = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir)]
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        image_path = self.img_paths[idx]
        image = Image.open(image_path)
        if self.apply_mask == True:
            # Open mask
            mask_path = self.mask_paths[idx]
            mask = read_mask(mask_path)
            
            # Apply mask to image
            image = Image.fromarray(np.array(image)*np.expand_dims(np.where(mask==2, self.mask_method, mask), axis=-1).astype('uint8'))
        
        # Convert to tensor. Tensors are needed already in the custom collate_fn and at the transforms (specified in self.transform).
        image = transforms.functional.to_tensor(image)
        
        if self.transform:
            image = self.transform(image)
            
        return image
    
    
    

# Custom dataset class for HSI data
class HsiDataset(Dataset):
    def __init__(self, img_paths, mask_dir, transform=None, apply_mask=False, mask_method=1,
                 min_wavelength=0, normalize=True, selected_bands=None, pca=None, scaler=None, polyorder=2, window_length=11, deriv = 2,  preprocess_method = "normalization", patch_probability=0.8):
        super().__init__()
        self.img_paths = img_paths
        self.mask_dir = mask_dir
        self.transform = transform
        self.apply_mask = apply_mask
        self.mask_method = mask_method
        self.min_wavelength = min_wavelength
        self.normalize = normalize
        self.selected_bands = selected_bands
        self.pca = pca
        self.scaler = scaler
        self.polyorder = polyorder
        self.window_length = window_length
        self.deriv = deriv
        self.preprocess_method = preprocess_method
        self.patch_probability = patch_probability
        
        # Load wavelengths from the first image (but could be any image since images have the same spectral bands)
        _, self.wlens = LoadHSI(self.img_paths[0], return_wlens=True)
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        image_path = self.img_paths[idx]
        hsi_np = LoadHSI(image_path, return_wlens=False)
        
        if self.apply_mask:
            filename = os.path.basename(image_path)
            if filename.endswith(".hdf5"):
                mask_name = filename.replace(".hdf5", ".png")
            elif filename.endswith(".h5"):
                mask_name = filename.replace(".h5", ".png")
            else:
                raise ValueError(f"Unexpected HDF5 file extension in {filename}")

            mask_path = os.path.join(self.mask_dir, mask_name)
            mask = read_mask(mask_path)
            hsi_np = hsi_np * np.where(mask == 2, self.mask_method, mask)
        
        if self.preprocess_method == "snv":
            hsi_np, _ = preprocessing_snv(hsi_np, self.wlens, min_wavelength=self.min_wavelength, selected_bands=self.selected_bands)
        elif self.preprocess_method == "savgol":
            hsi_np, _ = preprocessing_savgol(hsi_np, self.wlens, window_length=self.window_length, polyorder=self.polyorder, deriv=self.deriv,
                                            min_wavelength=self.min_wavelength, selected_bands=self.selected_bands)
        elif self.preprocess_method == "stacked":
            hsi_np, _ = stack_original_and_derivatives_as_channels(
                hsi_np,
                self.wlens,
                window_length=self.window_length,
                polyorder=self.polyorder,
                min_wavelength=self.min_wavelength,
                selected_bands=self.selected_bands,
                target_size=(128, 128)  # Resize before stacking
            )
            # NEW: Check for NaNs or Infs early
            if np.isnan(hsi_np).any() or np.isinf(hsi_np).any():
                print(f"[Dataset] Found NaN or Inf in sample {idx} after preprocessing!")
                np.set_printoptions(precision=3, suppress=True)
                print("Wavelengths:", self.wlens)
                print("Stacked sample stats - min:", np.nanmin(hsi_np), "max:", np.nanmax(hsi_np))
        else:
            # Apply preprocessing (wavelength filtering, negative-value correction, optional normalization, optional band selection)
            hsi_np, _ = preprocess(hsi_np, self.wlens,
                                min_wavelength=self.min_wavelength, normalize=self.normalize, selected_bands=self.selected_bands)
            
        # Transform data into PCA space with pre-fitted PCA model and scaler - if required!
        if self.pca is not None and self.scaler is not None:
            hsi_np = hsi_transform_to_pca_space(hsi_np, self.pca, self.scaler)
        
        # At this point hsi_np is a numpy.ndarray with shape (CxHxW).
        # However it will need to be changed to torch.Tensor (with transforms.functional.to_tensor or transforms.ToTensor).
                
        # --- Tensor conversion ---
        if self.preprocess_method == "stacked":
            # shape: (3, D, H, W)
            hsi_tensor = torch.from_numpy(hsi_np).float()
        else:
            # shape: (C, H, W) → (H, W, C)
            hsi_np = hsi_np.transpose((1, 2, 0))
            hsi_tensor = transforms.functional.to_tensor(hsi_np)
    
        # Randomly choose to return a patch or the full image
        patch_prob = self.patch_probability # or make it a class attribute
        if self.preprocess_method == "stacked":
            _, _, H, W = hsi_tensor.shape  # Correct unpacking for stacked
        else:
            _, H, W = hsi_tensor.shape  # Regular unpacking for non-stacked
        if np.random.rand() < patch_prob and H >= 128 and W >= 128:
            top = np.random.randint(0, H - 128 + 1)
            left = np.random.randint(0, W - 128 + 1)
            hsi_tensor = hsi_tensor[:, top:top+128, left:left+128]
        else:
            # Optionally center crop to 256x256 to remove border inconsistencies
            hsi_tensor = hsi_tensor
        
        # Now apply transforms
        if self.transform:
            hsi_tensor = self.transform(hsi_tensor)

        return hsi_tensor, idx
    
    def get_edge_mask(self, idx):
        image_path = self.img_paths[idx]
        mask_path = os.path.join(self.mask_dir, os.path.basename(image_path).replace('.hdf5', '.png'))
        mask = read_mask(mask_path)
        mask = Image.fromarray(mask)
        mask = mask.resize((256, 256), resample=Image.NEAREST)  # Use NEAREST to avoid interpolation artifacts
        mask = np.array(mask)
        edge_mask = extract_edge(mask, edge_width=4).astype(np.float32)
        edge_mask = torch.from_numpy(edge_mask).unsqueeze(0)  # Add channel dim for transforms

        # Apply the same transform as in test_transforms
        if self.transform:
            # Only apply geometric transforms
            # You may want to manually extract those from test_transforms if needed
            for t in self.transform.transforms:
                if isinstance(t, (v2.Resize, v2.RandomRotation, v2.RandomHorizontalFlip)):
                    edge_mask = t(edge_mask)

        return edge_mask.squeeze(0)  # (H, W
            
    
    

def padded_collate(batch):
    # batch: list of (tensor, index)
    tensors, indices = zip(*batch)

    # Get max dimensions
    max_depth = max(t.shape[1] for t in tensors)
    max_height = max(t.shape[2] for t in tensors)
    max_width  = max(t.shape[3] for t in tensors)

    padded_batch = []
    for t in tensors:
        padding = (
            0, max_width - t.shape[3],  # W
            0, max_height - t.shape[2],  # H
            0, max_depth - t.shape[1]   # D
        )
        padded = F.pad(t, padding, mode='constant', value=0)
        padded_batch.append(padded)

    return torch.stack(padded_batch), torch.tensor(indices)

def extract_full_edge(mask, edge_width=10):
    mask = mask.astype(bool)
    
    # Inward edge
    eroded = mask
    for _ in range(edge_width):
        eroded = binary_erosion(eroded)
    inner_edge = np.logical_xor(mask, eroded)
    
    # Outward edge
    dilated = mask
    for _ in range(edge_width):
        dilated = binary_dilation(dilated)
    outer_edge = np.logical_xor(dilated, mask)
    
    # Combine both
    full_edge = np.logical_or(inner_edge, outer_edge)
    
    return full_edge.astype(np.uint8)

