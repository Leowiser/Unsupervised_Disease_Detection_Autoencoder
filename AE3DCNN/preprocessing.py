import os
import torch
import torch.nn.functional as F
from utils import *
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def filter_filenames(folder_path, camera_id, date_stamps, tray_ids):
    """
    Filters filenames in a folder based on camera ID, date stamps, and tray IDs.

    Parameters:
    - folder_path (str): Path to the folder containing the files.
    - camera_id (str): Camera ID.
    - date_stamps (list): List of selected date stamps.
    - tray_ids (list): List of selected tray IDs.

    Returns:
    - list: Filtered list of full file paths that match the given criteria.
    """
    all_files = os.listdir(folder_path)

    filtered_files = [
        os.path.join(folder_path, f) for f in all_files
        if f.startswith(camera_id + "_") and 
           any(date in f for date in date_stamps) and 
           any(f.split("_")[2].startswith(tray) for tray in tray_ids)
    ]
    
    return filtered_files



def preprocess(hsi_np, wlens, min_wavelength=0, normalize=True, selected_bands=None):
    """
    Removes spectral bands with wavelengths below min_wavelength from the hyperspectral image.
    Also replaces negative values with 0 and applies optional normalization.
    Also apply optional band selection, but that is inteded to be used in the dataset class 
    after determining selected bands based on PCA loadings.

    Parameters:
    - hsi_np (nunmpy array): Hyperspectral image with shape (bands, height, width).
    - wlens (numpy array): Wavelength values with shape (bands, ) corresponding to bands in hsi_np
    - min_wavelength (int or float): Minimum wavelength (in nm) to keep in the hyperspectral data
    - normalize (bool): Whether to normalize the hyperspectral image data
    - selected_bands (list or None): List of exact wavelength values to keep. If None, all bands >= min_wavelength are kept.

    Returns:
    - hsi_np_filtered (numpy array): Hyperspectral image with selected spectral bands of shape (filtered bands, height, width)
    - wlens_filtered (numpy array): Updated wavelengths array of shape (filtered bands, )
    """
    # Determine wavelengths to keep
    valid_bands = wlens >= min_wavelength  

    # Filter hsi_np and wlens to keep only relevant bands
    hsi_np_filtered = hsi_np[valid_bands, :, :]
    wlens_filtered = wlens[valid_bands]
    
    # Set all negative values to 0 (these are noise)
    hsi_np_filtered = np.maximum(hsi_np_filtered, 0)
    
    # Normalize the data if required
    if normalize and np.max(hsi_np_filtered) > 0:    # Avoid division by zero
        hsi_np_filtered = hsi_np_filtered / np.max(hsi_np_filtered)
    
    # Perform band selection - will mainly be used in the dataset class for dimensionality reduction
    if selected_bands is not None:
        selected_bands = np.sort(selected_bands)
        # Get indices of the selected bands in wlens_filtered
        selected_band_indices = np.searchsorted(wlens_filtered, selected_bands)
        # Filter hsi_np and wlens to keep only selected bands
        hsi_np_filtered = hsi_np_filtered[selected_band_indices, :, :]
        wlens_filtered = wlens_filtered[selected_band_indices]
        
    return hsi_np_filtered, wlens_filtered

def apply_snv(hyper_image):
    # Input shape: (bands, H, W)
    bands, H, W = hyper_image.shape

    # Reshape to (H*W, bands) — each row is one pixel's spectrum
    pixel_spectra = hyper_image.reshape(bands, -1).T  # shape: (H*W, bands)

    # Apply SNV: subtract mean and divide by std per pixel
    mean = pixel_spectra.mean(axis=1, keepdims=True)
    std = pixel_spectra.std(axis=1, keepdims=True)
    snv_pixel_spectra = (pixel_spectra - mean) / (std + 1e-8)

    # Reshape back to (bands, H, W)
    snv_image = snv_pixel_spectra.T.reshape(bands, H, W)
    return snv_image

def preprocessing_snv(hsi_np, wlens, min_wavelength=0, selected_bands=None):
    """
    Removes spectral bands with wavelengths below min_wavelength from the hyperspectral image.
    Also replaces negative values with 0 and applies optional normalization.
    Also apply optional band selection, but that is inteded to be used in the dataset class 
    after determining selected bands based on PCA loadings.

    Parameters:
    - hsi_np (nunmpy array): Hyperspectral image with shape (bands, height, width).
    - wlens (numpy array): Wavelength values with shape (bands, ) corresponding to bands in hsi_np
    - min_wavelength (int or float): Minimum wavelength (in nm) to keep in the hyperspectral data
    - normalize (bool): Whether to normalize the hyperspectral image data
    - selected_bands (list or None): List of exact wavelength values to keep. If None, all bands >= min_wavelength are kept.

    Returns:
    - hsi_np_filtered (numpy array): Hyperspectral image with selected spectral bands of shape (filtered bands, height, width)
    - wlens_filtered (numpy array): Updated wavelengths array of shape (filtered bands, )
    """
    # Determine wavelengths to keep
    valid_bands = wlens >= min_wavelength  
    # Filter hsi_np and wlens to keep only relevant bands
    hsi_np_filtered = hsi_np[valid_bands, :, :]
    wlens_filtered = wlens[valid_bands]
    # Set all negative values to 0 (these are noise)
    hsi_np_filtered = np.maximum(hsi_np_filtered, 0)

    hsi_np_snv = apply_snv(hsi_np_filtered)

    # Perform band selection - will mainly be used in the dataset class for dimensionality reduction
    if selected_bands is not None:
        selected_bands = np.sort(selected_bands)
        # Get indices of the selected bands in wlens_filtered
        selected_band_indices = np.searchsorted(wlens_filtered, selected_bands)
        # Filter hsi_np and wlens to keep only selected bands
        hsi_np_snv = hsi_np_snv[selected_band_indices, :, :]
        wlens_filtered = wlens_filtered[selected_band_indices]

    return hsi_np_snv, wlens_filtered

def savgol_filter_hypercube(hyper_image, window_length=11, polyorder=2, deriv = 2):
    """
    Apply Savitzky-Golay filter along the spectral axis (per pixel).

    Args:
        hyper_image: numpy array of shape (bands, H, W)
        window_length: Window size for the filter (must be odd)
        polyorder: Polynomial order for the filter

    Returns:
        Smoothed hyper_image of same shape
    """
    bands, H, W = hyper_image.shape

    # Make sure window length is valid and odd
    window_length = min(window_length, bands if bands % 2 else bands - 1)

    # Reshape to (H*W, bands) for per-pixel filtering
    reshaped = hyper_image.transpose(1, 2, 0).reshape(-1, bands)

    # Apply SG filter across the spectral dimension for each pixel
    smoothed = savgol_filter(reshaped, window_length=window_length, polyorder=polyorder, deriv = deriv, axis=1)

    # Reshape back to (bands, H, W)
    smoothed_cube = smoothed.reshape(H, W, bands).transpose(2, 0, 1)

    return smoothed_cube

def preprocessing_savgol(hsi_np, wlens, window_length=11, polyorder=2, deriv = 2, min_wavelength=0, selected_bands=None):
    """
    Removes spectral bands with wavelengths below min_wavelength from the hyperspectral image.
    Also replaces negative values with 0 and applies optional normalization.
    Also apply optional band selection, but that is inteded to be used in the dataset class 
    after determining selected bands based on PCA loadings.

    Parameters:
    - hsi_np (nunmpy array): Hyperspectral image with shape (bands, height, width).
    - wlens (numpy array): Wavelength values with shape (bands, ) corresponding to bands in hsi_np
    - min_wavelength (int or float): Minimum wavelength (in nm) to keep in the hyperspectral data
    - normalize (bool): Whether to normalize the hyperspectral image data
    - selected_bands (list or None): List of exact wavelength values to keep. If None, all bands >= min_wavelength are kept.

    Returns:
    - hsi_np_filtered (numpy array): Hyperspectral image with selected spectral bands of shape (filtered bands, height, width)
    - wlens_filtered (numpy array): Updated wavelengths array of shape (filtered bands, )
    """
    # Determine wavelengths to keep
    valid_bands = wlens >= min_wavelength  
    # Filter hsi_np and wlens to keep only relevant bands
    hsi_np_filtered = hsi_np[valid_bands, :, :]
    wlens_filtered = wlens[valid_bands]
    hsi_np_savgol = savgol_filter_hypercube(hsi_np_filtered, window_length,polyorder, deriv = deriv)

    # Perform band selection - will mainly be used in the dataset class for dimensionality reduction
    if selected_bands is not None:
        selected_bands = np.sort(selected_bands)
        # Get indices of the selected bands in wlens_filtered
        selected_band_indices = np.searchsorted(wlens_filtered, selected_bands)
        # Filter hsi_np and wlens to keep only selected bands
        hsi_np_savgol = hsi_np_savgol[selected_band_indices, :, :]
        wlens_filtered = wlens_filtered[selected_band_indices]

    return hsi_np_savgol, wlens_filtered

def normalize_cube_bandwise(cube, debug=False):
    """
    Normalize each spectral band to [0, 1] individually across spatial dimensions.
    Handles constant bands and avoids NaNs/Infs.

    Parameters:
    - cube: np.ndarray of shape (bands, H, W)
    - debug: if True, prints details when anomalies are found

    Returns:
    - cube_norm: np.ndarray of same shape, normalized band-wise
    """
    cube_norm = np.empty_like(cube)

    for i in range(cube.shape[0]):
        band = cube[i]

        # Mask invalid values
        valid_mask = np.isfinite(band)
        if not np.any(valid_mask):
            # Entire band is NaN or Inf — zero it out
            if debug:
                print(f"[Band {i}] All values are NaN or Inf — zeroing band.")
            cube_norm[i] = 0.0
            continue

        valid_band = band[valid_mask]
        min_val, max_val = valid_band.min(), valid_band.max()

        if max_val > min_val:
            norm_band = (band - min_val) / (max_val - min_val)
        else:
            # Constant band: just subtract min_val
            norm_band = band - min_val

        # Replace remaining NaN/Inf (due to empty areas or division)
        norm_band = np.nan_to_num(norm_band, nan=0.0, posinf=0.0, neginf=0.0)
        cube_norm[i] = norm_band

    return cube_norm
        
def stack_original_and_derivatives_as_channels(hsi_np, wlens, window_length=11, polyorder=2, min_wavelength=0, selected_bands=None, target_size = (128,128)):
    valid_bands = wlens >= min_wavelength
    hsi_np_filtered = hsi_np[valid_bands, :, :]
    wlens_filtered = wlens[valid_bands]

    # Step 2: Convert to tensor and resize spatial dimensions
    hsi_tensor = torch.from_numpy(hsi_np_filtered).unsqueeze(0).float()  # shape: (1, D, H, W)
    hsi_resized = F.interpolate(hsi_tensor, size=target_size, mode='bilinear', align_corners=False)
    hsi_np_filtered = hsi_resized.squeeze(0).numpy()  # shape: (D, H, W)

    first_deriv = savgol_filter_hypercube(hsi_np_filtered, window_length, polyorder, deriv=1)
    second_deriv = savgol_filter_hypercube(hsi_np_filtered, window_length, polyorder, deriv=2)

    if selected_bands is not None:
        selected_bands = np.sort(selected_bands)
        selected_band_indices = np.searchsorted(wlens_filtered, selected_bands)
        hsi_np_filtered = hsi_np_filtered[selected_band_indices]
        first_deriv = first_deriv[selected_band_indices]
        second_deriv = second_deriv[selected_band_indices]
        wlens_filtered = wlens_filtered[selected_band_indices]

    hsi_np_filtered = normalize_cube_bandwise(hsi_np_filtered, debug=True)
    first_deriv = normalize_cube_bandwise(first_deriv, debug=True)
    second_deriv = normalize_cube_bandwise(second_deriv, debug=True)

    # Stack along new channel axis: (C=3, D, H, W)
    stacked_cube = np.stack([hsi_np_filtered, first_deriv, second_deriv], axis=0)

    return stacked_cube, wlens_filtered
    
def stack_original_and_derivatives(hsi_np, wlens, window_length=11, polyorder=2, min_wavelength=0, selected_bands=None):
    """
    Stacks the original spectrum, first derivative, and second derivative along the spectral dimension.

    Returns:
    - stacked_cube: numpy array of shape (3*bands, H, W)
    - stacked_wlens: numpy array of shape (3*bands) representing stacked wavelengths
    """

    # Step 1: Filter out wavelengths below the threshold
    valid_bands = wlens >= min_wavelength  
    hsi_np_filtered = hsi_np[valid_bands, :, :]
    wlens_filtered = wlens[valid_bands]

    # Step 2: Compute first and second derivatives
    first_deriv = savgol_filter_hypercube(hsi_np_filtered, window_length, polyorder, deriv=1)
    second_deriv = savgol_filter_hypercube(hsi_np_filtered, window_length, polyorder, deriv=2)

    # Step 2: Apply band selection if needed
    if selected_bands is not None:
        selected_bands = np.sort(selected_bands)
        selected_bands_indices = np.searchsorted(wlens_filtered, selected_bands)

        hsi_np_filtered = hsi_np_filtered[selected_bands_indices, :, :]
        first_deriv = first_deriv[selected_bands_indices, :, :]
        second_deriv = second_deriv[selected_bands_indices, :, :]
        wlens_filtered = wlens_filtered[selected_bands_indices]

    # Step 4: Normalize each part independently
    hsi_np_filtered = normalize_cube_bandwise(hsi_np_filtered)
    first_deriv = normalize_cube_bandwise(first_deriv)
    second_deriv = normalize_cube_bandwise(second_deriv)

    # Step 5: Stack all three
    stacked_cube = np.concatenate([hsi_np_filtered, first_deriv, second_deriv], axis=0)
    stacked_wlens = np.concatenate([wlens_filtered] * 3)
    

    return stacked_cube, stacked_wlens

def load_and_flatten_hsi(img_paths, mask_dir=None, individual_normalize=False, apply_mask=False, mask_method=1, min_wavelength=0):
    """
    Transforms the 3D hyperspectral images into a 2D array by flattening the spatial dimensions.
    The resulting "rows" are the pixels and the "columns" store their values for the different spectral bands. 
    Can be used with a single- or multiple HSI-s. If multiple HSI-s are provided, they are stacked together.

    Parameters:
    - img_paths (list of str): Paths to the HSI files to be loaded and flattened
    - mask_dir (str): Path to the folder containing the masks for the HSI-s
    - individual_normalize (bool): Whether to normalize each HSI individually before flattening and stacking
    - apply_mask (bool): Whether to apply the mask to the HSI-s
    - mask_method (int): 0 for keeping only leaf, 1 for keeping leaf+stem after applying the mask
    - min_wavelength (int or float): Minimum wavelength (in nm) to keep in the hyperspectral data

    Returns:
    - numpy array, shape (n_pixels, n_bands), the flattened and stacked HSI pixels
    """
    all_pixels = []
    
    # Load wavelengths from the first image (but could be any image since images have the same spectral bands)
    _, wlens = LoadHSI(img_paths[0], return_wlens=True)
    
    for file in img_paths:
        hsi_np = LoadHSI(file)
        
        # Load and apply mask if required
        if apply_mask and mask_dir:
            mask_file = os.path.join(mask_dir, os.path.basename(file).replace('.hdf5', '.png'))    # Find the mask for the HSI (same name)
            mask_np = read_mask(mask_file)
            hsi_np = hsi_np * np.where(mask_np == 2, mask_method, mask_np)
        
        # Preprocess the hyperspectral image   
        hsi_np, _ = preprocess(hsi_np, wlens, min_wavelength=min_wavelength, normalize=individual_normalize)

        # Flatten: (bands, height, width) → (height*width, bands)
        hsi_np = hsi_np.reshape(hsi_np.shape[0], -1).T
        
        # Remove background (zero) pixels (those that were masked out)
        if apply_mask:
            hsi_np = hsi_np[~np.all(hsi_np == 0, axis=1)]

        all_pixels.append(hsi_np)
    
    # Stack all pixels together
    return np.vstack(all_pixels)



def hsi_transform_to_pca_space(hsi_np, pca, scaler, mask_np=None, mask_method=1):
    """
    Apply pre-trained PCA on HSI to reduce spectral bands, i.e. transform data to PCA space.
    If a mask is provided, only the valid non-background pixels are transformed to the PCA space with the pre-trained PCA model. Also,
    the background pixels will be set to 0 in the PCA space as well.

    Parameters:
    - hsi_np (numpy array): Hyperspectral image with shape (bands, height, width).
    - pca: Pre-fitted PCA model.
    - scaler: Pre-fitted StandardScaler model.
    - mask_np (numpy array): Mask to apply on the HSI.
    - mask_method (int): 0 for keeping only leaf, 1 for keeping leaf+stem after applying the mask.

    Returns:
    - numpy array of PCA-transformed HSI with shape (pca_components, height, width).
    """
    # Set all negative values to 0 (these are noise)
    # hsi_np = np.maximum(hsi_np, 0)    # Q: Interestingly if we do this, the reconstruction errors show vertical lines
    # A: Because we would need to change negative values to 0 in the function that calculates reconstruction error as well to the
    # "original" hsi_np that we compare the reconstructed to. → Better apply np.maximum(hsi_np, 0) before calling the function and not inside it.
    
    # Flatten: (bands, height, width) → (height*width, bands)
    hsi_flattened = hsi_np.reshape(hsi_np.shape[0], -1).T
    
    # Apply mask if provided (exclude background from PCA)
    if mask_np is not None:
        mask_np = np.where(mask_np == 2, mask_method, mask_np)
        valid_indices = mask_np.flatten() != 0
        hsi_valid = hsi_flattened[valid_indices]    # Keep only non-background pixels
    else:
        hsi_valid = hsi_flattened

    # Standardize using the previously fitted scaler
    hsi_valid_scaled = scaler.transform(hsi_valid)

    # Apply the pre-trained PCA model
    hsi_pca_valid = pca.transform(hsi_valid_scaled)

    # Return to the original number of pixels, with 0s for background pixels if mask was applied and PCA-transformed pixels for the rest
    if mask_np is not None:
        hsi_pca = np.zeros((hsi_flattened.shape[0], pca.n_components_))    # Initialize empty array with the shape of the original pixels
        hsi_pca[valid_indices] = hsi_pca_valid    # Insert only valid transformed pixels
    else:
        hsi_pca = hsi_pca_valid    # No mask applied, just return transformed pixels

    # Reshape back to (pca_components, height, width)
    return hsi_pca.T.reshape(pca.n_components_, hsi_np.shape[1], hsi_np.shape[2])



def reconstruct_hsi_from_pca_space(hsi_pca, pca, scaler, mask_np=None, mask_method=1):
    """
    Back-transform PCA-transformed HSI to original space.
    If a mask is provided, only the valid non-background pixels are reconstructed from the PCA space to the original space, with background
    pixels set to 0.
    
    Parameters:
    - hsi_pca (numpy array): PCA-transformed HSI with shape (pca_components, height, width).
    - pca: Pre-fitted PCA model.
    - scaler: Pre-fitted StandardScaler model.
    - mask_np (numpy array): Mask to apply on the HSI.
    - mask_method (int): 0 for keeping only leaf, 1 for keeping leaf+stem after applying the mask.

    Returns:
    - numpy array of back-transformed (reconstructed) HSI with shape (bands, height, width).
    """
    # Flatten: (pca_components, height, width) → (height*width, pca_components)
    hsi_pca_flattened = hsi_pca.reshape(pca.n_components_, -1).T
    
    # Apply mask if provided (exclude background from PCA)
    if mask_np is not None:
        mask_np = np.where(mask_np == 2, mask_method, mask_np)
        valid_indices = mask_np.flatten() != 0
        hsi_valid = hsi_pca_flattened[valid_indices]    # Keep only non-background pixels
    else:
        hsi_valid = hsi_pca_flattened

    # Apply inverse PCA
    hsi_valid_reconstructed = pca.inverse_transform(hsi_valid)

    # Apply inverse scaling
    hsi_valid_reconstructed = scaler.inverse_transform(hsi_valid_reconstructed)
    
    # Reconstruct full spatial structure if mask was applied
    if mask_np is not None:
        hsi_reconstructed = np.zeros((hsi_pca_flattened.shape[0], hsi_valid_reconstructed.shape[1]))    # Initialize empty array of shape (original pixels, original bands)
        hsi_reconstructed[valid_indices] = hsi_valid_reconstructed    # Insert only valid transformed pixels
    else:
        hsi_reconstructed = hsi_valid_reconstructed    # No mask applied, just return reconstructed pixels

    # Reshape back to (bands, height, width)
    return hsi_reconstructed.T.reshape(-1, hsi_pca.shape[1], hsi_pca.shape[2])



def compress_and_reconstruct_hsi_pca(hsi_np, pca, scaler, mask_np=None, mask_method=1):
    '''
    Preform PCA compression and reconstruction right after on a HSI data.
    
    Parameters:
    - hsi_np (numpy array): Hyperspectral image with shape (bands, height, width).
    - pca: Pre-fitted PCA model.
    - scaler: Pre-fitted StandardScaler model.
    - mask_np (numpy array): Mask to apply on the HSI.
    - mask_method (int): 0 for keeping only leaf, 1 for keeping leaf+stem after applying the mask.    
    '''
    # Transform data to PCA space
    hsi_pca = hsi_transform_to_pca_space(hsi_np, pca, scaler, mask_np, mask_method)
    
    # Reconstruct data from PCA space
    hsi_reconstructed = reconstruct_hsi_from_pca_space(hsi_pca, pca, scaler, mask_np, mask_method)
    
    return hsi_reconstructed



def get_pca_reconstruction_error(hsi_np, pca, scaler, mask_np=None, mask_method=1, show_plot=True):
    """
    Reconstruct an input HSI using the pre-trained PCA, calculate the reconstruction error and optionally plot the sum of it across bands.
    Should be used with a single HSI file.

    Parameters:
    - hsi_np (numpy array): Hyperspectral image with shape (bands, height, width)
    - pca: pre-fitted PCA model
    - scaler: pre-fitted StandardScaler model
    - mask_np (numpy array): Mask to apply on the HSI
    - mask_method (int): 0 for keeping only leaf, 1 for keeping leaf+stem after applying the mask
    - show_plot (bool): Whether to show the plot of the sum of reconstruction errors across the bands

    Returns:
    - reconstruction_error (numpy array): The reconstruction error of the HSI (for each band) with shape (bands, height, width) 
    - Plot (optional): "Heatmap" of the sum of reconstruction errors across the bands
    """
    # Apply PCA and reconstruct
    hsi_pca = hsi_transform_to_pca_space(hsi_np, pca, scaler, mask_np, mask_method)
    hsi_reconstructed = reconstruct_hsi_from_pca_space(hsi_pca, pca, scaler, mask_np, mask_method)
    
    # Compute reconstruction error
    if mask_np is not None:
        mask_np = np.where(mask_np == 2, mask_method, mask_np)
        reconstruction_error = np.abs(hsi_np - hsi_reconstructed) * mask_np
    else:
        reconstruction_error = np.abs(hsi_np - hsi_reconstructed)
        
    if show_plot:
        # Sum reconstruction error across the bands
        pixel_errors = np.sum(reconstruction_error, axis=0)

        # Plot pixel errors
        plt.figure(figsize=(8, 6))
        plt.imshow(pixel_errors, cmap='hot')
        plt.colorbar(label='Total Reconstruction Error')
        plt.title('Total Reconstruction Error per Pixel')
        plt.axis('off')
        plt.show()
    
    return reconstruction_error
