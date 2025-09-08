import torch
import time
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import numpy as np
import gc
import time
import copy
import os
import umap as UMAP
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cv2
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.ndimage import binary_erosion, binary_dilation
from scipy import ndimage
from vein_detection import *
from datasets import *
from preprocessing import *
from utils import *




##############################
### Reconstruction analysis
def get_reconstruction_errors(model, dataloader, device, error_metric='mae',
                              mask_after=False, remove_edges=False, mask_resize=256):
    """
    Computes per-image reconstruction error (MAE or MSE), optionally masked and with edge exclusion.

    Args:
        model: Trained autoencoder model.
        dataloader: DataLoader yielding images (and optionally img_paths via dataset).
        device: Device to perform computation on.
        error_metric: "mae" or "mse".
        mask_after: Whether to apply annotation mask after reconstruction.
        remove_edges: Whether to exclude leaf edges (via annotation) from error.
        mask_resize: Size to resize the mask to (default: 256).

    Returns:
        List of reconstruction errors (one per image).
    """
    error_metric = error_metric.lower()
    if error_metric not in ['mae', 'mse']:
        raise ValueError("error_metric must be 'mae' or 'mse'")

    model.eval()
    reconstruction_errors = []

    with torch.no_grad():
        for batch_idx, (data, *_) in enumerate(dataloader):
            batch_size = data.unsqueeze(1).shape[0]
            data = data.unsqueeze(1).to(device)
            recon = model(data)

            for i in range(batch_size):
                original = data[i]         # [C, H, W]
                reconstructed = recon[i]  # [C, H, W]

                # Compute per-pixel error map
                if error_metric == 'mse':
                    error_map = (original - reconstructed) ** 2
                else:
                    error_map = torch.abs(original - reconstructed)

                # Aggregate over channels
                error_map = torch.mean(error_map, dim=0).cpu().numpy()  # [H, W]

                if mask_after:
                    # Get corresponding image path
                    idx = batch_idx * dataloader.batch_size + i
                    if idx >= len(dataloader.dataset.img_paths):
                        break

                    file_path = dataloader.dataset.img_paths[idx]
                    mask_path = os.path.join(dataloader.dataset.mask_dir,
                                             os.path.basename(file_path).replace('.hdf5', '.png'))

                    mask = read_mask(mask_path)
                    mask = cv2.resize(mask, (mask_resize, mask_resize), interpolation=cv2.INTER_NEAREST)

                    if remove_edges:
                        mask = (1 - extract_full_edge(mask)) * mask  # zero-out edge pixels

                    # Masked mean error (only nonzero mask pixels)
                    error_masked = error_map * mask
                    valid_pixels = error_masked[mask != 0]
                    overall_error = float(np.mean(valid_pixels)) if valid_pixels.size > 0 else 0.0
                else:
                    # Unmasked mean over whole image
                    overall_error = float(np.mean(error_map))

                reconstruction_errors.append(overall_error)

    return reconstruction_errors


def get_recon_error_threshold(model, dataloader, device, file_path,
                              dataloader_early=None, dataloader_mid=None, dataloader_late=None,
                              show_plot=True, error_metric="mae", mask_after=False, remove_edges=False):
    """
    Computes the maximum reconstruction error on the validation set to use as a threshold for classification.
    Supports masking and edge removal during error calculation.

    Args:
        model: Trained autoencoder model.
        dataloader: Validation dataloader (assumed to be healthy samples).
        device: Device to run inference on (e.g., 'cuda' or 'cpu').
        file_path: Path to save the plot.
        dataloader_early: Optional dataloader for early disease stage.
        dataloader_mid: Optional dataloader for mid disease stage.
        dataloader_late: Optional dataloader for late disease stage.
        show_plot: Whether to show the histogram plot.
        error_metric: 'mae' or 'mse'.
        mask_after: Whether to apply a mask to exclude background.
        remove_edges: Whether to remove edges from the mask region.

    Returns:
        float: Threshold based on max validation error.
    """

    # --- Compute reconstruction errors ---
    reconstruction_errors = get_reconstruction_errors(
        model, dataloader, device,
        error_metric=error_metric,
        mask_after=mask_after,
        remove_edges=remove_edges
    )

    max_reconstruction_error = max(reconstruction_errors)

    # Compute errors for optional datasets
    errors_early = get_reconstruction_errors(model, dataloader_early, device) if dataloader_early else []
    errors_mid = get_reconstruction_errors(model, dataloader_mid, device) if dataloader_mid else []
    errors_late = get_reconstruction_errors(model, dataloader_late, device) if dataloader_late else []

    # Find a common range for all histograms
    all_errors = reconstruction_errors + errors_early + errors_mid + errors_late
    min_value, max_value = min(all_errors), max(all_errors)

    if show_plot:
        plt.figure(figsize=(10, 5))
        
        # Plot histogram with fixed bins and range
        plt.hist(reconstruction_errors, bins=60, range=(min_value, max_value),
                 color='blue', alpha=0.9, edgecolor='black', label="Validation (Healthy)")

        if errors_early:
            plt.hist(errors_early, bins=60, range=(min_value, max_value),
                     color='green', alpha=0.7, edgecolor='black', label="Early Disease")
        if errors_mid:
            plt.hist(errors_mid, bins=60, range=(min_value, max_value),
                     color='orange', alpha=0.7, edgecolor='black', label="Mid Disease")
        if errors_late:
            plt.hist(errors_late, bins=60, range=(min_value, max_value),
                     color='red', alpha=0.7, edgecolor='black', label="Late Disease")

        # Add threshold line
        plt.axvline(max_reconstruction_error, color='black', linestyle='dashed', linewidth=2, 
                    label="Threshold (Max validation Error)")

        # Labels and title
        plt.xlabel("Reconstruction Error")
        plt.ylabel("Frequency")
        plt.title("Distribution of Reconstruction Errors")
        plt.legend()
        plt.savefig(f'{file_path}aggregated_errors_images.png')
        plt.show()

    return max_reconstruction_error



################################################
# Pixelwise reconstruction errors

# --- Function to compute pixelwise error for an entire dataloader ---
def get_pixel_reconstruction_errors(model, dataloader, device, error_metric = "MAE"):
    """
    Computes per-pixel reconstruction error (MSE/MAE summed over spectral bands) for entire dataloader.

    Args:
        model: Autoencoder.
        dataloader: DataLoader.
        device: torch.device.
        error_metric: Used error calculation (MSE or MAE)

    Returns:
        np.ndarray: Flattened array of all pixel errors.
    """
    model.eval()
    all_errors = []

    with torch.no_grad():
        for batch,_ in dataloader:
            batch = batch.unsqueeze(1).to(device)  # shape: [B, 1, C, H, W]
            recon = model(batch)                  # same shape
            if error_metric=="MSE":
                error = (recon - batch).pow(2).sum(dim=2)  # sum over spectral bands (C)
            else:
                error = (recon - batch).abs().sum(dim=2)
            # result: [B, 1, H, W]
            error = error.squeeze(1)  # now [B, H, W]
            all_errors.append(error.cpu().flatten())

    return torch.cat(all_errors).numpy()

# --- Function to compute pixelwise error for an entire dataloader and get ---
def get_pixel_error_threshold(model, dataloader, device, quantile = 0.75, error_metric = "MAE"):
    """
    Computes pixel-wise reconstruction error threshold using a quantile of errors from the given dataset.

    Args:
        model: Autoencoder.
        dataloader: DataLoader.
        device: torch.device.
        quantile: Threshold quantile (default 0.75).

    Returns:
        float: Pixel-level error threshold.
    """
    model.eval()
    all_errors = []

    with torch.no_grad():
        for batch,_ in dataloader:
            batch = batch.unsqueeze(1).to(device)  # shape: [B, 1, C, H, W]
            recon = model(batch)                  # same shape
            if error_metric == "MSE":
                error = (recon - batch).pow(2).sum(dim=2)  # sum over spectral bands (C)
            else:
                error = (recon - batch).abs().sum(dim=2)

            all_errors.append(error.cpu().flatten())
    all_pixel_errors = torch.cat(all_errors).numpy()
    threshold = np.quantile(all_pixel_errors, quantile)

    return threshold

def classify_leaves_pixel_error_aggregate(model, dataloader, device, threshold, error_metric = "MAE"):
    """
    Computes a score for each image based on the sum of pixel errors that exceed a given threshold.
    Uses absolute error aggregated over channels.

    Args:
        model: Trained autoencoder model.
        dataloader: DataLoader with input images.
        device: torch.device.
        threshold: Pixel-wise error threshold.

    Returns:
        List of image-level anomaly scores.
    """
    model.eval()
    image_scores =[]

    with torch.no_grad():
        for batch,_ in dataloader:
            batch = batch.unsqueeze(1).to(device)  # shape: [B, 1, C, H, W]
            recon = model(batch) 
            if error_metric == "MSE":
                pixel_errors = (recon - batch).abs().squeeze(1).mean(dim=1)
            else:
                pixel_errors = (recon - batch).abs().sum(dim=1)

            for err_map in pixel_errors:
               high_error_pixels = err_map[err_map>threshold]
               score = high_error_pixels.sum().mean()
               image_scores.append(score)

    return image_scores

# classify the leaves based on a threshold
def classify_leaves_pixel_error_aggregate_label(model, dataloader, label, device, threshold):
    model.eval()
    image_scores =[]
    labels = []

    with torch.no_grad():
        for batch,_ in dataloader:
            batch = batch.unsqueeze(1).to(device)  # shape: [B, 1, C, H, W]
            recon = model(batch) 
            pixel_errors = (recon-batch).abs().squeeze(1).sum(dim=1)

            for err_map in pixel_errors:
               high_error_pixels = err_map[err_map>threshold]
               score = high_error_pixels.sum().item()
               image_scores.append(score)
            labels.extend([label] * batch.size(0))

    return image_scores, labels

def classify_leaves_pixel_error_aggregate_label_edge_removal(model, dataloader, label, device, threshold, MASK_FOLDER):
    model.eval()
    image_scores =[]
    labels = []

    with torch.no_grad():
        for batch, idxs in dataloader:
            batch = batch.unsqueeze(1).to(device)  # [B, 1, C, H, W]
            recon = model(batch)
            pixel_errors = (recon - batch).abs().squeeze(1).sum(dim=1)  # [B, H, W]

            for i, err_map in enumerate(pixel_errors):
                idx = idxs[i].item()

                # === Load edge mask ===
                image_path = dataloader.dataset.img_paths[idx]
                mask_path = os.path.join(MASK_FOLDER, os.path.basename(image_path).replace('.hdf5', '.png'))
                mask = read_mask(mask_path)
                mask = Image.fromarray(mask).resize((256, 256), resample=Image.NEAREST)
                mask = np.array(mask)

                edge_mask = extract_full_edge(mask)               # [H, W]
                non_edge_mask = np.logical_not(edge_mask)         # exclude edge

                # Apply threshold + mask
                err_np = err_map.cpu().numpy()                    # [H, W]
                exceed_mask = (err_np > threshold) & non_edge_mask
                score = err_np[exceed_mask].sum()

                image_scores.append(score)
                labels.append(label)

    return image_scores, labels

def get_pixel_threshold_per_band(model, dataloader, device, quantile=0.75, error_metric="MAE"):
    """
    Compute per-band pixel error thresholds using absolute (MAE) or squared (MSE) error.
    
    Args:
        model: trained PyTorch model
        dataloader: DataLoader yielding [B, C, H, W] or [B, 1, C, H, W]
        device: torch.device
        quantile: float in (0, 1), e.g., 0.99 for top 1%
        error_metric: "MAE" (default) or "MSE"
    
    Returns:
        thresholds: np.ndarray of shape [C], per-band threshold
    """
    model.eval()
    all_errors_per_band = []

    with torch.no_grad():
        for batch, _ in dataloader:
            batch = batch.unsqueeze(1).to(device)  # Ensure [B, 1, C, H, W]
            recon = model(batch)                   # [B, 1, C, H, W]

            if error_metric == "MSE":
                error = (recon - batch).pow(2).squeeze(1)     # [B, C, H, W]
            else:  # Default to absolute error
                error = torch.abs(recon - batch).squeeze(1)   # [B, C, H, W]

            # Rearrange to [C, B*H*W] for each band
            error_per_band = error.permute(1, 0, 2, 3).reshape(error.shape[1], -1)
            all_errors_per_band.append(error_per_band.cpu())

    all_errors_per_band = torch.cat(all_errors_per_band, dim=1)  # [C, total_pixels]
    thresholds = np.quantile(all_errors_per_band.numpy(), quantile, axis=1)

    return thresholds

def classify_leaves_band_pixel_error_aggregate_label(model, dataloader, label, device, thresholds):
    """
    Computes a score per image by summing pixel errors that exceed per-band thresholds.
    Aggregates across all bands.

    Args:
        model: Trained autoencoder.
        dataloader: DataLoader yielding (images, _).
        label: Ground truth label to assign.
        device: torch.device.
        thresholds: List or np.ndarray of shape [C], per-band thresholds.

    Returns:
        image_scores: List of anomaly scores per image.
        labels: List of labels (same length).
    """
    model.eval()
    image_scores = []
    labels = []

    with torch.no_grad():
        for batch, _ in dataloader:
            batch = batch.unsqueeze(1).to(device)              # [B, 1, C, H, W]
            recon = model(batch)                               # [B, 1, C, H, W]
            pixel_errors = (recon - batch).abs().squeeze(1)    # [B, C, H, W]

            for img_errors in pixel_errors:  # [C, H, W]
                score = 0.0
                img_errors_np = img_errors.cpu().numpy()
                for c in range(img_errors_np.shape[0]):
                    band_errors = img_errors_np[c]                      # [H, W]
                    exceed_mask = band_errors > thresholds[c]
                    score += band_errors[exceed_mask].sum()
                image_scores.append(score)
                labels.append(label)

    return image_scores, labels


def get_pixel_threshold_per_band_edge_removal(model, dataloader, device, MASK_FOLDER, quantile=0.75, error_metric="MAE"):
    """
    Compute per-band pixel error thresholds using absolute (MAE) or squared (MSE) error.
    
    Args:
        model: trained PyTorch model
        dataloader: DataLoader yielding [B, C, H, W] or [B, 1, C, H, W]
        device: torch.device
        quantile: float in (0, 1), e.g., 0.99 for top 1%
        error_metric: "MAE" (default) or "MSE"
    
    Returns:
        thresholds: np.ndarray of shape [C], per-band threshold
    """
    model.eval()
    all_errors_per_band = []

    with torch.no_grad():
        for batch, idxs in dataloader:
            batch = batch.unsqueeze(1).to(device)  # Ensure [B, 1, C, H, W]
            recon = model(batch)                   # [B, 1, C, H, W]

            for data_img, recon_img, idx in zip(batch, recon, idxs):
                # Convert tensors to NumPy arrays
                data_np = data_img.squeeze(0).cpu().numpy()   # [C, H, W]
                recon_np = recon_img.squeeze(0).cpu().numpy() # [C, H, W]

                # === Load edge mask ===
                image_path = dataloader.dataset.img_paths[idx.item()]
                mask_path = os.path.join(MASK_FOLDER, os.path.basename(image_path).replace('.hdf5', '.png'))
                mask = read_mask(mask_path)
                mask = Image.fromarray(mask).resize((256, 256), resample=Image.NEAREST)
                mask = np.array(mask)

                edge_line = extract_full_edge(mask)  # [H, W]
                edge_removal = np.where((edge_line == 0) | (edge_line == 1), edge_line ^ 1, edge_line)  # [H, W]

                # === Compute error ===
                if error_metric == "MSE":
                    error = (data_np - recon_np) ** 2  # [C, H, W]
                else:
                    error = np.abs(data_np - recon_np) # [C, H, W]

                # Apply edge mask
                masked_error = error * edge_removal[None, :, :]  # [C, H, W]

                # Convert to tensor and extract valid values per band
                error_tensor = torch.from_numpy(masked_error).to(device)  # [C, H, W]
                band_values = []
                for c in range(error_tensor.shape[0]):
                    selected = error_tensor[c][edge_removal > 0].view(1, -1)  # [1, N_pixels]
                    band_values.append(selected.cpu())

                band_tensor = torch.cat(band_values, dim=0)  # [C, N_pixels]
                all_errors_per_band.append(band_tensor)

    all_errors_per_band = torch.cat(all_errors_per_band, dim=1)  # [C, total_pixels]
    thresholds = np.quantile(all_errors_per_band.numpy(), quantile, axis=1)

    return thresholds

def classify_leaves_band_pixel_error_aggregate_label_edge_removal(model, dataloader, label, device, thresholds, MASK_FOLDER):
    """
    Computes a score per image by summing reconstruction errors above per-band thresholds,
    with edge regions excluded.

    Args:
        model: Trained autoencoder.
        dataloader: DataLoader yielding (images, indices).
        label: Label to assign to all images.
        device: torch.device.
        thresholds: np.ndarray of shape [C], per-band thresholds.
        MASK_FOLDER: Path to annotation masks.

    Returns:
        image_scores: List of anomaly scores per image.
        labels: List of labels (same length).
    """
    model.eval()
    image_scores = []
    labels = []

    with torch.no_grad():
        for batch, idxs in dataloader:
            batch = batch.unsqueeze(1).to(device)         # [B, 1, C, H, W]
            recon = model(batch)                          # [B, 1, C, H, W]
            pixel_errors = (recon - batch).abs().squeeze(1)  # [B, C, H, W]

            for i in range(pixel_errors.shape[0]):
                idx = idxs[i].item()

                # === Load edge mask ===
                image_path = dataloader.dataset.img_paths[idx]
                mask_path = os.path.join(MASK_FOLDER, os.path.basename(image_path).replace('.hdf5', '.png'))
                mask = read_mask(mask_path)
                mask = Image.fromarray(mask).resize((256, 256), resample=Image.NEAREST)
                mask = np.array(mask)

                edge_mask = extract_full_edge(mask)              # [H, W]
                non_edge_mask = np.logical_not(edge_mask)        # [H, W]

                # Apply band-wise threshold + edge mask
                err_tensor = pixel_errors[i].cpu().numpy()       # [C, H, W]
                score = 0.0
                for c in range(err_tensor.shape[0]):
                    exceed_mask = (err_tensor[c] > thresholds[c]) & non_edge_mask
                    score += err_tensor[c][exceed_mask].sum()

                image_scores.append(score)
                labels.append(label)

    return image_scores, labels

def classify_leaves_pixel_error_mean(model, dataloader, device, threshold, error_metric = "MAE"):
    """
    Computes the mean reconstruction error per image over all pixels and bands using MSE.

    Args:
        model: Trained autoencoder.
        dataloader: DataLoader with input images.
        device: torch.device.
        threshold: Unused in this version.

    Returns:
        List of mean image-level reconstruction errors.
    """
    model.eval()
    image_scores =[]
    with torch.no_grad():
            for batch, _ in dataloader:
                batch = batch.unsqueeze(1).to(device)  # [B, 1, D, H, W]
                recon = model(batch)
                if error_metric == "MSE":
                    pixel_errors = (recon - batch).abs().squeeze(1).mean(dim=1)
                else:
                    pixel_errors = (recon - batch).abs().sum(dim=1)

                for err_map in pixel_errors:
                    high_error_pixels = err_map[err_map>threshold]
                    mean_error = high_error_pixels.mean().item()
                    image_scores.append(mean_error)
    return image_scores


def get_pixel_errors_per_band(model, dataloader, device):
    """
    Returns all pixel errors per spectral band, flattened and grouped by band.

    Args:
        model: Autoencoder.
        dataloader: DataLoader with inputs.
        device: torch.device.

    Returns:
        Tensor of shape [C, total_pixels] with per-band error values.
    """
    model.eval()
    all_errors_per_band = []  # list of [C, N_pixels] tensors

    with torch.no_grad():
        for batch, _ in dataloader:
            batch = batch.unsqueeze(1).to(device)  # [B, 1, C, H, W]
            recon = model(batch)                   # [B, 1, C, H, W]
            error = (recon - batch).pow(2).squeeze(1)  # [B, C, H, W]

            # Reshape to [C, B*H*W]
            error_per_band = error.permute(1, 0, 2, 3).reshape(error.shape[1], -1)
            all_errors_per_band.append(error_per_band.cpu())

    # Concatenate along pixel dimension
    all_errors_per_band = torch.cat(all_errors_per_band, dim=1)  # [C, total_pixels]
    return all_errors_per_band  # tensor [C, N_pixels_total]

def get_pixel_errors_per_band_edge_removal(model, dataloader, device, MASK_FOLDER, error_metric = "MAE"):
    """
    Returns all pixel errors per spectral band, flattened and grouped by band. It also removes the edges

    Args:
        model: Autoencoder.
        dataloader: DataLoader with inputs.
        device: torch.device.

    Returns:
        Tensor of shape [C, total_pixels] with per-band error values.
    """
    model.eval()
    all_errors_per_band = []  # list of [C, N_pixels] tensors

    with torch.no_grad():
        for batch, idxs in dataloader:
            batch = batch.unsqueeze(1).to(device)  # [B, 1, C, H, W]
            recon = model(batch)                              # [B, 1, C, H, W]
            for i, (data_img, recon_img, idx) in enumerate(zip(batch, recon, idxs)):
                # Remove channel dim: (1, 17, 256, 256) → (17, 256, 256)
                data_np = data_img.squeeze(0).cpu().numpy()
                recon_np = recon_img.squeeze(0).cpu().numpy()
                

                # Load disease annotation mask
                image_path = dataloader.dataset.img_paths[idx.item()]
                mask_path = os.path.join(MASK_FOLDER, os.path.basename(image_path).replace('.hdf5', '.png'))
                mask = read_mask(mask_path)
                mask = Image.fromarray(mask).resize((256, 256), resample=Image.NEAREST)
                mask = np.array(mask)

                edge_line = extract_full_edge(mask)
                edge_removal = np.where((edge_line == 0) | (edge_line == 1), edge_line ^ 1, edge_line)

                if error_metric == "MSE":
                    error = (data_np - recon_np)**2     # [B, C, H, W]
                else:  # Default to absolute error
                    error = np.abs(data_np - recon_np)  # Shape: (17, 256, 256)
                
                masked_error = error * edge_removal[None,:,:]
                # Convert to tensor and extract valid values per band
                error_tensor = torch.from_numpy(masked_error).to(device)  # [C, H, W]
                band_values = []
                for c in range(error_tensor.shape[0]):
                    selected = error_tensor[c][edge_removal > 0].view(1, -1)  # [1, N_pixels]
                    band_values.append(selected.cpu())

                band_tensor = torch.cat(band_values, dim=0)  # [C, N_pixels]
                all_errors_per_band.append(band_tensor)

    all_errors_per_band = torch.cat(all_errors_per_band, dim=1)  # [C, total_pixels]
    return all_errors_per_band



    """
    Computes:
    - Per-band % of pixels above threshold (excluding edge)
    - Overall % of pixels above threshold in any band (excluding edge)

    Args:
        model: Trained autoencoder.
        dataloader: DataLoader yielding (images, indices).
        device: torch.device.
        thresholds: np.ndarray of shape [C].
        select_img: Index of image in batch to evaluate.

    Returns:
        per_band_percentages: List of % exceedance per band (excluding edge).
        overall_percentage: % of all pixel-band values that exceed threshold (excluding edge).
    """
    model.eval()

    with torch.no_grad():
        for batch, idxs in dataloader:
            batch = batch.unsqueeze(1).to(device)  # [B, 1, C, H, W]
            recon = model(batch)
            break  # Use only one batch

    input_img = batch[select_img, 0].cpu().numpy()   # [C, H, W]
    recon_img = recon[select_img, 0].cpu().numpy()   # [C, H, W]
    error = np.abs(input_img - recon_img)            # [C, H, W]
    idx = idxs[select_img].item()

    C, H, W = error.shape

    # === Load and process annotation mask for edge exclusion ===
    image_path = dataloader.dataset.img_paths[idx]
    mask_path = os.path.join(MASK_FOLDER, os.path.basename(image_path).replace('.hdf5', '.png'))
    mask = read_mask(mask_path)
    mask = Image.fromarray(mask).resize((256, 256), resample=Image.NEAREST)
    mask = np.array(mask)

    edge_mask = extract_full_edge(mask)  # [H, W]
    non_edge_mask = np.logical_not(edge_mask)  # True for valid (non-edge) pixels
    total_valid_pixels = np.sum(non_edge_mask)
    total_valid_pixels_all_bands = total_valid_pixels * C

    per_band_percentages = []
    binary_mask_stack = []

    for c in range(C):
        # Apply per-band threshold and exclude edge
        mask = (error[c] > thresholds[c]) & non_edge_mask  # [H, W]
        percentage = (np.sum(mask) / total_valid_pixels) * 100
        per_band_percentages.append(percentage)
        binary_mask_stack.append(mask.astype(np.uint8))

    # Stack and compute overall exceedance
    binary_mask_stack = np.stack(binary_mask_stack, axis=0)  # [C, H, W]
    overall_exceedance = np.sum(binary_mask_stack)  # sum of all True values
    overall_percentage = (overall_exceedance / total_valid_pixels_all_bands) * 100

    return per_band_percentages, overall_percentage



    """
    Computes per-image, per-band anomaly scores based on pixel-wise band thresholds.

    Args:
        model: Autoencoder.
        dataloader: DataLoader.
        device: torch.device.
        thresholds: np.ndarray of shape [C], per-band pixel thresholds.

    Returns:
        List of lists: each inner list contains per-band scores for one image.
    """
    model.eval()
    image_scores_per_band = []

    with torch.no_grad():
        for batch, _ in dataloader:
            batch = batch.unsqueeze(1).to(device)  # [B, 1, C, H, W]
            recon = model(batch)                   # [B, 1, C, H, W]
            if error_metric == "MSE":
                error = (recon - batch).pow(2).squeeze(1)     # [B, C, H, W]
            else:  # Default to absolute error
                error = torch.abs(recon - batch).squeeze(1)   # [B, C, H, W]


            # Iterate over batch to compute per-image, per-band score
            for sample_error in error:  # sample_error: [C, H, W]
                scores = []
                for c in range(sample_error.shape[0]):
                    high_error_pixels = sample_error[c][sample_error[c] > thresholds[c]]
                    score = high_error_pixels.sum().item() / (high_error_pixels.numel() + 1e-8)
                    scores.append(score)
                image_scores_per_band.append(scores)

    return image_scores_per_band  # list of [17]-length lists (per image)











































#######################################
### Same functions for the stacke version

def get_reconstruction_errors_stacked(model, dataloader, device):
    model.eval()
    reconstruction_errors = []
    with torch.no_grad():
        for data,_ in dataloader:
            data = data.to(device)
            recon = model(data)
            # Compute reconstruction error per image
            batch_errors = torch.mean((recon - data) ** 2, dim=[1, 2, 3, 4])  # MSE per image
            reconstruction_errors.extend(batch_errors.cpu().numpy())  # Store in list
    return reconstruction_errors


def get_recon_error_threshold_stacked(model, dataloader, device, file_path, dataloader_early=None, dataloader_mid=None, dataloader_late=None, show_plot=True):
    """
    Computes the maximum reconstruction error on the provided data (intended to be the validation set) to use as a threshold for classification.

    Args:
        model: Trained autoencoder model.
        dataloader: Dataloader for the data (intended to be the validation set).
        device: Device to run inference on (e.g., 'cuda' or 'cpu').
        show_plot: Whether to show a histogram of the reconstruction errors.

    Returns:
        float: The threshold for classifying leaves as healthy or unhealthy.
    """
    reconstruction_errors = get_reconstruction_errors_stacked(model, dataloader, device)
    
    max_reconstruction_error = max(reconstruction_errors)

    # Compute errors for optional datasets
    errors_early = get_reconstruction_errors_stacked(model, dataloader_early, device) if dataloader_early else []
    errors_mid = get_reconstruction_errors_stacked(model, dataloader_mid, device) if dataloader_mid else []
    errors_late = get_reconstruction_errors_stacked(model, dataloader_late, device) if dataloader_late else []

    # Find a common range for all histograms
    all_errors = reconstruction_errors + errors_early + errors_mid + errors_late
    min_value, max_value = min(all_errors), max(all_errors)

    if show_plot:
        plt.figure(figsize=(10, 5))
        
        # Plot histogram with fixed bins and range
        plt.hist(reconstruction_errors, bins=60, range=(min_value, max_value),
                 color='blue', alpha=0.9, edgecolor='black', label="Validation (Healthy)")

        if errors_early:
            plt.hist(errors_early, bins=60, range=(min_value, max_value),
                     color='green', alpha=0.7, edgecolor='black', label="Early Disease")
        if errors_mid:
            plt.hist(errors_mid, bins=60, range=(min_value, max_value),
                     color='orange', alpha=0.7, edgecolor='black', label="Mid Disease")
        if errors_late:
            plt.hist(errors_late, bins=60, range=(min_value, max_value),
                     color='red', alpha=0.7, edgecolor='black', label="Late Disease")

        # Add threshold line
        plt.axvline(max_reconstruction_error, color='black', linestyle='dashed', linewidth=2, 
                    label="Threshold (Max validation Error)")

        # Labels and title
        plt.xlabel("Reconstruction Error")
        plt.ylabel("Frequency")
        plt.title("Distribution of Reconstruction Errors")
        plt.legend()
        plt.savefig(f'{file_path}aggregated_errors_images.png')
        plt.show()

    return max_reconstruction_error

### Also pixelwise
# --- Function to compute pixelwise error for an entire dataloader ---
def get_pixel_reconstruction_errors_stacked(model, dataloader, device):
    model.eval()
    all_errors = []

    with torch.no_grad():
        for batch,_ in dataloader:
            batch = batch.to(device)  # shape: [B, 1, C, H, W]
            recon = model(batch)                  # same shape
            error = (recon - batch).pow(2).sum(dim=2)  # sum over spectral bands (C)
            # result: [B, 1, H, W]
            error = error  # now [B, H, W]
            all_errors.append(error.cpu().flatten())

    return torch.cat(all_errors).numpy()

# --- Function to compute pixelwise error for an entire dataloader and get ---
def get_pixel_error_threshold_stacked(model, dataloader, device, quantile = 0.75):
    model.eval()
    all_errors = []

    with torch.no_grad():
        for batch,_ in dataloader:
            batch = batch.to(device)  # shape: [B, 1, C, H, W]
            recon = model(batch)                  # same shape
            error = (recon - batch).pow(2).sum(dim=2)  # sum over spectral bands (C)
            
            all_errors.append(error.cpu().flatten())
    all_pixel_errors = torch.cat(all_errors).numpy()
    threshold = np.quantile(all_pixel_errors, quantile)

    return threshold

def classify_leaves_pixel_error_aggregate_stacked(model, dataloader, device, threshold):
    model.eval()
    image_scores =[]

    with torch.no_grad():
        for batch,_ in dataloader:
            batch = batch.to(device)  # shape: [B, 1, C, H, W]
            recon = model(batch) 
            pixel_errors = (recon-batch).pow(2).squeeze(1).sum(dim=1)

            for err_map in pixel_errors:
               high_error_pixels = err_map[err_map>threshold]
               score = high_error_pixels.sum().mean()
               image_scores.append(score)

    return image_scores

def classify_leaves_pixel_error_mean_stacked(model, dataloader, device, threshold):
    model.eval()
    image_scores =[]
    with torch.no_grad():
            for batch, _ in dataloader:
                batch = batch.to(device)  # [B, 1, D, H, W]
                recon = model(batch)
                pixel_errors = (recon - batch).pow(2).squeeze(1).sum(dim=1)  # [B, H, W]

                for err_map in pixel_errors:
                    mean_error = err_map.mean().item()
                    image_scores.append(mean_error)
    return image_scores


def get_pixel_threshold_per_band_stacked(model, dataloader, device, quantile=0.75):
    model.eval()
    all_errors_per_band = []

    with torch.no_grad():
        for batch, _ in dataloader:
            batch = batch.to(device)  # [B, 1, C, H, W]
            recon = model(batch)                  # [B, 1, C, H, W]
            error = (recon - batch).pow(2).squeeze(1)  # [B, C, H, W]

            # Reshape to [B*C*H*W], grouped by band
            error_per_band = error.permute(1, 0, 2, 3).reshape(error.shape[1], -1)
            all_errors_per_band.append(error_per_band.cpu())

    all_errors_per_band = torch.cat(all_errors_per_band, dim=1)  # [C, N_pixels_total]
    thresholds = np.quantile(all_errors_per_band.numpy(), quantile, axis=1)

    return thresholds

def get_pixel_errors_per_band_stacked(model, dataloader, device):
    model.eval()
    all_errors_per_band = []  # list of [C, N_pixels] tensors

    with torch.no_grad():
        for batch, _ in dataloader:
            batch = batch.to(device)  # [B, 1, C, H, W]
            recon = model(batch)                   # [B, 1, C, H, W]
            error = (recon - batch).pow(2).squeeze(1)  # [B, C, H, W]

            # Reshape to [C, B*H*W]
            error_per_band = error.permute(1, 0, 2, 3).reshape(error.shape[1], -1)
            all_errors_per_band.append(error_per_band.cpu())

    # Concatenate along pixel dimension
    all_errors_per_band = torch.cat(all_errors_per_band, dim=1)  # [C, total_pixels]
    return all_errors_per_band  # tensor [C, N_pixels_total]

def classify_leaves_per_band_stacked(model, dataloader, device, thresholds):
    model.eval()
    image_scores_per_band = []

    with torch.no_grad():
        for batch, _ in dataloader:
            batch = batch.to(device)  # [B, 1, C, H, W]
            recon = model(batch)                   # [B, 1, C, H, W]
            error = (recon - batch).pow(2).squeeze(1)  # [B, C, H, W]

            # Iterate over batch to compute per-image, per-band score
            for sample_error in error:  # sample_error: [C, H, W]
                scores = []
                for c in range(sample_error.shape[0]):
                    high_error_pixels = sample_error[c][sample_error[c] > thresholds[c]]
                    score = high_error_pixels.sum().item() / (high_error_pixels.numel() + 1e-8)
                    scores.append(score)
                image_scores_per_band.append(scores)

    return image_scores_per_band  # list of [17]-length lists (per image)