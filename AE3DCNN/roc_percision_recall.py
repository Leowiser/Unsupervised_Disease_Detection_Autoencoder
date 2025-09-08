
import torch
import time
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import minmax_scale
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
from vein_detection import *
from reconstruction_error import *
from utils import *
from CNN_AE_helper import *

####################################
### ROC and Recall-Percission curve

def get_reconstruction_errors_and_labels(model, dataloader, label, device,
                                         error_metric='mse', mask_after=False,
                                         remove_edges=False, mask_resize=256):
    """
    Computes per-image reconstruction errors and returns them with the associated label.

    Args:
        model: Trained autoencoder model.
        dataloader: DataLoader yielding data (and optionally paths).
        label: Label to assign to all samples (e.g. 0 for healthy, 1 for diseased).
        device: Torch device.
        error_metric: "mae" or "mse".
        mask_after: Whether to apply annotation mask after reconstruction.
        remove_edges: Whether to exclude leaf edges from the mask.
        mask_resize: Size to resize the mask to (default: 256).

    Returns:
        errors: List of per-image reconstruction errors.
        labels: Corresponding labels (same length as errors).
    """
    error_metric = error_metric.lower()
    if error_metric not in ['mae', 'mse']:
        raise ValueError("error_metric must be 'mae' or 'mse'")

    model.eval()
    reconstruction_errors = []
    labels = []

    with torch.no_grad():
        for batch_idx, (data, *_) in enumerate(dataloader):
            batch_size = data.shape[0]
            data = data.unsqueeze(1).to(device)
            recon = model(data)

            for i in range(batch_size):
                original = data[i]
                reconstructed = recon[i]

                # Compute per-pixel error map
                if error_metric == 'mse':
                    error_map = (original - reconstructed) ** 2
                else:
                    error_map = torch.abs(original - reconstructed)

                error_map = torch.mean(error_map, dim=0).cpu().numpy()  # [H, W]

                if mask_after:
                    idx = batch_idx * dataloader.batch_size + i
                    if idx >= len(dataloader.dataset.img_paths):
                        continue  # avoid out-of-bounds

                    file_path = dataloader.dataset.img_paths[idx]
                    mask_path = os.path.join(dataloader.dataset.mask_dir,
                                             os.path.basename(file_path).replace('.hdf5', '.png'))

                    mask = read_mask(mask_path)
                    mask = cv2.resize(mask, (mask_resize, mask_resize), interpolation=cv2.INTER_NEAREST)

                    if remove_edges:
                        mask = (1 - extract_full_edge(mask)) * mask

                    error_masked = error_map * mask
                    valid_pixels = error_masked[mask != 0]
                    overall_error = float(np.mean(valid_pixels)) if valid_pixels.size > 0 else 0.0
                else:
                    overall_error = float(np.mean(error_map))

                reconstruction_errors.append(overall_error)
                labels.append(label)

    return reconstruction_errors, labels

# def get_reconstruction_errors_and_labels(model, dataloader, label, device):
#     model.eval()
#     reconstruction_errors = []
#     labels = []
#     with torch.no_grad():
#         for data, _ in dataloader:
#             data = data.unsqueeze(1).to(device)  # Adjust if not already (B, C, H, W, ...)
#             recon = model(data)
#             errors = torch.mean((recon - data) ** 2, dim=[1, 2, 3, 4])  # Per-sample MSE
#             reconstruction_errors.extend(errors.cpu().numpy())
#             labels.extend([label] * data.size(0))
#     return reconstruction_errors, labels


def get_reconstruction_errors(model, dataloader, device, error_type='mae',
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
    error_type = error_type.lower()
    if error_type not in ['mae', 'mse']:
        raise ValueError("error_type must be 'mae' or 'mse'")

    model.eval()
    reconstruction_errors = []

    with torch.no_grad():
        for batch_idx, (data, *_) in enumerate(dataloader):
            batch_size = data.unsqueeze(1).shape[0]
            data = data.unsqueeze(1).to(device)
            recon = model(data)

            for i in range(batch_size):
                original = data[i].squeeze(0)         # [C, H, W]
                reconstructed = recon[i].squeeze(0)   # [C, H, W]

                # Compute per-pixel error map
                if error_type == 'mse':
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

# def get_reconstruction_errors(model, dataloader, device, error_type='mae',
#                               mask_after=False, remove_edges=False, mask_resize=256):
#     """
#     Computes per-image reconstruction error (MAE or MSE), optionally masked and with edge exclusion.

#     Args:
#         model: Trained autoencoder model.
#         dataloader: DataLoader yielding images (and optionally img_paths via dataset).
#         device: Device to perform computation on.
#         error_metric: "mae" or "mse".
#         mask_after: Whether to apply annotation mask after reconstruction.
#         remove_edges: Whether to exclude leaf edges (via annotation) from error.
#         mask_resize: Size to resize the mask to (default: 256).

#     Returns:
#         List of reconstruction errors (one per image).
#     """
#     error_type = error_type.lower()
#     if error_type not in ['mae', 'mse']:
#         raise ValueError("error_type must be 'mae' or 'mse'")

#     model.eval()
#     reconstruction_errors = []

#     with torch.no_grad():
#         for data_batch, idxs in dataloader:
#             data_batch = data_batch.unsqueeze(1).to(device)  # Shape: (B, 1, 17, 256, 256)
#             recon_batch = model(data_batch)

#             for i, (data_img, recon_img, idx) in enumerate(zip(data_batch, recon_batch, idxs)):
#                 # Remove channel dim: (1, 17, 256, 256) → (17, 256, 256)
#                 data_np = data_img.squeeze(0).cpu().numpy()
#                 recon_np = recon_img.squeeze(0).cpu().numpy()
#                 # Load disease annotation mask
#                 image_path = dataloader.dataset.img_paths[idx.item()]
#                 mask_path = os.path.join(dataloader.dataset.mask_dir, os.path.basename(image_path).replace('.hdf5', '.png'))
#                 mask = read_mask(mask_path)
#                 mask = Image.fromarray(mask).resize((256, 256), resample=Image.NEAREST)
#                 mask = np.array(mask)

#                 edge_line = extract_full_edge(mask)
#                 edge_line_switch = np.where((edge_line == 0) | (edge_line == 1), edge_line ^ 1, edge_line)
#                 mask_combined = np.multiply(edge_line_switch, mask)
#                 mask_combined = np.stack([mask_combined] * 17)  # Shape: (17, 256, 256)
                
#                 # Compute per-pixel error map
#                 if error_type == 'mse':
#                     error_map = (data_np - recon_np) ** 2
#                 else:
#                     error_map = np.abs(data_np - recon_np)
                
#                 masked_error = error_map * mask_combined
#                 valid_vals  = masked_error[mask_combined == 1]
#                 overall_error = float(np.mean(valid_vals)) if valid_vals.size else 0.0
                
#                 reconstruction_errors.append(overall_error)

#     return reconstruction_errors


def get_reconstruction_errors_stacked(model, dataloader, device):
    
    model.eval()
    reconstruction_errors = []
    with torch.no_grad():
        for data,_ in dataloader:
            data = data.to(device)
            recon = model(data)
            # Compute reconstruction error per image
            batch_errors = torch.abs((recon - data), dim=[1, 2, 3, 4])  # MAE per image
            reconstruction_errors.extend(batch_errors.cpu().numpy())  # Store in list
    return reconstruction_errors

# classify the leaves based on a threshold
# def classify_leaves_pixel_error_aggregate_label(model, dataloader, label, device, threshold):
#     model.eval()
#     image_scores =[]
#     labels = []

#     with torch.no_grad():
#         for batch,_ in dataloader:
#             batch = batch.unsqueeze(1).to(device)  # shape: [B, 1, C, H, W]
#             recon = model(batch) 
#             pixel_errors = (recon-batch).pow(2).squeeze(1).sum(dim=1)

#             for err_map in pixel_errors:
#                high_error_pixels = err_map[err_map>threshold]
#                score = high_error_pixels.sum().item()
#                image_scores.append(score)
#             labels.extend([label] * batch.size(0))

#     return image_scores, labels


def get_band_pixel_error_percentage(model, dataloader, device, threshold_band, MASK_FOLDER):
    """
    Computes a score per image by summing reconstruction errors above per-band threshold_band,
    with edge regions excluded.

    Args:
        model: Trained autoencoder.
        dataloader: DataLoader yielding (images, indices).
        label: Label to assign to all images.
        device: torch.device.
        threshold_band: np.ndarray of shape [C], per-band threshold_band.
        MASK_FOLDER: Path to annotation masks.

    Returns:
        image_scores: List of anomaly scores per image.
        labels: List of labels (same length).
    """
    model.eval()
    image_scores = []
    

    with torch.no_grad():
        for batch, idxs in dataloader:
            batch = batch.unsqueeze(1).to(device)         # [B, 1, C, H, W]
            recon = model(batch)                          # [B, 1, C, H, W]
            pixel_errors = (recon - batch).abs().squeeze(1)  # [B, C, H, W]

            for i in range(pixel_errors.shape[0]):
                
                idx = idxs[i].item()

                # Load edge mask 
                image_path = dataloader.dataset.img_paths[idx]
                mask_path = os.path.join(MASK_FOLDER, os.path.basename(image_path).replace('.hdf5', '.png'))
                mask = read_mask(mask_path)
                mask = Image.fromarray(mask).resize((256, 256), resample=Image.NEAREST)
                mask = np.array(mask)

                edge_mask = extract_full_edge(mask)              # [H, W]
                non_edge_mask = np.logical_not(edge_mask)        # [H, W]

                # Apply band-wise threshold + edge mask
                err_tensor = pixel_errors[i].cpu().numpy()       # [C, H, W]
                threshold_band = np.copy(threshold_band)
                err_tensor = err_tensor         # Slice here for subset of bands
                thr = threshold_band             # Then use the same slicing here
                band_scores = []
                total_valid_pixels = 0
                total_exceeding_pixels = 0
                for c in range(err_tensor.shape[0]):
                    valid_mask = (mask > 0) & non_edge_mask  # [H, W], boolean mask
                    total_valid_pixels += valid_mask.sum()

                    exceed_mask = (err_tensor[c] > threshold_band[c]) & valid_mask
                    total_exceeding_pixels += exceed_mask.sum()

                    band_score = total_exceeding_pixels / total_valid_pixels if total_valid_pixels > 0 else 0.0
                    band_scores.append(band_score)

                image_scores.append(band_scores) 
    return image_scores

# Latent errors
def get_feature_reconstruction_error(model, dataloader, device):
    model.eval()
    feature_errors = []

    with torch.no_grad():
        for batch, _ in dataloader:
            batch = batch.unsqueeze(1).to(device)

            # Get latent code z
            _, z = model._encode(batch)  # [B, latent_channels, D', H', W']

            # Reconstruct image and re-encode it to get z_hat
            recon = model(batch)
            _, z_hat = model._encode(recon)  # [B, latent_channels, D', H', W']

            # Compute MSE in latent space (z - z_hat)^2, averaged per sample
            sq_diff = torch.sum((z - z_hat) ** 2, dim=[1, 2, 3, 4])
            # squared norm of the original latent
            sq_norm = torch.sum(z**2, dim=[1,2,3,4])
            # Benfenati et al. feature‐error
            s_z     = sq_diff / (sq_norm + 1e-8)   # +eps to avoid div0
            feature_errors.extend(s_z.cpu().numpy())
    return feature_errors


def calculate_and_plot_roc_auc(model, healthy_dataloader, diseased_dataloader, device, threshold_band = 0, error_metric = "image_mean", error_type='mse',
                              mask_after=False, remove_edges=False, mask_resize=256, plot=True, MASK_FOLDER = "No_folder", save_path = None):
    """
    Calculate and plot the ROC curve and AUC for disease detection.
    
    Args:
        model: Trained autoencoder model
        healthy_dataloader: DataLoader containing healthy samples
        diseased_dataloader: DataLoader containing diseased samples
        device: Device to run inference on
        error_type: Type of error to calculate ('mse' or 'mae')
        mask_after: Whether to apply mask after reconstruction
        remove_edges: Whether to remove edges from error calculation
        mask_resize: Size to resize mask to
        plot: Whether to plot the ROC curve
        
    Returns:
        float: The AUC value
    """
    from sklearn.metrics import roc_curve, auc
    if error_metric == "image_mean":
        # Get reconstruction errors for healthy data
        healthy_errors = get_reconstruction_errors(
            model, healthy_dataloader, device, error_type=error_type,
            mask_after=mask_after, remove_edges=remove_edges, mask_resize=mask_resize)
        
        # Get reconstruction errors for diseased data
        diseased_errors = get_reconstruction_errors(
            model, diseased_dataloader, device, error_type=error_type,
            mask_after=mask_after, remove_edges=remove_edges, mask_resize=mask_resize)
    elif error_metric == "extreme_bands_99":
        # Get reconstruction errors for healthy data
        healthy_img = get_band_pixel_error_percentage(model, healthy_dataloader, device, threshold_band = threshold_band, MASK_FOLDER = MASK_FOLDER)
        healthy_errors = [np.mean(band_scores) for band_scores in healthy_img]
        # Get reconstruction errors for diseased data
        diseased_img = get_band_pixel_error_percentage(model, diseased_dataloader, device, threshold_band = threshold_band, MASK_FOLDER = MASK_FOLDER)
        diseased_errors = [np.mean(band_scores) for band_scores in diseased_img]
    elif error_metric == "latent":
        healthy_errors = get_feature_reconstruction_error(model, healthy_dataloader, device)
        diseased_errors = get_feature_reconstruction_error(model, diseased_dataloader, device)
    elif error_metric == "veins":
        healthy_errors = get_vein_errors(model, healthy_dataloader, device, error_type = error_type, MASK_FOLDER = MASK_FOLDER)
        diseased_errors = get_vein_errors(model, diseased_dataloader, device, error_type = error_type, MASK_FOLDER = MASK_FOLDER)

    
    # Combine errors and create labels
    all_errors = healthy_errors + diseased_errors
    all_labels = [0] * len(healthy_errors) + [1] * len(diseased_errors)
    
    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(all_labels, all_errors)
    roc_auc = auc(fpr, tpr)
    
    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Disease Detection')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.show()
    
    return roc_auc

def compare_auc_across_stages(model, healthy_dataloader, diseased_dataloaders, stages_labels, device, threshold_band = 0, error_metric = "image_mean", error_type='mse',
                             mask_after=False, remove_edges=False, mask_resize=256, MASK_FOLDER = "No_folder", save_path = None):
    """
    Calculate and plot ROC curves and AUCs for multiple disease stages and for combined disease data.
    
    Args:
        model: Trained autoencoder model
        healthy_dataloader: DataLoader containing healthy samples
        diseased_dataloaders: List of DataLoaders for different disease stages
        stages_labels: List of labels for the disease stages (e.g., ["Early", "Mid", "Late"])
        device: Device to run inference on
        error_type: Type of error to calculate ('mse' or 'mae')
        mask_after: Whether to apply mask after reconstruction
        remove_edges: Whether to remove edges from error calculation
        mask_resize: Size to resize mask to
        
    Returns:
        dict: Dictionary mapping stage labels to AUC values, includes "Combined" for overall performance
    """
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    
    # Get reconstruction errors for healthy data once
    if error_metric == "image_mean":
        # Get reconstruction errors for healthy data
        healthy_errors = get_reconstruction_errors(
            model, healthy_dataloader, device, error_type=error_type,
            mask_after=mask_after, remove_edges=remove_edges, mask_resize=mask_resize)
    elif error_metric == "extreme_bands_99":
        # Get reconstruction errors for healthy data
        healthy_img = get_band_pixel_error_percentage(model, healthy_dataloader, device, threshold_band = threshold_band, MASK_FOLDER = MASK_FOLDER)
        healthy_errors = [np.mean(band_scores) for band_scores in healthy_img]
    elif error_metric == "latent":
        healthy_errors = get_feature_reconstruction_error(model, healthy_dataloader, device)
    elif error_metric == "veins":
        healthy_errors = get_vein_errors(model, healthy_dataloader, device, error_type = error_type, MASK_FOLDER = MASK_FOLDER)

    
    plt.figure(figsize=(10, 8))
    results = {}
    
    # Plot random chance line
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Chance')
    
    # Colors for different stages
    colors = ['green', 'orange', 'red', 'purple', 'brown']

    target_tpr = 0.95
    
    # Storage for all disease errors for combined analysis
    all_disease_errors = []
    
    # Process each disease stage
    for i, (dataloader, stage) in enumerate(zip(diseased_dataloaders, stages_labels)):
        # Get reconstruction errors for this disease stage
        if error_metric == "image_mean":
        # Get reconstruction errors for healthy data
            stage_errors = get_reconstruction_errors(
                model, dataloader, device, error_type=error_type,
                mask_after=mask_after, remove_edges=remove_edges, mask_resize=mask_resize)
        elif error_metric == "extreme_bands_99":
            # Get reconstruction errors for diseased data
            diseased_img = get_band_pixel_error_percentage(model, dataloader, device, threshold_band = threshold_band, MASK_FOLDER = MASK_FOLDER)
            stage_errors = [np.mean(band_scores) for band_scores in diseased_img]
        elif error_metric == "latent":
            stage_errors = get_feature_reconstruction_error(model, dataloader, device)
        elif error_metric == "veins":
            stage_errors = get_vein_errors(model, dataloader, device, error_type = error_type, MASK_FOLDER = MASK_FOLDER)
        
        # Store for combined analysis
        all_disease_errors.extend(stage_errors)
        
        # Combine with healthy errors and create labels
        all_errors = healthy_errors + stage_errors
        all_labels = [0] * len(healthy_errors) + [1] * len(stage_errors)
        
        # Calculate ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(all_labels, all_errors)
        roc_auc = auc(fpr, tpr)
        fpr_at_95_tpr = next((f for t, f in zip(tpr, fpr) if t >= target_tpr), None)
        if fpr_at_95_tpr is not None:
            print(f"FPR@95%TPR {stage}: {fpr_at_95_tpr:.4f}")
        else:
            print("TPR never reaches 95%")
        results[stage] = roc_auc
        
        # Plot ROC curve for this stage
        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2, 
                 label=f'{stage} after infection (AUC = {roc_auc:.3f})')
    
    # Calculate and plot ROC curve for all disease stages combined
    combined_errors = healthy_errors + all_disease_errors
    combined_labels = [0] * len(healthy_errors) + [1] * len(all_disease_errors)
    
    # Calculate ROC curve and AUC for combined data
    fpr_combined, tpr_combined, _ = roc_curve(combined_labels, combined_errors)
    roc_auc_combined = auc(fpr_combined, tpr_combined)
    
    fpr_at_95_tpr = next((f for t, f in zip(tpr_combined, fpr_combined) if t >= target_tpr), None)
    if fpr_at_95_tpr is not None:
        print(f"FPR@95%TPR: {fpr_at_95_tpr:.4f}")
    else:
        print("TPR never reaches 95%")
    results["Combined"] = roc_auc_combined
    
    # Plot the combined ROC curve with a distinctive style
    plt.plot(fpr_combined, tpr_combined, color='blue', lw=3, linestyle='-',
             label=f'Overall (AUC = {roc_auc_combined:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Disease Stages')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()
    
    return results

def compare_pr_auc_across_stages(model, healthy_dataloader, diseased_dataloaders, stages_labels, device, threshold_band = 0, error_metric = "image_mean", error_type='mse',
                             mask_after=False, remove_edges=False, mask_resize=256, MASK_FOLDER = "No_folder", save_path = None):
    """
    Calculate and plot Precision-Recall curves and AUCs for multiple disease stages and combined disease data.
    
    Args:
        model: Trained autoencoder model
        healthy_dataloader: DataLoader containing healthy samples
        diseased_dataloaders: List of DataLoaders for different disease stages
        stages_labels: List of labels for the disease stages (e.g., ["Early", "Mid", "Late"])
        device: Device to run inference on
        error_type: Type of error to calculate ('mse' or 'mae')
        mask_after: Whether to apply mask after reconstruction
        remove_edges: Whether to remove edges from error calculation
        mask_resize: Size to resize mask to
        
    Returns:
        dict: Dictionary mapping stage labels to PR-AUC values, includes "Combined" for overall performance
    """
    from sklearn.metrics import precision_recall_curve, auc, average_precision_score
    import matplotlib.pyplot as plt
    
    # Get reconstruction errors for healthy data once
    if error_metric == "image_mean":
        # Get reconstruction errors for healthy data
        healthy_errors = get_reconstruction_errors(
            model, healthy_dataloader, device, error_type=error_type,
            mask_after=mask_after, remove_edges=remove_edges, mask_resize=mask_resize)
    elif error_metric == "extreme_bands_99":
        # Get reconstruction errors for healthy data
        healthy_img = get_band_pixel_error_percentage(model, healthy_dataloader, device, threshold_band = threshold_band, MASK_FOLDER = MASK_FOLDER)
        healthy_errors = [np.mean(band_scores) for band_scores in healthy_img]
    elif error_metric == "latent":
        healthy_errors = get_feature_reconstruction_error(model, healthy_dataloader, device)
    elif error_metric == "veins":
        healthy_errors = get_vein_errors(model, healthy_dataloader, device, error_type = error_type, MASK_FOLDER = MASK_FOLDER)

    
    plt.figure(figsize=(10, 8))
    results = {}
    
    # Colors for different stages
    colors = ['green', 'orange', 'red', 'purple', 'brown']
    
    # Storage for all disease errors for combined analysis
    all_disease_errors = []
    
    # Process each disease stage
    for i, (dataloader, stage) in enumerate(zip(diseased_dataloaders, stages_labels)):
        # Get reconstruction errors for this disease stage
        if error_metric == "image_mean":
        # Get reconstruction errors for healthy data
            stage_errors = get_reconstruction_errors(
                model, dataloader, device, error_type=error_type,
                mask_after=mask_after, remove_edges=remove_edges, mask_resize=mask_resize)
        elif error_metric == "extreme_bands_99":
            # Get reconstruction errors for diseased data
            diseased_img = get_band_pixel_error_percentage(model, dataloader, device, threshold_band = threshold_band, MASK_FOLDER = MASK_FOLDER)
            stage_errors = [np.mean(band_scores) for band_scores in diseased_img]
        elif error_metric == "latent":
            stage_errors = get_feature_reconstruction_error(model, dataloader, device)
        elif error_metric == "veins":
            stage_errors = get_vein_errors(model, dataloader, device, error_type = error_type, MASK_FOLDER = MASK_FOLDER)
        
        # Store for combined analysis
        all_disease_errors.extend(stage_errors)
        
        # Combine with healthy errors and create labels
        all_errors = healthy_errors + stage_errors
        all_labels = [0] * len(healthy_errors) + [1] * len(stage_errors)
        
        # Calculate PR curve and average precision
        precision, recall, _ = precision_recall_curve(all_labels, all_errors)
        # For PR curves, we need to flip the x and y when calculating AUC
        pr_auc = auc(recall, precision)
        ap_score = average_precision_score(all_labels, all_errors)
        results[stage] = pr_auc
        
        # Plot PR curve for this stage
        plt.plot(recall, precision, color=colors[i % len(colors)], lw=2, 
                 label=f'{stage} Infected (PR-AUC = {pr_auc:.3f}, AP = {ap_score:.3f})')
    
    # Calculate and plot PR curve for all disease stages combined
    combined_errors = healthy_errors + all_disease_errors
    combined_labels = [0] * len(healthy_errors) + [1] * len(all_disease_errors)
    
    # Calculate PR curve and AUC for combined data
    precision_combined, recall_combined, _ = precision_recall_curve(combined_labels, combined_errors)
    pr_auc_combined = auc(recall_combined, precision_combined)
    ap_combined = average_precision_score(combined_labels, combined_errors)
    results["Combined"] = pr_auc_combined
    
    # Calculate combined baseline (no skill classifier that always predicts the positive class)
    combined_baseline = sum(combined_labels) / len(combined_labels)
    
    # Plot the combined PR curve with a distinctive style
    plt.plot(recall_combined, precision_combined, color='blue', lw=3, linestyle='-',
             label=f'Combined (PR-AUC = {pr_auc_combined:.3f}, AP = {ap_combined:.3f})')
    
    # Add a dotted line for the baseline (no skill classifier)
    plt.axhline(y=combined_baseline, color='navy', linestyle='--', alpha=0.8,
               label=f'No Skill Classifier ({combined_baseline:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves for Disease Stages')
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()
    
    return results









####################################################################################################################################
###################### STACKED VERSION
####################################################################################################################################


def get_reconstruction_errors_stacked(model, dataloader, device, error_type='mae',
                                      mask_after=False, remove_edges=False, mask_resize=256):
    error_type = error_type.lower()
    if error_type not in ['mae', 'mse']:
        raise ValueError("error_type must be 'mae' or 'mse'")

    model.eval()
    reconstruction_errors = []

    with torch.no_grad():
        for batch_idx, (data, *_) in enumerate(dataloader):
            data = data.to(device)
            recon = model(data)

            for i in range(data.size(0)):
                orig = data[i]            # [C, D, H, W]
                rec = recon[i]            # [C, D, H, W]
                if error_type == 'mse':
                    err_map = (orig - rec) ** 2
                else:
                    err_map = torch.abs(orig - rec)

                # mean over channels and bands → [H, W]
                err_map = torch.mean(err_map, dim=[0,1]).cpu().numpy()

                if mask_after:
                    idx = batch_idx * dataloader.batch_size + i
                    if idx >= len(dataloader.dataset.img_paths):
                        break
                    img_path = dataloader.dataset.img_paths[idx]
                    mask = read_mask(os.path.join(dataloader.dataset.mask_dir,
                                                  os.path.basename(img_path).replace('.hdf5','.png')))
                    mask = cv2.resize(mask, (mask_resize, mask_resize), interpolation=cv2.INTER_NEAREST)
                    if remove_edges:
                        mask = (1 - extract_full_edge(mask)) * mask
                    valid = err_map[mask!=0]
                    overall = float(np.mean(valid)) if valid.size>0 else 0.0
                else:
                    overall = float(np.mean(err_map))
                reconstruction_errors.append(overall)
    return reconstruction_errors


def get_feature_reconstruction_errors_stacked(model, dataloader, device):
    model.eval()
    feature_errors = []
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            _, z = model._encode(data)
            recon = model(data)
            _, z_hat = model._encode(recon)
            errs = torch.mean((z - z_hat)**2, dim=[1,2,3,4])
            feature_errors.extend(errs.cpu().numpy())
    return feature_errors


def get_stacked_vein_errors(model, dataloader, device, MASK_FOLDER,
                             error_type="mse", steger_sigma=1.0, line_thresh=0.023):
    model.eval()
    vein_errors = []
    with torch.no_grad():
        for data, idxs in dataloader:
            data = data.to(device)   # [B,C,D,H,W]
            recon = model(data)
            for i in range(data.size(0)):
                orig = data[i].cpu().numpy()  # [C,D,H,W]
                rec = recon[i].cpu().numpy()
                spec = orig[0]                # [D,H,W]
                resp = apply_steger_to_hsi(spec, sigma=steger_sigma)
                lm = np.mean(resp, axis=0)
                nl = minmax_scale(lm.ravel()).reshape(lm.shape)
                bin_mask = (nl>line_thresh).astype(np.uint8)
                idx = idxs[i].item()
                mask = read_mask(os.path.join(MASK_FOLDER,
                                              os.path.basename(dataloader.dataset.img_paths[idx]).replace('.hdf5','.png')))
                mask = Image.fromarray(mask).resize((bin_mask.shape[1], bin_mask.shape[0]), resample=Image.NEAREST)
                mask = np.array(mask)
                edge = extract_full_edge(mask)
                vein = edge^1
                cm = vein * bin_mask
                C,D,H,W = orig.shape
                mask_stack = np.broadcast_to(cm, (C,D,H,W))
                if error_type.lower()=="mse":
                    err = (orig-rec)**2
                else:
                    err = np.abs(orig-rec)
                masked = err*mask_stack
                total = mask_stack.sum()
                vein_errors.append(float(masked.sum()/total) if total>0 else 0.0)
    return vein_errors


def get_stacked_error_thresholds(model, dataloader, device, MASK_FOLDER, quantile=0.75, error_metric="MAE", max_samples_per_band=10000):
    """
    Compute per-channel-and-band error thresholds for stacked inputs with edge exclusion.
    Samples up to max_samples_per_band pixels per channel-band per batch to limit memory.

    Returns:
        thresholds: np.ndarray of shape [C, D] containing the quantile-based threshold for each channel and derivative band.
    """
    model.eval()
    errors = {}  # dict (c,d) -> list of sampled pixel errors

    with torch.no_grad():
        for data, idxs in dataloader:
            data = data.to(device)
            recon = model(data)
            batch_size = data.size(0)
            for i in range(batch_size):
                orig = data[i].cpu().numpy()  # [C, D, H, W]
                rec = recon[i].cpu().numpy()
                # compute error map
                if error_metric.upper() == "MSE":
                    err = (orig - rec) ** 2
                else:
                    err = np.abs(orig - rec)
                # load mask & compute non-edge mask
                idx = idxs[i].item()
                mask = read_mask(os.path.join(MASK_FOLDER,
                                              os.path.basename(dataloader.dataset.img_paths[idx]).replace('.hdf5', '.png')))
                mask = Image.fromarray(mask).resize((err.shape[2], err.shape[3]), resample=Image.NEAREST)
                mask = np.array(mask)
                non_edge = ~extract_full_edge(mask)
                C, D, H, W = err.shape
                # sample pixels for each channel-band
                for c in range(C):
                    for d in range(D):
                        pix = err[c, d][non_edge]
                        if pix.size:
                            if pix.size > max_samples_per_band:
                                # random subsample
                                idxs_sample = np.random.choice(pix.size, max_samples_per_band, replace=False)
                                pix = pix[idxs_sample]
                            errors.setdefault((c, d), []).append(pix)
    # determine array dimensions
    Cs = max(k[0] for k in errors) + 1 if errors else 0
    Ds = max(k[1] for k in errors) + 1 if errors else 0
    thresholds = np.zeros((Cs, Ds), dtype=float)
    for (c, d), lists in errors.items():
        all_pix = np.concatenate(lists)
        thresholds[c, d] = np.quantile(all_pix, quantile)
    return thresholds



def get_stacked_reconstruction_scores(model, dataloader, device, thresholds, MASK_FOLDER, error_metric="MAE"):
    model.eval()
    image_scores = []
    with torch.no_grad():
        for data, idxs in dataloader:
            data = data.to(device)
            recon = model(data)
            for i in range(data.size(0)):
                orig = data[i].cpu().numpy()
                rec = recon[i].cpu().numpy()
                err = (orig-rec)**2 if error_metric.upper()=="MSE" else np.abs(orig-rec)
                idx = idxs[i].item()
                mask = read_mask(os.path.join(MASK_FOLDER,
                                              os.path.basename(dataloader.dataset.img_paths[idx]).replace('.hdf5','.png')))
                mask = Image.fromarray(mask).resize((err.shape[2], err.shape[3]), resample=Image.NEAREST)
                mask = np.array(mask)
                non_edge = ~extract_full_edge(mask)
                C,D,H,W = err.shape
                scores = np.zeros((C,D),float)
                for c in range(C):
                    for d in range(D):
                        vals = err[c,d][non_edge]
                        scores[c,d] = float(np.mean(vals>thresholds[c,d])) if vals.size else 0.0
                image_scores.append(scores)
    return image_scores


def compare_stacked_auc_across_stages(
    model,
    healthy_dataloader,
    diseased_dataloaders,
    stages_labels,
    device,
    thresholds=None,
    error_metric="image_mean",
    error_type='mse',
    MASK_FOLDER="No_folder",
    save_path=None
):
    def flatten(scores): return [float(np.mean(s)) for s in scores]

    # healthy
    if error_metric=="image_mean":
        healthy = flatten(get_reconstruction_errors_stacked(model, healthy_dataloader, device,
                                                            error_type=error_type))
    elif error_metric=="extreme_bands_99":
        healthy = flatten(get_stacked_reconstruction_scores(model, healthy_dataloader, device, thresholds, MASK_FOLDER))
    elif error_metric=="latent":
        healthy = get_feature_reconstruction_errors_stacked(model, healthy_dataloader, device)
    else:
        healthy = get_stacked_vein_errors(model, healthy_dataloader, device, MASK_FOLDER, error_type=error_type)

    plt.figure(figsize=(10,8))
    plt.plot([0,1],[0,1],'--',color='gray')

    target_tpr = 0.95
    results={}
    all_scores=[]
    colors=['green','orange','red','purple','brown']
    for i,(dl,lab) in enumerate(zip(diseased_dataloaders,stages_labels)):
        if error_metric in ["image_mean","extreme_bands_99"]:
            scores = flatten(get_reconstruction_errors_stacked(model, dl, device,
                    error_type=error_type)) if error_metric=="image_mean" else flatten(
                    get_stacked_reconstruction_scores(model, dl, device, thresholds, MASK_FOLDER))
        elif error_metric=="latent":
            scores = get_feature_reconstruction_errors_stacked(model, dl, device)
        else:
            scores = get_stacked_vein_errors(model, dl, device, MASK_FOLDER, error_type=error_type)
        all_scores+=scores
        y_true = np.array([0]*len(healthy)+[1]*len(scores))
        fpr,tpr,_ = roc_curve(y_true,np.array(healthy+scores))
        
        fpr_at_95_tpr = next((f for t, f in zip(tpr, fpr) if t >= target_tpr), None)
        if fpr_at_95_tpr is not None:
            print(f"FPR@95%TPR {lab}: {fpr_at_95_tpr:.4f}")
        else:
            print("TPR never reaches 95%")
        aucv=auc(fpr,tpr)
        results[lab]=aucv
        plt.plot(fpr,tpr,color=colors[i],lw=2,label=f"{lab} (AUC={aucv:.3f})")
    # combined
    y_true=np.array([0]*len(healthy)+[1]*len(all_scores))
    fpr,tpr,_=roc_curve(y_true,np.array(healthy+all_scores))
    aucv=auc(fpr,tpr)
    
    fpr_at_95_tpr = next((f for t, f in zip(tpr, fpr) if t >= target_tpr), None)
    if fpr_at_95_tpr is not None:
        print(f"FPR@95%TPR: {fpr_at_95_tpr:.4f}")
    else:
        print("TPR never reaches 95%")
    results['Combined']=aucv
    plt.plot(fpr,tpr,color='blue',lw=3,label=f"Combined (AUC={aucv:.3f})")
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curves Stacked')
    plt.legend(loc='lower right')
    if save_path: plt.savefig(save_path,dpi=150,bbox_inches='tight')
    plt.show()
    return results

def compare_stacked_pr_auc_across_stages(
    model,
    healthy_dataloader,
    diseased_dataloaders,
    stages_labels,
    device,
    thresholds=None,
    error_metric="image_mean",
    error_type='mse',
    MASK_FOLDER="No_folder",
    save_path=None
):
    """
    Calculate and plot Precision–Recall curves and PR-AUCs for stacked inputs across stages.
    Returns a dict mapping each stage (and “Combined”) to its PR-AUC.
    """
    from sklearn.metrics import precision_recall_curve, auc, average_precision_score
    import matplotlib.pyplot as plt

    def flatten(scores): 
        # collapse per-(channel,band) scores into one scalar per image
        return [float(np.mean(s)) for s in scores]

    # 1) Compute healthy scores
    if error_metric == "image_mean":
        healthy = flatten(
            get_reconstruction_errors_stacked(
                model, healthy_dataloader, device, error_type=error_type
            )
        )
    elif error_metric == "extreme_bands_99":
        if thresholds is None:
            thresholds = get_stacked_error_thresholds(
                model, healthy_dataloader, device, MASK_FOLDER
            )
        healthy = flatten(
            get_stacked_reconstruction_scores(
                model, healthy_dataloader, device, thresholds, MASK_FOLDER
            )
        )
    elif error_metric == "latent":
        healthy = get_feature_reconstruction_errors_stacked(
            model, healthy_dataloader, device
        )
    else:  # veins
        healthy = get_stacked_vein_errors(
            model, healthy_dataloader, device, MASK_FOLDER, error_type=error_type
        )

    plt.figure(figsize=(10, 8))
    results = {}
    all_scores = []
    colors = ['green','orange','red','purple','brown']

    # 2) Each disease stage
    for i, (loader, label) in enumerate(zip(diseased_dataloaders, stages_labels)):
        if error_metric == "image_mean":
            scores = flatten(
                get_reconstruction_errors_stacked(
                    model, loader, device, error_type=error_type
                )
            )
        elif error_metric == "extreme_bands_99":
            scores = flatten(
                get_stacked_reconstruction_scores(
                    model, loader, device, thresholds, MASK_FOLDER
                )
            )
        elif error_metric == "latent":
            scores = get_feature_reconstruction_errors_stacked(
                model, loader, device
            )
        else:
            scores = get_stacked_vein_errors(
                model, loader, device, MASK_FOLDER, error_type=error_type
            )

        all_scores += scores

        y_true = np.array([0]*len(healthy) + [1]*len(scores))
        y_scores = np.array(healthy + scores)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        ap = average_precision_score(y_true, y_scores)
        results[label] = pr_auc

        plt.plot(
            recall,
            precision,
            color=colors[i % len(colors)],
            lw=2,
            label=f'{label} (PR-AUC={pr_auc:.3f}, AP={ap:.3f})'
        )

    # 3) Combined across all stages
    y_true = np.array([0]*len(healthy) + [1]*len(all_scores))
    y_scores = np.array(healthy + all_scores)
    precision_c, recall_c, _ = precision_recall_curve(y_true, y_scores)
    pr_auc_c = auc(recall_c, precision_c)
    ap_c = average_precision_score(y_true, y_scores)
    baseline = sum(y_true) / len(y_true)
    results["Combined"] = pr_auc_c

    plt.plot(
        recall_c,
        precision_c,
        color='blue',
        lw=3,
        label=f'Combined (PR-AUC={pr_auc_c:.3f}, AP={ap_c:.3f})'
    )
    plt.axhline(
        baseline,
        linestyle='--',
        color='gray',
        label=f'No‐Skill ({baseline:.3f})'
    )

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision–Recall Curves Across Stages (Stacked)')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()

    return results