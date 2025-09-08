import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np 
import joblib
from torch.utils.data import DataLoader
from preprocessing import *
from utils import *
from datasets import *
from CNN_AE_helper import *
from CNN3d import *
from vein_detection import *
from torchvision.transforms import v2
from scipy.ndimage import binary_erosion
from sklearn.manifold import TSNE
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, average_precision_score,
    precision_recall_fscore_support, balanced_accuracy_score, matthews_corrcoef,
    classification_report
)
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import umap.umap_ as UMAP
import os
from sklearn.metrics import PrecisionRecallDisplay

##############################
### Visualization Reconstruction error

def visu_reconstruction_error(model, dataloader, device, save_path=None, select_img=0,
                              RGB_bands=[9,4,6], error_type='mae', vmax=0.2, minmax = False):
    """
    Visualizes reconstruction error for a single image in a batch, consistent with the batch version.

    Args:
        model: Trained autoencoder model
        dataloader: DataLoader containing the image
        device: Torch device
        save_path: If provided, saves the output visualization to this path
        select_img: Index of the image within the batch to visualize
        RGB_bands: List of 3 integers specifying the RGB channels
        error_type: 'mse' or 'mae' for reconstruction error calculation
        vmax: Maximum value for colorbar in error heatmap
    """
    error_type = error_type.lower()
    if error_type not in ['mse', 'mae']:
        raise ValueError("error_type must be either 'mse' or 'mae'")

    model.eval()

    with torch.no_grad():
        for data, _ in dataloader:
            data = data.unsqueeze(1).to(device)
            recon = model(data)
            break  # Take only the first batch

    original = data[select_img].squeeze().cpu().numpy() # (C, H, W)
    reconstructed = recon[select_img].squeeze().cpu().numpy()  # (C, H, W)
    
    # Compute reconstruction error map
    if error_type == 'mse':
        error_map = (original - reconstructed) ** 2
    else:  # mae
        error_map = np.abs(original - reconstructed)

    error_map_aggregated = np.mean(error_map, axis=0)  # (H, W)
    if minmax:
        # Min-max scale the error map to [0, 1]
        min_val = error_map_aggregated.min()
        max_val = error_map_aggregated.max()
        if max_val > min_val:  # avoid division by zero
            error_map_aggregated = (error_map_aggregated - min_val) / (max_val - min_val)
        else:
            error_map_aggregated = np.zeros_like(error_map_aggregated)

    overall_error = np.mean(error_map_aggregated)

    # Plotting with consistent layout
    fig = plt.figure(figsize=(12, 4))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05])

    # Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original[RGB_bands, :, :].transpose(1, 2, 0))
    ax1.set_title("Original")
    ax1.axis('off')

    # Reconstructed image
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(reconstructed[RGB_bands, :, :].transpose(1, 2, 0))
    ax2.set_title("Reconstructed")
    ax2.axis('off')

    # Error map
    ax3 = fig.add_subplot(gs[0, 2])
    im = ax3.imshow(error_map_aggregated, cmap='hot', vmin=0, vmax=vmax)
    ax3.set_title(f"Reconstruction Error\n{error_type.upper()}: {overall_error:.4f}")
    ax3.axis('off')

    # Colorbar
    cbar_ax = fig.add_subplot(gs[0, 3])
    plt.colorbar(im, cax=cbar_ax)

    # Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()

def visu_reconstruction_error_edge_removed(model, dataloader, device, save_path=None, error_type='mae', vmax=0.2, minmax=False):
    """
    Visualizes reconstruction error for a single image in a batch, with optional mask and edge removal.

    Args:
        model: Trained autoencoder model
        dataloader: DataLoader containing the image
        device: Torch device
        save_path: If provided, saves the output visualization to this path
        select_img: Index of the image within the batch to visualize
        RGB_bands: List of 3 integers specifying the RGB channels
        error_type: 'mse' or 'mae' for reconstruction error calculation
        vmax: Maximum value for colorbar in error heatmap
        minmax: Whether to apply min-max normalization to the error map
        mask_after: Whether to apply annotation mask after reconstruction
        remove_edges: Whether to exclude leaf edges (via annotation) from error
        mask_resize: Size to resize the mask to (default: 256)
    """
    import os
    import cv2
    import numpy as np
    import torch
    from matplotlib import pyplot as plt

    # Validate error type
    error_type = error_type.lower()
    if error_type not in ['mse', 'mae']:
        raise ValueError("error_type must be either 'mse' or 'mae'")

    reconstruction_errors = []
    
    model.eval()
    with torch.no_grad():
        for data_batch, idxs in dataloader:
            data_batch = data_batch.unsqueeze(1).to(device)  # Shape: (B, 1, 17, 256, 256)
            recon_batch = model(data_batch)
            break

        for i, (data_img, recon_img, idx) in enumerate(zip(data_batch, recon_batch, idxs)):
                # Remove channel dim: (1, 17, 256, 256) → (17, 256, 256)
                data_np = data_img.squeeze(0).cpu().numpy()
                recon_np = recon_img.squeeze(0).cpu().numpy()
                # Apply Steger to original image
                
                # Load disease annotation mask
                image_path = dataloader.dataset.img_paths[idx.item()]
                mask_path = os.path.join(MASK_FOLDER, os.path.basename(image_path).replace('.hdf5', '.png'))
                mask = read_mask(mask_path)
                mask = Image.fromarray(mask).resize((256, 256), resample=Image.NEAREST)
                mask = np.array(mask)

                edge_line = extract_full_edge(mask)
                edge_line_switch = np.where((edge_line == 0) | (edge_line == 1), edge_line ^ 1, edge_line)
                mask_combined = np.multiply(edge_line_switch, mask)
                mask_combined = np.stack([mask_combined] * 17)  # Shape: (17, 256, 256)

                # Compute reconstruction error within the masked area
                error_map = np.abs(data_np - recon_np)  # Shape: (17, 256, 256)
                masked_error = error_map * mask_combined
                err2d  = np.mean(masked_error, axis=0)

                if minmax:
                    e_min, e_max = err2d.min(), err2d.max()
                    if e_max > e_min:
                        err2d = (err2d - e_min) / (e_max - e_min)
                    else:
                        # all values are identical → set to zero (or to 1, if you prefer)
                        err2d = np.zeros_like(err2d)
                
                valid_vals  = masked_error[mask_combined == 1]
                mean_error_in_mask = float(np.mean(valid_vals)) if valid_vals.size else 0.0
                                               
                
                # Visualize RGB composites (using bands 9, 3, 5)
                rgb = data_np[[9, 4, 6]]
                #rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
                rgb = rgb.transpose(1, 2, 0)
                rgb_reconst = recon_np[[9, 4, 6]]
                #rgb_reconst = (rgb_reconst - rgb_reconst.min()) / (rgb_reconst.max() - rgb_reconst.min())
                rgb_reconst = rgb_reconst.transpose(1, 2, 0)


                # Plotting with consistent layout
                fig = plt.figure(figsize=(12, 4))
                gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05])
                # Original image
                ax1 = fig.add_subplot(gs[0, 0])
                ax1.imshow(rgb)
                ax1.set_title("Original Image (RGB)")
                ax1.axis('off')

                # Reconstructed image
                ax2 = fig.add_subplot(gs[0, 1])
                ax2.imshow(rgb_reconst)
                ax2.set_title("Reconstructed Image (RGB)")
                ax2.axis('off')

                # Error map
                ax3 = fig.add_subplot(gs[0, 2])
                im = ax3.imshow(err2d, cmap='hot', vmin=0, vmax=vmax)
                ax3.set_title(f"Reconstruction Error MAE: {mean_error_in_mask:.4f}")
                ax3.axis('off')

                # Colorbar
                cbar_ax = fig.add_subplot(gs[0, 3])
                plt.colorbar(im, cax=cbar_ax)
                # Save or show
                if save_path:
                    plt.savefig(f'{save_path}_img{i}.png', bbox_inches='tight', dpi=150)
                plt.show()


def visu_error_per_band(model, dataloader, device,
                        save_path=None,
                        error_type='mae',
                        vmax=None,
                        minmax=False):
    """
    Visualizes reconstruction error per spectral band for a single image in a batch,
    with optional edge removal masking.
    """
    # Validate error type
    error_type = error_type.lower()
    if error_type not in ['mse', 'mae']:
        raise ValueError("error_type must be either 'mse' or 'mae'")
    
    model.eval()
    with torch.no_grad():
        # grab one batch
        for data_batch, idxs in dataloader:
            # ensure shape (B, 1, C, H, W)
            data_batch = data_batch.unsqueeze(1).to(device)
            recon_batch = model(data_batch)
            break

        # just take the first image in the batch
        data_img = data_batch[0].squeeze(0).cpu().numpy()    # (C, H, W)
        recon_img = recon_batch[0].squeeze(0).cpu().numpy()  # (C, H, W)
        idx = idxs[0].item()

        # load & prepare mask
        image_path = dataloader.dataset.img_paths[idx]
        mask_path = os.path.join(MASK_FOLDER,
                                 os.path.basename(image_path).replace('.hdf5','.png'))
        mask = read_mask(mask_path)
        mask = Image.fromarray(mask).resize((data_img.shape[1], data_img.shape[2]),
                                            resample=Image.NEAREST)
        mask = np.array(mask)
        edge_line = extract_full_edge(mask)
        # flip edge bits (assuming 0/1)
        edge_line_switch = np.where((edge_line == 0) | (edge_line == 1),
                                    edge_line ^ 1, edge_line)
        mask_combined = np.multiply(edge_line_switch, mask)
        # broadcast to all bands
        mask_stack = np.stack([mask_combined]*data_img.shape[0], axis=0)

        # compute error map
        if error_type == 'mse':
            error_map = (data_img - recon_img)**2
        else:
            error_map = np.abs(data_img - recon_img)

        # apply mask
        masked_error = error_map * mask_stack

        # optionally min–max normalize each band individually (within mask)
        if minmax:
            for b in range(masked_error.shape[0]):
                band = masked_error[b]
                mvals = band[mask_stack[b]==1]
                if mvals.size:
                    mn, mx = mvals.min(), mvals.max()
                    if mx > mn:
                        masked_error[b] = (band - mn) / (mx - mn)
                    else:
                        masked_error[b] = np.zeros_like(band)

        # plot a grid of per-band error maps
        n_bands = masked_error.shape[0]
        n_cols = 5
        n_rows = int(np.ceil(n_bands / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(n_cols*3, n_rows*3),
                                 constrained_layout=True)
        axes = axes.flatten()

        for b in range(n_bands):
            ax = axes[b]
            im = ax.imshow(masked_error[b], cmap='hot', vmin=0, vmax=vmax)
            ax.set_title(f'Band {b}')
            ax.axis('off')

        # turn off any extra axes
        for ax in axes[n_bands:]:
            ax.axis('off')

        # common colorbar
        cbar = fig.colorbar(im, ax=axes[:n_bands].tolist(),
                            orientation='vertical',
                            fraction=0.02, pad=0.01)
        cbar.set_label(f'{error_type.upper()} error')

        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()



def visualize_pixel_spectra_combined_3d(model, dataloader, device, n_pixels=5, error_type='mae',
                                        mask_after=False, remove_edges=False, mask_resize=256):
    """
    Visualizes spectra of multiple pixels with highest reconstruction errors on a single plot for 3D CNN autoencoders.
    
    Parameters:
        model (torch.nn.Module): Trained 3D CNN autoencoder
        dataloader (torch.utils.data.DataLoader): Dataloader yielding (image, label) pairs
        device (torch.device): Device for inference
        n_pixels (int): Number of top-error pixels to visualize
        error_type (str): 'mae' or 'mse'
        mask_after (bool): Whether to apply ROI mask
        remove_edges (bool): Remove edges from the mask
        mask_resize (int): Resize mask to this size (assumes square)
    """
    
    # === Get first image and path ===
    for batch_idx, (batch, _) in enumerate(dataloader):
        image = batch[0]  # Shape: (D, H, W)
        image = image.unsqueeze(0).unsqueeze(0)  # → (1, 1, D, H, W)
        if mask_after:
            file_path = dataloader.dataset.img_paths[batch_idx * dataloader.batch_size]
            file_name = os.path.splitext(os.path.basename(file_path))[0]
        break

    model.eval()
    with torch.no_grad():
        reconstruction = model(image.to(device)).squeeze().cpu()  # → (D, H, W)

    original = image.squeeze().cpu()  # → (D, H, W)

    # === Compute Error Map ===
    if error_type == 'mae':
        error_map = torch.abs(original - reconstruction).numpy()  # (D, H, W)
    else:
        error_map = ((original - reconstruction) ** 2).numpy()

    error = np.mean(error_map, axis=0)  # → (H, W)

    # === Optional: Mask ===
    if mask_after:
        mask_path = os.path.join(dataloader.dataset.mask_dir, file_name + '.png')
        mask = read_mask(mask_path)
        mask = np.where(mask == 2, dataloader.dataset.mask_method, mask)
        mask = cv2.resize(mask, (mask_resize, mask_resize), interpolation=cv2.INTER_LINEAR)
        if remove_edges:
            mask = (1 - extract_edge(mask, edge_width=4)) * mask
        error *= mask  # apply mask to error map

    # === Get top N pixel indices ===
    flat_error = error.flatten()
    if mask_after:
        valid_indices = np.where(mask.flatten() > 0)[0]
        valid_errors = flat_error[valid_indices]
        if len(valid_errors) > 0:
            top_relative = np.argsort(valid_errors)[-n_pixels:][::-1]
            top_indices = valid_indices[top_relative]
        else:
            top_indices = np.argsort(flat_error)[-n_pixels:][::-1]
    else:
        top_indices = np.argsort(flat_error)[-n_pixels:][::-1]

    H, W = error.shape
    y_coords = top_indices // W
    x_coords = top_indices % W

    #  RGB Rendering 
    rgb_bands = [9,3,5]  # Modify based on your bands and their indices
    rgb_img = original[rgb_bands, :, :].numpy().transpose(1, 2, 0)
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)

    #  Wavelengths 
    if hasattr(dataloader.dataset, 'selected_bands') and dataloader.dataset.selected_bands is not None:
        wavelengths = np.array(dataloader.dataset.selected_bands)
    else:
        wavelengths = dataloader.dataset.wlens[dataloader.dataset.wlens >= 430]

    #  Plot
    fig = plt.figure(figsize=(16, 10))

    # RGB Image
    ax_rgb = plt.subplot2grid((2, 2), (0, 0))
    ax_rgb.imshow(rgb_img)
    ax_rgb.set_title('RGB Image with Top Error Pixels')
    ax_rgb.set_xticks([])
    ax_rgb.set_yticks([])

    colors = plt.cm.tab10(np.linspace(0, 1, n_pixels))
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        ax_rgb.add_patch(Rectangle((x-2, y-2), 5, 5, edgecolor=colors[i], facecolor='none', linewidth=1.5))
        ax_rgb.text(x+3, y+3, str(i+1), color=colors[i], fontsize=10, weight='bold')

    # Error Map
    ax_err = plt.subplot2grid((2, 2), (0, 1))
    im = ax_err.imshow(error, cmap='hot', vmin=0, vmax=0.2)
    ax_err.set_title('Reconstruction Error Map' + (' (Masked)' if mask_after else ''))
    plt.colorbar(im, ax=ax_err)
    ax_err.set_xticks([])
    ax_err.set_yticks([])

    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        ax_err.add_patch(Rectangle((x-2, y-2), 5, 5, edgecolor=colors[i], facecolor='none', linewidth=1.5))
        ax_err.text(x+3, y+3, str(i+1), color=colors[i], fontsize=10, weight='bold')

    # Spectral Plots
    ax_spec = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        orig_spectrum = original[:, y, x].numpy()
        recon_spectrum = reconstruction[:, y, x].numpy()
        err_val = error[y, x]
        ax_spec.plot(wavelengths, orig_spectrum, color=colors[i], linestyle='-', label=f"Pixel {i+1} ({x},{y}) Err: {err_val:.4f}")
        ax_spec.plot(wavelengths, recon_spectrum, color=colors[i], linestyle='--', alpha=0.7)

    # Top error bands (vertical lines)
    avg_band_error = np.mean([error_map[:, y, x] for y, x in zip(y_coords, x_coords)], axis=0)
    top_bands = np.argsort(avg_band_error)[-3:]
    for idx in top_bands:
        ax_spec.axvline(wavelengths[idx], color='gray', linestyle=':', linewidth=1)
        ax_spec.text(wavelengths[idx], 0.95, f"{wavelengths[idx]:.1f}nm", rotation=90,
                     ha='right', va='bottom', fontsize=8, color='black',
                     transform=ax_spec.get_xaxis_transform())

    ax_spec.set_title("Spectra of Top Error Pixels\n(solid=original, dashed=reconstructed)")
    ax_spec.set_xlabel("Wavelength (nm)")
    ax_spec.set_ylabel("Reflectance / Intensity")
    ax_spec.set_ylim(0, 1)
    ax_spec.grid(True, linestyle='--', alpha=0.5)
    ax_spec.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

    return error, x_coords, y_coords



####################################################
######## Extreme pixel error rate per band #########
####################################################

# Visualize the Individuala pixels that exceed the threshold in at least one spectra
def visualize_threshold_exceedance(model, dataloader, device, thresholds, save_path, band_colors=None, select_img=0, bands_to_show=None):
    """
    Visualize pixels where reconstruction error exceeds band-wise thresholds.

    - band_colors: dict {band_idx: [R, G, B]} in [0-1] range.
    - bands_to_show: list of band indices to visualize (default: all).
    """

    model.eval()

    with torch.no_grad():
        for batch, _ in dataloader:
            batch = batch.unsqueeze(1).to(device)  # [B, 1, C, H, W]
            recon = model(batch)
            break  # Just use one batch/image for visualization

    input_img = batch[select_img, 0].cpu().numpy()  # [C, H, W]
    recon_img = recon[select_img, 0].cpu().numpy()  # [C, H, W]
    error = np.abs(input_img - recon_img)  # [C, H, W]

    C, H, W = error.shape
    if bands_to_show is None:
        bands_to_show = list(range(C))

    if band_colors is None:
        # Default: red, green, blue, yellow, magenta, cyan, white, orange...
        default_colors = [[1, 0, 0]
            # [1, 0, 0], [0, 1, 0], [0, 0, 1],
            # [1, 1, 0], [1, 0, 1], [0, 1, 1],
            # [1, 1, 1], [1, 0.5, 0]
        ]
        band_colors = {b: default_colors[i % len(default_colors)] for i, b in enumerate(bands_to_show)}

    # Build RGB highlight map
    rgb_mask = np.zeros((H, W, 3), dtype=np.float32)

    for b in bands_to_show:
        threshold = thresholds[b]
        mask = error[b] > threshold  # [H, W]
        color = np.array(band_colors.get(b, [0.5, 0.5, 0.5]))  # Default: gray
        rgb_mask[mask] += color

    rgb_mask = np.clip(rgb_mask, 0, 1)  # Ensure valid RGB range

    # Display
    plt.figure(figsize=(6, 6))
    plt.imshow(rgb_mask)
    plt.title("Pixels Above Threshold (Color-coded by Band)")
    plt.axis('off')
    plt.show()


def visualize_threshold_exceedance_edge_removal(
    model,
    dataloader,
    device,
    thresholds,
    save_path,
    MASK_FOLDER,
    band_colors=None,
    select_img=0,
    bands_to_show=None
):
    """
    Visualize pixels where reconstruction error exceeds per-band thresholds,
    excluding edge regions from the mask, and annotate band stats below.
    """
    import os
    import numpy as np
    import torch
    from PIL import Image
    from matplotlib import pyplot as plt

    model.eval()
    # Grab exactly one batch
    with torch.no_grad():
        imgs, idxs = next(iter(dataloader))            # imgs: [B, C, H, W]
        imgs = imgs.to(device)
        recons = model(imgs.unsqueeze(1)).squeeze(1)    # model expects [B,1,C,H,W]

    # Pick image
    data_np  = imgs[select_img].cpu().numpy()         # [C, H, W]
    recon_np = recons[select_img].cpu().numpy()       # [C, H, W]
    idx       = idxs[select_img].item()

    # Load + resize the annotation mask, then exclude edges
    mask_path = os.path.join(
        MASK_FOLDER,
        os.path.basename(dataloader.dataset.img_paths[idx]).replace('.hdf5','.png')
    )
    mask = Image.open(mask_path).convert('L')
    C, H, W = data_np.shape
    mask = mask.resize((W, H), resample=Image.NEAREST)
    mask = np.array(mask)
    edge = extract_full_edge(mask)
    valid_mask = 1 - edge  


    # Broadcast that mask across all C bands
    mask_stack = np.broadcast_to(valid_mask, (C, H, W))

    # Compute bandwise absolute error and apply the mask
    err_map = np.abs(data_np - recon_np)  # [C, H, W]
    masked = err_map * mask_stack

    # Determine which bands to visualize
    if bands_to_show is None:
        bands_to_show = list(range(C))

    # Default to red if no colors provided
    if band_colors is None:
        band_colors = {b: [1.0, 0.0, 0.0] for b in bands_to_show}

    # Build an RGB overlay and compute stats
    rgb_mask = np.zeros((H, W, 3), dtype=np.float32)
    stats_lines = []
    for b in bands_to_show:
        # compute exceedance mask & stats
        exceed_mask = (masked[b] > thresholds[b])
        total_exceed = int(exceed_mask.sum())
        total_valid = int(mask_stack[b].sum())
        band_score = (total_exceed / total_valid) if total_valid > 0 else 0.0
        band_perc = band_score*100

        # add to overlay
        color = np.array(band_colors.get(b, [0.5,0.5,0.5]))
        rgb_mask[exceed_mask] += color

    rgb_mask = np.clip(rgb_mask, 0, 1)

    # Plot & annotate
    fig, ax = plt.subplots(figsize=(6, 6))

    fig.subplots_adjust(bottom=0.25)

    ax.imshow(rgb_mask)
    # draw outline of leaf in white
    ax.contour(
        mask,           
        levels=[0.5],    
        colors='white',  
        linewidths=1     
    )
    ax.set_title(f"Band {b}: {total_exceed} px > {thresholds[b]:.3f} MAE, {band_perc:.3f}%")
    ax.axis('off')

    

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    

    # differences between the band
def band_differences_errors(x_test, x_diseased):
    differences = []
    for i, test in enumerate(x_test):
        diff = x_diseased[i]-test
        differences.append(diff)
    return differences

def plot_error_differences_band(x_test, x_diseased):
    differences = band_differences_errors(x_test, x_diseased)
    bands = list(range(17))
    plt.bar(bands, differences)
    plt.title('Differences in error percentage per band')
    plt.xlabel('bands')
    plt.ylabel('percentage difference of error')
    plt.show()




####################################################
######## Extreme pixel error rate per band #########
####################################################

