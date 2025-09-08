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
import scipy.ndimage as ndi
from sklearn.preprocessing import minmax_scale
from PIL import Image


####################################
### VEIN DETECTION
def steger_line_detection(image, sigma=1.0):
    # First and second derivatives
    Ix = gaussian_filter(image, sigma=sigma, order=(1, 0))
    Iy = gaussian_filter(image, sigma=sigma, order=(0, 1))

    Ixx = gaussian_filter(image, sigma=sigma, order=(2, 0))
    Iyy = gaussian_filter(image, sigma=sigma, order=(0, 2))
    Ixy = gaussian_filter(image, sigma=sigma, order=(1, 1))

    # Hessian eigen decomposition
    tmp = np.sqrt((Ixx - Iyy) ** 2 + 4 * Ixy ** 2)
    lambda1 = 0.5 * (Ixx + Iyy + tmp)
    lambda2 = 0.5 * (Ixx + Iyy - tmp)

    # Eigenvector direction (associated with lambda2)
    vx = 2 * Ixy
    vy = Iyy - Ixx + tmp
    norm = np.sqrt(vx ** 2 + vy ** 2)
    norm[norm == 0] = 1e-8  # Avoid division by zero
    vx /= norm
    vy /= norm

    # Subpixel interpolation along gradient direction (safe division)
    denom = lambda2 * (vx ** 2 + vy ** 2)
    denom[denom == 0] = 1e-8  # Avoid NaNs in t
    t = -(Ix * vx + Iy * vy) / denom

    # Subpixel positions (clipped to image boundaries)
    x_coords = np.tile(np.arange(image.shape[1])[None, :], (image.shape[0], 1))
    y_coords = np.tile(np.arange(image.shape[0])[:, None], (1, image.shape[1]))
    x_subpixel = np.clip(x_coords + t * vx, 0, image.shape[1] - 1)
    y_subpixel = np.clip(y_coords + t * vy, 0, image.shape[0] - 1)

    # Return line strength and subpixel location
    line_strength = np.abs(lambda2)
    return line_strength, x_subpixel, y_subpixel

# Apply across all hyperspectral bands
def apply_steger_to_hsi(hsi_cube, sigma=1.0):
    strength_stack = []
    for i in range(hsi_cube.shape[0]):
        strength, _, _ = steger_line_detection(hsi_cube[i], sigma)
        strength_stack.append(strength)
    return np.stack(strength_stack)

# Extract the edge using the mask and dilute and erode inwards and outwards 
def extract_full_edge(mask, edge_width=7):
    mask = mask.astype(bool)
    
    # Inward edge
    eroded = mask
    for _ in range(edge_width):
        eroded = ndi.binary_erosion(eroded)
    inner_edge = np.logical_xor(mask, eroded)
    
    # Outward edge
    dilated = mask
    for _ in range(edge_width):
        dilated = ndi.binary_dilation(dilated)
    outer_edge = np.logical_xor(dilated, mask)
    
    # Combine both
    full_edge = np.logical_or(inner_edge, outer_edge)
    
    return full_edge.astype(np.uint8)

from sklearn.preprocessing import minmax_scale
# Use the detected veins and the edges to just get the veins
def vein_error_and_labels(model, dataloader, label, device, MASK_FOLDER, save_path = None , plot=False, vmax = 0.2):
    model.eval()
    reconstruction_errors = []
    labels = []
    with torch.no_grad():
        for data_batch, idxs in dataloader:
            data_batch = data_batch.unsqueeze(1).to(device)  # Shape: (B, 1, 17, 256, 256)
            recon_batch = model(data_batch)

            for i, (data_img, recon_img, idx) in enumerate(zip(data_batch, recon_batch, idxs)):
                # Remove channel dim: (1, 17, 256, 256) → (17, 256, 256)
                data_np = data_img.squeeze(0).cpu().numpy()
                recon_np = recon_img.squeeze(0).cpu().numpy()
                # Apply Steger to original image
                steger_response = apply_steger_to_hsi(data_np, sigma=1.0)
                line_map = np.mean(steger_response, axis=0)
                norm_line_map = minmax_scale(line_map.ravel()).reshape(line_map.shape)
                binary_mask = (norm_line_map > 0.023).astype(np.uint8)

                # Load disease annotation mask
                image_path = dataloader.dataset.img_paths[idx.item()]
                mask_path = os.path.join(MASK_FOLDER, os.path.basename(image_path).replace('.hdf5', '.png'))
                mask = read_mask(mask_path)
                mask = Image.fromarray(mask).resize((256, 256), resample=Image.NEAREST)
                mask = np.array(mask)

                edge_line = extract_full_edge(mask)
                edge_line_switch = np.where((edge_line == 0) | (edge_line == 1), edge_line ^ 1, edge_line)
                mask_combined = np.multiply(edge_line_switch, binary_mask)
                mask_combined = np.stack([mask_combined] * 17)  # Shape: (17, 256, 256)

                # Compute reconstruction error within the masked area
                error_map = np.abs(data_np - recon_np)  # Shape: (17, 256, 256)
                masked_error = error_map * mask_combined
                num_masked_pixels = mask_combined.sum()
                mean_error_in_mask = masked_error.sum() / num_masked_pixels if num_masked_pixels > 0 else 0.0

                reconstruction_errors.append(mean_error_in_mask)
                labels.extend([label] * data_img.size(0))

                
                if plot:
                    # Visualize RGB composites (using bands 9, 3, 5)
                    rgb = data_np[[9, 3, 5]]
                    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
                    rgb = rgb.transpose(1, 2, 0)

                    rgb_reconst = recon_np[[9, 4, 6]]
                    rgb_reconst = (rgb_reconst - rgb_reconst.min()) / (rgb_reconst.max() - rgb_reconst.min())
                    rgb_reconst = rgb_reconst.transpose(1, 2, 0)


                    # Plotting with consistent layout
                    fig = plt.figure(figsize=(12, 4))
                    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05])

                    # Original image
                    ax1 = fig.add_subplot(gs[0, 0])
                    ax1.imshow(rgb_reconst)
                    ax1.set_title("Reconstructed")
                    ax1.axis('off')

                    # Reconstructed image
                    ax2 = fig.add_subplot(gs[0, 1])
                    ax2.imshow(mask_combined[0], cmap='binary')
                    ax2.set_title("Reconstructed")
                    ax2.axis('off')

                    # Error map
                    ax3 = fig.add_subplot(gs[0, 2])
                    im = ax3.imshow(np.mean(masked_error, axis=0), cmap='hot', vmin=0, vmax=vmax)
                    ax3.set_title(f"Vein Reconstruction Error MAE: {mean_error_in_mask:.4f}")
                    ax3.axis('off')

                    # Colorbar
                    cbar_ax = fig.add_subplot(gs[0, 3])
                    plt.colorbar(im, cax=cbar_ax)
                    # Save or show
                    if save_path:
                        plt.savefig(save_path, bbox_inches='tight', dpi=150)
                    plt.show()

    return reconstruction_errors, labels


def vein_detection(dataloader, device, MASK_FOLDER, save_path = None , plot=False):
    with torch.no_grad():
        for data_batch, idxs in dataloader:
            data_batch = data_batch.unsqueeze(1).to(device)  # Shape: (B, 1, 17, 256, 256)

            for i, (data_img, idx) in enumerate(zip(data_batch, idxs)):
                # Remove channel dim: (1, 17, 256, 256) → (17, 256, 256)
                data_np = data_img.squeeze(0).cpu().numpy()
                # Apply Steger to original image
                steger_response = apply_steger_to_hsi(data_np, sigma=1.0)
                line_map = np.mean(steger_response, axis=0)
                norm_line_map = minmax_scale(line_map.ravel()).reshape(line_map.shape)
                binary_mask = (norm_line_map > 0.023).astype(np.uint8)

                # Load disease annotation mask
                image_path = dataloader.dataset.img_paths[idx.item()]
                mask_path = os.path.join(MASK_FOLDER, os.path.basename(image_path).replace('.hdf5', '.png'))
                mask = read_mask(mask_path)
                mask = Image.fromarray(mask).resize((256, 256), resample=Image.NEAREST)
                mask = np.array(mask)

                edge_line = extract_full_edge(mask, edge_width=6)
                edge_line_switch = np.where((edge_line == 0) | (edge_line == 1), edge_line ^ 1, edge_line)
                mask_combined = np.multiply(edge_line_switch, binary_mask)
                mask_combined = np.stack([mask_combined] * 211)  # Shape: (17, 256, 256)
                
                if plot:
                    # Visualize RGB composite
                    rgb = data_np[[9, 4, 6]]
                    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
                    rgb = rgb.transpose(1, 2, 0)

                    # Create red overlay for full binary mask
                    overlay_color = [0, 1, 1]  # Red
                    alpha = 0.6  # Stronger opacity

                    # Full vein overlay
                    mask_rgb = np.zeros_like(rgb)
                    for c in range(3):
                        mask_rgb[..., c] = overlay_color[c] * binary_mask
                    overlay_img = (1 - alpha) * rgb + alpha * mask_rgb

                    # Cut mask overlay (edge-filtered)
                    mask_rgb_cut = np.zeros_like(rgb)
                    for c in range(3):
                        mask_rgb_cut[..., c] = overlay_color[c] * mask_combined[0]
                    overlay_img_cut = (1 - alpha) * rgb + alpha * mask_rgb_cut

                    # Plotting
                    fig = plt.figure(figsize=(12, 4))
                    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])

                    ax1 = fig.add_subplot(gs[0])
                    ax1.imshow(rgb)
                    ax1.set_title("Original RGB")
                    ax1.axis('off')

                    ax2 = fig.add_subplot(gs[1])
                    ax2.imshow(overlay_img)
                    ax2.set_title("Detected Veins (Blue)")
                    ax2.axis('off')

                    ax3 = fig.add_subplot(gs[2])
                    ax3.imshow(overlay_img_cut)
                    ax3.set_title("Edge-Masked Veins (Blue)")
                    ax3.axis('off')

                    if save_path:
                        plt.savefig(save_path, bbox_inches='tight', dpi=150)
                    plt.show()
                    break


def visu_vein_error(
    model,
    dataloader,
    label,
    device,
    MASK_FOLDER,
    save_path=None,
    plot=False
):
    import os
    import numpy as np
    import torch
    from PIL import Image
    from matplotlib import pyplot as plt
    from sklearn.preprocessing import minmax_scale

    with torch.no_grad():
        for data_batch, idxs in dataloader:
            B = data_batch.shape[0]
            data_batch = data_batch.unsqueeze(1).to(device)   # (B,1,C,H,W)
            recons     = model(data_batch).squeeze(1)         # (B,C,H,W)

            # Prepare storage
            rgbs = []
            overlay_cuts = []
            heatmaps = []

            # Process each image
            for i, (data_img, recon_img, idx) in enumerate(zip(data_batch, recons, idxs)):
                # Convert to numpy
                data_np  = data_img.squeeze(0).cpu().numpy()   # (C,H,W)
                recon_np = recon_img.squeeze(0).cpu().numpy()  # (C,H,W)

                # 1) Steger → binary vein mask
                steger_resp = apply_steger_to_hsi(data_np, sigma=1.0)
                line_map = np.mean(steger_resp, axis=0)
                norm_line_map = minmax_scale(line_map.ravel()).reshape(line_map.shape)
                binary_mask = (norm_line_map > 0.023).astype(np.uint8)

                # 2) Load annotation mask & remove edges
                image_path = dataloader.dataset.img_paths[idx.item()]
                mask_path  = os.path.join(
                    MASK_FOLDER,
                    os.path.basename(image_path).replace('.hdf5','.png')
                )
                mask = read_mask(mask_path)
                mask = Image.fromarray(mask).resize((256,256),Image.NEAREST)
                mask = np.array(mask)
                edge_line = extract_full_edge(mask, edge_width=6)
                edge_switch = np.where((edge_line==0)|(edge_line==1),
                                       edge_line ^ 1, edge_line)
                mask_combined = (edge_switch * binary_mask).astype(np.uint8)

                # 3) Build RGB composite
                rgb = data_np[[9,4,6]]
                rgb = rgb.transpose(1,2,0)
                rgbs.append(rgb)

                # 4) Build edge‐masked overlay
                overlay = rgb.copy()
                alpha = 0.6
                # color = cyan
                for c in range(3):
                    overlay[...,c] = (1-alpha)*rgb[...,c] + alpha*(mask_combined * [0,1,1][c])
                overlay_cuts.append(overlay)

                # 5) Compute 2D heatmap over veins
                err_map = np.abs(data_np - recon_np)         # (C,H,W)
                masked_err = err_map * np.stack([mask_combined]*data_np.shape[0])
                heat = masked_err.mean(axis=0)               # (H,W)
                mask2d = binary_mask.astype(bool)
                vals = heat[mask2d]
                mn, mx = vals.min(), vals.max()
                if mx>mn:
                    norm_heat = (heat - mn)/(mx-mn)
                else:
                    norm_heat = np.zeros_like(heat)
                norm_heat[~mask2d] = 0
                heatmaps.append(norm_heat)

            if plot:
                # for each image in the batch, make its own figure
                for i in range(B):
                    fig, (ax1, ax2, ax3) = plt.subplots(
                        ncols=3, figsize=(12, 4), constrained_layout=True
                    )

                    # panel 1: RGB
                    ax1.imshow(rgbs[i])
                    ax1.set_title(f"Img {i}: Original RGB")
                    ax1.axis('off')

                    # panel 2: edge‐masked veins
                    ax2.imshow(overlay_cuts[i])
                    ax2.set_title("Edge‐Masked Veins")
                    ax2.axis('off')

                    # panel 3: heatmap
                    im = ax3.imshow(heatmaps[i], cmap='hot', vmin=0, vmax=1)
                    ax3.set_title("Vein‐Error Heatmap")
                    ax3.axis('off')

                    # shared colorbar for this figure
                    cbar = fig.colorbar(im, ax=[ax3], location='right', shrink=0.8)
                    cbar.set_label(f"{label} error (norm)")

                    # build a unique filename
                    if save_path:
                        # e.g. "/outdir/healthy_idx42_img0.png"
                        base, ext = os.path.splitext(save_path)
                        fname = f"{base}_idx{idxs[i].item()}_img{i}{ext or '.png'}"
                        fig.savefig(fname, dpi=150, bbox_inches='tight')

                    plt.show()
                    plt.close(fig)

            # only process the first batch
            break
