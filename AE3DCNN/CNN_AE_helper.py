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
import optuna

def train_one_epoch(model, dataloader_train, optimizer, criterion, device, noise = False):
    model.train()
    running_batch_loss = 0.0
    for data, _ in dataloader_train:
        data = data.unsqueeze(1).to(device)
        if noise:
            noisy_data = data + 0.01 * torch.randn_like(data)
            noisy_data = torch.clamp(noisy_data, 0.0, 1.0)  # Keep input in [0,1] range
            optimizer.zero_grad()
            recon = model(noisy_data)
            loss = criterion(recon, data)
            loss.backward()
            optimizer.step()
        else:
            optimizer.zero_grad()
            recon = model(data)
            loss = criterion(recon, data)
            loss.backward()
            optimizer.step()

        running_batch_loss += loss.item()

        # Free memory
        del data, recon, loss
        torch.cuda.empty_cache()
        gc.collect()

    avg_batch_loss = running_batch_loss / len(dataloader_train)
    return avg_batch_loss

def train_one_epoch_stacked(model, dataloader_train, optimizer, criterion, device, clip_grad=True, debug=True):
    model.train()
    running_batch_loss = 0.0

    for batch_idx, (data, _) in enumerate(dataloader_train):
        # Move input to device
        data = data.to(device)

        # Debug: check input for NaN or Inf
        if debug and (torch.isnan(data).any() or torch.isinf(data).any()):
            print(f"[Batch {batch_idx}] Found NaNs or Infs in input data!")
            print(f"Min: {data.min().item()}, Max: {data.max().item()}")
            continue  # skip this batch

        # Forward + backward
        optimizer.zero_grad()
        recon = model(data)

        loss = criterion(recon, data)

        # Debug: check loss for NaN or Inf
        if debug and (torch.isnan(loss) or torch.isinf(loss)):
            print(f"[Batch {batch_idx}] NaN or Inf in loss: {loss.item()}")
            continue  # skip this batch

        loss.backward()

        # Clip gradients to avoid exploding updates
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_batch_loss += loss.item()

        # Optional: print some debug info
        if debug and batch_idx % 10 == 0:
            print(f"[Batch {batch_idx}] Loss: {loss.item():.6f}")

        # Free memory
        del data, recon, loss
        torch.cuda.empty_cache()
        gc.collect()

    avg_loss = running_batch_loss / len(dataloader_train)
    return avg_loss


def validate(model, dataloader_valid, criterion, device):
    if dataloader_valid is None:
        return 0
    model.eval()
    running_batch_loss = 0.0
    with torch.no_grad():
        for data, _ in dataloader_valid:
            data = data.unsqueeze(1).to(device)
            recon = model(data)
            loss = criterion(recon, data)
            running_batch_loss += loss.item()

            # Free memory
            del data, recon, loss
            torch.cuda.empty_cache()
            gc.collect()

    avg_batch_loss = running_batch_loss / len(dataloader_valid)
    return avg_batch_loss

def validate_stacked(model, dataloader_valid, criterion, device):
    if dataloader_valid is None:
        return 0
    model.eval()
    running_batch_loss = 0.0
    with torch.no_grad():
        for data, _ in dataloader_valid:
            data = data.to(device)
            recon = model(data)
            loss = criterion(recon, data)
            running_batch_loss += loss.item()

            # Free memory
            del data, recon, loss
            torch.cuda.empty_cache()
            gc.collect()

    avg_batch_loss = running_batch_loss / len(dataloader_valid)
    return avg_batch_loss

def save_checkpoint(model, optimizer, train_losses, val_losses, epoch, path):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path} at epoch {epoch}")




def train_autoencoder(num_epochs, model, dataloader_train, device, dataloader_valid=None,
                      criterion=torch.nn.L1Loss(), optimizer=None, save_model=True,
                      save_path='models/3D_autoencoder/checkpoint.pth', save_after_epoch=50, early_stopping=True, patience=15,
                      backup_dir='models/3D_autoencoder/backups/', noise = False, trial = None):
    
    os.makedirs(backup_dir, exist_ok=True)

    start_time = time.time()
    optimizer = optimizer if optimizer is not None else torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())  # Store best weights
    patience_counter = 0

    for epoch in range(num_epochs):
        train_batch_loss = train_one_epoch(
                model,
                dataloader_train,
                optimizer,
                criterion,
                device,
                noise = noise   #  adds some noise
            )
        val_criterion = torch.nn.L1Loss()
        val_batch_loss = validate(model, dataloader_valid, val_criterion, device)

        train_losses.append(train_batch_loss)
        val_losses.append(val_batch_loss)

        print(f"Epoch {epoch + 1}/{num_epochs} - "
              f"Train Avg. Loss: {train_batch_loss:.6f} | Val Avg. Loss: {val_batch_loss:.6f} | "
              f"Elapsed: {((time.time() - start_time) / 60):.2f} min")
        # Report to Optuna and check for pruning
        if trial is not None:
            trial.report(val_batch_loss, step=epoch)
            if trial.should_prune():
                print(f"Trial {trial.number} pruned at epoch {epoch + 1}")
                raise optuna.exceptions.TrialPruned()

        # Backup save after every 25 epochs
        if save_model and (epoch + 1) % 25 == 0:
            backup_path = os.path.join(backup_dir, f'backup_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, train_losses, val_losses, epoch + 1, backup_path)


        # Check for improvement
        if val_batch_loss < best_val_loss:
            best_val_loss = val_batch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if early_stopping and patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()

    # At the end, load best model weights and save
    model.load_state_dict(best_model_wts)
    if save_model:
        save_checkpoint(model, optimizer, train_losses, val_losses, epoch + 1, save_path)
        print(f"Best model saved to {save_path}")

    print(f'Total training time: {((time.time() - start_time) / 60):.2f} min')

    return train_losses, val_losses

def train_autoencoder_stacked(num_epochs, model, dataloader_train, device, dataloader_valid=None,
                      criterion=torch.nn.L1Loss(), optimizer=None, save_model=True,
                      save_path='models/3D_autoencoder/checkpoint.pth', save_after_epoch=50, early_stopping=True, patience=15,
                      backup_dir='models/3D_autoencoder/backups/'):
    
    os.makedirs(backup_dir, exist_ok=True)

    start_time = time.time()
    optimizer = optimizer if optimizer is not None else torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())  # Store best weights
    patience_counter = 0

    for epoch in range(num_epochs):
        train_batch_loss = train_one_epoch_stacked(model, dataloader_train, optimizer, criterion, device)
        val_batch_loss = validate_stacked(model, dataloader_valid, criterion, device)

        train_losses.append(train_batch_loss)
        val_losses.append(val_batch_loss)

        print(f"Epoch {epoch + 1}/{num_epochs} - "
              f"Train Avg. Loss: {train_batch_loss:.6f} | Val Avg. Loss: {val_batch_loss:.6f} | "
              f"Elapsed: {((time.time() - start_time) / 60):.2f} min")

        # Backup save after every 25 epochs
        if save_model and (epoch + 1) % 25 == 0:
            backup_path = os.path.join(backup_dir, f'backup_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, train_losses, val_losses, epoch + 1, backup_path)


        # Check for improvement
        if val_batch_loss < best_val_loss:
            best_val_loss = val_batch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if early_stopping and patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()

    # At the end, load best model weights and save
    model.load_state_dict(best_model_wts)
    if save_model:
        save_checkpoint(model, optimizer, train_losses, val_losses, epoch + 1, save_path)
        print(f"Best model saved to {save_path}")

    print(f'Total training time: {((time.time() - start_time) / 60):.2f} min')

    return train_losses, val_losses


def load_checkpoint(model, path, device, optimizer=None):
    """
    Loads a checkpoint and restores model, optimizer states, and training history.
    Ensures compatibility when switching between CPU and GPU.
    """
    checkpoint = torch.load(path, map_location=device)  # Ensures compatibility across devices
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()  # Set to evaluation mode

    train_losses = checkpoint["train_losses"]
    val_losses = checkpoint["val_losses"]
    epoch = checkpoint["epoch"]

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Checkpoint loaded from {path}, Last Epoch: {epoch}")
    return model, optimizer, train_losses, val_losses, epoch


def plot_losses(train_losses, val_losses, num_epochs, file_path):
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
        plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Over Epochs")
        plt.legend()
        plt.grid()
        plt.savefig(f'{file_path}_val_losses_model.png')
        plt.show()




######################################
### UMAP and latent space inference

def get_pseudo_rgb_image(img, RGB_bands=[9, 3, 5]):
    """
    Extracts pseudo-RGB image from a hyperspectral image tensor.
    
    Parameters:
        img (torch.Tensor): Hyperspectral image tensor with shape (C, H, W).
        RGB_bands (list): Band indices to use for R, G, and B.
        
    Returns:
        numpy.ndarray: Pseudo-RGB image (H, W, 3) with values normalized between 0 and 1.
    """
    # Extract the bands and convert to numpy array
    img_rgb = img[RGB_bands].cpu().numpy()  # shape: (3, H, W)
    img_rgb = img_rgb.transpose(1, 2, 0)       # shape: (H, W, 3)
    # Normalize for display
    img_rgb = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min() + 1e-6)
    return img_rgb

def plot_umap_interactive(latent_array, images, labels=None, RGB_bands=[9, 3, 5], image_scale=0.2, click_threshold=1.0, n_neighbors=15, min_dist=0.1):
    """
    Applies UMAP on latent representations and creates an interactive plot.
    Clicking on a point toggles display of its corresponding pseudo-RGB image.

    Parameters:
        latent_array (np.ndarray): Latent vectors (num_samples x latent_dim)
        images (torch.Tensor): Hyperspectral images (num_samples x C x H x W)
        labels (np.ndarray or None): Optional labels for coloring
        RGB_bands (list): Indices of hyperspectral bands used as R, G, B
        image_scale (float): Zoom level of the image thumbnails
        click_threshold (float): Max distance to register a click on a point
        n_neighbors (int): UMAP parameter for local structure
        min_dist (float): UMAP parameter for cluster spread
    """
    reducer = UMAP.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    umap_result = reducer.fit_transform(latent_array)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(umap_result[:, 0], umap_result[:, 1], c=labels if labels is not None else 'blue', cmap='viridis', s=50)
    if labels is not None:
        plt.colorbar(scatter, label="Label")

    ax.set_title("Interactive UMAP: Click a point to toggle image")
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")

    # Dictionary to hold the currently displayed annotation boxes.
    displayed_annotations = {}

    def on_click(event):
	# Only consider clicks inside the axes
        if event.inaxes != ax:
            return

	# Get click coordinates in data space
        x_click, y_click = event.xdata, event.ydata
	
	# Compute distances from the click to each point
        distances = np.sqrt((umap_result[:, 0] - x_click)**2 + (umap_result[:, 1] - y_click)**2)
        closest_idx = np.argmin(distances)

	# Only consider the click if it's close enough to a point
        if distances[closest_idx] > click_threshold:
            return

        # Toggle the image annotation for the closest point
        if closest_idx in displayed_annotations:
            # Remove the annotation if already displayed
            displayed_annotations[closest_idx].remove()
            del displayed_annotations[closest_idx]
        else:
            # Create an annotation box with the pseudo-RGB image
            img_rgb = get_pseudo_rgb_image(images[closest_idx], RGB_bands=RGB_bands)
            imagebox = OffsetImage(img_rgb, zoom=image_scale)
            ab = AnnotationBbox(imagebox, (umap_result[closest_idx, 0], umap_result[closest_idx, 1]),
                                frameon=False)
            displayed_annotations[closest_idx] = ab
            ax.add_artist(ab)

        fig.canvas.draw_idle()

    # Connect the click event handler
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

def get_latent_representations(model, dataloader, device, assigned_label=None):
    """
    Passes images through the encoder and collects latent representations, labels, and original images.

    Parameters:
        model (torch.nn.Module): Trained autoencoder model.
        dataloader (torch.utils.data.DataLoader): Dataloader for the dataset.
        device (torch.device): Device to run the model on.
        assigned_label (int, optional): Label to assign to all samples in this dataloader.

    Returns:
        latent_array (np.ndarray): Array of latent representations (num_samples x latent_dim).
        labels_array (np.ndarray): Array of labels for each sample.
        all_images (torch.Tensor): Original image tensors.
    """
    model.to(device).eval()
    latent_list = []
    labels_list = []
    image_list = []

    with torch.no_grad():
        for batch,_ in dataloader:
            images = batch.to(device)
            images = images.unsqueeze(1)  # [B, 1, C, H, W]
            latent = model.encoder(images)
            latent = latent.view(latent.size(0), -1)  # Flatten latent representation

            latent_list.append(latent.cpu().numpy())
            image_list.append(images.cpu())

            if assigned_label is not None:
                labels_list.extend([assigned_label] * images.size(0))

    latent_array = np.concatenate(latent_list, axis=0)
    labels_array = np.array(labels_list) if labels_list else None
    all_images = torch.cat(image_list, dim=0)

    return latent_array, labels_array, all_images