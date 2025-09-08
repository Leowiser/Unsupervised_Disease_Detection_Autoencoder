# Exploring Fully Unsupervised Autoencoders for Early Disease Detection in Strawberry Leaves using Hyperspectral Images
Unsupervised 2D and 3D Convolutional Autoencoders for early detection of Phytophthora cactorum infection in strawberry leaves using hyperspectral imaging.

The full thesis and its results can be found here:
[KU Leuven - Exploring Fully Unsupervised Autoencoders for Early Disease Detection in Strawberry Leaves using Hyperspectral Images](https://repository.teneo.libis.be/delivery/DeliveryManagerServlet?dps_pid=IE48017380&)


# AE3DCNN – Code for 3D CAE

**reconstruction_error.py**

Contains routines to compute, aggregate, and visualize pixel‐wise reconstruction errors (both per‐band and with edge‐pixels removed). The key functions (visu_error_per_band, visu_reconstruction_error_edge_removed, and get_pixel_reconstruction_errors) are used in the **Model_Inference.ipynb** notebook to examine how well a trained autoencoder reconstructs hyperspectral leaf images. 
________________________________________
**roc_percision_recall.py**

Implements stage‐by‐stage ROC and Precision‐Recall analyses plus a vein‐based error visualizer. In particular, compare_auc_across_stages and compare_pr_auc_across_stages compute and plot AUC/PR‐AUC curves for “Early/Mid/Late” disease stages, while visu_vein_error shows reconstruction errors restricted to vein pixels. All three of these functions are invoked by **Model_Inference.ipynb**.
________________________________________
**CNN_AE_helper.py**

Provides training‐and‐validation loops, checkpoint saving, and optional noise injection or gradient clipping for 3D autoencoders. Functions like train_one_epoch, validate, and train_autoencoder are called by **Model_Training.ipynb** to fit and monitor hyperspectral‐CNN autoencoder models. Stacked‐training variants (train_one_epoch_stacked, train_autoencoder_stacked) are also included for multi‐channel‐stacked inputs but are only needed if you feed three‐channel stacks instead of raw HSI.
________________________________________
**CNN3d.py**

Defines a collection of 3D‐CNN autoencoder architectures (e.g. Autoencoder, CNN3DAE, CNN3DAEFCMP, CNN3DAEMAX, and variants with different kernel/upsampling strategies). These classes encapsulate both encoder and decoder blocks (often with BatchNorm, Dropout, or FC‐bottlenecks) and are instantiated during training in **Model_Training.ipynb**. Any of these model classes can be passed to the training helper routines for fitting on hyperspectral inputs.
________________________________________
**datasets.py**

Implements two custom PyTorch datasets: RgbDataset for RGB‐image inputs (with optional masking) and HsiDataset for loading, preprocessing, and (optionally) masking hyperspectral .hdf5 files. Inside HsiDataset.__getitem__, raw HSI data are loaded via LoadHSI, preprocessed (e.g. SNV, Savitzky‐Golay, PCA, stacking), and returned as a tensor along with its index. This file is used by both **Model_Training.ipynb** and **Model_Inference.ipynb** to feed data into models and error‐analysis routines.
________________________________________
**preprocessing.py**

Houses spectral preprocessing functions that HsiDataset (dataset.py) call:

-	filter_filenames (to select files by camera/date/tray),
-	preprocess (min‐wavelength filtering, negative‐clipping, normalization, band selection),
-	preprocessing_snv and preprocessing_savgol (SNV and Savitzky‐Golay smoothing),
-	stack_original_and_derivatives_as_channels (builds 3‐channel “original + first derivative + second derivative”).

Model‐training and inference datasets rely on these routines to turn raw HSI into normalized, band‐selected tensors.
________________________________________
**utils.py**

Defines low‐level I/O helpers:

-	LoadHSI(path, return_wlens=False) loads a .hdf5 or .h5 hyperspectral dataset into a NumPy array (and retrieves wavelengths).
-	read_mask(file_path) reads a ground‐truth PNG mask, converts RGBA→integer labels {0,1,2}.

These are used throughout datasets.py, vein_detection.py, and the reconstruction‐error routines to fetch data and ground‐truth masks.
________________________________________
**vein_detection.py**

Contains Steger line‐detection code (steger_line_detection and apply_steger_to_hsi) plus utilities to extract leaf edges (extract_full_edge). Functions like vein_detection (show raw vs. edge‐masked vein maps) and vein_error_and_labels (compute mean reconstruction error along veins) are used in **Model_Inference.ipynb** or **roc_percision_recall.py** when you want to focus error analysis on vein pixels.
________________________________________
**Model_Training.ipynb**

This notebook brings together datasets.py, CNN3d.py, CNN_AE_helper.py, and preprocessing.py to train and validate chosen 3D‐CNN autoencoder architectures on hyperspectral leaf data. It uses the helper functions to fit models, track losses, and save checkpoints.
________________________________________
**Model_Inference.ipynb**

Imports reconstruction_error.py and roc_percision_recall.py (along with datasets.py, utils.py, and vein_detection.py) to load a trained model, run inference on new HSI data, and visualize reconstruction‐error histograms, per‐band errors, edge‐removed errors, ROC/PR curves by disease stage, and vein‐based error heatmaps.
________________________________________
**Summary of Usage Flow**

1.	**Training** → datasets.py loads and preprocesses leaves → CNN3d.py defines the model → CNN_AE_helper.py runs training/validation loops → optionally use preprocessing.py inside datasets.py.
2.	**Inference/Analysis** → load model weights in Model_Inference.ipynb → use datasets.py+utils.py to load new runs → reconstruction_error.py yields pixel‐ and edge‐removed‐error visualizations → roc_percision_recall.py computes stagewise ROC/PR AUC and vein‐error plots → vein_detection.py supplies Steger‐based vein masks.
Each file is thus purpose‐specific and plugged into either the training notebook or inference notebook as noted above.



----------------------------------------
**ROIReflection.py and ROI_diffplants.ipynb**

Files needed to create the spatial-spectral data exploration shown in the thesis

----------------------------------------
**data_exploration.ipynb and data_quality_checks.ipynb**

Basic data exploration for a general overview and checks of quality. The quality checks investigate the size and wavelength of the images and potential outliers.


----------------------------------------
**pca_experiment.ipynb and pca_dimreduct.ipynb and pca_bandselect.ipynb**

pca_experiment.ipynb experiments with PCA in generall.

pca_dimreduct.ipynb fits a PCA model (on healthy data only) which is saved and later re-used as a dimensionality reduction method before the autoencoder training.

pca_bandselect.ipynb selects the relevant bands as described in teh thesis. It is a final selection of 7 bands.
