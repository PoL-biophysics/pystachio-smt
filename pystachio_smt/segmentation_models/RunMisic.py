# -*- coding: utf-8 -*-
"""
Created for MiSiCv2 Segmentation
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skimage.color
import skimage.util

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import MiSiC wrapper
try:
    from misic.misic import MiSiC
except ImportError:
    raise ImportError("MiSiC is not installed. Please install it via pip or add it to Python path.")

def main(img_obj, modeldir, save_dir, mean_width=10, noise_variance=0.0001):
    """
    Runs MiSiCv2 segmentation on an input image object.
    
    Parameters:
    -----------
    img_obj : np.ndarray
        Input 2D gray/phase/fluorescence image array.
    modeldir : str
        Path to the MiSiCv2.h5 model weights file.
    save_dir : str
        Directory where output masks and plots will be saved.
    mean_width : float
        Expected average cell width in pixels (MiSiC parameter, default=10).
    noise_variance : float
        Noise variance threshold for MiSiC preprocessing.
    """
    os.makedirs(save_dir, exist_ok=True)

    print("Initializing MiSiC Model...")
    # 1. INITIALIZE & LOAD MODEL
    m = MiSiC()
    print(f"Loading MiSiCv2 weights from: {modeldir}")
    m.load_model(modeldir)
    print("Model loaded successfully!")

    # 2. PREPARE IMAGE
    img = img_obj.copy()
    if img.ndim == 3:
        # Convert RGB/multichannel to grayscale if needed
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Ensure float32 normalized image format for prediction
    img_norm = img.astype(np.float32)
    if img_norm.max() > 1.0:
        img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-8)

    print(f"Input image shape: {img.shape}, Mean width param: {mean_width}")

    # 3. RUN PREDICTION
    print("Running MiSiC evaluation...")
    raw_response = m.predict(img_norm, mean_width=mean_width, noise_variance=noise_variance)
    
    # Convert probability/response map into discrete instance labels
    if hasattr(m, "postprocess"):
        maski = m.postprocess(raw_response)
    else:
        # Fallback labeling if predict returns raw thresholded mask/labels directly
        maski = raw_response

    maski = np.squeeze(maski).astype(np.uint16)
    print(f"Final Mask shape: {maski.shape}, Identified cells: {maski.max()}")

    # 4. VISUALIZATION & PLOTTING
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    visual_mask = skimage.color.label2rgb(maski, image=img, bg_label=0, alpha=0.4)
    axes[1].imshow(visual_mask)
    axes[1].set_title(f"MiSiCv2 Segmentation ({maski.max()} cells)")
    axes[1].axis('off')
    
    plt.tight_layout()

    # 5. SAVE OUTPUTS
    # Raw Integer Label Mask (16-bit TIFF for analysis)
    raw_mask_path = os.path.join(save_dir, "output_raw_mask_misic.tif")
    print(f"Saving raw analysis mask ({raw_mask_path})...")
    skimage.io.imsave(raw_mask_path, maski.astype(np.uint16))

    # Colored Overlay Visual Mask (RGB 8-bit)
    visual_mask_ubyte = skimage.util.img_as_ubyte(visual_mask)
    visual_mask_path = os.path.join(save_dir, "output_visual_mask_misic.tif")
    print(f"Saving visual check mask ({visual_mask_path})...")
    skimage.io.imsave(visual_mask_path, visual_mask_ubyte)

    # Plot Figure
    plot_path = os.path.join(save_dir, "output_plot_visualization_misic.png")
    print(f"Saving plot visualization ({plot_path})...")
    plt.savefig(plot_path, dpi=300)
    
    print("All files saved successfully.")
    plt.show()

    return visual_mask_ubyte


if __name__ == "__main__":
    main(img_obj=None, modeldir="MiSiCv2.h5", save_dir=".")