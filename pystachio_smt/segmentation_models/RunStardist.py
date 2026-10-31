# -*- coding: utf-8 -*-
"""
Created for StarDist 2D Segmentation Models
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

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    from stardist.models import StarDist2D
    from csbdeep.utils import normalize
except ImportError:
    raise ImportError("StarDist or CSBDeep is not installed. Run `pip install stardist csbdeep` to use this helper.")


def main(
    img_obj,
    model_name="2D_versatile_fluo",
    model_dir=None,
    save_dir=".",
    prob_thresh=0.5,
    nms_thresh=0.4,
    norm_pmin=1.0,
    norm_pmax=99.8,
):
    """
    Runs StarDist 2D evaluation on an input image.

    Parameters:
    -----------
    img_obj : np.ndarray
        Input 2D grayscale or RGB image array.
    model_name : str
        Built-in StarDist model name ('2D_versatile_fluo', '2D_versatile_he', '2D_paper_dsb2018')
        OR the folder name of a custom-trained StarDist model.
    model_dir : str or None
        Parent directory containing the custom model folder (specified in `model_name`).
        If None, loads `model_name` as a pre-trained StarDist model.
    save_dir : str
        Directory where output masks and plots will be saved.
    prob_thresh : float
        Probability threshold for object detection (default: 0.5).
    nms_thresh : float
        Non-maximum suppression threshold for overlapping polygon suppression (default: 0.4).
    norm_pmin : float
        Lower percentile for image intensity normalization.
    norm_pmax : float
        Upper percentile for image intensity normalization.
    """
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 50)
    print("Initializing StarDist Model...")
    print("=" * 50)

    # 1. LOAD MODEL
    if model_dir is not None:
        print(f"Loading custom StarDist model '{model_name}' from: {model_dir}")
        model = StarDist2D(config=None, name=model_name, basedir=model_dir)
    else:
        print(f"Loading built-in pretrained model: '{model_name}'")
        model = StarDist2D.from_pretrained(model_name)

    # 2. PREPARE & NORMALIZE IMAGE
    img = img_obj.copy()
    print(f"Input image shape: {img.shape}")

    # StarDist models expect percentile-normalized input intensity
    axis_norm = (0, 1) if img.ndim in (2, 3) else None
    img_norm = normalize(img, pmin=norm_pmin, pmax=norm_pmax, axis=axis_norm)

    # 3. RUN EVALUATION
    print("Running StarDist prediction...")
    labels, details = model.predict_instances(
        img_norm,
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh,
    )

    maski = labels.astype(np.uint16)
    print(f"Final Mask shape: {maski.shape}")
    print(f"Total cells/objects segmented: {maski.max()}")

    # 4. VISUALIZATION & PLOTTING
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(img, cmap="gray" if img.ndim == 2 else None)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    visual_mask = skimage.color.label2rgb(maski, image=img, bg_label=0, alpha=0.4)
    axes[1].imshow(visual_mask)
    axes[1].set_title(f"StarDist Segmentation ({maski.max()} objects)")
    axes[1].axis("off")

    plt.tight_layout()

    # 5. SAVE OUTPUTS
    # 1. Raw Integer Label Mask (16-bit TIFF)
    raw_mask_path = os.path.join(save_dir, "output_raw_mask_stardist.tif")
    print(f"Saving raw analysis mask ({raw_mask_path})...")
    skimage.io.imsave(raw_mask_path, maski)

    # 2. Colored Overlay Visual Mask (RGB 8-bit)
    visual_mask_ubyte = skimage.util.img_as_ubyte(visual_mask)
    visual_mask_path = os.path.join(save_dir, "output_visual_mask_stardist.tif")
    print(f"Saving visual check mask ({visual_mask_path})...")
    skimage.io.imsave(visual_mask_path, visual_mask_ubyte)

    # 3. Polygon Coordinates / StarDist Details
    details_path = os.path.join(save_dir, "output_stardist_details.npy")
    print(f"Saving detection details & polygon coordinates ({details_path})...")
    np.save(details_path, details)

    # 4. Plot Figure
    plot_path = os.path.join(save_dir, "output_plot_visualization_stardist.png")
    print(f"Saving plot visualization ({plot_path})...")
    plt.savefig(plot_path, dpi=300)

    print("All files saved successfully.")
    plt.show()

    return visual_mask_ubyte


if __name__ == "__main__":
    main(img_obj=None, model_name="2D_versatile_fluo", save_dir=".")