# -*- coding: utf-8 -*-
"""
Created for standard Cellpose / Cellpose 3 models (cyto3, bact_phase_cp3, etc.)
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

from cellpose import models, io, plot


def main(
    img_obj,
    model_type="bact_phase_cp3",
    pretrained_model=None,
    save_dir=".",
    channels=[0, 0],
    diameter=None,
    flow_threshold=0.4,
    cellprob_threshold=0.0,
    gpu=False,
):
    """
    Runs standard Cellpose / CP3 evaluation on an input image.

    Parameters:
    -----------
    img_obj : np.ndarray
        Input 2D gray or multi-channel image array.
    model_type : str
        Built-in Cellpose model name ('bact_phase_cp3', 'cyto3', 'cyto', 'nuclei').
        Used if pretrained_model is None.
    pretrained_model : str or None
        Path to custom custom-trained Cellpose model weights file.
    save_dir : str
        Directory where output masks and plots will be saved.
    channels : list
        Channel configuration, e.g. [0,0] for grayscale, [1,2] for Green/Red.
    diameter : float or None
        Expected cell diameter in pixels. If None, Cellpose estimates it automatically.
    flow_threshold : float
        Maximum allowed flow error. Increase if cell masks are missing (default: 0.4).
    cellprob_threshold : float
        Threshold for cell probability (default: 0.0).
    gpu : bool
        Whether to use GPU acceleration (requires CUDA PyTorch).
    """
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 50)
    print("Initializing Cellpose Model...")
    print("=" * 50)

    # 1. INIT MODEL
    if pretrained_model:
        print(f"Loading custom model weights from: {pretrained_model}")
        model = models.CellposeModel(gpu=gpu, pretrained_model=pretrained_model)
        is_high_level = False
    else:
        print(f"Loading built-in model: '{model_type}'")
        model = models.Cellpose(gpu=gpu, model_type=model_type)
        is_high_level = True

    # 2. PREPARE IMAGE
    img = img_obj.copy()
    print(f"Input image shape: {img.shape}")

    # 3. RUN EVALUATION
    print("Running Cellpose evaluation...")
    if is_high_level:
        masks, flows, styles, diams = model.eval(
            img,
            channels=channels,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
        )
        print(f"Estimated diameter: {diams}")
    else:
        masks, flows, styles = model.eval(
            img,
            channels=channels,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
        )

    # 4. UNWRAPPING MASKS & FLOWS
    maski = masks
    while isinstance(maski, list):
        print(f"Drilling down mask layer... type is {type(maski)}")
        maski = maski[0]

    flowi = flows[0] if isinstance(flows, list) else flows
    if isinstance(flowi, list):
        flowi = flowi[0]  # Standard RGB flow display map

    print(f"Final Mask shape: {maski.shape}")
    print(f"Total cells segmented: {maski.max()}")

    # 5. VISUALIZATION & PLOTTING
    fig = plt.figure(figsize=(12, 5))
    plot.show_segmentation(fig, img, maski, flowi, channels=channels)
    plt.tight_layout()

    # 6. SAVE OUTPUTS
    # 1. Save Raw Analysis Mask (16-bit TIFF)
    raw_mask_path = os.path.join(save_dir, "output_raw_mask_cellpose.tif")
    print(f"Saving raw analysis mask ({raw_mask_path})...")
    skimage.io.imsave(raw_mask_path, maski.astype(np.uint16))

    # 2. Save Visual Overlay Mask (Colored RGB)
    print("Saving visual check mask (output_visual_mask_cellpose.tif)...")
    visual_mask = skimage.color.label2rgb(maski, image=img, bg_label=0)
    visual_mask_ubyte = skimage.util.img_as_ubyte(visual_mask)
    skimage.io.imsave(
        os.path.join(save_dir, "output_visual_mask_cellpose.tif"), visual_mask_ubyte
    )

    # 3. Save Flow Map
    if isinstance(flowi, np.ndarray):
        io.imsave(
            os.path.join(save_dir, "output_flows_cellpose.tif"),
            flowi.astype(np.float32),
        )

    # 4. Save Plot Figure
    plot_path = os.path.join(save_dir, "output_plot_visualization_cellpose.png")
    print(f"Saving plot visualization ({plot_path})...")
    plt.savefig(plot_path, dpi=300)

    print("All files saved successfully.")
    plt.show()

    return visual_mask_ubyte


if __name__ == "__main__":
    main(img_obj=None, model_type="bact_phase_cp3", save_dir=".")