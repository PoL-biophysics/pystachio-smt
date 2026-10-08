# -*- coding: utf-8 -*-
"""
UNet Segmentation Helper Module
"""

import os
import sys
import re
import numpy as np
import skimage.io
import skimage.color
import skimage.util
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def load_unet_threshold(threshold_param, default_thresh: float = 0.5) -> float:
    """
    Reads UNet threshold from a .dat / .txt parameter file path 
    or parses a direct numeric value. Defaults to default_thresh if omitted/unparseable.
    """
    if threshold_param is None:
        return default_thresh

    if isinstance(threshold_param, (int, float)):
        return float(threshold_param)

    if isinstance(threshold_param, str):
        threshold_param = threshold_param.strip()
        try:
            return float(threshold_param)
        except ValueError:
            pass

        if os.path.exists(threshold_param):
            try:
                with open(threshold_param, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]

                for line in lines:
                    try:
                        return float(line)
                    except ValueError:
                        pass

                    if any(k in line.lower() for k in ['thresh', 'threshold', 'cutoff', 'val']):
                        parts = line.replace('=', ':').split(':')
                        if len(parts) >= 2:
                            try:
                                return float(parts[1].strip())
                            except ValueError:
                                pass

                full_text = " ".join(lines)
                matches = re.findall(r"[-+]?\d*\.\d+|\d+", full_text)
                if matches:
                    return float(matches[0])
            except Exception as e:
                print(f"Warning: Could not parse threshold file '{threshold_param}': {e}. Defaulting to {default_thresh}.", flush=True)

    return default_thresh


def local_contrast_normalization(img: np.ndarray) -> np.ndarray:
    """Applies local contrast normalization to an image patch."""
    img = img.astype(np.float32)
    mean = np.mean(img)
    std = np.std(img)
    if std == 0:
        return np.zeros_like(img)
    return (img - mean) / std


def make_patches(img: np.ndarray, inv_bf: str = "False", patch_size: int = 256):
    """Slices image into patches of size patch_size x patch_size."""
    h, w = img.shape[:2]
    
    if str(inv_bf).lower() in ["true", "1"]:
        img = np.max(img) - img

    patches = []
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = img[i:i+patch_size, j:j+patch_size]
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                pad_h = patch_size - patch.shape[0]
                pad_w = patch_size - patch.shape[1]
                patch = np.pad(patch, ((0, pad_h), (0, pad_w)), mode='reflect')
            patches.append(patch)

    return patches, patch_size, patch_size


def stitch_patches(patches: list, original_shape: tuple, patch_h: int, patch_w: int) -> np.ndarray:
    """Stitches prediction patches back into original image dimensions."""
    h, w = original_shape[:2]
    full_mask = np.zeros((h, w), dtype=np.float32)
    idx = 0

    for i in range(0, h, patch_h):
        for j in range(0, w, patch_w):
            p = patches[idx]
            idx += 1
            actual_h = min(patch_h, h - i)
            actual_w = min(patch_w, w - j)
            full_mask[i:i+actual_h, j:j+actual_w] = p[:actual_h, :actual_w]

    return full_mask


def main(img_obj, model_or_path, save_dir, threshold_param=None, inv_bf="False", debug=False):
    """
    Main entry point for UNet segmentation.
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. Load Model (if file path passed)
    if isinstance(model_or_path, str):
        print(f"Loading UNet model from {model_or_path}...")
        import tensorflow as tf
        model = tf.keras.models.load_model(model_or_path, compile=False)
    else:
        model = model_or_path

    # 2. Extract dynamic threshold from parameter
    unet_thresh = load_unet_threshold(threshold_param, default_thresh=0.5)
    print(f"Executing UNet segmentation (Threshold: {unet_thresh})...", flush=True)

    # 3. Prepare Patches
    patches, patch_h, patch_w = make_patches(img_obj, inv_bf=inv_bf)
    preds = []

    # 4. Predict per patch
    for i, p in enumerate(patches):
        p_norm = local_contrast_normalization(p)

        if debug:
            plt.figure(figsize=(6, 5))
            plt.imshow(p_norm, cmap='gray')
            plt.title(f"DEBUG: Patch {i+1}/{len(patches)}")
            plt.colorbar()
            plt.savefig(os.path.join(save_dir, f"patch_{i+1}.png"))
            plt.close()

        input_tensor = p_norm.reshape(1, p_norm.shape[0], p_norm.shape[1], 1)
        pred_patch = model.predict(input_tensor, verbose=0)[0, :, :, 0]

        binary_patch = np.zeros_like(pred_patch, dtype=np.float32)
        binary_patch[pred_patch >= unet_thresh] = 255.0
        preds.append(binary_patch)

    # 5. Stitch Mask
    maski = stitch_patches(preds, img_obj.shape, patch_h, patch_w)

    # 6. Save Outputs
    print("Saving raw analysis mask (output_raw_mask_unet.tif)...")
    skimage.io.imsave(os.path.join(save_dir, "output_raw_mask_unet.tif"), maski.astype(np.uint16))

    print("Saving visual check mask (output_visual_mask_unet.tif)...")
    visual_mask = skimage.color.label2rgb(maski.astype(int), bg_label=0)
    visual_mask = skimage.util.img_as_ubyte(visual_mask)
    skimage.io.imsave(os.path.join(save_dir, "output_visual_mask_unet.tif"), visual_mask)

    # Plot visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img_obj, cmap='gray')
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(maski, cmap='gray')
    axes[1].set_title(f"UNet Mask (Threshold: {unet_thresh})")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "output_plot_visualization_unet.png"), dpi=300)
    plt.close()

    print("UNet processing complete.")
    return visual_mask


if __name__ == "__main__":
    main(img_obj=None, model_or_path="", save_dir=".")