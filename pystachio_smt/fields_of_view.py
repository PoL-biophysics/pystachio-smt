# -*- coding: utf-8 -*-

import os
import re
import glob
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io
from pystackreg import StackReg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

def get_sort_key(s):
    # Extract all numbers in path for natural sorting (e.g., Sample1/field2 -> [1, 2])
    nums = re.findall(r'\d+', s)
    return [int(n) for n in nums] if nums else [s]

def main(folder_name, save_dir, num_channels, channel, tmats_path):
    # Recursively find all 'results' directories inside folder_name
    search_pattern = os.path.join(folder_name, "**", "results")
    results_dirs = glob.glob(search_pattern, recursive=True)
    
    # Filter directories and exclude any folder starting with "BF"
    results_dirs = [
        r for r in results_dirs 
        if os.path.isdir(r) and not os.path.basename(os.path.dirname(r)).startswith("BF")
    ]
    
    if not results_dirs:
        print(f"No 'results' directories found under '{folder_name}'.")
        return

    results_dirs = sorted(results_dirs, key=get_sort_key)
    print(f"Found {len(results_dirs)} results folder(s):\n" + "\n".join(results_dirs))
    
    # Create matplotlib figure (squeeze=False ensures axes is always 2D array)
    if num_channels == 1:
        fig, axes = plt.subplots(len(results_dirs), 3, figsize=(15, 2.5 * len(results_dirs)), squeeze=False)
    else:
        fig, axes = plt.subplots(len(results_dirs), 5, figsize=(15, 2.5 * len(results_dirs)), squeeze=False)
        
    canvas = FigureCanvas(fig)
    blank = np.zeros((256, 256))
    
    # Load transformation matrix if provided and operating in 2-channel mode
    transformation_matrix = None
    if num_channels == 2 and tmats_path and os.path.exists(tmats_path):
        transformation_matrix = np.load(tmats_path)

    for i, results_path in enumerate(results_dirs):
        fov_path = os.path.dirname(results_path)
        
        # Format label name based on relative path (e.g., "Sample1/field1" or "field1")
        rel_fov = os.path.relpath(fov_path, folder_name)
        name = rel_fov if rel_fov != "." else os.path.basename(fov_path)
        
        print(f"Processing FOV: {name}")

        if num_channels == 1:
            color = "Reds" if channel == "R" else "Greens"
            img_file = os.path.join(results_path, "R_avg.tif" if channel == "R" else "L_avg.tif")
            
            try:
                img = io.imread(img_file)
            except FileNotFoundError:
                print(f"Warning: Missing channel image for {name}")
                img = blank.copy()

            try:
                mask = io.imread(os.path.join(results_path, "all_cells_mask.tif"))
            except FileNotFoundError:
                print(f"Warning: Missing mask for {name}, using blank.")
                mask = blank.copy()

            try:
                bf_img = io.imread(os.path.join(results_path, "BF.tif"))
            except FileNotFoundError:
                bf_img = blank.copy()

            axes[i, 0].imshow(bf_img, cmap='gray')
            axes[i, 0].axis('off')
            axes[i, 0].set_title(f'Brightfield\n{name}', fontsize=9)

            axes[i, 1].imshow(img, cmap=color)
            axes[i, 1].axis('off')
            axes[i, 1].set_title(f'{channel} Channel\n{name}', fontsize=9)

            axes[i, 2].imshow(mask, cmap='gray')
            axes[i, 2].axis('off')
            axes[i, 2].set_title(f'Mask\n{name}', fontsize=9)

        elif num_channels == 2:
            try:
                img_L = io.imread(os.path.join(results_path, "L_avg.tif"))
                img_R = io.imread(os.path.join(results_path, "R_avg.tif"))
                img_L = cv2.GaussianBlur(img_L, ksize=(5, 5), sigmaX=0)
                img_R = cv2.GaussianBlur(img_R, ksize=(5, 5), sigmaX=0)
            except FileNotFoundError:
                img_L, img_R = blank.copy(), blank.copy()

            img_L = img_L.astype(np.float64) - np.mean(img_L)
            img_L[img_L < 0] = 0

            try:
                mask_L = io.imread(os.path.join(results_path, "all_cells_mask.tif"))
            except FileNotFoundError:
                print(f"Warning: Missing mask for {name}, using blank.")
                mask_L = blank.copy()

            try:
                bf_img = io.imread(os.path.join(results_path, "BF.tif"))
            except FileNotFoundError:
                bf_img = blank.copy()

            img_R = img_R.astype(np.float64)
            if transformation_matrix is not None:
                sr = StackReg(StackReg.RIGID_BODY)
                bkg = np.mean(img_R)
                img_R = sr.transform(img_R, transformation_matrix)
                img_R[img_R == 0] = bkg
                img_R = img_R - bkg
                img_R[img_R < 0] = 0

            axes[i, 0].imshow(bf_img, cmap='gray')
            axes[i, 0].axis('off')
            axes[i, 0].set_title(f'Brightfield\n{name}', fontsize=9)

            axes[i, 1].imshow(img_L, cmap="Greens")
            axes[i, 1].axis('off')
            axes[i, 1].set_title(f'Left\n{name}', fontsize=9)

            axes[i, 2].imshow(img_R, cmap="Reds")
            axes[i, 2].axis('off')
            axes[i, 2].set_title(f'Right\n{name}', fontsize=9)

            max_R = np.amax(img_R)
            max_L = np.amax(img_L)
            img_R_norm = (img_R / max_R * 255).astype(np.uint8) if max_R > 0 else img_R.astype(np.uint8)
            img_L_norm = (img_L / max_L * 255).astype(np.uint8) if max_L > 0 else img_L.astype(np.uint8)

            composite_merge = np.zeros((img_L.shape[0], img_L.shape[1], 3), dtype=np.uint8)
            composite_merge[:, :, 0] = img_R_norm
            composite_merge[:, :, 1] = img_L_norm
            Image.fromarray(composite_merge).save(os.path.join(results_path, "merge_img.tif"))

            axes[i, 3].imshow(composite_merge)
            axes[i, 3].axis('off')
            axes[i, 3].set_title(f'Merge\n{name}', fontsize=9)

            axes[i, 4].imshow(mask_L, cmap="gray")
            axes[i, 4].axis('off')
            axes[i, 4].set_title(f'Mask\n{name}', fontsize=9)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out_fig_path = os.path.join(save_dir, "All_FoVs.png")
    plt.savefig(out_fig_path)
    print(f"Saved summary figure to: {out_fig_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate summary figures across FoVs.")
    parser.add_argument("--num_channels", default=2, type=int)
    parser.add_argument("--channel", default="L")
    parser.add_argument("--tmats_path", default=None)
    parser.add_argument("--save_dir", default=".")
    parser.add_argument("--folder", default=".")
    
    args = parser.parse_args()
    
    main(args.folder, args.save_dir, int(args.num_channels), args.channel, args.tmats_path)