# -*- coding: utf-8 -*-

import os
import re
import glob
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io, measure
from skimage import transform as ski_transform
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


def get_sort_key(s):
    # Extract all numbers in path for natural sorting (e.g., Sample1/field2 -> [1, 2])
    nums = re.findall(r'\d+', s)
    return [int(n) for n in nums] if nums else [s]


def overlay_fit_contours(ax, fits_mask_path):
    """Loads all_fits_mask.tif and plots green dotted cell outlines over the axis."""
    if os.path.exists(fits_mask_path):
        try:
            fits_mask = io.imread(fits_mask_path)
            if np.any(fits_mask > 0):
                contours = measure.find_contours(fits_mask, 0.5)
                for contour in contours:
                    ax.plot(contour[:, 1], contour[:, 0], color='lime', linestyle=':', linewidth=1.2)
        except Exception as e:
            print(f"Warning: Could not overlay fits mask contours from {fits_mask_path}: {e}")


def load_mask(results_path, name, blank):
    """Checks for all_cells_mask_colored first, then falls back to all_cells_mask."""
    candidates = [
        "all_cells_mask_colored.tif",
        "all_cells_mask.tif",
    ]
    for filename in candidates:
        filepath = os.path.join(results_path, filename)
        if os.path.exists(filepath):
            try:
                return io.imread(filepath)
            except Exception as e:
                print(f"Warning: Could not read {filepath}: {e}")
    print(f"Warning: Missing mask for {name}, using blank.")
    return blank.copy()


def resolve_transformation_matrix(tmats_path, results_path, folder_name):
    """Loads transformation_matrix.npy from tmats_path, results/parent directories, or subfolders."""
    candidates = []
    if tmats_path:
        candidates.append(tmats_path)

    for start_dir in (results_path, os.path.abspath(folder_name)):
        cur = os.path.abspath(start_dir)
        for _ in range(6):
            candidates.append(os.path.join(cur, "transformation_matrix.npy"))
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent

    if folder_name and os.path.isdir(folder_name):
        candidates.extend(
            sorted(glob.glob(os.path.join(os.path.abspath(folder_name), "**", "transformation_matrix.npy"), recursive=True))
        )

    seen = set()
    for cand in candidates:
        norm = os.path.normcase(os.path.abspath(cand))
        if norm in seen:
            continue
        seen.add(norm)
        if os.path.isfile(cand):
            try:
                mat = np.load(cand)
                if mat.shape == (3, 3):
                    return mat, cand
            except Exception as e:
                print(f"Warning: Could not read matrix from '{cand}': {e}")

    return None, None


def _ncc(ref, mov):
    a = ref - np.mean(ref)
    b = mov - np.mean(mov)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.sum(a * b) / denom) if denom > 0 else -1.0


def _warp_moving_channel(moving_img, ref_img, tmat):
    """Warps moving_img onto ref_img via skimage.transform.warp, selecting forward or inverse tform."""
    bkg = float(np.mean(moving_img))
    tform = ski_transform.AffineTransform(matrix=tmat)

    candidates = [tform]
    try:
        candidates.append(tform.inverse)
    except np.linalg.LinAlgError:
        pass

    best_warped = None
    best_score = -np.inf
    for tf in candidates:
        warped = ski_transform.warp(
            moving_img,
            tf,
            order=1,
            mode='constant',
            cval=bkg,
            preserve_range=True,
        )
        warped = np.maximum(warped - bkg, 0.0)
        score = _ncc(ref_img, warped)
        if score > best_score:
            best_score = score
            best_warped = warped

    return best_warped


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

    # Load transformation matrix if provided or found in parent directories in 2-channel mode
    transformation_matrix = None
    if num_channels == 2:
        transformation_matrix, loaded_tmat_path = resolve_transformation_matrix(
            tmats_path, results_dirs[0], folder_name
        )
        if transformation_matrix is not None:
            print(f"Loaded transformation matrix from: {loaded_tmat_path}")
        else:
            print("Warning: No transformation_matrix.npy found; Merge will not be aligned.")

    for i, results_path in enumerate(results_dirs):
        fov_path = os.path.dirname(results_path)

        # Format label name based on relative path (e.g., "Sample1/field1" or "field1")
        rel_fov = os.path.relpath(fov_path, folder_name)
        name = rel_fov if rel_fov != "." else os.path.basename(fov_path)

        print(f"Processing FOV: {name}")

        fits_mask_path = os.path.join(results_path, "all_fits_mask.tif")

        if num_channels == 1:
            color = "Reds" if channel == "R" else "Greens"
            img_file = os.path.join(results_path, "R_avg.tif" if channel == "R" else "L_avg.tif")

            try:
                img = np.squeeze(io.imread(img_file))
            except FileNotFoundError:
                print(f"Warning: Missing channel image for {name}")
                img = blank.copy()

            mask = load_mask(results_path, name, blank)

            try:
                bf_img = np.squeeze(io.imread(os.path.join(results_path, "BF.tif")))
            except FileNotFoundError:
                bf_img = blank.copy()

            axes[i, 0].imshow(bf_img, cmap='gray')
            axes[i, 0].axis('off')
            axes[i, 0].set_title(f'Brightfield\n{name}', fontsize=9)

            axes[i, 1].imshow(img, cmap=color)
            axes[i, 1].axis('off')
            axes[i, 1].set_title(f'{channel} Channel\n{name}', fontsize=9)

            if mask.ndim == 2:
                axes[i, 2].imshow(mask, cmap='gray')
            else:
                axes[i, 2].imshow(mask)
            overlay_fit_contours(axes[i, 2], fits_mask_path)
            axes[i, 2].axis('off')
            axes[i, 2].set_title(f'Mask\n{name}', fontsize=9)

        elif num_channels == 2:
            try:
                img_L = np.squeeze(io.imread(os.path.join(results_path, "L_avg.tif")))
                img_R = np.squeeze(io.imread(os.path.join(results_path, "R_avg.tif")))
                img_L = cv2.GaussianBlur(img_L, ksize=(5, 5), sigmaX=0)
                img_R = cv2.GaussianBlur(img_R, ksize=(5, 5), sigmaX=0)
            except FileNotFoundError:
                img_L, img_R = blank.copy(), blank.copy()

            mask_L = load_mask(results_path, name, blank)

            try:
                bf_img = np.squeeze(io.imread(os.path.join(results_path, "BF.tif")))
            except FileNotFoundError:
                bf_img = blank.copy()

            tmat = transformation_matrix
            if tmat is None:
                tmat, _ = resolve_transformation_matrix(None, results_path, folder_name)

            img_L = img_L.astype(np.float64)
            img_R = img_R.astype(np.float64)

            if tmat is not None:
                if channel.upper() == "R":
                    img_R = np.maximum(img_R - np.mean(img_R), 0.0)
                    img_L = _warp_moving_channel(img_L, img_R, tmat)
                else:
                    img_L = np.maximum(img_L - np.mean(img_L), 0.0)
                    img_R = _warp_moving_channel(img_R, img_L, tmat)
            else:
                img_L = np.maximum(img_L - np.mean(img_L), 0.0)
                img_R = np.maximum(img_R - np.mean(img_R), 0.0)

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

            if mask_L.ndim == 2:
                axes[i, 4].imshow(mask_L, cmap="gray")
            else:
                axes[i, 4].imshow(mask_L)
            overlay_fit_contours(axes[i, 4], fits_mask_path)
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