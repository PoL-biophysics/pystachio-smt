# -*- coding: utf-8 -*-
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from keras.models import load_model
import tensorflow as tf
import glob
import shutil
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mplpatches
from scipy import optimize, ndimage
import scipy.io as sio
from skimage import io, measure
from skimage import transform as ski_transform
from skimage.util import img_as_float64, img_as_uint
from skimage.filters import threshold_multiotsu
from PIL import Image
from pystackreg import StackReg
from roifile import ImagejRoi
import pandas as pd
import json
import tifffile
import traceback
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from scipy import ndimage as ndi
from matplotlib.widgets import RadioButtons
from keras import backend as K
from scipy.ndimage import gaussian_filter
import urllib.request
from urllib.parse import urlparse

# Strict 4-connectivity structuring element for scipy.ndimage operations
CONNECTIVITY_4_STRUCT = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)

class FileHandler:
    """Utility class for managing files and directory structures."""
    @staticmethod
    def delete_folders(pattern: str):
        matches = glob.glob(pattern)
        for folder in [d for d in matches if os.path.isdir(d)]:
            try:
                shutil.rmtree(folder)
            except Exception as e:
                print(f"Error deleting {folder}: {e}", flush=True)

    @staticmethod
    def save_parameters(args_dict: dict, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        with open(f'{save_dir}/parameters.txt', 'w', encoding="utf-8") as f:
            for k, v in args_dict.items(): 
                f.write(f"{k}: {v}\n")

class MathUtils:
    @staticmethod
    def exp_1(x, A, k, c): return A * np.exp(-k * x) + c
    @staticmethod
    def exp_2(x, A, k, A1, k1, c): return A * np.exp(-k * x) + A1 * np.exp(-1 * (k1) * x) + c
    @staticmethod
    def exp_3(x, A, k, A1, k1, A2, k2, c): return A * np.exp(-k * x) + A1 * np.exp(-1 * (k1) * x) + A2 * np.exp(-1 * (k2) * x) + c
    @staticmethod
    def chisq(ydata, ymodel, err): return np.sum((np.asarray(ydata) - np.asarray(ymodel))**2 / err)
    
    @staticmethod
    def dice_loss(y_true, y_pred, smooth=1e-6):
        """Computes the Dice loss between true masks and predicted masks."""
        try:
            y_true_f = K.flatten(tf.cast(y_true, tf.float32))
            y_pred_f = K.flatten(y_pred)
            intersection = K.sum(y_true_f * y_pred_f)
            dice_coef = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
            return 1.0 - dice_coef
        except Exception:
            return 0.0

class IntensityAnalyzer:
    """Handles plotting of intensity traces and fitting to exponential decay models."""
    @staticmethod
    def plot_intensity(intensity_data, channel, index, frames, save_dir):
        intensity_data_x = np.asarray(intensity_data[0])
        intensity_data_y = np.asarray(intensity_data[1])
        
        # 1. Full raw intensity trace
        fig = plt.figure(figsize=(7, 5))
        ax = fig.subplots()
        ax.plot(intensity_data_x[:frames], intensity_data_y[:frames], label=f'{channel}')
        ax.legend(fontsize=12)
        ax.set_xlabel("Frame Number")
        ax.set_ylabel("Intensity (Counts)")
        plt.savefig(f'{save_dir}/Int_intensity_{channel}_obj_{index}.png')
        plt.close()

        # 2. First 50 frames exponential decay fitting
        fig = plt.figure(figsize=(7, 5))
        ax = fig.subplots()
        
        fit_frames = min(50, len(intensity_data_x))
        x_fit_range = intensity_data_x[:fit_frames]
        y_fit_range = intensity_data_y[:fit_frames]
        
        ax.plot(x_fit_range, y_fit_range, label=f'{channel}')
        
        fit_quality = []
        baseline_index = min(50, len(intensity_data_y) - 1)
        baseline = intensity_data_y[baseline_index]
        
        guess_frames = min(10, len(intensity_data_x))
        x_, y_ = intensity_data_x[:guess_frames], intensity_data_y[:guess_frames]

        try:
            fit_exp_1 = lambda x, A, k: MathUtils.exp_1(x, A, k, baseline)
            params, cov = optimize.curve_fit(fit_exp_1, x_, y_, p0=(max(y_), 1))
            model_1 = MathUtils.exp_1(x_fit_range, params[0], params[1], baseline)
            rchisq_1 = MathUtils.chisq(y_fit_range, model_1, model_1) / max(1, (len(model_1) - 1 - 2))
            ax.plot(x_fit_range, model_1, linestyle="--", label="Single exp")
            fit_quality.append([params, rchisq_1])
        except Exception:
            print(f"Cell {index} ({channel}): Cannot fit to 1 exp", flush=True)
            fit_quality.append(0)
            
        try:
            fit_exp_2 = lambda x, A, k, A1, k1: MathUtils.exp_2(x, A, k, A1, k1, baseline)
            params, cov = optimize.curve_fit(fit_exp_2, x_, y_, p0=(1e5, 1, 1e5, 1))
            model_2 = MathUtils.exp_2(x_fit_range, params[0], params[1], params[2], params[3], baseline)
            rchisq_2 = MathUtils.chisq(y_fit_range, model_2, model_2) / max(1, (len(model_2) - 1 - 4))
            ax.plot(x_fit_range, model_2, linestyle="--", label="Double exp")
            fit_quality.append([params, rchisq_2])
        except Exception:
            print(f"Cell {index} ({channel}): Cannot fit to 2 exp", flush=True)
            fit_quality.append(0)

        try:    
            fit_exp_3 = lambda x, A, k, A1, k1, A2, k2: MathUtils.exp_3(x, A, k, A1, k1, A2, k2, baseline)
            params, cov = optimize.curve_fit(fit_exp_3, x_, y_, p0=(0.75e5, 1, 0.75e5, 1, 0.75e5, 1))
            model_3 = MathUtils.exp_3(x_fit_range, params[0], params[1], params[2], params[3], params[4], params[5], baseline)
            rchisq_3 = MathUtils.chisq(y_fit_range, model_3, model_3) / max(1, (len(model_3) - 1 - 6))
            ax.plot(x_fit_range, model_3, linestyle="--", label="Triple exp")
            fit_quality.append([params, rchisq_3])    
        except Exception:
            print(f"Cell {index} ({channel}): Cannot fit to 3 exp", flush=True)
            fit_quality.append(0)

        ax.legend(fontsize=8)
        ax.set_xlabel("Frame Number")
        ax.set_ylabel("Intensity (Counts)")
        plt.savefig(f'{save_dir}/Int_intensity_fit_{channel}_obj_{index}.png')
        plt.close()
        
        return fit_quality

class ImageProcessor:
    @staticmethod
    def separate_diagonal_pixels(mask: np.ndarray) -> np.ndarray:
        """
        Enforces 4-connectivity separation across binary or integer-labeled masks.
        Detects diagonal-only pixel contacts between distinct objects and severs the
        diagonal bridge to prevent 8-connected component merging.
        """
        if mask is None or not np.any(mask):
            return mask

        out_mask = mask.copy()
        binary = out_mask > 0
        labeled = measure.label(binary, connectivity=1)

        tl = labeled[:-1, :-1]
        tr = labeled[:-1, 1:]
        bl = labeled[1:, :-1]
        br = labeled[1:, 1:]

        # Pattern 1: Top-Left & Bottom-Right diagonal contact across different component IDs
        diag1 = (tl > 0) & (br > 0) & (tl != br) & (tr == 0) & (bl == 0)
        # Pattern 2: Top-Right & Bottom-Left diagonal contact across different component IDs
        diag2 = (tr > 0) & (bl > 0) & (tr != bl) & (tl == 0) & (br == 0)

        if np.any(diag1):
            y_d1, x_d1 = np.where(diag1)
            out_mask[y_d1 + 1, x_d1 + 1] = 0

        if np.any(diag2):
            y_d2, x_d2 = np.where(diag2)
            out_mask[y_d2 + 1, x_d2] = 0

        return out_mask

    @staticmethod
    def transform_mask(mask: np.ndarray, tmats: np.ndarray, src_channel: str = "L", dst_channel: str = "R", ref_channel: str = "L") -> np.ndarray:
        if tmats is None or not isinstance(tmats, np.ndarray) or tmats.shape != (3, 3):
            print(f"[ImageProcessor] transform_mask: tmats is None or invalid shape. Returning original mask.", flush=True)
            return ImageProcessor.separate_diagonal_pixels(mask)
            
        src_ch = src_channel.upper()
        dst_ch = dst_channel.upper()
        ref_ch = ref_channel.upper()
        
        if src_ch == dst_ch:
            return ImageProcessor.separate_diagonal_pixels(mask)
            
        try:
            tform = ski_transform.AffineTransform(matrix=tmats)
            
            if ref_ch == "L":
                transform_to_apply = tform.inverse if (src_ch == "L" and dst_ch == "R") else tform
            else:
                transform_to_apply = tform if (src_ch == "L" and dst_ch == "R") else tform.inverse

            warped = ski_transform.warp(mask.astype(float), transform_to_apply, order=0, preserve_range=True)
            warped_mask = np.round(warped).astype(mask.dtype) if (np.issubdtype(mask.dtype, np.integer) or mask.dtype == bool) else warped

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            unique_labels = np.unique(warped_mask)
            unique_labels = unique_labels[unique_labels > 0]
            
            cleaned_mask = np.zeros_like(warped_mask)
            if len(unique_labels) == 0:
                return cleaned_mask

            if (len(unique_labels) == 1 and unique_labels[0] == 1) or mask.dtype == bool or np.array_equal(unique_labels, [255]):
                bin_mask = (warped_mask > 0).astype(np.uint8)
                dilated = cv2.dilate(bin_mask, kernel, iterations=1)
                eroded = cv2.erode(dilated, kernel, iterations=1)
                max_val = np.max(warped_mask) if np.max(warped_mask) > 0 else 255
                cleaned_mask = (eroded > 0).astype(mask.dtype) if mask.dtype == bool else (eroded * max_val).astype(mask.dtype)
            else:
                for label_id in unique_labels:
                    single_cell = (warped_mask == label_id).astype(np.uint8)
                    dilated = cv2.dilate(single_cell, kernel, iterations=1)
                    eroded = cv2.erode(dilated, kernel, iterations=1)
                    cleaned_mask[eroded > 0] = label_id

            return ImageProcessor.separate_diagonal_pixels(cleaned_mask)
        except Exception as e:
            print(f"Warning: Failed to transform mask from {src_ch} to {dst_ch}: {e}", flush=True)
            return ImageProcessor.separate_diagonal_pixels(mask)

    @staticmethod
    def local_contrast_normalization(img, sigma=15.0, eps=1e-5):
        img_f = img.astype(np.float32)
        local_mean = gaussian_filter(img_f, sigma=sigma)
        img_zero_centered = img_f - local_mean
        
        local_var = gaussian_filter(img_zero_centered**2, sigma=sigma)
        local_std = np.sqrt(np.maximum(local_var, 0)) + eps
        
        lcn_img = img_zero_centered / local_std
        
        lcn_min, lcn_max = lcn_img.min(), lcn_img.max()
        if lcn_max > lcn_min:
            lcn_norm = (lcn_img - lcn_min) / (lcn_max - lcn_min)
        else:
            lcn_norm = np.zeros_like(lcn_img)
            
        return (lcn_norm * 255).astype(np.uint8)

    @staticmethod
    def read_roi(roi_path: str, roi_channel: str, current_channel: str, img_height: int, img_width: int, is_split_view: bool = True):
        half_width = img_width // 2 if is_split_view else img_width
        if not os.path.exists(roi_path):
            return [0, img_height, 0, half_width]
            
        try:
            roi = ImagejRoi.fromfile(roi_path)
            y1, x1, y2, x2 = int(roi.top), int(roi.left), int(roi.bottom), int(roi.right)
            
            if is_split_view:
                if x1 >= half_width: x1 -= half_width
                if x2 > half_width:  x2 -= half_width
                
            y1 = max(0, min(y1, img_height))
            y2 = max(0, min(y2, img_height))
            x1 = max(0, min(x1, half_width))
            x2 = max(0, min(x2, half_width))
            
            return [y1, y2, x1, x2]
        except Exception as e:
            print(f"Error reading ROI file: {e}", flush=True)
            return [0, img_height, 0, half_width]

    @staticmethod
    def interactive_edit_mask(mask=None, bg_img=None, fl_img=None, image_dict=None):
        if mask is None:
            ref_img = None
            if image_dict:
                ref_img = next(iter(image_dict.values()), None)
            if ref_img is None:
                ref_img = bg_img if bg_img is not None else fl_img
            if ref_img is None:
                raise ValueError("Must provide either a mask, bg_img, fl_img, or image_dict.")
            mask = np.zeros(ref_img.shape[:2], dtype=np.int32)
            
        editor = MaskEditor(mask=mask, bf_img=bg_img, fl_img=fl_img, image_dict=image_dict)
        return ImageProcessor.separate_diagonal_pixels(editor.mask)

    @staticmethod
    def fit_and_generate_cell_shape(points, shape, radius_hint=None):
        mask = np.zeros(shape, dtype=np.uint8)
        if len(points) < 2:
            return mask
            
        p1, p2 = np.array(points[0]), np.array(points[1])
        length = np.linalg.norm(p2 - p1)
        angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
        
        radius = radius_hint if radius_hint is not None else max(3.0, length / 4.0)
        
        y, x = np.ogrid[:shape[0], :shape[1]]
        dx = x - p1[0]
        dy = y - p1[1]
        
        ux = np.cos(angle)
        uy = np.sin(angle)
        
        proj = dx * ux + dy * uy
        proj_clamped = np.clip(proj, 0, length)
        
        closest_x = p1[0] + proj_clamped * ux
        closest_y = p1[1] + proj_clamped * uy
        
        dist_sq = (x - closest_x)**2 + (y - closest_y)**2
        mask[dist_sq <= radius**2] = 1
        return ImageProcessor.separate_diagonal_pixels(mask)

    @staticmethod
    def make_patches(input_image, inv_img):
        img = input_image.copy().astype(np.float64)
        bkg_1 = np.mean(img)
        
        h, w = img.shape
        pad_h = (256 - (h % 256)) % 256
        pad_w = (256 - (w % 256)) % 256
        
        img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=bkg_1)

        patches = []
        num_h_patches = img.shape[0] // 256
        num_w_patches = img.shape[1] // 256
        
        for i in range(num_h_patches):
            for j in range(num_w_patches):
                y, x = i * 256, j * 256
                patch = img[y:y+256, x:x+256]
                
                if str(inv_img).lower() == "true":
                    patch = np.amax(patch) - patch
                    bkg_2 = np.mean(patch)
                    patch = patch - bkg_2 + bkg_1
                
                if patch.max() > 0:
                    patch = patch / patch.max()
                    
                patches.append(patch)
                
        return patches, num_h_patches, num_w_patches

    @staticmethod
    def stitch_patches(predictions, original_shape, num_h_patches, num_w_patches):
        stitched_image = np.zeros((num_h_patches * 256, num_w_patches * 256))
        index = 0
        for i in range(num_h_patches):
            for j in range(num_w_patches):
                y1, x1 = i * 256, j * 256
                stitched_image[y1:y1+256, x1:x1+256] = predictions[index]
                index += 1
        out = stitched_image[:original_shape[0], :original_shape[1]].astype(np.uint8)
        return ImageProcessor.separate_diagonal_pixels(out)

    @staticmethod
    def apply_watershed_threshold(img_for_masking, dist_multiplier=0.3, use_otsu="False"):
        img_float = img_for_masking.astype(np.float32)
        p_low, p_high = np.percentile(img_float, (1.0, 99.5))
        img_clipped = np.clip(img_float, p_low, p_high)
        img_uint8 = cv2.normalize(img_clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_uint8)
        blurred = cv2.medianBlur(img_clahe, 5)
        
        otsu_str = str(use_otsu).lower()
        if otsu_str == "multi":
            thresholds = threshold_multiotsu(blurred, classes=3)
            thresh = (blurred > thresholds[0]).astype(np.uint8) * 255
        elif otsu_str in ["true", "1", "t", "y"]:
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 45, -2)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        sure_bg = cv2.dilate(opening, kernel, iterations=2)
        
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, dist_multiplier * dist_transform.max(), 255, 0)
        
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Enforce 4-connectivity on connected components step
        _, markers = cv2.connectedComponents(sure_fg, connectivity=4)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        img_color = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(img_color, markers)
        
        final_mask = np.zeros_like(img_uint8)
        final_mask[markers > 1] = 255
        filled_mask = ndimage.binary_fill_holes(final_mask).astype(np.uint8) * 255
        return ImageProcessor.separate_diagonal_pixels(filled_mask)

    @staticmethod
    def generate_dual_masks(mask, target_channel):
        blur = cv2.GaussianBlur(mask, (25, 25), 0)
        ret, inv_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        morph_kernel = np.ones((5, 5), np.uint8)
        inv_mask = cv2.dilate(inv_mask, morph_kernel, iterations=2)
        inv_mask = cv2.erode(inv_mask, morph_kernel, iterations=1)

        mask_sep = ImageProcessor.separate_diagonal_pixels(mask)
        inv_mask_sep = ImageProcessor.separate_diagonal_pixels(inv_mask)

        if target_channel == 'L': return mask_sep, inv_mask_sep
        elif target_channel == 'R': return inv_mask_sep, mask_sep
        else: return mask_sep, mask_sep

    @staticmethod
    def generate_channel_overlay(bg_img, mask_or_fits, save_path, color=(0, 255, 255), alpha=0.3):
        if bg_img is None or mask_or_fits is None:
            return
        try:
            bg_norm = cv2.normalize(bg_img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            bgr_bg = cv2.cvtColor(bg_norm, cv2.COLOR_GRAY2BGR)
            
            mask_bin = (mask_or_fits > 0).astype(np.uint8) * 255
            if mask_bin.shape != bg_norm.shape:
                mask_bin = cv2.resize(mask_bin, (bg_norm.shape[1], bg_norm.shape[0]), interpolation=cv2.INTER_NEAREST)
                
            if not np.any(mask_bin):
                cv2.imwrite(save_path, bgr_bg)
                return
            
            bgr_color = (color[2], color[1], color[0])
            colored_mask = np.zeros_like(bgr_bg)
            colored_mask[mask_bin > 0] = bgr_color
            
            overlay = cv2.addWeighted(bgr_bg, 1.0, colored_mask, alpha, 0)
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, bgr_color, thickness=1, lineType=cv2.LINE_AA)
            
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            cv2.imwrite(save_path, overlay)
        except Exception as e:
            print(f"Warning: Failed to generate overlay ({save_path}): {e}", flush=True)

class VideoProcessor:
    @staticmethod
    def process_video(video_path, start_frame, end_frame, roi_channel, roi_file, save_dir, num_channels, channel, ALEX, start_channel="L"):
        frames = io.imread(video_path)
        if frames.ndim == 2:
            frames = np.expand_dims(frames, axis=0)
        frames = frames[start_frame:end_frame]
            
        raw_shape = np.shape(frames)
        height, width = raw_shape[1], raw_shape[2]
        is_split_view = (int(num_channels) == 2) or (width >= height * 1.2 and width >= 500)
        half_width = width // 2 if is_split_view else width
        
        y1, y2, x1, x2 = ImageProcessor.read_roi(
            roi_file, roi_channel, channel, height, width, is_split_view=is_split_view
        )
        
        if not is_split_view:
            L_full = frames
            R_full = None
        elif str(ALEX).lower() == "false":
            L_full = frames[:, :, :half_width]
            R_full = frames[:, :, half_width:]
        else:
            if str(start_channel).upper().startswith("R"):
                R_full = frames[::2, :, half_width:]
                L_full = frames[1::2, :, :half_width]
            else:
                L_full = frames[::2, :, :half_width]
                R_full = frames[1::2, :, half_width:]
            
        L_channel = L_full[:, y1:y2, x1:x2] if L_full is not None else None
        R_channel = R_full[:, y1:y2, x1:x2] if R_full is not None else None
        
        if int(num_channels) == 1:
            if channel == "L": return L_channel, None, len(L_channel), raw_shape
            else: return None, R_channel, len(R_channel) if R_channel is not None else len(L_channel), raw_shape
        else:
            return L_channel, R_channel, len(L_channel), raw_shape

class BeadRegistrar:
    """Handles bead-based spatial registration using NCC or interactive manual alignment."""
    @staticmethod
    def calculate_ncc(image_a, image_b):
        image_a = image_a.astype(np.float64)
        image_b = image_b.astype(np.float64)
        mean_a, mean_b = np.mean(image_a), np.mean(image_b)
        centered_a, centered_b = image_a - mean_a, image_b - mean_b
        numerator = np.sum(centered_a * centered_b)
        denominator = np.sqrt(np.sum(centered_a ** 2) * np.sum(centered_b ** 2))
        return 0.0 if denominator == 0 else float(numerator / denominator)

    @staticmethod
    def normalize_channels(i1, i2):
        i1, i2 = i1.astype(np.float64), i2.astype(np.float64)
        m1, m2 = np.amax(i1), np.amin(i1)
        if m1 != m2: i1 = 1024 * (i1 - m2) / (m1 - m2)
        m1, m2 = np.amax(i2), np.amin(i2)
        if m1 != m2: i2 = 1024 * (i2 - m2) / (m1 - m2)

        m, n = np.amax(i2), np.amax(i1)
        if m != 0: i2 = n * i2 / m
        if np.amax(i1) != 0: i1 = 255 * i1 / np.amax(i1)
        if np.amax(i2) != 0: i2 = 255 * i2 / np.amax(i2)
            
        return i1, i2

    def save_registration_plot(self, static_img, moving_img, registered_img, ncc, name, save_dir):
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        axs[0].imshow((static_img + moving_img) / 2.0, cmap='gray')
        axs[0].set_title(f'Original Images - {name}')
        axs[0].set_xticks([]); axs[0].set_yticks([])

        axs[1].imshow(registered_img, cmap='gray')
        axs[1].set_title('Registered Image')
        axs[1].set_xticks([]); axs[1].set_yticks([])

        axs[2].imshow((static_img + registered_img) / 2.0, cmap='gray')
        axs[2].set_title(f'Overlay - NCC = {ncc:.4f}')
        axs[2].set_xticks([]); axs[2].set_yticks([])

        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        safe_name = os.path.basename(os.path.normpath(name))
        plt.savefig(os.path.join(save_dir, f"Registered_overlay_{safe_name}.png"))
        plt.close(fig)

    def save_manual_overlay_and_metrics(self, ch1_norm, ch2_norm, matrix, save_dir, method_label="Manual Registration"):
        tform = ski_transform.AffineTransform(matrix=matrix)
        warped_ch2 = ski_transform.warp(ch2_norm, tform.inverse)

        ncc_before = self.calculate_ncc(ch1_norm, ch2_norm)
        ncc_after = self.calculate_ncc(ch1_norm, warped_ch2)

        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        axs[0].imshow((ch1_norm + ch2_norm) / 2.0, cmap='gray')
        axs[0].set_title(f'Original Images (Unaligned Overlay) - NCC = {ncc_before:.4f}')
        axs[0].set_xticks([]); axs[0].set_yticks([])

        axs[1].imshow(warped_ch2, cmap='gray')
        axs[1].set_title('Registered Moving Channel (Channel 2)')
        axs[1].set_xticks([]); axs[1].set_yticks([])

        axs[2].imshow((ch1_norm + warped_ch2) / 2.0, cmap='gray')
        axs[2].set_title(f'{method_label} Overlay - NCC = {ncc_after:.4f}')
        axs[2].set_xticks([]); axs[2].set_yticks([])

        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "Registered_overlay_manual.png"))
        plt.savefig(os.path.join(save_dir, "Registered_overlay_summed_beads.png"))
        plt.close(fig)

        df_reg = pd.DataFrame([{
            'Best_Folder': method_label,
            'Best_NCC': ncc_after,
            'Matrix_00': matrix[0, 0], 'Matrix_01': matrix[0, 1], 'Matrix_02': matrix[0, 2],
            'Matrix_10': matrix[1, 0], 'Matrix_11': matrix[1, 1], 'Matrix_12': matrix[1, 2],
            'Matrix_20': matrix[2, 0], 'Matrix_21': matrix[2, 1], 'Matrix_22': matrix[2, 2]
        }])
        df_reg.to_csv(os.path.join(save_dir, "registration.csv"), index=False)
        print(f"[{method_label}] Saved visual overlays and registration.csv with NCC = {ncc_after:.4f}", flush=True)

    def calculate_transformation(self, bead_prefix: str, save_dir: str, channel: str):
        search_pattern = f"{bead_prefix}_*"
        bead_folders = sorted(glob.glob(search_pattern))
        
        if not bead_folders and os.path.exists(bead_prefix) and os.path.isdir(bead_prefix):
            bead_folders = [bead_prefix]

        if not bead_folders:
            print(f"[DEBUG Registration] NO bead folders found matching: '{search_pattern}'", flush=True)
            return None

        ncc_scores = []
        sr = StackReg(StackReg.AFFINE)

        summed_img = None
        ncc_before_list, ncc_after_list, folder_names = [], [], []

        for idx, folder in enumerate(bead_folders):
            files = glob.glob(f"{folder}/*.ome.tif") + glob.glob(f"{folder}/*.tif")
            if not files: continue
            
            beads = io.imread(files[0])
            if beads.ndim == 3: beads = np.mean(beads, axis=0)
                
            beads_float = img_as_float64(beads)
            if summed_img is not None: summed_img += beads_float
            else: summed_img = beads_float.copy()
                
            width = beads.shape[1]
            L, R = beads[:, :width//2], beads[:, width//2:]
            
            i1, i2 = (L.copy(), R.copy()) if channel.upper() == "L" else (R.copy(), L.copy())
            L_norm, R_norm = self.normalize_channels(L, R)

            tmat = sr.register(L_norm, R_norm) if channel.upper() == "L" else sr.register(R_norm, L_norm)
                
            i1_norm, i2_norm = self.normalize_channels(i1, i2)
            out = sr.transform(i2_norm, tmat)
            
            ncc_before = self.calculate_ncc(i1_norm, i2_norm)
            ncc_after = self.calculate_ncc(i1_norm, out)
            ncc_before_list.append(ncc_before)
            ncc_after_list.append(ncc_after)
            folder_names.append(f"beads_{idx+1}")
            
            out[out == 0] = np.mean(i2_norm)
            self.save_registration_plot(i1_norm, i2_norm, out, ncc_after, f"beads_{idx+1}", save_dir)
            ncc_scores.append({'folder': folder, 'ncc': ncc_after, 'matrix': tmat})

        if not ncc_scores: return None

        if summed_img is not None:
            summed_norm = cv2.normalize(summed_img, None, 0, 65535, cv2.NORM_MINMAX).astype(np.uint16)
            tifffile.imwrite(os.path.join(save_dir, "summed_beads.tif"), summed_norm, imagej=True)

            s_width = summed_img.shape[1]
            s_L, s_R = summed_img[:, :s_width//2], summed_img[:, s_width//2:]
            s1, s2 = (s_L.copy(), s_R.copy()) if channel.upper() == "L" else (s_R.copy(), s_L.copy())
            s1_norm, s2_norm = self.normalize_channels(s1, s2)
            
            summed_tmat = sr.register(s1_norm, s2_norm) if channel.upper() == "L" else sr.register(s2_norm, s1_norm)
            s_out = sr.transform(s2_norm, summed_tmat)
            
            summed_ncc_before = self.calculate_ncc(s1_norm, s2_norm)
            summed_ncc_after = self.calculate_ncc(s1_norm, s_out)

            folder_names.append("summed_beads")
            ncc_before_list.append(summed_ncc_before)
            ncc_after_list.append(summed_ncc_after)

            fig, axs = plt.subplots(3, 1, figsize=(10, 15))
            axs[0].imshow((s1_norm + s2_norm) / 2, 'gray')
            axs[0].set_title(f'Original Summed Beads - NCC = {summed_ncc_before:.4f}')
            axs[0].set_xticks([]); axs[0].set_yticks([])

            axs[1].imshow(s_out, 'gray')
            axs[1].set_title('Registered Summed Image')
            axs[1].set_xticks([]); axs[1].set_yticks([])

            axs[2].imshow(((s1_norm + s_out) / 2), cmap='gray')
            axs[2].set_title(f'Overlay Summed Beads - NCC = {summed_ncc_after:.4f}')
            axs[2].set_xticks([]); axs[2].set_yticks([])
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "Registered_overlay_summed_beads.png"))
            plt.close()

        df_ncc = pd.DataFrame({
            'Bead_Folder': folder_names,
            'NCC_Before': ncc_before_list,
            'NCC_After': ncc_after_list
        })
        df_ncc.to_csv(os.path.join(save_dir, "ncc_scores.csv"), index=False)

        fig, ax = plt.subplots(figsize=(9, 5))
        x_indices = np.arange(len(folder_names))
        ax.bar(x_indices - 0.2, ncc_before_list, width=0.4, label='NCC Before')
        ax.bar(x_indices + 0.2, ncc_after_list, width=0.4, label='NCC After')
        ax.set_xticks(x_indices)
        ax.set_xticklabels(folder_names, rotation=25, ha='right')
        ax.set_ylabel("Normalized Cross-Correlation (NCC)")
        ax.set_title("Bead Alignment NCC Scores (Individual & Summed)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "ncc_scores.png"))
        plt.close()

        best_entry = max(ncc_scores, key=lambda x: x['ncc'])
        tmats = best_entry['matrix']

        df_reg = pd.DataFrame([{
            'Best_Folder': best_entry['folder'],
            'Best_NCC': best_entry['ncc'],
            'Matrix_00': tmats[0, 0], 'Matrix_01': tmats[0, 1], 'Matrix_02': tmats[0, 2],
            'Matrix_10': tmats[1, 0], 'Matrix_11': tmats[1, 1], 'Matrix_12': tmats[1, 2],
            'Matrix_20': tmats[2, 0], 'Matrix_21': tmats[2, 1], 'Matrix_22': tmats[2, 2]
        }])
        df_reg.to_csv(os.path.join(save_dir, "registration.csv"), index=False)
        np.save(f"{save_dir}/transformation_matrix.npy", tmats)
        return tmats

    def _interactive_drag_registration(self, img, save_dir: str):
        h, w = img.shape[:2]
        if w >= h and w > 1:
            half_w = w // 2
            ch1 = img[:, :half_w].astype(float)
            ch2 = img[:, half_w:].astype(float)
        else:
            ch1, ch2 = img.astype(float), img.astype(float)

        ch1_norm = (ch1 - ch1.min()) / (ch1.max() - ch1.min() + 1e-8)
        ch2_norm = (ch2 - ch2.min()) / (ch2.max() - ch2.min() + 1e-8)

        dx, dy = 0.0, 0.0
        angle_deg = 0.0

        fig, ax = plt.subplots(figsize=(10, 8))
        rgb = np.zeros((ch1_norm.shape[0], ch1_norm.shape[1], 3), dtype=np.float32)
        rgb[:, :, 0] = ch2_norm
        rgb[:, :, 1] = ch1_norm
        
        im_artist = ax.imshow(rgb)
        ax.set_title("Interactive Drag Overlay\nLeft-Drag: Move | Scroll / Q,E: Rotate | Arrows: Nudge | R: Reset | D: Done")
        plt.axis('image')

        drag_state = {'press_xy': None, 'initial_dxdy': (0.0, 0.0)}

        def update_display():
            nonlocal dx, dy, angle_deg
            rad = np.deg2rad(angle_deg)
            center = (ch2_norm.shape[1] / 2.0, ch2_norm.shape[0] / 2.0)
            
            t_center = ski_transform.AffineTransform(translation=-np.array(center))
            t_rot = ski_transform.AffineTransform(rotation=rad)
            t_back = ski_transform.AffineTransform(translation=np.array(center) + np.array([dx, dy]))
            
            total_tform = t_center + t_rot + t_back
            warped = ski_transform.warp(ch2_norm, total_tform.inverse)
            
            rgb_updated = np.zeros_like(rgb)
            rgb_updated[:, :, 0] = warped
            rgb_updated[:, :, 1] = ch1_norm
            
            im_artist.set_data(rgb_updated)
            ax.set_title(f"Interactive Overlay Alignment\ndx: {dx:.2f} px | dy: {dy:.2f} px | Angle: {angle_deg:.2f}°")
            fig.canvas.draw_idle()

        def on_press(event):
            if event.inaxes != ax or event.button != 1: return
            if event.xdata is None or event.ydata is None: return
            drag_state['press_xy'] = (event.xdata, event.ydata)
            drag_state['initial_dxdy'] = (dx, dy)

        def on_motion(event):
            if drag_state['press_xy'] is None or event.inaxes != ax: return
            if event.xdata is None or event.ydata is None: return
            x0, y0 = drag_state['press_xy']
            init_dx, init_dy = drag_state['initial_dxdy']
            nonlocal dx, dy
            dx = init_dx + (event.xdata - x0)
            dy = init_dy + (event.ydata - y0)
            update_display()

        def on_release(event): drag_state['press_xy'] = None

        def on_scroll(event):
            nonlocal angle_deg
            step = 0.5 if event.button == 'up' else -0.5
            angle_deg += step
            update_display()

        def on_key(event):
            nonlocal dx, dy, angle_deg
            if event.key == 'q': angle_deg -= 0.5; update_display()
            elif event.key == 'e': angle_deg += 0.5; update_display()
            elif event.key == 'left': dx -= 0.5; update_display()
            elif event.key == 'right': dx += 0.5; update_display()
            elif event.key == 'up': dy -= 0.5; update_display()
            elif event.key == 'down': dy += 0.5; update_display()
            elif event.key == 'r': dx, dy, angle_deg = 0.0, 0.0, 0.0; update_display()
            elif event.key == 'd': plt.close(fig)

        fig.canvas.mpl_connect('button_press_event', on_press)
        fig.canvas.mpl_connect('motion_notify_event', on_motion)
        fig.canvas.mpl_connect('button_release_event', on_release)
        fig.canvas.mpl_connect('scroll_event', on_scroll)
        fig.canvas.mpl_connect('key_press_event', on_key)

        plt.show(block=True)

        rad = np.deg2rad(angle_deg)
        center = (ch2_norm.shape[1] / 2.0, ch2_norm.shape[0] / 2.0)
        t_center = ski_transform.AffineTransform(translation=-np.array(center))
        t_rot = ski_transform.AffineTransform(rotation=rad)
        t_back = ski_transform.AffineTransform(translation=np.array(center) + np.array([dx, dy]))
        final_tform = t_center + t_rot + t_back

        matrix = final_tform.params.astype(np.float64)

        os.makedirs(save_dir, exist_ok=True)
        out_npy = os.path.join(save_dir, "transformation_matrix.npy")
        np.save(out_npy, matrix)

        out_mat = os.path.join(save_dir, "channel_transform.mat")
        sio.savemat(out_mat, {'T_matrix': matrix.T, 'tform_type': 'affine'})

        self.save_manual_overlay_and_metrics(ch1_norm, ch2_norm, matrix, save_dir, method_label="Interactive Drag Registration")
        return matrix

    def interactive_manual_registration(self, image_path: str, save_dir: str, num_pairs: int = 10):
        target_path = image_path
        if os.path.isdir(image_path) or not os.path.exists(image_path):
            search_pattern = f"{image_path}_*" if not os.path.exists(image_path) else os.path.join(image_path, "*")
            bead_folders = sorted(glob.glob(search_pattern))
            if os.path.isdir(image_path): bead_folders = [image_path] + bead_folders
            found_file = None
            for folder in bead_folders:
                files = glob.glob(os.path.join(folder, "*.ome.tif")) + glob.glob(os.path.join(folder, "*.tif"))
                if files:
                    found_file = files[0]
                    break
            if found_file: target_path = found_file
            else: raise FileNotFoundError(f"Could not locate bead file at: '{image_path}'")

        img = io.imread(target_path)
        if img.ndim == 3: img = np.mean(img, axis=0) 

        if int(num_pairs) <= 0:
            return self._interactive_drag_registration(img, save_dir)
            
        fig, ax = plt.subplots(figsize=(12, 6))
        h, w = img.shape[:2]
        if w >= h and w > 1:
            half_w = w // 2
            ch1, ch2 = img[:, :half_w].astype(float), img[:, half_w:].astype(float)
            ch1_norm = (ch1 - ch1.min()) / (ch1.max() - ch1.min() + 1e-8)
            ch2_norm = (ch2 - ch2.min()) / (ch2.max() - ch2.min() + 1e-8)
            img_display = np.hstack((ch1_norm, ch2_norm))
        else:
            half_w = w
            ch1_norm = (img.astype(float) - img.min()) / (img.max() - img.min() + 1e-8)
            ch2_norm = ch1_norm.copy()
            img_display = ch1_norm

        ax.imshow(img_display, cmap='gray')
        plt.axis('image')
        pts = []

        def on_scroll(event):
            if event.inaxes != ax: return
            xdata, ydata = event.xdata, event.ydata
            if xdata is None or ydata is None: return
            cur_xlim, cur_ylim = ax.get_xlim(), ax.get_ylim()
            scale_factor = (1 / 1.25) if event.button == 'up' else 1.25
            new_w, new_h = (cur_xlim[1] - cur_xlim[0]) * scale_factor, (cur_ylim[1] - cur_ylim[0]) * scale_factor
            rel_x, rel_y = (xdata - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0]), (ydata - cur_ylim[0]) / (cur_ylim[1] - cur_ylim[0])
            ax.set_xlim([xdata - new_w * rel_x, xdata + new_w * (1 - rel_x)])
            ax.set_ylim([ydata - new_h * rel_y, ydata + new_h * (1 - rel_y)])
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect('scroll_event', on_scroll)

        def redraw_points():
            cur_xlim, cur_ylim = ax.get_xlim(), ax.get_ylim()
            while len(ax.lines) > 0: ax.lines[0].remove()
            while len(ax.texts) > 0: ax.texts[0].remove()
            for idx, (x, y) in enumerate(pts):
                pair_num, is_ch1 = (idx // 2) + 1, (idx % 2 == 0)
                color, label_color = ('g+', 'lime') if is_ch1 else ('r+', 'red')
                ax.plot(x, y, color, markersize=6, markeredgewidth=1.5)
                ax.text(x + 2, y - 2, str(pair_num), color=label_color, fontweight='bold', fontsize=7)
            ax.set_title(f"Manual Registration ({len(pts)//2}/{num_pairs} pairs picked)")
            ax.set_xlim(cur_xlim); ax.set_ylim(cur_ylim)
            fig.canvas.draw_idle()

        redraw_points()

        while len(pts) < num_pairs * 2:
            click = plt.ginput(1, timeout=-1, show_clicks=False)
            if not click:
                if len(pts) >= 6 or len(pts) == 0: break
                else:
                    if pts: pts.pop(); redraw_points()
                    continue
            pts.append(click[0])
            redraw_points()

        plt.close(fig)
        if len(pts) % 2 != 0: pts = pts[:-1]
        if len(pts) < 6: raise ValueError("Affine transform requires at least 3 pairs (6 points).")

        pts_ch1, pts_ch2 = [], []
        for i in range(0, len(pts), 2):
            p1, p2 = list(pts[i]), list(pts[i+1])
            if p1[0] > p2[0]: p1, p2 = p2, p1
            if p2[0] >= half_w: p2[0] -= half_w
            pts_ch1.append(p1); pts_ch2.append(p2)

        tform = ski_transform.estimate_transform('affine', src=np.array(pts_ch2), dst=np.array(pts_ch1))
        matrix = tform.params.astype(np.float64)

        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "transformation_matrix.npy"), matrix)
        sio.savemat(os.path.join(save_dir, "channel_transform.mat"), {'T_matrix': matrix.T, 'tform_type': 'affine'})
        self.save_manual_overlay_and_metrics(ch1_norm, ch2_norm, matrix, save_dir, method_label="Manual Point-Picking Registration")
        return matrix

class MaskEditor:
    """Interactive Mask Editor GUI component."""
    def __init__(self, mask, bf_img=None, fl_img=None, image_dict=None, rect_width=14, alpha=0.4, outline_alpha=0.7):
        conflicting_keys = ['f', 'c', 'r', 'z', 'a', 'p', 'x', 'd', '+', '-', '=']
        for param in plt.rcParams:
            if param.startswith('keymap.'):
                for key in conflicting_keys:
                    if key in plt.rcParams[param]:
                        plt.rcParams[param].remove(key)

        if mask.ndim == 3:
            collapsed = np.zeros(mask.shape[1:], dtype=np.int32)
            for idx in range(mask.shape[0]): collapsed[mask[idx] > 0] = idx + 1
            self.mask = ImageProcessor.separate_diagonal_pixels(collapsed)
        else:
            self.mask = ImageProcessor.separate_diagonal_pixels(mask.copy().astype(np.int32))
            unique_vals = np.unique(self.mask)
            if len(unique_vals) <= 2 or (len(unique_vals) == 3 and 0 in unique_vals):
                self.mask = measure.label(self.mask > 0, connectivity=1).astype(np.int32)

        self.image_dict = {}
        if image_dict and isinstance(image_dict, dict):
            for k, v in image_dict.items():
                if v is not None: self.image_dict[k] = v
        if not self.image_dict:
            if bf_img is not None: self.image_dict["Brightfield"] = bf_img
            if fl_img is not None: self.image_dict["Fluorescence"] = fl_img

        self.current_img_key = list(self.image_dict.keys())[0] if self.image_dict else "None"

        self.history, self.current_points = [], []
        self.rect_width, self.alpha, self.outline_alpha = rect_width, alpha, outline_alpha
        self.press_pos, self.last_pos, self.press_button, self.is_dragging = None, None, None, False
        self.rect_patch = None

        self.fig, (self.ax_ref, self.ax) = plt.subplots(1, 2, figsize=(14, 8), sharex=True, sharey=True)
        self.fig.subplots_adjust(top=0.80, bottom=0.05, left=0.05, right=0.98, wspace=0.05)

        if self.fig.canvas.toolbar is not None:
            try:
                self.fig.canvas.toolbar.mode = ''
            except Exception:
                pass

        if len(self.image_dict) > 1:
            self.ax_menu = self.fig.add_axes([0.35, 0.83, 0.30, 0.12], facecolor='#f0f0f0')
            self.ax_menu.set_zorder(999)
            self.img_selector = RadioButtons(self.ax_menu, list(self.image_dict.keys()), active=0)
            
            for lbl in self.img_selector.labels:
                lbl.set_fontsize(9)
                
            self.img_selector.on_clicked(self.on_switch_image)

        self._redraw()
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        try:
            if hasattr(self.fig.canvas, 'manager') and self.fig.canvas.manager is not None:
                self.fig.canvas.manager.window.raise_()
                self.fig.canvas.manager.window.activateWindow()
        except Exception:
            pass

        from PyQt6.QtCore import QEventLoop
        from PyQt6.QtWidgets import QApplication

        self.fig.show()
        if QApplication.instance():
            loop = QEventLoop()
            self.fig.canvas.mpl_connect('close_event', lambda e: loop.quit())
            loop.exec()
        else:
            plt.show(block=True)

    def on_switch_image(self, label):
        self.current_img_key = label
        self._redraw()

    def _draw_base(self):
        self.ax_ref.clear(); self.ax.clear()
        current_img = self.image_dict.get(self.current_img_key, None)
        if current_img is not None:
            cmap = "gray" if current_img.ndim == 2 else None
            self.ax_ref.imshow(current_img, cmap=cmap)
            self.ax.imshow(current_img, cmap=cmap)

        max_label = int(self.mask.max()) if self.mask.size else 0
        rng = np.random.RandomState(42)
        colors = rng.rand(max(max_label, 1) + 1, 3); colors[0] = [0, 0, 0]
        cmap = mcolors.ListedColormap(colors)
        masked = np.ma.masked_where(self.mask == 0, self.mask)
        self.ax.imshow(masked, cmap=cmap, interpolation="nearest", vmin=0, vmax=max(max_label, 1), alpha=self.alpha)
        if max_label > 0:
            self.ax.contour(self.mask > 0, levels=[0.5], colors=["white"], linewidths=0.5, alpha=self.outline_alpha)
        self.ax_ref.set_xticks([]); self.ax_ref.set_yticks([])
        self.ax.set_xticks([]); self.ax.set_yticks([])

    def _set_title(self):
        self.fig.suptitle("Left-Drag: Lasso | Right-Drag: Delete Box | C: Cut | A: Add | P: Poly | F: Fit | R: Reject | Z: Undo | D: Done", fontsize=9, y=0.98)
        self.fig.canvas.draw_idle()

    def on_press(self, event):
        if event.inaxes not in [self.ax, self.ax_ref] or event.button not in [1, 3]: return
        if event.xdata is None or event.ydata is None: return
        self.press_pos = (event.xdata, event.ydata)
        self.last_pos = self.press_pos
        self.press_button = event.button
        self.is_dragging = False

    def on_motion(self, event):
        if self.press_pos is None or event.inaxes not in [self.ax, self.ax_ref]: return
        if event.xdata is None or event.ydata is None: return
        x0, y0 = self.press_pos
        x1, y1 = event.xdata, event.ydata
        self.last_pos = (x1, y1)

        if abs(x1 - x0) > 3 or abs(y1 - y0) > 3:
            if not self.is_dragging:
                self.is_dragging = True
                if self.press_button == 1: self.current_points = [self.press_pos, (x1, y1)]; self._refresh_line()
            else:
                if self.press_button == 1: self.current_points.append((x1, y1)); self._refresh_line()
            
            if self.press_button == 3:
                xmin, xmax = min(x0, x1), max(x0, x1)
                ymin, ymax = min(y0, y1), max(y0, y1)
                if self.rect_patch is not None: self.rect_patch.remove()
                self.rect_patch = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1.5, edgecolor="red", facecolor="red", alpha=0.3, linestyle="--")
                self.ax.add_patch(self.rect_patch)
                self.fig.canvas.draw_idle()

    def on_release(self, event):
        if self.press_pos is None: return
        if self.rect_patch is not None: self.rect_patch.remove(); self.rect_patch = None
        release_x = event.xdata if event.xdata is not None else self.last_pos[0]
        release_y = event.ydata if event.ydata is not None else self.last_pos[1]

        if self.is_dragging:
            if self.press_button == 1: self.add_polygon_contour()
            elif self.press_button == 3:
                x0, y0 = self.press_pos
                x1, y1 = release_x, release_y
                h, w = self.mask.shape
                xmin, xmax = max(0, int(round(min(x0, x1)))), min(w, int(round(max(x0, x1))))
                ymin, ymax = max(0, int(round(min(y0, y1)))), min(h, int(round(max(y0, y1))))
                sub_region = self.mask[ymin:ymax+1, xmin:xmax+1]
                labels_to_delete = set(np.unique(sub_region)) - {0}
                if labels_to_delete:
                    self._save_history()
                    for label_id in labels_to_delete: self.mask[self.mask == label_id] = 0
                    self.mask = ImageProcessor.separate_diagonal_pixels(self.mask)
                    self._redraw()
        elif not self.is_dragging and self.press_button == 1:
            self.current_points.append(self.press_pos)
            self._refresh_line()

        self.press_pos, self.press_button, self.is_dragging = None, None, False

    def _redraw(self):
        self._draw_base()
        self.line_artist, = self.ax.plot([], [], "w-", linewidth=1.5)
        self.point_artist, = self.ax.plot([], [], "wo", markersize=4)
        self._refresh_line()
        self._set_title()

    def _refresh_line(self):
        xs, ys = zip(*self.current_points) if self.current_points else ([], [])
        self.line_artist.set_data(xs, ys)
        self.point_artist.set_data(xs, ys)
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key == "c": self.commit_cut()
        elif event.key == "a": self.add_new_mask()
        elif event.key == "p": self.add_polygon_contour()
        elif event.key == "f": self.fit_and_add_cell_shape()
        elif event.key == "r": self.reject_selected_masks()
        elif event.key == "z": self.undo()
        elif event.key == "x": self.current_points = []; self._refresh_line()
        elif event.key in ["-", "_"]: self.rect_width = max(2, self.rect_width - 2); self._set_title()
        elif event.key in ["+", "="]: self.rect_width += 2; self._set_title()
        elif event.key == "d": plt.close(self.fig)

    def _save_history(self):
        self.history.append(self.mask.copy())
        if len(self.history) > 15: self.history.pop(0)

    def commit_cut(self):
        if len(self.current_points) < 2: return
        pts = np.array(self.current_points)
        line_canvas = np.zeros(self.mask.shape, dtype=np.uint8)
        for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
            cv2.line(line_canvas, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), color=1, thickness=2)

        intersecting_labels = set(np.unique(self.mask[line_canvas == 1])) - {0}
        if not intersecting_labels: return

        self._save_history()
        cuts_performed, max_label = 0, int(self.mask.max())
        for label_id in intersecting_labels:
            region = (self.mask == label_id)
            cut_region = region & (line_canvas == 0)
            pieces, n_pieces = ndi.label(cut_region, structure=CONNECTIVITY_4_STRUCT)
            if n_pieces > 1:
                self.mask[region] = 0
                for piece_id in range(1, n_pieces + 1):
                    max_label += 1
                    self.mask[pieces == piece_id] = max_label
                cuts_performed += 1

        if cuts_performed > 0: 
            self.mask = ImageProcessor.separate_diagonal_pixels(self.mask)
            self.current_points = []
            self._redraw()
        else:
            if self.history: self.history.pop()

    def add_new_mask(self):
        if len(self.current_points) == 2:
            p1, p2 = self.current_points
            canvas = np.zeros(self.mask.shape, dtype=np.uint8)
            x1, y1, x2, y2 = int(round(p1[0])), int(round(p1[1])), int(round(p2[0])), int(round(p2[1]))
            rad = max(1, int(round(self.rect_width / 2)))
            cv2.line(canvas, (x1, y1), (x2, y2), color=1, thickness=self.rect_width)
            cv2.circle(canvas, (x1, y1), rad, color=1, thickness=-1)
            cv2.circle(canvas, (x2, y2), rad, color=1, thickness=-1)
            new_shape = canvas.astype(bool)
        elif len(self.current_points) >= 3:
            pts = np.round(np.array(self.current_points)).astype(np.int32)
            canvas = np.zeros(self.mask.shape, dtype=np.uint8)
            cv2.fillPoly(canvas, [pts], color=1)
            new_shape = canvas.astype(bool)
        else: return

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        forbidden_zone = cv2.dilate((self.mask > 0).astype(np.uint8), kernel, iterations=1)
        new_region = new_shape & (forbidden_zone == 0)
        if not np.any(new_region): return

        self._save_history()
        new_label = int(self.mask.max()) + 1
        self.mask[new_region] = new_label
        self.mask = ImageProcessor.separate_diagonal_pixels(self.mask)
        self.current_points = []
        self._redraw()

    def add_polygon_contour(self):
        if len(self.current_points) < 3: return
        pts = np.round(np.array(self.current_points)).astype(np.int32)
        canvas = np.zeros(self.mask.shape, dtype=np.uint8)
        cv2.fillPoly(canvas, [pts], color=1)
        kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel_smooth)
        new_shape = canvas.astype(bool)

        kernel_gap = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        forbidden_zone = cv2.dilate((self.mask > 0).astype(np.uint8), kernel_gap, iterations=1)
        new_region = new_shape & (forbidden_zone == 0)
        if not np.any(new_region): return

        self._save_history()
        new_label = int(self.mask.max()) + 1
        self.mask[new_region] = new_label
        self.mask = ImageProcessor.separate_diagonal_pixels(self.mask)
        self.current_points = []
        self._redraw()

    def fit_and_add_cell_shape(self):
        if len(self.current_points) < 2: return
        new_shape = ImageProcessor.fit_and_generate_cell_shape(self.current_points, self.mask.shape)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        forbidden_zone = cv2.dilate((self.mask > 0).astype(np.uint8), kernel, iterations=1)
        new_region = new_shape & (forbidden_zone == 0)
        if not np.any(new_region): return
            
        self._save_history()
        new_label = int(self.mask.max()) + 1
        self.mask[new_region] = new_label
        self.mask = ImageProcessor.separate_diagonal_pixels(self.mask)
        self.current_points = []
        self._redraw()

    def reject_selected_masks(self):
        """Rejects ONLY the individual cell mask labels intersecting the selected points or drawn boundary."""
        if not self.current_points: return
        h, w = self.mask.shape
        labels_to_reject = set()

        if len(self.current_points) == 1:
            pt = self.current_points[0]
            r, c = int(round(pt[1])), int(round(pt[0]))
            if 0 <= r < h and 0 <= c < w:
                lbl = self.mask[r, c]
                if lbl != 0: labels_to_reject.add(lbl)
        elif len(self.current_points) == 2:
            pts = np.round(np.array(self.current_points)).astype(np.int32)
            canvas = np.zeros(self.mask.shape, dtype=np.uint8)
            cv2.line(canvas, (pts[0][0], pts[0][1]), (pts[1][0], pts[1][1]), color=1, thickness=2)
            labels_to_reject = set(np.unique(self.mask[canvas == 1])) - {0}
        elif len(self.current_points) >= 3:
            pts = np.round(np.array(self.current_points)).astype(np.int32)
            canvas = np.zeros(self.mask.shape, dtype=np.uint8)
            cv2.fillPoly(canvas, [pts], color=1)
            labels_to_reject = set(np.unique(self.mask[canvas == 1])) - {0}

        if not labels_to_reject: return

        self._save_history()
        for lbl in labels_to_reject: 
            self.mask[self.mask == lbl] = 0
        self.mask = ImageProcessor.separate_diagonal_pixels(self.mask)
        self.current_points = []
        self._redraw()

    def undo(self):
        if not self.history: return
        self.mask = self.history.pop()
        self.current_points = []
        self._redraw()

class AnalysisPipeline:
    def __init__(self, params):
        self.args = params
        self.pxsize = getattr(params, 'pixel_size', getattr(params, 'pxsize', 0.120))
        self.ALEX = getattr(params, 'ALEX', False)
        self.save_dir = getattr(params, 'save_dir', None) or getattr(params, 'name', 'results')
        self.channel = getattr(params, 'use_channel', getattr(params, 'channel', 'L'))
        self.num_frames = None if getattr(params, 'num_frames', 0) == 0 else getattr(params, 'num_frames', None)
        self.frame_avg = getattr(params, 'frame_avg', 5)

        self.video_path = getattr(params, 'video_path', '')
        self.bead_path = getattr(params, 'bead_path', '')
        self.bf_path = getattr(params, 'bf_path', '')
        self.tmats_path = getattr(params, 'tmats_path', 'transformation_matrix.npy')
        self.manual_registration = getattr(params, 'manual_registration', False)
        self.manual_pairs = getattr(params, 'manual_pairs', 10)
        self.roi_channel = getattr(params, 'roi_channel', 'L')
        self.roi_file = getattr(params, 'roi_file', '')
        self.mask_prefix = getattr(params, 'mask_prefix', 'mask')
        self.overwrite = getattr(params, 'overwrite', True)
        self.use_otsu = getattr(params, 'use_otsu', 'False')
        self.cell_fitting = getattr(params, 'cell_fitting', False)
        self.area_filter = int(getattr(params, 'area_filter', 100))
        self.inv_bf = getattr(params, 'inv_bf', False)
        self.model_type = getattr(params, 'model_type', 'unet')
        self.raw_shape = None

        self.model = None
        if getattr(params, 'mask_type', 'AI') in ["AI", "BF", "FL_AI"]:
            model_val = getattr(params, 'model', '')
            if isinstance(model_val, str) and model_val.startswith(("http://", "https://")):
                parsed_url = urlparse(model_val)
                filename = os.path.basename(parsed_url.path) or "downloaded_model"
                os.makedirs(self.save_dir, exist_ok=True)
                local_model_path = os.path.join(self.save_dir, filename)
                if not os.path.exists(local_model_path):
                    req = urllib.request.Request(model_val, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req) as response, open(local_model_path, 'wb') as out_file:
                        shutil.copyfileobj(response, out_file)
                params.model = local_model_path
                model_val = local_model_path

            if self.model_type in ["unet", "keras"] and isinstance(model_val, str) and os.path.exists(model_val):
                try:
                    self.model = load_model(model_val, compile=False, safe_mode=False)
                except Exception:
                    try:
                        import tf_keras as tfk
                        self.model = tfk.models.load_model(model_val)
                    except Exception: pass
            elif self.model_type == "pytorch" and isinstance(model_val, str) and os.path.exists(model_val):
                self.model = torch.load(model_val, map_location=torch.device('cpu'))
                self.model.eval()

        os.makedirs(self.save_dir, exist_ok=True)

    def edit_existing_mask(self):
        """Loads existing output masks, launches interactive MaskEditor, and updates all saved files."""
        mask_path = os.path.join(self.save_dir, "all_cells_mask.tif")
        if not os.path.exists(mask_path):
            print(f"[AnalysisPipeline] ERROR: No existing mask found at '{mask_path}' to edit.", flush=True)
            return None

        existing_mask = tifffile.imread(mask_path)
        
        def _load_bg_image(name_base):
            for ext in [".tif", ".png", ".tiff"]:
                p = os.path.join(self.save_dir, f"{name_base}{ext}")
                if os.path.exists(p):
                    try:
                        return tifffile.imread(p) if ext.startswith(".tif") else io.imread(p)
                    except Exception:
                        pass
            return None

        image_dict = {}
        if self.channel == "R":
            bf_r_img = _load_bg_image("BF_R")
            if bf_r_img is not None: 
                image_dict["Brightfield (BF_R)"] = bf_r_img
            else:
                bf_img = _load_bg_image("BF")
                if bf_img is not None: image_dict["Brightfield (BF)"] = bf_img

            r_avg_img = _load_bg_image("R_avg")
            if r_avg_img is not None: 
                image_dict["Fluorescence (R_avg)"] = r_avg_img
            else:
                l_avg_img = _load_bg_image("L_avg")
                if l_avg_img is not None: image_dict["Fluorescence (L_avg)"] = l_avg_img
        else:
            bf_img = _load_bg_image("BF")
            if bf_img is not None: image_dict["Brightfield (BF)"] = bf_img

            l_avg_img = _load_bg_image("L_avg")
            if l_avg_img is not None: image_dict["Fluorescence (L_avg)"] = l_avg_img

        raw_edited_mask = ImageProcessor.interactive_edit_mask(
            mask=existing_mask, bg_img=image_dict.get(list(image_dict.keys())[0], None), image_dict=image_dict
        )
        
        mask = (ImageProcessor.separate_diagonal_pixels(raw_edited_mask) > 0).astype(np.uint8) * 255
        tmats = self.get_transformation_matrix()

        # Re-save primary binary mask
        tifffile.imwrite(mask_path, mask.astype(np.uint16), imagej=True)
        self.save_image_with_scalebar(mask, f"{self.save_dir}/all_cells_mask.png", cmap="viridis")

        # Re-transform L and R channel masks with strict diagonal pixel separation
        mask_l = mask if self.channel == "L" else ImageProcessor.transform_mask(mask, tmats, src_channel=self.channel, dst_channel="L", ref_channel=self.channel)
        mask_r = mask if self.channel == "R" else ImageProcessor.transform_mask(mask, tmats, src_channel=self.channel, dst_channel="R", ref_channel=self.channel)

        tifffile.imwrite(f"{self.save_dir}/all_cells_mask_L.tif", mask_l.astype(np.uint16), imagej=True)
        self.save_image_with_scalebar(mask_l, f"{self.save_dir}/all_cells_mask_L.png", cmap="viridis")

        if int(getattr(self.args, 'num_channels', 1)) == 2 or os.path.exists(os.path.join(self.save_dir, "R_avg.png")):
            tifffile.imwrite(f"{self.save_dir}/all_cells_mask_R.tif", mask_r.astype(np.uint16), imagej=True)
            self.save_image_with_scalebar(mask_r, f"{self.save_dir}/all_cells_mask_R.png", cmap="viridis")

        # Re-save labeled integer masks using connectivity=1 (4-connectivity)
        labeled_mask = measure.label(mask > 0, connectivity=1).astype(np.uint16)
        labeled_mask_l = measure.label(mask_l > 0, connectivity=1).astype(np.uint16)
        labeled_mask_r = measure.label(mask_r > 0, connectivity=1).astype(np.uint16)

        tifffile.imwrite(f"{self.save_dir}/all_cells_mask_labeled.tif", labeled_mask, imagej=True)
        tifffile.imwrite(f"{self.save_dir}/all_cells_mask_L_labeled.tif", labeled_mask_l, imagej=True)
        if int(getattr(self.args, 'num_channels', 1)) == 2 or os.path.exists(os.path.join(self.save_dir, "R_avg.png")):
            tifffile.imwrite(f"{self.save_dir}/all_cells_mask_R_labeled.tif", labeled_mask_r, imagej=True)

        # Re-save colored visualization masks
        if labeled_mask.max() > 0:
            rng = np.random.RandomState(42)
            max_lbl = max(labeled_mask.max(), labeled_mask_l.max(), labeled_mask_r.max())
            colors = rng.randint(60, 256, size=(max_lbl + 1, 3), dtype=np.uint8)
            colors[0] = [0, 0, 0]
            io.imsave(f"{self.save_dir}/all_cells_mask_colored.png", colors[labeled_mask])
            io.imsave(f"{self.save_dir}/all_cells_mask_L_colored.png", colors[labeled_mask_l])
            if int(getattr(self.args, 'num_channels', 1)) == 2 or os.path.exists(os.path.join(self.save_dir, "R_avg.png")):
                io.imsave(f"{self.save_dir}/all_cells_mask_R_colored.png", colors[labeled_mask_r])

        # Regenerate visual overlays
        l_avg = _load_bg_image("L_avg")
        r_avg = _load_bg_image("R_avg")
        bf_img = _load_bg_image("BF")
        if l_avg is not None:
            ImageProcessor.generate_channel_overlay(l_avg, mask_l, f"{self.save_dir}/L_mask_overlay.png", color=(0, 255, 255))
        if r_avg is not None:
            ImageProcessor.generate_channel_overlay(r_avg, mask_r, f"{self.save_dir}/R_mask_overlay.png", color=(0, 255, 255))
        if bf_img is not None:
            ImageProcessor.generate_channel_overlay(bf_img, mask, f"{self.save_dir}/BF_mask_overlay.png", color=(255, 255, 0))

        print(f"[AnalysisPipeline] Updated edited mask saved successfully to '{self.save_dir}'.", flush=True)
        return mask

    def get_transformation_matrix(self):
        video_dir = os.path.dirname(self.video_path) if self.video_path else ""
        parent_dir = os.path.dirname(video_dir) if video_dir else ""
        grandparent_dir = os.path.dirname(parent_dir) if parent_dir else ""
        
        tmat_candidates = [
            os.path.join(self.save_dir, "transformation_matrix.npy"),
            getattr(self, 'tmats_path', None),
            os.path.join(video_dir, "transformation_matrix.npy") if video_dir else None,
            os.path.join(parent_dir, "transformation_matrix.npy") if parent_dir else None,
            os.path.join(grandparent_dir, "transformation_matrix.npy") if grandparent_dir else None,
            "transformation_matrix.npy"
        ]
        
        for cand in tmat_candidates:
            if cand and os.path.exists(cand):
                try:
                    tmats = np.load(cand)
                    if isinstance(tmats, np.ndarray) and tmats.shape == (3, 3):
                        print(f"[AnalysisPipeline] Loaded shared transformation matrix from '{cand}'.", flush=True)
                        return np.ascontiguousarray(tmats, dtype=np.float64)
                except Exception as e:
                    print(f"[AnalysisPipeline] Error reading '{cand}': {e}", flush=True)
    
        print("[AnalysisPipeline] WARNING: No valid 3x3 transformation_matrix.npy found.", flush=True)
        return None

    def process_brightfield(self, raw_shape, sr, tmats):
        bf_sum, bf_cropped, bf_R_cropped = None, None, None
        if self.bf_path and os.path.exists(self.bf_path):
            bf_img = img_as_float64(io.imread(self.bf_path))
            bf_sum = np.sum(bf_img, axis=0) if bf_img.ndim == 3 else bf_img
            
            bf_height, bf_width = bf_sum.shape[0], bf_sum.shape[1]
            is_split_view = (int(getattr(self.args, 'num_channels', 1)) == 2) or (bf_width >= bf_height * 1.2 and bf_width >= 500)
            bf_width_split = bf_width // 2 if is_split_view else bf_width
            
            if is_split_view:
                L_bf_full = bf_sum[:, :bf_width_split]
                R_bf_full = bf_sum[:, bf_width_split:]
                
                if tmats is not None:
                    print(f"[DEBUG BF] tmats shape: {tmats.shape}, dtype: {tmats.dtype}", flush=True)
                    print(f"[DEBUG BF] Transformation Matrix:\n{tmats}", flush=True)
                    try:
                        try:
                            if sr is not None:
                                print(f"[DEBUG BF] Attempting StackReg AFFINE transform (Channel Target: {self.channel})...", flush=True)
                                if self.channel == "L":
                                    R_reg = sr.transform(R_bf_full, tmats)
                                    R_reg[R_reg == 0] = np.mean(R_bf_full)
                                    bf_overlay_full = (L_bf_full + R_reg) / 2.0
                                else:
                                    L_reg = sr.transform(L_bf_full, tmats)
                                    L_reg[L_reg == 0] = np.mean(L_bf_full)
                                    bf_overlay_full = (R_bf_full + L_reg) / 2.0
                                print("[DEBUG BF] Applied StackReg AFFINE matrix successfully.", flush=True)
                            else:
                                raise ValueError("StackReg instance is None")
                        except Exception as sr_err:
                            print(f"[DEBUG BF] StackReg transform failed ({sr_err}). Falling back to skimage.transform.warp...", flush=True)
                            if isinstance(tmats, np.ndarray) and tmats.shape == (3, 3):
                                tform = ski_transform.AffineTransform(matrix=tmats)
                                moving_img = R_bf_full if self.channel == "L" else L_bf_full
                                warped = ski_transform.warp(moving_img, tform.inverse, preserve_range=True)
                                warped[warped == 0] = np.mean(moving_img)
                                
                                if self.channel == "L":
                                    R_reg = warped
                                    bf_overlay_full = (L_bf_full + R_reg) / 2.0
                                else:
                                    L_reg = warped
                                    bf_overlay_full = (R_bf_full + L_reg) / 2.0
                                print("[DEBUG BF] Applied scikit-image AffineTransform (tform.inverse) successfully.", flush=True)
                            else:
                                raise sr_err
                    except Exception as e:
                        print(f"[WARNING BF] Registration failed: {e}. Falling back to unaligned overlay.", flush=True)
                        bf_overlay_full = (L_bf_full + R_bf_full) / 2.0
                else:
                    print("[WARNING BF] No tmats supplied (tmats is None). Using unaligned overlay.", flush=True)
                    bf_overlay_full = (L_bf_full + R_bf_full) / 2.0

                y1, y2, x1, x2 = ImageProcessor.read_roi(self.roi_file, self.roi_channel, self.channel, bf_height, bf_width, is_split_view=is_split_view)
                unreg_viz = cv2.normalize((L_bf_full + R_bf_full) / 2.0, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                unreg_bgr = cv2.cvtColor(unreg_viz, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(unreg_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imwrite(f"{self.save_dir}/BF_unregistered_roi.png", unreg_bgr)

                fig, axs = plt.subplots(3, 1, figsize=(10, 15))
                axs[0].imshow((cv2.normalize(L_bf_full, None, 0, 1, cv2.NORM_MINMAX) + cv2.normalize(R_bf_full, None, 0, 1, cv2.NORM_MINMAX)) / 2, cmap='gray')
                axs[0].set_title('Before: Original BF Images (Unaligned Overlay)')
                axs[0].set_xticks([]); axs[0].set_yticks([])

                moving_reg = R_reg if (self.channel == "L" and 'R_reg' in locals()) else (L_reg if 'L_reg' in locals() else R_bf_full)
                axs[1].imshow(cv2.normalize(moving_reg, None, 0, 1, cv2.NORM_MINMAX), cmap='gray')
                axs[1].set_title('Registered Moving Channel')
                axs[1].set_xticks([]); axs[1].set_yticks([])

                axs[2].imshow(cv2.normalize(bf_overlay_full, None, 0, 1, cv2.NORM_MINMAX), cmap='gray')
                axs[2].set_title('After: Final Registered BF Overlay')
                axs[2].set_xticks([]); axs[2].set_yticks([])
                plt.savefig(f"{self.save_dir}/BF_Registered_overlay.png", bbox_inches='tight')
                plt.close()

                bf_R_cropped = R_bf_full[y1:y2, x1:x2]
            else:
                bf_overlay_full = bf_sum
                y1, y2, x1, x2 = ImageProcessor.read_roi(self.roi_file, self.roi_channel, self.channel, bf_height, bf_width, is_split_view=False)
                bf_R_cropped = bf_overlay_full[y1:y2, x1:x2]

            bf_cropped = bf_overlay_full[y1:y2, x1:x2]
            bf_norm = cv2.normalize(bf_cropped, None, 0, 65535, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
            tifffile.imwrite(f"{self.save_dir}/BF.tif", bf_norm, imagej=True)
            self.save_image_with_scalebar(bf_norm, f"{self.save_dir}/BF.png", cmap="gray")
            tifffile.imwrite(f"{self.save_dir}/BF_reg_full.tif", img_as_uint(np.clip(cv2.normalize(bf_overlay_full, None, 0, 1, cv2.NORM_MINMAX), 0, 1)), imagej=True)

            bf_R_norm = cv2.normalize(bf_R_cropped, None, 0, 65535, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
            tifffile.imwrite(f"{self.save_dir}/BF_R.tif", bf_R_norm, imagej=True)
            self.save_image_with_scalebar(bf_R_norm, f"{self.save_dir}/BF_R.png", cmap="gray")

        return bf_sum, bf_cropped, bf_R_cropped

    def save_image_with_scalebar(self, image, save_path, cmap="gray", bar_length_um=1):
        fig, ax = plt.subplots()
        ax.imshow(image, cmap=cmap)
        ax.axis('off')
        pixels = bar_length_um / self.pxsize 
        scalebar = AnchoredSizeBar(
            ax.transData, pixels, f'{bar_length_um} µm', 'lower right', 
            pad=0.5, color='white', frameon=False, 
            size_vertical=max(1, int(image.shape[0] * 0.01)),
            fontproperties=fm.FontProperties(size=12)
        )
        ax.add_artist(scalebar)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

    def fit_cell(self, cell_mask, obj_num, channel, save_dir):
        contours = measure.find_contours(cell_mask, 0.5)
        if not contours: return None, None, None
        contour = max(contours, key=len)
        
        y_data, x_data = contour.T
        pts = np.vstack([x_data, y_data]).T
        mean = np.mean(pts, axis=0)

        cov = np.cov(pts, rowvar=False)
        evals, evecs = np.linalg.eigh(cov)
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:, idx]
        initial_theta = np.arctan2(evecs[1, 0], evecs[0, 0])

        centered = pts - mean
        proj_major, proj_minor = centered @ evecs[:, 0], centered @ evecs[:, 1]
        major_span = np.max(proj_major) - np.min(proj_major)
        minor_span = np.max(proj_minor) - np.min(proj_minor)

        initial_R = max(1.0, minor_span / 2.0)
        initial_L = max(0.1, major_span - 2.0 * initial_R)
        initial_xc, initial_yc = mean[0], mean[1]
        
        max_R = max(initial_R * 3.0, 50.0)
        max_L = max(initial_L * 3.0, 200.0)
        
        lower_bounds = [0.1, 0.0, -np.inf, -np.inf, -np.pi]
        upper_bounds = [max_R, max_L, np.inf, np.inf, np.pi]
        
        def residuals_capsule(params, x_data, y_data):
            R, L, xc, yc, theta = params
            dx, dy = x_data - xc, y_data - yc
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            x_local = dx * cos_t + dy * sin_t
            y_local = -dx * sin_t + dy * cos_t
            x_clamped = np.clip(x_local, -L / 2.0, L / 2.0)
            dist_to_segment = np.sqrt((x_local - x_clamped)**2 + y_local**2)
            return dist_to_segment - R

        p0 = [initial_R, initial_L, initial_xc, initial_yc, initial_theta]
        res = optimize.least_squares(residuals_capsule, p0, args=(x_data, y_data), bounds=(lower_bounds, upper_bounds))
        R_fit, L_fit, xc_fit, yc_fit, theta_fit = res.x
        fit_error = np.sqrt(np.mean(res.fun**2))
        
        scale_factor = 1e9 if self.pxsize < 1e-4 else 1000
        cell_length = (2 * R_fit + L_fit) * self.pxsize * scale_factor
        cell_width = 2 * R_fit * self.pxsize * scale_factor
        error_nm = fit_error * self.pxsize * scale_factor

        t_cap = np.linspace(-np.pi/2, np.pi/2, 100)
        x_right_cap, y_right_cap = L_fit/2 + R_fit * np.cos(t_cap), R_fit * np.sin(t_cap)
        x_left_cap, y_left_cap = -L_fit/2 - R_fit * np.cos(t_cap), -R_fit * np.sin(t_cap)
        x_m = np.concatenate([x_right_cap, x_left_cap, [x_right_cap[0]]])
        y_m = np.concatenate([y_right_cap, y_left_cap, [y_right_cap[0]]])

        x_fitted = xc_fit + x_m * np.cos(theta_fit) - y_m * np.sin(theta_fit)
        y_fitted = yc_fit + x_m * np.sin(theta_fit) + y_m * np.cos(theta_fit)

        plt.figure(figsize=(8, 6))
        plt.imshow(cell_mask, cmap='binary', alpha=0.5)
        plt.plot(x_data, y_data, 'b.', markersize=2, alpha=0.5, label='Mask Contour')
        plt.plot(x_fitted, y_fitted, 'r-', linewidth=2, label='Fitted Outline')
        
        title_text = (
            f'Fitted Cell Outline {obj_num}\n'
            rf'Length={cell_length:.2f} nm, Width={cell_width:.2f} nm, '
            rf'Center=({xc_fit:.2f}, {yc_fit:.2f}), $\theta$={np.rad2deg(theta_fit):.1f}° | '
            f'Error={error_nm:.3f} nm'
        )
        plt.title(title_text)
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.savefig(f"{save_dir}/cell_mask_{obj_num}_{channel}_fitted.png")
        plt.close()

        fit_params = {
            'Cell_Number': obj_num, 'Channel': channel,
            'Length_nm': round(cell_length, 2), 'Width_nm': round(cell_width, 2),
            'Center_X': round(xc_fit, 2), 'Center_Y': round(yc_fit, 2),
            'Rotation_deg': round(np.rad2deg(theta_fit), 2), 'Error_nm': round(error_nm, 3)
        }
        return fit_params, x_fitted, y_fitted

    def extract_skeleton_centerlines(self, fitted_mask, num_lines=5, spacing=1):
        from skimage.morphology import skeletonize
        from skimage.draw import line

        height, width = fitted_mask.shape
        skeleton_mask = skeletonize(fitted_mask)
        out_arr_skeleton = np.argwhere(skeleton_mask)
        if len(out_arr_skeleton) == 0: raise ValueError("Skeletonization failed.")
            
        y_coords, x_coords = out_arr_skeleton[:, 0], out_arr_skeleton[:, 1]
        slope, intercept = np.polyfit(x_coords, y_coords, 1)

        x0, y0 = 0, int(round(slope * 0 + intercept))
        x1, y1 = width - 1, int(round(slope * (width - 1) + intercept))
        rr, cc = line(y0, x0, y1, x1)
        
        half_n = num_lines // 2
        offsets = [i * spacing for i in range(-half_n, half_n + 1)]
        theta = np.arctan(slope)
        dx, dy = -np.sin(theta), np.cos(theta)
        
        all_rr = np.array([np.round(rr + offset * dy).astype(int) for offset in offsets])
        all_cc = np.array([np.round(cc + offset * dx).astype(int) for offset in offsets])
        
        valid_cols = ((all_rr >= 0) & (all_rr < height) & (all_cc >= 0) & (all_cc < width)).all(axis=0)
        all_rr_valid, all_cc_valid = all_rr[:, valid_cols], all_cc[:, valid_cols]
        
        center_line_idx = num_lines // 2
        mask_line_values = fitted_mask[all_rr_valid[center_line_idx], all_cc_valid[center_line_idx]]
        mask_indices = np.where(mask_line_values > 0)[0]
        if len(mask_indices) == 0: raise ValueError("No cell mask detected along centerline!")
            
        start_idx = max(0, mask_indices[0] - 5)
        end_idx = min(len(mask_line_values), mask_indices[-1] + 5 + 1)
        
        return {
            'all_rr_valid': all_rr_valid, 'all_cc_valid': all_cc_valid,
            'start_idx': start_idx, 'end_idx': end_idx,
            'skeleton_mask': skeleton_mask, 'slope': slope, 'intercept': intercept
        }

    def line_profile(self, fitted_mask_file, save_dir=".", is_ALEX=False, alex_start_frame=0):
        fitted_mask = tifffile.imread(fitted_mask_file)
        num_channels = int(getattr(self.args, 'num_channels', 1))

        skel = self.extract_skeleton_centerlines(fitted_mask, num_lines=5, spacing=1)
        all_rr_valid, all_cc_valid = skel['all_rr_valid'], skel['all_cc_valid']
        start_idx, end_idx = skel['start_idx'], skel['end_idx']

        tmats = self.get_transformation_matrix()

        def get_channel_avg(raw_path):
            if not os.path.exists(raw_path): return None
            raw = tifffile.imread(raw_path)
            if raw.ndim == 3:
                frames = raw[alex_start_frame::2] if is_ALEX else raw
                return np.mean(frames, axis=0)
            return raw

        def compute_stats(raw_img, ch_label):
            if ch_label != self.channel and tmats is not None:
                tform = ski_transform.AffineTransform(matrix=tmats)
                pts_src = np.column_stack((all_cc_valid.ravel(), all_rr_valid.ravel()))
                
                if self.channel == "L":
                    pts_target = tform(pts_src) if ch_label == "R" else pts_src
                else:
                    pts_target = tform(pts_src) if ch_label == "L" else pts_src

                rr_chan = np.clip(np.round(pts_target[:, 1]).astype(int), 0, raw_img.shape[0] - 1).reshape(all_rr_valid.shape)
                cc_chan = np.clip(np.round(pts_target[:, 0]).astype(int), 0, raw_img.shape[1] - 1).reshape(all_cc_valid.shape)
            else:
                rr_chan, cc_chan = all_rr_valid, all_cc_valid

            intensities = np.array([raw_img[r, c] for r, c in zip(rr_chan, cc_chan)])
            cropped = intensities[:, start_idx:end_idx]
            full_len = cropped.shape[1]
            x_full = np.arange(full_len)
            mean_full = np.mean(cropped, axis=0)
            sem_full = np.std(cropped, axis=0, ddof=1) / np.sqrt(5)
            
            mid = full_len // 2
            left_half = cropped[:, :mid]
            second_half_flipped = cropped[:, mid:mid*2][:, ::-1]
            
            x_half = np.arange(mid)
            mean_left = np.mean(left_half, axis=0)
            sem_left = np.std(left_half, axis=0, ddof=1) / np.sqrt(5)
            mean_right = np.mean(second_half_flipped, axis=0)
            sem_right = np.std(second_half_flipped, axis=0, ddof=1) / np.sqrt(5)
            
            return {'cropped': cropped, 'x_full': x_full, 'mean_full': mean_full, 'sem_full': sem_full,
                    'x_half': x_half, 'mean_left': mean_left, 'sem_left': sem_left, 'mean_right': mean_right, 'sem_right': sem_right}

        def save_single_channel_outputs(ch_label, stats, color_l="blue", color_r="orange", color_full="green"):
            plt.figure(figsize=(8, 5))
            plt.plot(stats['x_half'], stats['mean_left'], label="Left Half", color=color_l)
            plt.fill_between(stats['x_half'], stats['mean_left'] - stats['sem_left'], stats['mean_left'] + stats['sem_left'], color=color_l, alpha=0.2)
            plt.plot(stats['x_half'], stats['mean_right'], label="Right Half (Flipped)", color=color_r)
            plt.fill_between(stats['x_half'], stats['mean_right'] - stats['sem_right'], stats['mean_right'] + stats['sem_right'], color=color_r, alpha=0.2)
            plt.title(f"Channel {ch_label}: Left Half vs Right Half Profile")
            plt.xlabel("Pixel Distance from Edge/Center")
            plt.ylabel("Fluorescence Intensity")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"left_vs_right_half_profile_{ch_label}.png"))
            plt.close()

            plt.figure(figsize=(10, 5))
            plt.plot(stats['x_full'], stats['mean_full'], label=f"Channel {ch_label}", color=color_full)
            plt.fill_between(stats['x_full'], stats['mean_full'] - stats['sem_full'], stats['mean_full'] + stats['sem_full'], color=color_full, alpha=0.2)
            plt.title(f"Channel {ch_label}: Full Length Line Profile")
            plt.xlabel("Pixel Distance along Centerline")
            plt.ylabel("Fluorescence Intensity")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"full_length_line_profile_{ch_label}.png"))
            plt.close()

            df = pd.DataFrame({
                "Pixel_Index": stats['x_full'],
                f"Channel_{ch_label}_Line_-2": stats['cropped'][0],
                f"Channel_{ch_label}_Line_-1": stats['cropped'][1],
                f"Channel_{ch_label}_Line_0": stats['cropped'][2],
                f"Channel_{ch_label}_Line_+1": stats['cropped'][3],
                f"Channel_{ch_label}_Line_+2": stats['cropped'][4],
                f"Channel_{ch_label}_Mean": stats['mean_full'],
                f"Channel_{ch_label}_SEM": stats['sem_full']
            })
            df.to_csv(os.path.join(save_dir, f"line_profile_{ch_label}.csv"), index=False)

        channels_to_process = [self.channel] if num_channels == 1 else ["L", "R"]
        stats_dict = {}

        for ch in channels_to_process:
            ch_file = os.path.join(self.save_dir, f"{ch}_channel.tif")
            raw_avg = get_channel_avg(ch_file)
            if raw_avg is not None:
                stats = compute_stats(raw_avg, ch)
                stats_dict[ch] = stats
                ch_color = "green" if ch == "L" else "red"
                save_single_channel_outputs(ch, stats, color_full=ch_color)

        if num_channels == 2 and "L" in stats_dict and "R" in stats_dict:
            stats_L = stats_dict["L"]
            stats_R = stats_dict["R"]

            plt.figure(figsize=(10, 5))
            plt.plot(stats_L['x_full'], stats_L['mean_full'], label="Channel L", color="green")
            plt.fill_between(stats_L['x_full'], stats_L['mean_full'] - stats_L['sem_full'], stats_L['mean_full'] + stats_L['sem_full'], color="green", alpha=0.2)
            plt.plot(stats_R['x_full'], stats_R['mean_full'], label="Channel R", color="red")
            plt.fill_between(stats_R['x_full'], stats_R['mean_full'] - stats_R['sem_full'], stats_R['mean_full'] + stats_R['sem_full'], color="red", alpha=0.2)
            plt.title("Full Cell Profile: Channel L vs Channel R")
            plt.xlabel("Pixel Distance along Centerline")
            plt.ylabel("Fluorescence Intensity")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "full_length_line_profile_combined.png"))
            plt.close()

            df_combined = pd.DataFrame({
                "Pixel_Index": stats_L['x_full'],
                "Channel_L_Line_-2": stats_L['cropped'][0],
                "Channel_L_Line_-1": stats_L['cropped'][1],
                "Channel_L_Line_0": stats_L['cropped'][2],
                "Channel_L_Line_+1": stats_L['cropped'][3],
                "Channel_L_Line_+2": stats_L['cropped'][4],
                "Channel_L_Mean": stats_L['mean_full'],
                "Channel_L_SEM": stats_L['sem_full'],
                "Channel_R_Line_-2": stats_R['cropped'][0],
                "Channel_R_Line_-1": stats_R['cropped'][1],
                "Channel_R_Line_0": stats_R['cropped'][2],
                "Channel_R_Line_+1": stats_R['cropped'][3],
                "Channel_R_Line_+2": stats_R['cropped'][4],
                "Channel_R_Mean": stats_R['mean_full'],
                "Channel_R_SEM": stats_R['sem_full'],
            })
            df_combined.to_csv(os.path.join(save_dir, "line_profile_combined.csv"), index=False)

    def create_interactive_html(self, images_dict, objects, fit_results_list, fit_outlines_dict, tmats, save_dir):
        valid_images = {}
        for name, img_data in images_dict.items():
            if img_data is not None:
                valid_images[name] = cv2.normalize(img_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if not valid_images: return
        fig = go.Figure()
        image_keys = list(valid_images.keys())
        num_img_traces = len(image_keys)

        for idx, (img_name, norm_img) in enumerate(valid_images.items()):
            rgb_img = np.dstack([norm_img, norm_img, norm_img]) if norm_img.ndim == 2 else norm_img
            fig.add_trace(go.Image(z=rgb_img, name=img_name, visible=(idx == 0), hoverinfo='skip'))

        fit_dict = {f['Cell_Number']: f for f in fit_results_list}
        area_thresh = int(self.area_filter)

        for obj in objects:
            status = "Accepted" if obj.area > area_thresh else "Rejected"
            fill_color = "rgba(255, 255, 0, 0.2)" if status == "Accepted" else "rgba(255, 0, 0, 0.2)"
            line_color = "yellow" if status == "Accepted" else "red"
            obj_num = obj.label

            if obj.image.shape[0] < 2 or obj.image.shape[1] < 2: continue

            fit_data = fit_dict.get(obj_num)
            fit_outline = fit_outlines_dict.get(obj_num)

            hover_text = f"<b>Object {obj_num}</b><br>Status: {status}<br>Area: {obj.area} px<br>"
            if fit_data:
                hover_text += f"Length: {fit_data['Length_nm']} nm<br>Width: {fit_data['Width_nm']} nm<br>Error: {fit_data['Error_nm']} nm"

            min_y, min_x, max_y, max_x = obj.bbox
            padded_image = np.pad(obj.image, pad_width=1, mode='constant', constant_values=0)
            contours = measure.find_contours(padded_image, 0.5)

            for contour in contours:
                global_y, global_x = contour[:, 0] + min_y - 1, contour[:, 1] + min_x - 1
                fig.add_trace(go.Scatter(
                    x=global_x, y=global_y, mode='lines', fill='toself',
                    fillcolor=fill_color, line=dict(color=line_color, width=1.5),
                    name=f"Cell {obj_num}", text=hover_text, hoverinfo="text", showlegend=False, visible=True
                ))

            if fit_outline and status == "Accepted":
                x_fit, y_fit = fit_outline
                fig.add_trace(go.Scatter(
                    x=x_fit, y=y_fit, mode='lines', line=dict(color='cyan', width=2, dash='dot'),
                    name=f"Fit {obj_num}", hoverinfo='skip', showlegend=False, visible=True
                ))

        total_traces = len(fig.data)
        buttons = []
        for idx, name in enumerate(image_keys):
            vis_state = [False] * num_img_traces + [True] * (total_traces - num_img_traces)
            vis_state[idx] = True
            buttons.append(dict(
                label=f"Channel: {name}",
                method="update",
                args=[{"visible": vis_state}, {"title": f"Interactive Cell Overlay | Viewing: {name}"}]
            ))

        fig.update_layout(
            title=f"Interactive Cell Mask Overlay | Viewing: {image_keys[0]}",
            updatemenus=[dict(
                active=0, buttons=buttons, direction="down",
                pad={"r": 10, "t": 10}, showactive=True, x=0.01, xanchor="left", y=1.15, yanchor="top"
            )],
            margin=dict(l=0, r=0, b=0, t=60),
            coloraxis_showscale=False,
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False, autorange="reversed")
        )

        html_path = f"{save_dir}/interactive_cell_overlay.html"
        fig.write_html(html_path)

    def run_segmentation(self):
        os.makedirs(self.save_dir, exist_ok=True)
        FileHandler.save_parameters(vars(self.args) if hasattr(self.args, '__dict__') else {}, self.save_dir)

        if self.bead_path:
            target_tmat_path = os.path.join(self.save_dir, "transformation_matrix.npy")
            if not os.path.exists(target_tmat_path):
                registrar = BeadRegistrar()
                if str(self.manual_registration).lower() in ['true', '1', 't', 'y']:
                    registrar.interactive_manual_registration(self.bead_path, self.save_dir, int(self.manual_pairs))
                else:
                    registrar.calculate_transformation(self.bead_path, self.save_dir, self.channel)
            self.tmats_path = target_tmat_path

        tmats = self.get_transformation_matrix()
        if tmats is not None:
            target_tmat_path = os.path.join(self.save_dir, "transformation_matrix.npy")
            if not os.path.exists(target_tmat_path):
                try: np.save(target_tmat_path, tmats)
                except Exception: pass

        L_chan, R_chan, num_frames, raw_shape = VideoProcessor.process_video(
            self.video_path, 0, self.num_frames, self.roi_channel, 
            self.roi_file, self.save_dir, int(getattr(self.args, 'num_channels', 1)), 
            self.channel, self.ALEX, getattr(self.args, 'start_channel', 'L')
        )
        self.raw_shape = raw_shape

        if L_chan is not None:
            L_avg = np.mean(L_chan[:min(5, len(L_chan))], axis=0).astype(np.uint16)
            tifffile.imwrite(f"{self.save_dir}/L_avg.tif", L_avg, imagej=True)
            self.save_image_with_scalebar(L_avg, f"{self.save_dir}/L_avg.png")
            tifffile.imwrite(f"{self.save_dir}/L_channel.tif", L_chan, imagej=True)

        if R_chan is not None:
            R_avg = np.mean(R_chan[:min(5, len(R_chan))], axis=0).astype(np.uint16)
            tifffile.imwrite(f"{self.save_dir}/R_avg.tif", R_avg, imagej=True)
            self.save_image_with_scalebar(R_avg, f"{self.save_dir}/R_avg.png")
            tifffile.imwrite(f"{self.save_dir}/R_channel.tif", R_chan, imagej=True)

        sr = StackReg(StackReg.AFFINE)
        bf_sum, bf_cropped, bf_R_cropped = self.process_brightfield(raw_shape, sr, tmats)

        target_chan = L_chan if self.channel == "L" else R_chan
        fl_avg = np.mean(target_chan[:int(getattr(self.args, 'frame_avg', 5))], axis=0).astype(np.uint16) if target_chan is not None else None

        mask_type = getattr(self.args, 'mask_type', 'AI')
        img_for_masking = bf_cropped if (mask_type == "BF" and bf_cropped is not None) else fl_avg

        if mask_type in ["AI", "BF", "FL_AI"]:
            if self.model_type == "omnitorch":
                from segmentation_models import RunOmnipose
                visual_mask = RunOmnipose.main(img_for_masking, getattr(self.args, 'model', ''), self.save_dir)
                io.imsave(f'{self.save_dir}/visual_mask_omni.tif', visual_mask, check_contrast=False)
                prediction = np.zeros(img_for_masking.shape, dtype=np.uint8)
                flat_pixels = visual_mask.reshape(-1, visual_mask.shape[-1]) if visual_mask.ndim == 3 else visual_mask.reshape(-1, 1)
                unique_colors = [c for c in np.unique(flat_pixels, axis=0) if np.any(c > 0)]
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                for color in unique_colors:
                    mask_boolean = np.all(visual_mask == color, axis=-1) if visual_mask.ndim == 3 else (visual_mask == color[0])
                    object_mask = mask_boolean.astype(np.uint8) * 255
                    color_string = f"{color[0]}-{color[1]}-{color[2]}" if len(color) >= 3 else f"{color[0]}"
                    io.imsave(f'{self.save_dir}/object_mask_{color_string}.tif', object_mask, check_contrast=False)
                    prediction += cv2.erode(object_mask, kernel=kernel, iterations=1)
                prediction[prediction > 255] = 0
                mask = prediction
            elif self.model_type == "cellpose":
                from segmentation_models import RunCellpose
                model_val = getattr(self.args, 'model', '')
                pretrained_model = model_val if (model_val and os.path.exists(str(model_val))) else None
                cp_model_type = model_val if (model_val and not pretrained_model) else "bact_phase_cp3"
                channels = getattr(self.args, 'channels', [0, 0])
                diameter = getattr(self.args, 'diameter', None)
                flow_threshold = getattr(self.args, 'flow_threshold', 0.4)
                cellprob_threshold = getattr(self.args, 'cellprob_threshold', 0.0)
                gpu = getattr(self.args, 'gpu', torch.cuda.is_available() if 'torch' in globals() else False)

                RunCellpose.main(
                    img_obj=img_for_masking,
                    model_type=cp_model_type,
                    pretrained_model=pretrained_model,
                    save_dir=self.save_dir,
                    channels=channels,
                    diameter=diameter,
                    flow_threshold=flow_threshold,
                    cellprob_threshold=cellprob_threshold,
                    gpu=gpu
                )
                raw_mask_path = os.path.join(self.save_dir, "output_raw_mask_cellpose.tif")
                if os.path.exists(raw_mask_path):
                    mask = tifffile.imread(raw_mask_path)
                else:
                    mask = np.zeros_like(img_for_masking, dtype=np.uint8)
            elif self.model_type in ["unet", "keras"] and self.model is not None:
                from segmentation_models import RunUNet
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                num_dil = getattr(self.args, 'dilations', 1)
                dat_candidates = (
                    glob.glob(os.path.join(self.save_dir, "*params.dat")) +
                    glob.glob(os.path.join(os.path.dirname(self.video_path or ''), "*params.dat")) +
                    glob.glob("*params.dat")
                )
                if dat_candidates and os.path.exists(dat_candidates[0]):
                    try:
                        with open(dat_candidates[0], 'r', encoding='utf-8', errors='ignore') as f:
                            for line in f:
                                if 'dilations' in line.lower():
                                    parts = line.replace(':', ' ').replace('=', ' ').split()
                                    for idx, part in enumerate(parts):
                                        if part.lower() == 'dilations' and idx + 1 < len(parts):
                                            num_dil = int(parts[idx + 1])
                                            break
                    except Exception: pass

                thresh_param = getattr(self.args, 'threshold_param', getattr(self.args, 'model_params', 0.5))
                raw_mask = RunUNet.main(img_obj=img_for_masking, model_or_path=self.model, save_dir=self.save_dir, threshold_param=thresh_param, inv_bf=self.inv_bf, debug=False)
                io.imsave(f'{self.save_dir}/visual_mask_unet.tif', raw_mask, check_contrast=False)
                prediction = np.zeros(img_for_masking.shape, dtype=np.uint8)
                flat_pixels = raw_mask.reshape(-1, raw_mask.shape[-1]) if raw_mask.ndim == 3 else raw_mask.reshape(-1, 1)
                unique_colors = [c for c in np.unique(flat_pixels, axis=0) if np.any(c > 0)]
                for color in unique_colors:
                    mask_boolean = np.all(raw_mask == color, axis=-1) if raw_mask.ndim == 3 else (raw_mask == color[0])
                    object_mask = mask_boolean.astype(np.uint8) * 255
                    color_string = f"{color[0]}-{color[1]}-{color[2]}" if len(color) >= 3 else f"{color[0]}"
                    io.imsave(f'{self.save_dir}/object_mask_{color_string}.tif', object_mask, check_contrast=False)
                    processed_object = cv2.dilate(object_mask, kernel, iterations=num_dil) if num_dil > 0 else (cv2.erode(object_mask, kernel, iterations=abs(num_dil)) if num_dil < 0 else object_mask)
                    prediction += processed_object
                prediction[prediction > 255] = 0
                mask = prediction
            elif self.model_type == "pytorch" and self.model is not None:
                patches, h, w = ImageProcessor.make_patches(img_for_masking, self.inv_bf)
                preds = []
                with torch.no_grad():
                    for p in patches:
                        tensor_patch = torch.tensor(p, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                        out = self.model(tensor_patch)
                        preds.append(out.squeeze().cpu().numpy() * 255)
                mask = ImageProcessor.stitch_patches(preds, img_for_masking.shape, h, w)
            else:
                mask = np.zeros_like(img_for_masking, dtype=np.uint8)
        elif mask_type == "THRESHOLD":
            mask = ImageProcessor.apply_watershed_threshold(img_for_masking, dist_multiplier=0.3, use_otsu=self.use_otsu)
        elif mask_type == "WHOLE":
            mask = np.ones_like(img_for_masking, dtype=np.uint8) * 255
        elif mask_type == "MANUAL":
            existing_mask_path = os.path.join(self.save_dir, "all_cells_mask.tif")
            existing_mask = None
            if os.path.exists(existing_mask_path):
                try:
                    existing_mask = tifffile.imread(existing_mask_path)
                    print(f"[AnalysisPipeline] Loaded existing mask for manual editing: '{existing_mask_path}'", flush=True)
                except Exception as e:
                    print(f"[AnalysisPipeline] Could not load existing mask ({e}). Starting fresh.", flush=True)

            raw_manual_mask = ImageProcessor.interactive_edit_mask(mask=existing_mask, bg_img=bf_cropped, fl_img=fl_avg)
            mask = (raw_manual_mask > 0).astype(np.uint8) * 255
        elif mask_type == "file":
            mask_files = glob.glob(f"{self.save_dir}/{self.mask_prefix}*.tif")
            mask = io.imread(mask_files[0]) if mask_files else np.zeros_like(img_for_masking)
            mask = (mask > 0).astype(np.uint8) * 255
        else:
            mask = np.zeros_like(img_for_masking, dtype=np.uint8)

        # Apply global diagonal pixel separation pass to primary mask
        mask = (ImageProcessor.separate_diagonal_pixels(mask) > 0).astype(np.uint8) * 255

        # Transform masks for Left and Right channels dynamically with diagonal pixel separation
        mask_l = mask if self.channel == "L" else ImageProcessor.transform_mask(mask, tmats, src_channel=self.channel, dst_channel="L", ref_channel=self.channel)
        mask_r = mask if self.channel == "R" else ImageProcessor.transform_mask(mask, tmats, src_channel=self.channel, dst_channel="R", ref_channel=self.channel)

        # Save Primary Reference Mask
        tifffile.imwrite(f"{self.save_dir}/all_cells_mask.tif", mask.astype(np.uint16), imagej=True)
        self.save_image_with_scalebar(mask, f"{self.save_dir}/all_cells_mask.png", cmap="viridis")

        # Save Channel-Specific Registered Masks
        tifffile.imwrite(f"{self.save_dir}/all_cells_mask_L.tif", mask_l.astype(np.uint16), imagej=True)
        self.save_image_with_scalebar(mask_l, f"{self.save_dir}/all_cells_mask_L.png", cmap="viridis")

        if int(getattr(self.args, 'num_channels', 1)) == 2 or R_chan is not None:
            tifffile.imwrite(f"{self.save_dir}/all_cells_mask_R.tif", mask_r.astype(np.uint16), imagej=True)
            self.save_image_with_scalebar(mask_r, f"{self.save_dir}/all_cells_mask_R.png", cmap="viridis")

        # Save Integer-Labeled Masks with connectivity=1 (4-connectivity)
        labeled_mask = measure.label(mask > 0, connectivity=1).astype(np.uint16)
        labeled_mask_l = measure.label(mask_l > 0, connectivity=1).astype(np.uint16)
        labeled_mask_r = measure.label(mask_r > 0, connectivity=1).astype(np.uint16)

        tifffile.imwrite(f"{self.save_dir}/all_cells_mask_labeled.tif", labeled_mask, imagej=True)
        tifffile.imwrite(f"{self.save_dir}/all_cells_mask_L_labeled.tif", labeled_mask_l, imagej=True)
        if int(getattr(self.args, 'num_channels', 1)) == 2 or R_chan is not None:
            tifffile.imwrite(f"{self.save_dir}/all_cells_mask_R_labeled.tif", labeled_mask_r, imagej=True)

        # Save Multi-Color RGB Mask Images (Each Cell Has a Unique Color)
        if labeled_mask.max() > 0:
            rng = np.random.RandomState(42)
            max_lbl = max(labeled_mask.max(), labeled_mask_l.max(), labeled_mask_r.max())
            colors = rng.randint(60, 256, size=(max_lbl + 1, 3), dtype=np.uint8)
            colors[0] = [0, 0, 0]

            io.imsave(f"{self.save_dir}/all_cells_mask_colored.png", colors[labeled_mask])
            io.imsave(f"{self.save_dir}/all_cells_mask_L_colored.png", colors[labeled_mask_l])
            if int(getattr(self.args, 'num_channels', 1)) == 2 or R_chan is not None:
                io.imsave(f"{self.save_dir}/all_cells_mask_R_colored.png", colors[labeled_mask_r])

        # Generate Visual Overlays
        if L_chan is not None: 
            ImageProcessor.generate_channel_overlay(np.mean(L_chan[:5], axis=0), mask_l, f"{self.save_dir}/L_mask_overlay.png", color=(0, 255, 255))
        if R_chan is not None: 
            ImageProcessor.generate_channel_overlay(np.mean(R_chan[:5], axis=0), mask_r, f"{self.save_dir}/R_mask_overlay.png", color=(0, 255, 255))
        if bf_cropped is not None: 
            ImageProcessor.generate_channel_overlay(bf_cropped, mask, f"{self.save_dir}/BF_mask_overlay.png", color=(255, 255, 0))
        if bf_R_cropped is not None:
            ImageProcessor.generate_channel_overlay(bf_R_cropped, mask_r, f"{self.save_dir}/BF_R_mask_overlay.png", color=(255, 255, 0))

        labels = measure.label(mask > 0, connectivity=1)
        objects = measure.regionprops(labels)

        for obj_num, obj in enumerate(objects, start=1):
            if obj.area < self.area_filter: continue
            cell_dir = f"{self.save_dir}/cell_{obj_num}"
            os.makedirs(cell_dir, exist_ok=True)
            cell_mask = (labels == obj.label).astype(np.uint8)

            if bf_cropped is not None:
                bg_mask = (cell_mask == 0)
                masked_bf = bf_cropped.copy()
                if np.any(bg_mask): masked_bf[bg_mask] = np.mean(masked_bf[bg_mask])
                tifffile.imwrite(f"{cell_dir}/cellmasked_BF.tif", cv2.normalize(masked_bf, None, 0, 65535, cv2.NORM_MINMAX, dtype=cv2.CV_16U), imagej=True)
                io.imsave(f"{cell_dir}/cellmasked_BF.png", cv2.normalize(masked_bf, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), check_contrast=False)

        print("[AnalysisPipeline] Cell Segmentation completed successfully.", flush=True)

    def run_fitting(self):
        mask_path = f"{self.save_dir}/all_cells_mask.tif"
        if not os.path.exists(mask_path): return

        mask = tifffile.imread(mask_path)
        labels = measure.label(mask > 0, connectivity=1)
        objects = measure.regionprops(labels)

        tmats = self.get_transformation_matrix()

        all_fits_mask = np.zeros_like(mask, dtype=np.uint8)
        all_fit_results = []
        fit_outlines = {}
        all_objects_data = []
        mask_stack_l_list, mask_stack_r_list = [], []
        fits_stack_list = []

        L_file = os.path.join(self.save_dir, "L_channel.tif")
        R_file = os.path.join(self.save_dir, "R_channel.tif")
        L_chan = tifffile.imread(L_file) if os.path.exists(L_file) else None
        R_chan = tifffile.imread(R_file) if os.path.exists(R_file) else None

        mask_r_global = ImageProcessor.transform_mask(mask, tmats, src_channel=self.channel, dst_channel="R", ref_channel=self.channel) if (self.channel == "L" and tmats is not None) else mask
        global_median_bgs = {}
        
        if L_chan is not None:
            L_avg = np.mean(L_chan[:int(getattr(self.args, 'frame_avg', 5))], axis=0)
            bg_L = L_avg[mask == 0]
            global_median_bgs["L"] = float(np.median(bg_L)) if len(bg_L) > 0 else 0.0

        if R_chan is not None:
            R_avg = np.mean(R_chan[:int(getattr(self.args, 'frame_avg', 5))], axis=0)
            bg_R = R_avg[mask_r_global == 0]
            global_median_bgs["R"] = float(np.median(bg_R)) if len(bg_R) > 0 else 0.0

        for obj_num, obj in enumerate(objects, start=1):
            if obj.area < self.area_filter: continue
            cell_dir = f"{self.save_dir}/cell_{obj_num}"
            os.makedirs(cell_dir, exist_ok=True)

            cell_mask_ref = (labels == obj.label).astype(np.uint8)
            single_fit_mask = np.zeros_like(cell_mask_ref, dtype=np.uint8)

            single_mask_l = cell_mask_ref if self.channel == "L" else ImageProcessor.transform_mask(cell_mask_ref, tmats, src_channel=self.channel, dst_channel="L", ref_channel=self.channel)
            single_mask_r = cell_mask_ref if self.channel == "R" else ImageProcessor.transform_mask(cell_mask_ref, tmats, src_channel=self.channel, dst_channel="R", ref_channel=self.channel)

            mask_stack_l_list.append(single_mask_l)
            mask_stack_r_list.append(single_mask_r)

            tifffile.imwrite(f"{cell_dir}/single_cell_mask.tif", (cell_mask_ref * 255).astype(np.uint8))
            tifffile.imwrite(f"{cell_dir}/single_cell_mask_L.tif", (single_mask_l * 255).astype(np.uint8))
            tifffile.imwrite(f"{cell_dir}/single_cell_mask_R.tif", (single_mask_r * 255).astype(np.uint8))

            mat_data = {
                'cell_num': obj_num,
                'area': obj.area,
                'centroid': obj.centroid,
                'bbox': obj.bbox,
                'cell_mask': cell_mask_ref.astype(np.float64),
                'major_axis_length': getattr(obj, 'major_axis_length', 0),
                'minor_axis_length': getattr(obj, 'minor_axis_length', 0),
                'eccentricity': getattr(obj, 'eccentricity', 0),
                'orientation': getattr(obj, 'orientation', 0)
            }

            obj_data = {prop: getattr(obj, prop) for prop in ['label', 'area', 'bbox', 'centroid', 'eccentricity', 'major_axis_length', 'minor_axis_length', 'orientation', 'solidity', 'perimeter'] if hasattr(obj, prop)}
            extracted_intensities = {}

            for chan_name, chan_data in [("L", L_chan), ("R", R_chan)]:
                if chan_data is not None:
                    curr_cell_mask = single_mask_l if chan_name == "L" else single_mask_r
                    intensities = [np.mean(frame[curr_cell_mask > 0]) if np.any(curr_cell_mask > 0) else 0 for frame in chan_data]
                    extracted_intensities[chan_name] = intensities
                    chan_avg = np.mean(chan_data[:int(getattr(self.args, 'frame_avg', 5))], axis=0)
                    mask_pixels = chan_avg[curr_cell_mask > 0]
                    obj_data[f'mean_intensity_{chan_name}'] = float(np.mean(mask_pixels)) if len(mask_pixels) > 0 else 0.0
                    obj_data[f'total_intensity_{chan_name}'] = float(np.sum(mask_pixels)) if len(mask_pixels) > 0 else 0.0
                    obj_data[f'global_median_bg_{chan_name}'] = global_median_bgs.get(chan_name, 0.0)

                    masked_roi_img = chan_avg.astype(np.uint16).copy()
                    bg_mask = (curr_cell_mask == 0)
                    if np.any(bg_mask): masked_roi_img[bg_mask] = np.mean(masked_roi_img[bg_mask])
                    tifffile.imwrite(f"{cell_dir}/cellmasked_{chan_name}.tif", masked_roi_img, imagej=True)
                    io.imsave(f"{cell_dir}/cellmasked_{chan_name}.png", cv2.normalize(masked_roi_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), check_contrast=False)

                    frames_x = np.arange(len(intensities))
                    pd.DataFrame({'Frame': frames_x, 'Intensity': intensities}).to_csv(f"{cell_dir}/raw_intensity_{chan_name}.csv", index=False)
                    fit_quality = IntensityAnalyzer.plot_intensity([frames_x, intensities], chan_name, obj_num, len(intensities), cell_dir)

                    fit_rows = []
                    model_names = ["Single Exp", "Double Exp", "Triple Exp"]
                    for i, fit_res in enumerate(fit_quality):
                        if fit_res != 0:
                            params, rchisq = fit_res
                            row = {'Model': model_names[i], 'Reduced_ChiSq': rchisq, 'Status': 'Success'}
                            if len(params) >= 2: row.update({'A': params[0], 'k': params[1]})
                            if len(params) >= 4: row.update({'A1': params[2], 'k1': params[3]})
                            if len(params) >= 6: row.update({'A2': params[4], 'k2': params[5]})
                            fit_rows.append(row)
                        else:
                            fit_rows.append({'Model': model_names[i], 'Status': 'Fit Failed'})
                    pd.DataFrame(fit_rows).to_csv(f"{cell_dir}/intensity_fit_params_{chan_name}.csv", index=False)

            fit_result, x_fit, y_fit = self.fit_cell(cell_mask_ref, obj_num, self.channel, cell_dir)
            if fit_result:
                all_fit_results.append(fit_result)
                fit_outlines[obj_num] = (x_fit, y_fit)
                mat_data['fit_length_nm'] = fit_result['Length_nm']
                mat_data['fit_width_nm'] = fit_result['Width_nm']

                pts = np.round(np.column_stack((x_fit, y_fit))).astype(np.int32)

                # --- 1-PIXEL GAP BUFFER LOGIC ---
                single_fit = np.zeros_like(mask, dtype=np.uint8)
                cv2.fillPoly(single_fit, [pts], color=255)

                # Create a 1-pixel dilated forbidden zone from previously fitted cells
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                forbidden_zone = cv2.dilate((all_fits_mask > 0).astype(np.uint8), kernel, iterations=2)

                # Assign pixels to all_fits_mask only where there is no overlap/forbidden buffer
                available_region = (single_fit > 0) & (forbidden_zone == 0)
                all_fits_mask[available_region] = 255
                # --------------------------------

                cv2.fillPoly(single_fit_mask, [pts], color=1)

                single_mask_file = f"{cell_dir}/single_cell_mask.tif"
                try: self.line_profile(fitted_mask_file=single_mask_file, save_dir=cell_dir, is_ALEX=self.ALEX)
                except Exception as e: print(f"Line profile error obj {obj_num}: {e}", flush=True)

            fits_stack_list.append(single_fit_mask)
            all_objects_data.append(obj_data)

            if 'L' in extracted_intensities: mat_data['intensity_L'] = extracted_intensities['L']
            if 'R' in extracted_intensities: mat_data['intensity_R'] = extracted_intensities['R']
            sio.savemat(f"{cell_dir}/cell_{obj_num}_data.mat", mat_data)

        if all_fit_results:
            df_fits = pd.DataFrame(all_fit_results)
            df_fits.to_csv(f"{self.save_dir}/cell_fitting_parameters.csv", index=False)

            fits_l = all_fits_mask if self.channel == "L" else ImageProcessor.transform_mask(all_fits_mask, tmats, src_channel=self.channel, dst_channel="L", ref_channel=self.channel)
            fits_r = all_fits_mask if self.channel == "R" else ImageProcessor.transform_mask(all_fits_mask, tmats, src_channel=self.channel, dst_channel="R", ref_channel=self.channel)

            tifffile.imwrite(f"{self.save_dir}/all_fits_mask.tif", all_fits_mask.astype(np.uint16), imagej=True)
            tifffile.imwrite(f"{self.save_dir}/all_fits_mask_L.tif", fits_l.astype(np.uint16), imagej=True)
            tifffile.imwrite(f"{self.save_dir}/all_fits_mask_R.tif", fits_r.astype(np.uint16), imagej=True)
            self.save_image_with_scalebar(all_fits_mask, f"{self.save_dir}/all_fits_mask.png", cmap="viridis")

            L_tif, R_tif, BF_tif, BF_R_tif = os.path.join(self.save_dir, "L_avg.tif"), os.path.join(self.save_dir, "R_avg.tif"), os.path.join(self.save_dir, "BF.tif"), os.path.join(self.save_dir, "BF_R.tif")
            if os.path.exists(L_tif): ImageProcessor.generate_channel_overlay(tifffile.imread(L_tif), fits_l, os.path.join(self.save_dir, "L_fits_overlay.png"), color=(0, 255, 0))
            if os.path.exists(R_tif): ImageProcessor.generate_channel_overlay(tifffile.imread(R_tif), fits_r, os.path.join(self.save_dir, "R_fits_overlay.png"), color=(255, 0, 0))
            if os.path.exists(BF_tif): ImageProcessor.generate_channel_overlay(tifffile.imread(BF_tif), all_fits_mask, os.path.join(self.save_dir, "BF_fits_overlay.png"), color=(0, 255, 255))
            if os.path.exists(BF_R_tif): ImageProcessor.generate_channel_overlay(tifffile.imread(BF_R_tif), fits_r, os.path.join(self.save_dir, "BF_R_fits_overlay.png"), color=(0, 255, 255))

        images_dict = {}
        if os.path.exists(f"{self.save_dir}/BF.tif"): images_dict["Brightfield (BF)"] = tifffile.imread(f"{self.save_dir}/BF.tif")
        if os.path.exists(f"{self.save_dir}/BF_R.tif"): images_dict["Brightfield R (BF_R)"] = tifffile.imread(f"{self.save_dir}/BF_R.tif")
        if os.path.exists(f"{self.save_dir}/L_avg.tif"): images_dict["Channel L"] = tifffile.imread(f"{self.save_dir}/L_avg.tif")
        if os.path.exists(f"{self.save_dir}/R_avg.tif"): images_dict["Channel R"] = tifffile.imread(f"{self.save_dir}/R_avg.tif")
        if images_dict: self.create_interactive_html(images_dict, objects, all_fit_results, fit_outlines, tmats, self.save_dir)

        if all_objects_data and mask_stack_l_list:
            tifffile.imwrite(f"{self.save_dir}/mask_stack.tif", np.array(mask_stack_l_list) * 255, imagej=True)
            tifffile.imwrite(f"{self.save_dir}/mask_stack_L.tif", np.array(mask_stack_l_list) * 255, imagej=True)
            tifffile.imwrite(f"{self.save_dir}/mask_stack_R.tif", np.array(mask_stack_r_list) * 255, imagej=True)
            if fits_stack_list: tifffile.imwrite(f"{self.save_dir}/fits_stack.tif", np.array(fits_stack_list) * 255, imagej=True)

            cleaned_objects_data = []
            for obj_dict in all_objects_data:
                row_data = {}
                for key, value in obj_dict.items():
                    if isinstance(value, (tuple, list, np.ndarray)): row_data[key] = str(value)
                    else: row_data[key] = value
                cleaned_objects_data.append(row_data)

            all_objects_df = pd.DataFrame(cleaned_objects_data)
            all_objects_df.to_csv(f"{self.save_dir}/all_objects.csv", index=False)

            raw_shape = self.raw_shape if self.raw_shape is not None else (1, 512, 1024)
            is_split_view = (int(getattr(self.args, 'num_channels', 1)) == 2) or (raw_shape[2] >= raw_shape[1] * 1.2 and raw_shape[2] >= 500)
            y1, y2, x1, x2 = ImageProcessor.read_roi(self.roi_file, self.roi_channel, self.channel, raw_shape[1], raw_shape[2], is_split_view)
            p = {'laser_centre_X': (x1 + x2) / 2, 'laser_centre_Y': (y1 + y2) / 2, 'laser_radius_R': (x2 - x1) / 2, 'laser_angle': 0}
            bf_sum_val = tifffile.imread(f"{self.save_dir}/BF.tif") if os.path.exists(f"{self.save_dir}/BF.tif") else np.zeros((raw_shape[1], raw_shape[2]//2))
            
            matlab_dict = {
                'area': all_objects_df['area'].values if 'area' in all_objects_df else np.array([]),
                'BF': bf_sum_val,
                'bf2fl_tform': [0, 0],
                'CellObject': np.array(mask_stack_l_list).astype(np.float64),
                'p': p
            }
            if self.channel == "L" and L_chan is not None: matlab_dict['Cellframe0'] = L_chan[0]
            elif self.channel == "R" and R_chan is not None: matlab_dict['Cellframe0'] = R_chan[0]

            video_base = os.path.basename(self.video_path)
            mat_filename = os.path.join(self.save_dir, (os.path.splitext(video_base)[0] if video_base else "video") + "_segmentation.mat")
            try: sio.savemat(mat_filename, matlab_dict)
            except Exception: pass

            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            areas = [o['area'] for o in all_objects_data if 'area' in o]
            eccs = [o['eccentricity'] for o in all_objects_data if 'eccentricity' in o]
            if areas: ax[0].hist(areas); ax[0].set_xlabel("Area")
            if eccs: ax[1].hist(eccs); ax[1].set_xlabel("Eccentricity")
            if areas and eccs: ax[2].scatter(areas, eccs); ax[2].set_xlabel("Area"); ax[2].set_ylabel("Eccentricity")
            plt.savefig(f"{self.save_dir}/obj_feature_summary.png")
            plt.close()

        print("[AnalysisPipeline] Cell Shape Fitting completed successfully.", flush=True)

    def run(self):
        if str(self.overwrite).lower() in ['true', '1', 't', 'y']:
            if "results" in self.save_dir and os.path.exists(self.save_dir):
                try: shutil.rmtree(self.save_dir)
                except Exception: pass
        self.run_segmentation()
        if str(self.cell_fitting).lower() in ['true', '1', 't', 'y'] or self.cell_fitting is True:
            self.run_fitting()

def run_preprocessing(params):
    print("\n" + "="*50, flush=True)
    session_name = getattr(params, 'name', getattr(params, 'save_dir', 'session'))
    print(f"PREPROCESSING SESSION: {session_name}", flush=True)
    print("="*50 + "\n", flush=True)
    pipeline = AnalysisPipeline(params)
    pipeline.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PySTACHIO Image Preprocessing & Segmentation Pipeline")
    
    # Core Files & Directories
    parser.add_argument("--video_path", type=str, required=True, help="Path to raw input TIFF video")
    parser.add_argument("--bead_path", type=str, default="", help="Path/prefix to bead calibration images")
    parser.add_argument("--bf_path", type=str, default="", help="Path to Brightfield image")
    parser.add_argument("--save_dir", "--name", dest="save_dir", type=str, default="results", help="Output directory")
    
    # Channels & Optics
    parser.add_argument("--use_channel", "--channel", dest="use_channel", type=str, default="L", choices=["L", "R"], help="Target channel ('L' or 'R')")
    parser.add_argument("--num_channels", type=int, default=1, choices=[1, 2], help="Number of imaging channels")
    parser.add_argument("--start_channel", type=str, default="L", help="Starting channel for ALEX mode")
    parser.add_argument("--frame_avg", type=int, default=5, help="Number of frames to average")
    parser.add_argument("--num_frames", type=int, default=0, help="Number of frames to process (0 = all)")
    parser.add_argument("--ALEX", type=str, default="False", help="Enable ALEX alternating excitation (True/False)")
    parser.add_argument("--pixel_size", type=float, default=0.120, help="Pixel size in micrometers")
    
    # Segmentation & Masking
    parser.add_argument("--mask_type", type=str, default="AI", choices=["AI", "BF", "FL_AI", "THRESHOLD", "WHOLE", "MANUAL", "file"], help="Mask generation mode")
    parser.add_argument("--model_type", type=str, default="unet", choices=["unet", "keras", "pytorch", "cellpose", "omnitorch"], help="Segmentation model architecture")
    parser.add_argument("--model", type=str, default="", help="Path or URL to model file")
    parser.add_argument("--threshold_param", "--model_params", dest="threshold_param", type=float, default=0.5, help="Segmentation probability threshold")
    parser.add_argument("--area_filter", type=int, default=100, help="Minimum cell area filter in pixels")
    parser.add_argument("--inv_bf", type=str, default="False", help="Invert background intensity (True/False)")
    parser.add_argument("--mask_prefix", type=str, default="mask", help="Prefix for pre-existing mask files")

    # Registration & ROI
    parser.add_argument("--tmats_path", type=str, default="transformation_matrix.npy", help="Path to transformation matrix file")
    parser.add_argument("--roi_channel", type=str, default="L", help="ROI reference channel")
    parser.add_argument("--roi_file", type=str, default="", help="Path to ImageJ ROI file")
    parser.add_argument("--manual_registration", type=str, default="False", help="Enable manual bead registration (True/False)")
    parser.add_argument("--manual_pairs", type=int, default=10, help="Number of point pairs for manual registration")
    parser.add_argument("--use_otsu", type=str, default="False", help="Otsu threshold mode ('False', 'True', 'multi')")

    # Pipeline Flags
    parser.add_argument("--cell_fitting", type=str, default="False", help="Fit capsule shapes to cell masks (True/False)")
    parser.add_argument("--overwrite", type=str, default="True", help="Overwrite output directory if it exists (True/False)")

    parsed_args = parser.parse_args()
    
    parsed_args.channel = parsed_args.use_channel
    parsed_args.name = parsed_args.save_dir
    
    run_preprocessing(parsed_args)