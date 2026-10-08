# -*- coding: utf-8 -*-
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from keras.models import load_model
import tensorflow as tf
#import torch
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
        with open(f'{save_dir}/parameters.txt', 'w',encoding="utf-8") as f:
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
        """
        Computes the Dice loss between the true masks and predicted masks.
        """
        # Flatten the tensors
        y_true_f = K.flatten(tf.cast(y_true, tf.float32))
        y_pred_f = K.flatten(y_pred)
        
        # Calculate intersection
        intersection = K.sum(y_true_f * y_pred_f)
        
        # Calculate the Dice coefficient
        dice_coef = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        
        # Return loss (1 - coefficient)
        return 1.0 - dice_coef

class IntensityAnalyzer:
    """Handles plotting of intensity traces and fitting to exponential decay models."""
    
    @staticmethod
    def plot_intensity(intensity_data, channel, index, frames, save_dir):
        intensity_data_x = np.asarray(intensity_data[0])
        intensity_data_y = np.asarray(intensity_data[1])
        
        # 1. Plot the full raw intensity trace
        fig = plt.figure(figsize=(7, 5))
        ax = fig.subplots()
        ax.plot(intensity_data_x[:frames], intensity_data_y[:frames], label=f'{channel}')
        ax.legend(fontsize=12)
        ax.set_xlabel("Frame Number")
        ax.set_ylabel("Intensity (Counts)")
        plt.savefig(f'{save_dir}/Int_intensity_{channel}_obj_{index}.png')
        plt.close()

        # 2. Plot the first 50 frames and fit exponential decay models
        fig = plt.figure(figsize=(7, 5))
        ax = fig.subplots()
        
        # Safely determine frame ranges
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
    def local_contrast_normalization(img, sigma=15.0, eps=1e-5):
        """Standardizes image contrast across varying lighting."""
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
        # Calculate single channel width based on physical split, not user parameters
        half_width = img_width // 2 if is_split_view else img_width
        
        if not os.path.exists(roi_path):
            return [0, img_height, 0, half_width]
            
        try:
            roi = ImagejRoi.fromfile(roi_path)
            y1, x1, y2, x2 = int(roi.top), int(roi.left), int(roi.bottom), int(roi.right)
            
            # Convert any global coordinates to local single-channel coordinates (0 to 1024)
            if is_split_view:
                if x1 >= half_width: x1 -= half_width
                if x2 > half_width:  x2 -= half_width
                
            # Clamp the ROI so it never exceeds the boundaries of a single channel
            y1 = max(0, min(y1, img_height))
            y2 = max(0, min(y2, img_height))
            x1 = max(0, min(x1, half_width))
            x2 = max(0, min(x2, half_width))
            
            return [y1, y2, x1, x2]
            
        except Exception as e:
            print(f"Error reading ROI file: {e}", flush=True)
            return [0, img_height, 0, half_width]

    @staticmethod
    def interactive_edit_mask(mask=None, bg_img=None, fl_img=None):
        """
        Launches full interactive editor to cut, add, delete, and modify masks.
        Supports both Brightfield (bg_img) and Fluorescence (fl_img) canvases.
        """
        if mask is None:
            # Fall back to fl_img if bg_img is None, or vice versa
            ref_img = bg_img if bg_img is not None else fl_img
            if ref_img is None:
                raise ValueError("Must provide either a mask, bg_img, or fl_img to determine canvas dimensions.")
            mask = np.zeros(ref_img.shape[:2], dtype=np.int32)
            
        editor = MaskEditor(mask=mask, bf_img=bg_img, fl_img=fl_img)
        return editor.mask
    
    @staticmethod
    def fit_and_generate_cell_shape(points, shape, radius_hint=None):
        """
        Fits a 2-pole capsule (spherocylinder) shape to the image ROI 
        defined by user points and returns a binary mask of shape canvas.
        """
        mask = np.zeros(shape, dtype=np.uint8)
        
        if len(points) < 2:
            return mask
            
        # Extract endpoints from user clicks/selection
        p1, p2 = np.array(points[0]), np.array(points[1])
        
        # Calculate cell axis, orientation, and default dimensions
        length = np.linalg.norm(p2 - p1)
        angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
        
        # Use radius_hint if provided, otherwise fall back to length estimate
        radius = radius_hint if radius_hint is not None else max(3.0, length / 4.0)
        
        # Generate coordinate grid
        y, x = np.ogrid[:shape[0], :shape[1]]
        
        # Vector along cell axis
        dx = x - p1[0]
        dy = y - p1[1]
        
        ux = np.cos(angle)
        uy = np.sin(angle)
        
        # Projection along axis and perpendicular distance
        proj = dx * ux + dy * uy
        proj_clamped = np.clip(proj, 0, length)
        
        closest_x = p1[0] + proj_clamped * ux
        closest_y = p1[1] + proj_clamped * uy
        
        dist_sq = (x - closest_x)**2 + (y - closest_y)**2
        
        # Binary capsule mask
        mask[dist_sq <= radius**2] = 1
        return mask

    @staticmethod
    def make_patches(input_image, inv_img):
        # 1. Pad the image with the mean background (like the old script), NOT zeros
        img = input_image.copy().astype(np.float64)
        bkg_1 = np.mean(img)
        
        # Calculate padding needed to reach multiples of 256
        h, w = img.shape
        pad_h = (256 - (h % 256)) % 256
        pad_w = (256 - (w % 256)) % 256
        
        # Apply the background padding
        img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=bkg_1)

        patches = []
        num_h_patches = img.shape[0] // 256
        num_w_patches = img.shape[1] // 256
        
        for i in range(num_h_patches):
            for j in range(num_w_patches):
                y, x = i * 256, j * 256
                patch = img[y:y+256, x:x+256]
                
                # Apply inversion if requested
                if str(inv_img).lower() == "true":
                    patch = np.amax(patch) - patch
                    bkg_2 = np.mean(patch)
                    patch = patch - bkg_2 + bkg_1
                
                # 2. CRITICAL FIX: Normalize the individual patch by its max value
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
        return stitched_image[:original_shape[0], :original_shape[1]].astype(np.uint8)

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
            # Multi-Otsu with 3 classes: Background, Halos/Intermediate, Bright Cells
            thresholds = threshold_multiotsu(blurred, classes=3)
            # thresholds[0] separates background from foreground features
            thresh = (blurred > thresholds[0]).astype(np.uint8) * 255
            
        elif otsu_str in ["true", "1", "t", "y"]:
            # Standard single-threshold Otsu
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
        else:
            # Default Adaptive Gaussian Thresholding
            thresh = cv2.adaptiveThreshold(
                blurred, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 
                45, 
                -2  
            )
        #_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        sure_bg = cv2.dilate(opening, kernel, iterations=2)
        
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, dist_multiplier * dist_transform.max(), 255, 0)
        
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        img_color = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(img_color, markers)
        
        final_mask = np.zeros_like(img_uint8)
        final_mask[markers > 1] = 255
        
        return ndimage.binary_fill_holes(final_mask).astype(np.uint8) * 255

    @staticmethod
    def generate_dual_masks(mask, target_channel):
        """
        Generates an inverted background mask for the non-target channel 
        when processing dual-channel videos.
        """
        # 1. Generate the inverted mask with blur and morphology
        blur = cv2.GaussianBlur(mask, (25, 25), 0)
        ret, inv_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        morph_kernel = np.ones((5, 5), np.uint8)
        inv_mask = cv2.dilate(inv_mask, morph_kernel, iterations=2)
        inv_mask = cv2.erode(inv_mask, morph_kernel, iterations=1)

        # 2. Assign the standard mask and inv_mask to the correct channels
        if target_channel == 'L':
            return mask, inv_mask
        elif target_channel == 'R':
            return inv_mask, mask
        else:
            return mask, mask

class VideoProcessor:
    @staticmethod
    def process_video(video_path, start_frame, end_frame, roi_channel, roi_file, save_dir, num_channels, channel, ALEX):
        frames = io.imread(video_path)
        
        # Handle 2D single-frame TIFFs
        if frames.ndim == 2:
            frames = np.expand_dims(frames, axis=0)
            
        # Slice frames (end_frame can be None for 'all frames')
        frames = frames[start_frame:end_frame]
            
        raw_shape = np.shape(frames)
        height, width = raw_shape[1], raw_shape[2]
        half_width = width // 2
        
        # Always split view optics
        y1, y2, x1, x2 = ImageProcessor.read_roi(
            roi_file, roi_channel, channel, height, width, is_split_view=True
        )
        
        # Extract split halves based on ALEX mode
        if str(ALEX).lower() == "false":
            L_full = frames[:, :, :half_width]
            R_full = frames[:, :, half_width:]
        else:
            L_full = frames[::2, :, :half_width]
            R_full = frames[1::2, :, half_width:]
            
        L_channel = L_full[:, y1:y2, x1:x2]
        R_channel = R_full[:, y1:y2, x1:x2]
        
        # If 1-channel mode, extract only the active side (L or R) and leave the blank side None
        if int(num_channels) == 1:
            if channel == "L": 
                return L_channel, None, len(L_channel), raw_shape
            else: 
                return None, R_channel, len(R_channel), raw_shape
        else:
            # 2-channel mode: return both active channels
            return L_channel, R_channel, len(L_channel), raw_shape

class BeadRegistrar:
    """Handles bead-based spatial registration using the best NCC score."""
    
    @staticmethod
    def calculate_ncc(image_a, image_b):
        image_a = image_a.astype(np.float64)
        image_b = image_b.astype(np.float64)
        mean_a = np.mean(image_a)
        mean_b = np.mean(image_b)
        centered_a = image_a - mean_a
        centered_b = image_b - mean_b
        numerator = np.sum(centered_a * centered_b)
        denominator = np.sqrt(np.sum(centered_a ** 2) * np.sum(centered_b ** 2))
        return 0 if denominator == 0 else numerator / denominator

    @staticmethod
    def normalize_channels(i1, i2):
        i1 = i1.astype(np.float64)
        i2 = i2.astype(np.float64)
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
        axs[0].imshow((static_img + moving_img) / 2, cmap='gray')
        axs[1].imshow(registered_img, cmap='gray')
        axs[2].imshow((static_img + registered_img) / 2, cmap='gray')
        safe_name = os.path.basename(os.path.normpath(name))
        plt.tight_layout()
        plt.savefig(f"{save_dir}/Registered_overlay_{safe_name}.png")
        plt.close()

    def calculate_transformation(self, bead_prefix: str, save_dir: str, channel: str):
        bead_folders = glob.glob(f"{bead_prefix}_*")
        if not bead_folders: return None

        ncc_scores = []
        sr = StackReg(StackReg.RIGID_BODY)
        summed_img = None
        
        # Track NCCs for the line plot
        ncc_before_list = []
        ncc_after_list = []
        
        import pandas as pd

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
            
            # Store exact copies for individual plotting
            if channel.upper() == "L":
                i1, i2 = L.copy(), R.copy()
            else:
                i1, i2 = R.copy(), L.copy()

            L_norm, R_norm = self.normalize_channels(L, R)

            if channel.upper() == "L":
                tmat = sr.register(L_norm, R_norm)
            else:
                tmat = sr.register(R_norm, L_norm)
                
            # --- INDIVIDUAL BEAD OVERLAY LOGIC ---
            m1, m2 = np.amax(i1), np.amin(i1)
            if m1 != m2: i1 = 1024 * (i1 - m2) / (m1 - m2)
            m1, m2 = np.amax(i2), np.amin(i2)
            if m1 != m2: i2 = 1024 * (i2 - m2) / (m1 - m2)

            m, n = np.amax(i2), np.amax(i1)
            if m != 0: i2 = n * i2 / m
            if np.amax(i1) != 0: i1 = 255 * i1 / np.amax(i1)
            if np.amax(i2) != 0: i2 = 255 * i2 / np.amax(i2)
            
            out = sr.transform(i2, tmat)
            
            # Calculate NCCs
            ncc_before = self.calculate_ncc(i1, i2)
            ncc_after = self.calculate_ncc(i1, out)
            ncc_before_list.append(ncc_before)
            ncc_after_list.append(ncc_after)
            
            # Fix background padding for plot
            bkg_1 = np.mean(i2)
            out[out == 0] = bkg_1
            
            # 1. Save Individual Overlay (Registered_overlay_beads_1.png, etc)
            fig, axs = plt.subplots(3, 1, figsize=(10, 15))
            axs[0].imshow((i1 + i2) / 2, 'gray')
            axs[0].set_title(f'Original Images - beads_{idx+1}')
            axs[0].set_xticks([]); axs[0].set_yticks([])

            axs[1].imshow(out, 'gray')
            axs[1].set_title('Registered Image')
            axs[1].set_xticks([]); axs[1].set_yticks([])

            axs[2].imshow(((i1 + out) / 2), cmap='gray')
            axs[2].set_title(f'Overlay - NCC = {ncc_after}')
            axs[2].set_xticks([]); axs[2].set_yticks([])
            plt.tight_layout()
            plt.savefig(f"{save_dir}/Registered_overlay_beads_{idx+1}.png")
            plt.close()
            # -------------------------------------
            
            ncc_scores.append({'folder': folder, 'ncc': ncc_after, 'matrix': tmat})

        best_entry = max(ncc_scores, key=lambda x: x['ncc'])
        tmats = best_entry['matrix']
        np.save(f"{save_dir}/transformation_matrix.npy", tmats)

        # 2. Save Summed Beads TIF 
        tifffile.imwrite(f"{save_dir}/summed_beads.tif", summed_img.astype(np.float32), imagej=True)
        
        # Split summed image
        split_width = summed_img.shape[1]
        L_sum = summed_img[:, :split_width//2]
        R_sum = summed_img[:, split_width//2:]
        
        if channel.upper() == "L":
            i1, i2 = L_sum, R_sum
        else:
            i1, i2 = R_sum, L_sum
            
        m1, m2 = np.amax(i1), np.amin(i1)
        if m1 != m2: i1 = 1024 * (i1 - m2) / (m1 - m2)
        m1, m2 = np.amax(i2), np.amin(i2)
        if m1 != m2: i2 = 1024 * (i2 - m2) / (m1 - m2)

        m, n = np.amax(i2), np.amax(i1)
        if m != 0: i2 = n * i2 / m
        if np.amax(i1) != 0: i1 = 255 * i1 / np.amax(i1)
        if np.amax(i2) != 0: i2 = 255 * i2 / np.amax(i2)
        
        out = sr.transform(i2, tmats)
        ncc_score = best_entry['ncc']
        
        bkg_1 = np.mean(i2)
        out[out == 0] = bkg_1
        
        # 3. Save Summed Overlay (Registered_overlay_summed_beads.png)
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        axs[0].imshow((i1 + i2) / 2, 'gray')
        axs[0].set_title('Original Images - summed_beads')
        axs[0].set_xticks([]); axs[0].set_yticks([])

        axs[1].imshow(out, 'gray')
        axs[1].set_title('Registered Image')
        axs[1].set_xticks([]); axs[1].set_yticks([])

        axs[2].imshow(((i1 + out) / 2), cmap='gray')
        axs[2].set_title(f'Overlay - NCC = {ncc_score}')
        axs[2].set_xticks([]); axs[2].set_yticks([])
        plt.tight_layout()
        plt.savefig(f"{save_dir}/Registered_overlay_summed_beads.png")
        plt.close()
        
        # 4. Save Registration CSV
        df_reg = pd.DataFrame.from_dict({'sum': {'ncc_score': ncc_score, 'transformation_matrix': tmats.tolist()}}, orient='index')
        df_reg.to_csv(f"{save_dir}/registration.csv")

        # 5. Save ncc_scores.csv and ncc_scores.png
        df_ncc = pd.DataFrame({
            'Bead_Folder': range(1, len(ncc_before_list) + 1),
            'NCC_Before': ncc_before_list,
            'NCC_After': ncc_after_list
        })
        df_ncc.to_csv(f"{save_dir}/ncc_scores.csv", index=False)

        plt.figure(figsize=(8, 5))
        plt.plot(df_ncc['Bead_Folder'], df_ncc['NCC_Before'], label='Before Registration', color='red', marker='o')
        plt.plot(df_ncc['Bead_Folder'], df_ncc['NCC_After'], label='After Registration', color='blue', marker='o')
        plt.title('Normalized Cross-Correlation (NCC) Scores')
        plt.xlabel('Bead Folder / Frame')
        plt.ylabel('NCC Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_dir}/ncc_scores.png", dpi=300, bbox_inches='tight')
        plt.close()

        return tmats
    
    
    def interactive_manual_registration(self, image_path: str, save_dir: str, num_pairs: int = 10):
        """
        Interactive point-picking for manual affine registration.
        """
        print(f"Loading image for manual registration: {image_path}", flush=True)
        # Load and handle potential 3D stacks
        img = io.imread(image_path)
        if img.ndim == 3: 
            img = np.mean(img, axis=0) 
            
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Normalize for display so 16-bit images don't appear pure black
        img_display = img.astype(float)
        img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
        ax.imshow(img_display, cmap='gray')
        
        ax.set_title(f"Click {num_pairs} pairs of beads.\n"
                     f"Pattern: [Ch1 Bead, Ch2 Bead], [Ch1 Bead, Ch2 Bead]...\n"
                     f"Middle-click to undo, Right-click to stop early.")
        plt.axis('image')
        
        print(f"Waiting for user to select {num_pairs * 2} points on the image...", flush=True)
        pts = plt.ginput(num_pairs * 2, timeout=-1) 
        plt.close(fig)
        
        if len(pts) % 2 != 0:
            print("Warning: Odd number of points selected. Discarding the last point.", flush=True)
            pts = pts[:-1]
            
        if len(pts) < 6:
            raise ValueError("Error: An affine transform requires at least 3 pairs (6 points) to compute.")
            
        pts = np.array(pts)
        pts_ch1 = pts[0::2] # Reference
        pts_ch2 = pts[1::2] # Moving
        
        # Estimate Affine Transform
        tform = ski_transform.estimate_transform('affine', src=pts_ch2, dst=pts_ch1)
        
        print("\nEstimated Affine Transformation Matrix:", flush=True)
        print(tform.params, flush=True)
        
        # 1. Save as .npy to match the existing Python pipeline's format
        out_npy = os.path.join(save_dir, "transformation_matrix.npy")
        np.save(out_npy, tform.params)
        
        # 2. Save as .mat for MATLAB ADEMScode compatibility
        out_mat = os.path.join(save_dir, "channel_transform.mat")
        mat_dict = {
            'T_matrix': tform.params.T, # Transpose for MATLAB
            'tform_type': 'affine',
            'generated_by': 'Python scikit-image manual registration'
        }
        sio.savemat(out_mat, mat_dict)
        
        print(f"\nSaved transformation matrix to {out_npy} and {out_mat}", flush=True)
        return tform.params
    
class MaskEditor:
    """
    Interactive Mask Editor featuring:
    - Left-Drag: Draw freehand lasso contour around cell to automatically fill as mask
    - Right-Drag: Click-and-drag red box to delete all cell masks touched by box
    - Click points + 'c': Cut/split a single mask along drawn path
    - Click 2 points + 'a': Add new 'sausage' mask (width controlled by [ and ])
    - Click 3+ points + 'p': Draw free-form polygon contour around a shape without mathematical fitting
    - Click 2 points (poles) + 'f': Fit exact mathematical capsule (width controlled by [ and ])
    - Click points + 'r': Reject/delete selected cell masks
    - 'z': Undo last change | 'x': Clear current path points | 'd': Done/Save
    """
    def __init__(self, mask, bf_img=None, fl_img=None, rect_width=14, alpha=0.4, outline_alpha=0.7):
        conflicting_keys = ['f', 'c', 'r', 'z', 'a', 'p', 'x', 'd', '+', '-', '=']
        for param in plt.rcParams:
            if param.startswith('keymap.'):
                for key in conflicting_keys:
                    if key in plt.rcParams[param]:
                        plt.rcParams[param].remove(key)
        
        if mask.ndim == 3:
            collapsed = np.zeros(mask.shape[1:], dtype=np.int32)
            for idx in range(mask.shape[0]):
                collapsed[mask[idx] > 0] = idx + 1
            self.mask = collapsed
        else:
            self.mask = mask.copy().astype(np.int32)

        self.image_dict = {}
        if bf_img is not None: self.image_dict["Brightfield"] = bf_img
        if fl_img is not None: self.image_dict["Fluorescence"] = fl_img
        self.current_img_key = list(self.image_dict.keys())[0] if self.image_dict else "None"

        self.history = []
        self.current_points = []
        self.rect_width = rect_width
        self.alpha = alpha
        self.outline_alpha = outline_alpha

        self.press_pos = None
        self.last_pos = None
        self.press_button = None
        self.is_dragging = False
        self.rect_patch = None

        self.fig, (self.ax_ref, self.ax) = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
        
        if len(self.image_dict) > 1:
            self.ax_menu = plt.axes([0.84, 0.88, 0.14, 0.09], facecolor='lightgray')
            self.img_selector = RadioButtons(self.ax_menu, list(self.image_dict.keys()), active=0)
            self.img_selector.on_clicked(self.on_switch_image)

        try:
            if hasattr(self.fig.canvas, "manager") and self.fig.canvas.manager is not None:
                self.fig.canvas.manager.set_window_title("Interactive Mask Editor")
        except Exception:
            pass

        self._redraw()

        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        plt.show()

    def on_switch_image(self, label):
        self.current_img_key = label
        self._redraw()

    def _draw_base(self):
        self.ax_ref.clear()
        self.ax.clear()

        current_img = self.image_dict.get(self.current_img_key, None)

        if current_img is not None:
            cmap = "gray" if current_img.ndim == 2 else None
            self.ax_ref.imshow(current_img, cmap=cmap)
            self.ax.imshow(current_img, cmap=cmap)
        self.ax_ref.set_title(f"Reference Channel ({self.current_img_key})", fontsize=9)

        max_label = int(self.mask.max()) if self.mask.size else 0
        rng = np.random.RandomState(42)
        colors = rng.rand(max(max_label, 1) + 1, 3)
        colors[0] = [0, 0, 0] 
        cmap = mcolors.ListedColormap(colors)
        masked = np.ma.masked_where(self.mask == 0, self.mask)

        current_alpha = self.alpha if current_img is not None else 1.0
        self.ax.imshow(masked, cmap=cmap, interpolation="nearest", vmin=0, vmax=max(max_label, 1), alpha=current_alpha)

        if max_label > 0:
            self.ax.contour(self.mask > 0, levels=[0.5], colors=["white"], linewidths=0.5, alpha=self.outline_alpha)

        self.ax_ref.set_xticks([]); self.ax_ref.set_yticks([])
        self.ax.set_xticks([]); self.ax.set_yticks([])

    def _set_title(self):
        self.fig.suptitle(
            f"Left-Drag: Freehand Lasso | Right-Drag: Delete Box | C: Cut | A: Sausage/Add | P: Polygon | F: Fit Capsule | "
            f"Width: {self.rect_width}px (+/-) | R: Reject | Z: Undo | D: Done", 
            fontsize=9
        )
        self.fig.canvas.draw_idle()

    def on_press(self, event):
        if event.inaxes != self.ax or event.button not in [1, 3]: return
        if event.xdata is None or event.ydata is None: return
        
        self.press_pos = (event.xdata, event.ydata)
        self.last_pos = self.press_pos
        self.press_button = event.button
        self.is_dragging = False

    def on_motion(self, event):
        if self.press_pos is None or event.inaxes != self.ax: return
        if event.xdata is None or event.ydata is None: return

        x0, y0 = self.press_pos
        x1, y1 = event.xdata, event.ydata
        self.last_pos = (x1, y1)

        if abs(x1 - x0) > 3 or abs(y1 - y0) > 3:
            if not self.is_dragging:
                self.is_dragging = True
                if self.press_button == 1:
                    self.current_points = [self.press_pos, (x1, y1)]
                    self._refresh_line()
            else:
                if self.press_button == 1:
                    self.current_points.append((x1, y1))
                    self._refresh_line()
            
            if self.press_button == 3:
                xmin, xmax = min(x0, x1), max(x0, x1)
                ymin, ymax = min(y0, y1), max(y0, y1)

                if self.rect_patch is not None:
                    self.rect_patch.remove()

                self.rect_patch = patches.Rectangle(
                    (xmin, ymin), xmax - xmin, ymax - ymin,
                    linewidth=1.5, edgecolor="red", facecolor="red", alpha=0.3, linestyle="--"
                )
                self.ax.add_patch(self.rect_patch)
                self.fig.canvas.draw_idle()

    def on_release(self, event):
        if self.press_pos is None: return

        if self.rect_patch is not None:
            self.rect_patch.remove()
            self.rect_patch = None

        release_x = event.xdata if event.xdata is not None else self.last_pos[0]
        release_y = event.ydata if event.ydata is not None else self.last_pos[1]

        if self.is_dragging:
            if self.press_button == 1:
                self.add_polygon_contour()
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
                    for label_id in labels_to_delete:
                        self.mask[self.mask == label_id] = 0
                    print(f"Box drag deleted {len(labels_to_delete)} mask(s): {sorted(list(labels_to_delete))}")
                    self._redraw()
        elif not self.is_dragging and self.press_button == 1:
            self.current_points.append(self.press_pos)
            self._refresh_line()

        self.press_pos = None
        self.press_button = None
        self.is_dragging = False

    def _redraw(self):
        self._draw_base()
        self.line_artist, = self.ax.plot([], [], "w-", linewidth=1.5)
        self.point_artist, = self.ax.plot([], [], "wo", markersize=4)
        self._refresh_line()
        self._set_title()

    def _refresh_line(self):
        if self.current_points:
            xs, ys = zip(*self.current_points)
        else:
            xs, ys = [], []
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
        elif event.key == "x":
            self.current_points = []
            self._refresh_line()
        elif event.key in ["-", "_"]:
            self.rect_width = max(2, self.rect_width - 2)
            self._set_title()
        elif event.key in ["+", "="]:
            self.rect_width += 2
            self._set_title()
        elif event.key == "d":
            plt.close(self.fig)

    def _save_history(self):
        self.history.append(self.mask.copy())
        if len(self.history) > 15:
            self.history.pop(0)

    def commit_cut(self):
        if len(self.current_points) < 2:
            print("Click at least 2 points across the mask, then press 'c'.")
            return

        pts = np.array(self.current_points)
        line_canvas = np.zeros(self.mask.shape, dtype=np.uint8)
        for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
            cv2.line(line_canvas, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), color=1, thickness=2)

        intersecting_labels = set(np.unique(self.mask[line_canvas == 1])) - {0}

        if not intersecting_labels:
            print("Cut line did not intersect any cell masks. Draw the line across the cell you want to cut.")
            return

        self._save_history()
        cuts_performed = 0
        max_label = int(self.mask.max())

        for label_id in intersecting_labels:
            region = (self.mask == label_id)
            cut_region = region & (line_canvas == 0)
            pieces, n_pieces = ndi.label(cut_region, structure=np.ones((3, 3)))

            if n_pieces > 1:
                self.mask[region] = 0
                for piece_id in range(1, n_pieces + 1):
                    max_label += 1
                    self.mask[pieces == piece_id] = max_label
                cuts_performed += 1
                print(f"Mask {label_id} successfully split into {n_pieces} segments.")
            else:
                print(f"Line touched mask {label_id} but did not completely divide it into 2+ pieces.")

        if cuts_performed > 0:
            self.current_points = []
            self._redraw()
        else:
            if self.history:
                self.history.pop()
            
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
        else:
            print("Click 2 points for sausage or 3+ points for polygon, then press 'a'.")
            return

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        forbidden_zone = cv2.dilate((self.mask > 0).astype(np.uint8), kernel, iterations=1)
        new_region = new_shape & (forbidden_zone == 0)
        
        if not np.any(new_region): return

        self._save_history()
        new_label = int(self.mask.max()) + 1
        self.mask[new_region] = new_label

        self.current_points = []
        self._redraw()
        print(f"Added new mask with label {new_label}.")

    def add_polygon_contour(self):
        if len(self.current_points) < 3:
            print("Click at least 3 perimeter points around the shape contour, then press 'p'.")
            return

        pts = np.round(np.array(self.current_points)).astype(np.int32)
        canvas = np.zeros(self.mask.shape, dtype=np.uint8)
        
        cv2.fillPoly(canvas, [pts], color=1)
        
        kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel_smooth)
        
        new_shape = canvas.astype(bool)

        kernel_gap = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        forbidden_zone = cv2.dilate((self.mask > 0).astype(np.uint8), kernel_gap, iterations=1)
        new_region = new_shape & (forbidden_zone == 0)
        
        if not np.any(new_region):
            print("Drawn contour overlapped completely with existing masks or background.")
            return

        self._save_history()
        new_label = int(self.mask.max()) + 1
        self.mask[new_region] = new_label

        self.current_points = []
        self._redraw()
        print(f"Added free-form polygon contour mask with label {new_label} (no capsule fitting applied).")

    def fit_and_add_cell_shape(self):
        if len(self.current_points) < 2:
            print("Click 2 points (one at each pole) or 3+ perimeter points, then press 'f'.")
            return
            
        new_shape = ImageProcessor.fit_and_generate_cell_shape(
            None, self.current_points, self.mask.shape
        )
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        forbidden_zone = cv2.dilate((self.mask > 0).astype(np.uint8), kernel, iterations=1)
        new_region = new_shape & (forbidden_zone == 0)
        
        if not np.any(new_region):
            print("Fitted cell shape overlapped completely with existing masks or background.")
            return
            
        self._save_history()
        new_label = int(self.mask.max()) + 1
        self.mask[new_region] = new_label
        
        self.current_points = []
        self._redraw()
        print(f"Added fitted mathematical cell shape with label {new_label}.")

    def reject_selected_masks(self):
        if not self.current_points: return
        h, w = self.mask.shape
        labels_to_reject = set()

        for pt in self.current_points:
            r, c = int(round(pt[1])), int(round(pt[0]))
            if 0 <= r < h and 0 <= c < w:
                lbl = self.mask[r, c]
                if lbl != 0: labels_to_reject.add(lbl)

        if not labels_to_reject: return

        self._save_history()
        for lbl in labels_to_reject:
            self.mask[self.mask == lbl] = 0

        self.current_points = []
        self._redraw()
        print(f"Rejected {len(labels_to_reject)} mask(s).")

    def undo(self):
        if not self.history:
            print("Nothing to undo.")
            return
        self.mask = self.history.pop()
        self.current_points = []
        self._redraw()
        print("Undid last action.")

class AnalysisPipeline:
    def __init__(self, params):
        """
        Integrated PySTACHIO Initializer.
        This replaces the old argparse-based __init__.
        """
        #self.params = params
        self.args = params
        
        # --- 1. Map Universal PySTACHIO Params to Internal Names ---
        self.pxsize = params.pixel_size        # mapped from pixel_size
        self.ALEX = params.ALEX                # mapped from ALEX
        self.save_dir = params.save_dir if params.save_dir else params.name            # Use the seed 'name' as the directory
        self.channel = params.use_channel      # mapped from use_channel
        self.num_frames = None if params.num_frames == 0 else params.num_frames

        # --- 2. Map Preprocessing-Specific Params ---
        self.video_path = params.video_path
        self.bead_path = params.bead_path
        self.bf_path = params.bf_path
        self.tmats_path = params.tmats_path
        self.manual_registration = params.manual_registration
        self.manual_pairs = params.manual_pairs
        self.roi_channel = params.roi_channel
        self.roi_file = params.roi_file
        self.mask_prefix = params.mask_prefix
        self.overwrite = params.overwrite
        self.use_otsu = params.use_otsu
        self.cell_fitting = params.cell_fitting
        self.area_filter = int(params.area_filter)
        self.inv_bf = params.inv_bf

        # --- 3. Model Loading Logic (Supports local file paths and HTTP/HTTPS URLs) ---
        self.model = None
        self.model_params = getattr(params, 'model_params', None)
        
        if params.mask_type in ["AI", "BF", "FL_AI"]:
            # Check if model parameter is a URL and download locally if necessary
            if isinstance(params.model, str) and params.model.startswith(("http://", "https://")):
                parsed_url = urlparse(params.model)
                filename = os.path.basename(parsed_url.path) or "downloaded_model"
                os.makedirs(self.save_dir, exist_ok=True)
                local_model_path = os.path.join(self.save_dir, filename)
                
                if not os.path.exists(local_model_path):
                    print(f"Downloading model from URL: {params.model} -> {local_model_path}", flush=True)
                    req = urllib.request.Request(params.model, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req) as response, open(local_model_path, 'wb') as out_file:
                        shutil.copyfileobj(response, out_file)
                    print("Download complete.", flush=True)
                else:
                    print(f"Using cached model at: {local_model_path}", flush=True)
                
                params.model = local_model_path

            print(f"Loading Model: {params.model}", flush=True)
            
            if params.model_type in ["omnitorch", "cellpose"]:
                print(f"{params.model_type.capitalize()} execution deferred to external segmentation script.", flush=True)                
            
            elif params.model_type in ["unet", "keras"]:
                print("Loading Keras/U-Net model...", flush=True)
                try:
                    self.model = load_model(params.model, compile=False, safe_mode=False)
                except:
                    import tf_keras as tfk
                    self.model = tfk.models.load_model(params.model)
                
            elif params.model_type == "pytorch":
                print("Loading standard PyTorch model...", flush=True)
                self.model = torch.load(params.model, map_location=torch.device('cpu'))
                self.model.eval() 

        # --- 4. Setup Directory ---
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
            
    def process_brightfield(self, raw_shape, sr, tmats):
        bf_sum = None
        bf_cropped = None
        width = raw_shape[2]
        
        # ALWAYS process BF if a path is given, regardless of mask_type
        if self.args.bf_path and os.path.exists(self.args.bf_path):
            bf_img = img_as_float64(io.imread(self.args.bf_path))
            bf_sum = np.sum(bf_img, axis=0) if bf_img.ndim == 3 else bf_img
            
            is_split_view = width > 1500
            bf_width_split = width // 2 if is_split_view else width
            
            if is_split_view:
                L_bf_full = bf_sum[:, :bf_width_split]
                R_bf_full = bf_sum[:, bf_width_split:]
                
                # Create the registered overlay
                if self.args.channel == "L":
                    # L is the static reference, R is the moving channel that gets transformed
                    R_reg = sr.transform(R_bf_full, tmats) if tmats is not None else R_bf_full
                    
                    # Replace StackReg's 0-padding with the mean background to prevent black lines
                    if tmats is not None:
                        R_reg[R_reg == 0] = np.mean(R_bf_full)
                        
                    bf_overlay_full = (L_bf_full + R_reg) / 2
                else:
                    # R is the static reference, L is the moving channel that gets transformed
                    L_reg = sr.transform(L_bf_full, tmats) if tmats is not None else L_bf_full
                    
                    # Replace StackReg's 0-padding with the mean background to prevent black lines
                    if tmats is not None:
                        L_reg[L_reg == 0] = np.mean(L_bf_full)
                        
                    bf_overlay_full = (R_bf_full + L_reg) / 2
                    
                # --- Save Brightfield Registration Overlay Plot ---
                if tmats is not None:
                    L_viz = cv2.normalize(L_bf_full, None, 0, 1, cv2.NORM_MINMAX)
                    R_viz = cv2.normalize(R_bf_full, None, 0, 1, cv2.NORM_MINMAX)
                    overlay_viz = cv2.normalize(bf_overlay_full, None, 0, 1, cv2.NORM_MINMAX)
                    
                    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
                    
                    # Panel 1: Original Left & Right unaligned overlay
                    axs[0].imshow((L_viz + R_viz) / 2, cmap='gray')
                    axs[0].set_title('Before: Original BF Images (Unaligned Overlay)')
                    axs[0].set_xticks([]); axs[0].set_yticks([])

                    # Panel 2: The transformed moving channel
                    if self.args.channel == "L":
                        axs[1].imshow(cv2.normalize(R_reg, None, 0, 1, cv2.NORM_MINMAX), cmap='gray')
                        axs[1].set_title('Registered Moving Channel (Right mapped to Left)')
                    else:
                        axs[1].imshow(cv2.normalize(L_reg, None, 0, 1, cv2.NORM_MINMAX), cmap='gray')
                        axs[1].set_title('Registered Moving Channel (Left mapped to Right)')
                    axs[1].set_xticks([]); axs[1].set_yticks([])

                    # Panel 3: After overlay
                    axs[2].imshow(overlay_viz, cmap='gray')
                    axs[2].set_title('After: Final Registered BF Overlay')
                    axs[2].set_xticks([]); axs[2].set_yticks([])
                    
                    plt.savefig(f"{self.args.save_dir}/BF_Registered_overlay.png", bbox_inches='tight')
                    plt.close()
                # -----------------------------------------------------
            else:
                bf_overlay_full = bf_sum

            # 4. Get the exact same ROI coordinates we used for the FL video
            y1, y2, x1, x2 = ImageProcessor.read_roi(
                self.args.roi_file, self.args.roi_channel, self.args.channel, 
                raw_shape[1], raw_shape[2], is_split_view
            )
            
            # 5. CROP the properly registered Brightfield overlay identically to the FL video
            bf_cropped = bf_overlay_full[y1:y2, x1:x2]
            
            # Normalize for saving
            bf_norm = cv2.normalize(bf_cropped, None, 0, 65535, cv2.NORM_MINMAX, dtype=cv2.CV_16U)

            # Save exactly as 'BF.tif' so fields_of_view.py can find it!
            tifffile.imwrite(f"{self.args.save_dir}/BF.tif", bf_norm, imagej=True)
            self.save_image_with_scalebar(bf_norm, f"{self.args.save_dir}/BF.png", cmap="gray")
            
            # Save the full uncropped reg image just in case you need it later
            full_bf_norm = img_as_uint(np.clip(cv2.normalize(bf_overlay_full, None, 0, 1, cv2.NORM_MINMAX),0,1))
            
            tifffile.imwrite(f"{self.args.save_dir}/BF_reg_full.tif", full_bf_norm, imagej=True)

        return bf_sum, bf_cropped
    
    def save_image_with_scalebar(self, image, save_path, cmap="gray", bar_length_um=1):
        fig, ax = plt.subplots()
        ax.imshow(image, cmap=cmap)
        ax.axis('off')
        
        # Calculate how many pixels make up the desired micrometer length
        pixels = bar_length_um / self.pxsize 
        
        scalebar = AnchoredSizeBar(
            ax.transData, 
            pixels, 
            f'{bar_length_um} µm', 
            'lower right', 
            pad=0.5, 
            color='white', 
            frameon=False, 
            size_vertical=max(1, int(image.shape[0] * 0.01)), # Automatically scales bar thickness
            fontproperties=fm.FontProperties(size=12)
        )
        ax.add_artist(scalebar)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

    def fit_cell(self, cell_mask, obj_num, channel, save_dir):
        contours = measure.find_contours(cell_mask, 0.5)
        if not contours: return None, None, None
        contour = max(contours, key=len)
        
        # 1. PCA for initial geometric parameters (center, angle, length, radius)
        y_data, x_data = contour.T
        pts = np.vstack([x_data, y_data]).T
        mean = np.mean(pts, axis=0)

        cov = np.cov(pts, rowvar=False)
        evals, evecs = np.linalg.eigh(cov)

        # Sort eigenvectors by eigenvalue (major axis first)
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:, idx]

        # Orientation along principal axis
        initial_theta = np.arctan2(evecs[1, 0], evecs[0, 0])

        # Initial length and width span
        centered = pts - mean
        proj_major = centered @ evecs[:, 0]
        proj_minor = centered @ evecs[:, 1]

        major_span = np.max(proj_major) - np.min(proj_major)
        minor_span = np.max(proj_minor) - np.min(proj_minor)

        initial_R = max(1.0, minor_span / 2.0)
        initial_L = max(0.1, major_span - 2.0 * initial_R)
        initial_xc, initial_yc = mean[0], mean[1]
        
        max_R = max(initial_R * 3.0, 50.0)
        max_L = max(initial_L * 3.0, 200.0)
        
        lower_bounds = [0.1, 0.0, -np.inf, -np.inf, -np.pi]
        upper_bounds = [max_R, max_L, np.inf, np.inf, np.pi]
        
        # 2. Euclidean distance to central line segment: dist(point, segment) - R
        def residuals_capsule(params, x_data, y_data):
            R, L, xc, yc, theta = params
            
            dx = x_data - xc
            dy = y_data - yc
            
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            
            # Transform contour points into local rotated capsule coordinates
            x_local = dx * cos_t + dy * sin_t
            y_local = -dx * sin_t + dy * cos_t
            
            # Perpendicular Euclidean distance to line segment [-L/2, L/2] along x-axis
            x_clamped = np.clip(x_local, -L / 2.0, L / 2.0)
            dist_to_segment = np.sqrt((x_local - x_clamped)**2 + y_local**2)
            
            # Residual is signed distance to capsule boundary (radius R)
            return dist_to_segment - R

        # 3. Fit using least_squares with PCA estimates as p0 (including initial_theta)
        p0 = [initial_R, initial_L, initial_xc, initial_yc, initial_theta]
        res = optimize.least_squares(
            residuals_capsule, p0, 
            args=(x_data, y_data), bounds=(lower_bounds, upper_bounds)
        )
        R_fit, L_fit, xc_fit, yc_fit, theta_fit = res.x
        fit_error = np.sqrt(np.mean(res.fun**2))
        
        # Unit conversion
        scale_factor = 1e9 if self.pxsize < 1e-4 else 1000
        
        cell_length = (2 * R_fit + L_fit) * self.pxsize * scale_factor
        cell_width = 2 * R_fit * self.pxsize * scale_factor
        error_nm = fit_error * self.pxsize * scale_factor

        print(f"\n--- Optimal Fit for cell {obj_num} ---", flush=True)
        print(rf"Radius (R): {R_fit:.2f} px, Length (L): {L_fit:.2f} px, Angle: {np.rad2deg(theta_fit):.1f}°", flush=True)
        print(f"Residual Error: {fit_error:.3f} px", flush=True)
        
        # 4. Generate fitted capsule outline for visualization & HTML overlay
        plt.figure(figsize=(8, 6))
        plt.imshow(cell_mask, cmap='binary', alpha=0.5)
        plt.plot(x_data, y_data, 'b.', markersize=2, alpha=0.5, label='Mask Contour')

        t_cap = np.linspace(-np.pi/2, np.pi/2, 100)
        # Right cap semicircle
        x_right_cap = L_fit/2 + R_fit * np.cos(t_cap)
        y_right_cap = R_fit * np.sin(t_cap)
        # Left cap semicircle
        x_left_cap = -L_fit/2 - R_fit * np.cos(t_cap)
        y_left_cap = -R_fit * np.sin(t_cap)
        
        x_m = np.concatenate([x_right_cap, x_left_cap, [x_right_cap[0]]])
        y_m = np.concatenate([y_right_cap, y_left_cap, [y_right_cap[0]]])

        x_fitted = xc_fit + x_m * np.cos(theta_fit) - y_m * np.sin(theta_fit)
        y_fitted = yc_fit + x_m * np.sin(theta_fit) + y_m * np.cos(theta_fit)

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
            'Cell_Number': obj_num,
            'Channel': channel,
            'Length_nm': round(cell_length, 2),
            'Width_nm': round(cell_width, 2),
            'Center_X': round(xc_fit, 2),
            'Center_Y': round(yc_fit, 2),
            'Rotation_deg': round(np.rad2deg(theta_fit), 2),
            'Error_nm': round(error_nm, 3)
        }
        return fit_params, x_fitted, y_fitted
    
    def extract_skeleton_centerlines(self, fitted_mask, num_lines=5, spacing=1):
        """
        Skeletonizes binary cell mask, fits a centerline polynomial, and computes 
        parallel offset pixel coordinates bounded by cell mask dimensions.
        """
        from skimage.morphology import skeletonize
        from skimage.draw import line

        height, width = fitted_mask.shape
        skeleton_mask = skeletonize(fitted_mask)
        out_arr_skeleton = np.argwhere(skeleton_mask)
        
        if len(out_arr_skeleton) == 0:
            raise ValueError("Skeletonization failed: No mask pixels found.")
            
        y_coords, x_coords = out_arr_skeleton[:, 0], out_arr_skeleton[:, 1]
        slope, intercept = np.polyfit(x_coords, y_coords, 1)
        print(f"Skeleton Centerline: y = {slope:.4f}x + {intercept:.4f}", flush=True)

        # Base centerline line across image width
        x0, y0 = 0, int(round(slope * 0 + intercept))
        x1, y1 = width - 1, int(round(slope * (width - 1) + intercept))
        rr, cc = line(y0, x0, y1, x1)
        
        # Calculate parallel line offsets along normal vector
        half_n = num_lines // 2
        offsets = [i * spacing for i in range(-half_n, half_n + 1)]
        theta = np.arctan(slope)
        dx, dy = -np.sin(theta), np.cos(theta)
        
        all_rr = np.array([np.round(rr + offset * dy).astype(int) for offset in offsets])
        all_cc = np.array([np.round(cc + offset * dx).astype(int) for offset in offsets])
        
        # Mask valid coordinates within frame boundaries
        valid_cols = ((all_rr >= 0) & (all_rr < height) & (all_cc >= 0) & (all_cc < width)).all(axis=0)
        all_rr_valid = all_rr[:, valid_cols]
        all_cc_valid = all_cc[:, valid_cols]
        
        # Determine cropped start/end indices along central line
        center_line_idx = num_lines // 2
        mask_line_values = fitted_mask[all_rr_valid[center_line_idx], all_cc_valid[center_line_idx]]
        mask_indices = np.where(mask_line_values > 0)[0]
        
        if len(mask_indices) == 0:
            raise ValueError("No cell mask detected along centerline!")
            
        start_idx = max(0, mask_indices[0] - 5)
        end_idx = min(len(mask_line_values), mask_indices[-1] + 5 + 1)
        
        return {
            'all_rr_valid': all_rr_valid,
            'all_cc_valid': all_cc_valid,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'skeleton_mask': skeleton_mask,
            'slope': slope,
            'intercept': intercept
        }    

    def line_profile(self, fitted_mask_file, save_dir=".", is_ALEX=False, alex_start_frame=0):
        import tifffile
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import os

        fitted_mask = tifffile.imread(fitted_mask_file)
        num_channels = int(self.args.num_channels)

        # 1. Extract skeleton, centerline fit, and parallel line offset coordinates
        skel = self.extract_skeleton_centerlines(fitted_mask, num_lines=5, spacing=1)
        all_rr_valid, all_cc_valid = skel['all_rr_valid'], skel['all_cc_valid']
        start_idx, end_idx = skel['start_idx'], skel['end_idx']

        def get_channel_avg(raw_path):
            if not os.path.exists(raw_path):
                return None
            raw = tifffile.imread(raw_path)
            if raw.ndim == 3:
                frames = raw[alex_start_frame::2] if is_ALEX else raw
                return np.mean(frames, axis=0)
            return raw

        def compute_stats(raw_img):
            intensities = np.array([raw_img[r, c] for r, c in zip(all_rr_valid, all_cc_valid)])
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
            
            return {
                'cropped': cropped,
                'x_full': x_full,
                'mean_full': mean_full,
                'sem_full': sem_full,
                'x_half': x_half,
                'mean_left': mean_left,
                'sem_left': sem_left,
                'mean_right': mean_right,
                'sem_right': sem_right
            }

        def save_single_channel_outputs(ch_label, stats, color_l="blue", color_r="orange", color_full="green"):
            # Plot 1: Halves (Left vs Right)
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

            # Plot 2: Full Length Profile
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

            # Export Individual Channel CSV
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

        # 2. Determine target channels strictly by num_channels
        channels_to_process = [self.args.channel] if num_channels == 1 else ["L", "R"]

        stats_dict = {}
        for ch in channels_to_process:
            ch_file = os.path.join(self.args.save_dir, f"{ch}_channel.tif")
            raw_avg = get_channel_avg(ch_file)
            if raw_avg is not None:
                stats = compute_stats(raw_avg)
                stats_dict[ch] = stats
                
                # Assign Green for L and Red for R
                ch_color = "green" if ch == "L" else "red"
                save_single_channel_outputs(ch, stats, color_full=ch_color)
            else:
                print(f"Warning: Channel file {ch_file} not found for line profile.", flush=True)

        # 3. Dual-Channel Combined Comparison (Only when num_channels == 2)
        if num_channels == 2 and "L" in stats_dict and "R" in stats_dict:
            stats_L = stats_dict["L"]
            stats_R = stats_dict["R"]

            # Combined Full Length Plot (L=Green, R=Red)
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

            # Combined CSV
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
    
    def create_interactive_html(self, images_dict, objects, fit_results_list, fit_outlines_dict, save_dir):
        """
        Generates an interactive Plotly HTML overlay containing:
        - Toggleable background layers (Brightfield, Channel L, Channel R)
        - Interactive Cell contours with hover info
        - Mathematical capsule fit overlays
        """
        print("Generating interactive multi-channel HTML overlay with fits...", flush=True)
        
        # 1. Normalize and validate input channel images
        valid_images = {}
        for name, img_data in images_dict.items():
            if img_data is not None:
                norm_img = cv2.normalize(img_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                valid_images[name] = norm_img

        if not valid_images:
            print("No valid images provided for HTML generation.", flush=True)
            return

        fig = go.Figure()
        image_keys = list(valid_images.keys())
        num_img_traces = len(image_keys)

        # 2. Add background image traces for each channel
        for idx, (img_name, norm_img) in enumerate(valid_images.items()):
            # Convert 2D grayscale array to 3D RGB for Plotly rendering
            rgb_img = np.dstack([norm_img, norm_img, norm_img]) if norm_img.ndim == 2 else norm_img
            
            fig.add_trace(go.Image(
                z=rgb_img,
                name=img_name,
                visible=(idx == 0),  # Only the first channel is visible initially
                hoverinfo='skip'
            ))

        # 3. Add Cell Masks and Mathematical Fits
        fit_dict = {f['Cell_Number']: f for f in fit_results_list}
        area_thresh = int(self.args.area_filter)

        for obj in objects:
            status = "Accepted" if obj.area > area_thresh else "Rejected"
            color = None if status == "Accepted" else "rgba(255, 0, 0, 0.4)" 
            obj_num = obj.label

            if obj.image.shape[0] < 2 or obj.image.shape[1] < 2:
                continue

            fit_data = fit_dict.get(obj_num)
            fit_outline = fit_outlines_dict.get(obj_num)

            # Build rich hover tooltip
            hover_text = (
                f"<b>Object {obj_num}</b><br>"
                f"Status: {status}<br>"
                f"Area: {obj.area} px<br>"
            )
            if fit_data:
                hover_text += (
                    f"Length: {fit_data['Length_nm']} nm<br>"
                    f"Width: {fit_data['Width_nm']} nm<br>"
                    f"Error: {fit_data['Error_nm']} nm"
                )

            # Draw real cell segmentation boundary
            min_y, min_x, max_y, max_x = obj.bbox
            padded_image = np.pad(obj.image, pad_width=1, mode='constant', constant_values=0)
            contours = measure.find_contours(padded_image, 0.5)

            for contour in contours:
                global_y = contour[:, 0] + min_y - 1
                global_x = contour[:, 1] + min_x - 1

                fig.add_trace(go.Scatter(
                    x=global_x, y=global_y, mode='lines', fill='toself', fillcolor=color,
                    line=dict(color=color if color else 'yellow', width=1.5),
                    name=f"Cell {obj_num}", text=hover_text, hoverinfo="text", showlegend=False,
                    visible=True
                ))

            # Draw fitted capsule shape
            if fit_outline and status == "Accepted":
                x_fit, y_fit = fit_outline
                fig.add_trace(go.Scatter(
                    x=x_fit, y=y_fit, mode='lines',
                    line=dict(color='cyan', width=2, dash='dot'),
                    name=f"Fit {obj_num}", hoverinfo='skip', showlegend=False,
                    visible=True
                ))

        total_traces = len(fig.data)

        # 4. Construct Dropdown Buttons to switch active channel background
        buttons = []
        for idx, name in enumerate(image_keys):
            # Visibility mask: set trace `idx` to True, other image traces to False, and keep all contour/fit traces True
            vis_state = [False] * num_img_traces + [True] * (total_traces - num_img_traces)
            vis_state[idx] = True
            
            buttons.append(dict(
                label=f"Channel: {name}",
                method="update",
                args=[{"visible": vis_state},
                      {"title": f"Interactive Cell Overlay | Viewing: {name}"}]
            ))

        # 5. Apply layout settings
        fig.update_layout(
            title=f"Interactive Cell Mask Overlay | Viewing: {image_keys[0]}",
            updatemenus=[dict(
                active=0,
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.15,
                yanchor="top",
                bgcolor="rgba(255, 255, 255, 0.9)"
            )],
            margin=dict(l=0, r=0, b=0, t=60),
            coloraxis_showscale=False,
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False, autorange="reversed") # Standard image coordinate orientation
        )

        html_path = f"{save_dir}/interactive_cell_overlay.html"
        fig.write_html(html_path)
        print(f"Saved multi-channel interactive overlay to: {html_path}", flush=True)
    
    def run(self):
        print(f"Starting analysis with mask type: {self.args.mask_type}", flush=True)
        
        # FIX: Try/except block added back around shutil.rmtree to prevent silent wrapper crashes
        if str(self.args.overwrite).lower() in ['true', '1', 't', 'y']:
            if "results" in self.args.save_dir and os.path.exists(self.args.save_dir):
                try:
                    shutil.rmtree(self.args.save_dir)
                except Exception as e:
                    print(f"Notice: Could not completely clear save directory (file might be open): {e}", flush=True)
                    pass 
                
        os.makedirs(self.args.save_dir, exist_ok=True)
            
        FileHandler.save_parameters(vars(self.args), self.args.save_dir)
        
        if hasattr(self.args, 'bead_path') and self.args.bead_path:
            target_tmat_path = "transformation_matrix.npy"
            if not os.path.exists(target_tmat_path):
                registrar = BeadRegistrar()
                if str(self.args.manual_registration).lower() in ['true', '1', 't', 'y']:
                    # Pass the single calibration image path
                    registrar.interactive_manual_registration(
                        self.args.bead_path, 
                        ".", 
                        int(self.args.manual_pairs)
                    )
                else:
                    # Run the existing automated StackReg logic
                    registrar.calculate_transformation(self.args.bead_path, ".", self.args.channel)
            self.args.tmats_path = target_tmat_path
            
        # 1. Video Processing
        L_chan, R_chan, num_frames, raw_shape = VideoProcessor.process_video(
            self.args.video_path, 0, self.num_frames, self.args.roi_channel, 
            self.args.roi_file, self.args.save_dir, int(self.args.num_channels), 
            self.args.channel, self.args.ALEX
        )
        
        if L_chan is not None:
            #L_avg = np.mean(L_chan[:int(self.args.frame_avg)], axis=0).astype(np.uint16)
            L_avg = np.mean(L_chan[:5], axis=0).astype(np.uint16)
            tifffile.imwrite(f"{self.args.save_dir}/L_avg.tif", L_avg, imagej=True)
            self.save_image_with_scalebar(L_avg, f"{self.args.save_dir}/L_avg.png")
            tifffile.imwrite(f"{self.args.save_dir}/L_channel.tif", L_chan, imagej=True)
        if R_chan is not None:
            #R_avg = np.mean(R_chan[:int(self.args.frame_avg)], axis=0).astype(np.uint16)
            R_avg = np.mean(R_chan[:5], axis=0).astype(np.uint16)
            tifffile.imwrite(f"{self.args.save_dir}/R_avg.tif", R_avg, imagej=True)
            self.save_image_with_scalebar(R_avg, f"{self.args.save_dir}/R_avg.png")
            tifffile.imwrite(f"{self.args.save_dir}/R_channel.tif", R_chan, imagej=True)

        width = raw_shape[2]
        img_for_masking = None
        sr = StackReg(StackReg.AFFINE)
        tmats = np.load(self.args.tmats_path) if self.args.tmats_path and os.path.exists(self.args.tmats_path) else None

        # 2. Frame Averaging & Brightfield Processing
        img_for_masking = None
        bf_sum = None 
        
        bf_sum, bf_cropped = self.process_brightfield(raw_shape, sr, tmats)
        
        target_chan = L_chan if self.args.channel == "L" else R_chan
        fl_avg = None
        if target_chan is not None:
            fl_avg = np.mean(target_chan[:int(self.args.frame_avg)], axis=0).astype(np.uint16)
        
        # Set the image to be used for masking
        if self.args.mask_type == "BF" and self.args.bf_path and bf_cropped is not None:
            img_for_masking = bf_cropped
            
        # elif self.args.mask_type == "MANUAL":
        #     # Pass both brightfield and fluorescence images to populate the switcher menu
        #     target_chan = L_chan if self.args.channel == "L" else R_chan
        #     mask = ImageProcessor.interactive_edit_mask(bg_img=bf_cropped, fl_img=fl_avg)
            
        elif self.args.mask_type in ["FL_AI", "THRESHOLD", "AI", "WHOLE","MANUAL"]:
            target_chan = L_chan if self.args.channel == "L" else R_chan
            img_for_masking = np.mean(target_chan[:int(self.args.frame_avg)], axis=0).astype(np.uint16)
        
        # 3. Mask Generation & Labeling
        labels = None
        
        if self.args.mask_type in ["AI", "BF", "FL_AI"]:
            
            # --- PATH 1: OMNITORCH ---
            if self.args.model_type == "omnitorch":
                print("Omnipose execution deferred to external RunOmnipose script.", flush=True)
                from segmentation_models import RunOmnipose
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
                
                # 1. Generate the 3-channel visual mask via Omnipose
                visual_mask = RunOmnipose.main(img_for_masking, self.args.model, self.args.save_dir)
                io.imsave(f'{self.args.save_dir}/visual_mask_omni.tif', visual_mask, check_contrast=False)
                
                # 2. Create a blank 2D canvas based on the original image dimensions
                prediction = np.zeros(img_for_masking.shape, dtype=np.uint8)
                
                # 3. Find unique colors (cells) in the mask
                flat_pixels = visual_mask.reshape(-1, visual_mask.shape[-1])
                unique_colors = np.unique(flat_pixels, axis=0)
                unique_colors = [c for c in unique_colors if np.any(c > 0)] # Exclude black background
                
                print(f"Found {len(unique_colors)} unique objects.", flush=True)
                
                # 4. Loop through every unique cell color
                for i, color in enumerate(unique_colors):
                    
                    # Create a boolean mask where this specific color exists
                    mask_boolean = np.all(visual_mask == color, axis=-1)
                    
                    # Convert to an 8-bit image (255 for the cell, 0 for background)
                    object_mask = mask_boolean.astype(np.uint8) * 255
                    
                    # Save the individual cell mask
                    color_string = f"{color[0]}-{color[1]}-{color[2]}"
                    save_path = f'{self.args.save_dir}/object_mask_{color_string}.tif'
                    io.imsave(save_path, object_mask, check_contrast=False)
                    
                    # Erode the object and add it to our blank canvas
                    eroded_object = cv2.erode(object_mask, kernel=kernel, iterations=1)
                    prediction += eroded_object
                
                # 5. Cap the max value and output the final 1-channel mask
                prediction[prediction > 255] = 0
                print(f"Prediction shape {prediction.shape}", flush=True)    
                
                # Pass the processed 1-channel prediction to the rest of the pipeline
                mask = prediction
                
            # --- PATH 2: CELLPOSE ---
            elif self.args.model_type == "cellpose":
                print("Cellpose execution deferred to external RunCellpose script.", flush=True)
                from segmentation_models import RunCellpose

                pretrained_model = self.args.model if (self.args.model and os.path.exists(str(self.args.model))) else None
                cp_model_type = self.args.model if (self.args.model and not pretrained_model) else "bact_phase_cp3"

                channels = getattr(self.args, 'channels', [0, 0])
                diameter = getattr(self.args, 'diameter', None)
                flow_threshold = getattr(self.args, 'flow_threshold', 0.4)
                cellprob_threshold = getattr(self.args, 'cellprob_threshold', 0.0)
                gpu = getattr(self.args, 'gpu', torch.cuda.is_available() if 'torch' in globals() else False)

                RunCellpose.main(
                    img_obj=img_for_masking,
                    model_type=cp_model_type,
                    pretrained_model=pretrained_model,
                    save_dir=self.args.save_dir,
                    channels=channels,
                    diameter=diameter,
                    flow_threshold=flow_threshold,
                    cellprob_threshold=cellprob_threshold,
                    gpu=gpu
                )

                raw_mask_path = os.path.join(self.args.save_dir, "output_raw_mask_cellpose.tif")
                if os.path.exists(raw_mask_path):
                    mask = tifffile.imread(raw_mask_path)
                else:
                    raise FileNotFoundError(f"Cellpose output raw mask not found at {raw_mask_path}")
                
            # --- PATH 3: KERAS / U-NET ---
            
            elif self.args.model_type in ["unet", "keras"]:
                        from segmentation_models import RunUNet
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            
                        # --- Read dilations from *params.dat ---
                        num_dilations = getattr(self, 'dilations', 1)
                        dat_candidates = (
                            glob.glob(os.path.join(self.args.save_dir, "*params.dat")) +
                            glob.glob(os.path.join(os.path.dirname(getattr(self, 'video_path', '') or ''), "*params.dat")) +
                            glob.glob("*params.dat")
                        )
                        
                        if dat_candidates and os.path.exists(dat_candidates[0]):
                            dat_path = dat_candidates[0]
                            try:
                                with open(dat_path, 'r', encoding='utf-8', errors='ignore') as f:
                                    for line in f:
                                        if 'dilations' in line.lower():
                                            parts = line.replace(':', ' ').replace('=', ' ').split()
                                            for idx, part in enumerate(parts):
                                                if part.lower() == 'dilations' and idx + 1 < len(parts):
                                                    num_dilations = int(parts[idx + 1])
                                                    print(f"Read dilations={num_dilations} from {dat_path}", flush=True)
                                                    break
                            except Exception as e:
                                print(f"Error reading dilations from {dat_path}: {e}", flush=True)
            
                        thresh_param = getattr(self.args, 'model_params', None) or getattr(self.args, 'threshold_param', None)
                        
                        mask = RunUNet.main(
                            img_obj=img_for_masking,
                            model_or_path=self.model,  # Accepts loaded model instance OR path string
                            save_dir=self.args.save_dir,
                            threshold_param=thresh_param,
                            inv_bf=self.args.inv_bf,
                            debug=False
                        )
                        
                        io.imsave(f'{self.args.save_dir}/visual_mask_unet.tif', mask, check_contrast=False)
                        
                        # 2. Create a blank 2D canvas based on the original image dimensions
                        prediction = np.zeros(img_for_masking.shape, dtype=np.uint8)
                        
                        # 3. Find unique colors (cells) in the mask
                        flat_pixels = mask.reshape(-1, mask.shape[-1])
                        unique_colors = np.unique(flat_pixels, axis=0)
                        unique_colors = [c for c in unique_colors if np.any(c > 0)] # Exclude black background
                        
                        print(f"Found {len(unique_colors)} unique objects.", flush=True)
                        
                        # 4. Loop through every unique cell color
                        for i, color in enumerate(unique_colors):
                            
                            # Create a boolean mask where this specific color exists
                            mask_boolean = np.all(mask == color, axis=-1)
                            
                            # Convert to an 8-bit image (255 for the cell, 0 for background)
                            object_mask = mask_boolean.astype(np.uint8) * 255
                            
                            # Save the individual cell mask
                            color_string = f"{color[0]}-{color[1]}-{color[2]}"
                            save_path = f'{self.args.save_dir}/object_mask_{color_string}.tif'
                            io.imsave(save_path, object_mask, check_contrast=False)
                            
                            # Apply dilations / erosions based on *params.dat value
                            if num_dilations > 0:
                                processed_object = cv2.dilate(object_mask, kernel=kernel, iterations=num_dilations)
                            elif num_dilations < 0:
                                processed_object = cv2.erode(object_mask, kernel=kernel, iterations=abs(num_dilations))
                            else:
                                processed_object = object_mask
            
                            prediction += processed_object
                        
                        # 5. Cap the max value and output the final 1-channel mask
                        prediction[prediction > 255] = 0
                        print(f"Prediction shape {prediction.shape}", flush=True)    
                        
                        # Pass the processed 1-channel prediction to the rest of the pipeline
                        mask = prediction
            
            # --- PATH 3: STANDARD PYTORCH ---
            elif self.args.model_type == "pytorch":
                import torch
                patches, h, w = ImageProcessor.make_patches(img_for_masking, self.args.inv_bf)
                preds = []
                with torch.no_grad():
                    for p in patches:
                        # Convert patch to tensor [Batch, Channel, Height, Width]
                        tensor_patch = torch.tensor(p, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                        out = self.model(tensor_patch)
                        # Convert back to numpy array
                        pred_patch = out.squeeze().cpu().numpy() * 255
                        preds.append(pred_patch)
                mask = ImageProcessor.stitch_patches(preds, img_for_masking.shape, h, w)
                
        # --- FALLBACK: CLASSICAL THRESHOLDING ---
        elif self.args.mask_type == "THRESHOLD":
            mask = ImageProcessor.apply_watershed_threshold(
                img_for_masking, 
                dist_multiplier=0.3,
                use_otsu=self.args.use_otsu
            )
            
        elif self.args.mask_type == "WHOLE":
            mask = np.ones_like(img_for_masking, dtype=np.uint8) * 255
            
        elif self.args.mask_type == "MANUAL":
            # Get the raw manual mask containing distinct integer IDs for each drawn cell
            raw_manual_mask = ImageProcessor.interactive_edit_mask(bg_img=bf_cropped, fl_img=fl_avg)
            
            contracted_mask = np.zeros_like(raw_manual_mask, dtype=np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            
            # Iterate over each individual cell label ID (skipping background 0)
            unique_labels = [label_id for label_id in np.unique(raw_manual_mask) if label_id > 0]
            for label_id in unique_labels:
                # Isolate the current cell's binary mask
                single_cell_mask = (raw_manual_mask == label_id).astype(np.uint8)
                
                # Apply morphological erosion to contract the label slightly
                eroded_cell = cv2.erode(single_cell_mask, kernel, iterations=1)
                
                # Add contracted label back to the final binary mask canvas
                contracted_mask[eroded_cell > 0] = 255
                
            mask = contracted_mask 
            
        elif self.args.mask_type == "file":
            mask = io.imread(glob.glob(f"{self.args.save_dir}/{self.args.mask_prefix}*.tif")[0])
            mask = (mask > 0).astype(np.uint8) * 255
            
        tifffile.imwrite(f"{self.args.save_dir}/all_cells_mask.tif", mask.astype(np.uint16), imagej=True)
        self.save_image_with_scalebar(mask, f"{self.args.save_dir}/all_cells_mask.png", cmap="viridis")

        # ==========================================================
        # Dual Mask Generation 
        # ==========================================================
        if int(self.args.num_channels) == 2 and self.args.channel == "L":
            mask_l, mask_r = ImageProcessor.generate_dual_masks(mask, self.args.channel)
        else:
            # If only 1 channel, strictly assign the mask to the target channel
            mask_l = mask if self.args.channel == 'L' else None
            mask_r = mask if self.args.channel == 'R' else None
        # ==========================================================

        # 4. Region Extraction
        if labels is None:
            labels = measure.label(mask > 0)
            
        objects = measure.regionprops(labels)
        all_objects_data = []
        mask_stack_list = []
        all_fit_results = []
        fit_outlines = {}
        
        global_bg_mask = (mask == 0)
        global_median_bgs = {}
        
        for c_name, c_data in [("L", L_chan), ("R", R_chan)]:
            if c_data is not None:
                c_avg = np.mean(c_data[:int(self.args.frame_avg)], axis=0)
                bg_pixels = c_avg[global_bg_mask]
                global_median_bgs[c_name] = float(np.median(bg_pixels)) if len(bg_pixels) > 0 else 0.0
        
        for obj_num, obj in enumerate(objects, start=1):
            if obj.area < int(self.args.area_filter): continue
            
            cell_dir = f"{self.args.save_dir}/cell_{obj_num}"
            os.makedirs(cell_dir, exist_ok=True)
            
            obj_data = {prop: getattr(obj, prop) for prop in ['label', 'area', 'bbox', 'centroid', 'eccentricity', 'major_axis_length', 'minor_axis_length', 'orientation', 'solidity', 'perimeter'] if hasattr(obj, prop)}
            
            cell_mask = (labels == obj.label).astype(np.uint8)
            mask_stack_list.append(cell_mask)
             
            extracted_intensities = {}    
            
            # Extract the bounding box for this specific cell
            min_y, min_x, max_y, max_x = obj.bbox
            mask_crop = cell_mask[min_y:max_y, min_x:max_x]

            for chan_name, chan_data in [("L", L_chan), ("R", R_chan)]:
                if chan_data is not None:
                    # 1. Extract mathematical intensities from the full video
                    intensities = []
                    for frame in chan_data[:num_frames]:
                        pixels = frame[cell_mask > 0]
                        intensities.append(np.mean(pixels) if len(pixels) > 0 else 0)
                        
                    extracted_intensities[chan_name] = intensities
                    
                    # 2. Create the ROI-sized 2D average image
                    chan_avg = np.mean(chan_data[:int(self.args.frame_avg)], axis=0)
                    
                    mask_pixels = chan_avg[cell_mask > 0]
                    obj_data[f'mean_intensity_{chan_name}'] = float(np.mean(mask_pixels)) if len(mask_pixels) > 0 else 0.0
                    obj_data[f'total_intensity_{chan_name}'] = float(np.sum(mask_pixels)) if len(mask_pixels) > 0 else 0.0
                    obj_data[f'global_median_bg_{chan_name}'] = global_median_bgs[chan_name]

                    chan_avg_uint = chan_avg.astype(np.uint16)
                    
                    # Create a copy to apply the mask
                    masked_roi_img = chan_avg_uint.copy()
                    
                    # Identify the background (everything outside this specific cell's mask)
                    bg_mask = (cell_mask == 0)
                    
                    # Fill the background with the mean background value of those pixels
                    if np.any(bg_mask):
                        masked_roi_img[bg_mask] = np.mean(masked_roi_img[bg_mask])
                        
                    # Save the ROI-sized average as TIF
                    tifffile.imwrite(f"{cell_dir}/cellmasked_{chan_name}.tif", masked_roi_img, imagej=True)
                    
                    # Normalize to 8-bit and save as PNG for visual checking
                    cell_png_norm = cv2.normalize(masked_roi_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    io.imsave(f"{cell_dir}/cellmasked_{chan_name}.png", cell_png_norm, check_contrast=False)
                    
                    # Plot Intensity and Fit Exponentials
                    frames_x = np.arange(len(intensities))
                    
                    # Save raw intensity to CSV
                    df_intensity = pd.DataFrame({
                        'Frame': frames_x,
                        'Intensity': intensities
                    })
                    df_intensity.to_csv(f"{cell_dir}/raw_intensity_{chan_name}.csv", index=False)

                    # Capture the returned fit quality data
                    fit_quality = IntensityAnalyzer.plot_intensity([frames_x, intensities], chan_name, obj_num, len(intensities), cell_dir)
                    
                    # Parse the fit data into a structured format
                    fit_rows = []
                    model_names = ["Single Exp", "Double Exp", "Triple Exp"]
                    
                    for i, fit_res in enumerate(fit_quality):
                        if fit_res != 0:
                            params, rchisq = fit_res
                            row = {'Model': model_names[i], 'Reduced_ChiSq': rchisq, 'Status': 'Success'}
                            
                            if len(params) >= 2:
                                row.update({'A': params[0], 'k': params[1]})
                            if len(params) >= 4:
                                row.update({'A1': params[2], 'k1': params[3]})
                            if len(params) >= 6:
                                row.update({'A2': params[4], 'k2': params[5]})
                                
                            fit_rows.append(row)
                        else:
                            fit_rows.append({'Model': model_names[i], 'Status': 'Fit Failed'})
                            
                    df_fit_params = pd.DataFrame(fit_rows)
                    df_fit_params.to_csv(f"{cell_dir}/intensity_fit_params_{chan_name}.csv", index=False)
            
            # Append object data after all active channel intensity features are added
            all_objects_data.append(obj_data)
            
            mat_data = {
                'cell_num': obj_num,
                'area': obj.area,
                'centroid': obj.centroid,
                'bbox': obj.bbox,
                'cell_mask': cell_mask.astype(np.float64),
                'major_axis_length': obj.major_axis_length,
                'minor_axis_length': obj.minor_axis_length,
                'eccentricity': obj.eccentricity,
                'orientation': obj.orientation
            }
            
            if str(self.args.cell_fitting).lower() == "true":
                fit_result, x_fit, y_fit = self.fit_cell(cell_mask, obj_num, self.args.channel, cell_dir)
                if fit_result:
                    all_fit_results.append(fit_result)
                    fit_outlines[obj_num] = (x_fit, y_fit)
                    mat_data['fit_length_nm'] = fit_result['Length_nm']
                    mat_data['fit_width_nm'] = fit_result['Width_nm']
                    
                    # LINE PROFILE EXTRACTION
                    single_mask_file = f"{cell_dir}/single_cell_mask.tif"
                    tifffile.imwrite(single_mask_file, (cell_mask * 255).astype(np.uint8))

                    try:
                        self.line_profile(
                            fitted_mask_file=single_mask_file,
                            save_dir=cell_dir,
                            is_ALEX=str(self.args.ALEX).lower() in ['true', '1', 't', 'y']
                        )
                    except Exception as e:
                        print(f"Line fit failed for cell {obj_num}: {e}", flush=True)
            
            if 'L' in extracted_intensities: mat_data['intensity_L'] = extracted_intensities['L']
            if 'R' in extracted_intensities: mat_data['intensity_R'] = extracted_intensities['R']
                
            sio.savemat(f"{cell_dir}/cell_{obj_num}_data.mat", mat_data)

        # 5. Save Summary Outputs
        if all_fit_results:
            df_fits = pd.DataFrame(all_fit_results)
            csv_path = f"{self.args.save_dir}/cell_fitting_parameters.csv"
            df_fits.to_csv(csv_path, index=False)
            print(f"\nSaved all cell fitting parameters to: {csv_path}", flush=True)
        #if img_for_masking is not None:
        #    self.create_interactive_html(img_for_masking, objects, all_fit_results, fit_outlines, self.args.save_dir)
        images_dict = {}
        if bf_cropped is not None:
            images_dict["Brightfield (BF)"] = bf_cropped
        if L_chan is not None:
            images_dict["Channel L"] = np.mean(L_chan[:int(self.args.frame_avg)], axis=0).astype(np.uint16)
        if R_chan is not None:
            images_dict["Channel R"] = np.mean(R_chan[:int(self.args.frame_avg)], axis=0).astype(np.uint16)
        # Generate multi-channel HTML
        if images_dict:
            self.create_interactive_html(images_dict, objects, all_fit_results, fit_outlines, self.args.save_dir)
        if all_objects_data and mask_stack_list:
            stack_masks = np.array(mask_stack_list)
            tifffile.imwrite(f"{self.args.save_dir}/mask_stack.tif", stack_masks * 255, imagej=True)

            cleaned_objects_data = []
            for obj_dict in all_objects_data:
                row_data = {}
                for key, value in obj_dict.items():
                    if isinstance(value, (tuple, list, np.ndarray)): row_data[key] = str(value)
                    else: row_data[key] = value
                cleaned_objects_data.append(row_data)
            
            all_objects_df = pd.DataFrame(cleaned_objects_data)
            all_objects_df.to_csv(f"{self.args.save_dir}/all_objects.csv", index=False)
            
            is_split_view = raw_shape[2] > 1500
            y1, y2, x1, x2 = ImageProcessor.read_roi(
                self.args.roi_file, self.args.roi_channel, self.args.channel, 
                raw_shape[1], raw_shape[2], is_split_view
            )
            
            p = {
                'laser_centre_X': (x1 + x2) / 2, 
                'laser_centre_Y': (y1 + y2) / 2, 
                'laser_radius_R': (x2 - x1) / 2,
                'laser_angle': 0
            }
            
            matlab_dict = {
                'area': all_objects_df['area'].values,
                'BF': bf_sum if bf_sum is not None else np.zeros((raw_shape[1], raw_shape[2]//2)), 
                'bf2fl_tform': [0,0],
                'CellObject': stack_masks.astype(np.float64),  
                'p': p
            }
            
            if self.args.channel == "L" and L_chan is not None: matlab_dict['Cellframe0'] = L_chan[0]
            elif self.args.channel == "R" and R_chan is not None: matlab_dict['Cellframe0'] = R_chan[0]
            
            filename = self.args.save_dir + "/" + self.args.video_path.split("/")[-1][:-4] + "_segmentation.mat"
            sio.savemat(filename, matlab_dict)
            print(f"Saved global MATLAB segmentation data to {filename}", flush=True)

            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].hist(all_objects_df['area'].dropna())
            ax[0].set_xlabel("Area (pixels)")
            ax[1].hist(all_objects_df['eccentricity'].dropna())
            ax[1].set_xlabel("Eccentricity")
            ax[2].scatter(all_objects_df['area'], all_objects_df['eccentricity'])
            ax[2].set_xlabel("Area")
            ax[2].set_ylabel("Eccentricity")
            plt.savefig(f"{self.args.save_dir}/obj_feature_summary.png")
            plt.close()

        print("Analysis pipeline completed successfully.", flush=True)

def run_preprocessing(params):
    """
    This is the entry point called by PySTACHIO.
    """
    print("\n" + "="*50, flush=True)
    print(f"PREPROCESSING SESSION: {params.name}", flush=True)
    print("="*50, flush=True)

    print(f"Video Path  : {params.video_path}")
    print(f"Model       : {params.model}")
    print(f"ALEX Mode   : {params.ALEX}")
    print("="*50 + "\n", flush=True)

    try:
        pipeline = AnalysisPipeline(params)
        pipeline.run()
        
    except Exception as e:
        print(f"\nCRITICAL ERROR DURING PREPROCESSING:\n{traceback.format_exc()}", flush=True)
        raise e