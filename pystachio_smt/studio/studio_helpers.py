import sys
import os
import traceback
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QSlider, QLineEdit,
                             QLabel, QRadioButton, QGroupBox, QFileDialog, 
                             QButtonGroup, QFormLayout, QMessageBox, QCheckBox,
                             QTabWidget, QGridLayout, QToolTip, QComboBox)
from PyQt6.QtCore import Qt

# --- External Custom Modules ---
try:
    import images
    import spots
    import parameters
    import postprocessing
    import trajectories
    import tracking
    MODULES_LOADED = True
    
    # =========================================================================
    # MONKEY PATCH: Fix for ValueError: cannot convert float NaN to integer
    # =========================================================================
    from functools import wraps
    
    def patch_find_in_frame(original_func):
        @wraps(original_func)
        def safe_find_in_frame(self, *args, **kwargs):
            try:
                return original_func(self, *args, **kwargs)
            except ValueError as ve:
                if "cannot convert float NaN to integer" in str(ve):
                    if not hasattr(self, 'positions'): self.positions = []
                    if not hasattr(self, 'num_spots'): self.num_spots = 0
                    return None
                raise ve
        return safe_find_in_frame

    if hasattr(spots, 'Spots') and hasattr(spots.Spots, 'find_in_frame'):
        spots.Spots.find_in_frame = patch_find_in_frame(spots.Spots.find_in_frame)
        
    for attr_name in dir(spots):
        attr = getattr(spots, attr_name)
        if isinstance(attr, type) and hasattr(attr, 'find_in_frame'):
            setattr(attr, 'find_in_frame', patch_find_in_frame(getattr(attr, 'find_in_frame')))

    if hasattr(tracking, 'track_frame'):
        orig_track_frame = tracking.track_frame
        @wraps(orig_track_frame)
        def safe_track_frame(*args, **kwargs):
            try:
                return orig_track_frame(*args, **kwargs)
            except ValueError as ve:
                if "cannot convert float NaN to integer" in str(ve):
                    if hasattr(spots, 'Spots'):
                        dummy = spots.Spots()
                        dummy.positions, dummy.num_spots = [], 0
                        return dummy
                    else:
                        class DummySpots:
                            def __init__(self):
                                self.positions, self.num_spots = [], 0
                        return DummySpots()
                raise ve
        tracking.track_frame = safe_track_frame

except ImportError as e:
    print(f"Warning: Custom backend module missing ({e}). Backend disabled.")
    MODULES_LOADED = False
    class DummyParams: pass
    parameters = type('Dummy', (object,), {'Parameters': DummyParams})()

# --- Global Helpers ---
def to_imagej_compatible(arr):
    if arr is not None and arr.dtype == np.float64:
        return arr.astype(np.float32)
    return arr

def export_pystachio_params(params, file_path):
    try:
        with open(file_path, "w",encoding='utf-8') as f:
            for k in dir(params):
                if not k.startswith("__") and not callable(getattr(params, k)):
                    f.write(f"{k}={getattr(params, k)}\n")
    except Exception as e:
        print(f"Warning: Could not export PySTACHIO parameters. Error: {e}")

def show_popup(parent, title, text, critical=False):
    msg = QMessageBox(parent)
    msg.setWindowTitle(title)
    msg.setText(text)
    msg.setIcon(QMessageBox.Icon.Critical if critical else QMessageBox.Icon.Information)
    msg.setStyleSheet("QLabel{min-width: 600px; min-height: 100px;}")
    
    for label in msg.findChildren(QLabel):
        label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        label.setWordWrap(True)
        
    msg.exec()

def save_config_file(config_dict: dict, file_path: str, raw_lines: list = None, original_dict: dict = None):
    """
    Saves configuration parameters to a PySTACHIO config file.
    Preserves comments, sections, directives, and formatting of original file.
    Any parameters unchanged from original_dict/raw_lines remain identical.
    """
    written_keys = set()
    out_lines = []

    if raw_lines:
        for line in raw_lines:
            raw_line_stripped = line.strip()

            if raw_line_stripped and not raw_line_stripped.startswith('#') and '=' in line:
                if '#' in line:
                    kv_part, comment_part = line.split('#', 1)
                    comment_suffix = '  #' + comment_part.rstrip('\r\n')
                else:
                    kv_part = line.rstrip('\r\n')
                    comment_suffix = ''

                key_str, val_str = kv_part.split('=', 1)
                clean_key = key_str.strip().lower().replace(' ', '_')

                if clean_key in config_dict:
                    new_val = str(config_dict[clean_key]).strip()
                    orig_val = str(original_dict.get(clean_key, val_str.strip())).strip() if original_dict else val_str.strip()
                    written_keys.add(clean_key)

                    if orig_val == new_val:
                        out_lines.append(line.rstrip('\r\n') + '\n')
                    else:
                        new_line = f"{key_str.strip()}={new_val}{comment_suffix}\n"
                        out_lines.append(new_line)
                else:
                    out_lines.append(line.rstrip('\r\n') + '\n')
            else:
                out_lines.append(line.rstrip('\r\n') + '\n')

        # Append any new parameters not present in the original config
        unwritten = [k for k in config_dict if k not in written_keys and config_dict[k] is not None]
        if unwritten:
            out_lines.append("\n# --- NEW ADDED PARAMETERS ---\n")
            for k in unwritten:
                out_lines.append(f"{k}={config_dict[k]}\n")
    else:
        # Write default standard PySTACHIO config.txt template
        out_lines.append("# ==============================================================================\n")
        out_lines.append("# PYSTACHIO & PREPROCESSING MASTER CONFIGURATION\n")
        out_lines.append("# ==============================================================================\n")
        out_lines.append("preprocess\n\n")

        out_lines.append("# --- IMAGE PROPERTIES (UNIVERSAL) ---\n")
        for k in ['pixel_size', 'alex', 'num_frames', 'frame_time']:
            if k in config_dict: out_lines.append(f"{k}={config_dict[k]}\n")

        out_lines.append("\n# --- PREPROCESSING & SEGMENTATION ---\n")
        for k in ['video_path', 'bf_path', 'channel', 'start_channel', 'num_channels', 'frame_avg', 'save_dir', 'overwrite']:
            if k in config_dict: out_lines.append(f"{k}={config_dict[k]}\n")

        out_lines.append("\n# --- CROP & REGISTRATION PARAMETERS ---\n")
        for k in ['roi_file', 'roi_channel', 'bead_path', 'manual_registration', 'manual_pairs', 'tmats_path']:
            if k in config_dict: out_lines.append(f"{k}={config_dict[k]}\n")

        out_lines.append("\n# --- SEGMENTATION & MODELING ---\n")
        for k in ['model', 'model_type', 'model_params', 'mask_type', 'mask_prefix', 'use_otsu', 'inv_bf']:
            if k in config_dict: out_lines.append(f"{k}={config_dict[k]}\n")

        out_lines.append("\n# --- CELL FILTERING & FITTING ---\n")
        for k in ['area_filter', 'eccentricity_filter', 'cell_fit_err', 'cell_fitting']:
            if k in config_dict: out_lines.append(f"{k}={config_dict[k]}\n")

        written = {'pixel_size', 'alex', 'num_frames', 'frame_time', 'video_path', 'bf_path', 'channel', 
                   'start_channel', 'num_channels', 'frame_avg', 'save_dir', 'overwrite', 'roi_file', 
                   'roi_channel', 'bead_path', 'manual_registration', 'manual_pairs', 'tmats_path', 
                   'model', 'model_type', 'model_params', 'mask_type', 'mask_prefix', 'use_otsu', 
                   'inv_bf', 'area_filter', 'eccentricity_filter', 'cell_fit_err', 'cell_fitting'}
        remaining = [k for k in config_dict if k not in written and config_dict[k] is not None]
        if remaining:
            out_lines.append("\n# --- OTHER PARAMETERS ---\n")
            for k in remaining:
                out_lines.append(f"{k}={config_dict[k]}\n")

    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(out_lines)