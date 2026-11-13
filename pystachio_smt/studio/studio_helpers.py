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

def save_config_file(config_dict: dict, file_path: str, raw_lines: list = None, original_dict: dict = None, task_directive: str = "preprocess"):
    """
    Saves configuration parameters to a PySTACHIO config file.
    Ensures 'task_directive' (e.g., 'preprocess' or 'track,postprocess') is placed on Line 1.
    """
    written_keys = set()
    out_lines = [f"{task_directive}\n"]  # Dynamic task directive on Line 1

    known_tasks = {"preprocess", "track", "postprocess", "fovs", "track,postprocess", "postprocess,track"}

    if raw_lines:
        for line in raw_lines:
            raw_line_stripped = line.strip()

            # Skip existing standalone task directive lines in body
            if raw_line_stripped.lower() in known_tasks or (raw_line_stripped and not raw_line_stripped.startswith('#') and '=' not in raw_line_stripped):
                continue

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
                        out_lines.append(f"{key_str.strip()}={new_val}{comment_suffix}\n")
                else:
                    out_lines.append(line.rstrip('\r\n') + '\n')
            else:
                out_lines.append(line.rstrip('\r\n') + '\n')

        # Append any newly added parameters
        unwritten = [k for k in config_dict if k not in written_keys and config_dict[k] is not None]
        if unwritten:
            out_lines.append("\n# --- NEW ADDED PARAMETERS ---\n")
            for k in unwritten:
                out_lines.append(f"{k}={config_dict[k]}\n")

    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(out_lines)