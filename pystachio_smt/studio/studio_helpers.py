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
