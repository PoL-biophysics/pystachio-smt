#!/bin/env python3
import sys
import os
import glob
import csv
import subprocess
import traceback
import numpy as np
import tifffile as tf
import cv2
import matplotlib
matplotlib.use('QtAgg')

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QSlider, QLineEdit,
                             QLabel, QRadioButton, QGroupBox, QFileDialog, 
                             QButtonGroup, QFormLayout, QMessageBox, QCheckBox,
                             QTabWidget, QGridLayout, QToolTip, QComboBox)
from PyQt6.QtCore import Qt

from pystackreg import StackReg

# --- External Custom Modules ---
try:
    import images
    import spots
    import parameters
    import postprocessing
    import trajectories
    import tracking
    from studio.studio_helpers import *
    from studio.studio_tracking_tab import *
    from studio.studio_smFRET_tab import *
    from studio.studio_clickmode_tab import *
    from studio.studio_astigmatism_tab import *
    from studio.studio_results_viewer_tab import *
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


# =========================================================================
# APPLICATION SUITE MAIN ENTRY CONTAINER FRAMEWORK
# =========================================================================
class MainApplicationSuite(QMainWindow):
    def __init__(self, working_dir=".", initial_file=None):
        super().__init__()
        self.setWindowTitle("PySTACHIO Analysis Studio")
        self.setGeometry(100, 100, 1300, 850)
        
        self.working_dir = os.path.abspath(working_dir) if working_dir else os.getcwd()
        self.initial_file = os.path.abspath(initial_file) if initial_file else None
        
        self.tabs_container = QTabWidget()
        
        self.main_analysis_tab = TiffAnalyzerTab(parent_suite=self)
        self.smfret_dedicated_tab = SmFretTab(parent_suite=self)
        self.click_mode_tab = ClickModeTab(parent_suite=self)
        self.astigmatism_3d_tab = Astigmatism3DTab(parent=self)
        self.results_viewer_tab = ResultsViewerTab(parent=self, root_dir=self.working_dir)
        
        self.tabs_container.addTab(self.main_analysis_tab, "Automated Spot Tracking Pipeline")
        self.tabs_container.addTab(self.smfret_dedicated_tab, "Interactive Single-Molecule FRET Analysis")
        self.tabs_container.addTab(self.click_mode_tab, "Single-Channel Click Mode")
        self.tabs_container.addTab(self.astigmatism_3d_tab, "Astigmatism viewer")
        self.tabs_container.addTab(self.results_viewer_tab, "Results Viewer")
        
        self.setCentralWidget(self.tabs_container)

        # Auto-load file into all applicable tabs if provided
        if self.initial_file and os.path.exists(self.initial_file):
            self.load_initial_file_across_tabs(self.initial_file)

    def load_initial_file_across_tabs(self, filepath):
        """Attempts to load the command-line provided file into each tab."""
        tabs_to_load = [
            ("Automated Spot Tracking", self.main_analysis_tab),
            ("smFRET Analysis", self.smfret_dedicated_tab),
            ("Click Mode", self.click_mode_tab),
            ("Astigmatism Viewer", self.astigmatism_3d_tab)
        ]

        load_methods = ['load_file', 'open_file', 'load_stack', 'load_image', 'load_data', 'read_image', 'select_file']

        for tab_name, tab_widget in tabs_to_load:
            loaded = False
            for method in load_methods:
                if hasattr(tab_widget, method) and callable(getattr(tab_widget, method)):
                    try:
                        getattr(tab_widget, method)(filepath)
                        loaded = True
                        break
                    except Exception as e:
                        print(f"Info: Could not load '{filepath}' into {tab_name} via {method}(): {e}")
            if not loaded:
                # If tab stores filename as attribute directly
                if hasattr(tab_widget, 'source_fname'):
                    tab_widget.source_fname = filepath

    def reset_entire_suite(self):
        t_tab = self.main_analysis_tab
        t_tab.raw_stack = t_tab.source_fname = t_tab.crop_coords = None
        t_tab.current_frame = 0
        for sel in t_tab.selectors:
            sel.set_active(False)
            if hasattr(sel, 'disconnect_events'): sel.disconnect_events()
        t_tab.selectors.clear(); t_tab.fig.clear()
        t_tab.ax_single = t_tab.ax_left = t_tab.ax_right = t_tab.ax_donor = t_tab.ax_acceptor = t_tab.ax_fret = None
        t_tab.im_single = t_tab.im_left = t_tab.im_right = t_tab.im_donor = t_tab.im_acceptor = t_tab.im_fret = None
        t_tab.cb_single = t_tab.cb_left = t_tab.cb_right = t_tab.cb_donor = t_tab.cb_acceptor = t_tab.cb_fret = None
        t_tab.frame_slider.blockSignals(True); t_tab.frame_slider.setValue(0); t_tab.frame_slider.setMaximum(0); t_tab.frame_slider.setEnabled(False); t_tab.frame_slider.blockSignals(False)
        t_tab.frame_label.setText("0 / 0"); t_tab.canvas.draw_idle()

        s_tab = self.smfret_dedicated_tab
        s_tab.coords, s_tab.trajs_donor, s_tab.trajs_acceptor, s_tab.trajs_fret = [], [], [], []
        s_tab.donor_data = s_tab.acceptor_data = s_tab.fret_data = s_tab.source_fname = None
        s_tab.active_frame = 0
        s_tab.build_mosaic()
        s_tab.frame_slider.blockSignals(True); s_tab.frame_slider.setValue(0); s_tab.frame_slider.setMaximum(0); s_tab.frame_slider.setEnabled(False); s_tab.frame_slider.blockSignals(False)
        s_tab.frame_label.setText("0 / 0")

        c_tab = self.click_mode_tab
        c_tab.img_ch1 = c_tab.img_ch2 = None
        c_tab.active_frame = 0
        c_tab.clear_plot()
        c_tab.frame_slider.blockSignals(True); c_tab.frame_slider.setValue(0); c_tab.frame_slider.setMaximum(0); c_tab.frame_slider.setEnabled(False); c_tab.frame_slider.blockSignals(False)

        a_tab = self.astigmatism_3d_tab
        a_tab.data_store.clear()
        a_tab.plotted_tracks.clear()
        a_tab.ax_3d.clear()
        a_tab.canvas.draw_idle()
        
    def clear_registration(self):
        self.registration_matrix = self.main_analysis_tab.registration_matrix = self.smfret_dedicated_tab.registration_matrix = None
        show_popup(self, "Registration Cleared", "The calibration registration matrix has been deleted.")
        if self.main_analysis_tab.raw_stack is not None: self.main_analysis_tab.update_display()
        if self.smfret_dedicated_tab.donor_data is not None: self.smfret_dedicated_tab.refresh()

    def run_registration(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open Calibration Bead/Grid Multi-frame Stack", "", "TIFF Files (*.tif *.tiff)")
        if not fname: return
            
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            stack = tf.imread(fname)
        except:
            QApplication.restoreOverrideCursor()
            show_popup(self, "Registration Failure", f"Failed to read file {fname}", critical=True)

        img = np.mean(stack, axis=0)
        width = img.shape[1]
        i1 = img[:,:width//2]
        i2 = img[:,width//2:]
    
        # Rescale the images (optional)
        m1 = np.amax(i1)
        m2 = np.amin(i1)
        i1 = 1024 * (i1 - m2) / (m1 - m2)
        m1 = np.amax(i2)
        m2 = np.amin(i2)
        i2 = 1024 * (i2 - m2) / (m1 - m2)

        # Normalize i2 to i1 (optional)
        m = np.amax(i2)
        n = np.amax(i1)
        i2 = n * i2 / m
        i1 = 255 * i1/np.amax(i1)
        i2 = 255 * i2/np.amax(i2)
    
        sr = StackReg(StackReg.AFFINE)
        rot = sr.register(i1, i2)
    
        # Get the transformation matrix
        warp_matrix = sr.get_matrix()
        if not np.all(np.isfinite(warp_matrix)) or np.max(np.abs(warp_matrix)) > 10000:
            print("Warning: ECC optimization diverged. Falling back to SIFT/Phase Correlation.")
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            if 'shift' in locals():
                warp_matrix[0, 2], warp_matrix[1, 2] = shift[0], shift[1]

        self.registration_matrix = self.main_analysis_tab.registration_matrix = self.smfret_dedicated_tab.registration_matrix = np.vstack([warp_matrix, [0, 0, 1]])
        root_path, _ = os.path.splitext(fname)
        np.savetxt(f"{root_path}_registration_matrix.txt", self.registration_matrix)
            
        QApplication.restoreOverrideCursor()
        show_popup(self, "Registration Complete", f"Successfully generated Alignment Affine Transformation Matrix for Dual Channels.\n\nMatrix saved to:\n{root_path}_registration_matrix.txt")


if __name__ == "__main__":
    working_dir = "."
    initial_file = None

    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if args:
        target = os.path.abspath(args[0])
        if os.path.isdir(target):
            working_dir = target
            os.chdir(working_dir)
        elif os.path.isfile(target):
            initial_file = target
            working_dir = os.path.dirname(target)
            os.chdir(working_dir)

    app = QApplication(sys.argv)
    window = MainApplicationSuite(working_dir=working_dir, initial_file=initial_file)
    window.show()
    sys.exit(app.exec())