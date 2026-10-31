#!/bin/env python3
import sys
import os
import glob
import csv
import subprocess
import traceback
import tempfile
from pathlib import Path
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

def calculate_channel_registration(stack):
    """Fit physical left -> right coordinates (x, y, 1), in half-image pixels.

    StackReg uses a sampling map: register(left, right) returns the positions
    in right at which to sample each left coordinate. Store that 3x3 map once.
    """
    stack = np.asarray(stack, dtype=float)
    if stack.ndim == 2:
        image = stack
    elif stack.ndim == 3 and stack.shape[0] > 0:
        image = np.mean(stack, axis=0)
    else:
        raise ValueError('Calibration must be a 2-D image or a 3-D grayscale stack.')
    if min(image.shape) < 8 or image.shape[1] % 2 or image.shape[1] // 2 < 8:
        raise ValueError('Calibration needs two equal-width channel halves, at least 8x8 each.')
    if not np.isfinite(image).all():
        raise ValueError('Calibration contains NaN or infinite values.')
    left, right = np.hsplit(image, 2)
    def normalize(image):
        span = np.ptp(image)
        if not np.isfinite(span) or span <= 0:
            raise ValueError('Both calibration channels must contain nonconstant signal.')
        return np.ascontiguousarray((image - image.min()) / span, dtype=np.float64)
    left, right = normalize(left), normalize(right)
    sr = StackReg(StackReg.AFFINE)
    matrix = np.asarray(sr.register(left, right), dtype=float)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[2], [0, 0, 1])
            or abs(np.linalg.det(matrix[:2, :2])) < 1e-12
            or np.linalg.cond(matrix[:2, :2]) > 1e6):
        raise ValueError('Registration did not produce a valid invertible affine transform.')
    aligned = sr.transform(right)
    # Assess correlation only on pixels whose interpolation footprint is inside
    # the moving channel. This is a screening check, not localization accuracy.
    yy, xx = np.indices(left.shape)
    sample_x = matrix[0,0]*xx + matrix[0,1]*yy + matrix[0,2]
    sample_y = matrix[1,0]*xx + matrix[1,1]*yy + matrix[1,2]
    mask = ((sample_x >= 2) & (sample_y >= 2) &
            (sample_x < right.shape[1]-3) & (sample_y < right.shape[0]-3))
    overlap = float(mask.mean())
    if overlap < 0.25 or not np.isfinite(aligned[mask]).all():
        raise ValueError('Registration leaves less than 25% usable overlap or invalid warped pixels.')
    def correlation(a, b):
        if np.std(a) == 0 or np.std(b) == 0:
            return float('nan')
        return float(np.corrcoef(a, b)[0,1])
    before = correlation(left[mask], right[mask])
    after = correlation(left[mask], aligned[mask])
    if not np.isfinite(after) or after < 0.5 or (np.isfinite(before) and after < before-0.02):
        raise ValueError(f'Poor registration match (correlation before={before:.3f}, after={after:.3f}). '
                         'Use a calibration with corresponding beads/features in both halves.')
    return matrix.copy(), {'before': before, 'after': after, 'overlap': overlap}


class MainApplicationSuite(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PySTACHIO Analysis Studio")
        self.setGeometry(100, 100, 1300, 850)
        
        self.tabs_container = QTabWidget()
        
        self.main_analysis_tab = TiffAnalyzerTab(parent_suite=self)
        self.smfret_dedicated_tab = SmFretTab(parent_suite=self)
        self.click_mode_tab = ClickModeTab(parent_suite=self)
        self.results_viewer_tab = ResultsViewerTab(parent=self)
        self.astigmatism_3d_tab = Astigmatism3DTab(parent=self)
        
        self.tabs_container.addTab(self.main_analysis_tab, "Automated Spot Tracking Pipeline")
        self.tabs_container.addTab(self.smfret_dedicated_tab, "Interactive Single-Molecule FRET Analysis")
        self.tabs_container.addTab(self.click_mode_tab, "Single-Channel Click Mode")
        self.tabs_container.addTab(self.astigmatism_3d_tab, "Astigmatism viewer")
        self.tabs_container.addTab(self.results_viewer_tab, "Results Viewer")
        
        self.setCentralWidget(self.tabs_container)

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
        s_tab._selection_backup = None
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
        
    def _publish_registration(self, matrix):
        # Physical left -> right convention is independent of the chosen dye orientation.
        self.registration_matrix = None if matrix is None else matrix.copy()
        self.main_analysis_tab.registration_matrix = None if matrix is None else matrix.copy()
        self.smfret_dedicated_tab.registration_matrix = None if matrix is None else matrix.copy()
        # Old traces were measured at old coordinates and must not be reused.
        self.smfret_dedicated_tab.clear_plot()
        if self.main_analysis_tab.raw_stack is not None:
            self.main_analysis_tab.update_display()

    def _registration_busy(self):
        return any(worker is not None and worker.isRunning()
                   for worker in (self.smfret_dedicated_tab.worker,
                                  self.smfret_dedicated_tab.batch_worker,
                                  getattr(self.main_analysis_tab, '_worker', None)))

    def clear_registration(self):
        if self._registration_busy():
            show_popup(self, 'Analysis running', 'Wait for analysis to finish before changing registration.')
            return
        self._publish_registration(None)
        show_popup(self, 'Registration Cleared', 'Registration and previously extracted smFRET traces cleared. Re-extract spots.')

    def run_registration(self):
        if self._registration_busy():
            show_popup(self, 'Analysis running', 'Wait for analysis to finish before changing registration.')
            return
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Calibration Bead/Grid Image or Stack', '',
                                              'TIFF Files (*.tif *.tiff *.TIF *.TIFF)')
        if not fname:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            matrix, quality = calculate_channel_registration(tf.imread(fname))
            destination = Path(fname).with_suffix('')
            destination = destination.parent / (destination.name + '_registration_matrix.txt')
            # Commit only after fitting, validation and saving all succeed.
            with tempfile.TemporaryDirectory(dir=destination.parent, prefix='.registration_') as temp:
                staged = Path(temp) / destination.name
                np.savetxt(staged, matrix, header=(
                    'Physical LEFT -> RIGHT coordinates; column vectors [x, y, 1]; channel-local pixels.\n'
                    f'Correlation before={quality["before"]:.6g}, after={quality["after"]:.6g}; '
                    f'usable overlap={quality["overlap"]:.6g}'))
                staged.replace(destination)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            show_popup(self, 'Registration Failure',
                       f'{exc}\nPrevious calibration and traces were retained.', critical=True)
            return
        QApplication.restoreOverrideCursor()
        self._publish_registration(matrix)
        show_popup(self, 'Registration Complete',
                   f'Left-to-right affine calibration saved to:\n{destination}\n\n'
                   f'Image correlation: {quality["before"]:.3f} -> {quality["after"]:.3f}\n'
                   'Previous smFRET traces cleared; re-extract spots using this calibration.')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApplicationSuite()
    window.show()
    sys.exit(app.exec())
