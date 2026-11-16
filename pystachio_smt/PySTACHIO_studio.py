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
                             QTabWidget, QGridLayout, QToolTip, QComboBox,
                             QScrollArea, QSizePolicy, QLayout, QFrame,
                             QDialog, QDialogButtonBox, QPlainTextEdit, QInputDialog)
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
    from studio.studio_preprocessing_tab import PreprocessingTab
    from studio.studio_fov_tab import FieldsOfViewTab
    
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

class StudioMessageDialog(QDialog):
    """A screen-sized, resizable message viewer; long errors remain accessible."""
    def __init__(self, parent, title, text, critical=False):
        super().__init__(parent)
        self.setWindowTitle(str(title))
        self.setSizeGripEnabled(True)
        layout = QVBoxLayout(self)
        if critical:
            layout.addWidget(QLabel('Error'))
        self.message_text = QPlainTextEdit(self)
        self.message_text.setReadOnly(True)
        # Plain text preserves tracebacks, paths and literal <...> characters.
        self.message_text.setPlainText(str(text))
        layout.addWidget(self.message_text, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok, parent=self)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
        available = self.screen().availableGeometry()
        self.resize(min(720, int(available.width() * .85)),
                    min(460, int(available.height() * .8)))


def show_popup(parent, title, text, critical=False):
    return StudioMessageDialog(parent, title, text, critical).exec()


def install_studio_popup_handler():
    # Tabs import this helper by value. Replace those module bindings too;
    # leave QMessageBox questions/confirmation return values untouched.
    for name, module in tuple(sys.modules.items()):
        if name.startswith('studio.') and hasattr(module, 'show_popup'):
            module.show_popup = show_popup


def _scroll_container(content):
    scroll = QScrollArea()
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    scroll.setWidget(content)
    return scroll


def prepare_studio_page(tab):
    """Isolate tab size hints and keep controls accessible on small screens."""
    tab.setMinimumSize(0, 0)
    tab.setMaximumSize(16777215, 16777215)
    tab.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    layout = tab.layout()
    if layout is not None:
        layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)

    # Older tabs put the sidebar directly into a horizontal layout. Wrap
    # control columns, but never the plot column or an existing scroll area.
    if isinstance(layout, QHBoxLayout):
        for index in range(layout.count()):
            item = layout.itemAt(index)
            child_layout, content = item.layout(), item.widget()
            if isinstance(content, QScrollArea):
                continue
            if child_layout is not None:
                def contains_canvas(candidate):
                    for i in range(candidate.count()):
                        entry = candidate.itemAt(i)
                        widget = entry.widget()
                        if widget is not None and (isinstance(widget, FigureCanvas)
                                or widget.findChildren(FigureCanvas)):
                            return True
                        if entry.layout() is not None and contains_canvas(entry.layout()):
                            return True
                    return False
                if contains_canvas(child_layout):
                    continue
                stretch = layout.stretch(index)
                layout.takeAt(index)
                content = QWidget()
                content.setLayout(child_layout)
            elif content is not None and content.layout() is not None:
                if isinstance(content, FigureCanvas) or content.findChildren(FigureCanvas):
                    continue
                stretch = layout.stretch(index)
                layout.takeAt(index)
            else:
                continue
            content.setMinimumSize(0, 0)
            content.setMaximumSize(16777215, 16777215)
            content.layout().setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
            layout.insertWidget(index, _scroll_container(content), stretch)

    # A scroll area can only reach its last control if the content reports
    # its layout's minimum height instead of a fixed/clipped height.
    for scroll in tab.findChildren(QScrollArea):
        content = scroll.widget()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        if content is not None and content.layout() is not None:
            content.setMinimumSize(0, 0)
            content.setMaximumSize(16777215, 16777215)
            content.layout().setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
            content.layout().invalidate()
    # Fallback for tab designs not available here: oversized content scrolls
    # within its page rather than forcing the entire main window off-screen.
    return _scroll_container(tab)


def read_channel_registration(filename, direction):
    """Read a column-vector affine map in channel-local pixel coordinates.

    Always return physical LEFT -> RIGHT, independent of dye orientation.
    """
    if direction not in ('left-to-right', 'right-to-left'):
        raise ValueError('Select left-to-right or right-to-left mapping.')
    path = Path(filename)
    if path.suffix.lower() == '.npy':
        matrix = np.load(path, allow_pickle=False)
    else:
        matrix = np.loadtxt(path, delimiter=',' if path.suffix.lower() == '.csv' else None)
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape == (4, 3) and np.allclose(matrix[2:], [[0, 0, 1], [0, 0, 1]]):
        matrix = matrix[:3]
    if matrix.shape == (2, 3):
        matrix = np.vstack([matrix, [0, 0, 1]])
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[2], [0, 0, 1])
            or abs(np.linalg.det(matrix[:2, :2])) < 1e-12
            or np.linalg.cond(matrix[:2, :2]) > 1e6):
        raise ValueError('Registration must be a finite, invertible 2x3 or 3x3 affine matrix.')
    matrix = matrix.copy()
    matrix[2] = [0, 0, 1]
    if direction == 'right-to-left':
        matrix = np.linalg.inv(matrix)
    return matrix


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
    def __init__(self, working_dir=".", initial_file=None):
        super().__init__()
        install_studio_popup_handler()
        self.setWindowTitle("PySTACHIO Analysis Studio")
        available = self.screen().availableGeometry()
        self.resize(min(1300, int(available.width() * .9)),
                    min(850, int(available.height() * .9)))
        
        self.working_dir = os.path.abspath(working_dir) if working_dir else os.getcwd()
        self.initial_file = os.path.abspath(initial_file) if initial_file else None
        
        self.tabs_container = QTabWidget()
        self.tabs_container.setUsesScrollButtons(True)
        
        self.preprocessing_tab = PreprocessingTab(parent=self, root_dir=self.working_dir)  # <--- INSTANTIATION
        self.main_analysis_tab = TiffAnalyzerTab(parent_suite=self)
        self.smfret_dedicated_tab = SmFretTab(parent_suite=self)
        self.click_mode_tab = ClickModeTab(parent_suite=self)
        self.astigmatism_3d_tab = Astigmatism3DTab(parent=self)
        if FieldsOfViewTab is not None:
            self.fov_tab = FieldsOfViewTab(parent=self, root_dir=self.working_dir)
        else:
            self.fov_tab = QWidget()
        self.results_viewer_tab = ResultsViewerTab(parent=self, root_dir=self.working_dir)
        
        for tab, title in (
                (self.preprocessing_tab, "Image Preprocessing"),
                (self.main_analysis_tab, "Automated Spot Tracking Pipeline"),
                (self.smfret_dedicated_tab, "Interactive Single-Molecule FRET Analysis"),
                (self.click_mode_tab, "Single-Channel Click Mode"),
                (self.astigmatism_3d_tab, "Astigmatism viewer"),
                (self.fov_tab, "Fields of View"),
                (self.results_viewer_tab, "Results Viewer")):
            self.tabs_container.addTab(prepare_studio_page(tab), title)
        
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

    def load_registration(self):
        if self._registration_busy():
            show_popup(self, 'Analysis running', 'Wait for analysis to finish before changing registration.')
            return
        filename, _ = QFileDialog.getOpenFileName(
            self, 'Load Registration Matrix', '',
            'Registration matrices (*.txt *.dat *.csv *.npy);;All files (*)',
        )
        if not filename:
            return
        directions = ['Left to right', 'Right to left']
        selected, accepted = QInputDialog.getItem(
            self, 'Registration direction',
            'Direction of the matrix in this file:\n'
            'Coordinates use x/y pixels within each channel, not the full split image.\n'
            'Matrix convention: destination = M @ [x, y, 1].',
            directions, 0, False,
        )
        if not accepted:
            return
        try:
            matrix = read_channel_registration(
                filename, 'left-to-right' if selected == directions[0] else 'right-to-left'
            )
        except Exception as exc:
            show_popup(self, 'Registration Load Failure',
                       f'{exc}\nPrevious calibration and traces were retained.', critical=True)
            return
        self._publish_registration(matrix)
        show_popup(self, 'Registration Loaded',
                   'Calibration loaded for coordinate mapping. Images are unchanged.\n'
                   'Previous smFRET traces cleared; re-extract spots using this calibration.')


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
