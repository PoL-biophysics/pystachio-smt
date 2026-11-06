import sys
import os
import time
import glob
import re
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from PIL import Image
from scipy.stats import gaussian_kde, gamma, rayleigh

try:
    from pystackreg import StackReg
except ImportError:
    StackReg = None

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QPushButton, QFileDialog, QLabel, QComboBox,
    QSplitter, QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox,
    QCheckBox, QMessageBox, QDialog, QScrollArea, QProgressDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

import parameters
import fitting


# =========================================================================
# BACKGROUND WORKER THREAD FOR RESPONSIVE MODEL FITTING
# =========================================================================

class FitWorkerThread(QThread):
    """Runs jump distance or diffusion MLE fitting in a background thread with cancellation."""
    progress_signal = pyqtSignal(int, int, str)
    finished_signal = pyqtSignal(list)
    error_signal = pyqtSignal(str)

    def __init__(self, fit_type, data, max_comp, dt=None, loc_error=0.0, model_type=None, parent=None):
        super().__init__(parent)
        self.fit_type = fit_type  # 'diffusion' or 'jump'
        self.data = data
        self.max_comp = max_comp
        self.dt = dt
        self.loc_error = loc_error
        self.model_type = model_type
        self._is_canceled = False

    def cancel(self):
        self._is_canceled = True

    def is_canceled(self):
        return self._is_canceled

    def run(self):
        try:
            results = []
            for k in range(1, self.max_comp + 1):
                if self._is_canceled:
                    break
                self.progress_signal.emit(
                    k - 1, 
                    self.max_comp, 
                    f"Fitting component {k} of {self.max_comp}..."
                )
                
                if self.fit_type == 'diffusion':
                    res = fitting.fit_gamma_diffusion_mle(
                        self.data, 
                        num_components=k, 
                        cancel_check=self.is_canceled
                    )
                elif self.fit_type == 'jump':
                    res = fitting.fit_jump_distances_mle(
                        self.data, 
                        dt=self.dt, 
                        num_components=k, 
                        loc_error=self.loc_error, 
                        model_type=self.model_type,
                        cancel_check=self.is_canceled
                    )
                else:
                    res = None

                if self._is_canceled:
                    break

                if res:
                    results.append(res)

            if not self._is_canceled:
                self.progress_signal.emit(self.max_comp, self.max_comp, "Fitting completed.")
                self.finished_signal.emit(results)
        except InterruptedError:
            pass
        except Exception as e:
            self.error_signal.emit(str(e))


# =========================================================================
# CUSTOM INPUT CONTROLS PREVENTING ACCIDENTAL SCROLLING
# =========================================================================

class NoScrollSpinBox(QSpinBox):
    """QSpinBox that ignores mouse wheel scrolling unless explicitly focused by clicking."""
    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class NoScrollDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox that ignores mouse wheel scrolling unless explicitly focused by clicking."""
    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class NoScrollComboBox(QComboBox):
    """QComboBox that ignores mouse wheel scrolling unless explicitly focused by clicking."""
    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class ConfigSelectionDialog(QDialog):
    """Pop-up dialog asking user to select a config file when conflicts exist."""
    def __init__(self, config_file_map, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Conflicting Configuration Files Found")
        self.setMinimumWidth(580)

        self.config_file_map = config_file_map

        layout = QVBoxLayout(self)

        lbl = QLabel(
            "<b>Multiple configuration files were found with conflicting parameters.</b><br>"
            "Please select which configuration file to apply:"
        )
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

        self.combo = NoScrollComboBox()
        self.combo.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        for filepath, cfg in config_file_map.items():
            rel_path = os.path.relpath(filepath)
            details = []
            if 'pixel_size' in cfg:
                details.append(f"Pixel: {cfg['pixel_size']} μm")
            if 'frame_time' in cfg:
                details.append(f"Frame: {cfg['frame_time']} ms")
            if 'alex' in cfg:
                details.append(f"ALEX: {cfg['alex']}")
            if 'localisation_precision' in cfg:
                details.append(f"Loc Prec: {cfg['localisation_precision']} μm")
            if 'channel' in cfg:
                details.append(f"Channel: {cfg['channel']}")

            summary = ", ".join(details) if details else "No recognized parameters"
            label = f"{rel_path}  ({summary})"
            self.combo.addItem(label, userData=filepath)

        layout.addWidget(self.combo)

        btn_box = QHBoxLayout()
        btn_ok = QPushButton("Apply Selected Config")
        btn_ok.clicked.connect(self.accept)
        btn_box.addStretch()
        btn_box.addWidget(btn_ok)
        layout.addLayout(btn_box)

    def get_selected_filepath(self):
        return self.combo.currentData()


class MplCanvas(FigureCanvasQTAgg):
    """Reusable Matplotlib Canvas widget."""
    def __init__(self, parent=None, width=6, height=5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)


class ResultsViewerTab(QWidget):
    def __init__(self, parent=None, root_dir="."):
        super().__init__(parent)
        self.parent = parent

        self.root_dir = os.path.abspath(root_dir) if root_dir else ""
        self.loc_precision = None  # μm
        self.config_channel = "Left Channel (L)"

        # Data containers
        self.trajectories_df = pd.DataFrame()
        self.diffusion_summary_df = pd.DataFrame()
        self.field_dirs_map = {}  # field_name -> field_directory_path
        self.field_isingle_data = []  # Scanned field i_single values

        # Fitting results cache
        self.diff_fit_results = []   # List of fit dicts for diffusion models 1..N
        self.best_diff_model = None
        self.jump_fit_results = []   # List of fit dicts for jump distance models 1..N
        self.best_jump_model = None

        self.worker_thread = None
        self.progress_dialog = None

        self.init_ui()
        if self.root_dir and os.path.exists(self.root_dir):
            self.scan_and_load_directory()

    def get_parameter_default(self, key: str, fallback):
        """Safely retrieves default attribute values directly from the parameters module."""
        if parameters is not None and hasattr(parameters, 'Parameters'):
            try:
                p = parameters.Parameters()
                val = getattr(p, key, fallback)
                return val if val is not None else fallback
            except Exception:
                pass
        return fallback

    def init_ui(self):
        default_pixel = self.get_parameter_default('pixel_size', 0.120)       # μm/px
        raw_frame_time = self.get_parameter_default('frame_time', 0.005)      # seconds
        default_exposure = raw_frame_time * 1000.0 if raw_frame_time < 1.0 else raw_frame_time  # convert to ms
        default_alex = self.get_parameter_default('ALEX', False)
        default_isingle = self.get_parameter_default('I_single', 10000.0)
        default_isingle_frac = self.get_parameter_default('isingle_fraction', 66.0)

        main_layout = QHBoxLayout(self)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # ================= LEFT SIDEBAR (SCROLLABLE) =================
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(8, 8, 8, 8)
        sidebar_layout.setSpacing(10)

        # 1. Directory Info Group
        env_group = QGroupBox("Directory Info")
        env_layout = QFormLayout()
        env_layout.setVerticalSpacing(6)
        self.lbl_root_dir = QLabel(self.root_dir or "No directory selected")
        self.lbl_root_dir.setWordWrap(True)
        self.lbl_root_dir.setStyleSheet("font-size: 10px; color: #444;")

        self.btn_rescan = QPushButton("Rescan Directory")
        self.btn_rescan.clicked.connect(self.scan_and_load_directory)
        self.btn_change_dir = QPushButton("Open Directory...")
        self.btn_change_dir.clicked.connect(self.select_new_directory)

        env_layout.addRow("Target:", self.lbl_root_dir)
        env_layout.addRow(self.btn_rescan)
        env_layout.addRow(self.btn_change_dir)
        env_group.setLayout(env_layout)
        sidebar_layout.addWidget(env_group)

        # 2. Dataset & Channel Filter
        filter_group = QGroupBox("Sample & Channel Selection")
        filter_layout = QFormLayout()
        filter_layout.setVerticalSpacing(6)

        self.sample_selector = NoScrollComboBox()
        self.sample_selector.currentIndexChanged.connect(self.on_sample_changed)

        self.lbl_channel = QLabel("Channel View:")
        self.channel_selector = NoScrollComboBox()
        self.channel_selector.addItems(["Both Channels (Overlay)", "Left Channel (L)", "Right Channel (R)"])
        self.channel_selector.currentIndexChanged.connect(self.on_channel_changed)

        # Hide channel selector by default until multi-channel data is detected
        self.lbl_channel.setVisible(False)
        self.channel_selector.setVisible(False)

        filter_layout.addRow("Sample / Condition:", self.sample_selector)
        filter_layout.addRow(self.lbl_channel, self.channel_selector)
        filter_group.setLayout(filter_layout)
        sidebar_layout.addWidget(filter_group)

        # 3. Field Selection Dropdown Control
        field_group = QGroupBox("Field Selection")
        field_layout = QFormLayout()
        field_layout.setVerticalSpacing(6)

        self.field_selector = NoScrollComboBox()
        self.field_selector.currentIndexChanged.connect(self.on_field_changed)

        field_layout.addRow("Field Name:", self.field_selector)
        field_group.setLayout(field_layout)
        sidebar_layout.addWidget(field_group)

        # 4. Acquisition Settings
        acq_group = QGroupBox("Acquisition Settings")
        acq_layout = QFormLayout()
        acq_layout.setVerticalSpacing(6)

        self.spin_pixel_size = NoScrollDoubleSpinBox()
        self.spin_pixel_size.setDecimals(4)
        self.spin_pixel_size.setRange(0.0001, 10.0)
        self.spin_pixel_size.setValue(default_pixel)
        self.spin_pixel_size.setSingleStep(0.005)
        self.spin_pixel_size.setSuffix(" μm/px")

        self.spin_exposure = NoScrollDoubleSpinBox()
        self.spin_exposure.setDecimals(3)
        self.spin_exposure.setRange(0.001, 10000.0)
        self.spin_exposure.setValue(default_exposure)
        self.spin_exposure.setSuffix(" ms")

        self.chk_alex = QCheckBox("ALEX Imaging (Alternating Excitation)")
        self.chk_alex.setChecked(default_alex)

        acq_layout.addRow("Pixel Size:", self.spin_pixel_size)
        acq_layout.addRow("Frame Interval:", self.spin_exposure)
        acq_layout.addRow(self.chk_alex)
        acq_group.setLayout(acq_layout)
        sidebar_layout.addWidget(acq_group)

        # 5a. Track filtering (shared by every tab, always visible)
        track_group = QGroupBox("Track Filtering")
        track_layout = QFormLayout()
        track_layout.setVerticalSpacing(6)

        self.spin_min_len = NoScrollSpinBox()
        self.spin_min_len.setRange(2, 100)
        self.spin_min_len.setValue(4)

        self.spin_min_snr = NoScrollDoubleSpinBox()
        self.spin_min_snr.setDecimals(1)
        self.spin_min_snr.setRange(0.0, 100.0)
        self.spin_min_snr.setValue(0.0)  # Default 0.0 to avoid filtering tracks on initial load
        self.spin_min_snr.setSingleStep(0.5)

        self.spin_min_frame = NoScrollSpinBox()
        self.spin_min_frame.setRange(0, 1000000)
        self.spin_min_frame.setValue(0)

        self.spin_max_frame = NoScrollSpinBox()
        self.spin_max_frame.setRange(0, 1000000)
        self.spin_max_frame.setValue(1000000)

        self.chk_remove_neg = QCheckBox("Remove Negative D (D ≤ 0): 0")
        self.chk_remove_neg.setChecked(False)

        # SINGLE BUTTON TO APPLY ALL TRACK FILTERS IN THIS BOX
        self.btn_apply_track_filters = QPushButton("Apply Track Filters")
        self.btn_apply_track_filters.setStyleSheet("font-weight: bold; background-color: #008080; color: white; padding: 6px;")
        self.btn_apply_track_filters.clicked.connect(self.recalculate_and_refresh)

        track_layout.addRow("Min Track Length:", self.spin_min_len)
        track_layout.addRow("Min SNR Threshold:", self.spin_min_snr)
        track_layout.addRow("Min Frame:", self.spin_min_frame)
        track_layout.addRow("Max Frame:", self.spin_max_frame)
        track_layout.addRow(self.chk_remove_neg)
        track_layout.addRow(self.btn_apply_track_filters)
        track_group.setLayout(track_layout)
        sidebar_layout.addWidget(track_group)

        # 5b. iSINGLE TRACK FILTERING BOX (ALWAYS UNCHECKED BY DEFAULT)
        self.isingle_filter_group = QGroupBox("iSingle Track Filtering")
        isingle_layout = QFormLayout()
        isingle_layout.setVerticalSpacing(6)

        self.chk_filter_intensity = QCheckBox("Filter Trajectories by iSingle Range [0.5–1.5×]")
        self.chk_filter_intensity.setChecked(False)  # Ensure explicitly OFF by default

        # Channel L / Mono iSingle SpinBox
        self.lbl_isingle_L = QLabel("iSingle Value:")
        self.spin_isingle_L = NoScrollDoubleSpinBox()
        self.spin_isingle_L.setDecimals(1)
        self.spin_isingle_L.setRange(0.0, 1000000.0)
        self.spin_isingle_L.setValue(default_isingle)
        self.spin_isingle_L.setSingleStep(100.0)
        self.spin_isingle_L.setSuffix(" a.u.")

        # Channel R iSingle SpinBox (Visible when num_channels == 2)
        self.lbl_isingle_R = QLabel("iSingle (R):")
        self.spin_isingle_R = NoScrollDoubleSpinBox()
        self.spin_isingle_R.setDecimals(1)
        self.spin_isingle_R.setRange(0.0, 1000000.0)
        self.spin_isingle_R.setValue(default_isingle)
        self.spin_isingle_R.setSingleStep(100.0)
        self.spin_isingle_R.setSuffix(" a.u.")
        self.lbl_isingle_R.setVisible(False)
        self.spin_isingle_R.setVisible(False)

        isingle_layout.addRow(self.chk_filter_intensity)
        isingle_layout.addRow(self.lbl_isingle_L, self.spin_isingle_L)
        isingle_layout.addRow(self.lbl_isingle_R, self.spin_isingle_R)
        self.isingle_filter_group.setLayout(isingle_layout)
        sidebar_layout.addWidget(self.isingle_filter_group)

        # 5c. Diffusion histogram controls (shown on the Diffusion tab)
        self.diff_hist_group = QGroupBox("Diffusion Histogram Controls")
        diff_layout = QFormLayout()
        diff_layout.setVerticalSpacing(6)

        self.spin_bin_size = NoScrollDoubleSpinBox()
        self.spin_bin_size.setRange(0.0001, 100.0)
        self.spin_bin_size.setDecimals(4)
        self.spin_bin_size.setValue(0.02)
        self.spin_bin_size.setSingleStep(0.005)
        self.spin_bin_size.setSuffix(" μm²/s")

        self.spin_bins = NoScrollSpinBox()
        self.spin_bins.setRange(5, 200)
        self.spin_bins.setValue(35)

        self.chk_log_scale = QCheckBox("Log10 Scale [log10(D)]")
        self.chk_log_scale.setChecked(False)

        self.chk_kde = QCheckBox("Kernel Density Estimate (KDE)")
        self.chk_kde.setChecked(False)

        diff_layout.addRow("Bin Size (D):", self.spin_bin_size)
        diff_layout.addRow("Histogram Bins:", self.spin_bins)
        diff_layout.addRow(self.chk_log_scale)
        diff_layout.addRow(self.chk_kde)
        self.diff_hist_group.setLayout(diff_layout)
        sidebar_layout.addWidget(self.diff_hist_group)

        # 5d. Jump distance histogram controls (shown on the Jump Distance tab)
        self.jump_hist_group = QGroupBox("Jump Distance Histogram Controls")
        jump_layout = QFormLayout()
        jump_layout.setVerticalSpacing(6)

        self.spin_jump_lag = NoScrollSpinBox()
        self.spin_jump_lag.setRange(1, 100)
        self.spin_jump_lag.setValue(1)
        self.spin_jump_lag.setSuffix(" frames")

        self.spin_bins_jump = NoScrollSpinBox()
        self.spin_bins_jump.setRange(5, 200)
        self.spin_bins_jump.setValue(35)

        self.chk_kde_jump = QCheckBox("Kernel Density Estimate (KDE)")
        self.chk_kde_jump.setChecked(False)

        jump_layout.addRow("Jump Lag (N):", self.spin_jump_lag)
        jump_layout.addRow("Histogram Bins:", self.spin_bins_jump)
        jump_layout.addRow(self.chk_kde_jump)
        self.jump_hist_group.setLayout(jump_layout)
        sidebar_layout.addWidget(self.jump_hist_group)
        self.jump_hist_group.setVisible(False)

        # 6a. DIFFUSION MODEL FITTING (shown on the Diffusion tab)
        self.diff_fit_group = QGroupBox("Diffusion Model Fitting (BIC)")
        diff_fit_layout = QFormLayout()
        diff_fit_layout.setVerticalSpacing(6)

        self.spin_max_components = NoScrollSpinBox()
        self.spin_max_components.setRange(1, 4)
        self.spin_max_components.setValue(4)

        self.chk_show_subcomponents = QCheckBox("Show Sub-components (Best Model)")
        self.chk_show_subcomponents.setChecked(True)

        self.chk_show_non_best = QCheckBox("Show Non-Best Model Fits")
        self.chk_show_non_best.setChecked(False)

        self.btn_run_diff_fitting = QPushButton("Fit Diffusion Models & Compare (1-N)")
        self.btn_run_diff_fitting.setStyleSheet("font-weight: bold; background-color: #008080; color: white; padding: 6px;")
        self.btn_run_diff_fitting.clicked.connect(self.run_diffusion_fitting)

        self.btn_export_fits_diff = QPushButton("Export Fit Parameters (.npy)")
        self.btn_export_fits_diff.clicked.connect(self.export_fit_parameters_npy)

        self.lbl_diff_fit_results = QLabel("No fitting run yet.")
        self.lbl_diff_fit_results.setWordWrap(True)
        self.lbl_diff_fit_results.setStyleSheet("font-size: 10px; background-color: #f8f9fa; padding: 6px; border: 1px solid #ccc; border-radius: 4px;")

        diff_fit_layout.addRow("Max Components:", self.spin_max_components)
        diff_fit_layout.addRow(self.chk_show_subcomponents)
        diff_fit_layout.addRow(self.chk_show_non_best)
        diff_fit_layout.addRow(self.btn_run_diff_fitting)
        diff_fit_layout.addRow(self.btn_export_fits_diff)
        diff_fit_layout.addRow(self.lbl_diff_fit_results)
        self.diff_fit_group.setLayout(diff_fit_layout)
        sidebar_layout.addWidget(self.diff_fit_group)

        # 6b. JUMP DISTANCE MODEL FITTING (shown on the Jump Distance tab)
        self.jump_fit_group = QGroupBox("Jump Distance Model Fitting (BIC)")
        jump_fit_layout = QFormLayout()
        jump_fit_layout.setVerticalSpacing(6)

        self.spin_max_components_jump = NoScrollSpinBox()
        self.spin_max_components_jump.setRange(1, 4)
        self.spin_max_components_jump.setValue(4)

        self.lbl_jdd_model = QLabel("JDD Model:")

        self.combo_jdd_model = NoScrollComboBox()
        self.combo_jdd_model.addItems([
            "Rayleigh Distribution (with Loc Precision)",
            "Pure 2D Brownian Motion",
            "Anomalous 2D Diffusion",
            "Mixed (Pure Brownian + Anomalous)"
        ])
        self.combo_jdd_model.currentIndexChanged.connect(self.clear_jump_fits)

        self.chk_show_subcomponents_jump = QCheckBox("Show Sub-components (Best Model)")
        self.chk_show_subcomponents_jump.setChecked(True)

        self.chk_show_non_best_jump = QCheckBox("Show Non-Best Model Fits")
        self.chk_show_non_best_jump.setChecked(False)

        self.btn_run_jump_fitting = QPushButton("Fit Jump Distance Models & Compare (1-N)")
        self.btn_run_jump_fitting.setStyleSheet("font-weight: bold; background-color: #008080; color: white; padding: 6px;")
        self.btn_run_jump_fitting.clicked.connect(self.run_jump_fitting)

        self.btn_export_fits_jump = QPushButton("Export Fit Parameters (.npy)")
        self.btn_export_fits_jump.clicked.connect(self.export_fit_parameters_npy)

        self.lbl_jump_fit_results = QLabel("No fitting run yet.")
        self.lbl_jump_fit_results.setWordWrap(True)
        self.lbl_jump_fit_results.setStyleSheet("font-size: 10px; background-color: #f8f9fa; padding: 6px; border: 1px solid #ccc; border-radius: 4px;")

        jump_fit_layout.addRow("Max Components:", self.spin_max_components_jump)
        jump_fit_layout.addRow(self.lbl_jdd_model, self.combo_jdd_model)
        jump_fit_layout.addRow(self.chk_show_subcomponents_jump)
        jump_fit_layout.addRow(self.chk_show_non_best_jump)
        jump_fit_layout.addRow(self.btn_run_jump_fitting)
        jump_fit_layout.addRow(self.btn_export_fits_jump)
        jump_fit_layout.addRow(self.lbl_jump_fit_results)
        self.jump_fit_group.setLayout(jump_fit_layout)
        sidebar_layout.addWidget(self.jump_fit_group)
        self.jump_fit_group.setVisible(False)

        # 7. SPATIAL MAP OPTIONS PANEL (SIDEBAR TOGGLES & COLORING OPTIONS)
        self.spatial_options_group = QGroupBox("Spatial Map Options")
        spatial_opt_layout = QFormLayout()
        spatial_opt_layout.setVerticalSpacing(6)

        self.chk_spatial_show_L = QCheckBox("Show Left Channel (L) Tracks")
        self.chk_spatial_show_L.setChecked(True)
        self.chk_spatial_show_L.stateChanged.connect(lambda: self.plot_spatial_map(self.get_filtered_summary_df()))

        self.chk_spatial_show_R = QCheckBox("Show Right Channel (R) Tracks")
        self.chk_spatial_show_R.setChecked(True)
        self.chk_spatial_show_R.stateChanged.connect(lambda: self.plot_spatial_map(self.get_filtered_summary_df()))

        self.chk_spatial_transform_R = QCheckBox("Transform R Channel (transformation_matrix.npy)")
        self.chk_spatial_transform_R.setChecked(True)
        self.chk_spatial_transform_R.stateChanged.connect(lambda: self.plot_spatial_map(self.get_filtered_summary_df()))

        self.lbl_spatial_entity = QLabel("Spatial Entity Mapping:")
        self.combo_spatial_entity = NoScrollComboBox()
        self.combo_spatial_entity.addItems([
            "Map Full Trajectories",
            "Map Single Jump Steps"
        ])
        self.combo_spatial_entity.currentIndexChanged.connect(lambda: self.plot_spatial_map(self.get_filtered_summary_df()))

        self.lbl_spatial_color_mode = QLabel("Track Color Mode:")
        self.combo_spatial_color_mode = NoScrollComboBox()
        self.combo_spatial_color_mode.addItems([
            "Channel Color (Default)",
            "Color by Diffusion Coefficient (D)",
            "Color by Jump Distance (N frames)"
        ])
        self.combo_spatial_color_mode.currentIndexChanged.connect(lambda: self.plot_spatial_map(self.get_filtered_summary_df()))

        self.lbl_spatial_jump_lag = QLabel("Spatial Jump Lag (N):")
        self.spin_spatial_jump_lag = NoScrollSpinBox()
        self.spin_spatial_jump_lag.setRange(1, 100)
        self.spin_spatial_jump_lag.setValue(1)
        self.spin_spatial_jump_lag.setSuffix(" frames")
        self.spin_spatial_jump_lag.valueChanged.connect(lambda: self.plot_spatial_map(self.get_filtered_summary_df()))

        spatial_opt_layout.addRow(self.chk_spatial_show_L)
        spatial_opt_layout.addRow(self.chk_spatial_show_R)
        spatial_opt_layout.addRow(self.chk_spatial_transform_R)
        spatial_opt_layout.addRow(self.lbl_spatial_entity, self.combo_spatial_entity)
        spatial_opt_layout.addRow(self.lbl_spatial_color_mode, self.combo_spatial_color_mode)
        spatial_opt_layout.addRow(self.lbl_spatial_jump_lag, self.spin_spatial_jump_lag)
        self.spatial_options_group.setLayout(spatial_opt_layout)
        sidebar_layout.addWidget(self.spatial_options_group)
        self.spatial_options_group.setVisible(False)

        # 8. SINGLE FIELD INSPECTOR OPTIONS PANEL (SIDEBAR TOGGLES)
        self.field_inspector_group = QGroupBox("Field Inspector Options")
        field_insp_layout = QFormLayout()
        field_insp_layout.setVerticalSpacing(6)

        self.chk_overlay_field_tracks = QCheckBox("Overlay Trajectories on Microscopy Images")
        self.chk_overlay_field_tracks.setChecked(True)
        self.chk_overlay_field_tracks.stateChanged.connect(self.update_single_field_plot)

        field_insp_layout.addRow(self.chk_overlay_field_tracks)
        self.field_inspector_group.setLayout(field_insp_layout)
        sidebar_layout.addWidget(self.field_inspector_group)
        self.field_inspector_group.setVisible(False)

        # 9. INTENSITY PLOT DISPLAY SETTINGS (SHOWN ONLY ON INTENSITY TAB)
        self.intensity_plot_group = QGroupBox("Intensity Plot Settings")
        intensity_plot_layout = QFormLayout()
        intensity_plot_layout.setVerticalSpacing(6)

        # Dropdown selection for intensity plot view
        self.combo_intensity_dist = NoScrollComboBox()
        self.combo_intensity_dist.addItems([
            "Spot Intensity Histogram",
            "Trajectory Mean Intensity Histogram",
            "Log10(Spot Intensity) vs Frame Number",
            "Field iSingle Peaks Boxplot",
            "All Intensity Plots (2x2 Grid)"
        ])

        # DESELECTED BY DEFAULT AS REQUESTED
        self.chk_intensity_kde = QCheckBox("Fit KDE & Detect Peak (iSingle)")
        self.chk_intensity_kde.setChecked(False)

        # INDIVIDUAL AXES LOGGING OPTIONS - DESELECTED BY DEFAULT
        self.chk_log_x_intensity = QCheckBox("Log X Axis")
        self.chk_log_x_intensity.setChecked(False)

        self.chk_log_y_intensity = QCheckBox("Log Y Axis")
        self.chk_log_y_intensity.setChecked(False)

        # Frame Cutoff Controls
        self.spin_frame_cutoff = NoScrollSpinBox()
        self.spin_frame_cutoff.setRange(0, 1000000)
        self.spin_frame_cutoff.setValue(0)
        self.spin_frame_cutoff.setSuffix(" frames")

        self.chk_apply_frame_cutoff = QCheckBox("Filter Spot Plot by Frame Cutoff (Frame ≥ N)")
        self.chk_apply_frame_cutoff.setChecked(False)

        self.spin_isingle_fraction = NoScrollDoubleSpinBox()
        self.spin_isingle_fraction.setDecimals(1)
        self.spin_isingle_fraction.setRange(0.0, 100.0)
        self.spin_isingle_fraction.setValue(default_isingle_frac)
        self.spin_isingle_fraction.setSingleStep(5.0)
        self.spin_isingle_fraction.setSuffix(" %")

        self.btn_recalc_isingle = QPushButton("Calculate iSingle from Spot KDE")
        self.btn_recalc_isingle.setStyleSheet("font-weight: bold; background-color: #008080; color: white; padding: 6px;")
        self.btn_recalc_isingle.clicked.connect(self.recalculate_isingle_from_spots)

        self.btn_calc_pystachio_isingle = QPushButton("Calculate iSingle from PySTACHIO .dat Files")
        self.btn_calc_pystachio_isingle.setStyleSheet("font-weight: bold; background-color: #2e6da4; color: white; padding: 6px;")
        self.btn_calc_pystachio_isingle.clicked.connect(self.recalculate_isingle_from_pystachio_dat)

        intensity_plot_layout.addRow("Intensity View:", self.combo_intensity_dist)
        intensity_plot_layout.addRow(self.chk_intensity_kde)
        intensity_plot_layout.addRow(self.chk_log_x_intensity)
        intensity_plot_layout.addRow(self.chk_log_y_intensity)
        intensity_plot_layout.addRow("Frame Cutoff (N):", self.spin_frame_cutoff)
        intensity_plot_layout.addRow(self.chk_apply_frame_cutoff)
        intensity_plot_layout.addRow("Auto Cutoff (% Frame):", self.spin_isingle_fraction)
        intensity_plot_layout.addRow(self.btn_recalc_isingle)
        intensity_plot_layout.addRow(self.btn_calc_pystachio_isingle)
        self.intensity_plot_group.setLayout(intensity_plot_layout)
        sidebar_layout.addWidget(self.intensity_plot_group)
        self.intensity_plot_group.setVisible(False)

        # 10. MSD VS TIME LAG SETTINGS (SHOWN ONLY ON MSD TAB)
        self.msd_options_group = QGroupBox("MSD vs Time Lag Options")
        msd_opt_layout = QFormLayout()
        msd_opt_layout.setVerticalSpacing(6)

        self.spin_msd_max_lag = NoScrollSpinBox()
        self.spin_msd_max_lag.setRange(2, 200)
        self.spin_msd_max_lag.setValue(20)
        self.spin_msd_max_lag.setSuffix(" frames")
        self.spin_msd_max_lag.valueChanged.connect(lambda: self.plot_msd_vs_timelag(self.get_filtered_summary_df()))

        self.chk_msd_show_individual = QCheckBox("Show Individual Trajectory MSDs")
        self.chk_msd_show_individual.setChecked(True)
        self.chk_msd_show_individual.stateChanged.connect(lambda: self.plot_msd_vs_timelag(self.get_filtered_summary_df()))

        self.chk_msd_fit_linear = QCheckBox("Fit Linear MSD slope (4D·τ)")
        self.chk_msd_fit_linear.setChecked(True)
        self.chk_msd_fit_linear.stateChanged.connect(lambda: self.plot_msd_vs_timelag(self.get_filtered_summary_df()))

        self.lbl_msd_fit_results = QLabel("MSD fit summary will appear here.")
        self.lbl_msd_fit_results.setWordWrap(True)
        self.lbl_msd_fit_results.setStyleSheet("font-size: 10px; background-color: #f8f9fa; padding: 6px; border: 1px solid #ccc; border-radius: 4px;")

        msd_opt_layout.addRow("Max Time Lag:", self.spin_msd_max_lag)
        msd_opt_layout.addRow(self.chk_msd_show_individual)
        msd_opt_layout.addRow(self.chk_msd_fit_linear)
        msd_opt_layout.addRow(self.lbl_msd_fit_results)
        self.msd_options_group.setLayout(msd_opt_layout)
        sidebar_layout.addWidget(self.msd_options_group)
        self.msd_options_group.setVisible(False)

        # Event connections
        self.spin_pixel_size.valueChanged.connect(self.recalculate_and_refresh)
        self.spin_exposure.valueChanged.connect(self.on_frame_time_or_alex_changed)
        self.chk_alex.stateChanged.connect(self.on_frame_time_or_alex_changed)

        self.spin_jump_lag.valueChanged.connect(self.refresh_all_plots)
        self.spin_bin_size.valueChanged.connect(self.refresh_all_plots)
        self.spin_bins.valueChanged.connect(self.refresh_all_plots)
        self.chk_log_scale.stateChanged.connect(self.refresh_all_plots)
        self.chk_kde.stateChanged.connect(self.refresh_all_plots)
        self.chk_show_subcomponents.stateChanged.connect(self.refresh_all_plots)
        self.chk_show_non_best.stateChanged.connect(self.refresh_all_plots)
        self.chk_show_subcomponents_jump.stateChanged.connect(self.refresh_all_plots)
        self.spin_bins_jump.valueChanged.connect(self.refresh_all_plots)
        self.chk_kde_jump.stateChanged.connect(self.refresh_all_plots)
        self.chk_show_non_best_jump.stateChanged.connect(self.refresh_all_plots)
        self.chk_filter_intensity.stateChanged.connect(self.refresh_all_plots)
        self.spin_isingle_L.valueChanged.connect(self.refresh_all_plots)
        self.spin_isingle_R.valueChanged.connect(self.refresh_all_plots)
        self.spin_isingle_fraction.valueChanged.connect(self.refresh_all_plots)
        self.combo_intensity_dist.currentIndexChanged.connect(self.refresh_all_plots)
        self.chk_intensity_kde.stateChanged.connect(self.refresh_all_plots)
        self.chk_log_x_intensity.stateChanged.connect(self.refresh_all_plots)
        self.chk_log_y_intensity.stateChanged.connect(self.refresh_all_plots)
        self.spin_frame_cutoff.valueChanged.connect(self.refresh_all_plots)
        self.chk_apply_frame_cutoff.stateChanged.connect(self.refresh_all_plots)

        # Enforce strong focus policy so widgets only receive input when clicked
        for widget in sidebar.findChildren((NoScrollSpinBox, NoScrollDoubleSpinBox, NoScrollComboBox)):
            widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # 11. Summary Info Panel
        self.lbl_stats = QLabel("No data loaded")
        self.lbl_stats.setStyleSheet("font-size: 11px; background-color: #f5f5f5; padding: 6px; border: 1px solid #ddd; border-radius: 4px;")
        self.lbl_stats.setWordWrap(True)
        sidebar_layout.addWidget(self.lbl_stats)

        sidebar_layout.addStretch()

        scroll_area.setWidget(sidebar)
        splitter.addWidget(scroll_area)

        # ================= RIGHT PANEL =================
        self.tabs = QTabWidget()
        splitter.addWidget(self.tabs)

        # TAB 1: Diffusion Coefficient Histogram
        self.tab_diff = QWidget()
        self.diff_canvas = MplCanvas(self)
        self.diff_toolbar = NavigationToolbar(self.diff_canvas, self)
        layout_diff = QVBoxLayout(self.tab_diff)
        layout_diff.addWidget(self.diff_toolbar)
        layout_diff.addWidget(self.diff_canvas)
        self.tabs.addTab(self.tab_diff, "Diffusion Coefficient Histogram (D)")

        # TAB 2: Jump Distance Histogram
        self.tab_jump = QWidget()
        self.jump_canvas = MplCanvas(self)
        self.jump_toolbar = NavigationToolbar(self.jump_canvas, self)
        layout_jump = QVBoxLayout(self.tab_jump)
        layout_jump.addWidget(self.jump_toolbar)
        layout_jump.addWidget(self.jump_canvas)
        self.tabs.addTab(self.tab_jump, "Jump Distance Histogram")

        # TAB 3: Spatial Trajectories Map
        self.tab_spatial = QWidget()
        layout_spatial = QVBoxLayout(self.tab_spatial)

        spatial_top_bar = QHBoxLayout()
        lbl_spatial_bg = QLabel("Background Overlay:")
        lbl_spatial_bg.setStyleSheet("font-weight: bold;")

        self.combo_spatial_bg = NoScrollComboBox()
        self.combo_spatial_bg.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.combo_spatial_bg.addItems([
            "L_channel Average (L_avg.tif)",
            "R_channel Average (R_avg.tif)",
            "Merged Left and Right (Merge)",
            "Brightfield Image (BF.tif)",
            "None (Blank Grid)"
        ])
        self.combo_spatial_bg.currentIndexChanged.connect(lambda: self.plot_spatial_map(self.get_filtered_summary_df()))

        spatial_top_bar.addWidget(lbl_spatial_bg)
        spatial_top_bar.addWidget(self.combo_spatial_bg, 1)

        self.spatial_canvas = MplCanvas(self)
        self.spatial_toolbar = NavigationToolbar(self.spatial_canvas, self)

        layout_spatial.addLayout(spatial_top_bar)
        layout_spatial.addWidget(self.spatial_toolbar)
        layout_spatial.addWidget(self.spatial_canvas)
        self.tabs.addTab(self.tab_spatial, "Spatial Map")

        # TAB 4: Single Field Inspector
        self.tab_field = QWidget()
        layout_field = QVBoxLayout(self.tab_field)

        field_top_bar = QHBoxLayout()
        lbl_field_image = QLabel("Image / PySTACHIO Output:")
        lbl_field_image.setStyleSheet("font-weight: bold;")

        self.combo_field_image = NoScrollComboBox()
        self.combo_field_image.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.combo_field_image.currentIndexChanged.connect(self.update_single_field_plot)

        field_top_bar.addWidget(lbl_field_image)
        field_top_bar.addWidget(self.combo_field_image, 1)

        self.field_canvas = MplCanvas(self)
        self.field_toolbar = NavigationToolbar(self.field_canvas, self)

        layout_field.addLayout(field_top_bar)
        layout_field.addWidget(self.field_toolbar)
        layout_field.addWidget(self.field_canvas)
        self.tabs.addTab(self.tab_field, "Single Field Inspector")

        # TAB 5: Intensity Distribution
        self.tab_stoich = QWidget()
        self.stoich_canvas = MplCanvas(self)
        self.stoich_toolbar = NavigationToolbar(self.stoich_canvas, self)
        layout_stoich = QVBoxLayout(self.tab_stoich)
        layout_stoich.addWidget(self.stoich_toolbar)
        layout_stoich.addWidget(self.stoich_canvas)
        self.tabs.addTab(self.tab_stoich, "Intensity Distribution")

        # TAB 6: MSD vs Time Lag (NEW TAB)
        self.tab_msd = QWidget()
        self.msd_canvas = MplCanvas(self)
        self.msd_toolbar = NavigationToolbar(self.msd_canvas, self)
        layout_msd = QVBoxLayout(self.tab_msd)
        layout_msd.addWidget(self.msd_toolbar)
        layout_msd.addWidget(self.msd_canvas)
        self.tabs.addTab(self.tab_msd, "MSD vs Time Lag")

        splitter.setSizes([340, 960])

        self.tabs.currentChanged.connect(self.on_tab_changed)
        self.on_tab_changed(self.tabs.currentIndex())

    def on_tab_changed(self, index):
        """Swaps sidebar control boxes to match the active tab."""
        active_tab = self.tabs.widget(index)
        is_diff = active_tab is self.tab_diff
        is_jump = active_tab is self.tab_jump
        is_spatial = active_tab is self.tab_spatial
        is_field = active_tab is self.tab_field
        is_intensity = active_tab is self.tab_stoich
        is_msd = active_tab is self.tab_msd

        self.diff_hist_group.setVisible(is_diff)
        self.diff_fit_group.setVisible(is_diff)
        self.jump_hist_group.setVisible(is_jump)
        self.jump_fit_group.setVisible(is_jump)
        self.spatial_options_group.setVisible(is_spatial)
        self.field_inspector_group.setVisible(is_field)
        self.intensity_plot_group.setVisible(is_intensity)
        self.msd_options_group.setVisible(is_msd)

    def update_channel_visibility(self):
        """Shows or hides channel selector and channel-specific iSingle spinboxes based on num_channels."""
        if self.trajectories_df.empty or 'channel' not in self.trajectories_df.columns:
            has_multiple_channels = False
        else:
            selected_sample = self.sample_selector.currentText()
            if selected_sample == "All Samples Combined" or not selected_sample:
                active_df = self.trajectories_df
            else:
                active_df = self.trajectories_df[self.trajectories_df['sample'] == selected_sample]

            unique_channels = active_df['channel'].nunique()
            has_multiple_channels = unique_channels > 1

        self.lbl_channel.setVisible(has_multiple_channels)
        self.channel_selector.setVisible(has_multiple_channels)

        if has_multiple_channels:
            self.lbl_isingle_L.setText("iSingle (L):")
            self.lbl_isingle_R.setVisible(True)
            self.spin_isingle_R.setVisible(True)
            target_chan = getattr(self, 'config_channel', "Left Channel (L)") or "Left Channel (L)"
            idx = self.channel_selector.findText(target_chan)
            if idx >= 0:
                self.channel_selector.blockSignals(True)
                self.channel_selector.setCurrentIndex(idx)
                self.channel_selector.blockSignals(False)
        else:
            self.lbl_isingle_L.setText("iSingle Value:")
            self.lbl_isingle_R.setVisible(False)
            self.spin_isingle_R.setVisible(False)
            self.channel_selector.blockSignals(True)
            self.channel_selector.setCurrentIndex(0)
            self.channel_selector.blockSignals(False)

# ================= ISINGLE RECALCULATION & KDE PEAK DETECTION ROUTINES =================

    def recalculate_isingle_from_spots(self):
        """Reruns KDE fitting on spot intensity distributions independently for L and R channels
        to calculate single fluorophore intensity (iSingle) from the peak location.
        """
        if self.trajectories_df.empty:
            QMessageBox.warning(self, "No Data", "No trajectory data available to calculate iSingle.")
            return

        selected_sample = self.sample_selector.currentText()
        if selected_sample == "All Samples Combined" or not selected_sample:
            active_df = self.trajectories_df.copy()
        else:
            active_df = self.trajectories_df[self.trajectories_df['sample'] == selected_sample].copy()

        min_f = self.spin_min_frame.value()
        max_f = self.spin_max_frame.value()
        active_df = active_df[(active_df['frame'] >= min_f) & (active_df['frame'] <= max_f)]

        selected_field = self.field_selector.currentText()
        if selected_field and selected_field != "-- All Fields --":
            active_df = active_df[active_df['field'] == selected_field]

        if active_df.empty:
            QMessageBox.warning(self, "No Data", "No matching spot data for current selection.")
            return

        intensity_col = 'integrated_intensity' if 'integrated_intensity' in active_df.columns else (
            'intensity' if 'intensity' in active_df.columns else (
                'spot_intensity' if 'spot_intensity' in active_df.columns else None
            )
        )

        if not intensity_col or intensity_col not in active_df.columns:
            QMessageBox.warning(self, "Missing Column", "Intensity column not found in trajectory dataset.")
            return

        max_frame = active_df['frame'].max()
        user_cutoff = self.spin_frame_cutoff.value()
        
        if user_cutoff > 0:
            cutoff_frame = user_cutoff
        else:
            frac = self.spin_isingle_fraction.value() / 100.0
            cutoff_frame = int(max_frame * frac)
            self.spin_frame_cutoff.blockSignals(True)
            self.spin_frame_cutoff.setValue(cutoff_frame)
            self.spin_frame_cutoff.blockSignals(False)

        cutoff_df = active_df[active_df['frame'] >= cutoff_frame]
        
        has_multi = active_df['channel'].nunique() > 1
        calculated_peaks = {}

        channels_to_process = ["Left Channel (L)", "Right Channel (R)"] if has_multi else [
            active_df['channel'].iloc[0] if 'channel' in active_df.columns else "Single Channel"
        ]

        for chan in channels_to_process:
            if has_multi:
                chan_df = cutoff_df[cutoff_df['channel'] == chan]
            else:
                chan_df = cutoff_df

            intensities = chan_df[intensity_col].dropna().values
            intensities = intensities[intensities > 1]

            if len(intensities) >= 10:
                try:
                    kde = gaussian_kde(intensities, bw_method=0.1)
                    x_grid = np.linspace(np.min(intensities), np.max(intensities), 1000)
                    pdf = kde.evaluate(x_grid)
                    peak = float(x_grid[np.argmax(pdf)])
                    calculated_peaks[chan] = (peak, len(intensities))
                except Exception as e:
                    print(f"Error computing KDE peak for {chan}: {e}")

        if not calculated_peaks:
            QMessageBox.warning(
                self, "Insufficient Data",
                f"Need at least 10 spot intensities on or after frame {cutoff_frame}."
            )
            return

        info_lines = []
        if "Left Channel (L)" in calculated_peaks or not has_multi:
            key = "Left Channel (L)" if "Left Channel (L)" in calculated_peaks else list(calculated_peaks.keys())[0]
            peak_L, n_L = calculated_peaks[key]
            self.spin_isingle_L.blockSignals(True)
            self.spin_isingle_L.setValue(peak_L)
            self.spin_isingle_L.blockSignals(False)
            label_L = "iSingle (L)" if has_multi else "iSingle"
            info_lines.append(f"• <b>{label_L}:</b> {peak_L:.1f} a.u. (KDE Peak, n={n_L} spots)")

        if "Right Channel (R)" in calculated_peaks and has_multi:
            peak_R, n_R = calculated_peaks["Right Channel (R)"]
            self.spin_isingle_R.blockSignals(True)
            self.spin_isingle_R.setValue(peak_R)
            self.spin_isingle_R.blockSignals(False)
            info_lines.append(f"• <b>iSingle (R):</b> {peak_R:.1f} a.u. (KDE Peak, n={n_R} spots)")

        QMessageBox.information(
            self,
            "KDE iSingle Peak Recalculation Successful",
            f"Calculated iSingle Peaks from Spot KDE Fit (Frame Cutoff ≥ {cutoff_frame}):<br>" + "<br>".join(info_lines)
        )
        self.refresh_all_plots()

    def recalculate_isingle_from_pystachio_dat(self):
        """Calculates iSingle values from scanned PySTACHIO .dat output files (*_channel_i_single_peak.dat) using KDE peak detection."""
        if not self.field_isingle_data:
            QMessageBox.warning(
                self, "No PySTACHIO dat files",
                "No scanned PySTACHIO iSingle .dat files found in the loaded experiment directory."
            )
            return

        df_is = pd.DataFrame(self.field_isingle_data)
        info_lines = []

        # Left / Single Channel
        df_L = df_is[df_is['channel'].isin(['Left Channel (L)', 'Single Channel'])]
        if not df_L.empty:
            vals_L = df_L['i_single'].dropna().values
            vals_L = vals_L[vals_L > 0]
            if len(vals_L) >= 3:
                try:
                    kde_L = gaussian_kde(vals_L, bw_method=0.15)
                    x_grid_L = np.linspace(vals_L.min() * 0.5, vals_L.max() * 1.5, 1000)
                    pdf_L = kde_L.evaluate(x_grid_L)
                    peak_L = float(x_grid_L[np.argmax(pdf_L)])
                    self.spin_isingle_L.blockSignals(True)
                    self.spin_isingle_L.setValue(peak_L)
                    self.spin_isingle_L.blockSignals(False)
                    info_lines.append(f"• <b>iSingle (L) [PySTACHIO dat KDE]:</b> {peak_L:.1f} a.u. (n={len(vals_L)} files)")
                except Exception as e:
                    print(f"Error computing KDE on PySTACHIO L dat files: {e}")

        # Right Channel
        df_R = df_is[df_is['channel'] == 'Right Channel (R)']
        if not df_R.empty:
            vals_R = df_R['i_single'].dropna().values
            vals_R = vals_R[vals_R > 0]
            if len(vals_R) >= 3:
                try:
                    kde_R = gaussian_kde(vals_R, bw_method=0.15)
                    x_grid_R = np.linspace(vals_R.min() * 0.5, vals_R.max() * 1.5, 1000)
                    pdf_R = kde_R.evaluate(x_grid_R)
                    peak_R = float(x_grid_R[np.argmax(pdf_R)])
                    self.spin_isingle_R.blockSignals(True)
                    self.spin_isingle_R.setValue(peak_R)
                    self.spin_isingle_R.blockSignals(False)
                    info_lines.append(f"• <b>iSingle (R) [PySTACHIO dat KDE]:</b> {peak_R:.1f} a.u. (n={len(vals_R)} files)")
                except Exception as e:
                    print(f"Error computing KDE on PySTACHIO R dat files: {e}")

        if info_lines:
            QMessageBox.information(
                self,
                "PySTACHIO .dat File KDE Successful",
                "Calculated iSingle values from PySTACHIO .dat outputs:<br><br>" + "<br>".join(info_lines)
            )
            self.refresh_all_plots()
        else:
            QMessageBox.warning(
                self, "Insufficient Data",
                "Not enough PySTACHIO .dat values found to perform KDE peak detection."
            )

# ================= MLE FITTING & BIC EVALUATION =================

    def _on_fit_progress(self, current, total, message):
        """Updates progress dialog step count and status text."""
        if self.progress_dialog:
            self.progress_dialog.setLabelText(message)
            self.progress_dialog.setValue(current)

    def _on_fit_error(self, error_msg):
        """Handles background fitting errors."""
        self.btn_run_diff_fitting.setEnabled(True)
        self.btn_run_jump_fitting.setEnabled(True)
        if self.progress_dialog:
            self.progress_dialog.close()
        QMessageBox.critical(self, "Fitting Error", f"An error occurred during fitting:\n{error_msg}")

    def run_diffusion_fitting(self):
        """Spawns background thread to fit gamma-mixture models 1..max_components."""
        summary_df = self.get_filtered_summary_df()

        if summary_df.empty:
            QMessageBox.warning(self, "No Data", "No valid trajectory data available to fit.")
            return

        max_comp = self.spin_max_components.value()
        d_values = summary_df['D'].dropna().values
        d_clean = d_values[d_values > 0]

        if len(d_clean) < 10:
            QMessageBox.warning(self, "Insufficient Data", "Need at least 10 positive D values to fit mixture models.")
            return

        self.diff_fit_results = []

        self.progress_dialog = QProgressDialog("Initializing diffusion fitting...", "Cancel", 0, max_comp, self)
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.setValue(0)

        self.worker_thread = FitWorkerThread(
            fit_type='diffusion',
            data=d_clean,
            max_comp=max_comp,
            parent=self
        )

        self.worker_thread.progress_signal.connect(self._on_fit_progress)
        self.worker_thread.finished_signal.connect(self._on_diffusion_fit_finished)
        self.worker_thread.error_signal.connect(self._on_fit_error)
        self.progress_dialog.canceled.connect(self.worker_thread.cancel)

        self.btn_run_diff_fitting.setEnabled(False)
        self.worker_thread.start()

    def _on_diffusion_fit_finished(self, results):
        """Callback executed on the UI thread when diffusion fitting completes."""
        self.btn_run_diff_fitting.setEnabled(True)
        self.diff_fit_results = results

        if not self.diff_fit_results:
            self.best_diff_model = None
            self.lbl_diff_fit_results.setText("Diffusion fitting was canceled or failed to converge.")
            return

        self.best_diff_model = min(self.diff_fit_results, key=lambda x: x['bic'])
        best_diff_k = self.best_diff_model['components']

        chan_label = self.channel_selector.currentText() if self.channel_selector.isVisible() else "Single Channel"
        summary_lines = [
            f"<b>Best Diffusion Model ({chan_label}): {best_diff_k}-Component(s)</b>"
        ]
        for res in self.diff_fit_results:
            k = res['components']
            bic = res['bic']
            delta_bic = bic - self.best_diff_model['bic']
            mark = "★ BEST" if k == best_diff_k else f"(ΔBIC: +{delta_bic:.1f})"
            summary_lines.append(f"• <b>{k}-Comp:</b> BIC = {bic:.1f} {mark}")

            if k == best_diff_k:
                for idx, p in enumerate(res['params'], 1):
                    w = p.get('weight', 1.0) * 100
                    mean_d = p.get('mean_D', p.get('D', 0.0))
                    summary_lines.append(f"  └ State {idx}: {w:.1f}%, D ≈ {mean_d:.3f} μm²/s")

        self.lbl_diff_fit_results.setText("<br>".join(summary_lines))
        self.refresh_all_plots()

    def run_jump_fitting(self):
        """Spawns background thread to fit jump distance models 1..max_components."""
        summary_df = self.get_filtered_summary_df()

        if summary_df.empty:
            QMessageBox.warning(self, "No Data", "No valid trajectory data available to fit.")
            return

        max_comp = self.spin_max_components_jump.value()

        dt_ms = self.spin_exposure.value()
        dt = (dt_ms / 1000.0) * (2.0 if self.chk_alex.isChecked() else 1.0)
        lag_N = self.spin_jump_lag.value()
        dt_lag = dt * lag_N

        jumps = self.extract_jump_distances(summary_df)
        if len(jumps) < 10:
            QMessageBox.warning(self, "Insufficient Data", "Need at least 10 jump distances to fit mixture models.")
            return

        sigma_loc = self.loc_precision if self.loc_precision is not None else 0.0
        selected_model = self.combo_jdd_model.currentText()

        self.jump_fit_results = []

        self.progress_dialog = QProgressDialog("Initializing jump distance fitting...", "Cancel", 0, max_comp, self)
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.setValue(0)

        self.worker_thread = FitWorkerThread(
            fit_type='jump',
            data=jumps,
            max_comp=max_comp,
            dt=dt_lag,
            loc_error=sigma_loc,
            model_type=selected_model,
            parent=self
        )

        self.worker_thread.progress_signal.connect(self._on_fit_progress)
        self.worker_thread.finished_signal.connect(self._on_jump_fit_finished)
        self.worker_thread.error_signal.connect(self._on_fit_error)
        self.progress_dialog.canceled.connect(self.worker_thread.cancel)

        self.btn_run_jump_fitting.setEnabled(False)
        self.worker_thread.start()

    def _on_jump_fit_finished(self, results):
        """Callback executed on the UI thread when jump distance fitting completes."""
        self.btn_run_jump_fitting.setEnabled(True)
        self.jump_fit_results = results

        if not self.jump_fit_results:
            self.best_jump_model = None
            self.lbl_jump_fit_results.setText("Jump distance fitting was canceled or failed to converge.")
            return

        valid_models = [m for m in self.jump_fit_results if not m.get('has_degenerate', False)]
        if valid_models:
            self.best_jump_model = min(valid_models, key=lambda x: x['bic'])
        else:
            self.best_jump_model = min(self.jump_fit_results, key=lambda x: x['bic'])

        best_j_k = self.best_jump_model['components']
        selected_model = self.best_jump_model.get('model_type', self.combo_jdd_model.currentText())
        chan_label = self.channel_selector.currentText() if self.channel_selector.isVisible() else "Single Channel"

        summary_lines = [
            f"<b>JDD Model:</b> {selected_model}",
            f"<b>Best Jump Distance Model ({chan_label}): {best_j_k}-Component(s)</b>"
        ]

        best_params = self.best_jump_model.get('params', [])
        if selected_model == "Mixed (Pure Brownian + Anomalous)":
            if best_j_k == 1:
                pure_pct = 100.0
                anom_pct = 0.0
            else:
                pure_pct = sum(p.get('weight', 0.0) for p in best_params[:-1]) * 100.0
                anom_pct = best_params[-1].get('weight', 0.0) * 100.0 if best_params else 0.0
            
            summary_lines.append(f"📊 <b>Proportions:</b> Pure Brownian: <b>{pure_pct:.1f}%</b> | Anomalous: <b>{anom_pct:.1f}%</b>")

        elif "Pure" in selected_model or "Rayleigh" in selected_model:
            summary_lines.append("📊 <b>Proportions:</b> Pure Brownian: <b>100.0%</b> | Anomalous: <b>0.0%</b>")

        elif selected_model == "Anomalous 2D Diffusion":
            summary_lines.append("📊 <b>Proportions:</b> Pure Brownian: <b>0.0%</b> | Anomalous: <b>100.0%</b>")

        for res in self.jump_fit_results:
            k = res['components']
            bic = res['bic']
            delta_bic = bic - self.best_jump_model['bic']
            is_best = (k == best_j_k)
            is_degen = res.get('has_degenerate', False)

            degen_tag = " [Degenerate <1%]" if is_degen else ""
            mark = "★ BEST" if is_best else f"(ΔBIC: +{delta_bic:.1f}){degen_tag}"
            summary_lines.append(f"• <b>{k}-Comp:</b> BIC = {bic:.1f} {mark}")

            if is_best:
                for idx, p in enumerate(res['params'], 1):
                    w = p.get('weight', 1.0) * 100.0
                    d_val = p.get('D', p.get('mean_D', 0.0))
                    alpha_val = p.get('alpha', 1.0)

                    if selected_model == "Mixed (Pure Brownian + Anomalous)":
                        is_anom = (best_j_k > 1) and (idx == len(res['params']))
                        comp_type = "Anomalous" if is_anom else "Pure"
                        summary_lines.append(
                            f"  └ State {idx} ({comp_type}): D ≈ {d_val:.3f} μm²/s, α = {alpha_val:.2f} ({w:.1f}%)"
                        )
                    elif selected_model == "Anomalous 2D Diffusion":
                        summary_lines.append(
                            f"  └ State {idx}: D ≈ {d_val:.3f} μm²/s, α = {alpha_val:.2f} ({w:.1f}%)"
                        )
                    else:
                        summary_lines.append(f"  └ State {idx}: D ≈ {d_val:.3f} μm²/s ({w:.1f}%)")

        self.lbl_jump_fit_results.setText("<br>".join(summary_lines))
        self.refresh_all_plots()

    def export_fit_parameters_npy(self):
        """Exports model fit parameters and acquisition metadata to a NumPy .npy binary file."""
        if not self.diff_fit_results and not self.jump_fit_results:
            QMessageBox.warning(
                self,
                "No Fit Results",
                "No model fit parameters are available to export. Please run a fit first."
            )
            return

        default_filename = "model_fit_parameters.npy"
        default_path = os.path.join(self.root_dir, default_filename) if self.root_dir else default_filename

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Model Fit Parameters",
            default_path,
            "NumPy Files (*.npy);;All Files (*)"
        )

        if not filepath:
            return

        if not filepath.endswith('.npy'):
            filepath += '.npy'

        def sanitize_params(param_list):
            sanitized = []
            for p in param_list:
                clean_p = {k: v for k, v in p.items() if not callable(v)}
                sanitized.append(clean_p)
            return sanitized

        export_dict = {
            'metadata': {
                'sample': self.sample_selector.currentText(),
                'channel': (
                    self.channel_selector.currentText()
                    if self.channel_selector.isVisible()
                    else "Single Channel"
                ),
                'field': self.field_selector.currentText(),
                'pixel_size_um': self.spin_pixel_size.value(),
                'frame_interval_ms': self.spin_exposure.value(),
                'alex_mode': self.chk_alex.isChecked(),
                'localisation_precision_um': self.loc_precision,
                'min_track_length': self.spin_min_len.value(),
                'min_snr_threshold': self.spin_min_snr.value(),
                'min_frame': self.spin_min_frame.value(),
                'max_frame': self.spin_max_frame.value(),
                'negative_D_removed': self.chk_remove_neg.isChecked(),
                'jdd_model': self.combo_jdd_model.currentText(),
                'jump_lag_frames': self.spin_jump_lag.value(),
                'spatial_jump_lag_frames': self.spin_spatial_jump_lag.value() if hasattr(self, 'spin_spatial_jump_lag') else 1,
                'spatial_color_mode': self.combo_spatial_color_mode.currentText() if hasattr(self, 'combo_spatial_color_mode') else "Default",
                'effective_dt_seconds': (
                    (self.spin_exposure.value() / 1000.0)
                    * (2.0 if self.chk_alex.isChecked() else 1.0)
                    * self.spin_jump_lag.value()
                ),
                'histogram_bins': self.spin_bins.value(),
                'jump_histogram_bins': self.spin_bins_jump.value(),
                'diffusion_bin_size': self.spin_bin_size.value(),
                'log_scale_enabled': self.chk_log_scale.isChecked(),
                'kde_enabled': self.chk_kde.isChecked(),
                'jump_kde_enabled': self.chk_kde_jump.isChecked(),
                'intensity_filter_enabled': self.chk_filter_intensity.isChecked(),
                'isingle_value_L': self.spin_isingle_L.value(),
                'isingle_value_R': self.spin_isingle_R.value(),
                'isingle_fraction': self.spin_isingle_fraction.value(),
                'intensity_frame_cutoff': self.spin_frame_cutoff.value(),
                'max_components_tested': self.spin_max_components.value(),
                'max_components_tested_jump': self.spin_max_components_jump.value(),
                'num_tracks_used': len(self.get_filtered_summary_df()),
                'num_jump_distances': len(
                    self.extract_jump_distances(
                        self.get_filtered_summary_df()
                    )
                )
            },
            'diffusion_coefficient_fits': {
                'all_models': [
                    {
                        'components': fit.get('components'),
                        'bic': fit.get('bic'),
                        'log_likelihood': fit.get(
                            'log_likelihood',
                            fit.get('log_lh', None)
                        ),
                        'params': sanitize_params(
                            fit.get('params', [])
                        )
                    }
                    for fit in self.diff_fit_results
                ],
                'best_model': {
                    'components': self.best_diff_model.get('components'),
                    'bic': self.best_diff_model.get('bic'),
                    'params': sanitize_params(
                        self.best_diff_model.get('params', [])
                    )
                } if self.best_diff_model else None
            },
            'jump_distance_fits': {
                'all_models': [
                    {
                        'model_type': self.combo_jdd_model.currentText(),
                        'components': fit.get('components'),
                        'bic': fit.get('bic'),
                        'aic': fit.get('aic'),
                        'log_likelihood': fit.get(
                            'log_likelihood',
                            fit.get('log_lh', None)
                        ),
                        'params': sanitize_params(
                            fit.get('params', [])
                        )
                    }
                    for fit in self.jump_fit_results
                ],
                'best_model': {
                    'model_type': self.combo_jdd_model.currentText(),
                    'components': self.best_jump_model.get('components'),
                    'bic': self.best_jump_model.get('bic'),
                    'aic': self.best_jump_model.get('aic'),
                    'log_likelihood': self.best_jump_model.get(
                        'log_likelihood',
                        self.best_jump_model.get('log_lh', None)
                    ),
                    'params': sanitize_params(
                        self.best_jump_model.get('params', [])
                    )
                } if self.best_jump_model else None
            }
        }

        try:
            np.save(filepath, export_dict, allow_pickle=True)
            QMessageBox.information(
                self,
                "Export Successful",
                f"Model fit parameters successfully exported to:\n{filepath}"
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export model fit parameters:\n{str(e)}"
            )

    def clear_diff_fits(self):
        """Resets cached diffusion fitting models."""
        self.diff_fit_results = []
        self.best_diff_model = None
        self.lbl_diff_fit_results.setText("No fitting run yet.")

    def clear_jump_fits(self):
        """Resets cached jump distance fitting models."""
        self.jump_fit_results = []
        self.best_jump_model = None
        self.lbl_jump_fit_results.setText("No fitting run yet.")

    def clear_fits(self):
        """Resets all cached fitting models."""
        self.clear_diff_fits()
        self.clear_jump_fits()

    def extract_jump_distances(self, summary_df):
        """Extracts jump distances array matching current sidebar filtering."""
        if self.trajectories_df.empty or summary_df.empty:
            return np.array([])

        allowed_ids = set(summary_df['global_track_id'].unique())
        df = self.trajectories_df[self.trajectories_df['global_track_id'].isin(allowed_ids)]

        min_f = self.spin_min_frame.value()
        max_f = self.spin_max_frame.value()
        df = df[(df['frame'] >= min_f) & (df['frame'] <= max_f)]

        pixel_size = self.spin_pixel_size.value()
        lag_N = self.spin_jump_lag.value()

        jumps = []
        for track_id, track in df.groupby('global_track_id'):
            if len(track) <= lag_N:
                continue

            track_sorted = track.sort_values('frame')
            frames = track_sorted['frame'].values
            x = track_sorted['x'].values * pixel_size
            y = track_sorted['y'].values * pixel_size

            frame_diffs = frames[lag_N:] - frames[:-lag_N]
            valid_mask = (frame_diffs == lag_N)

            if not np.any(valid_mask):
                continue

            dx = (x[lag_N:] - x[:-lag_N])[valid_mask]
            dy = (y[lag_N:] - y[:-lag_N])[valid_mask]
            step_distances = np.sqrt(dx**2 + dy**2)
            jumps.extend(step_distances)

        return np.array(jumps)

    # ================= TRANSFORMATION MATRIX HELPERS =================

    def find_transformation_matrix(self, field_name=None):
        """Looks for transformation_matrix.npy first in relevant field results folder, then in directory where config is located."""
        candidates = []
        if field_name and field_name in self.field_dirs_map:
            field_dir = self.field_dirs_map[field_name]
            candidates.append(os.path.join(field_dir, "results", "transformation_matrix.npy"))
            candidates.append(os.path.join(field_dir, "transformation_matrix.npy"))

        if self.root_dir:
            candidates.append(os.path.join(self.root_dir, "transformation_matrix.npy"))
            glob_matches = glob.glob(os.path.join(self.root_dir, "**", "transformation_matrix.npy"), recursive=True)
            for gm in glob_matches:
                if gm not in candidates:
                    candidates.append(gm)

        for path in candidates:
            if os.path.exists(path):
                try:
                    tmat = np.load(path)
                    return tmat, path
                except Exception as e:
                    print(f"Error loading transformation matrix from {path}: {e}")
        return None, None

    def transform_track_coordinates(self, x, y, tmat):
        """Transforms coordinate arrays (x, y) using transformation_matrix.npy."""
        if tmat is None:
            return x, y
        try:
            x_arr = np.asarray(x, dtype=np.float64)
            y_arr = np.asarray(y, dtype=np.float64)
            pts = np.vstack([x_arr, y_arr, np.ones_like(x_arr)])

            # Ensure 3x3 matrix shape for inverse calculation
            if tmat.shape == (2, 3):
                tmat_3x3 = np.vstack([tmat, [0, 0, 1]])
            elif tmat.shape == (3, 3):
                tmat_3x3 = tmat
            else:
                return x, y

            # StackReg matrices map Target (L) -> Source (R) for image resampling.
            # To map track coordinates from Source (R) -> Target (L), invert the matrix.
            inv_tmat = np.linalg.inv(tmat_3x3)
            res = inv_tmat @ pts

            if np.any(res[2] != 0):
                return res[0] / res[2], res[1] / res[2]
            return res[0], res[1]
        except Exception as e:
            print(f"Error transforming track coordinates: {e}")
            return x, y

    # ================= CONFIG PARSING =================

    def parse_config_file(self, filepath):
        config = {}
        PIXEL_KEYS = {'pixel_size', 'pixel_length', 'pixelsize', 'px_size', 'pixel_sz', 'pixel', 'px'}
        FRAME_KEYS = {'frame_time', 'frame_interval', 'frametime', 'exposure_time', 'exposure', 'time_step', 'timestep', 'dt', 'frame_duration', 'frame_period'}
        ALEX_KEYS = {'alex', 'alex_flag', 'alex_mode', 'use_alex', 'is_alex'}
        PRECISION_KEYS = {'localisation_precision', 'localization_precision', 'loc_prec', 'loc_precision', 'precision', 'sigma'}
        CHANNEL_KEYS = {'channel', 'channel_name', 'chan', 'channels', 'active_channel', 'target_channel', 'im_channel', 'image_channel'}

        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.split('#')[0].split('//')[0].strip()
                    if not line:
                        continue

                    if '=' in line:
                        key, val = line.split('=', 1)
                    elif ':' in line:
                        key, val = line.split(':', 1)
                    elif '\t' in line:
                        key, val = line.split('\t', 1)
                    else:
                        parts = line.split(maxsplit=1)
                        if len(parts) == 2:
                            key, val = parts
                        else:
                            continue

                    key = key.strip().lower().replace(' ', '_')
                    val = val.strip()

                    if key in PIXEL_KEYS:
                        m = re.search(r"[-+]?\d*\.\d+|\d+", val)
                        if m:
                            config['pixel_size'] = float(m.group())
                    elif key in FRAME_KEYS:
                        m = re.search(r"[-+]?\d*\.\d+|\d+", val)
                        if m:
                            ft = float(m.group())
                            if ft < 0.1:
                                ft *= 1000.0
                            config['frame_time'] = ft
                    elif key in ALEX_KEYS:
                        val_lower = val.lower()
                        if val_lower in ['true', '1', 'yes', 't', 'on']:
                            config['alex'] = True
                        elif val_lower in ['false', '0', 'no', 'f', 'off']:
                            config['alex'] = False
                    elif key in PRECISION_KEYS:
                        m = re.search(r"[-+]?\d*\.\d+|\d+", val)
                        if m:
                            config['localisation_precision'] = float(m.group())
                    elif key in CHANNEL_KEYS:
                        val_lower = val.lower()
                        if 'right' in val_lower or val_lower == 'r' or val_lower == '2':
                            config['channel'] = "Right Channel (R)"
                        elif 'left' in val_lower or val_lower == 'l' or val_lower == '1':
                            config['channel'] = "Left Channel (L)"
                        elif 'both' in val_lower or 'overlay' in val_lower or val_lower == '0':
                            config['channel'] = "Both Channels (Overlay)"
                        else:
                            config['channel'] = val.strip()
        except Exception as e:
            print(f"Error parsing config file {filepath}: {e}")
        return config

    def has_config_conflicts(self, parsed_configs):
        configs = list(parsed_configs.values())
        if len(configs) <= 1:
            return False

        keys = ['pixel_size', 'frame_time', 'alex', 'localisation_precision', 'channel']
        for k in keys:
            vals = [c[k] for c in configs if k in c]
            if len(vals) > 1:
                first = vals[0]
                for v in vals[1:]:
                    if isinstance(first, float) or isinstance(v, float):
                        if abs(first - v) > 1e-6:
                            return True
                    else:
                        if first != v:
                            return True
        return False

    def update_default_bin_size(self):
        dt_ms = self.spin_exposure.value()
        is_alex = self.chk_alex.isChecked()

        dt_sec = (dt_ms / 1000.0) * (2.0 if is_alex else 1.0)

        # Set a clean, realistic default bin size in um^2/s
        if self.loc_precision is not None:
            sigma = self.loc_precision
            bin_size = (sigma ** 2) / (4.0 * dt_sec) if dt_sec > 0 else 0.02
        else:
            bin_size = 0.02

        bin_size = float(np.clip(bin_size, 0.001, 2.0))

        self.spin_bin_size.blockSignals(True)
        self.spin_bin_size.setValue(bin_size)
        self.spin_bin_size.blockSignals(False)

    def on_frame_time_or_alex_changed(self):
        self.update_default_bin_size()
        self.recalculate_and_refresh()

    def on_sample_changed(self):
        self.clear_fits()
        self.update_field_selector_items()
        self.update_channel_visibility()
        self.recalculate_and_refresh()

    def on_channel_changed(self):
        self.clear_fits()
        self.refresh_all_plots()

    def on_field_changed(self):
        self.populate_field_images()
        self.refresh_all_plots()

    # ================= DIRECTORY & TRAJECTORY SCANNING =================

    def scan_and_load_directory(self):
        """Scans directory for config files and field results folders containing trajectories."""
        if not self.root_dir or not os.path.exists(self.root_dir):
            return

        config_pattern = os.path.join(self.root_dir, "**", "*config*.txt")
        config_files = [f for f in glob.glob(config_pattern, recursive=True) if f.endswith('.txt')]

        parsed_configs = {}
        for cfg_file in config_files:
            cfg = self.parse_config_file(cfg_file)
            if cfg:
                parsed_configs[cfg_file] = cfg

        selected_cfg = {}
        if len(parsed_configs) == 1:
            selected_cfg = list(parsed_configs.values())[0]
        elif len(parsed_configs) > 1:
            if self.has_config_conflicts(parsed_configs):
                dialog = ConfigSelectionDialog(parsed_configs, parent=self)
                if dialog.exec():
                    chosen_file = dialog.get_selected_filepath()
                    selected_cfg = parsed_configs.get(chosen_file, {})
                else:
                    selected_cfg = list(parsed_configs.values())[0]
            else:
                for cfg in parsed_configs.values():
                    selected_cfg.update(cfg)

        if selected_cfg:
            if 'pixel_size' in selected_cfg:
                self.spin_pixel_size.blockSignals(True)
                self.spin_pixel_size.setValue(selected_cfg['pixel_size'])
                self.spin_pixel_size.blockSignals(False)
            if 'frame_time' in selected_cfg:
                self.spin_exposure.blockSignals(True)
                self.spin_exposure.setValue(selected_cfg['frame_time'])
                self.spin_exposure.blockSignals(False)
            if 'alex' in selected_cfg:
                self.chk_alex.blockSignals(True)
                self.chk_alex.setChecked(selected_cfg['alex'])
                self.chk_alex.blockSignals(False)
            if 'localisation_precision' in selected_cfg:
                self.loc_precision = selected_cfg['localisation_precision']
            else:
                self.loc_precision = None
            if 'channel' in selected_cfg:
                self.config_channel = selected_cfg['channel']
            else:
                self.config_channel = "Left Channel (L)"

        self.update_default_bin_size()

        # Scan Trajectories
        search_pattern = os.path.join(self.root_dir, "**", "*_trajectories.tsv")
        all_files = glob.glob(search_pattern, recursive=True)
        
        print(f"\n[INFO] Found {len(all_files)} trajectory file(s) in '{self.root_dir}'\n", flush=True)

        if not all_files:
            self.lbl_stats.setText("No *_trajectories.tsv files found.")
            self.trajectories_df = pd.DataFrame()
            self.field_dirs_map = {}
            self.field_isingle_data = []
            self.populate_sample_dropdown([])
            self.update_channel_visibility()
            return

        traj_list = []
        self.field_dirs_map = {}

        for traj_path in all_files:
            norm_path = os.path.normpath(traj_path)
            path_parts = norm_path.split(os.sep)

            if any(part.startswith("cell_") or part.startswith("cell") for part in path_parts[:-1]):
                continue

            filename = os.path.basename(traj_path)

            if "results" in path_parts:
                res_idx = path_parts.index("results")
                field_name = path_parts[res_idx - 1] if res_idx > 0 else os.path.basename(os.path.dirname(traj_path))
                field_dir = os.path.dirname(os.path.dirname(norm_path))
                sample_label = path_parts[res_idx - 2] if res_idx > 1 else os.path.basename(self.root_dir)
            else:
                field_name = os.path.basename(os.path.dirname(norm_path))
                field_dir = os.path.dirname(norm_path)
                sample_label = path_parts[-3] if len(path_parts) >= 3 else os.path.basename(self.root_dir)

            if sample_label.lower() == "results":
                sample_label = os.path.basename(self.root_dir)

            self.field_dirs_map[field_name] = field_dir

            if "R_channel" in filename or "_R_" in filename or "_R." in filename:
                channel = "Right Channel (R)"
                chan_code = "R"
            elif "L_channel" in filename or "_L_" in filename or "_L." in filename:
                channel = "Left Channel (L)"
                chan_code = "L"
            else:
                channel = "Left Channel (L)"
                chan_code = "L"

            try:
                df = pd.read_csv(traj_path, sep='\t')
                df['sample'] = sample_label
                df['field'] = field_name
                df['channel'] = channel

                raw_id_col = 'trajectory' if 'trajectory' in df.columns else (
                    'track_id' if 'track_id' in df.columns else None
                )

                if raw_id_col and raw_id_col in df.columns:
                    df['global_track_id'] = field_name + "_" + chan_code + "_" + df[raw_id_col].astype(str)
                else:
                    df['global_track_id'] = field_name + "_" + chan_code + "_tr" + df.index.astype(str)

                traj_list.append(df)
            except Exception as e:
                print(f"Error reading {traj_path}: {e}")

        if traj_list:
            self.trajectories_df = pd.concat(traj_list, ignore_index=True)

            # Dynamically set max frame spinbox to highest frame in dataset
            if 'frame' in self.trajectories_df.columns and not self.trajectories_df.empty:
                max_dataset_frame = int(self.trajectories_df['frame'].max())
                self.spin_max_frame.blockSignals(True)
                self.spin_max_frame.setValue(max_dataset_frame)
                self.spin_max_frame.blockSignals(False)

            samples = sorted(self.trajectories_df['sample'].unique().tolist())
            self.populate_sample_dropdown(samples)
        else:
            self.trajectories_df = pd.DataFrame()
            self.populate_sample_dropdown([])

        # Scan and process field iSingle dat files across discovered fields
        self.load_and_process_field_isingle_files()
        self.update_channel_visibility()
        self.refresh_all_plots()

    def _read_dat_floats(self, filepath):
        """Helper to parse numeric float values from iSingle .dat files."""
        vals = []
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                matches = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", content)
                for m in matches:
                    v = float(m)
                    if v > 0:
                        vals.append(v)
        except Exception as e:
            print(f"Error reading i_single dat file {filepath}: {e}")
        return vals

    def load_and_process_field_isingle_files(self):
        """Scans trajectory directories for channel i_single peak files, calculates channel-specific averages,
        updates spinbox values independently for L and R, and prepares boxplot data.
        """
        self.field_isingle_data = []

        if not self.field_dirs_map:
            return

        for field_name, field_dir in self.field_dirs_map.items():
            search_dirs = [field_dir, os.path.join(field_dir, "results")]

            for s_dir in search_dirs:
                if not os.path.exists(s_dir):
                    continue

                l_files = (
                    glob.glob(os.path.join(s_dir, "*L_channel_i_single_peak.dat")) +
                    glob.glob(os.path.join(s_dir, "*L_channel_i_single.dat"))
                )
                r_files = (
                    glob.glob(os.path.join(s_dir, "*R_channel_i_single_peak.dat")) +
                    glob.glob(os.path.join(s_dir, "*R_channel_i_single.dat"))
                )
                gen_files = (
                    glob.glob(os.path.join(s_dir, "*_channel_i_single_peak.dat")) +
                    glob.glob(os.path.join(s_dir, "*_channel_i_single.dat")) +
                    glob.glob(os.path.join(s_dir, "*_i_single_peak.dat")) +
                    glob.glob(os.path.join(s_dir, "*_i_single.dat"))
                )

                for fpath in set(l_files):
                    vals = self._read_dat_floats(fpath)
                    for v in vals:
                        self.field_isingle_data.append({
                            'field': field_name,
                            'channel': 'Left Channel (L)',
                            'i_single': v
                        })

                for fpath in set(r_files):
                    vals = self._read_dat_floats(fpath)
                    for v in vals:
                        self.field_isingle_data.append({
                            'field': field_name,
                            'channel': 'Right Channel (R)',
                            'i_single': v
                        })

                if not l_files and not r_files:
                    for fpath in set(gen_files):
                        vals = self._read_dat_floats(fpath)
                        for v in vals:
                            self.field_isingle_data.append({
                                'field': field_name,
                                'channel': 'Single Channel',
                                'i_single': v
                            })

        if self.field_isingle_data:
            df_is = pd.DataFrame(self.field_isingle_data)
            
            # Left channel / Single channel average
            df_L = df_is[df_is['channel'].isin(['Left Channel (L)', 'Single Channel'])]
            if not df_L.empty:
                mean_L = df_L['i_single'].mean()
                if not np.isnan(mean_L) and mean_L > 0:
                    self.spin_isingle_L.blockSignals(True)
                    self.spin_isingle_L.setValue(float(mean_L))
                    self.spin_isingle_L.blockSignals(False)

            # Right channel average
            df_R = df_is[df_is['channel'] == 'Right Channel (R)']
            if not df_R.empty:
                mean_R = df_R['i_single'].mean()
                if not np.isnan(mean_R) and mean_R > 0:
                    self.spin_isingle_R.blockSignals(True)
                    self.spin_isingle_R.setValue(float(mean_R))
                    self.spin_isingle_R.blockSignals(False)

    def select_new_directory(self):
        selected_dir = QFileDialog.getExistingDirectory(self, "Select Experiment Directory", self.root_dir or ".")
        if selected_dir:
            self.root_dir = os.path.abspath(selected_dir)
            self.lbl_root_dir.setText(self.root_dir)
            self.scan_and_load_directory()

    def populate_sample_dropdown(self, samples):
        self.sample_selector.blockSignals(True)
        self.sample_selector.clear()

        if samples:
            if len(samples) > 1:
                self.sample_selector.addItem("All Samples Combined")
            for s in samples:
                self.sample_selector.addItem(s)

        self.sample_selector.blockSignals(False)
        self.update_field_selector_items()
        self.update_channel_visibility()
        self.recalculate_and_refresh()

    def update_field_selector_items(self):
        self.field_selector.blockSignals(True)
        self.field_selector.clear()

        if not self.trajectories_df.empty:
            selected_sample = self.sample_selector.currentText()
            if selected_sample == "All Samples Combined" or not selected_sample:
                fields = sorted(self.trajectories_df['field'].unique().tolist())
            else:
                sample_df = self.trajectories_df[self.trajectories_df['sample'] == selected_sample]
                fields = sorted(sample_df['field'].unique().tolist())

            self.field_selector.addItem("-- All Fields --")
            for f in fields:
                self.field_selector.addItem(f)

        self.field_selector.blockSignals(False)
        self.populate_field_images()
        self.refresh_all_plots()

    def populate_field_images(self):
        """Populates the Single Field Inspector image selector dropdown with live filtered plot choices and disk images."""
        selected_field = self.field_selector.currentText()

        self.combo_field_image.blockSignals(True)
        self.combo_field_image.clear()

        # Add Dynamic Live Reproduce Options at top of dropdown
        self.combo_field_image.addItem("📊 [Live Filtered] Diffusion Coefficient (D)", userData="LIVE_DIFF")
        self.combo_field_image.addItem("📊 [Live Filtered] Jump Distance (N frames)", userData="LIVE_JUMP")
        self.combo_field_image.addItem("📊 [Live Filtered] Stoichiometry Scatter (N vs D)", userData="LIVE_SCATTER")
        self.combo_field_image.addItem("📊 [Live Filtered] Trajectory Intensity Traces", userData="LIVE_TRACES")
        self.combo_field_image.addItem("📊 [Live Filtered] Stoichiometry Histogram", userData="LIVE_STOICH_HIST")
        self.combo_field_image.addItem("📊 [Live Filtered] Spot Intensity vs Frame", userData="LIVE_SPOTS")

        # Add Static Image Files from Disk (PNGs, TIFs, etc.)
        if selected_field and selected_field != "-- All Fields --" and selected_field in self.field_dirs_map:
            field_dir = self.field_dirs_map[selected_field]
            search_dirs = [field_dir, os.path.join(field_dir, "results")]

            img_files = []
            for s_dir in search_dirs:
                if os.path.exists(s_dir):
                    for ext in ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.PNG', '*.JPG', '*.JPEG', '*.TIF', '*.TIFF'):
                        img_files.extend(glob.glob(os.path.join(s_dir, ext)))

            unique_images = sorted(list(set(img_files)))

            if unique_images:
                def sort_key(filepath):
                    fname = os.path.basename(filepath).lower()
                    if 'avg' in fname:
                        return (0, fname)
                    return (1, fname)

                sorted_imgs = sorted(unique_images, key=sort_key)
                for fpath in sorted_imgs:
                    rel_name = os.path.basename(fpath)
                    parent_dir = os.path.basename(os.path.dirname(fpath))
                    if parent_dir == "results":
                        rel_name = f"results/{rel_name}"
                    self.combo_field_image.addItem(f"📁 Static File: {rel_name}", userData=fpath)

        self.combo_field_image.blockSignals(False)

    # ================= DIFFUSION & SNR CALCULATIONS =================

    def recalculate_and_refresh(self):
        self.clear_fits()

        if self.trajectories_df.empty:
            self.diffusion_summary_df = pd.DataFrame()
            self.lbl_stats.setText("No trajectories loaded.")
            self.chk_remove_neg.setText("Remove Negative D (D ≤ 0): 0")
            self.refresh_all_plots()
            return

        selected_sample = self.sample_selector.currentText()

        if selected_sample == "All Samples Combined" or not selected_sample:
            active_df = self.trajectories_df.copy()
        else:
            active_df = self.trajectories_df[self.trajectories_df['sample'] == selected_sample].copy()

        # Apply Frame Range Filter using spinbox values directly
        min_f = self.spin_min_frame.value()
        max_f = self.spin_max_frame.value()
        active_df = active_df[(active_df['frame'] >= min_f) & (active_df['frame'] <= max_f)]

        pixel_size = self.spin_pixel_size.value()
        dt_ms = self.spin_exposure.value()
        dt = (dt_ms / 1000.0) * (2.0 if self.chk_alex.isChecked() else 1.0)
        min_len = self.spin_min_len.value()

        results = []
        for track_id, track in active_df.groupby('global_track_id'):
            if len(track) < min_len:
                continue

            track_sorted = track.sort_values('frame')
            dx = np.diff(track_sorted['x'].values) * pixel_size
            dy = np.diff(track_sorted['y'].values) * pixel_size
            sq_disp = dx**2 + dy**2

            msd_1step = np.mean(sq_disp)
            D = msd_1step / (4.0 * dt) if dt > 0 else np.nan

            channel = track['channel'].iloc[0]
            sample = track['sample'].iloc[0]
            field = track['field'].iloc[0]

            intensity_col = 'integrated_intensity' if 'integrated_intensity' in track.columns else (
                'intensity' if 'intensity' in track.columns else (
                    'spot_intensity' if 'spot_intensity' in track.columns else None
                )
            )
            mean_intensity = track[intensity_col].mean() if intensity_col and intensity_col in track.columns else np.nan

            snr_col = 'snr' if 'snr' in track.columns else (
                'SNR' if 'SNR' in track.columns else (
                    'signal_to_noise' if 'signal_to_noise' in track.columns else None
                )
            )
            mean_snr = track[snr_col].mean() if snr_col and snr_col in track.columns else np.nan

            results.append({
                'global_track_id': track_id,
                'sample': sample,
                'field': field,
                'channel': channel,
                'D': D,
                'log_D': np.log10(D) if D > 0 and not np.isnan(D) else np.nan,
                'track_length': len(track),
                'mean_intensity': mean_intensity,
                'mean_snr': mean_snr
            })

        self.diffusion_summary_df = pd.DataFrame(results)

        num_negative = (self.diffusion_summary_df['D'] <= 0).sum() if not self.diffusion_summary_df.empty else 0
        self.chk_remove_neg.setText(f"Remove Negative D (D ≤ 0): {num_negative}")
        self.refresh_all_plots()

    def get_filtered_summary_df(self):
        """Filters trajectory summary DataFrame based on Channel, Min SNR, Negative D, Intensity (per channel), and Field filters."""
        if self.diffusion_summary_df.empty:
            return pd.DataFrame()

        df = self.diffusion_summary_df.copy()

        if self.channel_selector.isVisible():
            selected_chan = self.channel_selector.currentText()
            if selected_chan == "Left Channel (L)":
                df = df[df['channel'] == "Left Channel (L)"]
            elif selected_chan == "Right Channel (R)":
                df = df[df['channel'] == "Right Channel (R)"]

        # SNR Filter: Only apply if min_snr > 0 to avoid dropping NaN or valid tracks by default
        if 'mean_snr' in df.columns:
            min_snr = self.spin_min_snr.value()
            if min_snr > 0:
                df = df[df['mean_snr'].notna() & (df['mean_snr'] >= min_snr)]

        if self.chk_remove_neg.isChecked():
            df = df[df['D'] > 0]

        # iSingle Intensity Filter: Applied ONLY if checkbox is explicitly checked
        if self.chk_filter_intensity.isChecked():
            isingle_L = self.spin_isingle_L.value()
            low_bound_L, high_bound_L = 0.5 * isingle_L, 1.5 * isingle_L
            
            isingle_R = self.spin_isingle_R.value()
            low_bound_R, high_bound_R = 0.5 * isingle_R, 1.5 * isingle_R

            mask_L = (df['channel'] != "Right Channel (R)") & (df['mean_intensity'] >= low_bound_L) & (df['mean_intensity'] <= high_bound_L)
            mask_R = (df['channel'] == "Right Channel (R)") & (df['mean_intensity'] >= low_bound_R) & (df['mean_intensity'] <= high_bound_R)

            df = df[mask_L | mask_R]

        selected_field = self.field_selector.currentText()
        if selected_field and selected_field != "-- All Fields --":
            df = df[df['field'] == selected_field]

        return df

    # ================= PLOTTING LOGIC =================

    def refresh_all_plots(self):
        summary_df = self.get_filtered_summary_df()

        if not self.diffusion_summary_df.empty:
            total_raw = self.trajectories_df['global_track_id'].nunique() if not self.trajectories_df.empty else 0
            valid_tracks = len(self.diffusion_summary_df)
            displayed_tracks = len(summary_df)
            neg_count = (self.diffusion_summary_df['D'] <= 0).sum()

            field_label = self.field_selector.currentText() or "-- All Fields --"
            chan_label = self.channel_selector.currentText() if self.channel_selector.isVisible() else "Single Channel"
            
            if self.chk_filter_intensity.isChecked():
                if self.channel_selector.isVisible() and self.spin_isingle_R.isVisible():
                    intensity_filter_label = f"Active [L: {self.spin_isingle_L.value():.1f}, R: {self.spin_isingle_R.value():.1f}]"
                else:
                    intensity_filter_label = f"Active [{self.spin_isingle_L.value():.1f}]"
            else:
                intensity_filter_label = "Off"

            snr_val = self.spin_min_snr.value()
            snr_filter_label = f"≥ {snr_val:.1f}" if snr_val > 0 else "Off"
            frame_range_label = f"{self.spin_min_frame.value()}–{self.spin_max_frame.value()}"

            self.lbl_stats.setText(
                f"<b>Scope:</b> {self.sample_selector.currentText()}<br>"
                f"<b>Channel Filter:</b> {chan_label}<br>"
                f"<b>Field:</b> {field_label}<br>"
                f"<b>Total Raw Tracks:</b> {total_raw}<br>"
                f"<b>Frame Range Filter:</b> {frame_range_label}<br>"
                f"<b>Valid Length Tracks:</b> {valid_tracks}<br>"
                f"<b>SNR Filter:</b> {snr_filter_label}<br>"
                f"<b>Negative/Zero D Tracks:</b> {neg_count}<br>"
                f"<b>Intensity Filter:</b> {intensity_filter_label}<br>"
                f"<b>Displayed Tracks:</b> {displayed_tracks}"
            )

        self.plot_diffusion_histogram(summary_df)
        self.plot_jump_distance_histogram(summary_df)
        self.plot_spatial_map(summary_df)
        self.update_single_field_plot()
        self.plot_intensity_distribution(summary_df)
        self.plot_msd_vs_timelag(summary_df)

    def plot_diffusion_histogram(self, summary_df):
        ax = self.diff_canvas.axes
        ax.clear()

        if summary_df.empty:
            ax.grid(False)
            self.diff_canvas.draw()
            return

        selected_chan = self.channel_selector.currentText() if self.channel_selector.isVisible() else "Single Channel"
        use_log = self.chk_log_scale.isChecked()

        val_col = 'log_D' if use_log else 'D'
        x_label = r"$\log_{10}(D \ [\mu\mathrm{m}^2/\mathrm{s}])$" if use_log else r"$D \ [\mu\mathrm{m}^2/\mathrm{s}]$"

        valid_vals = summary_df[val_col].dropna()
        if valid_vals.empty:
            ax.grid(False)
            self.diff_canvas.draw()
            return

        if use_log:
            bins = self.spin_bins.value()
        else:
            bin_width = self.spin_bin_size.value()
            if bin_width > 0:
                d_min = valid_vals.min()
                d_max = valid_vals.max()
                bins = np.arange(d_min, d_max + bin_width, bin_width)
                if len(bins) < 2:
                    bins = self.spin_bins.value()
            else:
                bins = self.spin_bins.value()

        is_overlay = self.channel_selector.isVisible() and selected_chan == "Both Channels (Overlay)" and summary_df['channel'].nunique() > 1

        if is_overlay:
            df_L = summary_df[summary_df['channel'] == "Left Channel (L)"]
            df_R = summary_df[summary_df['channel'] == "Right Channel (R)"]

            if not df_L.empty:
                ax.hist(
                    df_L[val_col].dropna().values, bins=bins, color='#008080', alpha=0.45,
                    label=f"Left Channel (L) (n={len(df_L)})", edgecolor='black', linewidth=0.5
                )
                if self.chk_kde.isChecked() and len(df_L) > 1:
                    data_L = df_L[val_col].dropna().values
                    x_grid_L = np.linspace(data_L.min(), data_L.max(), 300)
                    kde_L = gaussian_kde(data_L)
                    b_w_L = np.mean(np.diff(bins)) if isinstance(bins, np.ndarray) and len(bins) > 1 else 1.0
                    ax.plot(x_grid_L, kde_L(x_grid_L) * len(data_L) * b_w_L, color='#004D4D', linewidth=2, label="Left KDE")

            if not df_R.empty:
                ax.hist(
                    df_R[val_col].dropna().values, bins=bins, color='#D81B60', alpha=0.45,
                    label=f"Right Channel (R) (n={len(df_R)})", edgecolor='black', linewidth=0.5
                )
                if self.chk_kde.isChecked() and len(df_R) > 1:
                    data_R = df_R[val_col].dropna().values
                    x_grid_R = np.linspace(data_R.min(), data_R.max(), 300)
                    kde_R = gaussian_kde(data_R)
                    b_w_R = np.mean(np.diff(bins)) if isinstance(bins, np.ndarray) and len(bins) > 1 else 1.0
                    ax.plot(x_grid_R, kde_R(x_grid_R) * len(data_R) * b_w_R, color='#880e4f', linewidth=2, label="Right KDE")
        else:
            ax.hist(
                summary_df[val_col].dropna().values, bins=bins, color='#008080', alpha=0.45,
                label=f"Data [{selected_chan}] (n={len(summary_df)})", edgecolor='black', linewidth=0.5
            )

            if self.chk_kde.isChecked() and len(summary_df) > 1:
                data = summary_df[val_col].dropna().values
                x_grid = np.linspace(data.min(), data.max(), 300)
                kde = gaussian_kde(data)
                b_width = np.mean(np.diff(bins)) if isinstance(bins, np.ndarray) and len(bins) > 1 else 1.0
                ax.plot(x_grid, kde(x_grid) * len(data) * b_width, color='#004D4D', linewidth=2, label="KDE Distribution")

        if self.diff_fit_results and not use_log:
            d_clean = summary_df['D'].dropna().values
            d_clean = d_clean[d_clean > 0]

            if len(d_clean) > 0:
                x_grid = np.linspace(d_clean.min(), d_clean.max(), 350)
                bin_scale = np.diff(bins)[0] * len(d_clean) if isinstance(bins, np.ndarray) and len(bins) > 1 else len(d_clean)
                best_k = self.best_diff_model['components'] if self.best_diff_model else None

                show_non_best = self.chk_show_non_best.isChecked()
                show_subcomps = self.chk_show_subcomponents.isChecked()

                styles = ['--', '-.', ':', '-']
                colors = ['#1f77b4', '#d62728', '#9467bd', '#2ca02c']
                comp_colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2']

                for idx, fit in enumerate(self.diff_fit_results):
                    k = fit['components']
                    bic = fit['bic']
                    is_best = (k == best_k)

                    if not is_best and not show_non_best:
                        continue

                    color = '#D81B60' if is_best else colors[idx % len(colors)]
                    lw = 2.5 if is_best else 1.2
                    ls = '-' if is_best else styles[idx % len(styles)]
                    label = f"{k}-Comp Fit (BIC: {bic:.1f})" + (" ★ BEST" if is_best else "")

                    y_fit = fit['pdf_func'](x_grid) * bin_scale
                    ax.plot(x_grid, y_fit, color=color, linestyle=ls, linewidth=lw, label=label)

                    if is_best and show_subcomps and k > 1:
                        params = fit.get('params', [])
                        sub_pdfs = fit.get('component_pdfs', fit.get('comp_pdfs', []))

                        for c_idx, p in enumerate(params):
                            w = p.get('weight', 1.0)
                            mean_d = p.get('mean_D', p.get('D', 0.0))
                            c_color = comp_colors[c_idx % len(comp_colors)]
                            c_label = f"  └ State {c_idx+1}: D={mean_d:.3f}, {w*100:.1f}%"

                            y_sub = None
                            if c_idx < len(sub_pdfs) and callable(sub_pdfs[c_idx]):
                                try:
                                    y_sub = sub_pdfs[c_idx](x_grid) * bin_scale
                                except Exception:
                                    y_sub = None

                            if y_sub is None and 'pdf_func' in p and callable(p['pdf_func']):
                                try:
                                    y_sub = p['pdf_func'](x_grid) * bin_scale
                                except Exception:
                                    y_sub = None

                            if y_sub is None:
                                n = p.get('shape', p.get('n', 4))
                                if mean_d > 0 and n > 0:
                                    scale = mean_d / n
                                    y_sub = w * gamma.pdf(x_grid, a=n, scale=scale) * bin_scale

                            if y_sub is not None:
                                ax.plot(x_grid, y_sub, color=c_color, linestyle=':', linewidth=1.8, label=c_label)

        selected_field = self.field_selector.currentText()
        field_suffix = f" | Field: {selected_field}" if selected_field and selected_field != "-- All Fields --" else ""

        ax.set_title(f"Diffusion Coefficient Distribution — {self.sample_selector.currentText()} [{selected_chan}]{field_suffix}", fontsize=11)
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel("Trajectory Count", fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(False)

        self.diff_canvas.draw()

    def plot_jump_distance_histogram(self, summary_df):
        ax = self.jump_canvas.axes
        ax.clear()

        if self.trajectories_df.empty or summary_df.empty:
            ax.grid(False)
            self.jump_canvas.draw()
            return

        lag_N = self.spin_jump_lag.value()
        dt_ms = self.spin_exposure.value()
        is_alex = self.chk_alex.isChecked()
        dt_sec = (dt_ms / 1000.0) * (2.0 if is_alex else 1.0)
        time_lag_ms = lag_N * dt_sec * 1000.0
        bins_count = self.spin_bins_jump.value()
        selected_chan = self.channel_selector.currentText() if self.channel_selector.isVisible() else "Single Channel"

        is_overlay = self.channel_selector.isVisible() and selected_chan == "Both Channels (Overlay)" and summary_df['channel'].nunique() > 1

        if is_overlay:
            summary_df_L = summary_df[summary_df['channel'] == "Left Channel (L)"]
            summary_df_R = summary_df[summary_df['channel'] == "Right Channel (R)"]

            jumps_L = self.extract_jump_distances(summary_df_L)
            jumps_R = self.extract_jump_distances(summary_df_R)

            if len(jumps_L) == 0 and len(jumps_R) == 0:
                ax.grid(False)
                self.jump_canvas.draw()
                return

            all_jumps = np.concatenate([jumps_L, jumps_R]) if len(jumps_L) > 0 and len(jumps_R) > 0 else (jumps_L if len(jumps_L) > 0 else jumps_R)
            bin_edges = np.linspace(all_jumps.min(), all_jumps.max(), bins_count + 1)

            if len(jumps_L) > 0:
                ax.hist(
                    jumps_L, bins=bin_edges, color='#008080', alpha=0.45,
                    label=f"Left Channel (L) (N={len(jumps_L)} steps)", edgecolor='black', linewidth=0.5
                )
                if self.chk_kde_jump.isChecked() and len(jumps_L) > 1:
                    x_grid_L = np.linspace(jumps_L.min(), jumps_L.max(), 300)
                    kde_L = gaussian_kde(jumps_L)
                    b_w_L = np.mean(np.diff(bin_edges))
                    ax.plot(x_grid_L, kde_L(x_grid_L) * len(jumps_L) * b_w_L, color='#004D4D', linewidth=2, label="Left KDE")

            if len(jumps_R) > 0:
                ax.hist(
                    jumps_R, bins=bin_edges, color='#D81B60', alpha=0.45,
                    label=f"Right Channel (R) (N={len(jumps_R)} steps)", edgecolor='black', linewidth=0.5
                )
                if self.chk_kde_jump.isChecked() and len(jumps_R) > 1:
                    x_grid_R = np.linspace(jumps_R.min(), jumps_R.max(), 300)
                    kde_R = gaussian_kde(jumps_R)
                    b_w_R = np.mean(np.diff(bin_edges))
                    ax.plot(x_grid_R, kde_R(x_grid_R) * len(jumps_R) * b_w_R, color='#880e4f', linewidth=2, label="Right KDE")

            jumps = all_jumps
        else:
            jumps = self.extract_jump_distances(summary_df)
            if len(jumps) == 0:
                ax.grid(False)
                self.jump_canvas.draw()
                return

            counts, bin_edges, _ = ax.hist(
                jumps, bins=bins_count, color='#008080', alpha=0.45,
                label=f"Jump Data [{selected_chan}] (N={len(jumps)} steps)", edgecolor='black', linewidth=0.5
            )

            if self.chk_kde_jump.isChecked() and len(jumps) > 1:
                x_grid = np.linspace(jumps.min(), jumps.max(), 300)
                kde = gaussian_kde(jumps)
                b_width = np.mean(np.diff(bin_edges)) if len(bin_edges) > 1 else 1.0
                ax.plot(x_grid, kde(x_grid) * len(jumps) * b_width, color='#004D4D', linewidth=2, label="Jump KDE")

        if self.jump_fit_results and len(jumps) > 0:
            x_grid = np.linspace(jumps.min(), jumps.max(), 350)
            bin_scale = np.diff(bin_edges)[0] * len(jumps) if len(bin_edges) > 1 else len(jumps)
            best_k = self.best_jump_model['components'] if self.best_jump_model else None

            show_non_best = self.chk_show_non_best_jump.isChecked()
            show_subcomps = self.chk_show_subcomponents_jump.isChecked()

            styles = ['--', '-.', ':', '-']
            colors = ['#1f77b4', '#d62728', '#9467bd', '#2ca02c']
            comp_colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2']

            dt_lag = dt_sec * lag_N
            sigma_loc = self.loc_precision if self.loc_precision is not None else 0.0

            for idx, fit in enumerate(self.jump_fit_results):
                k = fit['components']
                bic = fit['bic']
                is_best = (k == best_k)

                if not is_best and not show_non_best:
                    continue

                color = '#D81B60' if is_best else colors[idx % len(colors)]
                lw = 2.5 if is_best else 1.2
                ls = '-' if is_best else styles[idx % len(styles)]
                label = f"Best Fit ({k}-Comp, BIC: {bic:.1f}) ★ BEST" if is_best else f"{k}-Comp Fit (BIC: {bic:.1f})"

                y_fit = fit['pdf_func'](x_grid) * bin_scale
                ax.plot(x_grid, y_fit, color=color, linestyle=ls, linewidth=lw, label=label)

                if is_best and show_subcomps and k > 1:
                    params = fit.get('params', [])
                    sub_pdfs = fit.get('component_pdfs', fit.get('comp_pdfs', []))

                    for c_idx, p in enumerate(params):
                        w = p.get('weight', 1.0)
                        d_val = p.get('D', p.get('mean_D', 0.0))
                        c_color = comp_colors[c_idx % len(comp_colors)]
                        c_label = f"  └ State {c_idx+1}: D={d_val:.3f}, {w*100:.1f}%"

                        y_sub = None
                        if c_idx < len(sub_pdfs) and callable(sub_pdfs[c_idx]):
                            try:
                                y_sub = sub_pdfs[c_idx](x_grid) * bin_scale
                            except Exception:
                                y_sub = None

                        if y_sub is None and 'pdf_func' in p and callable(p['pdf_func']):
                            try:
                                y_sub = p['pdf_func'](x_grid) * bin_scale
                            except Exception:
                                y_sub = None

                        if y_sub is None:
                            sigma_sq = 2.0 * d_val * dt_lag + 2.0 * (sigma_loc ** 2)
                            if sigma_sq > 0:
                                sigma_ray = np.sqrt(sigma_sq / 2.0)
                                y_sub = w * rayleigh.pdf(x_grid, scale=sigma_ray) * bin_scale

                        if y_sub is not None:
                            ax.plot(x_grid, y_sub, color=c_color, linestyle=':', linewidth=1.8, label=c_label)

        selected_field = self.field_selector.currentText()
        field_suffix = f" | Field: {selected_field}" if selected_field and selected_field != "-- All Fields --" else ""

        ax.set_title(f"Jump Distance Distribution ($\Delta t = {time_lag_ms:.1f}\mathrm{{ms}}$, {lag_N} frames) — {self.sample_selector.currentText()} [{selected_chan}]{field_suffix}", fontsize=11)
        ax.set_xlabel(r"Jump Distance $r$ $[\mu\mathrm{m}]$", fontsize=11)
        ax.set_ylabel("Jump Frequency / Count", fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(False)

        self.jump_canvas.draw()

    def plot_spatial_map(self, summary_df):
        fig = self.spatial_canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)

        if self.trajectories_df.empty or summary_df.empty:
            ax.grid(False)
            self.spatial_canvas.draw()
            return

        selected_field = self.field_selector.currentText()
        bg_choice = self.combo_spatial_bg.currentText()
        img_data = None

        tmat, tmat_path = self.find_transformation_matrix(selected_field)

        if selected_field and selected_field != "-- All Fields --" and selected_field in self.field_dirs_map:
            field_dir = self.field_dirs_map[selected_field]
            search_dirs = [field_dir, os.path.join(field_dir, "results")]

            def find_image_file(patterns):
                for s_dir in search_dirs:
                    if os.path.exists(s_dir):
                        for pat in patterns:
                            matches = glob.glob(os.path.join(s_dir, pat))
                            if matches:
                                return matches[0]
                return None

            if bg_choice == "L_channel Average (L_avg.tif)":
                l_file = find_image_file(["*L_avg*.[tT][iI][fF]*", "l_avg.[tT][iI][fF]*"])
                if l_file:
                    try:
                        img_data = np.array(Image.open(l_file))
                    except Exception as e:
                        print(f"Error loading {l_file}: {e}")

            elif bg_choice == "R_channel Average (R_avg.tif)":
                r_file = find_image_file(["*R_avg*.[tT][iI][fF]*", "r_avg.[tT][iI][fF]*"])
                if r_file:
                    try:
                        img_data = np.array(Image.open(r_file)).astype(np.float64)
                        if self.chk_spatial_transform_R.isChecked() and tmat is not None and StackReg is not None:
                            try:
                                sr = StackReg(StackReg.RIGID_BODY)
                                bkg = np.mean(img_data)
                                img_data = sr.transform(img_data, tmat)
                                img_data[img_data == 0] = bkg
                                img_data = img_data - bkg
                                img_data[img_data < 0] = 0
                            except Exception as e:
                                print(f"StackReg transform failed: {e}")
                    except Exception as e:
                        print(f"Error loading {r_file}: {e}")

            elif bg_choice == "Merged Left and Right (Merge)":
                l_file = find_image_file(["*L_avg*.[tT][iI][fF]*", "l_avg.[tT][iI][fF]*"])
                r_file = find_image_file(["*R_avg*.[tT][iI][fF]*", "r_avg.[tT][iI][fF]*"])

                if l_file and r_file:
                    try:
                        img_L = np.array(Image.open(l_file)).astype(np.float64)
                        img_R = np.array(Image.open(r_file)).astype(np.float64)

                        img_L = img_L - np.mean(img_L)
                        img_L[img_L < 0] = 0

                        if self.chk_spatial_transform_R.isChecked() and tmat is not None and StackReg is not None:
                            try:
                                sr = StackReg(StackReg.RIGID_BODY)
                                bkg = np.mean(img_R)
                                img_R = sr.transform(img_R, tmat)
                                img_R[img_R == 0] = bkg
                                img_R = img_R - bkg
                                img_R[img_R < 0] = 0
                            except Exception as e:
                                print(f"StackReg transform failed: {e}")

                        max_R = np.max(img_R) if np.max(img_R) > 0 else 1.0
                        max_L = np.max(img_L) if np.max(img_L) > 0 else 1.0

                        img_R_norm = (np.clip(img_R, 0, None) / max_R * 255).astype(np.uint8)
                        img_L_norm = (np.clip(img_L, 0, None) / max_L * 255).astype(np.uint8)

                        composite_merge = np.zeros((img_L.shape[0], img_L.shape[1], 3), dtype=np.uint8)
                        composite_merge[:, :, 0] = img_R_norm  # Red Channel
                        composite_merge[:, :, 1] = img_L_norm  # Green Channel
                        img_data = composite_merge
                    except Exception as e:
                        print(f"Error merging L and R images: {e}")

            elif bg_choice == "Brightfield Image (BF.tif)":
                bf_file = find_image_file(["*BF*.[tT][iI][fF]*", "bf.[tT][iI][fF]*"])
                if bf_file:
                    try:
                        img_data = np.array(Image.open(bf_file))
                    except Exception as e:
                        print(f"Error loading {bf_file}: {e}")

        if img_data is not None:
            if img_data.ndim == 3 and img_data.shape[0] < 10 and img_data.shape[2] != 3:
                img_data = img_data[0]
            if img_data.ndim == 2:
                ax.imshow(img_data, cmap='gray', origin='lower')
            else:
                ax.imshow(img_data, origin='lower')

        allowed_ids = set(summary_df['global_track_id'].unique())
        df = self.trajectories_df[self.trajectories_df['global_track_id'].isin(allowed_ids)]

        min_f = self.spin_min_frame.value()
        max_f = self.spin_max_frame.value()
        df = df[(df['frame'] >= min_f) & (df['frame'] <= max_f)]

        show_L = self.chk_spatial_show_L.isChecked()
        show_R = self.chk_spatial_show_R.isChecked()
        transform_R = self.chk_spatial_transform_R.isChecked()

        entity_mode = self.combo_spatial_entity.currentText() if hasattr(self, 'combo_spatial_entity') else "Map Full Trajectories"
        color_mode = self.combo_spatial_color_mode.currentText() if hasattr(self, 'combo_spatial_color_mode') else "Channel Color (Default)"
        selected_chan = self.channel_selector.currentText() if self.channel_selector.isVisible() else "Single Channel"

        try:
            cmap = matplotlib.colormaps['viridis']
        except AttributeError:
            import matplotlib.cm as cm
            cmap = cm.get_cmap('viridis')

        # FIXED COLORING BY DIFFUSION COEFFICIENT AND SINGLE JUMPS
        if color_mode == "Color by Diffusion Coefficient (D)":
            d_map = dict(zip(summary_df['global_track_id'], summary_df['D']))
            segments = []
            segment_vals = []

            for track_id, track in df.groupby('global_track_id'):
                channel = track['channel'].iloc[0]
                if channel == "Left Channel (L)" and not show_L:
                    continue
                if channel == "Right Channel (R)" and not show_R:
                    continue

                d_val = d_map.get(track_id, np.nan)
                if np.isnan(d_val) or d_val <= 0:
                    continue

                track_sorted = track.sort_values('frame')
                x_coords = track_sorted['x'].values
                y_coords = track_sorted['y'].values

                if channel == "Right Channel (R)" and transform_R and tmat is not None:
                    x_coords, y_coords = self.transform_track_coordinates(x_coords, y_coords, tmat)

                pts = np.column_stack([x_coords, y_coords])

                if entity_mode == "Map Single Jump Steps":
                    for i in range(len(pts) - 1):
                        segments.append(pts[i:i+2])
                        segment_vals.append(d_val)
                else:  # Full Trajectories
                    if len(pts) > 1:
                        for i in range(len(pts) - 1):
                            segments.append(pts[i:i+2])
                            segment_vals.append(d_val)

            if segments:
                segment_vals = np.array(segment_vals)
                vmin = np.percentile(segment_vals, 2) if len(segment_vals) > 10 else segment_vals.min()
                vmax = np.percentile(segment_vals, 98) if len(segment_vals) > 10 else segment_vals.max()
                if vmin == vmax:
                    vmax = vmin + 1e-5

                norm = Normalize(vmin=vmin, vmax=vmax)
                lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=1.2, alpha=0.85)
                lc.set_array(segment_vals)
                ax.add_collection(lc)

                cbar = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label(r"Diffusion Coefficient $D$ ($\mu\mathrm{m}^2/\mathrm{s}$)", fontsize=10)

        elif color_mode == "Color by Jump Distance (N frames)":
            pixel_size = self.spin_pixel_size.value()
            lag_N = self.spin_spatial_jump_lag.value() if hasattr(self, 'spin_spatial_jump_lag') else 1

            segments = []
            segment_vals = []

            for track_id, track in df.groupby('global_track_id'):
                channel = track['channel'].iloc[0]
                if channel == "Left Channel (L)" and not show_L:
                    continue
                if channel == "Right Channel (R)" and not show_R:
                    continue

                track_sorted = track.sort_values('frame')
                frames = track_sorted['frame'].values
                x_coords = track_sorted['x'].values
                y_coords = track_sorted['y'].values

                if channel == "Right Channel (R)" and transform_R and tmat is not None:
                    x_coords, y_coords = self.transform_track_coordinates(x_coords, y_coords, tmat)

                if len(frames) <= lag_N:
                    continue

                pts = np.column_stack([x_coords, y_coords])

                for i in range(len(frames) - lag_N):
                    if frames[i + lag_N] - frames[i] == lag_N:
                        dx = (x_coords[i + lag_N] - x_coords[i]) * pixel_size
                        dy = (y_coords[i + lag_N] - y_coords[i]) * pixel_size
                        dist = np.sqrt(dx**2 + dy**2)

                        if entity_mode == "Map Single Jump Steps":
                            segments.append(np.array([pts[i], pts[i + lag_N]]))
                        else:
                            segments.append(pts[i:i + lag_N + 1])
                        segment_vals.append(dist)

            if segments:
                segment_vals = np.array(segment_vals)
                vmin = np.percentile(segment_vals, 2) if len(segment_vals) > 10 else segment_vals.min()
                vmax = np.percentile(segment_vals, 98) if len(segment_vals) > 10 else segment_vals.max()
                if vmin == vmax:
                    vmax = vmin + 1e-5

                norm = Normalize(vmin=vmin, vmax=vmax)
                lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=1.2, alpha=0.85)
                lc.set_array(segment_vals)
                ax.add_collection(lc)

                cbar = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label(f"Jump Distance $r$ ($\mu\mathrm{{m}}$, N={lag_N} frames)", fontsize=10)

        else:  # Channel Color (Default)
            for track_id, track in df.groupby('global_track_id'):
                channel = track['channel'].iloc[0]

                if channel == "Left Channel (L)" and not show_L:
                    continue
                if channel == "Right Channel (R)" and not show_R:
                    continue

                track_sorted = track.sort_values('frame')
                x_coords = track_sorted['x'].values
                y_coords = track_sorted['y'].values

                if channel == "Right Channel (R)" and transform_R and tmat is not None:
                    x_coords, y_coords = self.transform_track_coordinates(x_coords, y_coords, tmat)

                color = '#00FFFF' if channel == "Left Channel (L)" else '#FF00FF' if img_data is not None else ('#008080' if channel == "Left Channel (L)" else '#D81B60')
                
                if entity_mode == "Map Single Jump Steps":
                    for i in range(len(x_coords) - 1):
                        ax.plot(x_coords[i:i+2], y_coords[i:i+2], color=color, alpha=0.6, linewidth=0.8)
                else:
                    ax.plot(x_coords, y_coords, color=color, alpha=0.6, linewidth=0.8)

        field_suffix = f" ({selected_field})" if selected_field and selected_field != "-- All Fields --" else ""
        tmat_info = f" [tmat: {os.path.basename(tmat_path)}]" if (transform_R and tmat_path) else ""
        
        ax.set_title(f"Spatial Map — {self.sample_selector.currentText()} [{selected_chan}]{field_suffix}{tmat_info}")
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(False)
        self.spatial_canvas.draw()

    # ================= SINGLE FIELD INSPECTOR & LIVE PLOT REPRODUCTION =================

    def update_single_field_plot(self):
        """Displays either microscopy images, saved PySTACHIO PNGs, or live reproduced analysis plots using active filters."""
        ax = self.field_canvas.axes
        ax.clear()

        selected_field = self.field_selector.currentText()

        if not selected_field or selected_field == "-- All Fields --":
            ax.text(0.5, 0.5, "Select a specific field from the 'Field Selection' dropdown to view images & outputs.",
                    ha='center', va='center', transform=ax.transAxes, fontsize=11)
            ax.grid(False)
            self.field_canvas.draw()
            return

        selected_item = self.combo_field_image.currentText()
        item_data = self.combo_field_image.currentData()

        # Check if user explicitly selected a Live Dynamic option
        is_live = isinstance(item_data, str) and item_data.startswith("LIVE_")

        summary_df = self.get_filtered_summary_df()
        field_summary = summary_df[summary_df['field'] == selected_field] if not summary_df.empty else pd.DataFrame()

        if not self.trajectories_df.empty:
            allowed_ids = set(field_summary['global_track_id'].unique()) if not field_summary.empty else set()
            df_field = self.trajectories_df[(self.trajectories_df['field'] == selected_field) & (self.trajectories_df['global_track_id'].isin(allowed_ids))]
            min_f = self.spin_min_frame.value()
            max_f = self.spin_max_frame.value()
            df_field = df_field[(df_field['frame'] >= min_f) & (df_field['frame'] <= max_f)]
        else:
            df_field = pd.DataFrame()

        selected_chan = self.channel_selector.currentText() if self.channel_selector.isVisible() else "Single Channel"

        if is_live:
            # Render live dynamic interactive plot
            if item_data == "LIVE_DIFF":
                self._render_live_diff_coeff(ax, field_summary, selected_field, selected_chan)
            elif item_data == "LIVE_JUMP":
                self._render_live_jump_distance(ax, field_summary, df_field, selected_field, selected_chan)
            elif item_data == "LIVE_SCATTER":
                self._render_live_stoich_scatter(ax, field_summary, selected_field, selected_chan)
            elif item_data == "LIVE_TRACES":
                self._render_live_trajectory_traces(ax, df_field, selected_field, selected_chan)
            elif item_data == "LIVE_STOICH_HIST":
                self._render_live_stoich_hist(ax, field_summary, selected_field, selected_chan)
            elif item_data == "LIVE_SPOTS":
                self._render_live_spot_intensity(ax, df_field, selected_field, selected_chan)
        else:
            # Load static image file directly from disk (TIFF or static PySTACHIO PNG)
            img_path = item_data
            if not img_path or not os.path.exists(img_path):
                ax.text(0.5, 0.5, f"No image selected or file not found for field '{selected_field}'.",
                        ha='center', va='center', transform=ax.transAxes, fontsize=11)
                ax.grid(False)
                self.field_canvas.draw()
                return

            try:
                img_data = np.array(Image.open(img_path))
            except Exception as e:
                ax.text(0.5, 0.5, f"Error loading image:\n{e}", ha='center', va='center', transform=ax.transAxes, fontsize=11)
                ax.grid(False)
                self.field_canvas.draw()
                return

            filename = os.path.basename(img_path)
            is_plot_png = filename.lower().endswith(('.png', '.jpg', '.jpeg'))

            if img_data.ndim == 3 and img_data.shape[0] < 10:
                img_data = img_data[0]

            if is_plot_png:
                ax.imshow(img_data, origin='upper', aspect='auto')
                ax.axis('off')
            else:
                ax.imshow(img_data, cmap='gray', origin='lower', aspect='equal')
                ax.set_xlabel("X (px)")
                ax.set_ylabel("Y (px)")
                ax.grid(False)

                if self.chk_overlay_field_tracks.isChecked() and not df_field.empty:
                    for track_id, track in df_field.groupby('global_track_id'):
                        channel = track['channel'].iloc[0]
                        color = '#00FFFF' if channel == "Left Channel (L)" else '#FF00FF'
                        ax.plot(track['x'], track['y'], color=color, alpha=0.8, linewidth=1.0)

            ax.set_title(f"Single Field Inspector: {selected_field} | {filename} [{selected_chan}]", fontsize=10)

        self.field_canvas.draw()

    # --- HELPER ROUTINES FOR LIVE PYSTACHIO PLOT REPRODUCTION ---

    def _render_live_diff_coeff(self, ax, field_summary, field_name, chan_label):
        if field_summary.empty or field_summary['D'].dropna().empty:
            ax.text(0.5, 0.5, f"No valid tracks match filter criteria in {field_name}", ha='center', va='center', transform=ax.transAxes)
            ax.grid(False)
            return

        d_vals = field_summary['D'].dropna().values
        d_clean = d_vals[d_vals > 0] if self.chk_remove_neg.isChecked() else d_vals

        bins_count = self.spin_bins.value()
        ax.hist(d_clean, bins=bins_count, color='#008080', alpha=0.55, edgecolor='black', linewidth=0.6, label=f"Filtered Tracks (n={len(d_clean)})")

        if len(d_clean) > 1 and self.chk_kde.isChecked():
            kde = gaussian_kde(d_clean)
            x_grid = np.linspace(d_clean.min(), d_clean.max(), 300)
            b_w = (d_clean.max() - d_clean.min()) / bins_count
            ax.plot(x_grid, kde(x_grid) * len(d_clean) * b_w, color='#004D4D', linewidth=2, label="KDE Fit")

        mean_d = np.mean(d_clean)
        ax.axvline(mean_d, color='red', linestyle='--', linewidth=1.5, label=f"Mean D: {mean_d:.3f} μm²/s")

        ax.set_title(f"Live Reproduced: Diffusion Coefficient — {field_name} [{chan_label}]", fontsize=10)
        ax.set_xlabel(r"Diffusion Coefficient $D$ $[\mu\mathrm{m}^2/\mathrm{s}]$", fontsize=10)
        ax.set_ylabel("Track Count", fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(False)

    def _render_live_jump_distance(self, ax, field_summary, df_field, field_name, chan_label):
        if df_field.empty or field_summary.empty:
            ax.text(0.5, 0.5, f"No valid tracks match filter criteria in {field_name}", ha='center', va='center', transform=ax.transAxes)
            ax.grid(False)
            return

        jumps = self.extract_jump_distances(field_summary)
        if len(jumps) == 0:
            ax.text(0.5, 0.5, f"No jump distance data for filtered tracks in {field_name}", ha='center', va='center', transform=ax.transAxes)
            ax.grid(False)
            return

        lag_N = self.spin_spatial_jump_lag.value() if hasattr(self, 'spin_spatial_jump_lag') else self.spin_jump_lag.value()
        bins_count = self.spin_bins_jump.value()
        ax.hist(jumps, bins=bins_count, color='#008080', alpha=0.55, edgecolor='black', linewidth=0.6, label=f"Jump Steps (n={len(jumps)})")

        if len(jumps) > 1 and self.chk_kde_jump.isChecked():
            kde = gaussian_kde(jumps)
            x_grid = np.linspace(jumps.min(), jumps.max(), 300)
            b_w = (jumps.max() - jumps.min()) / bins_count
            ax.plot(x_grid, kde(x_grid) * len(jumps) * b_w, color='#004D4D', linewidth=2, label="KDE Fit")

        mean_j = np.mean(jumps)
        ax.axvline(mean_j, color='red', linestyle='--', linewidth=1.5, label=f"Mean Jump: {mean_j:.3f} μm")

        ax.set_title(f"Live Reproduced: Jump Distance (N={lag_N} frames) — {field_name} [{chan_label}]", fontsize=10)
        ax.set_xlabel(r"Jump Distance $r$ $[\mu\mathrm{m}]$", fontsize=10)
        ax.set_ylabel("Step Count", fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(False)

    def _render_live_stoich_scatter(self, ax, field_summary, field_name, chan_label):
        if field_summary.empty or field_summary['mean_intensity'].dropna().empty:
            ax.text(0.5, 0.5, f"No valid tracks match filter criteria in {field_name}", ha='center', va='center', transform=ax.transAxes)
            ax.grid(False)
            return

        isingle = self.spin_isingle_L.value() if "Right" not in chan_label else self.spin_isingle_R.value()
        stoich = field_summary['mean_intensity'] / isingle
        d_vals = field_summary['D']

        ax.scatter(d_vals, stoich, color='#008080', alpha=0.65, s=25, edgecolors='black', linewidth=0.3, label=f"Tracks (n={len(field_summary)})")

        max_stoich = int(np.ceil(stoich.max())) if not stoich.empty and not np.isnan(stoich.max()) else 5
        for i in range(1, min(max_stoich + 1, 10)):
            ax.axhline(i, color='gray', linestyle=':', alpha=0.6, linewidth=0.8)

        ax.set_title(f"Live Reproduced: Stoichiometry vs Diffusion — {field_name} [{chan_label}]", fontsize=10)
        ax.set_xlabel(r"Diffusion Coefficient $D$ $[\mu\mathrm{m}^2/\mathrm{s}]$", fontsize=10)
        ax.set_ylabel(r"Stoichiometry ($I_{\mathrm{mean}} / I_{\mathrm{single}}$)", fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(False)

    def _render_live_trajectory_traces(self, ax, df_field, field_name, chan_label):
        if df_field.empty:
            ax.text(0.5, 0.5, f"No spot data for filtered trajectories in {field_name}", ha='center', va='center', transform=ax.transAxes)
            ax.grid(False)
            return

        intensity_col = 'integrated_intensity' if 'integrated_intensity' in df_field.columns else (
            'intensity' if 'intensity' in df_field.columns else 'spot_intensity'
        )

        isingle = self.spin_isingle_L.value() if "Right" not in chan_label else self.spin_isingle_R.value()

        for track_id, track in df_field.groupby('global_track_id'):
            track_sorted = track.sort_values('frame')
            ax.plot(track_sorted['frame'], track_sorted[intensity_col], alpha=0.6, linewidth=1.1)

        ax.axhline(isingle, color='black', linestyle='--', linewidth=1.5, label=f"iSingle ({isingle:.1f} a.u.)")

        ax.set_title(f"Live Reproduced: Trajectory Intensity Traces — {field_name} [{chan_label}]", fontsize=10)
        ax.set_xlabel("Frame Number", fontsize=10)
        ax.set_ylabel("Integrated Intensity (a.u.)", fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(False)

    def _render_live_stoich_hist(self, ax, field_summary, field_name, chan_label):
        if field_summary.empty or field_summary['mean_intensity'].dropna().empty:
            ax.text(0.5, 0.5, f"No valid tracks match filter criteria in {field_name}", ha='center', va='center', transform=ax.transAxes)
            ax.grid(False)
            return

        isingle = self.spin_isingle_L.value() if "Right" not in chan_label else self.spin_isingle_R.value()
        stoich = (field_summary['mean_intensity'] / isingle).dropna()

        bins = np.arange(0, stoich.max() + 0.5, 0.25) if not stoich.empty else 20
        ax.hist(stoich, bins=bins, color='#008080', alpha=0.55, edgecolor='black', linewidth=0.6, label=f"Filtered Tracks (n={len(stoich)})")

        for i in range(1, int(np.ceil(stoich.max())) + 1 if not stoich.empty else 5):
            ax.axvline(i, color='darkred', linestyle=':', alpha=0.7, linewidth=1.0)

        ax.set_title(f"Live Reproduced: Stoichiometry Histogram — {field_name} [{chan_label}]", fontsize=10)
        ax.set_xlabel(r"Stoichiometry ($N = I_{\mathrm{mean}} / I_{\mathrm{single}}$)", fontsize=10)
        ax.set_ylabel("Trajectory Count", fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(False)

    def _render_live_spot_intensity(self, ax, df_field, field_name, chan_label):
        if df_field.empty:
            ax.text(0.5, 0.5, f"No spot data for filtered trajectories in {field_name}", ha='center', va='center', transform=ax.transAxes)
            ax.grid(False)
            return

        intensity_col = 'integrated_intensity' if 'integrated_intensity' in df_field.columns else (
            'intensity' if 'intensity' in df_field.columns else 'spot_intensity'
        )

        ax.scatter(df_field['frame'], df_field[intensity_col], color='#008080', alpha=0.35, s=10, edgecolors='none', label=f"Spots (n={len(df_field)})")

        isingle = self.spin_isingle_L.value() if "Right" not in chan_label else self.spin_isingle_R.value()
        ax.axhline(isingle, color='red', linestyle='--', linewidth=1.5, label=f"iSingle ({isingle:.1f} a.u.)")

        ax.set_title(f"Live Reproduced: Spot Intensity vs Frame — {field_name} [{chan_label}]", fontsize=10)
        ax.set_xlabel("Frame Number", fontsize=10)
        ax.set_ylabel("Spot Integrated Intensity (a.u.)", fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(False)

    # ================= INTENSITY PLOTTING & INDIVIDUAL LOG AXES =================

    def plot_intensity_distribution(self, summary_df):
        fig = self.stoich_canvas.fig
        fig.clf()

        selected_dist = self.combo_intensity_dist.currentText()
        use_log_x = self.chk_log_x_intensity.isChecked()
        use_log_y = self.chk_log_y_intensity.isChecked()
        has_isingle_data = bool(self.field_isingle_data)
        fit_kde = self.chk_intensity_kde.isChecked()

        # Build axes layout based on drop-down view selector
        if selected_dist == "Spot Intensity Histogram":
            axes_list = [fig.add_subplot(1, 1, 1)]
            plot_modes = ["spot_hist"]
        elif selected_dist == "Trajectory Mean Intensity Histogram":
            axes_list = [fig.add_subplot(1, 1, 1)]
            plot_modes = ["traj_hist"]
        elif selected_dist == "Log10(Spot Intensity) vs Frame Number":
            axes_list = [fig.add_subplot(1, 1, 1)]
            plot_modes = ["scatter"]
        elif selected_dist == "Field iSingle Peaks Boxplot":
            axes_list = [fig.add_subplot(1, 1, 1)]
            plot_modes = ["boxplot"]
        else:  # "All Intensity Plots (2x2 Grid)"
            if has_isingle_data:
                axes_list = [fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 2), fig.add_subplot(2, 2, 3), fig.add_subplot(2, 2, 4)]
                plot_modes = ["spot_hist", "traj_hist", "scatter", "boxplot"]
            else:
                axes_list = [fig.add_subplot(1, 3, 1), fig.add_subplot(1, 3, 2), fig.add_subplot(1, 3, 3)]
                plot_modes = ["spot_hist", "traj_hist", "scatter"]

        for ax in axes_list:
            ax.grid(False)

        if self.trajectories_df.empty:
            for ax in axes_list:
                ax.text(0.5, 0.5, "No data loaded", ha='center', va='center', transform=ax.transAxes)
                ax.grid(False)
            fig.tight_layout()
            self.stoich_canvas.draw()
            return

        selected_sample = self.sample_selector.currentText()
        if selected_sample == "All Samples Combined" or not selected_sample:
            spots_df = self.trajectories_df.copy()
        else:
            spots_df = self.trajectories_df[self.trajectories_df['sample'] == selected_sample].copy()

        min_f = self.spin_min_frame.value()
        max_f = self.spin_max_frame.value()
        spots_df = spots_df[(spots_df['frame'] >= min_f) & (spots_df['frame'] <= max_f)]

        selected_field = self.field_selector.currentText()
        if selected_field and selected_field != "-- All Fields --":
            spots_df = spots_df[spots_df['field'] == selected_field]

        intensity_col = 'integrated_intensity' if 'integrated_intensity' in spots_df.columns else (
            'intensity' if 'intensity' in spots_df.columns else (
                'spot_intensity' if 'spot_intensity' in spots_df.columns else None
            )
        )

        if self.chk_apply_frame_cutoff.isChecked() and self.spin_frame_cutoff.value() > 0:
            cutoff = self.spin_frame_cutoff.value()
            spots_df = spots_df[spots_df['frame'] >= cutoff]

        selected_chan = self.channel_selector.currentText() if self.channel_selector.isVisible() else "Single Channel"

        if selected_chan == "Left Channel (L)":
            spots_df_filter = spots_df[spots_df['channel'] == "Left Channel (L)"]
        elif selected_chan == "Right Channel (R)":
            spots_df_filter = spots_df[spots_df['channel'] == "Right Channel (R)"]
        else:
            spots_df_filter = spots_df

        bins_count = self.spin_bins.value()
        selected_chan_lbl = self.channel_selector.currentText() if self.channel_selector.isVisible() else "Single Channel"

        for mode, ax in zip(plot_modes, axes_list):
            ax.grid(False)

            if use_log_x and mode != "boxplot":
                ax.set_xscale('log')
            if use_log_y:
                ax.set_yscale('log')

            if mode == "spot_hist":
                if not intensity_col or spots_df_filter.empty:
                    ax.text(0.5, 0.5, "No spot intensity data", ha='center', va='center', transform=ax.transAxes)
                    continue

                df_L = spots_df_filter[spots_df_filter['channel'] == "Left Channel (L)"]
                df_R = spots_df_filter[spots_df_filter['channel'] == "Right Channel (R)"]
                df_single = spots_df_filter[~spots_df_filter['channel'].isin(["Left Channel (L)", "Right Channel (R)"])]

                if not df_L.empty:
                    vals_L = df_L[intensity_col].dropna().values
                    vals_L = vals_L[vals_L > 0]
                    if len(vals_L) > 0:
                        lbl_L = "Left Channel (L)" if self.channel_selector.isVisible() else "Spot Intensities"
                        b = np.logspace(np.log10(vals_L.min()), np.log10(vals_L.max()), bins_count) if use_log_x else bins_count
                        counts_L, bins_L, _ = ax.hist(vals_L, bins=b, color='#008080', alpha=0.45, label=lbl_L, edgecolor='black', linewidth=0.5)

                        if fit_kde and len(vals_L) > 1:
                            x_grid = np.geomspace(vals_L.min(), vals_L.max(), 300) if use_log_x else np.linspace(vals_L.min(), vals_L.max(), 300)
                            kde = gaussian_kde(vals_L, bw_method=0.1)
                            pdf = kde(x_grid)
                            b_width = np.mean(np.diff(bins_L)) if isinstance(bins_L, np.ndarray) and len(bins_L) > 1 else 1.0
                            y_kde = pdf * len(vals_L) * b_width
                            peak_L = float(x_grid[np.argmax(pdf)])
                            ax.plot(x_grid, y_kde, color='#004D4D', linewidth=2, label=f"Left KDE Peak: {peak_L:.1f}")

                if not df_R.empty:
                    vals_R = df_R[intensity_col].dropna().values
                    vals_R = vals_R[vals_R > 0]
                    if len(vals_R) > 0:
                        b = np.logspace(np.log10(vals_R.min()), np.log10(vals_R.max()), bins_count) if use_log_x else bins_count
                        counts_R, bins_R, _ = ax.hist(vals_R, bins=b, color='#D81B60', alpha=0.45, label="Right Channel (R)", edgecolor='black', linewidth=0.5)

                        if fit_kde and len(vals_R) > 1:
                            x_grid = np.geomspace(vals_R.min(), vals_R.max(), 300) if use_log_x else np.linspace(vals_R.min(), vals_R.max(), 300)
                            kde = gaussian_kde(vals_R, bw_method=0.1)
                            pdf = kde(x_grid)
                            b_width = np.mean(np.diff(bins_R)) if isinstance(bins_R, np.ndarray) and len(bins_R) > 1 else 1.0
                            y_kde = pdf * len(vals_R) * b_width
                            peak_R = float(x_grid[np.argmax(pdf)])
                            ax.plot(x_grid, y_kde, color='#880e4f', linewidth=2, label=f"Right KDE Peak: {peak_R:.1f}")

                if not df_single.empty and df_L.empty and df_R.empty:
                    vals_s = df_single[intensity_col].dropna().values
                    vals_s = vals_s[vals_s > 0]
                    if len(vals_s) > 0:
                        b = np.logspace(np.log10(vals_s.min()), np.log10(vals_s.max()), bins_count) if use_log_x else bins_count
                        counts_s, bins_s, _ = ax.hist(vals_s, bins=b, color='#008080', alpha=0.45, label="Spot Intensities", edgecolor='black', linewidth=0.5)
                        
                        if fit_kde and len(vals_s) > 1:
                            x_grid = np.geomspace(vals_s.min(), vals_s.max(), 300) if use_log_x else np.linspace(vals_s.min(), vals_s.max(), 300)
                            kde = gaussian_kde(vals_s, bw_method=0.1)
                            pdf = kde(x_grid)
                            b_width = np.mean(np.diff(bins_s)) if isinstance(bins_s, np.ndarray) and len(bins_s) > 1 else 1.0
                            y_kde = pdf * len(vals_s) * b_width
                            peak_s = float(x_grid[np.argmax(pdf)])
                            ax.plot(x_grid, y_kde, color='#004D4D', linewidth=2, label=f"KDE Peak: {peak_s:.1f}")

                if selected_chan != "Right Channel (R)":
                    isingle_L = self.spin_isingle_L.value()
                    ax.axvline(isingle_L, color='#008080', linestyle='--', linewidth=1.5, label=f"iSingle L ({isingle_L:.1f})")
                
                if self.channel_selector.isVisible() and self.spin_isingle_R.isVisible() and selected_chan in ["Right Channel (R)", "Both Channels (Overlay)"]:
                    isingle_R = self.spin_isingle_R.value()
                    ax.axvline(isingle_R, color='#D81B60', linestyle='--', linewidth=1.5, label=f"iSingle R ({isingle_R:.1f})")

                frame_txt = f" (Frame ≥ {self.spin_frame_cutoff.value()})" if self.chk_apply_frame_cutoff.isChecked() and self.spin_frame_cutoff.value() > 0 else ""
                ax.set_title(f"Spot Intensity Histogram [{selected_chan_lbl}]{frame_txt}", fontsize=10)
                ax.set_xlabel("Spot Intensity (a.u.)", fontsize=10)
                ax.set_ylabel("Spot Count", fontsize=10)
                ax.legend(loc='upper right', fontsize=8)

            elif mode == "traj_hist":
                if summary_df.empty:
                    ax.text(0.5, 0.5, "No trajectory summary data", ha='center', va='center', transform=ax.transAxes)
                    continue

                df_L = summary_df[summary_df['channel'] == "Left Channel (L)"]
                df_R = summary_df[summary_df['channel'] == "Right Channel (R)"]

                if not df_L.empty:
                    vals_L = df_L['mean_intensity'].dropna().values
                    vals_L = vals_L[vals_L > 0]
                    if len(vals_L) > 0:
                        lbl_L = "Left Channel (L)" if self.channel_selector.isVisible() else "Trajectory Mean Intensity"
                        b = np.logspace(np.log10(vals_L.min()), np.log10(vals_L.max()), bins_count) if use_log_x else bins_count
                        ax.hist(vals_L, bins=b, color='#008080', alpha=0.45, label=lbl_L, edgecolor='black', linewidth=0.5)

                        if fit_kde and len(vals_L) > 1:
                            x_grid = np.geomspace(vals_L.min(), vals_L.max(), 300) if use_log_x else np.linspace(vals_L.min(), vals_L.max(), 300)
                            kde = gaussian_kde(vals_L, bw_method=0.1)
                            pdf = kde(x_grid)
                            b_width = np.mean(np.diff(b)) if isinstance(b, np.ndarray) and len(b) > 1 else 1.0
                            ax.plot(x_grid, pdf * len(vals_L) * b_width, color='#004D4D', linewidth=2, label="Left KDE")

                if not df_R.empty:
                    vals_R = df_R['mean_intensity'].dropna().values
                    vals_R = vals_R[vals_R > 0]
                    if len(vals_R) > 0:
                        b = np.logspace(np.log10(vals_R.min()), np.log10(vals_R.max()), bins_count) if use_log_x else bins_count
                        ax.hist(vals_R, bins=b, color='#D81B60', alpha=0.45, label="Right Channel (R)", edgecolor='black', linewidth=0.5)

                        if fit_kde and len(vals_R) > 1:
                            x_grid = np.geomspace(vals_R.min(), vals_R.max(), 300) if use_log_x else np.linspace(vals_R.min(), vals_R.max(), 300)
                            kde = gaussian_kde(vals_R, bw_method=0.1)
                            pdf = kde(x_grid)
                            b_width = np.mean(np.diff(b)) if isinstance(b, np.ndarray) and len(b) > 1 else 1.0
                            ax.plot(x_grid, pdf * len(vals_R) * b_width, color='#880e4f', linewidth=2, label="Right KDE")

                if selected_chan != "Right Channel (R)":
                    isingle_L = self.spin_isingle_L.value()
                    ax.axvline(isingle_L, color='#008080', linestyle='--', linewidth=1.5, label=f"iSingle L ({isingle_L:.1f})")

                if self.channel_selector.isVisible() and self.spin_isingle_R.isVisible() and selected_chan in ["Right Channel (R)", "Both Channels (Overlay)"]:
                    isingle_R = self.spin_isingle_R.value()
                    ax.axvline(isingle_R, color='#D81B60', linestyle='--', linewidth=1.5, label=f"iSingle R ({isingle_R:.1f})")

                if self.channel_selector.isVisible() and self.spin_isingle_R.isVisible():
                    isingle_R = self.spin_isingle_R.value()
                    ax.axvline(isingle_R, color='#D81B60', linestyle='--', linewidth=1.5, label=f"iSingle R ({isingle_R:.1f})")

                ax.set_title(f"Trajectory Mean Intensity Distribution [{selected_chan_lbl}]", fontsize=10)
                ax.set_xlabel("Mean Integrated Intensity (a.u.)", fontsize=10)
                ax.set_ylabel("Trajectory Count", fontsize=10)
                ax.legend(loc='upper right', fontsize=8)

            elif mode == "scatter":
                if not intensity_col or spots_df_filter.empty:
                    ax.text(0.5, 0.5, "No spot intensity data", ha='center', va='center', transform=ax.transAxes)
                    continue

                valid_spots = spots_df_filter[(spots_df_filter[intensity_col] > 0) & np.isfinite(spots_df_filter['frame'])].copy()

                if not valid_spots.empty:
                    df_sp_L = valid_spots[valid_spots['channel'] == "Left Channel (L)"]
                    df_sp_R = valid_spots[valid_spots['channel'] == "Right Channel (R)"]
                    df_sp_gen = valid_spots[~valid_spots['channel'].isin(["Left Channel (L)", "Right Channel (R)"])]

                    if not df_sp_L.empty:
                        lbl = "Left Channel (L)" if self.channel_selector.isVisible() else "Spots"
                        ax.scatter(df_sp_L['frame'], df_sp_L[intensity_col], color='#008080', alpha=0.35, s=8, label=lbl, edgecolors='none')

                    if not df_sp_R.empty:
                        ax.scatter(df_sp_R['frame'], df_sp_R[intensity_col], color='#D81B60', alpha=0.35, s=8, label="Right Channel (R)", edgecolors='none')

                    if not df_sp_gen.empty and df_sp_L.empty and df_sp_R.empty:
                        ax.scatter(df_sp_gen['frame'], df_sp_gen[intensity_col], color='#008080', alpha=0.35, s=8, label="Spots", edgecolors='none')

                    if selected_chan != "Right Channel (R)":
                        isingle_L = self.spin_isingle_L.value()
                        if isingle_L > 0:
                            ax.axhline(isingle_L, color='#008080', linestyle='--', linewidth=1.5, label=f"iSingle L ({isingle_L:.1f})")

                    if self.channel_selector.isVisible() and self.spin_isingle_R.isVisible() and selected_chan in ["Right Channel (R)", "Both Channels (Overlay)"]:
                        isingle_R = self.spin_isingle_R.value()
                        if isingle_R > 0:
                            ax.axhline(isingle_R, color='#D81B60', linestyle='--', linewidth=1.5, label=f"iSingle R ({isingle_R:.1f})")

                    if self.spin_frame_cutoff.value() > 0:
                        cutoff_val = self.spin_frame_cutoff.value()
                        ax.axvline(cutoff_val, color='gray', linestyle=':', linewidth=1.2, label=f"Cutoff Frame ({cutoff_val})")

                    ax.set_title("Spot Intensity vs. Frame Number", fontsize=10)
                    ax.set_xlabel("Frame Number", fontsize=10)
                    ax.set_ylabel("Spot Intensity [a.u.]", fontsize=10)
                    ax.legend(loc='upper right', fontsize=8)

            elif mode == "boxplot":
                if not has_isingle_data:
                    ax.text(0.5, 0.5, "No scanned field iSingle .dat files found", ha='center', va='center', transform=ax.transAxes)
                    continue

                df_is = pd.DataFrame(self.field_isingle_data)

                if self.channel_selector.isVisible():
                    selected_chan = self.channel_selector.currentText()
                    if selected_chan == "Left Channel (L)":
                        df_is = df_is[df_is['channel'] == "Left Channel (L)"]
                    elif selected_chan == "Right Channel (R)":
                        df_is = df_is[df_is['channel'] == "Right Channel (R)"]

                if not df_is.empty:
                    channels = df_is['channel'].unique()
                    data_by_chan = [df_is[df_is['channel'] == ch]['i_single'].values for ch in channels]

                    bp = ax.boxplot(data_by_chan, tick_labels=channels, patch_artist=True, widths=0.35)

                    palette = ['#008080', '#D81B60', '#4A90E2']
                    for patch, color in zip(bp['boxes'], palette[:len(channels)]):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.5)

                    for idx_c, (ch, vals) in enumerate(zip(channels, data_by_chan), 1):
                        jitter = np.random.normal(0, 0.04, size=len(vals))
                        ax.scatter(np.full_like(vals, idx_c) + jitter, vals, color='black', alpha=0.7, zorder=3, s=22)

                    overall_mean = df_is['i_single'].mean()
                    ax.axhline(overall_mean, color='darkgreen', linestyle='--', linewidth=1.5, label=f"Mean iSingle: {overall_mean:.1f} a.u.")

                    ax.set_title(f"Field iSingle Peaks (n={len(df_is)} fields/files)", fontsize=10)
                    ax.set_ylabel("Single Fluorophore Intensity (a.u.)", fontsize=10)
                    ax.legend(loc='upper right', fontsize=8)

            ax.grid(False)

        fig.tight_layout()
        self.stoich_canvas.draw()

    # ================= MSD VS TIME LAG PLOTTING (NEW TAB) =================

    def plot_msd_vs_timelag(self, summary_df):
        """Calculates and plots Mean Squared Displacement (MSD) vs Time Lag (tau) for active selection."""
        ax = self.msd_canvas.axes
        ax.clear()

        if self.trajectories_df.empty or summary_df.empty:
            ax.grid(False)
            self.msd_canvas.draw()
            return

        allowed_ids = set(summary_df['global_track_id'].unique())
        df = self.trajectories_df[self.trajectories_df['global_track_id'].isin(allowed_ids)]

        min_f = self.spin_min_frame.value()
        max_f = self.spin_max_frame.value()
        df = df[(df['frame'] >= min_f) & (df['frame'] <= max_f)]

        pixel_size = self.spin_pixel_size.value()
        dt_ms = self.spin_exposure.value()
        is_alex = self.chk_alex.isChecked()
        dt_sec = (dt_ms / 1000.0) * (2.0 if is_alex else 1.0)

        max_lag = self.spin_msd_max_lag.value()
        show_indiv = self.chk_msd_show_individual.isChecked()
        fit_linear = self.chk_msd_fit_linear.isChecked()

        lags = np.arange(1, max_lag + 1)
        tau_times = lags * dt_sec

        msd_per_lag = {lag: [] for lag in lags}
        valid_tracks_count = 0

        for track_id, track in df.groupby('global_track_id'):
            if len(track) < 3:
                continue

            track_sorted = track.sort_values('frame')
            x = track_sorted['x'].values * pixel_size
            y = track_sorted['y'].values * pixel_size
            frames = track_sorted['frame'].values

            n_points = len(frames)
            track_msds = []

            for lag in lags:
                if lag >= n_points:
                    track_msds.append(np.nan)
                    continue

                dx = x[lag:] - x[:-lag]
                dy = y[lag:] - y[:-lag]
                dframes = frames[lag:] - frames[:-lag]

                valid_step_mask = (dframes == lag)
                if not np.any(valid_step_mask):
                    track_msds.append(np.nan)
                    continue

                sq_disp = (dx[valid_step_mask]**2 + dy[valid_step_mask]**2)
                mean_sq = np.mean(sq_disp)
                track_msds.append(mean_sq)
                msd_per_lag[lag].append(mean_sq)

            valid_tracks_count += 1

            if show_indiv and not np.all(np.isnan(track_msds)):
                ax.plot(tau_times, track_msds, color='#008080', alpha=0.12, linewidth=0.7)

        avg_msd = []
        std_msd = []

        for lag in lags:
            vals = msd_per_lag[lag]
            if len(vals) > 0:
                avg_msd.append(np.mean(vals))
                std_msd.append(np.std(vals))
            else:
                avg_msd.append(np.nan)
                std_msd.append(np.nan)

        avg_msd = np.array(avg_msd)
        std_msd = np.array(std_msd)

        valid_mask = ~np.isnan(avg_msd)
        if np.any(valid_mask):
            ax.plot(tau_times[valid_mask], avg_msd[valid_mask], 'o-', color='#D81B60', linewidth=2.2, markersize=5, label=f"Ensemble Average MSD (n={valid_tracks_count})")

            # Fit linear slope: MSD = 4 * D * tau + offset (first 4 lags)
            fit_text = "No linear fit performed."
            if fit_linear and np.sum(valid_mask) >= 2:
                fit_lags = min(4, np.sum(valid_mask))
                x_fit = tau_times[:fit_lags]
                y_fit = avg_msd[:fit_lags]

                if len(x_fit) >= 2:
                    coeffs = np.polyfit(x_fit, y_fit, 1)  # slope, intercept
                    slope, intercept = coeffs[0], coeffs[1]
                    d_msd = slope / 4.0

                    fit_line = slope * tau_times + intercept
                    ax.plot(tau_times, fit_line, '--', color='black', linewidth=1.5, label=f"Linear Fit ($4D\\tau$): $D = {d_msd:.3f}\\ \\mu\\mathrm{{m}}^2/\\mathrm{{s}}$")

                    chan_lbl = self.channel_selector.currentText() if self.channel_selector.isVisible() else "Single Channel"
                    fit_text = (
                        f"<b>MSD Linear Fit Summary [{chan_lbl}]:</b><br>"
                        f"• <b>Apparent Diffusion ($D_{{MSD}}$):</b> {d_msd:.4f} μm²/s<br>"
                        f"• <b>Slope (4D):</b> {slope:.4f} μm²/s<br>"
                        f"• <b>Y-Intercept (Loc Precision ²):</b> {intercept:.4f} μm²<br>"
                        f"• <b>Tracks Analyzed:</b> {valid_tracks_count}"
                    )
            self.lbl_msd_fit_results.setText(fit_text)

        selected_chan = self.channel_selector.currentText() if self.channel_selector.isVisible() else "Single Channel"
        selected_field = self.field_selector.currentText()
        field_suffix = f" | Field: {selected_field}" if selected_field and selected_field != "-- All Fields --" else ""

        ax.set_title(f"Mean Squared Displacement (MSD) vs Time Lag — {self.sample_selector.currentText()} [{selected_chan}]{field_suffix}", fontsize=11)
        ax.set_xlabel(r"Time Lag $\tau$ [s]", fontsize=11)
        ax.set_ylabel(r"MSD $\langle r^2(\tau) \rangle$ $[\mu\mathrm{m}^2]$", fontsize=11)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(False)

        self.msd_canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ResultsViewerTab(root_dir=".")
    viewer.setWindowTitle("PySTACHIO Studio - Results Viewer")
    viewer.resize(1300, 850)
    viewer.show()
    sys.exit(app.exec())