
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

from PIL import Image
from scipy.stats import gaussian_kde, gamma, rayleigh

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

        # Data containers
        self.trajectories_df = pd.DataFrame()
        self.diffusion_summary_df = pd.DataFrame()
        self.field_dirs_map = {}  # field_name -> field_directory_path

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
        self.field_selector.currentIndexChanged.connect(self.refresh_all_plots)

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

        self.chk_remove_neg = QCheckBox("Remove Negative D (D ≤ 0): 0")
        self.chk_remove_neg.setChecked(False)

        track_layout.addRow("Min Track Length:", self.spin_min_len)
        track_layout.addRow(self.chk_remove_neg)
        track_group.setLayout(track_layout)
        sidebar_layout.addWidget(track_group)

        # 5b. Diffusion histogram controls (shown on the Diffusion tab)
        self.diff_hist_group = QGroupBox("Diffusion Histogram Controls")
        diff_layout = QFormLayout()
        diff_layout.setVerticalSpacing(6)

        self.spin_bin_size = NoScrollDoubleSpinBox()
        self.spin_bin_size.setRange(0.0001, 100.0)
        self.spin_bin_size.setDecimals(4)
        self.spin_bin_size.setValue(0.09)
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

        # 5c. Jump distance histogram controls (shown on the Jump Distance tab)
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
        self.jump_hist_group.setVisible(False)  # Diffusion tab is shown first

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
        self.jump_fit_group.setVisible(False)  # Diffusion tab is shown first

        # 7. Intensity Filtering Controls
        intensity_group = QGroupBox("Intensity Filtering Controls")
        intensity_layout = QFormLayout()
        intensity_layout.setVerticalSpacing(6)

        self.chk_filter_intensity = QCheckBox("Filter by Single Fluorophore Intensity")
        self.chk_filter_intensity.setChecked(False)

        self.spin_isingle = NoScrollDoubleSpinBox()
        self.spin_isingle.setDecimals(1)
        self.spin_isingle.setRange(0.0, 1000000.0)
        self.spin_isingle.setValue(default_isingle)
        self.spin_isingle.setSingleStep(100.0)
        self.spin_isingle.setSuffix(" a.u.")

        intensity_layout.addRow(self.chk_filter_intensity)
        intensity_layout.addRow("iSingle Value:", self.spin_isingle)
        intensity_group.setLayout(intensity_layout)
        sidebar_layout.addWidget(intensity_group)

        # Event connections
        self.spin_pixel_size.valueChanged.connect(self.recalculate_and_refresh)
        self.spin_exposure.valueChanged.connect(self.on_frame_time_or_alex_changed)
        self.chk_alex.stateChanged.connect(self.on_frame_time_or_alex_changed)
        self.spin_min_len.valueChanged.connect(self.recalculate_and_refresh)
        self.spin_jump_lag.valueChanged.connect(self.refresh_all_plots)
        self.spin_bin_size.valueChanged.connect(self.refresh_all_plots)
        self.spin_bins.valueChanged.connect(self.refresh_all_plots)
        self.chk_log_scale.stateChanged.connect(self.refresh_all_plots)
        self.chk_kde.stateChanged.connect(self.refresh_all_plots)
        self.chk_remove_neg.stateChanged.connect(self.refresh_all_plots)
        self.chk_show_subcomponents.stateChanged.connect(self.refresh_all_plots)
        self.chk_show_non_best.stateChanged.connect(self.refresh_all_plots)
        self.chk_show_subcomponents_jump.stateChanged.connect(self.refresh_all_plots)
        self.spin_bins_jump.valueChanged.connect(self.refresh_all_plots)
        self.chk_kde_jump.stateChanged.connect(self.refresh_all_plots)
        self.chk_show_non_best_jump.stateChanged.connect(self.refresh_all_plots)
        self.chk_filter_intensity.stateChanged.connect(self.refresh_all_plots)
        self.spin_isingle.valueChanged.connect(self.refresh_all_plots)

        # Enforce strong focus policy so widgets only receive input when clicked
        for widget in sidebar.findChildren((NoScrollSpinBox, NoScrollDoubleSpinBox, NoScrollComboBox)):
            widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # 8. Summary Info Panel
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
        self.spatial_canvas = MplCanvas(self)
        self.spatial_toolbar = NavigationToolbar(self.spatial_canvas, self)
        layout_spatial = QVBoxLayout(self.tab_spatial)
        layout_spatial.addWidget(self.spatial_toolbar)
        layout_spatial.addWidget(self.spatial_canvas)
        self.tabs.addTab(self.tab_spatial, "Spatial Map")

        # TAB 4: Single Field Inspector
        self.tab_field = QWidget()
        self.field_canvas = MplCanvas(self)
        self.field_toolbar = NavigationToolbar(self.field_canvas, self)
        layout_field = QVBoxLayout(self.tab_field)
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

        splitter.setSizes([340, 960])

        # Swap the fitting box when switching between the Diffusion / Jump Distance tabs.
        # Both boxes (and their widgets / results) persist, so state is kept when switching back.
        self.tabs.currentChanged.connect(self.on_tab_changed)
        self.on_tab_changed(self.tabs.currentIndex())

    def on_tab_changed(self, index):
        """Swaps the histogram-control and fitting boxes to match the active tab.

        The Jump Distance tab gets its own boxes; every other tab uses the diffusion ones
        (the Intensity tab, for example, reads the diffusion bin count / KDE settings).
        All boxes persist when hidden, so their state is kept when switching back.
        """
        is_jump = self.tabs.widget(index) is self.tab_jump
        self.diff_hist_group.setVisible(not is_jump)
        self.diff_fit_group.setVisible(not is_jump)
        self.jump_hist_group.setVisible(is_jump)
        self.jump_fit_group.setVisible(is_jump)

    def update_channel_visibility(self):
        """Shows or hides channel selector controls based on actual presence of >1 channels."""
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

        if not has_multiple_channels:
            self.channel_selector.blockSignals(True)
            self.channel_selector.setCurrentIndex(0)
            self.channel_selector.blockSignals(False)

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

        # Create progress dialog with a working Cancel button
        self.progress_dialog = QProgressDialog("Initializing diffusion fitting...", "Cancel", 0, max_comp, self)
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.setValue(0)

        # Instantiate background worker thread
        self.worker_thread = FitWorkerThread(
            fit_type='diffusion',
            data=d_clean,
            max_comp=max_comp,
            parent=self
        )

        # Wire up progress, finish, error, and cancel signals
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

        # Create progress dialog with a working Cancel button
        self.progress_dialog = QProgressDialog("Initializing jump distance fitting...", "Cancel", 0, max_comp, self)
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.setValue(0)

        # Instantiate background worker thread
        self.worker_thread = FitWorkerThread(
            fit_type='jump',
            data=jumps,
            max_comp=max_comp,
            dt=dt_lag,
            loc_error=sigma_loc,
            model_type=selected_model,
            parent=self
        )

        # Wire up progress, finish, error, and cancel signals
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

        self.best_jump_model = min(self.jump_fit_results, key=lambda x: x['bic'])
        best_j_k = self.best_jump_model['components']

        selected_model = self.combo_jdd_model.currentText()
        chan_label = self.channel_selector.currentText() if self.channel_selector.isVisible() else "Single Channel"
        summary_lines = [
            f"<b>JDD Model:</b> {selected_model}",
            f"<b>Best Jump Distance Model ({chan_label}): {best_j_k}-Component(s)</b>"
        ]
        for res in self.jump_fit_results:
            k = res['components']
            bic = res['bic']
            delta_bic = bic - self.best_jump_model['bic']
            mark = "★ BEST" if k == best_j_k else f"(ΔBIC: +{delta_bic:.1f})"
            summary_lines.append(f"• <b>{k}-Comp:</b> BIC = {bic:.1f} {mark}")

            if k == best_j_k:
                for p in res['params']:
                    w = p.get('weight', 1.0) * 100
                    d_val = p.get('D', p.get('mean_D', 0.0))
                    summary_lines.append(f"  └ D ≈ {d_val:.3f} μm²/s ({w:.1f}%)")

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
            """Removes non-serializable callables from parameter dictionaries."""
            sanitized = []
            for p in param_list:
                clean_p = {k: v for k, v in p.items() if not callable(v)}
                sanitized.append(clean_p)
            return sanitized

        export_dict = {
            'metadata': {
                # Dataset selection
                'sample': self.sample_selector.currentText(),
                'channel': (
                    self.channel_selector.currentText()
                    if self.channel_selector.isVisible()
                    else "Single Channel"
                ),
                'field': self.field_selector.currentText(),
        
                # Acquisition
                'pixel_size_um': self.spin_pixel_size.value(),
                'frame_interval_ms': self.spin_exposure.value(),
                'alex_mode': self.chk_alex.isChecked(),
                'localisation_precision_um': self.loc_precision,
        
                # Diffusion settings
                'min_track_length': self.spin_min_len.value(),
                'negative_D_removed': self.chk_remove_neg.isChecked(),
        
                # JDD settings
                'jdd_model': self.combo_jdd_model.currentText(),
                'jump_lag_frames': self.spin_jump_lag.value(),
                'effective_dt_seconds': (
                    (self.spin_exposure.value() / 1000.0)
                    * (2.0 if self.chk_alex.isChecked() else 1.0)
                    * self.spin_jump_lag.value()
                ),
        
                # Histogram settings
                'histogram_bins': self.spin_bins.value(),
                'jump_histogram_bins': self.spin_bins_jump.value(),
                'diffusion_bin_size': self.spin_bin_size.value(),
                'log_scale_enabled': self.chk_log_scale.isChecked(),
                'kde_enabled': self.chk_kde.isChecked(),
                'jump_kde_enabled': self.chk_kde_jump.isChecked(),
        
                # Intensity filtering
                'intensity_filter_enabled': self.chk_filter_intensity.isChecked(),
                'isingle_value': self.spin_isingle.value(),
        
                # Model fitting controls
                'max_components_tested': self.spin_max_components.value(),
                'max_components_tested_jump': self.spin_max_components_jump.value(),
        
                # Data statistics
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

    # ================= CONFIG PARSING =================

    def parse_config_file(self, filepath):
        config = {}
        PIXEL_KEYS = {'pixel_size', 'pixel_length', 'pixelsize', 'px_size', 'pixel_sz', 'pixel', 'px'}
        FRAME_KEYS = {'frame_time', 'frame_interval', 'frametime', 'exposure_time', 'exposure', 'time_step', 'timestep', 'dt', 'frame_duration', 'frame_period'}
        ALEX_KEYS = {'alex', 'alex_flag', 'alex_mode', 'use_alex', 'is_alex'}
        PRECISION_KEYS = {'localisation_precision', 'localization_precision', 'loc_prec', 'loc_precision', 'precision', 'sigma'}

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
        except Exception as e:
            print(f"Error parsing config file {filepath}: {e}")
        return config

    def has_config_conflicts(self, parsed_configs):
        configs = list(parsed_configs.values())
        if len(configs) <= 1:
            return False

        keys = ['pixel_size', 'frame_time', 'alex', 'localisation_precision']
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

        if self.loc_precision is not None:
            sigma = self.loc_precision
            bin_size = (sigma ** 2) / (4.0 * dt_sec) if dt_sec > 0 else 0.09
        else:
            base_bin_size = 0.045 if is_alex else 0.09
            bin_size = base_bin_size * (3.5 / dt_ms) if dt_ms > 0 else base_bin_size

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

        self.update_default_bin_size()

        # Scan Trajectories
        search_pattern = os.path.join(self.root_dir, "**", "*_trajectories.tsv")
        all_files = glob.glob(search_pattern, recursive=True)
        
        print(f"\n[INFO] Found {len(all_files)} trajectory file(s) in '{self.root_dir}'\n", flush=True)

        if not all_files:
            self.lbl_stats.setText("No *_trajectories.tsv files found.")
            self.trajectories_df = pd.DataFrame()
            self.field_dirs_map = {}
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
            samples = sorted(self.trajectories_df['sample'].unique().tolist())
            self.populate_sample_dropdown(samples)
        else:
            self.trajectories_df = pd.DataFrame()
            self.populate_sample_dropdown([])

        self.update_channel_visibility()

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
        self.refresh_all_plots()

    # ================= DIFFUSION CALCULATIONS =================

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
                'intensity' if 'intensity' in track.columns else None
            )
            mean_intensity = track[intensity_col].mean() if intensity_col else np.nan

            results.append({
                'global_track_id': track_id,
                'sample': sample,
                'field': field,
                'channel': channel,
                'D': D,
                'log_D': np.log10(D) if D > 0 and not np.isnan(D) else np.nan,
                'track_length': len(track),
                'mean_intensity': mean_intensity
            })

        self.diffusion_summary_df = pd.DataFrame(results)

        num_negative = (self.diffusion_summary_df['D'] <= 0).sum() if not self.diffusion_summary_df.empty else 0
        self.chk_remove_neg.setText(f"Remove Negative D (D ≤ 0): {num_negative}")
        self.refresh_all_plots()

    def get_filtered_summary_df(self):
        """Filters trajectory summary DataFrame based on Channel, Negative D, Intensity, and Field filters."""
        if self.diffusion_summary_df.empty:
            return pd.DataFrame()

        df = self.diffusion_summary_df.copy()

        # 1. Channel Filter
        if self.channel_selector.isVisible():
            selected_chan = self.channel_selector.currentText()
            if selected_chan == "Left Channel (L)":
                df = df[df['channel'] == "Left Channel (L)"]
            elif selected_chan == "Right Channel (R)":
                df = df[df['channel'] == "Right Channel (R)"]

        # 2. Negative D Filter
        if self.chk_remove_neg.isChecked():
            df = df[df['D'] > 0]

        # 3. Intensity Filter
        if self.chk_filter_intensity.isChecked():
            isingle = self.spin_isingle.value()
            low_bound = 0.5 * isingle
            high_bound = 1.5 * isingle
            df = df[(df['mean_intensity'] >= low_bound) & (df['mean_intensity'] <= high_bound)]

        # 4. Field Filter
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
            intensity_filter_label = (
                f"Active [0.5 - 1.5 × {self.spin_isingle.value():.1f}]" if self.chk_filter_intensity.isChecked() else "Off"
            )

            self.lbl_stats.setText(
                f"<b>Scope:</b> {self.sample_selector.currentText()}<br>"
                f"<b>Channel Filter:</b> {chan_label}<br>"
                f"<b>Field:</b> {field_label}<br>"
                f"<b>Total Raw Tracks:</b> {total_raw}<br>"
                f"<b>Valid Length Tracks:</b> {valid_tracks}<br>"
                f"<b>Negative/Zero D Tracks:</b> {neg_count}<br>"
                f"<b>Intensity Filter:</b> {intensity_filter_label}<br>"
                f"<b>Displayed Tracks:</b> {displayed_tracks}"
            )

        self.plot_diffusion_histogram(summary_df)
        self.plot_jump_distance_histogram(summary_df)
        self.plot_spatial_map(summary_df)
        self.update_single_field_plot()
        self.plot_intensity_distribution(summary_df)

    def plot_diffusion_histogram(self, summary_df):
        ax = self.diff_canvas.axes
        ax.clear()

        if summary_df.empty:
            self.diff_canvas.draw()
            return

        selected_chan = self.channel_selector.currentText() if self.channel_selector.isVisible() else "Single Channel"
        use_log = self.chk_log_scale.isChecked()

        val_col = 'log_D' if use_log else 'D'
        x_label = r"$\log_{10}(D \ [\mu\mathrm{m}^2/\mathrm{s}])$" if use_log else r"$D \ [\mu\mathrm{m}^2/\mathrm{s}]$"

        if use_log:
            bins = self.spin_bins.value()
        else:
            bin_width = self.spin_bin_size.value()
            valid_vals = summary_df['D'].dropna()
            if not valid_vals.empty and bin_width > 0:
                d_min = valid_vals.min()
                d_max = valid_vals.max()
                bins = np.arange(d_min, d_max + bin_width, bin_width)
                if len(bins) < 2:
                    bins = self.spin_bins.value()
            else:
                bins = self.spin_bins.value()

        counts, bin_edges, _ = ax.hist(
            summary_df[val_col].dropna().values,
            bins=bins, color='#008080', alpha=0.4, label=f"Data [{selected_chan}] (n={len(summary_df)})", edgecolor='black', linewidth=0.5
        )

        if self.chk_kde.isChecked() and len(summary_df) > 1:
            data = summary_df[val_col].dropna().values
            x_grid = np.linspace(data.min(), data.max(), 300)
            kde = gaussian_kde(data)
            b_width = np.mean(np.diff(bin_edges)) if isinstance(bin_edges, np.ndarray) and len(bin_edges) > 1 else 1.0
            y_kde = kde(x_grid) * len(data) * b_width
            ax.plot(x_grid, y_kde, color='#004D4D', linewidth=2, label="KDE Distribution")

        if self.diff_fit_results and not use_log:
            d_clean = summary_df['D'].dropna().values
            d_clean = d_clean[d_clean > 0]

            if len(d_clean) > 0:
                x_grid = np.linspace(d_clean.min(), d_clean.max(), 350)
                bin_scale = np.diff(bin_edges)[0] * len(d_clean) if len(bin_edges) > 1 else len(d_clean)
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
        ax.grid(True, linestyle='--', alpha=0.5)

        self.diff_canvas.draw()

    def plot_jump_distance_histogram(self, summary_df):
        ax = self.jump_canvas.axes
        ax.clear()

        if self.trajectories_df.empty or summary_df.empty:
            self.jump_canvas.draw()
            return

        jumps = self.extract_jump_distances(summary_df)
        if len(jumps) == 0:
            self.jump_canvas.draw()
            return

        lag_N = self.spin_jump_lag.value()
        dt_ms = self.spin_exposure.value()
        is_alex = self.chk_alex.isChecked()
        dt_sec = (dt_ms / 1000.0) * (2.0 if is_alex else 1.0)
        time_lag_ms = lag_N * dt_sec * 1000.0
        bins_count = self.spin_bins_jump.value()
        selected_chan = self.channel_selector.currentText() if self.channel_selector.isVisible() else "Single Channel"

        counts, bin_edges, _ = ax.hist(
            jumps, bins=bins_count, color='#008080', alpha=0.5,
            label=f"Jump Data [{selected_chan}] (N={len(jumps)} steps)", edgecolor='black', linewidth=0.5
        )

        if self.chk_kde_jump.isChecked() and len(jumps) > 1:
            x_grid = np.linspace(jumps.min(), jumps.max(), 300)
            kde = gaussian_kde(jumps)
            b_width = np.mean(np.diff(bin_edges)) if len(bin_edges) > 1 else 1.0
            y_kde = kde(x_grid) * len(jumps) * b_width
            ax.plot(x_grid, y_kde, color='#004D4D', linewidth=2, label="Jump KDE")

        if self.jump_fit_results:
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
        ax.grid(True, linestyle='--', alpha=0.5)

        self.jump_canvas.draw()

    def plot_spatial_map(self, summary_df):
        ax = self.spatial_canvas.axes
        ax.clear()

        if self.trajectories_df.empty or summary_df.empty:
            self.spatial_canvas.draw()
            return

        selected_field = self.field_selector.currentText()
        img_data = None

        if selected_field and selected_field != "-- All Fields --" and selected_field in self.field_dirs_map:
            field_dir = self.field_dirs_map[selected_field]
            target_img_path = None

            candidate_files = ["L_avg.tif", "L_avg.tiff", "l_avg.tif", "l_avg.tiff"]
            for cand in candidate_files:
                p1 = os.path.join(field_dir, cand)
                p2 = os.path.join(field_dir, "results", cand)
                if os.path.exists(p1):
                    target_img_path = p1
                    break
                elif os.path.exists(p2):
                    target_img_path = p2
                    break

            if not target_img_path:
                glob_matches = glob.glob(os.path.join(field_dir, "*L_avg*.[tT][iI][fF]*")) + \
                               glob.glob(os.path.join(field_dir, "results", "*L_avg*.[tT][iI][fF]*"))
                if glob_matches:
                    target_img_path = glob_matches[0]

            if target_img_path:
                try:
                    img_data = np.array(Image.open(target_img_path))
                except Exception as e:
                    print(f"Error loading {target_img_path}: {e}")

        if img_data is not None:
            if img_data.ndim == 3 and img_data.shape[0] < 10:
                img_data = img_data[0]
            ax.imshow(img_data, cmap='gray', origin='lower')

        allowed_ids = set(summary_df['global_track_id'].unique())
        df = self.trajectories_df[self.trajectories_df['global_track_id'].isin(allowed_ids)]

        for track_id, track in df.groupby('global_track_id'):
            channel = track['channel'].iloc[0]

            if img_data is not None:
                color = '#00FFFF' if channel == "Left Channel (L)" else '#FF00FF'
                alpha = 0.8
                linewidth = 0.9
            else:
                color = '#008080' if channel == "Left Channel (L)" else '#D81B60'
                alpha = 0.5
                linewidth = 0.8

            ax.plot(track['x'], track['y'], color=color, alpha=alpha, linewidth=linewidth)

        field_suffix = f" ({selected_field})" if selected_field and selected_field != "-- All Fields --" else ""
        selected_chan = self.channel_selector.currentText() if self.channel_selector.isVisible() else "Single Channel"

        ax.set_title(f"Spatial Map — {self.sample_selector.currentText()} [{selected_chan}]{field_suffix}")
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(False if img_data is not None else True, linestyle='--', alpha=0.4)
        self.spatial_canvas.draw()

    def update_single_field_plot(self):
        ax = self.field_canvas.axes
        ax.clear()

        selected_field = self.field_selector.currentText()

        if not selected_field or selected_field == "-- All Fields --":
            ax.text(0.5, 0.5, "Select a specific field from the 'Field Selection' dropdown to view image overlay.",
                    ha='center', va='center', transform=ax.transAxes, fontsize=11)
            self.field_canvas.draw()
            return

        if selected_field not in self.field_dirs_map:
            ax.text(0.5, 0.5, f"Directory for field '{selected_field}' not found.",
                    ha='center', va='center', transform=ax.transAxes, fontsize=11)
            self.field_canvas.draw()
            return

        field_dir = self.field_dirs_map[selected_field]

        img_extensions = ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg']
        found_images = []
        for ext in img_extensions:
            found_images.extend(glob.glob(os.path.join(field_dir, ext)))
            found_images.extend(glob.glob(os.path.join(field_dir, "results", ext)))

        img_data = None
        if found_images:
            img_path = found_images[0]
            try:
                img_data = np.array(Image.open(img_path))
            except Exception as e:
                print(f"Error loading field image {img_path}: {e}")

        if img_data is not None:
            if img_data.ndim == 3 and img_data.shape[0] < 10:
                img_data = img_data[0]
            ax.imshow(img_data, cmap='gray', origin='lower')

        summary_df = self.get_filtered_summary_df()
        if not summary_df.empty:
            allowed_ids = set(summary_df[summary_df['field'] == selected_field]['global_track_id'].unique())
            df_field = self.trajectories_df[self.trajectories_df['global_track_id'].isin(allowed_ids)]

            for track_id, track in df_field.groupby('global_track_id'):
                channel = track['channel'].iloc[0]
                color = '#00FFFF' if channel == "Left Channel (L)" else '#FF00FF'
                ax.plot(track['x'], track['y'], color=color, alpha=0.8, linewidth=1.0)

        selected_chan = self.channel_selector.currentText() if self.channel_selector.isVisible() else "Single Channel"
        ax.set_title(f"Single Field Inspector: {selected_field} [{selected_chan}]")
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")
        ax.grid(False if img_data is not None else True)

        self.field_canvas.draw()

    def plot_intensity_distribution(self, summary_df):
        ax = self.stoich_canvas.axes
        ax.clear()

        if summary_df.empty:
            self.stoich_canvas.draw()
            return

        bins_count = self.spin_bins.value()

        df_L = summary_df[summary_df['channel'] == "Left Channel (L)"]
        df_R = summary_df[summary_df['channel'] == "Right Channel (R)"]

        if not df_L.empty:
            vals_L = df_L['mean_intensity'].dropna().values
            lbl_L = "Left Channel (L)" if self.channel_selector.isVisible() else "Single Channel Trajectories"
            n_L, bins_L, _ = ax.hist(vals_L, bins=bins_count, color='#008080', alpha=0.6, label=lbl_L, edgecolor='black')

            if self.chk_kde.isChecked() and len(vals_L) > 1:
                x_grid = np.linspace(vals_L.min(), vals_L.max(), 300)
                kde = gaussian_kde(vals_L)
                b_width = np.mean(np.diff(bins_L)) if len(bins_L) > 1 else 1.0
                y_kde = kde(x_grid) * len(vals_L) * b_width
                ax.plot(x_grid, y_kde, color='#004D4D', linewidth=2, label="Intensity KDE")

        if not df_R.empty and self.channel_selector.isVisible():
            vals_R = df_R['mean_intensity'].dropna().values
            n_R, bins_R, _ = ax.hist(vals_R, bins=bins_count, color='#D81B60', alpha=0.5, label="Right Channel (R)", edgecolor='black')

            if self.chk_kde.isChecked() and len(vals_R) > 1:
                x_grid = np.linspace(vals_R.min(), vals_R.max(), 300)
                kde = gaussian_kde(vals_R)
                b_width = np.mean(np.diff(bins_R)) if len(bins_R) > 1 else 1.0
                y_kde = kde(x_grid) * len(vals_R) * b_width
                ax.plot(x_grid, y_kde, color='#880e4f', linewidth=2, label="Right Channel KDE")

        if self.chk_filter_intensity.isChecked():
            isingle = self.spin_isingle.value()
            ax.axvline(isingle, color='black', linestyle='--', linewidth=1.5, label=f"iSingle ({isingle:.1f})")
            ax.axvline(0.5 * isingle, color='red', linestyle=':', linewidth=1.2, label="0.5 × iSingle")
            ax.axvline(1.5 * isingle, color='red', linestyle=':', linewidth=1.2, label="1.5 × iSingle")

        selected_chan = self.channel_selector.currentText() if self.channel_selector.isVisible() else "Single Channel"
        ax.set_title(f"Mean Intensity Distribution per Trajectory [{selected_chan}]")
        ax.set_xlabel("Integrated Intensity (a.u.)")
        ax.set_ylabel("Frequency")
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.5)

        self.stoich_canvas.draw()