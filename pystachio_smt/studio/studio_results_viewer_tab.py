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
try:
    from .spatial_map_tab import SpatialMapTab
except ImportError:
    from spatial_map_tab import SpatialMapTab

try:
    from .single_field_inspector_tab import SingleFieldInspectorTab
except ImportError:
    from single_field_inspector_tab import SingleFieldInspectorTab


# =========================================================================
# BACKGROUND WORKER THREAD FOR RESPONSIVE MODEL FITTING
# =========================================================================

class FitWorkerThread(QThread):
    """Runs jump distance or diffusion MLE fitting in a background thread with cancellation."""
    progress_signal = pyqtSignal(int, int, str)
    finished_signal = pyqtSignal(list)
    error_signal = pyqtSignal(str)

    def __init__(
        self,
        fit_type,
        data,
        max_comp,
        dt=None,
        loc_error=0.0,
        model_type=None,
        fix_d_static=False,
        d_static=0.045,
        fix_shape=False,
        shape_val=4.0,
        n_bootstrap=100,
        parent=None,
    ):
        super().__init__(parent)
        self.fit_type = fit_type  # 'diffusion' or 'jump'
        self.data = data
        self.max_comp = max_comp
        self.dt = dt
        self.loc_error = loc_error
        self.model_type = model_type
        self.fix_d_static = fix_d_static
        self.d_static = d_static
        self.fix_shape = fix_shape
        self.shape_val = shape_val
        self.n_bootstrap = n_bootstrap
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

                def _boot_progress(b_cur, b_tot, comp=k):
                    self.progress_signal.emit(
                        comp - 1,
                        self.max_comp,
                        f"Component {comp}/{self.max_comp}: bootstrap {b_cur}/{b_tot}..."
                    )

                if self.fit_type == 'diffusion':
                    res = fitting.fit_gamma_diffusion_mle(
                        self.data,
                        num_components=k,
                        fix_d_static=self.fix_d_static,
                        d_static=self.d_static,
                        fix_shape=self.fix_shape,
                        shape_val=self.shape_val,
                        n_bootstrap=self.n_bootstrap,
                        cancel_check=self.is_canceled,
                        progress_callback=_boot_progress,
                    )
                elif self.fit_type == 'jump':
                    res = fitting.fit_jump_distances_mle(
                        self.data,
                        dt=self.dt,
                        num_components=k,
                        loc_error=self.loc_error,
                        model_type=self.model_type,
                        fix_d_static=self.fix_d_static,
                        d_static=self.d_static,
                        n_bootstrap=self.n_bootstrap,
                        cancel_check=self.is_canceled,
                        progress_callback=_boot_progress,
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
    INTENSITY_PLOT_CHOICES = [
        ("Spot Intensity Histogram", "spot_hist"),
        ("Trajectory Mean Intensity Histogram", "traj_hist"),
        ("Log10(Spot Intensity) vs Frame Number", "scatter"),
        ("Field iSingle Peaks Boxplot", "boxplot"),
        ("All Intensity Plots (2x2 Grid)", "all"),
        ("None (Disable Plotting)", "none"),
    ]

    def __init__(self, parent=None, root_dir="."):
        super().__init__(parent)
        self.parent = parent

        self.root_dir = os.path.abspath(root_dir) if root_dir else ""
        self.loc_precision = None  # μm
        self.config_channel = "Left Channel (L)"

        # Data containers
        self.trajectories_df = pd.DataFrame()
        self.diffusion_summary_df = pd.DataFrame()
        self.diff_loc_precision_map = {}  # global_track_id -> diffusion coeff loc precision
        self.radial_filter_circle = {}    # Saved radial filter circle geometry per field
        self.field_dirs_map = {}  # field_name -> field_directory_path
        self.field_isingle_data = []  # Scanned field i_single values
        self.field_checkboxes = {}  # field_name -> QCheckBox for multi-field selection
        self.applied_selected_fields = []

        # Fitting results cache
        self.diff_fit_results = []   # List of fit dicts for diffusion models 1..N
        self.best_diff_model = None
        self.jump_fit_results = []   # List of fit dicts for jump distance models 1..N
        self.best_jump_model = None

        # Dirty flags per tab for lazy on-demand rendering
        self._tab_dirty = {}

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

        # ================= RIGHT SIDEBAR (SCROLLABLE OPTIONS PANELS) =================
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

        # 3. Field Selection Dropdown & Multi-Field Checkbox Control
        field_group = QGroupBox("Field Selection")
        field_group_layout = QVBoxLayout()
        field_group_layout.setSpacing(6)

        field_form = QFormLayout()
        field_form.setVerticalSpacing(6)
        self.field_selector = NoScrollComboBox()
        self.field_selector.currentIndexChanged.connect(self.on_field_changed)
        field_form.addRow("Field Name:", self.field_selector)
        field_group_layout.addLayout(field_form)

        self.selected_fields_container = QWidget()
        sel_fields_layout = QVBoxLayout(self.selected_fields_container)
        sel_fields_layout.setContentsMargins(0, 4, 0, 0)
        sel_fields_layout.setSpacing(4)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        self.btn_select_all_fields = QPushButton("Select All")
        self.btn_select_all_fields.setStyleSheet("font-size: 10px; padding: 2px 6px;")
        self.btn_select_all_fields.clicked.connect(self.select_all_fields)

        self.btn_deselect_all_fields = QPushButton("Deselect All")
        self.btn_deselect_all_fields.setStyleSheet("font-size: 10px; padding: 2px 6px;")
        self.btn_deselect_all_fields.clicked.connect(self.deselect_all_fields)

        btn_row.addWidget(self.btn_select_all_fields)
        btn_row.addWidget(self.btn_deselect_all_fields)
        sel_fields_layout.addLayout(btn_row)

        self.fields_checkbox_scroll = QScrollArea()
        self.fields_checkbox_scroll.setWidgetResizable(True)
        self.fields_checkbox_scroll.setMaximumHeight(160)
        self.fields_checkbox_scroll.setStyleSheet("QScrollArea { border: 1px solid #ccc; border-radius: 3px; background: #fafafa; }")

        self.fields_checkbox_host = QWidget()
        self.fields_checkbox_layout = QVBoxLayout(self.fields_checkbox_host)
        self.fields_checkbox_layout.setContentsMargins(6, 4, 6, 4)
        self.fields_checkbox_layout.setSpacing(3)
        self.fields_checkbox_layout.addStretch()

        self.fields_checkbox_scroll.setWidget(self.fields_checkbox_host)
        sel_fields_layout.addWidget(self.fields_checkbox_scroll)

        self.btn_update_selected_fields = QPushButton("Update Selected Fields")
        self.btn_update_selected_fields.setStyleSheet(
            "font-weight: bold; background-color: #008080; color: white; padding: 5px;"
        )
        self.btn_update_selected_fields.clicked.connect(self.apply_selected_fields)
        sel_fields_layout.addWidget(self.btn_update_selected_fields)

        self.selected_fields_container.setVisible(False)
        field_group_layout.addWidget(self.selected_fields_container)

        field_group.setLayout(field_group_layout)
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

        self.spin_start_frame = NoScrollSpinBox()
        self.spin_start_frame.setRange(0, 1000000)
        self.spin_start_frame.setValue(0)
        self.spin_start_frame.setToolTip("Minimum start frame for whole trajectories")

        self.spin_end_frame = NoScrollSpinBox()
        self.spin_end_frame.setRange(0, 1000000)
        self.spin_end_frame.setValue(1000000)
        self.spin_end_frame.setToolTip("Maximum end frame for whole trajectories")

        self.spin_min_diff_loc_prec = NoScrollDoubleSpinBox()
        self.spin_min_diff_loc_prec.setDecimals(4)
        self.spin_min_diff_loc_prec.setRange(0.0, 100.0)
        self.spin_min_diff_loc_prec.setValue(0.0)
        self.spin_min_diff_loc_prec.setSingleStep(0.005)
        self.spin_min_diff_loc_prec.setSuffix(" μm²/s")
        self.spin_min_diff_loc_prec.setToolTip(
            "Minimum diffusion coefficient localisation precision from *_diff_coeff_loc_precision_data.tsv (0.0 = Off)"
        )

        self.spin_min_spot_prec = NoScrollDoubleSpinBox()
        self.spin_min_spot_prec.setDecimals(4)
        self.spin_min_spot_prec.setRange(0.0, 10000.0)
        self.spin_min_spot_prec.setValue(0.0)
        self.spin_min_spot_prec.setSingleStep(0.005)
        self.spin_min_spot_prec.setSuffix(" px")
        self.spin_min_spot_prec.setToolTip(
            "Minimum mean spot precision sqrt(x_prec^2 + y_prec^2) from trajectories file (0.0 = Off)"
        )

        self.spin_max_radial_dist = NoScrollDoubleSpinBox()
        self.spin_max_radial_dist.setDecimals(1)
        self.spin_max_radial_dist.setRange(0.0, 200.0)
        self.spin_max_radial_dist.setValue(0.0)
        self.spin_max_radial_dist.setSingleStep(5.0)
        self.spin_max_radial_dist.setSuffix(" %")
        self.spin_max_radial_dist.setToolTip(
            "Maximum mean radial distance from the field center as a percentage of field half-width (0.0 = Off)"
        )

        self.chk_remove_neg = QCheckBox("Remove Negative D (D ≤ 0): 0")
        self.chk_remove_neg.setChecked(False)

        # SINGLE BUTTON TO APPLY ALL TRACK FILTERS IN THIS BOX
        self.btn_apply_track_filters = QPushButton("Apply Track Filters")
        self.btn_apply_track_filters.setStyleSheet("font-weight: bold; background-color: #008080; color: white; padding: 6px;")
        self.btn_apply_track_filters.clicked.connect(self.recalculate_and_refresh)

        track_layout.addRow("Min Track Length:", self.spin_min_len)
        track_layout.addRow("Min SNR Threshold:", self.spin_min_snr)
        track_layout.addRow("Min Track Start Frame:", self.spin_start_frame)
        track_layout.addRow("Max Track End Frame:", self.spin_end_frame)
        track_layout.addRow("Min D Loc Precision:", self.spin_min_diff_loc_prec)
        track_layout.addRow("Min Spot Precision:", self.spin_min_spot_prec)
        track_layout.addRow("Max Radial Dist (Center):", self.spin_max_radial_dist)
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

        self.chk_fix_d_static_diff = QCheckBox("Fix Static D (State 1)")
        self.chk_fix_d_static_diff.setChecked(False)

        self.spin_d_static_diff = NoScrollDoubleSpinBox()
        self.spin_d_static_diff.setDecimals(4)
        self.spin_d_static_diff.setRange(0.0001, 100.0)
        self.spin_d_static_diff.setValue(0.0450)
        self.spin_d_static_diff.setSingleStep(0.005)
        self.spin_d_static_diff.setSuffix(" μm²/s")
        self.spin_d_static_diff.setEnabled(False)
        self.chk_fix_d_static_diff.toggled.connect(self.spin_d_static_diff.setEnabled)
        self.chk_fix_d_static_diff.toggled.connect(self.clear_diff_fits)
        self.spin_d_static_diff.valueChanged.connect(self.clear_diff_fits)

        self.chk_fix_shape_diff = QCheckBox("Fix Gamma Shape (α)")
        self.chk_fix_shape_diff.setChecked(False)

        self.spin_shape_diff = NoScrollDoubleSpinBox()
        self.spin_shape_diff.setDecimals(2)
        self.spin_shape_diff.setRange(0.50, 50.0)
        self.spin_shape_diff.setValue(4.00)
        self.spin_shape_diff.setSingleStep(0.50)
        self.spin_shape_diff.setEnabled(False)
        self.chk_fix_shape_diff.toggled.connect(self.spin_shape_diff.setEnabled)
        self.chk_fix_shape_diff.toggled.connect(self.clear_diff_fits)
        self.spin_shape_diff.valueChanged.connect(self.clear_diff_fits)

        self.spin_n_bootstrap_diff = NoScrollSpinBox()
        self.spin_n_bootstrap_diff.setRange(0, 2000)
        self.spin_n_bootstrap_diff.setValue(100)
        self.spin_n_bootstrap_diff.setSingleStep(25)
        self.spin_n_bootstrap_diff.setSuffix(" runs")
        self.spin_n_bootstrap_diff.setToolTip("Non-parametric bootstrap resamples for parameter error bars (0 = Off)")

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
        diff_fit_layout.addRow(self.chk_fix_d_static_diff, self.spin_d_static_diff)
        diff_fit_layout.addRow(self.chk_fix_shape_diff, self.spin_shape_diff)
        diff_fit_layout.addRow("Bootstrap Resamples:", self.spin_n_bootstrap_diff)
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

        self.chk_fix_d_static_jump = QCheckBox("Fix Static D (State 1)")
        self.chk_fix_d_static_jump.setChecked(False)

        self.spin_d_static_jump = NoScrollDoubleSpinBox()
        self.spin_d_static_jump.setDecimals(4)
        self.spin_d_static_jump.setRange(0.0001, 100.0)
        self.spin_d_static_jump.setValue(0.0450)
        self.spin_d_static_jump.setSingleStep(0.005)
        self.spin_d_static_jump.setSuffix(" μm²/s")
        self.spin_d_static_jump.setEnabled(False)
        self.chk_fix_d_static_jump.toggled.connect(self.spin_d_static_jump.setEnabled)
        self.chk_fix_d_static_jump.toggled.connect(self.clear_jump_fits)
        self.spin_d_static_jump.valueChanged.connect(self.clear_jump_fits)

        self.spin_n_bootstrap_jump = NoScrollSpinBox()
        self.spin_n_bootstrap_jump.setRange(0, 2000)
        self.spin_n_bootstrap_jump.setValue(100)
        self.spin_n_bootstrap_jump.setSingleStep(25)
        self.spin_n_bootstrap_jump.setSuffix(" runs")
        self.spin_n_bootstrap_jump.setToolTip("Non-parametric bootstrap resamples for parameter error bars (0 = Off)")

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
        jump_fit_layout.addRow(self.chk_fix_d_static_jump, self.spin_d_static_jump)
        jump_fit_layout.addRow("Bootstrap Resamples:", self.spin_n_bootstrap_jump)
        jump_fit_layout.addRow(self.chk_show_subcomponents_jump)
        jump_fit_layout.addRow(self.chk_show_non_best_jump)
        jump_fit_layout.addRow(self.btn_run_jump_fitting)
        jump_fit_layout.addRow(self.btn_export_fits_jump)
        jump_fit_layout.addRow(self.lbl_jump_fit_results)
        self.jump_fit_group.setLayout(jump_fit_layout)
        sidebar_layout.addWidget(self.jump_fit_group)
        self.jump_fit_group.setVisible(False)

        # 7. SPATIAL MAP OPTIONS PANEL (DELEGATED TO spatial_map_tab.py)
        self.spatial_tab = SpatialMapTab(self)
        sidebar_layout.addWidget(self.spatial_tab.options_group)

        # 8. SINGLE FIELD INSPECTOR TAB & OPTIONS PANEL (DELEGATED TO single_field_inspector_tab.py)
        self.single_field_tab = SingleFieldInspectorTab(self)
        sidebar_layout.addWidget(self.single_field_tab.options_group)

        # 9. INTENSITY PLOT DISPLAY SETTINGS (SHOWN ONLY ON INTENSITY TAB)
        self.intensity_plot_group = QGroupBox("Intensity Plot Settings")
        intensity_plot_layout = QFormLayout()
        intensity_plot_layout.setVerticalSpacing(6)

        self.combo_intensity_dist = NoScrollComboBox()
        for label, code in self.INTENSITY_PLOT_CHOICES:
            self.combo_intensity_dist.addItem(label, userData=code)

        self.chk_intensity_kde = QCheckBox("Fit KDE & Detect Peak (iSingle)")
        self.chk_intensity_kde.setChecked(False)

        self.chk_log_x_intensity = QCheckBox("Log X Axis")
        self.chk_log_x_intensity.setChecked(False)

        self.chk_log_y_intensity = QCheckBox("Log Y Axis")
        self.chk_log_y_intensity.setChecked(False)

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

        intensity_plot_layout.addRow("Plot Selector:", self.combo_intensity_dist)
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

        # Dedicated Channel Filter Dropdown for MSD tab
        self.lbl_msd_channel = QLabel("Channel Filter:")
        self.combo_msd_channel = NoScrollComboBox()
        self.combo_msd_channel.addItems([
            "Both Channels (Overlay)",
            "Left Channel (L)",
            "Right Channel (R)"
        ])
        self.combo_msd_channel.currentIndexChanged.connect(lambda: self.plot_msd_vs_timelag(self.get_filtered_summary_df()))

        self.spin_msd_max_lag = NoScrollSpinBox()
        self.spin_msd_max_lag.setRange(2, 200)
        self.spin_msd_max_lag.setValue(20)
        self.spin_msd_max_lag.setSuffix(" frames")
        self.spin_msd_max_lag.valueChanged.connect(lambda: self.plot_msd_vs_timelag(self.get_filtered_summary_df()))

        # Number of initial MSD points to fit (Jin et al. 2021 uses initial 4 points by default)
        self.spin_msd_fit_points = NoScrollSpinBox()
        self.spin_msd_fit_points.setRange(2, 100)
        self.spin_msd_fit_points.setValue(4)
        self.spin_msd_fit_points.setSuffix(" points")
        self.spin_msd_fit_points.valueChanged.connect(lambda: self.plot_msd_vs_timelag(self.get_filtered_summary_df()))

        # Model selector: Linear (4D·τ) vs Confined 2D Diffusion (Jin et al. 2021)
        self.combo_msd_model = NoScrollComboBox()
        self.combo_msd_model.addItems([
            "Linear Fit (4D·τ + 4σ²)",
            "Confined 2D Diffusion (Circular Domain r)"
        ])
        self.combo_msd_model.currentIndexChanged.connect(lambda: self.plot_msd_vs_timelag(self.get_filtered_summary_df()))

        self.chk_msd_show_individual = QCheckBox("Show Individual Trajectory MSDs")
        self.chk_msd_show_individual.setChecked(True)
        self.chk_msd_show_individual.stateChanged.connect(lambda: self.plot_msd_vs_timelag(self.get_filtered_summary_df()))

        self.chk_msd_fit_linear = QCheckBox("Fit Ensemble MSD Model")
        self.chk_msd_fit_linear.setChecked(True)
        self.chk_msd_fit_linear.stateChanged.connect(lambda: self.plot_msd_vs_timelag(self.get_filtered_summary_df()))

        self.lbl_msd_fit_results = QLabel("MSD fit summary will appear here.")
        self.lbl_msd_fit_results.setWordWrap(True)
        self.lbl_msd_fit_results.setStyleSheet("font-size: 10px; background-color: #f8f9fa; padding: 6px; border: 1px solid #ccc; border-radius: 4px;")

        msd_opt_layout.addRow(self.lbl_msd_channel, self.combo_msd_channel)
        msd_opt_layout.addRow("Max Time Lag:", self.spin_msd_max_lag)
        msd_opt_layout.addRow("Fit Points (N):", self.spin_msd_fit_points)
        msd_opt_layout.addRow("MSD Model:", self.combo_msd_model)
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

        # Synchronized Cutoff Signal Connections
        self.spin_isingle_fraction.valueChanged.connect(self.on_isingle_fraction_changed)
        self.spin_frame_cutoff.valueChanged.connect(self.on_frame_cutoff_changed)

        self.combo_intensity_dist.currentIndexChanged.connect(self._on_sidebar_intensity_plot_changed)
        self.chk_intensity_kde.stateChanged.connect(self.refresh_all_plots)
        self.chk_log_x_intensity.stateChanged.connect(self.refresh_all_plots)
        self.chk_log_y_intensity.stateChanged.connect(self.refresh_all_plots)
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

        # ================= LEFT PANEL (PLOTS) =================
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
        self.tabs.addTab(self.spatial_tab, "Spatial Map")

        # TAB 4: Single Field Inspector (DELEGATED TO single_field_inspector_tab.py)
        self.tabs.addTab(self.single_field_tab, "Single Field Inspector")

        # TAB 5: Intensity Distribution (with Top Bar Plot Selector)
        self.tab_stoich = QWidget()
        layout_stoich = QVBoxLayout(self.tab_stoich)

        intensity_top_bar = QHBoxLayout()
        intensity_top_bar.setSpacing(8)
        lbl_intensity_sel = QLabel("Plot Selector:")
        lbl_intensity_sel.setStyleSheet("font-weight: bold;")

        self.combo_intensity_top = NoScrollComboBox()
        self.combo_intensity_top.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        for label, code in self.INTENSITY_PLOT_CHOICES:
            self.combo_intensity_top.addItem(label, userData=code)
        self.combo_intensity_top.currentIndexChanged.connect(self._on_top_intensity_plot_changed)

        intensity_top_bar.addWidget(lbl_intensity_sel)
        intensity_top_bar.addWidget(self.combo_intensity_top, 1)

        self.stoich_canvas = MplCanvas(self)
        self.stoich_toolbar = NavigationToolbar(self.stoich_canvas, self)
        layout_stoich.addLayout(intensity_top_bar)
        layout_stoich.addWidget(self.stoich_toolbar)
        layout_stoich.addWidget(self.stoich_canvas)
        self.tabs.addTab(self.tab_stoich, "Intensity Distribution")

        # TAB 6: MSD vs Time Lag
        self.tab_msd = QWidget()
        self.msd_canvas = MplCanvas(self)
        self.msd_toolbar = NavigationToolbar(self.msd_canvas, self)
        layout_msd = QVBoxLayout(self.tab_msd)
        layout_msd.addWidget(self.msd_toolbar)
        layout_msd.addWidget(self.msd_canvas)
        self.tabs.addTab(self.tab_msd, "MSD vs Time Lag")

        # Add scrollable options panel to the RIGHT side of the splitter
        splitter.addWidget(scroll_area)
        splitter.setSizes([960, 340])

        self.tabs.currentChanged.connect(self.on_tab_changed)
        self.on_tab_changed(self.tabs.currentIndex())

    def _on_top_intensity_plot_changed(self, index):
        self.combo_intensity_dist.blockSignals(True)
        self.combo_intensity_dist.setCurrentIndex(index)
        self.combo_intensity_dist.blockSignals(False)
        self._render_tab_widget(self.tab_stoich)

    def _on_sidebar_intensity_plot_changed(self, index):
        self.combo_intensity_top.blockSignals(True)
        self.combo_intensity_top.setCurrentIndex(index)
        self.combo_intensity_top.blockSignals(False)
        self._render_tab_widget(self.tab_stoich)

    def on_tab_changed(self, index):
        """Swaps sidebar control boxes to match the active tab, synchronizes spatial map coloring mode,
        and renders the newly selected tab if its plot is out-of-date."""
        active_tab = self.tabs.widget(index)
        is_diff = active_tab is self.tab_diff
        is_jump = active_tab is self.tab_jump
        is_spatial = active_tab is self.spatial_tab
        is_field = active_tab is self.single_field_tab
        is_intensity = active_tab is self.tab_stoich
        is_msd = active_tab is self.tab_msd

        if is_diff:
            self.active_analysis_mode = 'diffusion'
            self.spatial_tab.sync_from_analysis_mode('diffusion', redraw=False)
        elif is_jump:
            self.active_analysis_mode = 'jump'
            self.spatial_tab.sync_from_analysis_mode(
                'jump', jump_lag=self.spin_jump_lag.value(), redraw=False
            )
        elif is_spatial:
            mode = getattr(self, 'active_analysis_mode', 'diffusion')
            self.spatial_tab.sync_from_analysis_mode(
                mode, jump_lag=self.spin_jump_lag.value(), redraw=True
            )
            self._tab_dirty[self.spatial_tab] = False

        self.diff_hist_group.setVisible(is_diff)
        self.diff_fit_group.setVisible(is_diff)
        self.jump_hist_group.setVisible(is_jump)
        self.jump_fit_group.setVisible(is_jump)
        self.spatial_tab.options_group.setVisible(is_spatial)
        self.single_field_tab.options_group.setVisible(is_field)
        self.intensity_plot_group.setVisible(is_intensity)
        self.msd_options_group.setVisible(is_msd)

        if self._tab_dirty.get(active_tab, False):
            self._render_tab_widget(active_tab)

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

# ================= BIDIRECTIONAL CUTOFF SYNCHRONIZATION HELPERS =================

    def get_max_dataset_frame(self):
        """Returns maximum frame in the loaded trajectory dataset."""
        if not self.trajectories_df.empty and 'frame' in self.trajectories_df.columns:
            return max(1, int(self.trajectories_df['frame'].max()))
        return 1

    def on_isingle_fraction_changed(self, value):
        """Recalculates frame cutoff number when percentage fraction changes."""
        max_f = self.get_max_dataset_frame()
        calc_cutoff = int(np.round(max_f * (value / 100.0)))

        self.spin_frame_cutoff.blockSignals(True)
        self.spin_frame_cutoff.setValue(calc_cutoff)
        self.spin_frame_cutoff.blockSignals(False)
        self.refresh_all_plots()

    def on_frame_cutoff_changed(self, value):
        """Recalculates percentage fraction when frame cutoff number changes."""
        max_f = self.get_max_dataset_frame()
        calc_frac = float(np.clip((value / max_f) * 100.0, 0.0, 100.0))

        self.spin_isingle_fraction.blockSignals(True)
        self.spin_isingle_fraction.setValue(calc_frac)
        self.spin_isingle_fraction.blockSignals(False)
        self.refresh_all_plots()

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

        min_start = self.spin_start_frame.value()
        max_end = self.spin_end_frame.value()
        if min_start > 0 or max_end < self.get_max_dataset_frame():
            track_bounds = active_df.groupby('global_track_id')['frame'].agg(['min', 'max'])
            valid_ids = track_bounds[
                (track_bounds['min'] >= min_start) & (track_bounds['max'] <= max_end)
            ].index
            active_df = active_df[active_df['global_track_id'].isin(valid_ids)]

        active_df = self.filter_df_by_field(active_df)

        min_f = self.spin_min_frame.value()
        max_f = self.spin_max_frame.value()
        active_df = active_df[(active_df['frame'] >= min_f) & (active_df['frame'] <= max_f)]
        active_df = self.filter_df_by_field(active_df)

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

        cutoff_frame = self.spin_frame_cutoff.value()

        # Auto-check frame cutoff box so plot KDE matches iSingle calculation subset
        self.chk_apply_frame_cutoff.blockSignals(True)
        self.chk_apply_frame_cutoff.setChecked(True)
        self.chk_apply_frame_cutoff.blockSignals(False)

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

    @staticmethod
    def _format_param_with_errors(val, fit_err, boot_err, decimals=3, is_fixed=False, unit=""):
        """Formats a fitted parameter alongside its asymptotic MLE (fit) and bootstrap (boot) errors."""
        suffix = f" {unit}" if unit else ""
        if is_fixed:
            return f"{val:.{decimals}f}{suffix} (fixed)"

        parts = [f"{val:.{decimals}f}"]
        err_parts = []
        if fit_err is not None and np.isfinite(fit_err):
            err_parts.append(f"± {fit_err:.{decimals}f} fit")
        if boot_err is not None and np.isfinite(boot_err):
            err_parts.append(f"± {boot_err:.{decimals}f} boot")

        if err_parts:
            parts.append(f"({' / '.join(err_parts)})")
        return f"{' '.join(parts)}{suffix}"

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
            fix_d_static=self.chk_fix_d_static_diff.isChecked(),
            d_static=self.spin_d_static_diff.value(),
            fix_shape=self.chk_fix_shape_diff.isChecked(),
            shape_val=self.spin_shape_diff.value(),
            n_bootstrap=self.spin_n_bootstrap_diff.value(),
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

        valid_models = [m for m in self.diff_fit_results if not m.get('has_degenerate', False)]
        if valid_models:
            self.best_diff_model = min(valid_models, key=lambda x: x['bic'])
        else:
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
            is_best = (k == best_diff_k)
            is_degen = res.get('has_degenerate', False)
            degen_tag = " [Degenerate <1%]" if is_degen else ""
            mark = "★ BEST" if is_best else f"(ΔBIC: +{delta_bic:.1f}){degen_tag}"
            summary_lines.append(f"• <b>{k}-Comp:</b> BIC = {bic:.1f} {mark}")

            if is_best:
                for idx, p in enumerate(res['params'], 1):
                    w = p.get('weight', 1.0) * 100.0
                    w_fit = p.get('weight_fit_err', 0.0) * 100.0
                    w_boot = p.get('weight_boot_err', np.nan) * 100.0
                    w_str = self._format_param_with_errors(
                        w, w_fit, w_boot, decimals=1, is_fixed=(k == 1), unit="%"
                    )

                    mean_d = p.get('mean_D', p.get('D', 0.0))
                    d_str = self._format_param_with_errors(
                        mean_d,
                        p.get('D_fit_err', 0.0),
                        p.get('D_boot_err', np.nan),
                        decimals=4 if p.get('is_static', False) else 3,
                        is_fixed=p.get('is_static', False),
                        unit="μm²/s",
                    )

                    shape_v = p.get('shape', p.get('n', 4.0))
                    shape_str = self._format_param_with_errors(
                        shape_v,
                        p.get('shape_fit_err', 0.0),
                        p.get('shape_boot_err', np.nan),
                        decimals=2,
                        is_fixed=p.get('is_fixed_shape', False),
                    )

                    state_tag = " [Static]" if p.get('is_static', False) else ""
                    summary_lines.append(
                        f"  └ State {idx}{state_tag}: {w_str}<br>"
                        f"     D = {d_str}<br>"
                        f"     α (shape) = {shape_str}"
                    )

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
            fix_d_static=self.chk_fix_d_static_jump.isChecked(),
            d_static=self.spin_d_static_jump.value(),
            n_bootstrap=self.spin_n_bootstrap_jump.value(),
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
                summary_lines.append(f"📊 <b>Proportions:</b> Pure Brownian: <b>{pure_pct:.1f}%</b> | Anomalous: <b>{anom_pct:.1f}%</b>")
            else:
                anom_p = best_params[-1]
                anom_w = anom_p.get('weight', 0.0) * 100.0
                anom_fit = anom_p.get('weight_fit_err', 0.0) * 100.0
                anom_boot = anom_p.get('weight_boot_err', np.nan) * 100.0
                pure_w = sum(p.get('weight', 0.0) for p in best_params[:-1]) * 100.0
                pure_str = self._format_param_with_errors(pure_w, anom_fit, anom_boot, decimals=1, unit="%")
                anom_str = self._format_param_with_errors(anom_w, anom_fit, anom_boot, decimals=1, unit="%")
                summary_lines.append(f"📊 <b>Proportions:</b> Pure: <b>{pure_str}</b> | Anomalous: <b>{anom_str}</b>")

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
                    w_fit = p.get('weight_fit_err', 0.0) * 100.0
                    w_boot = p.get('weight_boot_err', np.nan) * 100.0
                    w_str = self._format_param_with_errors(
                        w, w_fit, w_boot, decimals=1, is_fixed=(k == 1), unit="%"
                    )

                    d_val = p.get('D', p.get('mean_D', 0.0))
                    d_str = self._format_param_with_errors(
                        d_val,
                        p.get('D_fit_err', 0.0),
                        p.get('D_boot_err', np.nan),
                        decimals=4 if p.get('is_static', False) else 3,
                        is_fixed=p.get('is_static', False),
                        unit="μm²/s",
                    )

                    alpha_val = p.get('alpha', 1.0)
                    alpha_str = self._format_param_with_errors(
                        alpha_val,
                        p.get('alpha_fit_err', 0.0),
                        p.get('alpha_boot_err', np.nan),
                        decimals=2,
                        is_fixed=False,
                    )

                    static_tag = ", Static" if p.get('is_static', False) else ""

                    if selected_model == "Mixed (Pure Brownian + Anomalous)":
                        is_anom = (best_j_k > 1) and (idx == len(res['params']))
                        comp_type = "Anomalous" if is_anom else f"Pure{static_tag}"
                        if is_anom:
                            summary_lines.append(
                                f"  └ State {idx} ({comp_type}): {w_str}<br>"
                                f"     D = {d_str}<br>"
                                f"     α = {alpha_str}"
                            )
                        else:
                            summary_lines.append(
                                f"  └ State {idx} ({comp_type}): {w_str}<br>"
                                f"     D = {d_str}"
                            )
                    elif selected_model == "Anomalous 2D Diffusion":
                        state_tag = " [Static]" if p.get('is_static', False) else ""
                        summary_lines.append(
                            f"  └ State {idx}{state_tag}: {w_str}<br>"
                            f"     D = {d_str}<br>"
                            f"     α = {alpha_str}"
                        )
                    else:
                        state_tag = " [Static]" if p.get('is_static', False) else ""
                        summary_lines.append(
                            f"  └ State {idx}{state_tag}: {w_str}<br>"
                            f"     D = {d_str}"
                        )

        self.lbl_jump_fit_results.setText("<br>".join(summary_lines))
        self.refresh_all_plots()

    def export_fit_parameters_npy(self):
        """Exports model fit parameters, fitting/bootstrap errors, and acquisition metadata to a NumPy .npy binary file."""
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

        field_meta = (
            f"Selected Fields ({', '.join(self.get_selected_fields())})"
            if self.field_selector.currentText() == "-- Selected Fields --"
            else self.field_selector.currentText()
        )

        export_dict = {
            'metadata': {
                'sample': self.sample_selector.currentText(),
                'channel': (
                    self.channel_selector.currentText()
                    if self.channel_selector.isVisible()
                    else "Single Channel"
                ),
                'field': field_meta,
                'pixel_size_um': self.spin_pixel_size.value(),
                'frame_interval_ms': self.spin_exposure.value(),
                'alex_mode': self.chk_alex.isChecked(),
                'localisation_precision_um': self.loc_precision,
                'min_track_length': self.spin_min_len.value(),
                'min_snr_threshold': self.spin_min_snr.value(),
                'min_track_start_frame': self.spin_start_frame.value(),
                'max_track_end_frame': self.spin_end_frame.value(),
                'min_diff_loc_precision': self.spin_min_diff_loc_prec.value(),
                'min_spot_precision': self.spin_min_spot_prec.value(),
                'max_radial_dist_pct': self.spin_max_radial_dist.value(),
                'radial_filter_circle': self.radial_filter_circle,
                'negative_D_removed': self.chk_remove_neg.isChecked(),
                'jdd_model': self.combo_jdd_model.currentText(),
                'jump_lag_frames': self.spin_jump_lag.value(),
                'spatial_jump_lag_frames': self.spatial_tab.spin_spatial_jump_lag.value(),
                'spatial_color_mode': self.spatial_tab.combo_spatial_color_mode.currentText(),
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
                'fix_d_static_diff': self.chk_fix_d_static_diff.isChecked(),
                'd_static_diff_um2_s': self.spin_d_static_diff.value(),
                'fix_shape_diff': self.chk_fix_shape_diff.isChecked(),
                'fixed_shape_val_diff': self.spin_shape_diff.value(),
                'n_bootstrap_diff': self.spin_n_bootstrap_diff.value(),
                'fix_d_static_jump': self.chk_fix_d_static_jump.isChecked(),
                'd_static_jump_um2_s': self.spin_d_static_jump.value(),
                'n_bootstrap_jump': self.spin_n_bootstrap_jump.value(),
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
                        'aic': fit.get('aic'),
                        'log_likelihood': fit.get(
                            'log_likelihood',
                            fit.get('log_lh', None)
                        ),
                        'has_degenerate': fit.get('has_degenerate', False),
                        'params': sanitize_params(
                            fit.get('params', [])
                        )
                    }
                    for fit in self.diff_fit_results
                ],
                'best_model': {
                    'components': self.best_diff_model.get('components'),
                    'bic': self.best_diff_model.get('bic'),
                    'aic': self.best_diff_model.get('aic'),
                    'log_likelihood': self.best_diff_model.get(
                        'log_likelihood',
                        self.best_diff_model.get('log_lh', None)
                    ),
                    'params': sanitize_params(
                        self.best_diff_model.get('params', [])
                    )
                } if self.best_diff_model else None
            },
            'jump_distance_fits': {
                'all_models': [
                    {
                        'model_type': fit.get('model_type', self.combo_jdd_model.currentText()),
                        'components': fit.get('components'),
                        'bic': fit.get('bic'),
                        'aic': fit.get('aic'),
                        'log_likelihood': fit.get(
                            'log_likelihood',
                            fit.get('log_lh', None)
                        ),
                        'has_degenerate': fit.get('has_degenerate', False),
                        'params': sanitize_params(
                            fit.get('params', [])
                        )
                    }
                    for fit in self.jump_fit_results
                ],
                'best_model': {
                    'model_type': self.best_jump_model.get('model_type', self.combo_jdd_model.currentText()),
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

    # ================= TRANSFORMATION MATRIX HELPERS =================

    def find_transformation_matrix(self, field_name=None):
        """Looks for transformation_matrix.npy first in relevant field results folder, then in directory where config is located."""
        candidates = []
        if field_name and field_name in self.field_dirs_map:
            for s_dir in self.get_search_dirs_for_field(field_name):
                candidates.append(os.path.join(s_dir, "transformation_matrix.npy"))

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

    def _get_checked_fields(self):
        """Returns list of field names currently checked in the UI checkboxes."""
        return [fname for fname, cb in self.field_checkboxes.items() if cb.isChecked()]

    def get_selected_fields(self):
        """Returns list of field names currently applied in the multi-field selector."""
        return list(self.applied_selected_fields)

    def _update_selected_fields_button_state(self):
        """Updates the Update Selected Fields button appearance when pending changes exist."""
        if not hasattr(self, 'btn_update_selected_fields'):
            return
        has_pending = (self._get_checked_fields() != self.applied_selected_fields)
        if has_pending:
            self.btn_update_selected_fields.setText("Update Selected Fields*")
            self.btn_update_selected_fields.setStyleSheet(
                "font-weight: bold; background-color: #d97706; color: white; padding: 5px;"
            )
        else:
            self.btn_update_selected_fields.setText("Update Selected Fields")
            self.btn_update_selected_fields.setStyleSheet(
                "font-weight: bold; background-color: #008080; color: white; padding: 5px;"
            )

    def filter_df_by_field(self, df):
        """Filters a DataFrame containing a 'field' column according to the active Field Selection mode."""
        if df.empty or 'field' not in df.columns:
            return df
        selected_field = self.field_selector.currentText()
        if selected_field == "-- Selected Fields --":
            return df[df['field'].isin(self.get_selected_fields())]
        elif selected_field and selected_field != "-- All Fields --":
            return df[df['field'] == selected_field]
        return df

    def get_field_title_suffix(self):
        """Builds a concise title suffix describing the active field selection."""
        selected_field = self.field_selector.currentText()
        if not selected_field or selected_field == "-- All Fields --":
            return ""
        if selected_field == "-- Selected Fields --":
            chosen = self.get_selected_fields()
            total = len(self.field_checkboxes)
            if len(chosen) <= 3:
                return f" | Fields: {', '.join(chosen) if chosen else 'None'}"
            return f" | Selected Fields ({len(chosen)}/{total})"
        return f" | Field: {selected_field}"

    def select_all_fields(self):
        for cb in self.field_checkboxes.values():
            cb.blockSignals(True)
            cb.setChecked(True)
            cb.blockSignals(False)
        self._update_selected_fields_button_state()

    def deselect_all_fields(self):
        for cb in self.field_checkboxes.values():
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)
        self._update_selected_fields_button_state()

    def apply_selected_fields(self):
        """Applies checked fields and refreshes fits and plots."""
        self.applied_selected_fields = self._get_checked_fields()
        self._update_selected_fields_button_state()
        self.clear_fits()
        self.refresh_all_plots()

    def on_field_changed(self):
        is_multi = (self.field_selector.currentText() == "-- Selected Fields --")
        self.selected_fields_container.setVisible(is_multi)
        self.clear_fits()
        self.populate_field_images()
        self.refresh_all_plots()

    def update_field_selector_items(self):
        self.field_selector.blockSignals(True)
        self.field_selector.clear()

        # Clear existing checkboxes
        for cb in self.field_checkboxes.values():
            self.fields_checkbox_layout.removeWidget(cb)
            cb.deleteLater()
        self.field_checkboxes.clear()
        self.applied_selected_fields = []

        if not self.trajectories_df.empty:
            selected_sample = self.sample_selector.currentText()
            if selected_sample == "All Samples Combined" or not selected_sample:
                fields = sorted(self.trajectories_df['field'].unique().tolist())
            else:
                sample_df = self.trajectories_df[self.trajectories_df['sample'] == selected_sample]
                fields = sorted(sample_df['field'].unique().tolist())

            self.field_selector.addItem("-- All Fields --")
            if len(fields) > 1:
                self.field_selector.addItem("-- Selected Fields --")

            for f in fields:
                self.field_selector.addItem(f)
                cb = QCheckBox(f)
                cb.setChecked(True)
                cb.stateChanged.connect(self._update_selected_fields_button_state)
                self.fields_checkbox_layout.insertWidget(self.fields_checkbox_layout.count() - 1, cb)
                self.field_checkboxes[f] = cb

            self.applied_selected_fields = list(fields)

        self._update_selected_fields_button_state()
        self.selected_fields_container.setVisible(False)
        self.field_selector.blockSignals(False)
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
        cached_path = getattr(QApplication.instance(), '_pystachio_selected_config', None)

        if isinstance(cached_path, str) and cached_path in parsed_configs:
            selected_cfg = parsed_configs[cached_path]
        elif len(parsed_configs) == 1:
            selected_cfg = list(parsed_configs.values())[0]
        elif len(parsed_configs) > 1:
            if self.has_config_conflicts(parsed_configs):
                # Silently use the first config instead of opening a second popup
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

        # Scan Diffusion Coefficient Localisation Precision files (*_diff_coeff_loc_precision_data.tsv)
        self.diff_loc_precision_map = {}
        lp_pattern = os.path.join(self.root_dir, "**", "*diff_coeff_loc_precision_data*.tsv")
        lp_files = glob.glob(lp_pattern, recursive=True)

        for lp_path in lp_files:
            norm_lp = os.path.normpath(lp_path)
            lp_parts = norm_lp.split(os.sep)
            if any(part.startswith("cell_") or part.startswith("cell") for part in lp_parts[:-1]):
                continue

            lp_filename = os.path.basename(lp_path)
            lp_parent_dir = os.path.dirname(norm_lp)
            lp_parent_folder = os.path.basename(lp_parent_dir)

            if lp_parent_folder == "results":
                lp_field_dir = os.path.dirname(lp_parent_dir)
                lp_field_name = os.path.basename(lp_field_dir)
            elif lp_parent_folder.endswith("_results"):
                lp_field_name = lp_parent_folder[:-8].rstrip('_')
            else:
                lp_field_name = lp_parent_folder[:-8].rstrip('_') if lp_parent_folder.endswith("_results") else lp_parent_folder

            if "R_channel" in lp_filename or "_R_" in lp_filename or "_R." in lp_filename:
                lp_chan_code = "R"
            else:
                lp_chan_code = "L"

            try:
                lp_df = pd.read_csv(lp_path, sep='\t', header=None)
                if lp_df.shape[1] >= 2:
                    tids = pd.to_numeric(lp_df.iloc[:, 0], errors='coerce')
                    vals = pd.to_numeric(lp_df.iloc[:, 1], errors='coerce')
                    for tid_val, lp_val in zip(tids, vals):
                        if pd.notna(tid_val) and pd.notna(lp_val):
                            tid_str = str(int(round(tid_val))) if float(tid_val).is_integer() else str(tid_val)
                            g_id = f"{lp_field_name}_{lp_chan_code}_{tid_str}"
                            self.diff_loc_precision_map[g_id] = float(lp_val)
            except Exception as e:
                print(f"Error reading diffusion loc precision file {lp_path}: {e}")

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
            parent_dir = os.path.dirname(norm_path)
            parent_folder = os.path.basename(parent_dir)

            # 1. Nested preprocess directory: .../<Sample>/<Field_01>/results/
            if parent_folder == "results":
                field_dir = os.path.dirname(parent_dir)                      # .../<Sample>/<Field_01>
                field_name = os.path.basename(field_dir)                     # Field_01
                sample_dir = os.path.dirname(field_dir)                      # .../<Sample>
                sample_label = os.path.basename(sample_dir)

            # 2. PySTACHIO batch directory: .../<Sample>/<Field_01_results>/
            elif parent_folder.endswith("_results"):
                field_dir = parent_dir                                       # .../<Sample>/<Field_01_results>
                field_name = parent_folder[:-8].rstrip('_')                  # Field_01
                sample_dir = os.path.dirname(parent_dir)                     # .../<Sample>
                sample_label = os.path.basename(sample_dir)

            # 3. Fallback
            else:
                field_name = parent_folder[:-8].rstrip('_') if parent_folder.endswith("_results") else parent_folder
                field_dir = parent_dir
                sample_dir = os.path.dirname(parent_dir)
                sample_label = os.path.basename(sample_dir)

            # Store both field directory and sample directory for flexible image lookup
            self.field_dirs_map[field_name] = {
                "traj_dir": field_dir,
                "sample_dir": sample_dir
            }

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

            # Dynamically set whole-track frame bounds and synchronized cutoff frame
            if 'frame' in self.trajectories_df.columns and not self.trajectories_df.empty:
                max_dataset_frame = int(self.trajectories_df['frame'].max())

                self.spin_start_frame.blockSignals(True)
                self.spin_start_frame.setValue(0)
                self.spin_start_frame.blockSignals(False)

                self.spin_end_frame.blockSignals(True)
                self.spin_end_frame.setValue(max_dataset_frame)
                self.spin_end_frame.blockSignals(False)

                initial_cutoff = int(np.round(max_dataset_frame * (self.spin_isingle_fraction.value() / 100.0)))
                self.spin_frame_cutoff.blockSignals(True)
                self.spin_frame_cutoff.setValue(initial_cutoff)
                self.spin_frame_cutoff.blockSignals(False)

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
            search_dirs = self.get_search_dirs_for_field(field_name)

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

    def populate_field_images(self):
        """Delegates image list population to SingleFieldInspectorTab."""
        if hasattr(self, 'single_field_tab'):
            self.single_field_tab.populate_field_images()

    # ================= DIFFUSION & SNR CALCULATIONS =================

    def _get_field_geometry(self, fields):
        """Returns a dict mapping field_name -> {'center_x', 'center_y', 'ref_radius_px', 'width', 'height'}."""
        geom_map = {}
        default_frame_size = self.get_parameter_default('frame_size', None)
        for field in fields:
            geom = None
            for s_dir in self.get_search_dirs_for_field(field):
                for pat in ("*L_avg*.[tT][iI][fF]*", "*R_avg*.[tT][iI][fF]*", "*BF*.[tT][iI][fF]*", "*.[tT][iI][fF]*"):
                    matches = glob.glob(os.path.join(s_dir, pat))
                    if matches:
                        try:
                            with Image.open(matches[0]) as im:
                                w, h = im.size
                                if w > 0 and h > 0:
                                    geom = {
                                        'center_x': w / 2.0,
                                        'center_y': h / 2.0,
                                        'ref_radius_px': min(w, h) / 2.0,
                                        'width': float(w),
                                        'height': float(h),
                                    }
                                    break
                        except Exception:
                            pass
                if geom is not None:
                    break

            if geom is None and default_frame_size is not None:
                try:
                    if isinstance(default_frame_size, (tuple, list)) and len(default_frame_size) >= 2:
                        w, h = float(default_frame_size[0]), float(default_frame_size[1])
                        if w > 0 and h > 0:
                            geom = {
                                'center_x': w / 2.0,
                                'center_y': h / 2.0,
                                'ref_radius_px': min(w, h) / 2.0,
                                'width': w,
                                'height': h,
                            }
                    elif isinstance(default_frame_size, (int, float)) and default_frame_size > 0:
                        w = h = float(default_frame_size)
                        geom = {
                            'center_x': w / 2.0,
                            'center_y': h / 2.0,
                            'ref_radius_px': w / 2.0,
                            'width': w,
                            'height': h,
                        }
                except Exception:
                    pass

            if geom is None and not self.trajectories_df.empty:
                f_df = self.trajectories_df[self.trajectories_df['field'] == field]
                if not f_df.empty and 'x' in f_df.columns and 'y' in f_df.columns:
                    min_x, max_x = float(f_df['x'].min()), float(f_df['x'].max())
                    min_y, max_y = float(f_df['y'].min()), float(f_df['y'].max())
                    cx = (min_x + max_x) / 2.0
                    cy = (min_y + max_y) / 2.0
                    ref_r = max((max_x - min_x) / 2.0, (max_y - min_y) / 2.0, 1.0)
                    geom = {
                        'center_x': cx,
                        'center_y': cy,
                        'ref_radius_px': ref_r,
                        'width': max_x - min_x,
                        'height': max_y - min_y,
                    }

            geom_map[field] = geom if geom is not None else {
                'center_x': 0.0,
                'center_y': 0.0,
                'ref_radius_px': 1.0,
                'width': 0.0,
                'height': 0.0,
            }
        return geom_map

    def _save_radial_filter_circle(self, field_geom_map):
        """Builds and saves the radial filter circle specification as radial_filter_circle.npy for SpatialMapTab overlay."""
        radial_pct = float(self.spin_max_radial_dist.value())
        fields_circles = {}
        for field, geom in field_geom_map.items():
            ref_r = float(geom.get('ref_radius_px', 0.0))
            radius_px = (radial_pct / 100.0) * ref_r if (radial_pct > 0 and ref_r > 0) else 0.0
            fields_circles[field] = {
                'center_x': float(geom.get('center_x', 0.0)),
                'center_y': float(geom.get('center_y', 0.0)),
                'radius_px': float(radius_px),
                'radius_pct': radial_pct,
                'ref_radius_px': ref_r,
                'width': float(geom.get('width', 0.0)),
                'height': float(geom.get('height', 0.0)),
            }

        first_circle = next(iter(fields_circles.values()), {
            'center_x': 0.0,
            'center_y': 0.0,
            'radius_px': 0.0,
            'radius_pct': radial_pct,
            'ref_radius_px': 0.0,
        })

        self.radial_filter_circle = {
            'enabled': radial_pct > 0,
            'radius_pct': radial_pct,
            'center_x': first_circle['center_x'],
            'center_y': first_circle['center_y'],
            'radius_px': first_circle['radius_px'],
            'ref_radius_px': first_circle['ref_radius_px'],
            'fields': fields_circles,
        }

        if radial_pct > 0 and self.root_dir and os.path.exists(self.root_dir):
            try:
                out_root = os.path.join(self.root_dir, "radial_filter_circle.npy")
                np.save(out_root, self.radial_filter_circle, allow_pickle=True)
            except Exception as e:
                print(f"Error saving radial_filter_circle.npy to root_dir: {e}")

            for field, c_data in fields_circles.items():
                for s_dir in self.get_search_dirs_for_field(field):
                    try:
                        np.save(os.path.join(s_dir, "radial_filter_circle.npy"), c_data, allow_pickle=True)
                        break
                    except Exception:
                        pass

    def get_radial_filter_circle(self, field_name=None):
        """Returns circle dict {'center_x', 'center_y', 'radius_px', 'radius_pct'} for the requested field,
        checking active state first and falling back to saved radial_filter_circle.npy on disk."""
        data = self.radial_filter_circle if isinstance(self.radial_filter_circle, dict) else {}

        if not data or data.get('radius_px', 0.0) <= 0:
            candidate_paths = []
            if field_name and field_name in self.field_dirs_map:
                for s_dir in self.get_search_dirs_for_field(field_name):
                    candidate_paths.append(os.path.join(s_dir, "radial_filter_circle.npy"))
            if self.root_dir:
                candidate_paths.append(os.path.join(self.root_dir, "radial_filter_circle.npy"))

            for p in candidate_paths:
                if os.path.exists(p):
                    try:
                        loaded = np.load(p, allow_pickle=True)
                        if hasattr(loaded, 'item'):
                            loaded = loaded.item()
                        if isinstance(loaded, dict):
                            data = loaded
                            break
                    except Exception as e:
                        print(f"Error loading radial_filter_circle.npy from {p}: {e}")

        if not data:
            return None

        fields_dict = data.get('fields', {})
        if field_name and field_name in fields_dict:
            c_info = fields_dict[field_name]
        elif fields_dict:
            c_info = next(iter(fields_dict.values()))
        else:
            c_info = data

        if c_info.get('radius_px', 0.0) > 0:
            return c_info
        return None

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

        field_geom_map = self._get_field_geometry(active_df['field'].unique())
        self._save_radial_filter_circle(field_geom_map)

        # Detect spot precision columns in trajectories DataFrame
        prec_col_pairs = [
            ('x_precision', 'y_precision'),
            ('precision_x', 'precision_y'),
            ('x_prec', 'y_prec'),
            ('sigma_x', 'sigma_y'),
            ('loc_precision_x', 'loc_precision_y'),
        ]
        x_prec_col, y_prec_col = None, None
        lower_cols = {c.lower(): c for c in active_df.columns}
        for cx, cy in prec_col_pairs:
            if cx in lower_cols and cy in lower_cols:
                x_prec_col, y_prec_col = lower_cols[cx], lower_cols[cy]
                break

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
            start_frame = int(track_sorted['frame'].iloc[0])
            end_frame = int(track_sorted['frame'].iloc[-1])

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

            diff_loc_prec = self.diff_loc_precision_map.get(track_id, np.nan)

            if x_prec_col and y_prec_col:
                px_vals = pd.to_numeric(track_sorted[x_prec_col], errors='coerce').values
                py_vals = pd.to_numeric(track_sorted[y_prec_col], errors='coerce').values
                spot_precs = np.sqrt(px_vals**2 + py_vals**2)
                mean_spot_precision = float(np.nanmean(spot_precs)) if np.any(np.isfinite(spot_precs)) else np.nan
            else:
                mean_spot_precision = np.nan

            geom = field_geom_map.get(field, {'center_x': 0.0, 'center_y': 0.0, 'ref_radius_px': 1.0})
            cx, cy = geom['center_x'], geom['center_y']
            ref_r = geom['ref_radius_px']
            radial_dists_px = np.sqrt((track_sorted['x'].values - cx)**2 + (track_sorted['y'].values - cy)**2)
            radial_dists_pct = (radial_dists_px / ref_r) * 100.0 if ref_r > 0 else np.full_like(radial_dists_px, np.nan)
            mean_radial_dist_pct = float(np.nanmean(radial_dists_pct)) if len(radial_dists_pct) > 0 else np.nan

            results.append({
                'global_track_id': track_id,
                'sample': sample,
                'field': field,
                'channel': channel,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'D': D,
                'log_D': np.log10(D) if D > 0 and not np.isnan(D) else np.nan,
                'track_length': len(track),
                'mean_intensity': mean_intensity,
                'mean_snr': mean_snr,
                'diff_loc_precision': diff_loc_prec,
                'mean_spot_precision': mean_spot_precision,
                'mean_radial_dist_pct': mean_radial_dist_pct
            })

        self.diffusion_summary_df = pd.DataFrame(results)

        num_negative = (self.diffusion_summary_df['D'] <= 0).sum() if not self.diffusion_summary_df.empty else 0
        self.chk_remove_neg.setText(f"Remove Negative D (D ≤ 0): {num_negative}")
        self.refresh_all_plots()

    def get_filtered_summary_df(self):
        """Filters trajectory summary DataFrame based on Channel, Min SNR, Start/End Frame, Loc Precision, Spot Precision, Radial Distance, Negative D, Intensity (per channel), and Field filters."""
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

        # Whole-trajectory Start/End Frame Filter
        if 'start_frame' in df.columns and 'end_frame' in df.columns:
            min_start = self.spin_start_frame.value()
            max_end = self.spin_end_frame.value()
            df = df[(df['start_frame'] >= min_start) & (df['end_frame'] <= max_end)]

        # Diffusion Coefficient Localisation Precision Filter (Minimum threshold, 0.0 = Off)
        if 'diff_loc_precision' in df.columns:
            min_diff_lp = self.spin_min_diff_loc_prec.value()
            if min_diff_lp > 0:
                df = df[df['diff_loc_precision'].notna() & (df['diff_loc_precision'] >= min_diff_lp)]

        # Spot Precision Filter sqrt(x_prec^2 + y_prec^2) (Minimum threshold, 0.0 = Off)
        if 'mean_spot_precision' in df.columns:
            min_spot_prec = self.spin_min_spot_prec.value()
            if min_spot_prec > 0:
                df = df[df['mean_spot_precision'].notna() & (df['mean_spot_precision'] >= min_spot_prec)]

        # Radial Distance from Field Center Filter in % (0.0 = Off)
        if 'mean_radial_dist_pct' in df.columns:
            max_radial_pct = self.spin_max_radial_dist.value()
            if max_radial_pct > 0:
                df = df[df['mean_radial_dist_pct'].notna() & (df['mean_radial_dist_pct'] <= max_radial_pct)]

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

        df = self.filter_df_by_field(df)

        return df

    # ================= PLOTTING LOGIC =================

    def _render_tab_widget(self, tab_widget, summary_df=None):
        """Renders only the specified tab widget and clears its dirty flag."""
        if summary_df is None:
            summary_df = self.get_filtered_summary_df()

        if tab_widget is self.tab_diff:
            self.plot_diffusion_histogram(summary_df)
        elif tab_widget is self.tab_jump:
            self.plot_jump_distance_histogram(summary_df)
        elif tab_widget is self.spatial_tab:
            self.plot_spatial_map(summary_df)
        elif tab_widget is self.single_field_tab:
            self.update_single_field_plot()
        elif tab_widget is self.tab_stoich:
            self.plot_intensity_distribution(summary_df)
        elif tab_widget is self.tab_msd:
            self.plot_msd_vs_timelag(summary_df)

        self._tab_dirty[tab_widget] = False

    def refresh_all_plots(self):
        summary_df = self.get_filtered_summary_df()

        if not self.diffusion_summary_df.empty:
            total_raw = self.trajectories_df['global_track_id'].nunique() if not self.trajectories_df.empty else 0
            valid_tracks = len(self.diffusion_summary_df)
            displayed_tracks = len(summary_df)
            neg_count = (self.diffusion_summary_df['D'] <= 0).sum()

            selected_field_txt = self.field_selector.currentText() or "-- All Fields --"
            if selected_field_txt == "-- Selected Fields --":
                field_label = f"Selected ({len(self.get_selected_fields())}/{len(self.field_checkboxes)})"
            else:
                field_label = selected_field_txt

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
            track_bounds_label = f"{self.spin_start_frame.value()}–{self.spin_end_frame.value()}"
            diff_lp_val = self.spin_min_diff_loc_prec.value()
            diff_lp_label = f"≥ {diff_lp_val:.4f} μm²/s" if diff_lp_val > 0 else "Off"
            spot_prec_val = self.spin_min_spot_prec.value()
            spot_prec_label = f"≥ {spot_prec_val:.4f} px" if spot_prec_val > 0 else "Off"
            radial_val = self.spin_max_radial_dist.value()
            radial_label = f"≤ {radial_val:.1f}%" if radial_val > 0 else "Off"

            self.lbl_stats.setText(
                f"<b>Scope:</b> {self.sample_selector.currentText()}<br>"
                f"<b>Channel Filter:</b> {chan_label}<br>"
                f"<b>Field:</b> {field_label}<br>"
                f"<b>Total Raw Tracks:</b> {total_raw}<br>"
                f"<b>Track Start/End Bounds:</b> {track_bounds_label}<br>"
                f"<b>Valid Length Tracks:</b> {valid_tracks}<br>"
                f"<b>SNR Filter:</b> {snr_filter_label}<br>"
                f"<b>D Loc Precision Filter:</b> {diff_lp_label}<br>"
                f"<b>Spot Precision Filter:</b> {spot_prec_label}<br>"
                f"<b>Radial Center Filter:</b> {radial_label}<br>"
                f"<b>Negative/Zero D Tracks:</b> {neg_count}<br>"
                f"<b>Intensity Filter:</b> {intensity_filter_label}<br>"
                f"<b>Displayed Tracks:</b> {displayed_tracks}"
            )

        # Mark all tabs dirty, then immediately render only the currently visible tab
        for i in range(self.tabs.count()):
            self._tab_dirty[self.tabs.widget(i)] = True

        active_tab = self.tabs.currentWidget()
        if active_tab is not None:
            self._render_tab_widget(active_tab, summary_df=summary_df)

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
                            d_err = p.get('D_boot_err', np.nan)
                            if not np.isfinite(d_err):
                                d_err = p.get('D_fit_err', 0.0)
                            c_color = comp_colors[c_idx % len(comp_colors)]
                            d_tag = f"{mean_d:.3f} (fixed)" if p.get('is_static', False) else f"{mean_d:.3f}±{d_err:.3f}"
                            c_label = f"  └ State {c_idx+1}: D={d_tag}, {w*100:.1f}%"
                            
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

        field_suffix = self.get_field_title_suffix()

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
                        d_err = p.get('D_boot_err', np.nan)
                        if not np.isfinite(d_err):
                            d_err = p.get('D_fit_err', 0.0)
                        c_color = comp_colors[c_idx % len(comp_colors)]
                        d_tag = f"{d_val:.3f} (fixed)" if p.get('is_static', False) else f"{d_val:.3f}±{d_err:.3f}"
                        c_label = f"  └ State {c_idx+1}: D={d_tag}, {w*100:.1f}%"
                        
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

        field_suffix = self.get_field_title_suffix()

        ax.set_title(f"Jump Distance Distribution ($\Delta t = {time_lag_ms:.1f}\mathrm{{ms}}$, {lag_N} frames) — {self.sample_selector.currentText()} [{selected_chan}]{field_suffix}", fontsize=11)
        ax.set_xlabel(r"Jump Distance $r$ $[\mu\mathrm{m}]$", fontsize=11)
        ax.set_ylabel("Jump Frequency / Count", fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(False)

        self.jump_canvas.draw()

    def get_search_dirs_for_field(self, field_name):
        """Returns all potential image/result directories for a given field name, checking:
        1. Preprocess outputs: <Sample>/<Field>/results/ (e.g. R_avg.tif)
        2. Tracking outputs:   <Sample>/<Field>_results/
        3. Field root:         <Sample>/<Field>/
        """
        if not field_name or field_name not in self.field_dirs_map:
            return []

        info = self.field_dirs_map[field_name]
        if isinstance(info, dict):
            sample_dir = info.get("sample_dir", self.root_dir)
            traj_dir = info.get("traj_dir", "")
        else:
            traj_dir = info
            sample_dir = os.path.dirname(traj_dir)

        search_dirs = []
        if sample_dir and field_name:
            search_dirs.append(os.path.join(sample_dir, field_name, "results"))   # Preprocess: <Sample>/<Field>/results
            search_dirs.append(os.path.join(sample_dir, f"{field_name}_results")) # Tracking:   <Sample>/<Field>_results
            search_dirs.append(os.path.join(sample_dir, field_name))               # Field root:  <Sample>/<Field>

        if traj_dir and traj_dir not in search_dirs:
            search_dirs.append(traj_dir)
            search_dirs.append(os.path.join(traj_dir, "results"))

        return [d for d in list(dict.fromkeys(search_dirs)) if os.path.exists(d)]

    def plot_spatial_map(self, summary_df):
        """Delegates spatial map rendering to SpatialMapTab."""
        self.spatial_tab.plot(summary_df)

    def update_single_field_plot(self):
        """Delegates Single Field Inspector rendering to SingleFieldInspectorTab."""
        if hasattr(self, 'single_field_tab'):
            self.single_field_tab.plot()

    # ================= INTENSITY PLOTTING & INDIVIDUAL LOG AXES =================

    def plot_intensity_distribution(self, summary_df):
        fig = self.stoich_canvas.fig
        fig.clf()

        mode = self.combo_intensity_dist.currentData() or "spot_hist"

        # Fast exit if plotting is disabled via dropdown selector
        if mode == "none":
            ax = fig.add_subplot(1, 1, 1)
            ax.text(
                0.5, 0.5,
                "Intensity Distribution plotting is paused ('None' selected).\n"
                "Choose a plot from the Plot Selector dropdown above to render.",
                ha='center', va='center', transform=ax.transAxes, fontsize=10, color='#555'
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            self.stoich_canvas.draw()
            return

        use_log_x = self.chk_log_x_intensity.isChecked()
        use_log_y = self.chk_log_y_intensity.isChecked()
        has_isingle_data = bool(self.field_isingle_data)
        fit_kde = self.chk_intensity_kde.isChecked()

        # Build axes layout based on drop-down plot selector
        if mode in ("spot_hist", "traj_hist", "scatter", "boxplot"):
            axes_list = [fig.add_subplot(1, 1, 1)]
            plot_modes = [mode]
        else:  # "all" -> All Intensity Plots (2x2 Grid)
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

        selected_chan = self.channel_selector.currentText() if self.channel_selector.isVisible() else "Single Channel"
        selected_chan_lbl = selected_chan
        bins_count = self.spin_bins.value()

        # Lazily slice spot-level DataFrame ONLY when a spot-level plot is actually produced
        needs_spots = any(m in ("spot_hist", "scatter") for m in plot_modes)
        spots_df_filter = pd.DataFrame()
        intensity_col = None

        if needs_spots:
            selected_sample = self.sample_selector.currentText()
            if selected_sample == "All Samples Combined" or not selected_sample:
                spots_df = self.trajectories_df
            else:
                spots_df = self.trajectories_df[self.trajectories_df['sample'] == selected_sample]

            min_start = self.spin_start_frame.value()
            max_end = self.spin_end_frame.value()
            if min_start > 0 or max_end < self.get_max_dataset_frame():
                track_bounds = spots_df.groupby('global_track_id')['frame'].agg(['min', 'max'])
                valid_ids = track_bounds[
                    (track_bounds['min'] >= min_start) & (track_bounds['max'] <= max_end)
                ].index
                spots_df = spots_df[spots_df['global_track_id'].isin(valid_ids)]

            spots_df = self.filter_df_by_field(spots_df)

            intensity_col = 'integrated_intensity' if 'integrated_intensity' in spots_df.columns else (
                'intensity' if 'intensity' in spots_df.columns else (
                    'spot_intensity' if 'spot_intensity' in spots_df.columns else None
                )
            )

            if self.chk_apply_frame_cutoff.isChecked() and self.spin_frame_cutoff.value() > 0:
                cutoff = self.spin_frame_cutoff.value()
                spots_df = spots_df[spots_df['frame'] >= cutoff]

            if selected_chan == "Left Channel (L)":
                spots_df_filter = spots_df[spots_df['channel'] == "Left Channel (L)"]
            elif selected_chan == "Right Channel (R)":
                spots_df_filter = spots_df[spots_df['channel'] == "Right Channel (R)"]
            else:
                spots_df_filter = spots_df

        for current_mode, ax in zip(plot_modes, axes_list):
            ax.grid(False)

            if use_log_x and current_mode != "boxplot":
                ax.set_xscale('log')
            if use_log_y:
                ax.set_yscale('log')

            if current_mode == "spot_hist":
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

            elif current_mode == "traj_hist":
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

                ax.set_title(f"Trajectory Mean Intensity Distribution [{selected_chan_lbl}]", fontsize=10)
                ax.set_xlabel("Mean Integrated Intensity (a.u.)", fontsize=10)
                ax.set_ylabel("Trajectory Count", fontsize=10)
                ax.legend(loc='upper right', fontsize=8)

            elif current_mode == "scatter":
                if not intensity_col or spots_df_filter.empty:
                    ax.text(0.5, 0.5, "No spot intensity data", ha='center', va='center', transform=ax.transAxes)
                    continue

                valid_spots = spots_df_filter[(spots_df_filter[intensity_col] > 0) & np.isfinite(spots_df_filter['frame'])]

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

            elif current_mode == "boxplot":
                if not has_isingle_data:
                    ax.text(0.5, 0.5, "No scanned field iSingle .dat files found", ha='center', va='center', transform=ax.transAxes)
                    continue

                df_is = pd.DataFrame(self.field_isingle_data)

                if self.channel_selector.isVisible():
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
        """Calculates and plots Mean Squared Displacement (MSD) vs Time Lag (tau) for active selection.
        Differentiates L and R channel colors, filters channels using the MSD channel selector,
        and delegates model fitting to fitting.fit_msd_curve().
        """
        ax = self.msd_canvas.axes
        ax.clear()

        if self.trajectories_df.empty or summary_df.empty:
            ax.grid(False)
            self.msd_canvas.draw()
            return

        allowed_ids = set(summary_df['global_track_id'].unique())
        df = self.trajectories_df[self.trajectories_df['global_track_id'].isin(allowed_ids)]

        # Filter tracks by MSD tab's dedicated channel selector
        msd_chan_sel = self.combo_msd_channel.currentText() if hasattr(self, 'combo_msd_channel') else "Both Channels (Overlay)"
        if msd_chan_sel == "Left Channel (L)":
            df = df[df['channel'] == "Left Channel (L)"]
        elif msd_chan_sel == "Right Channel (R)":
            df = df[df['channel'] == "Right Channel (R)"]

        if df.empty:
            ax.grid(False)
            self.msd_canvas.draw()
            return

        pixel_size = self.spin_pixel_size.value()
        dt_ms = self.spin_exposure.value()
        is_alex = self.chk_alex.isChecked()
        dt_sec = (dt_ms / 1000.0) * (2.0 if is_alex else 1.0)

        max_lag = self.spin_msd_max_lag.value()
        fit_n_points = self.spin_msd_fit_points.value() if hasattr(self, 'spin_msd_fit_points') else 4
        selected_model = self.combo_msd_model.currentText() if hasattr(self, 'combo_msd_model') else "Linear Fit (4D·τ + 4σ²)"
        show_indiv = self.chk_msd_show_individual.isChecked()
        fit_enabled = self.chk_msd_fit_linear.isChecked()

        lags = np.arange(1, max_lag + 1)
        tau_times = lags * dt_sec

        channels_in_data = df['channel'].unique().tolist()

        # Color definitions: L = Teal (#008080), R = Crimson (#D81B60)
        chan_colors = {
            "Left Channel (L)": {'indiv': '#008080', 'avg': '#008080', 'fit': '#004D4D'},
            "Right Channel (R)": {'indiv': '#D81B60', 'avg': '#D81B60', 'fit': '#880e4f'},
            "Single Channel": {'indiv': '#008080', 'avg': '#008080', 'fit': '#004D4D'}
        }

        fit_summary_lines = []
        loc_prec = self.loc_precision if self.loc_precision is not None else 0.04

        for chan in channels_in_data:
            c_info = chan_colors.get(chan, chan_colors["Single Channel"])
            df_chan = df[df['channel'] == chan]

            msd_per_lag = {lag: [] for lag in lags}
            valid_tracks_count = 0

            for track_id, track in df_chan.groupby('global_track_id'):
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
                    ax.plot(tau_times, track_msds, color=c_info['indiv'], alpha=0.12, linewidth=0.7)

            avg_msd = np.array([np.mean(msd_per_lag[lag]) if len(msd_per_lag[lag]) > 0 else np.nan for lag in lags])
            valid_mask = ~np.isnan(avg_msd)

            if np.any(valid_mask):
                ax.plot(tau_times[valid_mask], avg_msd[valid_mask], 'o-', color=c_info['avg'], linewidth=2.2, markersize=5, label=f"Ensemble Avg [{chan}] (n={valid_tracks_count})")

                # DELEGATED FIT TO fitting.py
                if fit_enabled:
                    fit_res = fitting.fit_msd_curve(
                        tau_times, avg_msd,
                        model_type=selected_model,
                        num_points=fit_n_points,
                        loc_err=loc_prec
                    )

                    if fit_res is not None:
                        d_fit = fit_res['D']
                        fit_lbl = (
                            f"Confined Fit [{chan}]: $D={d_fit:.3f}\\ \\mu\\mathrm{{m}}^2/\\mathrm{{s}}$, $r={fit_res['r_nm']:.0f}\\ \\mathrm{{nm}}$"
                            if fit_res['model_type'] == 'Confined 2D'
                            else f"Linear Fit [{chan}]: $D={d_fit:.3f}\\ \\mu\\mathrm{{m}}^2/\\mathrm{{s}}$"
                        )
                        ax.plot(tau_times, fit_res['fit_curve'], '--', color=c_info['fit'], linewidth=2.0, label=fit_lbl)
                        fit_summary_lines.append(f"• <b>{chan} ({fit_res['model_type']} {fit_res['num_points']} pts):</b> {fit_res['summary']}")
                    else:
                        fit_summary_lines.append(f"• <b>{chan}:</b> Fit failed or insufficient points.")

        if fit_summary_lines:
            self.lbl_msd_fit_results.setText("<b>MSD Model Fit Summary:</b><br>" + "<br>".join(fit_summary_lines))
        else:
            self.lbl_msd_fit_results.setText("No model fit performed.")

        selected_chan = msd_chan_sel
        field_suffix = self.get_field_title_suffix()

        ax.set_title(f"Mean Squared Displacement (MSD) vs Time Lag — {self.sample_selector.currentText()} [{selected_chan}]{field_suffix}", fontsize=11)
        ax.set_xlabel(r"Time Lag $\tau$ [s]", fontsize=11)
        ax.set_ylabel(r"MSD $\langle r^2(\tau) \rangle$ $[\mu\mathrm{m}^2]$", fontsize=11)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(False)

        self.msd_canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ResultsViewerTab(root_dir=".")
    viewer.setWindowTitle("PySTACHIO Studio - Results Viewer")
    viewer.resize(1300, 850)
    viewer.show()
    sys.exit(app.exec())
