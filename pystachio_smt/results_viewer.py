import sys
import os
import glob
import re
import argparse
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QPushButton, QFileDialog, QLabel, QComboBox,
    QSplitter, QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox,
    QCheckBox, QMessageBox, QDialog
)
from PyQt5.QtCore import Qt

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from scipy.stats import gaussian_kde
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


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

        self.combo = QComboBox()
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


class PyStachioResultsViewer(QMainWindow):
    def __init__(self, root_dir):
        super().__init__()
        self.setWindowTitle("PySTACHIO Results Viewer — Diffusion & Channel Analysis")
        self.setGeometry(100, 100, 1300, 850)

        self.root_dir = os.path.abspath(root_dir)
        self.loc_precision = None  # μm

        # Data containers
        self.trajectories_df = pd.DataFrame()
        self.diffusion_summary_df = pd.DataFrame()
        self.field_dirs_map = {}  # field_name -> field_directory_path

        self._init_ui()
        self.scan_and_load_directory()

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # ================= LEFT SIDEBAR =================
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)

        # Directory Info Group
        env_group = QGroupBox("Directory Info")
        env_layout = QFormLayout()
        self.lbl_root_dir = QLabel(self.root_dir)
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

        # Dataset & Channel Filter
        filter_group = QGroupBox("Sample & Channel Selection")
        filter_layout = QFormLayout()

        self.sample_selector = QComboBox()
        self.sample_selector.currentIndexChanged.connect(self.on_sample_changed)

        self.channel_selector = QComboBox()
        self.channel_selector.addItems(["Both Channels (Overlay)", "Left Channel (L)", "Right Channel (R)"])
        self.channel_selector.currentIndexChanged.connect(self.refresh_all_plots)

        filter_layout.addRow("Sample / Condition:", self.sample_selector)
        filter_layout.addRow("Channel View:", self.channel_selector)
        filter_group.setLayout(filter_layout)
        sidebar_layout.addWidget(filter_group)

        # Field Selection Control
        field_group = QGroupBox("Field Selection")
        field_layout = QFormLayout()

        self.field_selector = QComboBox()
        self.field_selector.currentIndexChanged.connect(self.refresh_all_plots)

        field_layout.addRow("Field Name:", self.field_selector)
        field_group.setLayout(field_layout)
        sidebar_layout.addWidget(field_group)

        # Physics & Analysis Parameters
        param_group = QGroupBox("Acquisition & Filtering Controls")
        param_layout = QFormLayout()

        self.spin_pixel_size = QDoubleSpinBox()
        self.spin_pixel_size.setDecimals(4)
        self.spin_pixel_size.setRange(0.0001, 10.0)
        self.spin_pixel_size.setValue(0.117)
        self.spin_pixel_size.setSingleStep(0.005)
        self.spin_pixel_size.setSuffix(" μm/px")

        self.spin_exposure = QDoubleSpinBox()
        self.spin_exposure.setDecimals(3)
        self.spin_exposure.setRange(0.001, 10000.0)
        self.spin_exposure.setValue(3.5)
        self.spin_exposure.setSuffix(" ms")

        self.chk_alex = QCheckBox("ALEX Imaging (Alternating Excitation)")
        self.chk_alex.setChecked(False)

        self.spin_min_len = QSpinBox()
        self.spin_min_len.setRange(2, 100)
        self.spin_min_len.setValue(4)

        self.spin_jump_lag = QSpinBox()
        self.spin_jump_lag.setRange(1, 100)
        self.spin_jump_lag.setValue(1)
        self.spin_jump_lag.setSuffix(" frames")

        self.spin_bin_size = QDoubleSpinBox()
        self.spin_bin_size.setRange(0.0001, 100.0)
        self.spin_bin_size.setDecimals(4)
        self.spin_bin_size.setValue(0.09)
        self.spin_bin_size.setSingleStep(0.005)
        self.spin_bin_size.setSuffix(" μm²/s")

        self.spin_bins = QSpinBox()
        self.spin_bins.setRange(5, 200)
        self.spin_bins.setValue(35)

        self.chk_log_scale = QCheckBox("Log10 Scale [log10(D)]")
        self.chk_log_scale.setChecked(False)

        self.chk_kde = QCheckBox("Kernel Density Estimate (KDE)")
        self.chk_kde.setChecked(False)

        self.chk_remove_neg = QCheckBox("Remove Negative D (D ≤ 0): 0")
        self.chk_remove_neg.setChecked(False)

        # Intensity Filtering Controls
        self.chk_filter_intensity = QCheckBox("Filter by Single Fluorophore Intensity")
        self.chk_filter_intensity.setChecked(False)

        self.spin_isingle = QDoubleSpinBox()
        self.spin_isingle.setDecimals(1)
        self.spin_isingle.setRange(0.0, 1000000.0)
        self.spin_isingle.setValue(1000.0)
        self.spin_isingle.setSingleStep(100.0)
        self.spin_isingle.setSuffix(" a.u.")

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
        self.chk_filter_intensity.stateChanged.connect(self.refresh_all_plots)
        self.spin_isingle.valueChanged.connect(self.refresh_all_plots)

        param_layout.addRow("Pixel Size:", self.spin_pixel_size)
        param_layout.addRow("Frame Interval:", self.spin_exposure)
        param_layout.addRow(self.chk_alex)
        param_layout.addRow("Min Track Length:", self.spin_min_len)
        param_layout.addRow("Jump Lag (N):", self.spin_jump_lag)
        param_layout.addRow("Bin Size (D):", self.spin_bin_size)
        param_layout.addRow("Histogram Bins:", self.spin_bins)
        param_layout.addRow(self.chk_log_scale)
        param_layout.addRow(self.chk_kde)
        param_layout.addRow(self.chk_remove_neg)
        param_layout.addRow(self.chk_filter_intensity)
        param_layout.addRow("iSingle Value:", self.spin_isingle)
        param_group.setLayout(param_layout)
        sidebar_layout.addWidget(param_group)

        # Summary Info Panel
        self.lbl_stats = QLabel("No data")
        self.lbl_stats.setStyleSheet("font-size: 11px; background-color: #f5f5f5; padding: 6px; border: 1px solid #ddd; border-radius: 4px;")
        self.lbl_stats.setWordWrap(True)
        sidebar_layout.addWidget(self.lbl_stats)

        sidebar_layout.addStretch()
        splitter.addWidget(sidebar)

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
        self.update_field_selector_items()
        self.recalculate_and_refresh()

    # ================= DIRECTORY & TRAJECTORY SCANNING =================

    def scan_and_load_directory(self):
        """Scans directory for config files and field results folders containing trajectories."""
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
                if dialog.exec_() == QDialog.Accepted:
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

        if not all_files:
            self.lbl_stats.setText("No *_trajectories.tsv files found.")
            self.trajectories_df = pd.DataFrame()
            self.field_dirs_map = {}
            self.populate_sample_dropdown([])
            return

        traj_list = []
        self.field_dirs_map = {}

        for traj_path in all_files:
            norm_path = os.path.normpath(traj_path)
            path_parts = norm_path.split(os.sep)

            # Skip cell-level individual track subfolders if present
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

            if "L_channel" in filename or "_L_" in filename:
                channel = "Left Channel (L)"
                chan_code = "L"
            elif "R_channel" in filename or "_R_" in filename:
                channel = "Right Channel (R)"
                chan_code = "R"
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

    def select_new_directory(self):
        selected_dir = QFileDialog.getExistingDirectory(self, "Select Experiment Directory", self.root_dir)
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
        self.recalculate_and_refresh()

    def update_field_selector_items(self):
        """Populates field dropdown selector with '-- All Fields --' and field folder names."""
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

        if not self.diffusion_summary_df.empty:
            num_negative = (self.diffusion_summary_df['D'] <= 0).sum()
        else:
            num_negative = 0

        self.chk_remove_neg.setText(f"Remove Negative D (D ≤ 0): {num_negative}")
        self.refresh_all_plots()

    def get_filtered_summary_df(self):
        """Applies field selection, negative D removal, and intensity (iSingle) range filtering."""
        if self.diffusion_summary_df.empty:
            return pd.DataFrame()

        df = self.diffusion_summary_df.copy()

        if self.chk_remove_neg.isChecked():
            df = df[df['D'] > 0]

        if self.chk_filter_intensity.isChecked():
            isingle = self.spin_isingle.value()
            low_bound = 0.5 * isingle
            high_bound = 1.5 * isingle
            df = df[(df['mean_intensity'] >= low_bound) & (df['mean_intensity'] <= high_bound)]

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
            intensity_filter_label = (
                f"Active [0.5 - 1.5 × {self.spin_isingle.value():.1f}]" if self.chk_filter_intensity.isChecked() else "Off"
            )

            self.lbl_stats.setText(
                f"<b>Scope:</b> {self.sample_selector.currentText()}<br>"
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

        channel_mode = self.channel_selector.currentText()
        use_log = self.chk_log_scale.isChecked()

        df_L = summary_df[summary_df['channel'] == "Left Channel (L)"]
        df_R = summary_df[summary_df['channel'] == "Right Channel (R)"]

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

        if channel_mode in ["Both Channels (Overlay)", "Left Channel (L)"] and not df_L.empty:
            data_L = df_L[val_col].dropna().values
            ax.hist(data_L, bins=bins, color='#008080', alpha=0.6, label=f"Left Channel (n={len(data_L)})", edgecolor='black', linewidth=0.5)

            if self.chk_kde.isChecked():
                if HAS_SCIPY and len(data_L) > 1:
                    x_grid = np.linspace(data_L.min(), data_L.max(), 300)
                    kde = gaussian_kde(data_L)
                    b_width = np.mean(np.diff(bins)) if isinstance(bins, np.ndarray) and len(bins) > 1 else (
                        (data_L.max() - data_L.min()) / self.spin_bins.value() if data_L.max() != data_L.min() else 1.0
                    )
                    y_kde = kde(x_grid) * len(data_L) * b_width
                    ax.plot(x_grid, y_kde, color='#004D4D', linewidth=2, label="Left Channel KDE")

        if channel_mode in ["Both Channels (Overlay)", "Right Channel (R)"] and not df_R.empty:
            data_R = df_R[val_col].dropna().values
            ax.hist(data_R, bins=bins, color='#D81B60', alpha=0.5, label=f"Right Channel (n={len(data_R)})", edgecolor='black', linewidth=0.5)

            if self.chk_kde.isChecked():
                if HAS_SCIPY and len(data_R) > 1:
                    x_grid = np.linspace(data_R.min(), data_R.max(), 300)
                    kde = gaussian_kde(data_R)
                    b_width = np.mean(np.diff(bins)) if isinstance(bins, np.ndarray) and len(bins) > 1 else (
                        (data_R.max() - data_R.min()) / self.spin_bins.value() if data_R.max() != data_R.min() else 1.0
                    )
                    y_kde = kde(x_grid) * len(data_R) * b_width
                    ax.plot(x_grid, y_kde, color='#880e4f', linewidth=2, label="Right Channel KDE")

        if self.chk_kde.isChecked() and not HAS_SCIPY:
            ax.text(0.02, 0.95, "SciPy is required for KDE curves.", transform=ax.transAxes, color='red', fontsize=9)

        selected_field = self.field_selector.currentText()
        field_suffix = f" | Field: {selected_field}" if selected_field and selected_field != "-- All Fields --" else ""

        ax.set_title(f"Diffusion Coefficient Distribution — {self.sample_selector.currentText()}{field_suffix}", fontsize=11)
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel("Trajectory Count", fontsize=11)
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.5)

        self.diff_canvas.draw()

    def plot_jump_distance_histogram(self, summary_df):
        """Calculates single-molecule displacement (jump distance) over specified N frame lag."""
        ax = self.jump_canvas.axes
        ax.clear()

        if self.trajectories_df.empty or summary_df.empty:
            self.jump_canvas.draw()
            return

        allowed_ids = set(summary_df['global_track_id'].unique())
        df = self.trajectories_df[self.trajectories_df['global_track_id'].isin(allowed_ids)]

        pixel_size = self.spin_pixel_size.value()
        lag_N = self.spin_jump_lag.value()
        dt_ms = self.spin_exposure.value()
        is_alex = self.chk_alex.isChecked()
        dt_sec = (dt_ms / 1000.0) * (2.0 if is_alex else 1.0)
        time_lag_ms = lag_N * dt_sec * 1000.0

        jumps_L, jumps_R = [], []

        for track_id, track in df.groupby('global_track_id'):
            if len(track) <= lag_N:
                continue

            track_sorted = track.sort_values('frame')
            frames = track_sorted['frame'].values
            x = track_sorted['x'].values * pixel_size
            y = track_sorted['y'].values * pixel_size

            # Match exact frame differences of N steps
            frame_diffs = frames[lag_N:] - frames[:-lag_N]
            valid_mask = (frame_diffs == lag_N)

            if not np.any(valid_mask):
                continue

            dx = (x[lag_N:] - x[:-lag_N])[valid_mask]
            dy = (y[lag_N:] - y[:-lag_N])[valid_mask]
            step_distances = np.sqrt(dx**2 + dy**2)

            channel = track_sorted['channel'].iloc[0]
            if channel == "Left Channel (L)":
                jumps_L.extend(step_distances)
            else:
                jumps_R.extend(step_distances)

        channel_mode = self.channel_selector.currentText()
        bins_count = self.spin_bins.value()

        if channel_mode in ["Both Channels (Overlay)", "Left Channel (L)"] and jumps_L:
            arr_L = np.array(jumps_L)
            n_L, bins_L, _ = ax.hist(arr_L, bins=bins_count, color='#008080', alpha=0.6,
                                     label=f"Left Channel (N={len(arr_L)} jumps)", edgecolor='black', linewidth=0.5)

            if self.chk_kde.isChecked() and HAS_SCIPY and len(arr_L) > 1:
                x_grid = np.linspace(arr_L.min(), arr_L.max(), 300)
                kde = gaussian_kde(arr_L)
                b_width = np.mean(np.diff(bins_L)) if len(bins_L) > 1 else 1.0
                y_kde = kde(x_grid) * len(arr_L) * b_width
                ax.plot(x_grid, y_kde, color='#004D4D', linewidth=2, label="Left Channel KDE")

        if channel_mode in ["Both Channels (Overlay)", "Right Channel (R)"] and jumps_R:
            arr_R = np.array(jumps_R)
            n_R, bins_R, _ = ax.hist(arr_R, bins=bins_count, color='#D81B60', alpha=0.5,
                                     label=f"Right Channel (N={len(arr_R)} jumps)", edgecolor='black', linewidth=0.5)

            if self.chk_kde.isChecked() and HAS_SCIPY and len(arr_R) > 1:
                x_grid = np.linspace(arr_R.min(), arr_R.max(), 300)
                kde = gaussian_kde(arr_R)
                b_width = np.mean(np.diff(bins_R)) if len(bins_R) > 1 else 1.0
                y_kde = kde(x_grid) * len(arr_R) * b_width
                ax.plot(x_grid, y_kde, color='#880e4f', linewidth=2, label="Right Channel KDE")

        selected_field = self.field_selector.currentText()
        field_suffix = f" | Field: {selected_field}" if selected_field and selected_field != "-- All Fields --" else ""

        ax.set_title(f"Jump Distance Distribution ($\Delta t = {time_lag_ms:.1f}\mathrm{{ms}}$, {lag_N} frames) — {self.sample_selector.currentText()}{field_suffix}", fontsize=11)
        ax.set_xlabel(r"Jump Distance $r$ $[\mu\mathrm{m}]$", fontsize=11)
        ax.set_ylabel("Jump Frequency / Count", fontsize=11)
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.5)

        self.jump_canvas.draw()

    def plot_spatial_map(self, summary_df):
        """Plots spatial trajectories and overlays them onto L_avg.tif when a field is selected."""
        ax = self.spatial_canvas.axes
        ax.clear()

        if self.trajectories_df.empty or summary_df.empty:
            self.spatial_canvas.draw()
            return

        selected_field = self.field_selector.currentText()
        img_data = None

        # Load L_avg.tif background if a specific field is selected
        if selected_field and selected_field != "-- All Fields --" and selected_field in self.field_dirs_map:
            field_dir = self.field_dirs_map[selected_field]
            target_img_path = None

            # Look for exact or wildcard L_avg image paths
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
                    if HAS_PIL:
                        img_data = np.array(Image.open(target_img_path))
                    else:
                        img_data = matplotlib.image.imread(target_img_path)
                except Exception as e:
                    print(f"Error loading {target_img_path}: {e}")

        if img_data is not None:
            if img_data.ndim == 3 and img_data.shape[0] < 10:  # multi-channel or stack frame slice
                img_data = img_data[0]
            ax.imshow(img_data, cmap='gray', origin='lower')

        allowed_ids = set(summary_df['global_track_id'].unique())
        df = self.trajectories_df[self.trajectories_df['global_track_id'].isin(allowed_ids)]

        channel_mode = self.channel_selector.currentText()

        for track_id, track in df.groupby('global_track_id'):
            channel = track['channel'].iloc[0]
            if channel_mode == "Left Channel (L)" and channel != "Left Channel (L)":
                continue
            if channel_mode == "Right Channel (R)" and channel != "Right Channel (R)":
                continue

            # Contrast colors depending on background presence
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

        ax.set_title(f"Spatial Map — {self.sample_selector.currentText()}{field_suffix}")
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(False if img_data is not None else True, linestyle='--', alpha=0.4)
        self.spatial_canvas.draw()

    def update_single_field_plot(self):
        """Displays chosen field image and overlays trajectories matching the selected field folder name."""
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
                if HAS_PIL:
                    img_data = np.array(Image.open(img_path))
                else:
                    img_data = matplotlib.image.imread(img_path)
            except Exception as e:
                print(f"Error loading field image {img_path}: {e}")

        if img_data is not None:
            if img_data.ndim == 3 and img_data.shape[0] < 10:  # multi-channel frame stack
                img_data = img_data[0]
            ax.imshow(img_data, cmap='gray', origin='lower')

        summary_df = self.get_filtered_summary_df()
        if not summary_df.empty:
            allowed_ids = set(summary_df[summary_df['field'] == selected_field]['global_track_id'].unique())
            df_field = self.trajectories_df[self.trajectories_df['global_track_id'].isin(allowed_ids)]

            channel_mode = self.channel_selector.currentText()

            for track_id, track in df_field.groupby('global_track_id'):
                channel = track['channel'].iloc[0]
                if channel_mode == "Left Channel (L)" and channel != "Left Channel (L)":
                    continue
                if channel_mode == "Right Channel (R)" and channel != "Right Channel (R)":
                    continue

                color = '#00FFFF' if channel == "Left Channel (L)" else '#FF00FF'
                ax.plot(track['x'], track['y'], color=color, alpha=0.8, linewidth=1.0)

        ax.set_title(f"Single Field Inspector: {selected_field}")
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

        channel_mode = self.channel_selector.currentText()
        bins_count = self.spin_bins.value()

        df_L = summary_df[summary_df['channel'] == "Left Channel (L)"]
        df_R = summary_df[summary_df['channel'] == "Right Channel (R)"]

        if channel_mode in ["Both Channels (Overlay)", "Left Channel (L)"] and not df_L.empty:
            vals_L = df_L['mean_intensity'].dropna().values
            n_L, bins_L, _ = ax.hist(vals_L, bins=bins_count, color='#008080', alpha=0.6, label="Left Channel", edgecolor='black')

            if self.chk_kde.isChecked() and HAS_SCIPY and len(vals_L) > 1:
                x_grid = np.linspace(vals_L.min(), vals_L.max(), 300)
                kde = gaussian_kde(vals_L)
                b_width = np.mean(np.diff(bins_L)) if len(bins_L) > 1 else 1.0
                y_kde = kde(x_grid) * len(vals_L) * b_width
                ax.plot(x_grid, y_kde, color='#004D4D', linewidth=2, label="Left Channel KDE")

        if channel_mode in ["Both Channels (Overlay)", "Right Channel (R)"] and not df_R.empty:
            vals_R = df_R['mean_intensity'].dropna().values
            n_R, bins_R, _ = ax.hist(vals_R, bins=bins_count, color='#D81B60', alpha=0.5, label="Right Channel", edgecolor='black')

            if self.chk_kde.isChecked() and HAS_SCIPY and len(vals_R) > 1:
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

        ax.set_title("Mean Intensity Distribution per Trajectory")
        ax.set_xlabel("Integrated Intensity (a.u.)")
        ax.set_ylabel("Frequency")
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.5)
        self.stoich_canvas.draw()


def main():
    parser = argparse.ArgumentParser(description="PySTACHIO Terminal Results Viewer")
    parser.add_argument("target_directory", nargs="?", default=".", help="Target experiment directory")
    args = parser.parse_args()

    target_path = os.path.abspath(args.target_directory)
    print(f"--> Launching PySTACHIO Results Viewer")
    print(f"--> Target Directory: {target_path}")

    app = QApplication(sys.argv)
    viewer = PyStachioResultsViewer(root_dir=target_path)
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()