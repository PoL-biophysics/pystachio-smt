# =========================================================================
# TAB 1: TRACKING PIPELINE
# =========================================================================

import sys
import os
import glob
import csv
import re
import shutil
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

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QSplitter,
                             QHBoxLayout, QPushButton, QSlider, QLineEdit,
                             QLabel, QRadioButton, QGroupBox, QFileDialog, 
                             QButtonGroup, QFormLayout, QMessageBox, QCheckBox,
                             QTabWidget, QGridLayout, QToolTip, QComboBox,
                             QScrollArea)
from PyQt6.QtCore import Qt

from pystackreg import StackReg

import images
import spots
import parameters
import postprocessing
import trajectories
import tracking

# Import studio_helpers logic if available
try:
    import studio_helpers
    from studio_helpers import save_config_file, show_popup, export_pystachio_params, to_imagej_compatible
except ImportError:
    try:
        from studio.studio_helpers import save_config_file, show_popup, export_pystachio_params, to_imagej_compatible
    except ImportError:
        save_config_file = None
        def show_popup(parent, title, text, critical=False):
            msg = QMessageBox(parent)
            msg.setWindowTitle(title)
            msg.setText(text)
            msg.setIcon(QMessageBox.Icon.Critical if critical else QMessageBox.Icon.Information)
            msg.exec()

MODULES_LOADED = True


class TiffAnalyzerTab(QWidget):
    def __init__(self, parent_suite=None):
        super().__init__()
        self.parent_suite = parent_suite
        self.raw_stack = self.source_fname = self.crop_coords = self.registration_matrix = None
        self.current_frame = 0
        self.params = parameters.Parameters()
        
        # Raw config cache for preserving unedited lines/formatting when saving
        self.raw_config_lines = None
        self.original_config_dict = {}

        self.ax_single = self.ax_left = self.ax_right = self.ax_donor = self.ax_acceptor = self.ax_fret = None
        self.im_single = self.im_left = self.im_right = self.im_donor = self.im_acceptor = self.im_fret = None
        self.cb_single = self.cb_left = self.cb_right = self.cb_donor = self.cb_acceptor = self.cb_fret = None
        self.selectors = []

        self.init_ui()
        self.set_defaults()

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        canvas_widget = QWidget()
        canvas_layout = QVBoxLayout(canvas_widget)
        canvas_layout.setContentsMargins(0, 0, 0, 0)

        self.fig = plt.figure(figsize=(10, 8), constrained_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        canvas_layout.addWidget(self.toolbar)
        canvas_layout.addWidget(self.canvas)

        slider_layout = QHBoxLayout()
        self.slider_title_label = QLabel("Frame:")
        slider_layout.addWidget(self.slider_title_label)
        
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self.on_frame_change)
        slider_layout.addWidget(self.frame_slider)
        
        self.frame_label = QLabel("0 / 0")
        slider_layout.addWidget(self.frame_label)
        canvas_layout.addLayout(slider_layout)

        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        self.tabs.setMinimumHeight(340)
        self.tab_display, self.tab_tracking, self.tab_alex = QWidget(), QWidget(), QWidget()
        self.tabs.addTab(self.tab_display, "Display & Layout")
        self.tabs.addTab(self.tab_tracking, "Tracking Params")
        self.tabs.addTab(self.tab_alex, "ALEX Config")
        
        self.setup_display_tab()
        self.setup_tracking_tab()
        self.setup_alex_tab()
        
        control_layout.addWidget(self.tabs, stretch=1)
        
        # --- EXECUTION PIPELINE GROUP ---
        execution_group = QGroupBox("Execution Pipeline")
        exec_layout = QVBoxLayout()
        
        self.btn_find_spots = QPushButton("Find spots (current frame)")
        self.btn_find_spots.clicked.connect(self.find_spots)
        self.btn_export_data = QPushButton("Export processed image data stacks")
        self.btn_export_data.clicked.connect(self.export_image_data)
        
        self.btn_run_analysis = QPushButton("Run full PySTACHIO analysis")
        self.btn_run_analysis.setStyleSheet("background-color: #2b5b84; color: white; font-weight: bold; padding: 10px;")
        self.btn_run_analysis.clicked.connect(self.run_pystachio)
        
        self.btn_apply_workflow = QPushButton("Run PySTACHIO on all Tiff files")
        self.btn_apply_workflow.setStyleSheet("background-color: #842b5b; color: white; font-weight: bold; padding: 10px;")
        self.btn_apply_workflow.clicked.connect(self.apply_workflow_to_directory)

        self.chk_subdirs = QCheckBox("Include all subdirectories")
        self.chk_subdirs.setChecked(True)
        
        self.chk_preprocessed = QCheckBox("Preprocess run (save 2 subfolders up)")
        self.chk_preprocessed.setChecked(False)
        
        exec_layout.addWidget(self.btn_find_spots)
        exec_layout.addWidget(self.btn_export_data)
        exec_layout.addWidget(self.btn_run_analysis)
        exec_layout.addWidget(self.btn_apply_workflow)
        exec_layout.addWidget(self.chk_subdirs)
        exec_layout.addWidget(self.chk_preprocessed)

        # --- EMBEDDED PIPELINE & MULTI-CHANNEL CONFIG BOX ---
        pipeline_group = QGroupBox("Pipeline & Multi-Channel Config Overrides")
        pipeline_layout = QFormLayout()

        self.master_config_box = QLineEdit()
        self.btn_browse_master_config = QPushButton("Browse...")
        self.btn_browse_master_config.clicked.connect(lambda: self.browse_config_file(self.master_config_box))
        master_row = QHBoxLayout()
        master_row.addWidget(self.master_config_box)
        master_row.addWidget(self.btn_browse_master_config)
        pipeline_layout.addRow("Master Config File:", master_row)

        self.left_config_box = QLineEdit()
        self.btn_browse_left_config = QPushButton("Browse...")
        self.btn_browse_left_config.clicked.connect(lambda: self.browse_config_file(self.left_config_box))
        left_row = QHBoxLayout()
        left_row.addWidget(self.left_config_box)
        left_row.addWidget(self.btn_browse_left_config)
        pipeline_layout.addRow("Left Channel Config:", left_row)

        self.right_config_box = QLineEdit()
        self.btn_browse_right_config = QPushButton("Browse...")
        self.btn_browse_right_config.clicked.connect(lambda: self.browse_config_file(self.right_config_box))
        right_row = QHBoxLayout()
        right_row.addWidget(self.right_config_box)
        right_row.addWidget(self.btn_browse_right_config)
        pipeline_layout.addRow("Right Channel Config:", right_row)

        self.txt_pipeline_prefix = QLineEdit()
        self.txt_pipeline_prefix.setPlaceholderText("FL prefix (e.g., field_, cell_)")
        pipeline_layout.addRow("Video Prefix (-p):", self.txt_pipeline_prefix)

        self.txt_pipeline_bf_prefix = QLineEdit()
        self.txt_pipeline_bf_prefix.setPlaceholderText("Brightfield prefix (e.g., bf_)")
        pipeline_layout.addRow("Brightfield Prefix (-b):", self.txt_pipeline_bf_prefix)

        range_layout = QHBoxLayout()
        self.range_start_box = QLineEdit("2")
        self.range_end_box = QLineEdit("max")
        self.range_step_box = QLineEdit("2")
        range_layout.addWidget(QLabel("Start:"))
        range_layout.addWidget(self.range_start_box)
        range_layout.addWidget(QLabel("End:"))
        range_layout.addWidget(self.range_end_box)
        range_layout.addWidget(QLabel("Step:"))
        range_layout.addWidget(self.range_step_box)
        pipeline_layout.addRow("Field Range (field_*_i):", range_layout)

        self.chk_preprocessed.toggled.connect(self.on_preprocessed_toggled)
        self.on_preprocessed_toggled(self.chk_preprocessed.isChecked())

        tasks_layout = QHBoxLayout()
        self.chk_pipeline_track = QCheckBox("Track")
        self.chk_pipeline_track.setChecked(True)
        self.chk_pipeline_postprocess = QCheckBox("Postprocess")
        self.chk_pipeline_postprocess.setChecked(True)
        tasks_layout.addWidget(self.chk_pipeline_track)
        tasks_layout.addWidget(self.chk_pipeline_postprocess)
        pipeline_layout.addRow("Pipeline Tasks:", tasks_layout)

        self.btn_exec_pipeline = QPushButton("Run Automated Pipeline (run_pipeline.py)")
        self.btn_exec_pipeline.setStyleSheet("background-color: #2b845b; color: white; font-weight: bold; padding: 8px;")
        self.btn_exec_pipeline.clicked.connect(self.run_pipeline_subprocess)
        pipeline_layout.addRow(self.btn_exec_pipeline)

        pipeline_group.setLayout(pipeline_layout)
        exec_layout.addWidget(pipeline_group)

        execution_group.setLayout(exec_layout)
        control_layout.addWidget(execution_group, stretch=0)

        control_scroll = QScrollArea()
        control_scroll.setWidget(control_widget)
        control_scroll.setWidgetResizable(True)
        control_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        splitter.addWidget(canvas_widget)
        splitter.addWidget(control_scroll)
        splitter.setSizes([750, 380])
        main_layout.addWidget(splitter)

    def on_preprocessed_toggled(self, checked):
        """Disables prefix/path and field range inputs when Preprocess run is enabled."""
        self.txt_pipeline_prefix.setEnabled(not checked)
        self.txt_pipeline_bf_prefix.setEnabled(not checked)
        self.range_start_box.setEnabled(not checked)
        self.range_end_box.setEnabled(not checked)
        self.range_step_box.setEnabled(not checked)

    def get_output_root_path(self, source_fname=None):
        """Resolves output directory and file path prefix for PySTACHIO tracking."""
        target_file = source_fname or self.source_fname
        if not target_file:
            return "", ".tif"

        filename = os.path.basename(target_file)
        base_name, ext = os.path.splitext(filename)
        file_dir = os.path.dirname(os.path.abspath(target_file))

        if hasattr(self, 'chk_preprocessed') and self.chk_preprocessed.isChecked():
            field_dir = os.path.dirname(file_dir)
            field_name = os.path.basename(field_dir)
            parent_dir = os.path.dirname(field_dir)
            
            if not field_name or field_name == os.path.basename(parent_dir):
                field_name = base_name
        else:
            field_name = base_name
            for suffix in ['_left', '_right', '_cropped', '_donor', '_acceptor', '_fret', '_L_channel', '_R_channel', 'L_channel', 'R_channel']:
                if field_name.endswith(suffix):
                    field_name = field_name[:-len(suffix)].rstrip('_')
                    break
            if not field_name:
                field_name = base_name
            
            parent_dir = file_dir

        output_dir = os.path.join(parent_dir, f"{field_name}_results")
        os.makedirs(output_dir, exist_ok=True)
        
        return os.path.join(output_dir, base_name), ext

    def setup_display_tab(self):
        container = QWidget()
        layout = QVBoxLayout(container)

        self.btn_load = QPushButton("Load TIF Stack File")
        self.btn_load.clicked.connect(self.load_file)
        self.btn_reset_crop = QPushButton("Reset Region of Interest (Crop)")
        self.btn_reset_crop.clicked.connect(self.reset_crop)
        layout.addWidget(self.btn_load)
        layout.addWidget(self.btn_reset_crop)
        
        mode_group = QGroupBox("Analysis Mode Selection")
        mode_layout = QVBoxLayout()
        self.mode_cluster = QButtonGroup(self)
        self.rb_mode_single = QRadioButton("Single View (Full Frame)")
        self.rb_mode_dual = QRadioButton("Dual Channel (Standard Vertical Split)")
        self.rb_mode_fret = QRadioButton("smFRET Mode (3-Channel Spatial/Temporal)")
        self.rb_mode_single.setChecked(True)
        self.mode_cluster.addButton(self.rb_mode_single)
        self.mode_cluster.addButton(self.rb_mode_dual)
        self.mode_cluster.addButton(self.rb_mode_fret)
        mode_layout.addWidget(self.rb_mode_single)
        mode_layout.addWidget(self.rb_mode_dual)
        mode_layout.addWidget(self.rb_mode_fret)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        self.rb_mode_single.toggled.connect(self.on_structural_setting_changed)
        self.rb_mode_dual.toggled.connect(self.on_structural_setting_changed)
        self.rb_mode_fret.toggled.connect(self.on_structural_setting_changed)
        
        temporal_group = QGroupBox("Temporal Deinterleaving (ALEX View)")
        temporal_layout = QVBoxLayout()
        self.cb_alex = QCheckBox("Deinterleave Temporally")
        self.cb_alex.setEnabled(False)  
        self.cb_alex.stateChanged.connect(self.on_structural_setting_changed)
        self.temporal_cluster = QButtonGroup(self)
        self.rb_temp_even = QRadioButton("Channel 1 / Even Frames First")
        self.rb_temp_odd = QRadioButton("Channel 2 / Odd Frames First")
        self.rb_temp_even.setChecked(True)
        self.rb_temp_even.setEnabled(False)
        self.rb_temp_odd.setEnabled(False)
        self.temporal_cluster.addButton(self.rb_temp_even)
        self.temporal_cluster.addButton(self.rb_temp_odd)
        temporal_layout.addWidget(self.cb_alex)
        temporal_layout.addWidget(self.rb_temp_even)
        temporal_layout.addWidget(self.rb_temp_odd)
        temporal_group.setLayout(temporal_layout)
        layout.addWidget(temporal_group)
        self.rb_temp_even.toggled.connect(self.update_display)
        self.rb_temp_odd.toggled.connect(self.update_display)
        
        spatial_group = QGroupBox("Spatial Orientation Map (smFRET)")
        spatial_layout = QVBoxLayout()
        self.spatial_cluster = QButtonGroup(self)
        self.rb_orient_norm = QRadioButton("Left: Donor | Right: FRET/Acceptor")
        self.rb_orient_inv = QRadioButton("Left: FRET/Acceptor | Right: Donor")
        self.rb_orient_norm.setChecked(True)
        self.rb_orient_norm.setEnabled(False)
        self.rb_orient_inv.setEnabled(False)
        self.spatial_cluster.addButton(self.rb_orient_norm)
        self.spatial_cluster.addButton(self.rb_orient_inv)
        spatial_layout.addWidget(self.rb_orient_norm)
        spatial_layout.addWidget(self.rb_orient_inv)
        spatial_group.setLayout(spatial_layout)
        layout.addWidget(spatial_group)
        self.rb_orient_norm.toggled.connect(self.update_display)
        self.rb_orient_inv.toggled.connect(self.update_display)
        
        calib_group = QGroupBox("Calibration & Registration")
        calib_layout = QVBoxLayout()
        self.btn_registration = QPushButton("Calculate Registration Matrix")
        self.btn_registration.clicked.connect(self.parent_suite.run_registration if self.parent_suite else lambda: None)
        self.btn_clear_registration = QPushButton("Clear Registration Matrix")
        self.btn_clear_registration.clicked.connect(self.parent_suite.clear_registration if self.parent_suite else lambda: None)
        calib_layout.addWidget(self.btn_registration)
        calib_layout.addWidget(self.btn_clear_registration)
        calib_group.setLayout(calib_layout)
        layout.addWidget(calib_group)
        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        tab_layout = QVBoxLayout(self.tab_display)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.addWidget(scroll)

    def setup_tracking_tab(self):
        container = QWidget()
        layout = QVBoxLayout(container)

        grid = QGridLayout()
        self.num_procs_box, self.frame_time_box, self.pixel_size_box, self.start_frame_box = QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit()
        self.isingle_box, self.isingle_fraction_box, self.snr_box, self.maxdisp_box = QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit()
        self.mintraj_box, self.bwthresh_box, self.sdisk_box, self.subarray_box = QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit()
        self.innermask_box = QLineEdit()
        
        boxes = [("num_procs:", self.num_procs_box), ("frame_time:", self.frame_time_box), ("pixel_size:", self.pixel_size_box), 
                 ("start_frame:", self.start_frame_box), ("I_single:", self.isingle_box), ("isingle_fraction:", self.isingle_fraction_box),
                 ("SNR cutoff:", self.snr_box), ("max_displacement:", self.maxdisp_box), ("min_traj_length:", self.mintraj_box), 
                 ("bw_threshold_tolerance:", self.bwthresh_box), ("struct_disk_radius:", self.sdisk_box), ("subarray_halfwidth:", self.subarray_box), 
                 ("inner_mask_radius:", self.innermask_box)]
        for i, (label, widget) in enumerate(boxes):
            grid.addWidget(QLabel(label), i, 0)
            grid.addWidget(widget, i, 1)
        layout.addLayout(grid)
        
        calc_isingle_group = QGroupBox("Calculate Isingle?")
        calc_isingle_layout = QHBoxLayout()
        self.calc_isingle_yes, self.calc_isingle_no = QRadioButton("Yes"), QRadioButton("No")
        self.calc_isingle_yes.toggled.connect(self.on_calc_isingle_toggled)
        self.calc_isingle_no.toggled.connect(self.on_calc_isingle_toggled)
        calc_isingle_layout.addWidget(self.calc_isingle_yes)
        calc_isingle_layout.addWidget(self.calc_isingle_no)
        calc_isingle_group.setLayout(calc_isingle_layout)
        layout.addWidget(calc_isingle_group)
        
        # --- CONFIG LOAD / SAVE BUTTONS ---
        btn_layout = QVBoxLayout()
        self.btn_load_config = QPushButton("Load Parameters from Config File")
        self.btn_load_config.setStyleSheet("font-weight: bold; background-color: #2e6da4; color: white; padding: 6px;")
        self.btn_load_config.clicked.connect(self.load_config_file)
        btn_layout.addWidget(self.btn_load_config)

        self.btn_save_config = QPushButton("Save Parameters to Config File...")
        self.btn_save_config.setStyleSheet("font-weight: bold; background-color: #007bff; color: white; padding: 6px;")
        self.btn_save_config.clicked.connect(self.save_config_file_dialog)
        btn_layout.addWidget(self.btn_save_config)

        self.btn_defaults = QPushButton("Restore Parameter Defaults")
        self.btn_defaults.clicked.connect(self.set_defaults)
        btn_layout.addWidget(self.btn_defaults)
        layout.addLayout(btn_layout)

        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        tab_layout = QVBoxLayout(self.tab_tracking)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.addWidget(scroll)

    def on_calc_isingle_toggled(self):
        """Leaves I_single blank when Calculate Isingle is Yes, or isingle_fraction blank when No."""
        if self.calc_isingle_yes.isChecked():
            self.isingle_box.clear()
            if not self.isingle_fraction_box.text().strip():
                self.isingle_fraction_box.setText('20.0')
        else:
            self.isingle_fraction_box.clear()
            if not self.isingle_box.text().strip():
                self.isingle_box.setText('120')

    def browse_config_file(self, target_line_edit):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Configuration File", filter="Config Files (*.txt *.cfg *.dat);;All Files (*)")
        if fname:
            target_line_edit.setText(fname)

    def collect_config_dict(self):
        """Collects tracking UI parameters mapped to standard PySTACHIO configuration key names."""
        calc_isingle = self.calc_isingle_yes.isChecked()
        cfg = {
            'pixel_size': self.pixel_size_box.text().strip(),
            'frame_time': self.frame_time_box.text().strip(),
            'start_frame': self.start_frame_box.text().strip(),
            'calculate_isingle': 'True' if calc_isingle else 'False',
            'I_single': '' if calc_isingle else self.isingle_box.text().strip(),
            'isingle_fraction': self.isingle_fraction_box.text().strip() if calc_isingle else '',
            'snr_filter_cutoff': self.snr_box.text().strip(),
            'max_displacement': self.maxdisp_box.text().strip(),
            'min_traj_len': self.mintraj_box.text().strip(),
            'bw_threshold_tolerance': self.bwthresh_box.text().strip(),
            'struct_disk_radius': self.sdisk_box.text().strip(),
            'subarray_halfwidth': self.subarray_box.text().strip(),
            'inner_mask_radius': self.innermask_box.text().strip(),
            'num_procs': self.num_procs_box.text().strip(),
            'ALEX': 'True' if self.backend_alex_yes.isChecked() else 'False',
            'start_channel': self.start_channel_box.text().strip(),
            'L_isingle': self.l_isingle_box.text().strip(),
            'R_isingle': self.r_isingle_box.text().strip(),
            'colocalize': 'True' if self.colocalize_yes.isChecked() else 'False',
            'colocalize_distance': self.colocalize_dist_box.text().strip(),
            'colocalize_n_frames': self.colocalize_n_box.text().strip(),
        }

        v_prefix = self.txt_pipeline_prefix.text().strip()
        if v_prefix:
            cfg['video_path'] = v_prefix

        bf_prefix = self.txt_pipeline_bf_prefix.text().strip()
        if bf_prefix:
            cfg['bf_path'] = bf_prefix

        return cfg

    def save_config_file_dialog(self):
        """Saves current tracking and ALEX UI parameters to a configuration text file."""
        default_path = self.master_config_box.text().strip() or os.path.join(os.getcwd(), "config_tracking.txt")
        fname, _ = QFileDialog.getSaveFileName(
            self,
            "Save Configuration File",
            default_path,
            "Config Files (*config*.txt *.txt *.cfg);;All Files (*)"
        )
        if not fname:
            return

        config_dict = self.collect_config_dict()

        tasks = []
        if hasattr(self, 'chk_pipeline_track') and self.chk_pipeline_track.isChecked():
            tasks.append("track")
        if hasattr(self, 'chk_pipeline_postprocess') and self.chk_pipeline_postprocess.isChecked():
            tasks.append("postprocess")
        task_directive = ",".join(tasks) if tasks else "track,postprocess"

        # Map lowercase key aliases back to canonical keys in config_dict
        alias_to_canonical = {
            'i_single': 'I_single', 'isingle': 'I_single',
            'isingle_fraction': 'isingle_fraction', 'isingle_frac': 'isingle_fraction',
            'snr_filter_cutoff': 'snr_filter_cutoff', 'snr': 'snr_filter_cutoff', 'snr_cutoff': 'snr_filter_cutoff',
            'max_displacement': 'max_displacement', 'maxdisp': 'max_displacement', 'max_disp': 'max_displacement',
            'min_traj_len': 'min_traj_len', 'min_traj_length': 'min_traj_len', 'mintraj': 'min_traj_len',
            'bw_threshold_tolerance': 'bw_threshold_tolerance', 'bw_thresh': 'bw_threshold_tolerance',
            'struct_disk_radius': 'struct_disk_radius', 'sdisk_radius': 'struct_disk_radius',
            'subarray_halfwidth': 'subarray_halfwidth', 'subarray': 'subarray_halfwidth',
            'inner_mask_radius': 'inner_mask_radius', 'innermask': 'inner_mask_radius',
            'calculate_isingle': 'calculate_isingle', 'calc_isingle': 'calculate_isingle',
            'alex': 'ALEX', 'use_alex': 'ALEX', 'is_alex': 'ALEX',
            'l_isingle': 'L_isingle', 'isingle_l': 'L_isingle',
            'r_isingle': 'R_isingle', 'isingle_r': 'R_isingle',
            'colocalize': 'colocalize', 'colocalisation': 'colocalize',
            'colocalize_distance': 'colocalize_distance', 'colocalize_dist': 'colocalize_distance',
            'colocalize_n_frames': 'colocalize_n_frames', 'colocalize_n': 'colocalize_n_frames',
            'video_path': 'video_path', 'video_prefix': 'video_path',
            'bf_path': 'bf_path', 'bf_prefix': 'bf_path',
        }
        for k in config_dict:
            alias_to_canonical.setdefault(k.lower(), k)

        try:
            out_lines = []
            written_canonical_keys = set()

            if self.raw_config_lines:
                directive_written = False
                for line in self.raw_config_lines:
                    stripped = line.strip()
                    if not directive_written and stripped and not stripped.startswith('#') and '=' not in stripped:
                        out_lines.append(f"{task_directive}\n")
                        directive_written = True
                        continue

                    if '=' in stripped and not stripped.startswith('#'):
                        raw_key, raw_rest = line.split('=', 1)
                        key_lower = raw_key.strip().lower()
                        canon_key = alias_to_canonical.get(key_lower)
                        if canon_key and canon_key in config_dict:
                            comment_part = ""
                            if '#' in raw_rest:
                                comment_part = "  #" + raw_rest.split('#', 1)[1].rstrip('\r\n')
                            out_lines.append(f"{raw_key.strip()}={config_dict[canon_key]}{comment_part}\n")
                            written_canonical_keys.add(canon_key)
                            continue

                    out_lines.append(line if line.endswith('\n') else line + '\n')

                if not directive_written:
                    out_lines.insert(0, f"{task_directive}\n")
            else:
                out_lines.append(f"{task_directive}\n")

            # Append all tracking/ALEX parameters not already updated in-place
            for k, v in config_dict.items():
                if k not in written_canonical_keys:
                    out_lines.append(f"{k}={v}\n")

            with open(fname, "w", encoding="utf-8") as f:
                f.writelines(out_lines)

            self.raw_config_lines = out_lines
            self.original_config_dict = {k.lower(): v for k, v in config_dict.items()}
            self.master_config_box.setText(fname)
            show_popup(self, "Config Saved", f"Configuration successfully saved to:\n{fname}")
        except Exception as e:
            show_popup(self, "Save Error", f"Failed to save configuration file:\n{e}", critical=True)

    def run_pipeline_subprocess(self):
        """Triggers run_pipeline.py (raw runs) or direct batch tracking (preprocessed runs)."""
        is_preprocessed = hasattr(self, 'chk_preprocessed') and self.chk_preprocessed.isChecked()
        include_subdirs = hasattr(self, 'chk_subdirs') and self.chk_subdirs.isChecked()

        master_cfg = self.master_config_box.text().strip()
        left_cfg = self.left_config_box.text().strip()
        right_cfg = self.right_config_box.text().strip()

        # Resolve working directory from config files or loaded file
        if master_cfg and os.path.exists(master_cfg):
            work_dir = os.path.dirname(os.path.abspath(master_cfg))
        elif left_cfg and os.path.exists(left_cfg):
            work_dir = os.path.dirname(os.path.abspath(left_cfg))
        elif right_cfg and os.path.exists(right_cfg):
            work_dir = os.path.dirname(os.path.abspath(right_cfg))
        elif self.source_fname and os.path.exists(self.source_fname):
            src_dir = os.path.dirname(os.path.abspath(self.source_fname))
            if is_preprocessed and os.path.basename(src_dir).lower() == "results":
                work_dir = os.path.dirname(os.path.dirname(src_dir))
            else:
                work_dir = src_dir
        else:
            work_dir = os.getcwd()

        base_cfg = master_cfg if (master_cfg and os.path.exists(master_cfg)) else (
            left_cfg if (left_cfg and os.path.exists(left_cfg)) else (
                right_cfg if (right_cfg and os.path.exists(right_cfg)) else ""
            )
        )

        # Task directives
        tasks = []
        if self.chk_pipeline_track.isChecked():
            tasks.append("track")
        if self.chk_pipeline_postprocess.isChecked():
            tasks.append("postprocess")

        if not tasks:
            show_popup(self, "Input Warning", "Please select at least one task (Track or Postprocess).")
            return

        def _apply_cfg_to_params(cfg_path):
            self.fetch_params_from_ui()
            if not cfg_path or not os.path.exists(cfg_path):
                return
            parsed = {}
            with open(cfg_path, 'r', encoding='utf-8', errors='ignore') as cf:
                for line in cf:
                    line_clean = line.strip()
                    if not line_clean or line_clean.startswith('#') or '=' not in line_clean:
                        continue
                    k, v = line_clean.split('=', 1)
                    parsed[k.strip().lower()] = v.split('#')[0].strip().strip('"').strip("'")

            if parsed.get('frame_time'): self.params.frame_time = float(parsed['frame_time'])
            if parsed.get('pixel_size'): self.params.pixel_size = float(parsed['pixel_size'])
            if parsed.get('start_frame'): self.params.start_frame = int(parsed['start_frame'])
            if 'calculate_isingle' in parsed:
                self.params.calculate_isingle = parsed['calculate_isingle'].lower() in ['true', '1', 'yes', 't', 'on']
            if parsed.get('i_single'): self.params.I_single = float(parsed['i_single'])
            if parsed.get('isingle_fraction'): self.params.isingle_fraction = float(parsed['isingle_fraction'])
            if parsed.get('snr_filter_cutoff'): self.params.snr_filter_cutoff = float(parsed['snr_filter_cutoff'])
            if parsed.get('max_displacement'): self.params.max_displacement = int(parsed['max_displacement'])
            if parsed.get('min_traj_len'): self.params.min_traj_len = int(parsed['min_traj_len'])
            if parsed.get('bw_threshold_tolerance'): self.params.bw_threshold_tolerance = float(parsed['bw_threshold_tolerance'])
            if parsed.get('struct_disk_radius'): self.params.struct_disk_radius = int(parsed['struct_disk_radius'])
            if parsed.get('subarray_halfwidth'): self.params.subarray_halfwidth = int(parsed['subarray_halfwidth'])
            if parsed.get('inner_mask_radius'): self.params.inner_mask_radius = int(parsed['inner_mask_radius'])
            if parsed.get('num_procs'): self.params.num_procs = int(parsed['num_procs'])
            if 'alex' in parsed:
                self.params.ALEX = parsed['alex'].lower() in ['true', '1', 'yes', 't', 'on']
            if parsed.get('start_channel'): self.params.start_channel = parsed['start_channel']
            if parsed.get('l_isingle'): self.params.L_isingle = float(parsed['l_isingle'])
            if parsed.get('r_isingle'): self.params.R_isingle = float(parsed['r_isingle'])
            if 'colocalize' in parsed:
                self.params.colocalize = parsed['colocalize'].lower() in ['true', '1', 'yes', 't', 'on']
            if parsed.get('colocalize_distance'): self.params.colocalize_distance = int(parsed['colocalize_distance'])
            if parsed.get('colocalize_n_frames'): self.params.colocalize_n_frames = int(parsed['colocalize_n_frames'])

        # --- PREPROCESSED RUN: IGNORE PREFIX/PATH BOXES AND FIELD RANGE (-r / --step) ---
        if is_preprocessed:
            if include_subdirs:
                l_files = glob.glob(os.path.join(work_dir, "**", "results", "*L_channel*.tif"), recursive=True) + \
                          glob.glob(os.path.join(work_dir, "**", "results", "*L_channel*.tiff"), recursive=True)
                r_files = glob.glob(os.path.join(work_dir, "**", "results", "*R_channel*.tif"), recursive=True) + \
                          glob.glob(os.path.join(work_dir, "**", "results", "*R_channel*.tiff"), recursive=True)
            else:
                l_files = glob.glob(os.path.join(work_dir, "results", "*L_channel*.tif")) + \
                          glob.glob(os.path.join(work_dir, "results", "*L_channel*.tiff"))
                r_files = glob.glob(os.path.join(work_dir, "results", "*R_channel*.tif")) + \
                          glob.glob(os.path.join(work_dir, "results", "*R_channel*.tiff"))

            preprocessed_tifs = sorted(list(set(l_files + r_files)))

            if not preprocessed_tifs:
                show_popup(self, "Preprocessed Files Missing",
                           f"No *L_channel*.tif or *R_channel*.tif files were found in 'results' directories under:\n{work_dir}",
                           critical=True)
                return

            print("\n" + "=" * 72, flush=True)
            print(f"[PySTACHIO Pipeline] Mode        : Preprocessed Run (Prefixes & Field Range ignored)", flush=True)
            print(f"[PySTACHIO Pipeline] Working Dir : {work_dir}", flush=True)
            print(f"[PySTACHIO Pipeline] Tasks       : {','.join(tasks)}", flush=True)
            print(f"[PySTACHIO Pipeline] Files Found : {len(preprocessed_tifs)}", flush=True)
            print("=" * 72, flush=True)

            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                for idx_f, tif_path in enumerate(preprocessed_tifs, start=1):
                    QApplication.processEvents()
                    bname = os.path.basename(tif_path)
                    if "L_channel" in bname and left_cfg and os.path.exists(left_cfg):
                        active_cfg = left_cfg
                    elif "R_channel" in bname and right_cfg and os.path.exists(right_cfg):
                        active_cfg = right_cfg
                    else:
                        active_cfg = base_cfg

                    _apply_cfg_to_params(active_cfg)
                    output_root, ext = self.get_output_root_path(tif_path)
                    target_tif = f"{output_root}{ext}"
                    if os.path.abspath(tif_path) != os.path.abspath(target_tif) and not os.path.exists(target_tif):
                        shutil.copy2(tif_path, target_tif)

                    self.params.name = output_root

                    print(f"[{idx_f}/{len(preprocessed_tifs)}] Processing: {tif_path}", flush=True)
                    print(f"    Output Root : {output_root}", flush=True)
                    print(f"    Config      : {active_cfg or 'UI parameters'} | "
                          f"px={self.params.pixel_size}, dt={self.params.frame_time}, "
                          f"calc_isingle={self.params.calculate_isingle}, "
                          f"I_single={self.params.I_single}, isingle_frac={self.params.isingle_fraction}, "
                          f"SNR={self.params.snr_filter_cutoff}, min_traj={self.params.min_traj_len}",
                          flush=True)

                    export_pystachio_params(self.params, f"{output_root}_PySTACHIO_params.dat")

                    if "track" in tasks:
                        print("    -> Running tracking.track()...", flush=True)
                        tracking.track(self.params)
                        plt.close('all')

                    if "postprocess" in tasks:
                        print("    -> Running postprocessing.postprocess()...", flush=True)
                        postprocessing.postprocess(self.params)
                        plt.close('all')

                print("[PySTACHIO Pipeline] Preprocessed batch pipeline finished.\n", flush=True)
                QApplication.restoreOverrideCursor()
                show_popup(self, "Pipeline Complete",
                           f"Preprocessed PySTACHIO pipeline completed successfully on {len(preprocessed_tifs)} file(s)!\n\n"
                           f"Tasks: {','.join(tasks)}\n"
                           f"Working Dir: {work_dir}")
            except Exception:
                QApplication.restoreOverrideCursor()
                traceback.print_exc()
                show_popup(self, "Pipeline Failure", f"Preprocessed tracking error:\n\n{traceback.format_exc()}", critical=True)
            return

        # --- STANDARD RUN (USES PREFIXES & FIELD RANGE -r AND --step) ---
        tab_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(tab_dir)

        candidate_paths = [
            os.path.join(parent_dir, "run_pipeline.py"),
            os.path.join(tab_dir, "run_pipeline.py"),
            os.path.join(os.getcwd(), "run_pipeline.py")
        ]

        pipeline_script = next((path for path in candidate_paths if os.path.exists(path)), None)
        if not pipeline_script:
            searched_list = "\n".join(f" - {p}" for p in candidate_paths)
            show_popup(self, "Pipeline Error",
                       f"Could not find run_pipeline.py in any expected locations:\n{searched_list}",
                       critical=True)
            return

        cmd = [sys.executable, "-u", pipeline_script]
        if base_cfg:
            cmd.extend(["-c", base_cfg])
        if left_cfg and os.path.exists(left_cfg):
            cmd.extend(["--config-left", left_cfg])
        if right_cfg and os.path.exists(right_cfg):
            cmd.extend(["--config-right", right_cfg])

        v_prefix = self.txt_pipeline_prefix.text().strip()
        if v_prefix:
            cmd.extend(["-p", v_prefix])

        bf_prefix = self.txt_pipeline_bf_prefix.text().strip()
        if bf_prefix:
            cmd.extend(["-b", bf_prefix])

        start_val = self.range_start_box.text().strip() or "2"
        end_val = self.range_end_box.text().strip() or "max"
        step_val = self.range_step_box.text().strip() or "2"
        range_arg = f"{start_val},{end_val}"
        cmd.extend(["-r", range_arg, "--step", step_val, "--task", ",".join(tasks)])

        cmd_str = subprocess.list2cmdline(cmd)
        print("\n" + "=" * 72, flush=True)
        print(f"[PySTACHIO Pipeline] Working Dir : {work_dir}", flush=True)
        print(f"[PySTACHIO Pipeline] Command     : {cmd_str}", flush=True)
        print("=" * 72, flush=True)

        captured_lines = []
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            if proc.stdout is not None:
                for line in proc.stdout:
                    print(line, end="", flush=True)
                    captured_lines.append(line)
                    QApplication.processEvents()
            return_code = proc.wait()
            QApplication.restoreOverrideCursor()

            if return_code != 0:
                err_output = "".join(captured_lines[-40:])
                show_popup(self, "Pipeline Failure",
                           f"run_pipeline.py exited with code {return_code}.\n\n"
                           f"Command:\n{cmd_str}\n\nOutput:\n{err_output}",
                           critical=True)
                return

            show_popup(self, "Pipeline Complete",
                       f"Automated PySTACHIO pipeline completed successfully!\n\n"
                       f"Command:\n{cmd_str}\n\n"
                       f"Working Dir: {work_dir}\n"
                       f"Target Range: {range_arg} (step={step_val})")
        except Exception:
            QApplication.restoreOverrideCursor()
            traceback.print_exc()
            show_popup(self, "Execution Error", f"Failed to execute pipeline subprocess:\n\n{traceback.format_exc()}", critical=True)

    def load_config_file(self, fname=None):
        """Loads tracking and ALEX parameters from a PySTACHIO config file (e.g. config.txt)."""
        if not fname:
            fname, _ = QFileDialog.getOpenFileName(self, "Load Configuration File", filter="Config Files (*.txt *.cfg *.dat);;All Files (*)")
        if not fname or not os.path.exists(fname):
            return

        parsed = {}
        try:
            with open(fname, 'r', encoding='utf-8', errors='ignore') as f:
                self.raw_config_lines = f.readlines()

            self.original_config_dict = {}
            for line in self.raw_config_lines:
                line_clean = line.strip()
                if not line_clean or line_clean.startswith('#'):
                    continue
                if '=' in line_clean:
                    key, val = line_clean.split('=', 1)
                    clean_val = val.split('#')[0].strip().strip('"').strip("'")
                    key_clean = key.strip().lower()
                    parsed[key_clean] = clean_val
                    self.original_config_dict[key_clean] = clean_val

            if not parsed:
                show_popup(self, "Config Warning", f"No valid parameters found in config file:\n{fname}")
                return

            def update_field(keys, widget):
                for k in keys:
                    if k in parsed and parsed[k] != "":
                        widget.setText(str(parsed[k]))
                        return

            # --- TRACKING PARAMS TAB ---
            update_field(['num_procs', 'n_procs', 'procs', 'num_processes'], self.num_procs_box)
            update_field(['frame_time', 'frametime', 'exposure_time', 'frame_interval', 'dt'], self.frame_time_box)
            update_field(['pixel_size', 'pixelsize', 'px_size', 'pixel_length'], self.pixel_size_box)
            update_field(['start_frame', 'startframe'], self.start_frame_box)
            update_field(['i_single', 'isingle'], self.isingle_box)
            update_field(['isingle_fraction', 'isingle_frac'], self.isingle_fraction_box)
            update_field(['snr_filter_cutoff', 'snr', 'snr_cutoff', 'snr_threshold'], self.snr_box)
            update_field(['max_displacement', 'maxdisp', 'max_disp'], self.maxdisp_box)
            update_field(['min_traj_len', 'min_traj_length', 'mintraj', 'min_len'], self.mintraj_box)
            update_field(['bw_threshold_tolerance', 'bw_thresh', 'bwthresh'], self.bwthresh_box)
            update_field(['struct_disk_radius', 'sdisk_radius', 'sdisk'], self.sdisk_box)
            update_field(['subarray_halfwidth', 'subarray'], self.subarray_box)
            update_field(['inner_mask_radius', 'innermask'], self.innermask_box)

            for k in ['calculate_isingle', 'calc_isingle']:
                if k in parsed:
                    val_bool = parsed[k].lower() in ['true', '1', 'yes', 't', 'on']
                    self.calc_isingle_yes.setChecked(val_bool)
                    self.calc_isingle_no.setChecked(not val_bool)
                    break

            # --- ALEX CONFIG TAB ---
            for k in ['alex', 'use_alex', 'is_alex', 'alex_mode']:
                if k in parsed:
                    val_bool = parsed[k].lower() in ['true', '1', 'yes', 't', 'on']
                    self.backend_alex_yes.setChecked(val_bool)
                    self.backend_alex_no.setChecked(not val_bool)
                    break

            update_field(['start_channel', 'channel'], self.start_channel_box)
            update_field(['l_isingle', 'isingle_l'], self.l_isingle_box)
            update_field(['r_isingle', 'isingle_r'], self.r_isingle_box)
            
            for k in ['colocalize', 'colocalisation', 'colocalize_mode']:
                if k in parsed:
                    val_bool = parsed[k].lower() in ['true', '1', 'yes', 't', 'on']
                    self.colocalize_yes.setChecked(val_bool)
                    self.colocalize_no.setChecked(not val_bool)
                    break

            update_field(['colocalize_distance', 'colocalize_dist'], self.colocalize_dist_box)
            update_field(['colocalize_n_frames', 'colocalize_n'], self.colocalize_n_box)

            # Prefixes
            update_field(['video_path', 'video_prefix'], self.txt_pipeline_prefix)
            update_field(['bf_path', 'bf_prefix'], self.txt_pipeline_bf_prefix)

            self.master_config_box.setText(fname)
            self.fetch_params_from_ui()
            show_popup(self, "Config Loaded", f"Successfully loaded parameters from:\n{os.path.basename(fname)}")

        except Exception as e:
            show_popup(self, "Config Read Error", f"Failed to parse config file:\n{str(e)}", critical=True)

    def setup_alex_tab(self):
        container = QWidget()
        layout = QVBoxLayout(container)

        alex_group = QGroupBox("Enable ALEX Mode in Backend Tracking?")
        alex_layout = QHBoxLayout()
        self.backend_alex_yes, self.backend_alex_no = QRadioButton("Yes"), QRadioButton("No")
        alex_layout.addWidget(self.backend_alex_yes)
        alex_layout.addWidget(self.backend_alex_no)
        alex_group.setLayout(alex_layout)
        layout.addWidget(alex_group)
        
        form_group = QGroupBox("ALEX Colocalization Parameters")
        form_layout = QFormLayout()
        self.start_channel_box, self.l_isingle_box, self.r_isingle_box = QLineEdit(), QLineEdit(), QLineEdit()
        self.colocalize_dist_box, self.colocalize_n_box = QLineEdit(), QLineEdit()
        
        self.colocalize_yes, self.colocalize_no = QRadioButton("True"), QRadioButton("False")
        colocalize_layout = QHBoxLayout()
        colocalize_layout.addWidget(self.colocalize_yes)
        colocalize_layout.addWidget(self.colocalize_no)
        colocalize_widget = QWidget()
        colocalize_widget.setLayout(colocalize_layout)
        
        form_layout.addRow("Start Channel (L/R):", self.start_channel_box)
        form_layout.addRow("L_isingle:", self.l_isingle_box)
        form_layout.addRow("R_isingle:", self.r_isingle_box)
        form_layout.addRow("Colocalize:", colocalize_widget)
        form_layout.addRow("Colocalize Distance:", self.colocalize_dist_box)
        form_layout.addRow("Colocalize n_frames:", self.colocalize_n_box)
        form_group.setLayout(form_layout)
        layout.addWidget(form_group)
        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        tab_layout = QVBoxLayout(self.tab_alex)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.addWidget(scroll)

    def set_defaults(self):
        self.num_procs_box.setText('1'); self.frame_time_box.setText('0.005')
        self.pixel_size_box.setText('0.120'); self.start_frame_box.setText('0')
        self.isingle_box.setText('120'); self.isingle_fraction_box.setText('20.0'); self.snr_box.setText('0.4')
        self.maxdisp_box.setText('5'); self.mintraj_box.setText('3')
        self.bwthresh_box.setText('1.0'); self.sdisk_box.setText('5')
        self.subarray_box.setText('8'); self.innermask_box.setText('5')
        self.calc_isingle_no.setChecked(True); self.backend_alex_no.setChecked(True)
        self.start_channel_box.setText('L'); self.l_isingle_box.setText('10000.0')
        self.r_isingle_box.setText('10000.0'); self.colocalize_no.setChecked(True)
        self.colocalize_dist_box.setText('5'); self.colocalize_n_box.setText('5')

    def fetch_params_from_ui(self):
        if not MODULES_LOADED: return
        self.params.frame_time = float(self.frame_time_box.text())
        self.params.pixel_size = float(self.pixel_size_box.text())
        self.params.start_frame = int(self.start_frame_box.text())
        self.params.calculate_isingle = self.calc_isingle_yes.isChecked()
        self.params.I_single = float(self.isingle_box.text().strip() or '120')
        self.params.isingle_fraction = float(self.isingle_fraction_box.text().strip() or '20.0')
        self.params.snr_filter_cutoff = float(self.snr_box.text())
        self.params.max_displacement = int(self.maxdisp_box.text())
        self.params.min_traj_len = int(self.mintraj_box.text())
        self.params.bw_threshold_tolerance = float(self.bwthresh_box.text())
        self.params.struct_disk_radius = int(self.sdisk_box.text())
        self.params.subarray_halfwidth = int(self.subarray_box.text())
        self.params.inner_mask_radius = int(self.innermask_box.text())
        self.params.num_procs = int(self.num_procs_box.text())
        self.params.ALEX = self.backend_alex_yes.isChecked()
        self.params.start_channel = self.start_channel_box.text()
        self.params.L_isingle = float(self.l_isingle_box.text())
        self.params.R_isingle = float(self.r_isingle_box.text())
        self.params.colocalize = self.colocalize_yes.isChecked()
        self.params.colocalize_distance = int(self.colocalize_dist_box.text())
        self.params.colocalize_n_frames = int(self.colocalize_n_box.text())

    def load_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open TIF Stack", filter="TIFF Files (*.tif *.tiff)")
        if fname:
            try:
                if self.parent_suite: self.parent_suite.reset_entire_suite()
                self.raw_stack = tf.imread(fname)
                self.source_fname = fname  
                if self.raw_stack.ndim < 3: self.raw_stack = self.raw_stack[np.newaxis, :, :]
                self.reset_crop()
            except Exception as e:
                show_popup(self, "Loading Error", f"Failed to open file:\n{str(e)}", critical=True)

    def reset_crop(self):
        if self.raw_stack is not None:
            self.crop_coords = None; self.current_frame = 0; self.update_display()

    def on_structural_setting_changed(self):
        self.crop_coords = None; self.current_frame = 0
        is_dual, is_fret = self.rb_mode_dual.isChecked(), self.rb_mode_fret.isChecked()
        
        self.cb_alex.setEnabled(is_dual)
        enable_temporal = is_fret or (is_dual and self.cb_alex.isChecked())
        
        self.rb_temp_even.setEnabled(enable_temporal)
        self.rb_temp_odd.setEnabled(enable_temporal)
        self.rb_orient_norm.setEnabled(is_fret)
        self.rb_orient_inv.setEnabled(is_fret)
        self.update_display()

    def on_frame_change(self, val):
        self.current_frame = val; self.update_display_frame_only()

    def process_data(self):
        if self.raw_stack is None: return {"mode": "single", "ch1": np.zeros((1, 1, 1))}
        n_frames, height, width = self.raw_stack.shape

        if self.rb_mode_single.isChecked():
            working = self.raw_stack
            if self.crop_coords:
                xmin, xmax, ymin, ymax = self.crop_coords
                working = working[:, ymin:ymax, xmin:xmax]
            return {"mode": "single", "ch1": working}
            
        mid_x = width // 2
        ch_left, ch_right = self.raw_stack[:, :, :mid_x], self.raw_stack[:, :, mid_x:]

        if self.crop_coords:
            xmin, xmax, ymin, ymax = self.crop_coords
            xmin_rel, xmax_rel = (xmin - mid_x, xmax - mid_x) if xmin >= mid_x else (xmin, xmax) if xmax <= mid_x else (xmin, mid_x)
            ch_left, ch_right = ch_left[:, ymin:ymax, xmin_rel:xmax_rel], ch_right[:, ymin:ymax, xmin_rel:xmax_rel]

        if self.rb_mode_dual.isChecked():
            if self.cb_alex.isChecked():
                return {"mode": "dual", "ch1": ch_left[0::2] if self.rb_temp_even.isChecked() else ch_left[1::2],
                        "ch2": ch_right[0::2] if self.rb_temp_even.isChecked() else ch_right[1::2]}
            return {"mode": "dual", "ch1": ch_left, "ch2": ch_right}

        if self.rb_mode_fret.isChecked():
            path_donor, path_acceptor_fret = (ch_left, ch_right) if self.rb_orient_norm.isChecked() else (ch_right, ch_left)
            if self.rb_temp_even.isChecked():
                donor_out, fret_out, acceptor_out = path_donor[0::2], path_acceptor_fret[0::2], path_acceptor_fret[1::2]
            else:
                donor_out, fret_out, acceptor_out = path_donor[1::2], path_acceptor_fret[1::2], path_acceptor_fret[0::2]
                
            min_len = min(len(donor_out), len(acceptor_out), len(fret_out))
            return {"mode": "fret", "ch1": donor_out[:min_len], "ch2": acceptor_out[:min_len], "ch3": fret_out[:min_len]}

    def apply_crop_callback(self, eclick, erelease):
        if None in (eclick.xdata, erelease.xdata, eclick.ydata, erelease.ydata): return
        x1, y1, x2, y2 = int(eclick.xdata), int(eclick.ydata), int(erelease.xdata), int(erelease.ydata)
        if abs(x2 - x1) < 2 or abs(y2 - y1) < 2: return
        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])
        
        if self.crop_coords:
            base_xmin, _, base_ymin, _ = self.crop_coords
            xmin += base_xmin; xmax += base_xmin; ymin += base_ymin; ymax += base_ymin

        self.crop_coords = (xmin, xmax, ymin, ymax)
        self.current_frame = 0 
        self.update_display()

    def update_display(self):
        if self.raw_stack is None: return
        for sel in self.selectors:
            sel.set_active(False)
            if hasattr(sel, 'disconnect_events'): sel.disconnect_events()
        self.selectors.clear(); self.fig.clear()
        
        self.ax_single = self.ax_left = self.ax_right = self.ax_donor = self.ax_acceptor = self.ax_fret = None
        self.im_single = self.im_left = self.im_right = self.im_donor = self.im_acceptor = self.im_fret = None
        self.cb_single = self.cb_left = self.cb_right = self.cb_donor = self.cb_acceptor = self.cb_fret = None
        
        dataset = self.process_data()
        self.slider_title_label.setText("Deinterleaved Index:" if dataset.get("deinterleaved", False) else "Frame:")

        axes_list = []
        if dataset["mode"] == "single":
            self.ax_single = self.fig.add_subplot(111)
            max_frames = len(dataset["ch1"]) - 1
            axes_list = [self.ax_single]
        elif dataset["mode"] == "dual":
            self.ax_left, self.ax_right = self.fig.add_subplot(121), self.fig.add_subplot(122)
            max_frames = min(len(dataset["ch1"]), len(dataset["ch2"])) - 1
            axes_list = [self.ax_left, self.ax_right]
        elif dataset["mode"] == "fret":
            self.ax_donor, self.ax_acceptor, self.ax_fret = self.fig.add_subplot(131), self.fig.add_subplot(132), self.fig.add_subplot(133)
            max_frames = len(dataset["ch1"]) - 1
            axes_list = [self.ax_donor, self.ax_acceptor, self.ax_fret]

        self.frame_slider.setEnabled(True)
        self.frame_slider.setMaximum(max_frames)
        if self.current_frame > max_frames:
            self.current_frame = 0
            self.frame_slider.blockSignals(True); self.frame_slider.setValue(0); self.frame_slider.blockSignals(False)

        for ax in axes_list:
            self.selectors.append(RectangleSelector(ax, self.apply_crop_callback, useblit=True, button=[1], minspanx=5, minspany=5, interactive=True))
        self.update_display_frame_only()

    def _render_channel(self, ax, im, cb, img_data, title=None):
        if im is None:
            if title: ax.set_title(title)
            im = ax.imshow(img_data, cmap='Greys_r', aspect='equal')
            cb = self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            im.set_data(img_data)
            im.set_clim(vmin=img_data.min(), vmax=img_data.max())
            cb.update_normal(im)
        return im, cb

    def update_display_frame_only(self):
        if self.raw_stack is None: return
        dataset = self.process_data()
        idx = self.current_frame

        for ax in [self.ax_single, self.ax_left, self.ax_right, self.ax_donor, self.ax_acceptor, self.ax_fret]:
            if ax:
                try: [coll.remove() for coll in list(ax.collections)]
                except: pass

        if dataset["mode"] == "single" and idx < len(dataset["ch1"]):
            self.im_single, self.cb_single = self._render_channel(self.ax_single, self.im_single, self.cb_single, dataset["ch1"][idx], "Full Frame View")
        elif dataset["mode"] == "dual":
            if idx < len(dataset["ch1"]): self.im_left, self.cb_left = self._render_channel(self.ax_left, self.im_left, self.cb_left, dataset["ch1"][idx], "Channel 1 (Left)")
            if idx < len(dataset["ch2"]): self.im_right, self.cb_right = self._render_channel(self.ax_right, self.im_right, self.cb_right, dataset["ch2"][idx], "Channel 2 (Right)")
        elif dataset["mode"] == "fret":
            if idx < len(dataset["ch1"]): self.im_donor, self.cb_donor = self._render_channel(self.ax_donor, self.im_donor, self.cb_donor, dataset["ch1"][idx], "Donor Channel")
            if idx < len(dataset["ch2"]): self.im_acceptor, self.cb_acceptor = self._render_channel(self.ax_acceptor, self.im_acceptor, self.cb_acceptor, dataset["ch2"][idx], "Acceptor Channel")
            if idx < len(dataset["ch3"]): self.im_fret, self.cb_fret = self._render_channel(self.ax_fret, self.im_fret, self.cb_fret, dataset["ch3"][idx], "FRET Channel")

        self.frame_label.setText(f"Index: {idx} / {self.frame_slider.maximum()}")
        self.canvas.draw_idle()

    def _get_pystachio_image(self, numpy_array):
        self.params.name = self.source_fname[:-4]
        img = images.ImageData()
        img.read(self.source_fname, self.params)
        img.pixel_data = numpy_array
        img.num_frames = numpy_array.shape[0]
        img.frame_size = [numpy_array.shape[2], numpy_array.shape[1]]
        img.has_mask = True
        img.mask_data = np.ones((numpy_array.shape[1], numpy_array.shape[2]))
        return img

    def find_spots(self):
        if not MODULES_LOADED or self.raw_stack is None: return
        self.fetch_params_from_ui()
        dataset, idx = self.process_data(), self.current_frame
        
        for ax in [self.ax_single, self.ax_left, self.ax_right, self.ax_donor, self.ax_acceptor, self.ax_fret]:
            if ax:
                try: [coll.remove() for coll in list(ax.collections)]
                except: pass

        if dataset["mode"] == "single":
            spots_res = tracking.track_frame(self._get_pystachio_image(dataset["ch1"])[idx], idx, self.params)
            if spots_res: [self.ax_single.scatter(p[0], p[1], marker='x', color='blue', s=40) for p in spots_res.positions]
        elif dataset["mode"] == "dual":
            s1 = tracking.track_frame(self._get_pystachio_image(dataset["ch1"])[idx], idx, self.params)
            s2 = tracking.track_frame(self._get_pystachio_image(dataset["ch2"])[idx], idx, self.params)
            if s1: [self.ax_left.scatter(p[0], p[1], marker='x', color='blue', s=40) for p in s1.positions]
            if s2: [self.ax_right.scatter(p[0], p[1], marker='x', color='blue', s=40) for p in s2.positions]
        elif dataset["mode"] == "fret":
            s1 = tracking.track_frame(self._get_pystachio_image(dataset["ch1"])[idx], idx, self.params)
            s2 = tracking.track_frame(self._get_pystachio_image(dataset["ch2"])[idx], idx, self.params)
            s3 = tracking.track_frame(self._get_pystachio_image(dataset["ch3"])[idx], idx, self.params)
            if s1: [self.ax_donor.scatter(p[0], p[1], marker='x', color='blue', s=40) for p in s1.positions]
            if s2: [self.ax_acceptor.scatter(p[0], p[1], marker='x', color='blue', s=40) for p in s2.positions]
            if s3: [self.ax_fret.scatter(p[0], p[1], marker='x', color='blue', s=40) for p in s3.positions]
            
        self.canvas.draw_idle()

    def run_pystachio(self):
        if self.raw_stack is None or self.source_fname is None: return
        self.fetch_params_from_ui()
        dataset = self.process_data()
        root_path, ext = self.get_output_root_path()
        
        export_pystachio_params(self.params, f"{root_path}_PySTACHIO_params.dat")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        
        try:
            if dataset["mode"] == "single":
                out_ch1 = to_imagej_compatible(dataset["ch1"])
                tf.imwrite(f"{root_path}_cropped{ext}", out_ch1, imagej=True)
                self.params.name = f"{root_path}_cropped"
                tracking.track(self.params)
                plt.close('all')
                postprocessing.postprocess(self.params)
                plt.close('all')
                
            elif dataset["mode"] == "dual":
                out_ch1 = to_imagej_compatible(dataset["ch1"])
                out_ch2 = to_imagej_compatible(dataset["ch2"])
                tf.imwrite(f"{root_path}_left{ext}", out_ch1, imagej=True)
                tf.imwrite(f"{root_path}_right{ext}", out_ch2, imagej=True)
                for suffix in ["_left", "_right"]:
                    self.params.name = f"{root_path}{suffix}"
                    tracking.track(self.params)
                    plt.close('all')
                    postprocessing.postprocess(self.params)
                    plt.close('all')
                    
            elif dataset["mode"] == "fret":
                out_ch1 = to_imagej_compatible(dataset["ch1"])
                out_ch2 = to_imagej_compatible(dataset["ch2"])
                out_ch3 = to_imagej_compatible(dataset["ch3"])
                tf.imwrite(f"{root_path}_donor{ext}", out_ch1, imagej=True)
                tf.imwrite(f"{root_path}_acceptor{ext}", out_ch2, imagej=True)
                tf.imwrite(f"{root_path}_fret{ext}", out_ch3, imagej=True)
                
                for suffix in ["_donor", "_acceptor", "_fret"]:
                    self.params.name = f"{root_path}{suffix}"
                    tracking.track(self.params)
                    plt.close('all')
                    postprocessing.postprocess(self.params)
                    plt.close('all')
                    
            QApplication.restoreOverrideCursor()
            show_popup(self, "Analysis Completed", "Background PySTACHIO engine pipeline completed successfully!")
        except Exception as e:
            QApplication.restoreOverrideCursor()
            traceback.print_exc()
            show_popup(self, "Pipeline Failure", f"Tracking runtime error:\n\n{traceback.format_exc()}", critical=True)

    def export_image_data(self):
        if self.raw_stack is None or self.source_fname is None: return
        dataset = self.process_data()
        root_path, ext = self.get_output_root_path()
        
        try:
            if self.registration_matrix is not None:
                np.savetxt(f"{root_path}_registration_matrix.txt", self.registration_matrix)
                
            if dataset["mode"] == "single":
                out_ch1 = to_imagej_compatible(dataset["ch1"])
                tf.imwrite(f"{root_path}_cropped{ext}", out_ch1, imagej=True)
            elif dataset["mode"] == "dual":
                out_ch1 = to_imagej_compatible(dataset["ch1"])
                out_ch2 = to_imagej_compatible(dataset["ch2"])
                tf.imwrite(f"{root_path}_left{ext}", out_ch1, imagej=True)
                tf.imwrite(f"{root_path}_right{ext}", out_ch2, imagej=True)
            elif dataset["mode"] == "fret":
                out_ch1 = to_imagej_compatible(dataset["ch1"])
                out_ch2 = to_imagej_compatible(dataset["ch2"])
                out_ch3 = to_imagej_compatible(dataset["ch3"])
                tf.imwrite(f"{root_path}_donor{ext}", out_ch1, imagej=True)
                tf.imwrite(f"{root_path}_acceptor{ext}", out_ch2, imagej=True)
                tf.imwrite(f"{root_path}_fret{ext}", out_ch3, imagej=True)
            show_popup(self, "Export Complete", "Successfully generated multi-frame TIFF image stack files on disk.")
        except Exception as e:
            show_popup(self, "Export Error", f"Failed to dump data arrays to multi-frame TIFF stack:\n{e}", critical=True)

    def apply_workflow_to_directory(self):
        """Batch processing workflow across a directory tree."""
        if self.source_fname is None:
            show_popup(self, "Error", "Please load a file first to define the working directory and settings.", critical=True)
            return
            
        directory = os.path.dirname(self.source_fname)
        self.fetch_params_from_ui()

        # Handle Preprocessed Mode (searching for L_channel and R_channel tif files in results folders)
        is_preprocessed = hasattr(self, 'chk_preprocessed') and self.chk_preprocessed.isChecked()
        
        if is_preprocessed:
            l_files = glob.glob(os.path.join(directory, "**", "*L_channel*.tif"), recursive=True) + \
                      glob.glob(os.path.join(directory, "**", "*L_channel*.tiff"), recursive=True)
            r_files = glob.glob(os.path.join(directory, "**", "*R_channel*.tif"), recursive=True) + \
                      glob.glob(os.path.join(directory, "**", "*R_channel*.tiff"), recursive=True)
            
            all_target_files = sorted(list(set(l_files + r_files)))
            if not all_target_files:
                show_popup(self, "Info", "No preprocessed L_channel or R_channel .tif files found in results folders.")
                return

            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                left_cfg = self.left_config_box.text().strip()
                right_cfg = self.right_config_box.text().strip()

                for fname in all_target_files:
                    QApplication.processEvents()
                    if "L_channel" in os.path.basename(fname) and left_cfg and os.path.exists(left_cfg):
                        self.load_config_file(left_cfg)
                    elif "R_channel" in os.path.basename(fname) and right_cfg and os.path.exists(right_cfg):
                        self.load_config_file(right_cfg)
                    else:
                        self.fetch_params_from_ui()

                    output_root, ext = self.get_output_root_path(fname)
                    target_tif = f"{output_root}{ext}"
                    if os.path.abspath(fname) != os.path.abspath(target_tif) and not os.path.exists(target_tif):
                        shutil.copy2(fname, target_tif)

                    self.params.name = output_root
                    export_pystachio_params(self.params, f"{output_root}_PySTACHIO_params.dat")
                    
                    tracking.track(self.params)
                    plt.close('all')
                    postprocessing.postprocess(self.params)
                    plt.close('all')

                QApplication.restoreOverrideCursor()
                show_popup(self, "Batch Preprocessed Tracking Completed", 
                           f"Successfully tracked {len(all_target_files)} preprocessed channel files across results folders!")
                return
            except Exception as e:
                QApplication.restoreOverrideCursor()
                traceback.print_exc()
                show_popup(self, "Pipeline Failure", f"Preprocessed tracking runtime error:\n\n{traceback.format_exc()}", critical=True)
                return

        # Standard non-preprocessed directory scan
        if hasattr(self, 'chk_subdirs') and self.chk_subdirs.isChecked():
            tif_files = set(glob.glob(os.path.join(directory, "**", "*.tif"), recursive=True) + 
                            glob.glob(os.path.join(directory, "**", "*.tiff"), recursive=True))
        else:
            tif_files = set(glob.glob(os.path.join(directory, "*.tif")) + 
                            glob.glob(os.path.join(directory, "*.tiff")))
        
        skip_suffixes = ("_cropped.tif", "_cropped.tiff", "_left.tif", "_left.tiff", 
                         "_right.tif", "_right.tiff", "_donor.tif", "_donor.tiff", 
                         "_acceptor.tif", "_acceptor.tiff", "_fret.tif", "_fret.tiff")
        
        files_to_process = [f for f in tif_files if not f.endswith(skip_suffixes)]
        
        if not files_to_process:
            show_popup(self, "Info", "No valid un-processed .tif files found in this directory.")
            return
            
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        
        orig_raw_stack = self.raw_stack
        orig_source_fname = self.source_fname
        
        try:
            for fname in files_to_process:
                QApplication.processEvents()
                try:
                    stack = tf.imread(fname)
                    if stack.ndim < 3: stack = stack[np.newaxis, :, :]
                    self.raw_stack = stack
                    self.source_fname = fname
                except Exception as e:
                    print(f"Skipping {fname} due to read error: {e}")
                    continue
                
                dataset = self.process_data()
                root_path, ext = self.get_output_root_path(fname)
                
                if self.registration_matrix is not None:
                    np.savetxt(f"{root_path}_registration_matrix.txt", self.registration_matrix)
                
                export_pystachio_params(self.params, f"{root_path}_PySTACHIO_params.dat")
                
                if dataset["mode"] == "single":
                    out_ch1 = to_imagej_compatible(dataset["ch1"])
                    tf.imwrite(f"{root_path}_cropped{ext}", out_ch1, imagej=True)
                    self.params.name = f"{root_path}_cropped"
                    tracking.track(self.params)
                    plt.close('all')
                    postprocessing.postprocess(self.params)
                    plt.close('all')
                    
                elif dataset["mode"] == "dual":
                    out_ch1 = to_imagej_compatible(dataset["ch1"])
                    out_ch2 = to_imagej_compatible(dataset["ch2"])
                    tf.imwrite(f"{root_path}_left{ext}", out_ch1, imagej=True)
                    tf.imwrite(f"{root_path}_right{ext}", out_ch2, imagej=True)
                    for suffix in ["_left", "_right"]:
                        self.params.name = f"{root_path}{suffix}"
                        tracking.track(self.params)
                        plt.close('all')
                        postprocessing.postprocess(self.params)
                        plt.close('all')
                        
                elif dataset["mode"] == "fret":
                    out_ch1 = to_imagej_compatible(dataset["ch1"])
                    out_ch2 = to_imagej_compatible(dataset["ch2"])
                    out_ch3 = to_imagej_compatible(dataset["ch3"])
                    tf.imwrite(f"{root_path}_donor{ext}", out_ch1, imagej=True)
                    tf.imwrite(f"{root_path}_acceptor{ext}", out_ch2, imagej=True)
                    tf.imwrite(f"{root_path}_fret{ext}", out_ch3, imagej=True)
                    
                    for suffix in ["_donor", "_acceptor", "_fret"]:
                        self.params.name = f"{root_path}{suffix}"
                        tracking.track(self.params)
                        plt.close('all')
                        postprocessing.postprocess(self.params)
                        plt.close('all')
                        
            QApplication.restoreOverrideCursor()
            show_popup(self, "Batch Analysis Completed", f"Successfully ran workflow on {len(files_to_process)} files in the directory tree!")
        except Exception as e:
            QApplication.restoreOverrideCursor()
            traceback.print_exc()
            show_popup(self, "Pipeline Failure", f"Batch tracking runtime error:\n\n{traceback.format_exc()}", critical=True)
        finally:
            self.raw_stack = orig_raw_stack
            self.source_fname = orig_source_fname
            if self.source_fname:
                self.params.name = self.source_fname[:-4]
            self.update_display()