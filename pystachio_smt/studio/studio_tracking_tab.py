# =========================================================================
# TAB 1: TRACKING PIPELINE
# =========================================================================

import sys
import os
import glob
import csv
import subprocess
import traceback
import json
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

import images
import spots
import parameters
import postprocessing
import trajectories
import tracking

from studio.studio_helpers import *

MODULES_LOADED = True


def tracking_registration_metadata(matrix, raw_shape, crop, mode, donor_left=True):
    """Describe registration without resampling photometry or moving tracks.

    Parent registration maps physical LEFT -> RIGHT in uncropped half-frame
    coordinates. Cropped outputs require T(-crop_origin) M T(crop_origin).
    """
    if matrix is None:
        return dict(registration_active=False, applied_to_images=False,
                    applied_to_trajectories=False, mode=mode,
                    raw_shape=[int(n) for n in raw_shape],
                    crop=list(crop) if crop is not None else None,
                    calibration_left_to_right=None, output_left_to_right=None,
                    output_donor_to_acceptor=None)
    m = np.asarray(matrix, dtype=float)
    if m.shape == (4, 3) and np.allclose(m[2:], [[0, 0, 1], [0, 0, 1]]):
        m = m[:3]  # Earlier parent duplicated the homogeneous row.
    if m.shape == (2, 3):
        m = np.vstack([m, [0., 0., 1.]])
    if (m.shape != (3, 3) or not np.isfinite(m).all()
            or not np.allclose(m[2], [0, 0, 1])
            or abs(np.linalg.det(m[:2, :2])) < 1e-12):
        raise ValueError('Registration must be an invertible finite affine matrix.')
    if mode not in ('single', 'dual', 'fret'):
        raise ValueError('Unknown tracking view mode.')
    meta = dict(
        registration_active=True,
        convention='column vectors [x, y, 1]; channel-local pixels',
        calibration_direction='physical left -> physical right',
        calibration_left_to_right=m.tolist(),
        applied_to_images=False, applied_to_trajectories=False,
        mode=mode, crop=list(crop) if crop is not None else None,
        raw_shape=[int(n) for n in raw_shape],
        output_left_to_right=None, output_donor_to_acceptor=None)
    if mode != 'single':
        if len(raw_shape) != 3 or raw_shape[2] % 2:
            raise ValueError('Registered split-channel data requires a 3-D stack with even width.')
        cx = cy = 0
        if crop is not None:
            xmin, xmax, ymin, ymax = crop
            middle = raw_shape[2] // 2
            # Match exactly the half-frame crop used by process_data().
            cx = xmin - middle if xmin >= middle else xmin
            cy = ymin
        shift = np.array([[1.,0.,cx],[0.,1.,cy],[0.,0.,1.]])
        cropped = np.linalg.inv(shift) @ m @ shift
        meta['output_left_to_right'] = cropped.tolist()
        if mode == 'fret':
            meta['donor_side'] = 'left' if donor_left else 'right'
            meta['output_donor_to_acceptor'] = (cropped if donor_left else np.linalg.inv(cropped)).tolist()
    return meta


class TiffAnalyzerTab(QWidget):
    def __init__(self, parent_suite=None):
        super().__init__()
        self.parent_suite = parent_suite
        self.raw_stack = self.source_fname = self.crop_coords = self.registration_matrix = None
        self.current_frame = 0
        self.params = parameters.Parameters()
        
        self.ax_single = self.ax_left = self.ax_right = self.ax_donor = self.ax_acceptor = self.ax_fret = None
        self.im_single = self.im_left = self.im_right = self.im_donor = self.im_acceptor = self.im_fret = None
        self.cb_single = self.cb_left = self.cb_right = self.cb_donor = self.cb_acceptor = self.cb_fret = None
        self.selectors = []

        self.init_ui()
        self.set_defaults()

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        canvas_layout = QVBoxLayout()
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
        main_layout.addLayout(canvas_layout, stretch=4)

        control_layout = QVBoxLayout()
        self.tabs = QTabWidget()
        self.tab_display, self.tab_tracking, self.tab_alex = QWidget(), QWidget(), QWidget()
        self.tabs.addTab(self.tab_display, "Display & Layout")
        self.tabs.addTab(self.tab_tracking, "Tracking Params")
        self.tabs.addTab(self.tab_alex, "ALEX Config")
        
        self.setup_display_tab()
        self.setup_tracking_tab()
        self.setup_alex_tab()
        
        control_layout.addWidget(self.tabs, stretch=1)
        
        execution_group = QGroupBox("Execution Pipeline")
        exec_layout = QVBoxLayout()
        self.btn_find_spots = QPushButton("Find spots (current frame)")
        self.btn_find_spots.clicked.connect(self.find_spots)
        self.btn_export_data = QPushButton("Export processed image data stacks")
        self.btn_export_data.clicked.connect(self.export_image_data)
        
        self.btn_run_analysis = QPushButton("Run full PySTACHIO analysis")
        self.btn_run_analysis.setStyleSheet("background-color: #2b5b84; color: white; font-weight: bold; padding: 10px;")
        self.btn_run_analysis.clicked.connect(self.run_pystachio)
        
        self.btn_apply_workflow = QPushButton("Apply workflow (all files in this directory)")
        self.btn_apply_workflow.setStyleSheet("background-color: #842b5b; color: white; font-weight: bold; padding: 10px;")
        self.btn_apply_workflow.clicked.connect(self.apply_workflow_to_directory)

        self.chk_subdirs = QCheckBox("Include all subdirectories")
        self.chk_subdirs.setChecked(True)
        
        exec_layout.addWidget(self.btn_find_spots)
        exec_layout.addWidget(self.btn_export_data)
        exec_layout.addWidget(self.btn_run_analysis)
        exec_layout.addWidget(self.btn_apply_workflow)
        exec_layout.addWidget(self.chk_subdirs)
        execution_group.setLayout(exec_layout)
        control_layout.addWidget(execution_group, stretch=0)
        main_layout.addLayout(control_layout, stretch=1)

    def setup_display_tab(self):
        layout = QVBoxLayout(self.tab_display)
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

    def setup_tracking_tab(self):
        layout = QVBoxLayout(self.tab_tracking)
        grid = QGridLayout()
        self.num_procs_box, self.frame_time_box, self.pixel_size_box, self.start_frame_box = QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit()
        self.isingle_box, self.snr_box, self.maxdisp_box, self.mintraj_box = QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit()
        self.bwthresh_box, self.sdisk_box, self.subarray_box, self.innermask_box = QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit()
        
        boxes = [("num_procs:", self.num_procs_box), ("frame_time:", self.frame_time_box), ("pixel_size:", self.pixel_size_box), 
                 ("start_frame:", self.start_frame_box), ("I_single:", self.isingle_box), ("SNR cutoff:", self.snr_box), 
                 ("max_displacement:", self.maxdisp_box), ("min_traj_length:", self.mintraj_box), ("bw_threshold_tolerance:", self.bwthresh_box), 
                 ("struct_disk_radius:", self.sdisk_box), ("subarray_halfwidth:", self.subarray_box), ("inner_mask_radius:", self.innermask_box)]
        for i, (label, widget) in enumerate(boxes):
            grid.addWidget(QLabel(label), i, 0)
            grid.addWidget(widget, i, 1)
        layout.addLayout(grid)
        
        calc_isingle_group = QGroupBox("Calculate Isingle?")
        calc_isingle_layout = QHBoxLayout()
        self.calc_isingle_yes, self.calc_isingle_no = QRadioButton("Yes"), QRadioButton("No")
        calc_isingle_layout.addWidget(self.calc_isingle_yes)
        calc_isingle_layout.addWidget(self.calc_isingle_no)
        calc_isingle_group.setLayout(calc_isingle_layout)
        layout.addWidget(calc_isingle_group)
        
        self.btn_defaults = QPushButton("Restore Parameter Defaults")
        self.btn_defaults.clicked.connect(self.set_defaults)
        layout.addWidget(self.btn_defaults)
        layout.addStretch()

    def setup_alex_tab(self):
        layout = QVBoxLayout(self.tab_alex)
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

    def set_defaults(self):
        self.num_procs_box.setText('1'); self.frame_time_box.setText('0.005')
        self.pixel_size_box.setText('0.120'); self.start_frame_box.setText('0')
        self.isingle_box.setText('120'); self.snr_box.setText('0.4')
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
        self.params.I_single = float(self.isingle_box.text())
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

    def _write_registration_metadata(self, root_path, dataset):
        metadata = tracking_registration_metadata(
            self.registration_matrix, self.raw_stack.shape, self.crop_coords,
            dataset['mode'], donor_left=self.rb_orient_norm.isChecked())
        # Retain the existing calibration filename/convention for consumers.
        # Processed-image transforms belong in a separate descriptive sidecar.
        base = Path(root_path)
        with tempfile.TemporaryDirectory(dir=base.parent, prefix='.tracking_registration_') as stage:
            matrix_path = Path(stage) / (base.name + '_registration_matrix.txt')
            metadata_path = Path(stage) / (base.name + '_tracking_registration.json')
            if metadata['registration_active']:
                np.savetxt(matrix_path, np.asarray(metadata['calibration_left_to_right']),
                           header='Physical LEFT -> RIGHT; uncropped half-frame pixels; [x,y,1] column vectors. Not applied to tracking images/coordinates.')
            metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
            if metadata['registration_active']:
                matrix_path.replace(base.parent / matrix_path.name)
            metadata_path.replace(base.parent / metadata_path.name)

    def run_pystachio(self):
        if self.raw_stack is None or self.source_fname is None: return
        self.fetch_params_from_ui()
        dataset = self.process_data()
        root_path, ext = os.path.splitext(self.source_fname)
        
        export_pystachio_params(self.params, f"{root_path}_PySTACHIO_params.dat")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        
        try:
            self._write_registration_metadata(root_path, dataset)
            if dataset["mode"] == "single":
                out_ch1 = to_imagej_compatible(dataset["ch1"])
                tf.imwrite(f"{root_path}_cropped{ext}", out_ch1, imagej=True)
                self.params.name = f"{root_path}_cropped"
                tracking.track(self.params)
                postprocessing.postprocess(self.params)
                
            elif dataset["mode"] == "dual":
                out_ch1 = to_imagej_compatible(dataset["ch1"])
                out_ch2 = to_imagej_compatible(dataset["ch2"])
                tf.imwrite(f"{root_path}_left{ext}", out_ch1, imagej=True)
                tf.imwrite(f"{root_path}_right{ext}", out_ch2, imagej=True)
                for suffix in ["_left", "_right"]:
                    self.params.name = f"{root_path}{suffix}"
                    tracking.track(self.params)
                    postprocessing.postprocess(self.params)
                    
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
                    postprocessing.postprocess(self.params)
                    
            QApplication.restoreOverrideCursor()
            show_popup(self, "Analysis Completed", "Background PySTACHIO engine pipeline completed successfully!")
        except Exception as e:
            QApplication.restoreOverrideCursor()
            show_popup(self, "Pipeline Failure", f"Tracking runtime error:\n\n{traceback.format_exc()}", critical=True)

    def export_image_data(self):
        if self.raw_stack is None or self.source_fname is None: return
        dataset = self.process_data()
        root_path, ext = os.path.splitext(self.source_fname)
        try:
            self._write_registration_metadata(root_path, dataset)
                
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
        if self.source_fname is None:
            show_popup(self, "Error", "Please load a file first to define the working directory and settings.", critical=True)
            return
            
        directory = os.path.dirname(self.source_fname)
        
        # --- NEW RECURSIVE SEARCH LOGIC ---
        if hasattr(self, 'chk_subdirs') and self.chk_subdirs.isChecked():
            tif_files = set(glob.glob(os.path.join(directory, "**", "*.tif"), recursive=True) + 
                            glob.glob(os.path.join(directory, "**", "*.tiff"), recursive=True))
        else:
            tif_files = set(glob.glob(os.path.join(directory, "*.tif")) + 
                            glob.glob(os.path.join(directory, "*.tiff")))
        # ----------------------------------
        
        skip_suffixes = ("_cropped.tif", "_cropped.tiff", "_left.tif", "_left.tiff", 
                         "_right.tif", "_right.tiff", "_donor.tif", "_donor.tiff", 
                         "_acceptor.tif", "_acceptor.tiff", "_fret.tif", "_fret.tiff")
        
        files_to_process = [f for f in tif_files if not f.endswith(skip_suffixes)]
        
        if not files_to_process:
            show_popup(self, "Info", "No valid un-processed .tif files found in this directory.")
            return
            
        self.fetch_params_from_ui()
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
                root_path, ext = os.path.splitext(self.source_fname)
                
                self._write_registration_metadata(root_path, dataset)
                
                export_pystachio_params(self.params, f"{root_path}_PySTACHIO_params.dat")
                
                if dataset["mode"] == "single":
                    out_ch1 = to_imagej_compatible(dataset["ch1"])
                    tf.imwrite(f"{root_path}_cropped{ext}", out_ch1, imagej=True)
                    self.params.name = f"{root_path}_cropped"
                    tracking.track(self.params)
                    postprocessing.postprocess(self.params)
                    
                elif dataset["mode"] == "dual":
                    out_ch1 = to_imagej_compatible(dataset["ch1"])
                    out_ch2 = to_imagej_compatible(dataset["ch2"])
                    tf.imwrite(f"{root_path}_left{ext}", out_ch1, imagej=True)
                    tf.imwrite(f"{root_path}_right{ext}", out_ch2, imagej=True)
                    for suffix in ["_left", "_right"]:
                        self.params.name = f"{root_path}{suffix}"
                        tracking.track(self.params)
                        postprocessing.postprocess(self.params)
                        
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
                        postprocessing.postprocess(self.params)
                        
            QApplication.restoreOverrideCursor()
            show_popup(self, "Batch Analysis Completed", f"Successfully ran workflow on {len(files_to_process)} files in the directory tree!")
        except Exception as e:
            QApplication.restoreOverrideCursor()
            show_popup(self, "Pipeline Failure", f"Batch tracking runtime error:\n\n{traceback.format_exc()}", critical=True)
        finally:
            self.raw_stack = orig_raw_stack
            self.source_fname = orig_source_fname
            if self.source_fname:
                self.params.name = self.source_fname[:-4]
            self.update_display()

