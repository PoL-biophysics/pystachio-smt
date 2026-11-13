# -*- coding: utf-8 -*-
import sys
import os
import glob
import re
import time
import subprocess
import traceback
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.patches as mplpatches

# Save original plt.show reference globally before any worker thread modifies it
ORIGINAL_PLT_SHOW = plt.show

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QGroupBox, QFormLayout,
    QLabel, QPushButton, QFileDialog, QLineEdit, QCheckBox, QComboBox,
    QSpinBox, QDoubleSpinBox, QTabWidget, QScrollArea, QProgressDialog,
    QMessageBox, QTextEdit, QDialog, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QCoreApplication, QTimer
from PyQt6.QtGui import QColor, QFont

# Import studio_helpers logic
try:
    import studio_helpers
    from studio_helpers import save_config_file
except ImportError:
    save_config_file = None

# Import preprocess backend logic
try:
    import preprocess
    from preprocess import AnalysisPipeline, BeadRegistrar, VideoProcessor, FileHandler, ImageProcessor
    import tifffile
    import skimage.io as io
    from skimage import transform as ski_transform
    import scipy.io as sio
    from roifile import ImagejRoi
    PREPROCESS_LOADED = True
    print("[DEBUG GUI INIT] Successfully imported `preprocess` backend module.", flush=True)
except ImportError as e:
    print(f"[DEBUG GUI INIT] WARNING: Could not import `preprocess` module ({e}). Preprocessing tab operating in standalone mode.", flush=True)
    PREPROCESS_LOADED = False


class QtLogStream:
    """Stream object that redirects stdout/stderr writes safely to a Qt signal."""
    def __init__(self, signal=None):
        self.signal = signal

    def write(self, text):
        if text:
            if hasattr(self, 'signal') and self.signal is not None:
                try:
                    self.signal.emit(str(text))
                except Exception:
                    sys.__stdout__.write(str(text))
            else:
                sys.__stdout__.write(str(text))

    def flush(self):
        pass


class NoScrollSpinBox(QSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event):
        if self.hasFocus(): super().wheelEvent(event)
        else: event.ignore()


class NoScrollDoubleSpinBox(QDoubleSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event):
        if self.hasFocus(): super().wheelEvent(event)
        else: event.ignore()


class NoScrollComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event):
        if self.hasFocus(): super().wheelEvent(event)
        else: event.ignore()


class ConfigSelectionDialog(QDialog):
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
        for filepath, cfg in config_file_map.items():
            rel_path = os.path.relpath(filepath)
            details = []
            if 'pixel_size' in cfg: details.append(f"Pixel: {cfg['pixel_size']} μm")
            if 'frame_time' in cfg: details.append(f"Frame: {cfg['frame_time']} ms")
            if 'alex' in cfg: details.append(f"ALEX: {cfg['alex']}")
            if 'num_channels' in cfg: details.append(f"Channels: {cfg['num_channels']}")
            if 'roi_file' in cfg: details.append(f"ROI: {os.path.basename(cfg['roi_file'])}")

            summary = ", ".join(details) if details else "No recognized parameters"
            self.combo.addItem(f"{rel_path}  ({summary})", userData=filepath)

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
    def __init__(self, parent=None, width=6, height=5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)


def sync_registration_files(src_dir: str, results_dir: str, config_dir: str = None):
    """Copies all registration outputs (CSVs, PNGs, matrices) to results and config directories."""
    import shutil, glob
    targets = {os.path.abspath(results_dir)}
    if config_dir and os.path.exists(config_dir):
        targets.add(os.path.abspath(config_dir))
    
    patterns = [
        "Registered_overlay_*", "*ncc_scores*", "*registration*",
        "*summed_beads*", "transformation_matrix.npy", "channel_transform.mat"
    ]
    files_to_copy = []
    for pat in patterns:
        files_to_copy.extend(glob.glob(os.path.join(src_dir, pat)))
        
    for target in targets:
        if os.path.abspath(src_dir) != target:
            os.makedirs(target, exist_ok=True)
            for fpath in files_to_copy:
                if os.path.exists(fpath):
                    try:
                        shutil.copy2(fpath, os.path.join(target, os.path.basename(fpath)))
                    except Exception as e:
                        print(f"[SYNC WARNING] Failed to copy {fpath} to {target}: {e}", flush=True)


class PreprocessParams:
    def __init__(self, **kwargs):
        self.config_dir = kwargs.get('config_dir', '')
        self.name = kwargs.get('name', 'preprocessing_results')
        self.video_path = kwargs.get('video_path', '')
        self.bf_path = kwargs.get('bf_path', '')
        self.bead_path = kwargs.get('bead_path', '')
        self.roi_file = kwargs.get('roi_file', '')
        self.roi_channel = kwargs.get('roi_channel', 'L')
        self.single_reg_target = kwargs.get('single_reg_target', 'L_avg / R_avg')
        self.save_dir = kwargs.get('save_dir', '.')
        self.tmats_path = kwargs.get('tmats_path', 'transformation_matrix.npy')
        self.param_file = kwargs.get('param_file', '')
        
        self.manual_registration = kwargs.get('manual_registration', False)
        self.manual_pairs = kwargs.get('manual_pairs', 10)
        
        self.pixel_size = kwargs.get('pixel_size', 0.120)
        self.num_channels = kwargs.get('num_channels', 1)
        self.use_channel = kwargs.get('use_channel', 'L')
        self.channel = kwargs.get('use_channel', 'L')
        self.start_channel = kwargs.get('start_channel', 'L')
        self.ALEX = kwargs.get('ALEX', False)
        self.num_frames = kwargs.get('num_frames', 0)
        self.frame_avg = kwargs.get('frame_avg', 5)
        
        self.mask_type = kwargs.get('mask_type', 'AI')
        self.model_type = kwargs.get('model_type', 'omnitorch')
        self.model = kwargs.get('model', '')
        self.model_params = kwargs.get('model_params', 0.5)
        self.threshold_param = kwargs.get('threshold_param', 0.5)
        self.dilations = kwargs.get('dilations', 1)
        self.inv_bf = kwargs.get('inv_bf', False)
        self.use_otsu = kwargs.get('use_otsu', 'False')
        self.mask_prefix = kwargs.get('mask_prefix', 'mask')
        
        self.cell_fitting = kwargs.get('cell_fitting', False)
        self.area_filter = kwargs.get('area_filter', 100)
        self.eccentricity_filter = kwargs.get('eccentricity_filter', 0.0)
        self.cell_fit_err = kwargs.get('cell_fit_err', 0.0)
        self.overwrite = kwargs.get('overwrite', True)
        self.overwrite_tmat = kwargs.get('overwrite_tmat', True)


class PreprocessWorkerThread(QThread):
    progress_signal = pyqtSignal(str)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, params, task_type="full", parent=None):
        super().__init__(parent)
        self.params = params
        self.task_type = task_type

    def run(self):
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stream = QtLogStream(self.log_signal)
        sys.stdout = stream
        sys.stderr = stream
        
        start_time = time.time()

        try:
            import matplotlib
            import matplotlib.pyplot as plt
            
            if not hasattr(plt, '_original_show'):
                plt._original_show = ORIGINAL_PLT_SHOW if 'ORIGINAL_PLT_SHOW' in globals() else plt.show

            matplotlib.use('Agg', force=True)
            plt.ioff()
            plt.show = lambda *args, **kwargs: None

            print(f"[WORKER THREAD] Starting task '{self.task_type}' cleanly...", flush=True)

            if not PREPROCESS_LOADED:
                raise ImportError("The `preprocess` module could not be loaded. Please check backend files.")

            os.makedirs(self.params.save_dir, exist_ok=True)

            if self.task_type == "full":
                self.progress_signal.emit("Running Full Preprocessing Pipeline...")
                pipeline = AnalysisPipeline(self.params)
                pipeline.run()

            elif self.task_type == "bead":
                self.progress_signal.emit("Running Auto Bead Alignment Registration...")
                registrar = BeadRegistrar()

                if self.params.bead_path and os.path.isdir(self.params.bead_path):
                    bead_dir = self.params.bead_path
                elif self.params.bead_path and os.path.exists(os.path.dirname(os.path.abspath(self.params.bead_path))):
                    bead_dir = os.path.dirname(os.path.abspath(self.params.bead_path))
                elif getattr(self.params, 'config_dir', None) and os.path.exists(self.params.config_dir):
                    bead_dir = self.params.config_dir
                else:
                    bead_dir = os.path.dirname(os.path.abspath(self.params.bead_path or "."))

                tmat = registrar.calculate_transformation(
                    self.params.bead_path, bead_dir, self.params.use_channel
                )
                if tmat is None:
                    raise RuntimeError("Bead registration failed to produce a matrix.")

                sync_registration_files(bead_dir, self.params.save_dir, getattr(self.params, 'config_dir', None))

            elif self.task_type == "video":
                self.progress_signal.emit("Extracting video channels and creating frame averages...")

                requested_frames = self.params.num_frames
                if requested_frames <= 0 and self.params.video_path and os.path.exists(self.params.video_path):
                    try:
                        with tifffile.TiffFile(self.params.video_path) as tf:
                            requested_frames = len(tf.pages)
                    except Exception as e:
                        print(f"[WORKER THREAD] Failed to count TIFF pages: {e}", flush=True)
                        requested_frames = None
                elif requested_frames <= 0:
                    requested_frames = None

                roi_file = self.params.roi_file.strip() if getattr(self.params, 'roi_file', None) else ""
                roi_channel = getattr(self.params, 'roi_channel', 'L')

                if roi_file and os.path.exists(roi_file):
                    print(f"[DEBUG WORKER] ROI file successfully detected: '{roi_file}' (Channel: '{roi_channel}')", flush=True)
                elif roi_file:
                    print(f"[DEBUG WORKER] WARNING: ROI file path specified but does NOT exist on disk: '{roi_file}'. Falling back to full frame.", flush=True)
                    roi_file = ""
                else:
                    print("[DEBUG WORKER] No ROI file specified. Proceeding with full-frame processing.", flush=True)

                L_chan, R_chan, num_frames, raw_shape = VideoProcessor.process_video(
                    self.params.video_path, 0, requested_frames, roi_channel if roi_file else "L",
                    roi_file, self.params.save_dir, int(self.params.num_channels),
                    self.params.use_channel, self.params.ALEX, getattr(self.params, 'start_channel', 'L')
                )

                if L_chan is not None and isinstance(L_chan, np.ndarray) and L_chan.ndim >= 2:
                    avg_count = min(int(self.params.frame_avg), L_chan.shape[0]) if L_chan.ndim == 3 else 1
                    L_avg = np.mean(L_chan[:avg_count], axis=0) if L_chan.ndim == 3 else L_chan.copy()
                    L_avg_tif = np.clip(L_avg, 0, 65535).astype(np.uint16)
                    tifffile.imwrite(f"{self.params.save_dir}/L_avg.tif", L_avg_tif, imagej=True)
                    tifffile.imwrite(f"{self.params.save_dir}/L_channel.tif", L_chan, imagej=True)
                    try:
                        io.imsave(f"{self.params.save_dir}/L_avg.png", (L_avg_tif / (L_avg_tif.max() or 1) * 255).astype(np.uint8))
                    except Exception: pass

                if R_chan is not None and isinstance(R_chan, np.ndarray) and R_chan.ndim >= 2:
                    avg_count = min(int(self.params.frame_avg), R_chan.shape[0]) if R_chan.ndim == 3 else 1
                    R_avg = np.mean(R_chan[:avg_count], axis=0) if R_chan.ndim == 3 else R_chan.copy()
                    R_avg_tif = np.clip(R_avg, 0, 65535).astype(np.uint16)
                    tifffile.imwrite(f"{self.params.save_dir}/R_avg.tif", R_avg_tif, imagej=True)
                    tifffile.imwrite(f"{self.params.save_dir}/R_channel.tif", R_chan, imagej=True)
                    try:
                        io.imsave(f"{self.params.save_dir}/R_avg.png", (R_avg_tif / (R_avg_tif.max() or 1) * 255).astype(np.uint8))
                    except Exception: pass

            elif self.task_type == "brightfield":
                print("[DEBUG GUI WORKER] Executing Brightfield Task...", flush=True)

                if getattr(self.params, 'overwrite_tmat', True):
                    import shutil
                    if self.params.bead_path:
                        src_dir = self.params.bead_path if os.path.isdir(self.params.bead_path) else os.path.dirname(os.path.abspath(self.params.bead_path))
                    else:
                        src_dir = os.path.dirname(os.path.abspath(self.params.video_path or self.params.bf_path or "."))

                    src_mat = os.path.join(src_dir, "transformation_matrix.npy")
                    dst_mat = os.path.join(self.params.save_dir, "transformation_matrix.npy")

                    if os.path.exists(src_mat) and os.path.abspath(src_mat) != os.path.abspath(dst_mat):
                        shutil.copy2(src_mat, dst_mat)
                        print(f"[DEBUG GUI WORKER] Successfully copied master matrix from '{src_mat}' -> '{dst_mat}'", flush=True)
                    elif not os.path.exists(src_mat):
                        print(f"[DEBUG GUI WORKER] Master matrix not found at '{src_mat}'. Using existing results matrix if present.", flush=True)

                pipeline = AnalysisPipeline(self.params)
                
                tmats = pipeline.get_transformation_matrix()
                if tmats is not None:
                    print(f"[DEBUG GUI WORKER] Loaded Affine Matrix (Shape: {tmats.shape}, dtype: {tmats.dtype}):\n{tmats}", flush=True)
                else:
                    print("[DEBUG GUI WORKER] WARNING: transformation_matrix.npy is None!", flush=True)

                raw_shape = (1, 512, 1024)
                if self.params.video_path and os.path.exists(self.params.video_path):
                    try:
                        with tifffile.TiffFile(self.params.video_path) as tf:
                            raw_shape = (1, tf.pages[0].shape[0], tf.pages[0].shape[1])
                    except Exception: pass

                from pystackreg import StackReg
                sr = StackReg(StackReg.AFFINE)
                print("[DEBUG GUI WORKER] Initialized StackReg with AFFINE mode.", flush=True)
                
                pipeline.process_brightfield(raw_shape, sr, tmats)
                print("[DEBUG GUI WORKER] Brightfield processing completed.", flush=True)
            
            elif self.task_type == "segmentation":
                self.progress_signal.emit("Executing step: Cell Segmentation & Masking...")
                pipeline = AnalysisPipeline(self.params)
                pipeline.run_segmentation()

            elif self.task_type == "fitting":
                self.progress_signal.emit("Executing step: Cell Shape Fitting & Feature Extraction...")
                pipeline = AnalysisPipeline(self.params)
                pipeline.run_fitting()

            elapsed = time.time() - start_time
            print(f"[WORKER THREAD] Task '{self.task_type}' finished in {elapsed:.2f} seconds.", flush=True)
            
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            self.finished_signal.emit(self.params.save_dir)

        except Exception as e:
            err_details = f"[WORKER THREAD EXCEPTION] Error in task '{self.task_type}': {str(e)}\n\n{traceback.format_exc()}"
            print(err_details, flush=True)
            
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            self.error_signal.emit(err_details)
        finally:
            import matplotlib
            import matplotlib.pyplot as plt
            if hasattr(plt, '_original_show'):
                plt.show = plt._original_show
            elif 'ORIGINAL_PLT_SHOW' in globals():
                plt.show = ORIGINAL_PLT_SHOW
            matplotlib.use('QtAgg', force=True)
            plt.close('all')
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class PreprocessingTab(QWidget):
    _selected_config_by_root = {}
    _scan_dialog_active = False

    def __init__(self, parent=None, root_dir="."):
        super().__init__(parent)
        self.parent = parent
        self.root_dir = os.path.abspath(root_dir) if root_dir else os.getcwd()
        self.config_dir = self.root_dir
        self.worker_thread = None
        self.progress_dialog = None
        self.df_all_objects = None
        
        # Original config file cache for preserving unchanged options
        self.raw_config_lines = None
        self.original_config_dict = {}
        
        print(f"[DEBUG PreprocessingTab] Initializing GUI tab with root_dir: '{self.root_dir}'", flush=True)
        self.init_ui()
        if self.root_dir and os.path.exists(self.root_dir):
            QTimer.singleShot(0, self.scan_and_load_config)

    def append_log(self, text):
        if text:
            self.txt_status_log.moveCursor(self.txt_status_log.textCursor().MoveOperation.End)
            self.txt_status_log.insertPlainText(text)
            self.txt_status_log.moveCursor(self.txt_status_log.textCursor().MoveOperation.End)
            try:
                QCoreApplication.processEvents()
            except Exception:
                pass

    def log(self, msg):
        formatted_msg = f"[DEBUG GUI LOG] {msg}"
        print(formatted_msg, flush=True)
        self.append_log(formatted_msg + "\n")

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setMinimumWidth(420)

        sidebar = QWidget()
        sidebar.setStyleSheet("""
            QGroupBox { margin-top: 6px; padding-top: 10px; font-weight: bold; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { min-width: 40px; padding: 2px 4px; }
            QPushButton { padding: 3px 6px; }
        """)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(4, 4, 4, 4)
        sidebar_layout.setSpacing(6)

        # 1. Config Setup Group
        config_group = QGroupBox("General Configuration Setup")
        config_layout = QFormLayout()
        self.txt_config_file = QLineEdit()
        self.txt_config_file.setPlaceholderText("Select config.txt or dataset options file")
        btn_browse_config = QPushButton("Browse...")
        btn_browse_config.clicked.connect(self.browse_config_file)
        
        lay_config = QHBoxLayout()
        lay_config.addWidget(self.txt_config_file)
        lay_config.addWidget(btn_browse_config)
        
        btn_save_config = QPushButton("Save Parameters to Config File...")
        btn_save_config.setStyleSheet("font-weight: bold; background-color: #007bff; color: white; padding: 6px;")
        btn_save_config.clicked.connect(self.save_config_file_dialog)
        
        config_layout.addRow("Dataset Config:", lay_config)
        config_layout.addRow(btn_save_config)
        config_group.setLayout(config_layout)
        sidebar_layout.addWidget(config_group)

        # 2. Bead Alignment Group
        bead_group = QGroupBox("Bead Alignment & Registration")
        bead_layout = QFormLayout()
        self.combo_bead_mode = NoScrollComboBox()
        self.combo_bead_mode.addItem("Auto (StackReg Affine)", userData="auto")
        self.combo_bead_mode.addItem("Manual Registration", userData="manual")
        self.combo_bead_mode.addItem("Load Existing Matrix File", userData="load_existing")

        self.txt_bead_path = QLineEdit()
        self.txt_bead_path.setPlaceholderText("Bead folder prefix (e.g. /path/to/beads)")
        btn_browse_beads = QPushButton("Browse...")
        btn_browse_beads.clicked.connect(self.browse_beads)
        lay_beads = QHBoxLayout()
        lay_beads.addWidget(self.txt_bead_path)
        lay_beads.addWidget(btn_browse_beads)

        self.spin_num_pairs = NoScrollSpinBox()
        self.spin_num_pairs.setRange(0, 100)
        self.spin_num_pairs.setValue(10)

        self.btn_run_beads = QPushButton("Run / Apply Bead Registration")
        self.btn_run_beads.setStyleSheet("font-weight: bold; background-color: #2e6da4; color: white; padding: 6px;")
        self.btn_run_beads.clicked.connect(lambda: self.run_task("bead"))

        bead_layout.addRow("Alignment Mode:", self.combo_bead_mode)
        bead_layout.addRow("Bead Prefix:", lay_beads)
        bead_layout.addRow("Pairs (num_pairs):", self.spin_num_pairs)
        bead_layout.addRow(self.btn_run_beads)
        bead_group.setLayout(bead_layout)
        sidebar_layout.addWidget(bead_group)

        # 3. File & Directory Setup Group
        files_group = QGroupBox("File & Directory Setup")
        files_layout = QFormLayout()

        self.txt_video_path = QLineEdit()
        self.txt_video_path.editingFinished.connect(lambda: self.update_save_dir_from_video(self.txt_video_path.text()))
        btn_browse_video = QPushButton("Browse...")
        btn_browse_video.clicked.connect(self.browse_video)
        lay_video = QHBoxLayout()
        lay_video.addWidget(self.txt_video_path)
        lay_video.addWidget(btn_browse_video)

        self.txt_bf_path = QLineEdit()
        btn_browse_bf = QPushButton("Browse...")
        btn_browse_bf.clicked.connect(self.browse_bf)
        lay_bf = QHBoxLayout()
        lay_bf.addWidget(self.txt_bf_path)
        lay_bf.addWidget(btn_browse_bf)

        self.txt_roi_file = QLineEdit()
        btn_browse_roi = QPushButton("Browse...")
        btn_browse_roi.clicked.connect(self.browse_roi)
        lay_roi = QHBoxLayout()
        lay_roi.addWidget(self.txt_roi_file)
        lay_roi.addWidget(btn_browse_roi)

        self.txt_save_dir = QLineEdit(os.path.join(self.root_dir, "results"))
        self.txt_save_dir.editingFinished.connect(lambda: self.load_and_render_results(self.txt_save_dir.text()))
        btn_browse_save = QPushButton("Browse...")
        btn_browse_save.clicked.connect(self.browse_save)
        lay_save = QHBoxLayout()
        lay_save.addWidget(self.txt_save_dir)
        lay_save.addWidget(btn_browse_save)
        
        self.chk_overwrite_tmat = QCheckBox("Overwrite Results Matrix with Config/Bead Matrix")
        self.chk_overwrite_tmat.setChecked(True)

        btn_run_bf = QPushButton("Process Brightfield Image")
        btn_run_bf.setStyleSheet("font-weight: bold; background-color: #6f42c1; color: white; padding: 6px;")
        btn_run_bf.clicked.connect(lambda: self.run_task("brightfield"))

        files_layout.addRow("Fluorescence Video:", lay_video)
        files_layout.addRow("Brightfield Image:", lay_bf)
        files_layout.addRow("ROI File (.roi):", lay_roi)
        files_layout.addRow("Output Directory:", lay_save)
        files_layout.addRow(self.chk_overwrite_tmat)
        files_layout.addRow(btn_run_bf)
        files_group.setLayout(files_layout)
        sidebar_layout.addWidget(files_group)

        # 4. Acquisition & Channel Settings
        acq_group = QGroupBox("Acquisition & Channel Settings")
        acq_layout = QFormLayout()

        self.spin_pixel_size = NoScrollDoubleSpinBox()
        self.spin_pixel_size.setDecimals(4)
        self.spin_pixel_size.setRange(0.0001, 10.0)
        self.spin_pixel_size.setValue(0.120)
        self.spin_pixel_size.setSuffix(" μm/px")

        self.combo_num_channels = NoScrollComboBox()
        self.combo_num_channels.addItems(["1", "2"])

        self.combo_use_channel = NoScrollComboBox()
        self.combo_use_channel.addItems(["L", "R"])

        self.combo_start_channel = NoScrollComboBox()
        self.combo_start_channel.addItems(["L", "R"])

        self.combo_roi_channel = NoScrollComboBox()
        self.combo_roi_channel.addItems(["L", "R"])

        self.chk_alex = QCheckBox("ALEX Excitation Mode")

        self.spin_num_frames = NoScrollSpinBox()
        self.spin_num_frames.setRange(0, 100000)
        self.spin_num_frames.setValue(0)

        self.spin_frame_avg = NoScrollSpinBox()
        self.spin_frame_avg.setRange(1, 500)
        self.spin_frame_avg.setValue(5)

        btn_run_video = QPushButton("Extract & Average Video Stack")
        btn_run_video.setStyleSheet("font-weight: bold; background-color: #fd7e14; color: white; padding: 6px;")
        btn_run_video.clicked.connect(lambda: self.run_task("video"))

        acq_layout.addRow("Pixel Size:", self.spin_pixel_size)
        acq_layout.addRow("Channels:", self.combo_num_channels)
        acq_layout.addRow("Target Channel:", self.combo_use_channel)
        acq_layout.addRow("Start Channel:", self.combo_start_channel)
        acq_layout.addRow("ROI Channel:", self.combo_roi_channel)
        acq_layout.addRow(self.chk_alex)
        acq_layout.addRow("Max Frames:", self.spin_num_frames)
        acq_layout.addRow("Frames to Average:", self.spin_frame_avg)
        acq_layout.addRow(btn_run_video)
        acq_group.setLayout(acq_layout)
        sidebar_layout.addWidget(acq_group)

        # 5. Masking & Model Configuration
        mask_group = QGroupBox("Segmentation & AI Models")
        mask_layout = QFormLayout()

        self.combo_mask_type = NoScrollComboBox()
        self.combo_mask_type.addItems(["AI", "BF", "FL_AI", "THRESHOLD", "WHOLE", "MANUAL", "file"])

        self.combo_model_type = NoScrollComboBox()
        self.combo_model_type.addItems(["omnitorch", "cellpose", "unet", "keras", "pytorch"])

        self.txt_model_path = QLineEdit()
        self.txt_model_path.setPlaceholderText("Model path or preset name (e.g. bact_phase_cp3)")
        btn_browse_model = QPushButton("Browse...")
        btn_browse_model.clicked.connect(self.browse_model)
        lay_model = QHBoxLayout()
        lay_model.addWidget(self.txt_model_path)
        lay_model.addWidget(btn_browse_model)

        self.txt_seg_param_file = QLineEdit()
        self.txt_seg_param_file.setPlaceholderText("Optional UNet params (*params.dat)")
        self.txt_seg_param_file.editingFinished.connect(lambda: self.load_seg_param_file(self.txt_seg_param_file.text()))
        btn_browse_seg_param = QPushButton("Browse...")
        btn_browse_seg_param.clicked.connect(self.browse_seg_param_file)
        lay_seg_param = QHBoxLayout()
        lay_seg_param.addWidget(self.txt_seg_param_file)
        lay_seg_param.addWidget(btn_browse_seg_param)

        self.spin_model_param = NoScrollDoubleSpinBox()
        self.spin_model_param.setRange(0.0, 1.0)
        self.spin_model_param.setSingleStep(0.05)
        self.spin_model_param.setValue(0.5)

        self.spin_dilations = NoScrollSpinBox()
        self.spin_dilations.setRange(-20, 20)
        self.spin_dilations.setValue(1)

        self.chk_inv_bf = QCheckBox("Invert Brightfield Contrast")
        self.combo_use_otsu = NoScrollComboBox()
        self.combo_use_otsu.addItems(["False", "True", "Multi"])

        self.combo_single_reg_target = NoScrollComboBox()
        self.combo_single_reg_target.addItems(["L_avg / R_avg", "BF"])

        btn_register_single = QPushButton("Register Single Image")
        btn_register_single.setStyleSheet("font-weight: bold; background-color: #17a2b8; color: white; padding: 6px;")
        btn_register_single.clicked.connect(self.register_single_image)

        btn_edit_mask = QPushButton("Edit Current Mask Interactively")
        btn_edit_mask.setStyleSheet("font-weight: bold; background-color: #6c757d; color: white; padding: 6px;")
        btn_edit_mask.clicked.connect(self.interactive_mask_edit)

        btn_run_seg = QPushButton("Run Cell Segmentation / Masking")
        btn_run_seg.setStyleSheet("font-weight: bold; background-color: #28a745; color: white; padding: 6px;")
        btn_run_seg.clicked.connect(lambda: self.run_task("segmentation"))

        mask_layout.addRow("Masking Type:", self.combo_mask_type)
        mask_layout.addRow("Model Architecture:", self.combo_model_type)
        mask_layout.addRow("Model Weights / Path:", lay_model)
        mask_layout.addRow("UNet Param File:", lay_seg_param)
        mask_layout.addRow("Threshold / Param:", self.spin_model_param)
        mask_layout.addRow("Dilations:", self.spin_dilations)
        mask_layout.addRow(self.chk_inv_bf)
        mask_layout.addRow("Otsu Mode:", self.combo_use_otsu)
        mask_layout.addRow("Single Reg Target:", self.combo_single_reg_target)
        mask_layout.addRow(btn_register_single)
        mask_layout.addRow(btn_edit_mask)
        mask_layout.addRow(btn_run_seg)
        mask_group.setLayout(mask_layout)
        sidebar_layout.addWidget(mask_group)

        # 6. Cell Shape Fitting & Filtering
        fit_group = QGroupBox("Cell Shape Fitting & Filtering")
        fit_layout = QFormLayout()

        self.chk_cell_fitting = QCheckBox("Fit Mathematical Capsule Shapes")
        self.chk_cell_fitting.setChecked(True)

        self.spin_area_filter = NoScrollSpinBox()
        self.spin_area_filter.setRange(0, 100000)
        self.spin_area_filter.setValue(100)
        self.spin_area_filter.setSuffix(" px")
        self.spin_area_filter.valueChanged.connect(lambda: self.on_cell_folder_changed(self.combo_cell_select.currentIndex()))

        self.spin_ecc_filter = NoScrollDoubleSpinBox()
        self.spin_ecc_filter.setRange(0.0, 1.0)
        self.spin_ecc_filter.setSingleStep(0.05)
        self.spin_ecc_filter.setDecimals(2)
        self.spin_ecc_filter.setValue(0.0)
        self.spin_ecc_filter.valueChanged.connect(lambda: self.on_cell_folder_changed(self.combo_cell_select.currentIndex()))

        self.spin_cell_fit_err = NoScrollDoubleSpinBox()
        self.spin_cell_fit_err.setRange(0.0, 10000.0)
        self.spin_cell_fit_err.setSingleStep(5.0)
        self.spin_cell_fit_err.setDecimals(2)
        self.spin_cell_fit_err.setSuffix(" nm")
        self.spin_cell_fit_err.setValue(0.0)
        self.spin_cell_fit_err.valueChanged.connect(lambda: self.on_cell_folder_changed(self.combo_cell_select.currentIndex()))

        self.chk_overwrite = QCheckBox("Overwrite Existing Results")
        self.chk_overwrite.setChecked(True)

        btn_run_fit = QPushButton("Fit Cell Shapes & Extract Features")
        btn_run_fit.setStyleSheet("font-weight: bold; background-color: #17a2b8; color: white; padding: 6px;")
        btn_run_fit.clicked.connect(lambda: self.run_task("fitting"))

        fit_layout.addRow(self.chk_cell_fitting)
        fit_layout.addRow("Min Area Filter:", self.spin_area_filter)
        fit_layout.addRow("Min Eccentricity:", self.spin_ecc_filter)
        fit_layout.addRow("Max Fit Error Filter:", self.spin_cell_fit_err)
        fit_layout.addRow(self.chk_overwrite)
        fit_layout.addRow(btn_run_fit)
        fit_group.setLayout(fit_layout)
        sidebar_layout.addWidget(fit_group)

        # 7. Single Field Pipeline Execution Group
        exec_group = QGroupBox("Single Field Pipeline Execution")
        exec_layout = QVBoxLayout()
        self.btn_run_pipeline = QPushButton("Run Full Preprocessing Pipeline (Current Field)")
        self.btn_run_pipeline.setStyleSheet("font-weight: bold; background-color: #008080; color: white; padding: 8px;")
        self.btn_run_pipeline.clicked.connect(lambda: self.run_task("full"))
        exec_layout.addWidget(self.btn_run_pipeline)
        exec_group.setLayout(exec_layout)
        sidebar_layout.addWidget(exec_group)

        # 8. Batch & Multi-Field Pipeline Group (run_pipeline.py Integration)
        batch_pipeline_group = QGroupBox("Batch & Multi-Field Pipeline (run_pipeline.py)")
        batch_pipeline_layout = QFormLayout()

        self.txt_pipeline_master_config = QLineEdit()
        btn_browse_p_master = QPushButton("Browse...")
        btn_browse_p_master.clicked.connect(lambda: self.browse_generic_file(self.txt_pipeline_master_config, "Select Master Config"))
        lay_p_master = QHBoxLayout()
        lay_p_master.addWidget(self.txt_pipeline_master_config)
        lay_p_master.addWidget(btn_browse_p_master)
        batch_pipeline_layout.addRow("Master Config (-c):", lay_p_master)

        self.txt_pipeline_left_config = QLineEdit()
        btn_browse_p_left = QPushButton("Browse...")
        btn_browse_p_left.clicked.connect(lambda: self.browse_generic_file(self.txt_pipeline_left_config, "Select Left Channel Config"))
        lay_p_left = QHBoxLayout()
        lay_p_left.addWidget(self.txt_pipeline_left_config)
        lay_p_left.addWidget(btn_browse_p_left)
        batch_pipeline_layout.addRow("Left Channel Config:", lay_p_left)

        self.txt_pipeline_right_config = QLineEdit()
        btn_browse_p_right = QPushButton("Browse...")
        btn_browse_p_right.clicked.connect(lambda: self.browse_generic_file(self.txt_pipeline_right_config, "Select Right Channel Config"))
        lay_p_right = QHBoxLayout()
        lay_p_right.addWidget(self.txt_pipeline_right_config)
        lay_p_right.addWidget(btn_browse_p_right)
        batch_pipeline_layout.addRow("Right Channel Config:", lay_p_right)

        self.txt_pipeline_prefix = QLineEdit()
        self.txt_pipeline_prefix.setPlaceholderText("FL prefix (e.g., field_, cell_)")
        batch_pipeline_layout.addRow("Video Prefix (-p):", self.txt_pipeline_prefix)

        self.txt_pipeline_bf_prefix = QLineEdit()
        self.txt_pipeline_bf_prefix.setPlaceholderText("Leave blank for video_prefix (i-1)")
        batch_pipeline_layout.addRow("Brightfield Prefix (-b):", self.txt_pipeline_bf_prefix)

        range_layout = QHBoxLayout()
        range_layout.setSpacing(3)
        self.spin_pipeline_start = NoScrollSpinBox()
        self.spin_pipeline_start.setRange(0, 100000)
        self.spin_pipeline_start.setValue(2)

        self.txt_pipeline_end = QLineEdit("max")
        self.txt_pipeline_end.setFixedWidth(42)

        self.spin_pipeline_step = NoScrollSpinBox()
        self.spin_pipeline_step.setRange(1, 100)
        self.spin_pipeline_step.setValue(2)

        range_layout.addWidget(QLabel("Start:"))
        range_layout.addWidget(self.spin_pipeline_start)
        range_layout.addWidget(QLabel("End:"))
        range_layout.addWidget(self.txt_pipeline_end)
        range_layout.addWidget(QLabel("Step:"))
        range_layout.addWidget(self.spin_pipeline_step)
        batch_pipeline_layout.addRow("Field Range:", range_layout)

        self.btn_run_pipeline_script = QPushButton("Run Pipeline Subprocess (run_pipeline.py)")
        self.btn_run_pipeline_script.setStyleSheet("font-weight: bold; background-color: #2b845b; color: white; padding: 8px;")
        self.btn_run_pipeline_script.clicked.connect(self.run_pipeline_subprocess)
        batch_pipeline_layout.addRow(self.btn_run_pipeline_script)

        batch_pipeline_group.setLayout(batch_pipeline_layout)
        sidebar_layout.addWidget(batch_pipeline_group)

        self.txt_status_log = QTextEdit()
        self.txt_status_log.setReadOnly(True)
        self.txt_status_log.setMaximumHeight(180)
        sidebar_layout.addWidget(self.txt_status_log)

        for lay in (config_layout, bead_layout, files_layout, acq_layout, mask_layout, fit_layout, batch_pipeline_layout):
            lay.setHorizontalSpacing(6)
            lay.setVerticalSpacing(4)
            lay.setContentsMargins(6, 6, 6, 6)
            lay.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        for sub_lay in (lay_config, lay_beads, lay_video, lay_bf, lay_roi, lay_save, lay_model, lay_seg_param, lay_p_master, lay_p_left, lay_p_right):
            sub_lay.setSpacing(4)

        for btn in (btn_browse_config, btn_browse_beads, btn_browse_video, btn_browse_bf, btn_browse_roi, btn_browse_save, btn_browse_model, btn_browse_seg_param, btn_browse_p_master, btn_browse_p_left, btn_browse_p_right):
            btn.setMaximumWidth(64)

        scroll_area.setWidget(sidebar)

        # Left Display Panel Tabs
        self.tabs = QTabWidget()
        splitter.addWidget(self.tabs)

        # Tab 1: Bead Registration
        self.tab_beads = QWidget()
        layout_bead_tab = QVBoxLayout(self.tab_beads)
        self.lbl_matrix_display = QLabel("No transformation matrix loaded.")
        self.lbl_matrix_display.setStyleSheet("font-family: monospace; font-size: 11px; background-color: #f8f9fa; border: 1px solid #ddd; padding: 6px;")
        self.lbl_ncc_display = QLabel("NCC Performance: N/A")
        self.lbl_ncc_display.setStyleSheet("font-family: monospace; font-size: 11px; background-color: #f8f9fa; border: 1px solid #ddd; padding: 6px;")
        
        lay_matrix = QHBoxLayout()
        lay_matrix.addWidget(self.lbl_matrix_display, stretch=2)
        lay_matrix.addWidget(self.lbl_ncc_display, stretch=1)
        layout_bead_tab.addLayout(lay_matrix)

        self.combo_bead_view = NoScrollComboBox()
        self.combo_bead_view.currentIndexChanged.connect(self.on_bead_view_changed)
        layout_bead_tab.addWidget(self.combo_bead_view)

        self.canvas_beads = MplCanvas(self)
        layout_bead_tab.addWidget(NavigationToolbar(self.canvas_beads, self))
        layout_bead_tab.addWidget(self.canvas_beads)
        self.tabs.addTab(self.tab_beads, "Bead Registration")

        # Tab 2: Channels (Fluorescence Previews)
        self.tab_channels = QWidget()
        layout_ch = QVBoxLayout(self.tab_channels)
        self.combo_ch_view = NoScrollComboBox()
        self.combo_ch_view.currentIndexChanged.connect(self.on_ch_view_changed)
        layout_ch.addWidget(self.combo_ch_view)
        self.canvas_channels = MplCanvas(self)
        layout_ch.addWidget(NavigationToolbar(self.canvas_channels, self))
        layout_ch.addWidget(self.canvas_channels)
        self.tabs.addTab(self.tab_channels, "Fluorescence Previews")

        # Tab 3: Brightfield
        self.tab_bf = QWidget()
        layout_bf = QVBoxLayout(self.tab_bf)
        self.combo_bf_view = NoScrollComboBox()
        self.combo_bf_view.currentIndexChanged.connect(self.on_bf_view_changed)
        layout_bf.addWidget(self.combo_bf_view)
        self.canvas_bf = MplCanvas(self)
        layout_bf.addWidget(NavigationToolbar(self.canvas_bf, self))
        layout_bf.addWidget(self.canvas_bf)
        self.tabs.addTab(self.tab_bf, "Brightfield & Alignment")

        # Tab 4: Masks & Outlines
        self.tab_masks = QWidget()
        layout_m = QVBoxLayout(self.tab_masks)
        self.combo_mask_view = NoScrollComboBox()
        self.combo_mask_view.currentIndexChanged.connect(self.on_mask_view_changed)
        layout_m.addWidget(self.combo_mask_view)
        self.canvas_masks = MplCanvas(self)
        layout_m.addWidget(NavigationToolbar(self.canvas_masks, self))
        layout_m.addWidget(self.canvas_masks)
        self.tabs.addTab(self.tab_masks, "Masks & Outlines")

        # Tab 5: Feature Summaries
        self.tab_features = QWidget()
        layout_f = QVBoxLayout(self.tab_features)
        self.canvas_features = MplCanvas(self)
        layout_f.addWidget(NavigationToolbar(self.canvas_features, self))
        layout_f.addWidget(self.canvas_features)
        self.tabs.addTab(self.tab_features, "Cell Feature Summary")

        # Tab 6: Individual Cell Inspector
        self.tab_cell_inspector = QWidget()
        layout_cell_insp = QVBoxLayout(self.tab_cell_inspector)

        self.lbl_cell_metadata = QLabel("Status: N/A  |  Cell Number: N/A  |  Mask Area: N/A  |  Mask Eccentricity: N/A  |  Fit Error: N/A")
        self.lbl_cell_metadata.setStyleSheet("font-family: monospace; font-size: 12px; font-weight: bold; background-color: #e9ecef; border: 1px solid #ced4da; padding: 8px; border-radius: 4px;")
        layout_cell_insp.addWidget(self.lbl_cell_metadata)

        lay_cell_controls = QHBoxLayout()
        
        lbl_cell_sel = QLabel("Select Cell Folder:")
        self.combo_cell_select = NoScrollComboBox()
        self.combo_cell_select.currentIndexChanged.connect(self.on_cell_folder_changed)
        
        lbl_plot_sel = QLabel("Select Plot / Image:")
        self.combo_cell_plot_select = NoScrollComboBox()
        self.combo_cell_plot_select.currentIndexChanged.connect(self.on_cell_plot_changed)

        lay_cell_controls.addWidget(lbl_cell_sel)
        lay_cell_controls.addWidget(self.combo_cell_select, stretch=1)
        lay_cell_controls.addSpacing(15)
        lay_cell_controls.addWidget(lbl_plot_sel)
        lay_cell_controls.addWidget(self.combo_cell_plot_select, stretch=2)

        layout_cell_insp.addLayout(lay_cell_controls)

        self.canvas_cell_plot = MplCanvas(self)
        layout_cell_insp.addWidget(NavigationToolbar(self.canvas_cell_plot, self))
        layout_cell_insp.addWidget(self.canvas_cell_plot)

        self.tabs.addTab(self.tab_cell_inspector, "Cell Plot Inspector")

        # Right Control Sidebar (Options)
        splitter.addWidget(scroll_area)
        splitter.setSizes([860, 440])

    def browse_generic_file(self, target_line_edit, title="Select File"):
        fname, _ = QFileDialog.getOpenFileName(self, title, self.root_dir, "Config Files (*.txt *.cfg *.dat);;All Files (*)")
        if fname:
            target_line_edit.setText(fname)

    def load_config_file(self, filepath):
        if not filepath or not os.path.exists(filepath): return
        self.log(f"Loading config file: '{filepath}'")
        self.config_dir = os.path.dirname(os.path.abspath(filepath))
        app = QApplication.instance()
        if app is not None:
            app._pystachio_selected_config = filepath
        if self.root_dir and os.path.exists(self.root_dir):
            PreprocessingTab._selected_config_by_root[os.path.realpath(self.root_dir)] = filepath
        self.txt_config_file.setText(filepath)
        if not self.txt_pipeline_master_config.text().strip():
            self.txt_pipeline_master_config.setText(filepath)
        
        def resolve_path(p):
            p = p.strip("'\"")
            return p if os.path.isabs(p) else os.path.normpath(os.path.join(self.config_dir, p))

        def to_bool(v):
            return v.strip().lower() in ['true', '1', 'yes', 'on', 't', 'y']

        try:
            with open(filepath, 'r', encoding='utf-8-sig', errors='ignore') as f:
                self.raw_config_lines = f.readlines()
            
            self.original_config_dict = {}

            for line in self.raw_config_lines:
                clean_line = line.split('#')[0].split('//')[0].strip()
                if not clean_line: continue
                parts = re.split(r'[:=\t]+', clean_line, maxsplit=1)
                if len(parts) < 2: continue
                key, val = parts[0].strip().lower().replace(' ', '_'), parts[1].strip()
                self.original_config_dict[key] = val

                if key in ['pixel_size', 'pixelsize', 'pxsize']:
                    m = re.search(r"[-+]?\d*\.\d+|\d+", val)
                    if m: self.spin_pixel_size.setValue(float(m.group()))
                elif key in ['num_channels', 'channels']:
                    m = re.search(r"\d+", val)
                    if m and self.combo_num_channels.findText(m.group()) != -1:
                        self.combo_num_channels.setCurrentText(m.group())
                elif key in ['channel', 'use_channel']:
                    ch = val.strip("'\"").upper()
                    if self.combo_use_channel.findText(ch) != -1:
                        self.combo_use_channel.setCurrentText(ch)
                elif key == 'start_channel':
                    ch = val.strip("'\"").upper()
                    if self.combo_start_channel.findText(ch) != -1:
                        self.combo_start_channel.setCurrentText(ch)
                elif key == 'roi_channel':
                    ch = val.strip("'\"").upper()
                    if self.combo_roi_channel.findText(ch) != -1:
                        self.combo_roi_channel.setCurrentText(ch)
                elif key in ['alex', 'alex_mode']:
                    self.chk_alex.setChecked(to_bool(val))
                elif key == 'num_frames':
                    m = re.search(r"\d+", val)
                    if m: self.spin_num_frames.setValue(int(m.group()))
                elif key in ['frame_avg', 'avg_frames']:
                    m = re.search(r"\d+", val)
                    if m: self.spin_frame_avg.setValue(int(m.group()))
                elif key in ['video_path', 'video_prefix']:
                    clean_val = val.strip("'\"")
                    self.txt_pipeline_prefix.setText(clean_val)
                    res_p = resolve_path(val)
                    if os.path.exists(res_p):
                        self.txt_video_path.setText(res_p)
                elif key in ['bf_path', 'bf_prefix']:
                    clean_val = val.strip("'\"")
                    self.txt_pipeline_bf_prefix.setText(clean_val)
                    res_p = resolve_path(val)
                    if os.path.exists(res_p):
                        self.txt_bf_path.setText(res_p)
                elif key in ['roi_file', 'roi']:
                    self.txt_roi_file.setText(resolve_path(val))
                elif key in ['bead_path', 'beads']:
                    self.txt_bead_path.setText(resolve_path(val))
                elif key == 'manual_registration':
                    idx = self.combo_bead_mode.findData("manual" if to_bool(val) else "auto")
                    if idx != -1:
                        self.combo_bead_mode.setCurrentIndex(idx)
                elif key in ['manual_pairs', 'num_pairs']:
                    m = re.search(r"\d+", val)
                    if m: self.spin_num_pairs.setValue(int(m.group()))
                elif key == 'mask_type':
                    mt = val.strip("'\"")
                    idx = self.combo_mask_type.findText(mt, Qt.MatchFlag.MatchFixedString)
                    if idx != -1:
                        self.combo_mask_type.setCurrentIndex(idx)
                elif key == 'model_type':
                    mt = val.strip("'\"").lower()
                    idx = self.combo_model_type.findText(mt, Qt.MatchFlag.MatchFixedString)
                    if idx != -1:
                        self.combo_model_type.setCurrentIndex(idx)
                elif key in ['model', 'model_path']:
                    clean_val = val.strip("'\"")
                    res_p = resolve_path(val)
                    self.txt_model_path.setText(res_p if os.path.exists(res_p) else clean_val)
                elif key in ['param_file', 'seg_param_file', 'unet_param']:
                    res_p = resolve_path(val)
                    self.txt_seg_param_file.setText(res_p)
                    self.load_seg_param_file(res_p)
                elif key in ['model_params', 'threshold_param', 'threshold']:
                    m = re.search(r"[-+]?\d*\.\d+|\d+", val)
                    if m: self.spin_model_param.setValue(float(m.group()))
                elif key == 'dilations':
                    m = re.search(r"[-+]?\d+", val)
                    if m: self.spin_dilations.setValue(int(m.group()))
                elif key == 'inv_bf':
                    self.chk_inv_bf.setChecked(to_bool(val))
                elif key == 'use_otsu':
                    otsu_val = val.strip("'\"")
                    idx = self.combo_use_otsu.findText(otsu_val, Qt.MatchFlag.MatchFixedString)
                    if idx != -1:
                        self.combo_use_otsu.setCurrentIndex(idx)
                elif key in ['save_dir', 'output_dir']:
                    res_p = resolve_path(val)
                    self.txt_save_dir.setText(res_p)
                elif key == 'cell_fitting':
                    self.chk_cell_fitting.setChecked(to_bool(val))
                elif key == 'overwrite':
                    self.chk_overwrite.setChecked(to_bool(val))
                elif key in ['area_filter', 'min_area']:
                    m = re.search(r"\d+", val)
                    if m: self.spin_area_filter.setValue(int(m.group()))
                elif key in ['eccentricity_filter', 'ecc_filter', 'eccentricity']:
                    m = re.search(r"[-+]?\d*\.\d+|\d+", val)
                    if m: self.spin_ecc_filter.setValue(float(m.group()))
                elif key in ['cell_fit_err', 'fit_err_filter', 'max_fit_err', 'cell_fit_error']:
                    m = re.search(r"[-+]?\d*\.\d+|\d+", val)
                    if m: self.spin_cell_fit_err.setValue(float(m.group()))
        except Exception as e:
            self.log(f"Error parsing config file '{filepath}': {e}")

        self.load_and_render_results(self.txt_save_dir.text())

    def collect_config_dict(self):
        """Collects current UI parameter values mapped to standard config keys."""
        video_prefix = self.txt_pipeline_prefix.text().strip() or self.txt_video_path.text().strip()
        bf_prefix = self.txt_pipeline_bf_prefix.text().strip() or self.txt_bf_path.text().strip()

        return {
            'pixel_size': str(self.spin_pixel_size.value()),
            'alex': 'True' if self.chk_alex.isChecked() else 'False',
            'num_frames': str(self.spin_num_frames.value()),
            'frame_avg': str(self.spin_frame_avg.value()),
            'video_path': video_prefix,
            'bf_path': bf_prefix,
            'channel': self.combo_use_channel.currentText(),
            'start_channel': self.combo_start_channel.currentText(),
            'num_channels': self.combo_num_channels.currentText(),
            'save_dir': self.txt_save_dir.text().strip(),
            'overwrite': 'True' if self.chk_overwrite.isChecked() else 'False',
            'roi_file': self.txt_roi_file.text().strip(),
            'roi_channel': self.combo_roi_channel.currentText(),
            'bead_path': self.txt_bead_path.text().strip(),
            'manual_registration': 'True' if self.combo_bead_mode.currentData() == 'manual' else 'False',
            'manual_pairs': str(self.spin_num_pairs.value()),
            'model': self.txt_model_path.text().strip(),
            'model_type': self.combo_model_type.currentText(),
            'model_params': str(self.spin_model_param.value()),
            'mask_type': self.combo_mask_type.currentText(),
            'use_otsu': self.combo_use_otsu.currentText(),
            'inv_bf': 'True' if self.chk_inv_bf.isChecked() else 'False',
            'area_filter': str(self.spin_area_filter.value()),
            'eccentricity_filter': str(self.spin_ecc_filter.value()),
            'cell_fit_err': str(self.spin_cell_fit_err.value()),
            'cell_fitting': 'True' if self.chk_cell_fitting.isChecked() else 'False',
        }

    def save_config_file_dialog(self):
        """Saves active GUI parameters to a user-chosen config file using studio_helpers.save_config_file."""
        default_path = self.txt_config_file.text().strip() or os.path.join(self.root_dir, "config.txt")
        fname, _ = QFileDialog.getSaveFileName(self, "Save Configuration File", default_path, "Config Files (*config*.txt *.txt *.cfg);;All Files (*)")
        if not fname:
            return

        config_dict = self.collect_config_dict()

        try:
            if save_config_file is not None:
                save_config_file(
                    config_dict,
                    fname,
                    raw_lines=self.raw_config_lines,
                    original_dict=self.original_config_dict,
                    task_directive="preprocess"
                )
            else:
                # Fallback inline writer
                with open(fname, "w", encoding="utf-8") as f:
                    f.write("preprocess\n")
                    for k, v in config_dict.items():
                        f.write(f"{k}={v}\n")

            self.txt_config_file.setText(fname)
            self.log(f"Successfully saved configuration parameters to: '{fname}'")
            QMessageBox.information(self, "Config Saved", f"Configuration successfully saved to:\n{fname}")
        except Exception as e:
            self.log(f"Error saving config file '{fname}': {e}\n{traceback.format_exc()}")
            QMessageBox.critical(self, "Save Error", f"Failed to save configuration file:\n{e}")

    def run_pipeline_subprocess(self):
        """Triggers run_pipeline.py as an un-buffered subprocess using UI settings."""
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
            QMessageBox.critical(
                self, "Pipeline Script Error",
                f"Could not find run_pipeline.py in any of the expected locations:\n{searched_list}"
            )
            return

        cmd = [sys.executable, pipeline_script]

        master_cfg = self.txt_pipeline_master_config.text().strip() or self.txt_config_file.text().strip()
        left_cfg = self.txt_pipeline_left_config.text().strip()
        right_cfg = self.txt_pipeline_right_config.text().strip()

        if master_cfg and os.path.exists(master_cfg):
            cmd.extend(["-c", master_cfg])
        if left_cfg and os.path.exists(left_cfg):
            cmd.extend(["--config-left", left_cfg])
        if right_cfg and os.path.exists(right_cfg):
            cmd.extend(["--config-right", right_cfg])
            
        prefix_val = self.txt_pipeline_prefix.text().strip()
        if prefix_val:
            cmd.extend(["-p", prefix_val])

        bf_prefix_val = self.txt_pipeline_bf_prefix.text().strip()
        if bf_prefix_val:
            cmd.extend(["-b", bf_prefix_val])

        start_val = self.spin_pipeline_start.value()
        end_val = self.txt_pipeline_end.text().strip() or "max"
        step_val = self.spin_pipeline_step.value()

        range_arg = f"{start_val},{end_val}"
        cmd.extend(["-r", range_arg, "--step", str(step_val)])

        # Hardcode preprocess task for Preprocessing Tab execution
        cmd.extend(["--task", "preprocess"])

        # Run inside the dataset/config directory so relative prefixes and subfolders resolve properly
        if master_cfg and os.path.exists(master_cfg):
            work_dir = os.path.dirname(os.path.abspath(master_cfg))
        elif self.config_dir and os.path.exists(self.config_dir):
            work_dir = self.config_dir
        elif self.root_dir and os.path.exists(self.root_dir):
            work_dir = self.root_dir
        else:
            work_dir = os.getcwd()

        self.log(f"Launching batch pipeline: {' '.join(cmd)}")
        self.log(f"Working Directory: '{work_dir}'")

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            res = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True, check=True)
            QApplication.restoreOverrideCursor()
            self.log(f"Pipeline stdout:\n{res.stdout}")
            self.load_and_render_results(self.txt_save_dir.text())
            QMessageBox.information(
                self, "Pipeline Complete",
                f"Automated PySTACHIO pipeline finished successfully!\n\n"
                f"Script: {os.path.basename(pipeline_script)}\n"
                f"Target Range: {range_arg} (step={step_val})"
            )
        except subprocess.CalledProcessError as e:
            QApplication.restoreOverrideCursor()
            err_msg = e.stderr if e.stderr else e.stdout
            self.log(f"Pipeline failure error output:\n{err_msg}")
            QMessageBox.critical(self, "Pipeline Failure", f"run_pipeline error (code {e.returncode}):\n\n{err_msg}")
        except Exception as e:
            QApplication.restoreOverrideCursor()
            self.log(f"Subprocess execution error: {e}")
            QMessageBox.critical(self, "Execution Error", f"Failed to execute pipeline subprocess:\n\n{str(e)}")

    def register_single_image(self):
        import matplotlib
        import matplotlib.pyplot as plt

        if hasattr(plt, '_original_show'):
            plt.show = plt._original_show
        elif 'ORIGINAL_PLT_SHOW' in globals():
            plt.show = ORIGINAL_PLT_SHOW

        matplotlib.use('QtAgg', force=True)
        plt.ion()

        save_dir = self.txt_save_dir.text()
        if not save_dir:
            save_dir = os.path.join(self.root_dir, "results")
        os.makedirs(save_dir, exist_ok=True)

        target = self.combo_single_reg_target.currentText()
        self.log(f"Starting single image drag registration for target: '{target}'")

        img_to_register = None

        if "BF" in target or "Brightfield" in target:
            bf_path = self.txt_bf_path.text()
            if not bf_path or not os.path.exists(bf_path):
                for d in [save_dir, self.config_dir, self.root_dir]:
                    for name in ["BF.tif", "BF.png", "BF_R.png"]:
                        p = os.path.join(d, name)
                        if os.path.exists(p):
                            bf_path = p
                            break
                    if bf_path and os.path.exists(bf_path): break

            if not bf_path or not os.path.exists(bf_path):
                QMessageBox.warning(self, "Image Not Found", "Brightfield image not found. Please select a valid Brightfield path.")
                return

            try:
                bf_img = np.array(Image.open(bf_path))
                if bf_img.ndim == 3:
                    bf_img = bf_img[..., 0]

                roi_file = self.txt_roi_file.text()
                if roi_file and os.path.exists(roi_file):
                    try:
                        roi = ImagejRoi.fromfile(roi_file)
                        if hasattr(roi, 'bounds'):
                            ymin, xmin, ymax, xmax = roi.bounds
                            bf_img = bf_img[ymin:ymax, xmin:xmax]
                    except Exception as e_roi:
                        self.log(f"Warning: Could not apply ROI cropping to BF image: {e_roi}")

                img_to_register = bf_img
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Failed to load brightfield image: {e}")
                return

        else:
            video_path = self.txt_video_path.text()
            if not video_path or not os.path.exists(video_path):
                QMessageBox.warning(
                    self,
                    "Video Not Found",
                    "Fluorescence video path is missing or invalid. Please select a valid TIFF video file."
                )
                return

            try:
                import tifffile
                with tifffile.TiffFile(video_path) as tf:
                    frame0 = tf.pages[0].asarray()

                if frame0.ndim == 3:
                    frame0 = frame0[0]

                img_to_register = frame0
                self.log(f"Successfully loaded Frame 0 from video stack: '{video_path}' (Shape: {frame0.shape})")
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Failed to read Frame 0 from video stack: {e}")
                return

        if img_to_register is None:
            QMessageBox.warning(self, "Error", "Could not acquire image for drag registration.")
            return

        try:
            registrar = BeadRegistrar()
            img_norm = (img_to_register.astype(np.float32) - img_to_register.min()) / (img_to_register.max() - img_to_register.min() + 1e-8)

            if hasattr(registrar, '_interactive_drag_registration'):
                tmat = registrar._interactive_drag_registration(img_norm, save_dir=save_dir)
            else:
                QMessageBox.warning(self, "Error", "BeadRegistrar._interactive_drag_registration not available.")
                return

            if tmat is not None:
                tmat_out_path = os.path.join(save_dir, "transformation_matrix.npy")
                np.save(tmat_out_path, tmat)
                self.log(f"Saved single-image transformation matrix to: '{tmat_out_path}'")
                self.load_and_render_results(save_dir)
            else:
                self.log("Single image drag registration cancelled or returned None.")
        except Exception as e:
            self.log(f"Error during single image drag registration: {e}\n{traceback.format_exc()}")
            QMessageBox.critical(self, "Registration Error", f"An error occurred during drag registration:\n{e}")

    def interactive_mask_edit(self):
        import matplotlib
        import matplotlib.pyplot as plt
        
        if hasattr(plt, '_original_show'):
            plt.show = plt._original_show
        elif 'ORIGINAL_PLT_SHOW' in globals():
            plt.show = ORIGINAL_PLT_SHOW

        matplotlib.use('QtAgg', force=True)
        plt.ion()

        params = self.build_params()
        save_dir = params.save_dir
        mask_path = os.path.join(save_dir, "all_cells_mask.tif")

        if not os.path.exists(mask_path):
            QMessageBox.warning(
                self, 
                "Mask Not Found", 
                f"No existing mask found in output directory:\n{mask_path}\n\n"
                "Please run Segmentation first or select 'MANUAL' masking mode."
            )
            return

        self.log(f"Launching interactive MaskEditor for '{mask_path}'...")
        try:
            pipeline = AnalysisPipeline(params)
            updated_mask = pipeline.edit_existing_mask()

            if updated_mask is not None:
                self.log("Interactive mask edit complete. Updating GUI tab displays...")
                self.load_and_render_results(save_dir)
        except Exception as e:
            self.log(f"Error during interactive mask edit: {e}\n{traceback.format_exc()}")
            QMessageBox.critical(self, "Edit Mask Error", f"An error occurred while editing mask:\n{e}")

    def load_seg_param_file(self, filepath):
        if not filepath or not os.path.exists(filepath): return
        self.log(f"Reading UNet parameter file: '{filepath}'")
        try:
            thresh_found = None
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.split('#')[0].split('//')[0].strip()
                    if not line: continue
                    
                    parts = re.split(r'[:=\t]+', line, maxsplit=1)
                    if len(parts) >= 2:
                        key, val = parts[0].strip().lower().replace(' ', '_'), parts[1].strip()
                        if key in ['threshold', 'threshold_param', 'model_params', 'param', 'thresh']:
                            m = re.search(r"[-+]?\d*\.\d+|\d+", val)
                            if m:
                                thresh_found = float(m.group())
                                break
                    else:
                        m = re.search(r"^[-+]?\d*\.\d+|\d+$", line)
                        if m:
                            val_float = float(m.group())
                            if 0.0 <= val_float <= 1.0:
                                thresh_found = val_float
                                break

            if thresh_found is not None:
                self.spin_model_param.setValue(thresh_found)
                self.log(f"Updated threshold value to {thresh_found} from '{filepath}'")
            else:
                self.log(f"Could not find threshold float value in '{filepath}'")
        except Exception as e:
            self.log(f"Failed to parse UNet parameter file '{filepath}': {e}")

    def scan_and_load_config(self):
        if getattr(self, "_config_scan_done", False):
            return
        self._config_scan_done = True

        # 1. Skip if a config file was already loaded into this tab
        existing_txt = self.txt_config_file.text().strip()
        if existing_txt and os.path.exists(existing_txt):
            return

        app = QApplication.instance()
        state_holder = app if app is not None else PreprocessingTab

        # 2. Defer if another modal dialog is currently open
        if getattr(state_holder, "_pystachio_scan_dialog_active", False) or QApplication.activeModalWidget() is not None:
            self._config_scan_done = False
            QTimer.singleShot(100, self.scan_and_load_config)
            return

        # 3. Reuse any config already chosen in this session (across all instances/root_dirs)
        cached_cfg = getattr(state_holder, "_pystachio_selected_config", None)
        if not cached_cfg and self.root_dir and os.path.exists(self.root_dir):
            cached_cfg = PreprocessingTab._selected_config_by_root.get(os.path.realpath(self.root_dir))
        if not cached_cfg:
            for owner in (self.parent, self.window()):
                if owner is not None:
                    for attr in ("config_file", "config_path", "selected_config", "master_config"):
                        val = getattr(owner, attr, None)
                        if isinstance(val, str) and val and os.path.exists(val):
                            cached_cfg = val
                            break
                if cached_cfg:
                    break

        if cached_cfg and os.path.exists(cached_cfg):
            self.log(f"Using previously selected config: '{cached_cfg}'")
            self.load_config_file(cached_cfg)
            return

        if not self.root_dir or not os.path.exists(self.root_dir):
            return

        real_root = os.path.realpath(self.root_dir)
        self.log(f"Scanning for config files in: '{self.root_dir}'")

        seen_paths = set()
        config_matches = []
        for p in glob.glob(os.path.join(self.root_dir, "**", "*config*.txt"), recursive=True):
            if os.path.basename(p).startswith(".temp_"):
                continue
            norm_p = os.path.normcase(os.path.realpath(p))
            if norm_p not in seen_paths:
                seen_paths.add(norm_p)
                config_matches.append(p)

        if not config_matches:
            return

        # If exactly one config is found, load it directly without prompting
        if len(config_matches) == 1:
            self.log(f"Found single config file: '{config_matches[0]}'")
            state_holder._pystachio_selected_config = config_matches[0]
            PreprocessingTab._selected_config_by_root[real_root] = config_matches[0]
            self.load_config_file(config_matches[0])
            return

        # If multiple configs exist, prompt once for the entire session
        self.log(f"Found {len(config_matches)} config files. Prompting user selection...")
        config_file_map = {}
        for filepath in config_matches:
            cfg = {}
            try:
                with open(filepath, 'r', encoding='utf-8-sig', errors='ignore') as f:
                    for line in f:
                        clean_line = line.split('#')[0].split('//')[0].strip()
                        if not clean_line or '=' not in clean_line:
                            continue
                        k, v = clean_line.split('=', 1)
                        cfg[k.strip().lower().replace(' ', '_')] = v.strip()
                config_file_map[filepath] = cfg
            except Exception as e:
                self.log(f"Error reading {filepath}: {e}")

        state_holder._pystachio_scan_dialog_active = True
        try:
            dlg_parent = self.window() if (self.window() and self.window().isVisible()) else None
            dialog = ConfigSelectionDialog(config_file_map, dlg_parent)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                selected_path = dialog.get_selected_filepath() or config_matches[0]
            else:
                selected_path = config_matches[0]
            state_holder._pystachio_selected_config = selected_path
            PreprocessingTab._selected_config_by_root[real_root] = selected_path
            self.log(f"User selected config: '{selected_path}'")
            self.load_config_file(selected_path)
        finally:
            state_holder._pystachio_scan_dialog_active = False

    def browse_config_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Config File", self.root_dir, "Config Files (*config*.txt *.cfg *.txt)")
        if fname:
            self.txt_config_file.setText(fname)
            self.load_config_file(fname)

    def browse_seg_param_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select UNet Parameter File", self.root_dir, "UNet Params (*params.dat *.dat *.txt)")
        if fname:
            self.txt_seg_param_file.setText(fname)
            self.load_seg_param_file(fname)

    def update_save_dir_from_video(self, video_path):
        if video_path and os.path.exists(video_path):
            save_dir = os.path.join(os.path.dirname(os.path.abspath(video_path)), "results")
            self.txt_save_dir.setText(save_dir)
            self.load_and_render_results(save_dir)

    def browse_video(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Video", self.root_dir, "TIFF (*.tif *.tiff)")
        if fname:
            self.txt_video_path.setText(fname)
            self.update_save_dir_from_video(fname)

    def browse_bf(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Brightfield Image", self.root_dir, "TIFF (*.tif *.tiff)")
        if fname: self.txt_bf_path.setText(fname)

    def browse_beads(self):
        dialog = QFileDialog(self, "Select Bead Folder or File", self.root_dir)
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)

        def on_current_changed(path):
            if os.path.isdir(path):
                dialog.setFileMode(QFileDialog.FileMode.Directory)
            else:
                dialog.setFileMode(QFileDialog.FileMode.ExistingFile)

        dialog.currentChanged.connect(on_current_changed)

        if dialog.exec():
            selected = dialog.selectedFiles()
            if selected:
                res_path = selected[0]
                if os.path.isdir(res_path):
                    res_path = re.sub(r'_\d+$', '', res_path)
                self.txt_bead_path.setText(res_path)

    def browse_roi(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select ROI File", self.root_dir, "ImageJ ROI (*.roi)")
        if fname: self.txt_roi_file.setText(fname)

    def browse_save(self):
        dname = QFileDialog.getExistingDirectory(self, "Select Output Directory", self.root_dir)
        if dname:
            self.txt_save_dir.setText(dname)
            self.load_and_render_results(dname)

    def browse_model(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Model Weights", self.root_dir, "All Files (*)")
        if fname: self.txt_model_path.setText(fname)

    def build_params(self):
        target = self.combo_single_reg_target.currentText()
        roi_file_input = self.txt_roi_file.text().strip()
        
        roi_file = roi_file_input if roi_file_input else ""
        roi_channel = self.combo_roi_channel.currentText()

        p = PreprocessParams(
            config_dir=self.config_dir,
            video_path=self.txt_video_path.text().strip(),
            bf_path=self.txt_bf_path.text().strip(),
            bead_path=self.txt_bead_path.text().strip(),
            manual_registration=(self.combo_bead_mode.currentData() == "manual"),
            manual_pairs=self.spin_num_pairs.value(),
            roi_file=roi_file,
            roi_channel=roi_channel,
            single_reg_target=target,
            save_dir=self.txt_save_dir.text().strip(),
            pixel_size=self.spin_pixel_size.value(),
            num_channels=int(self.combo_num_channels.currentText()),
            use_channel=self.combo_use_channel.currentText(),
            start_channel=self.combo_start_channel.currentText(),
            ALEX=self.chk_alex.isChecked(),
            num_frames=self.spin_num_frames.value(),
            frame_avg=self.spin_frame_avg.value(),
            mask_type=self.combo_mask_type.currentText(),
            model_type=self.combo_model_type.currentText(),
            model=self.txt_model_path.text().strip(),
            model_params=self.spin_model_param.value(),
            dilations=self.spin_dilations.value(),
            inv_bf=self.chk_inv_bf.isChecked(),
            use_otsu=self.combo_use_otsu.currentText(),
            cell_fitting=self.chk_cell_fitting.isChecked(),
            area_filter=self.spin_area_filter.value(),
            eccentricity_filter=self.spin_ecc_filter.value(),
            cell_fit_err=self.spin_cell_fit_err.value(),
            overwrite=self.chk_overwrite.isChecked(),
            overwrite_tmat=self.chk_overwrite_tmat.isChecked(),
            tmats_path=os.path.join(self.config_dir, "transformation_matrix.npy") if self.config_dir else "transformation_matrix.npy",
        )
        self.log(f"Built PreprocessParams -> Target: '{target}', ROI File: '{roi_file or 'None'}', Mask: '{p.mask_type}'")
        return p

    def run_task(self, task_type: str):
        self.log(f"\n==================================================")
        self.log(f"TRIGGERED TASK EXECUTION: '{task_type.upper()}'")
        self.log(f"==================================================")
        
        params = self.build_params()

        if task_type == "bead" and getattr(params, 'manual_registration', False):
            self.log("Launching interactive manual bead registration...")
            try:
                registrar = BeadRegistrar()

                if params.bead_path and os.path.isdir(params.bead_path):
                    bead_dir = params.bead_path
                elif params.bead_path and os.path.exists(os.path.dirname(os.path.abspath(params.bead_path))):
                    bead_dir = os.path.dirname(os.path.abspath(params.bead_path))
                elif self.config_dir and os.path.exists(self.config_dir):
                    bead_dir = self.config_dir
                else:
                    bead_dir = os.path.dirname(os.path.abspath(params.bead_path or "."))

                tmat = registrar.interactive_manual_registration(
                    params.bead_path, bead_dir, num_pairs=params.manual_pairs
                )
                if tmat is not None:
                    sync_registration_files(bead_dir, params.save_dir, self.config_dir)
                    self.log(f"Manual bead registration complete. Files saved in '{bead_dir}' and synced to '{params.save_dir}'.")
                    self.load_and_render_results(params.save_dir)
            except Exception as e:
                self.log(f"Error during manual registration: {e}")
                QMessageBox.critical(self, "Registration Error", str(e))
            return

        self.progress_dialog = QProgressDialog(f"Running Task: {task_type.capitalize()}...", "Cancel", 0, 100, self)
        self.progress_dialog.show()

        print(f"[DEBUG GUI TAB] Launching PreprocessWorkerThread for task '{task_type}'...", flush=True)
        self.worker_thread = PreprocessWorkerThread(params, task_type=task_type, parent=self)
        self.worker_thread.progress_signal.connect(self.on_pipeline_progress)
        self.worker_thread.log_signal.connect(self.append_log)
        self.worker_thread.finished_signal.connect(self.on_pipeline_finished)
        self.worker_thread.error_signal.connect(self.on_pipeline_error)
        self.worker_thread.start()

    def on_pipeline_progress(self, msg):
        self.log(f"[PROGRESS SIGNAL] {msg}")
        if self.progress_dialog:
            self.progress_dialog.setLabelText(msg)

    def on_pipeline_finished(self, save_dir):
        if self.progress_dialog: self.progress_dialog.close()
        self.log(f"[FINISHED SIGNAL] Task completed successfully! Loading results from '{save_dir}'...\n")
        self.load_and_render_results(save_dir)

    def on_pipeline_error(self, error_msg):
        if self.progress_dialog: self.progress_dialog.close()
        self.log(f"[ERROR SIGNAL] Task failed with exception:\n{error_msg}\n")
        QMessageBox.critical(self, "Task Error", f"Execution error occurred:\n{error_msg}")

    def on_bead_view_changed(self, index):
        path = self.combo_bead_view.currentData()
        if path and os.path.exists(path):
            self.log(f"Displaying Bead image: '{path}'")
            ax = self.canvas_beads.axes
            ax.clear()
            ax.imshow(Image.open(path))
            ax.set_xticks([]); ax.set_yticks([])
            self.canvas_beads.draw()

    def on_ch_view_changed(self, index):
        data = self.combo_ch_view.currentData()
        if not data: return

        fig = self.canvas_channels.fig
        fig.clear()

        if isinstance(data, (tuple, list)):
            path_l, path_r = data
            if os.path.exists(path_l) and os.path.exists(path_r):
                self.log(f"Displaying Side-by-Side: '{path_l}' | '{path_r}'")
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)

                ax1.imshow(Image.open(path_l))
                ax1.set_title(os.path.basename(path_l), fontsize=9)
                ax1.set_xticks([]); ax1.set_yticks([])

                ax2.imshow(Image.open(path_r))
                ax2.set_title(os.path.basename(path_r), fontsize=9)
                ax2.set_xticks([]); ax2.set_yticks([])

                fig.tight_layout()
                self.canvas_channels.draw()
        elif isinstance(data, str) and os.path.exists(data):
            self.log(f"Displaying Fluorescence image: '{data}'")
            ax = fig.add_subplot(111)
            ax.imshow(Image.open(data))
            ax.set_title(os.path.basename(data), fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            fig.tight_layout()
            self.canvas_channels.draw()

    def on_bf_view_changed(self, index):
        path = self.combo_bf_view.currentData()
        if path and os.path.exists(path):
            self.log(f"Displaying Brightfield image: '{path}'")
            ax = self.canvas_bf.axes
            ax.clear()
            ax.imshow(Image.open(path))
            ax.set_xticks([]); ax.set_yticks([])
            self.canvas_bf.draw()

    def on_mask_view_changed(self, index):
        path = self.combo_mask_view.currentData()
        if path and os.path.exists(path):
            self.log(f"Displaying Mask image: '{path}'")
            ax = self.canvas_masks.axes
            ax.clear()
            ax.imshow(Image.open(path))
            ax.set_xticks([]); ax.set_yticks([])
            self.canvas_masks.draw()

    def on_cell_folder_changed(self, index):
        folder_path = self.combo_cell_select.currentData()
        folder_name = self.combo_cell_select.currentText()
        if not folder_path or not os.path.exists(folder_path):
            self.lbl_cell_metadata.setText("Status: N/A  |  Cell Number: N/A  |  Mask Area: N/A  |  Mask Eccentricity: N/A  |  Fit Error: N/A")
            return

        cell_match = re.search(r'\d+', folder_name)
        cell_num = int(cell_match.group()) if cell_match else "N/A"

        area_val, ecc_val, fit_err_val = None, None, None
        area_str, ecc_str, fit_err_str = "N/A", "N/A", "N/A"
        
        if self.df_all_objects is not None and not self.df_all_objects.empty:
            num_col = None
            for c in ['label', 'Cell_Number', 'cell_num', 'label_id']:
                if c in self.df_all_objects.columns:
                    num_col = c
                    break
            
            if num_col and cell_match:
                matched_row = self.df_all_objects[self.df_all_objects[num_col] == cell_num]
                if not matched_row.empty:
                    if 'area' in matched_row.columns:
                        area_val = float(matched_row['area'].values[0])
                        area_str = f"{area_val:.1f} px"
                    if 'eccentricity' in matched_row.columns:
                        ecc_val = float(matched_row['eccentricity'].values[0])
                        ecc_str = f"{ecc_val:.4f}"
                    for err_col in ['Error_nm', 'fit_error', 'cell_fit_err', 'fit_err']:
                        if err_col in matched_row.columns:
                            fit_err_val = float(matched_row[err_col].values[0])
                            fit_err_str = f"{fit_err_val:.2f} nm"
                            break

        if area_val is None or ecc_val is None or fit_err_val is None:
            mat_files = glob.glob(os.path.join(folder_path, "*.mat"))
            if mat_files:
                try:
                    mat_data = sio.loadmat(mat_files[0])
                    if 'area' in mat_data and area_val is None:
                        area_val = float(np.squeeze(mat_data['area']))
                        area_str = f"{area_val:.1f} px"
                    if 'eccentricity' in mat_data and ecc_val is None:
                        ecc_val = float(np.squeeze(mat_data['eccentricity']))
                        ecc_str = f"{ecc_val:.4f}"
                    if fit_err_val is None:
                        for err_key in ['Error_nm', 'fit_error', 'cell_fit_err', 'fit_err']:
                            if err_key in mat_data:
                                fit_err_val = float(np.squeeze(mat_data[err_key]))
                                fit_err_str = f"{fit_err_val:.2f} nm"
                                break
                except Exception:
                    pass

        if area_val is not None and ecc_val is not None:
            min_area = self.spin_area_filter.value()
            min_ecc = self.spin_ecc_filter.value()
            max_err = self.spin_cell_fit_err.value()

            is_valid_area = (area_val >= min_area)
            is_valid_ecc = (ecc_val >= min_ecc)
            is_valid_err = (max_err == 0.0 or fit_err_val is None or fit_err_val <= max_err)

            if is_valid_area and is_valid_ecc and is_valid_err:
                status_html = '<span style="color: #28a745; font-weight: bold;">Accepted</span>'
            else:
                status_html = '<span style="color: #dc3545; font-weight: bold;">Rejected</span>'
        else:
            status_html = 'N/A'

        self.lbl_cell_metadata.setText(
            f"Status: {status_html}  |  Cell Number: {cell_num}  |  Mask Area: {area_str}  |  Mask Eccentricity: {ecc_str}  |  Fit Error: {fit_err_str}"
        )

        self.combo_cell_plot_select.blockSignals(True)
        self.combo_cell_plot_select.clear()

        plot_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]

        def plot_priority_key(fname):
            flower = fname.lower()
            if 'line_profile_location' in flower:
                return (0, flower)
            elif 'fitted' in flower:
                return (1, flower)
            elif 'full_length' in flower or 'left_vs_right' in flower or 'line_profile' in flower:
                return (2, flower)
            elif 'intensity' in flower:
                return (3, flower)
            else:
                return (4, flower)

        plot_files.sort(key=plot_priority_key)

        for p_file in plot_files:
            full_p = os.path.join(folder_path, p_file)
            self.combo_cell_plot_select.addItem(p_file, userData=full_p)

        self.combo_cell_plot_select.blockSignals(False)
        if self.combo_cell_plot_select.count() > 0:
            self.on_cell_plot_changed(0)

    def on_cell_plot_changed(self, index):
        plot_path = self.combo_cell_plot_select.currentData()
        if plot_path and os.path.exists(plot_path):
            self.log(f"Displaying Cell Inspector Plot: '{plot_path}'")
            ax = self.canvas_cell_plot.axes
            ax.clear()
            ax.imshow(Image.open(plot_path))
            ax.set_title(os.path.basename(plot_path), fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            self.canvas_cell_plot.fig.tight_layout()
            self.canvas_cell_plot.draw()

    def load_and_render_results(self, save_dir):
        raw_dirs = [save_dir, self.config_dir, self.root_dir]
        base_search_dirs = []
        for d in raw_dirs:
            if d and os.path.exists(d):
                real_d = os.path.realpath(d)
                if real_d not in base_search_dirs:
                    base_search_dirs.append(real_d)

        self.log(f"Searching for output results in unique directories: {base_search_dirs}")
        
        self.df_all_objects = None
        for d in base_search_dirs:
            csv_path = os.path.join(d, "all_objects.csv")
            if os.path.exists(csv_path):
                try:
                    self.df_all_objects = pd.read_csv(csv_path)
                    self.log(f"Loaded 'all_objects.csv' with {len(self.df_all_objects)} object records.")
                    break
                except Exception as e:
                    self.log(f"Failed to read 'all_objects.csv': {e}")

        for d in base_search_dirs:
            tmat_p = os.path.join(d, "transformation_matrix.npy")
            if os.path.exists(tmat_p):
                tmat = np.load(tmat_p)
                m_str = "\n".join(["[" + "  ".join([f"{v:10.5f}" for v in row]) + "]" for row in tmat])
                self.lbl_matrix_display.setText(f"Matrix ({os.path.basename(d)}/transformation_matrix.npy):\n{m_str}")
                self.log(f"Loaded matrix from '{tmat_p}'")
                break

        for d in base_search_dirs:
            reg_csv_path = os.path.join(d, "registration.csv")
            ncc_csv_path = os.path.join(d, "ncc_scores.csv")
            if os.path.exists(reg_csv_path):
                try:
                    df_reg = pd.read_csv(reg_csv_path)
                    if 'Best_NCC' in df_reg.columns:
                        best_ncc = df_reg['Best_NCC'].iloc[0]
                        best_folder = df_reg['Best_Folder'].iloc[0] if 'Best_Folder' in df_reg.columns else "N/A"
                        clean_name = os.path.basename(os.path.normpath(str(best_folder)))
                        
                        self.lbl_ncc_display.setText(f"Best File/Folder: {clean_name}\nNCC Score: {best_ncc:.4f}")
                        self.log(f"Loaded Best Folder ('{clean_name}') with NCC ({best_ncc:.4f}) from '{reg_csv_path}'")
                        break
                except Exception as e:
                    self.log(f"Error reading '{reg_csv_path}': {e}")
            elif os.path.exists(ncc_csv_path):
                try:
                    df_ncc = pd.read_csv(ncc_csv_path)
                    if 'NCC_After' in df_ncc.columns and 'Bead_Folder' in df_ncc.columns:
                        best_idx = df_ncc['NCC_After'].idxmax()
                        best_folder = df_ncc.loc[best_idx, 'Bead_Folder']
                        best_ncc = df_ncc.loc[best_idx, 'NCC_After']
                        clean_name = os.path.basename(os.path.normpath(str(best_folder)))
                        
                        self.lbl_ncc_display.setText(f"Best File/Folder: {clean_name}\nNCC Score: {best_ncc:.4f}")
                        self.log(f"Loaded Best Folder ('{clean_name}') with NCC ({best_ncc:.4f}) from '{ncc_csv_path}'")
                        break
                except Exception as e:
                    self.log(f"Error reading '{ncc_csv_path}': {e}")

        # 1. Populate Bead Registration Tab
        self.combo_bead_view.blockSignals(True)
        self.combo_bead_view.clear()
        added_bead_files = set()
        for d in base_search_dirs:
            for fname in os.listdir(d):
                fname_lower = fname.lower()
                is_bead_file = any(k in fname_lower for k in ['bead', 'registration', 'registered', 'ncc'])                
                is_mask_file = any(k in fname_lower for k in ['mask', 'fits', 'cell'])
                
                if fname not in added_bead_files and is_bead_file and not is_mask_file and fname.endswith(('.png', '.jpg', '.tif', '.tiff')):
                    p = os.path.join(d, fname)
                    self.combo_bead_view.addItem(fname, userData=p)
                    added_bead_files.add(fname)
                    self.log(f"Added bead view image: '{fname}'")
        self.combo_bead_view.blockSignals(False)
        if self.combo_bead_view.count() > 0: self.on_bead_view_changed(0)

        # 2. Populate Fluorescence Channels Tab
        self.combo_ch_view.blockSignals(True)
        self.combo_ch_view.clear()

        l_avg_p, r_avg_p = None, None
        l_mask_p, r_mask_p = None, None
        l_fits_p, r_fits_p = None, None

        for d in base_search_dirs:
            if not l_avg_p and os.path.exists(os.path.join(d, "L_avg.png")): l_avg_p = os.path.join(d, "L_avg.png")
            if not r_avg_p and os.path.exists(os.path.join(d, "R_avg.png")): r_avg_p = os.path.join(d, "R_avg.png")
            if not l_mask_p and os.path.exists(os.path.join(d, "L_mask_overlay.png")): l_mask_p = os.path.join(d, "L_mask_overlay.png")
            if not r_mask_p and os.path.exists(os.path.join(d, "R_mask_overlay.png")): r_mask_p = os.path.join(d, "R_mask_overlay.png")
            if not l_fits_p and os.path.exists(os.path.join(d, "L_fits_overlay.png")): l_fits_p = os.path.join(d, "L_fits_overlay.png")
            if not r_fits_p and os.path.exists(os.path.join(d, "R_fits_overlay.png")): r_fits_p = os.path.join(d, "R_fits_overlay.png")

        if l_avg_p and r_avg_p: self.combo_ch_view.addItem("L & R Side-by-Side (Averages)", userData=(l_avg_p, r_avg_p))
        if l_mask_p and r_mask_p: self.combo_ch_view.addItem("L & R Side-by-Side (Mask Overlays)", userData=(l_mask_p, r_mask_p))
        if l_fits_p and r_fits_p: self.combo_ch_view.addItem("L & R Side-by-Side (Fit Overlays)", userData=(l_fits_p, r_fits_p))

        fl_targets = [
            "L_avg.png", "R_avg.png", "L_mask_overlay.png", "R_mask_overlay.png",
            "L_fits_overlay.png", "R_fits_overlay.png",
            "line_profile_location_L.png", "line_profile_location_R.png", "line_profile_location.png"
        ]
        added_fl_files = set()
        for fname in fl_targets:
            for d in base_search_dirs:
                p = os.path.join(d, fname)
                if os.path.exists(p) and fname not in added_fl_files:
                    self.combo_ch_view.addItem(fname, userData=p)
                    added_fl_files.add(fname)
                    break

        for d in base_search_dirs:
            for fname in os.listdir(d):
                fname_lower = fname.lower()
                if fname not in added_fl_files and fname_lower.endswith(('.png', '.jpg', '.tif', '.tiff')):
                    if any(k in fname_lower for k in ['avg', 'chan', 'line_profile']):
                        p = os.path.join(d, fname)
                        self.combo_ch_view.addItem(fname, userData=p)
                        added_fl_files.add(fname)

        self.combo_ch_view.blockSignals(False)
        if self.combo_ch_view.count() > 0: self.on_ch_view_changed(0)

        # 3. Populate Brightfield Tab
        self.combo_bf_view.blockSignals(True)
        self.combo_bf_view.clear()
        bf_targets = ["BF.png", "BF_R.png", "BF_unregistered_roi.png", "BF_Registered_overlay.png"]
        added_bf_files = set()
        for fname in bf_targets:
            if fname in added_bf_files: continue
            for d in base_search_dirs:
                p = os.path.join(d, fname)
                if os.path.exists(p):
                    self.combo_bf_view.addItem(fname, userData=p)
                    added_bf_files.add(fname)
                    break

        for d in base_search_dirs:
            for fname in os.listdir(d):
                fname_lower = fname.lower()
                if fname not in added_bf_files and fname_lower.endswith(('.png', '.jpg', '.tif', '.tiff')):
                    if 'bf' in fname_lower or 'brightfield' in fname_lower:
                        p = os.path.join(d, fname)
                        self.combo_bf_view.addItem(fname, userData=p)
                        added_bf_files.add(fname)

        self.combo_bf_view.blockSignals(False)
        if self.combo_bf_view.count() > 0: self.on_bf_view_changed(0)

        # 4. Populate Masks & Outlines Tab
        self.combo_mask_view.blockSignals(True)
        self.combo_mask_view.clear()
        mask_targets = [
            "accepted_rejected_cells.png",  
            "all_cells_mask_colored.png", 
            "all_cells_mask_L_colored.png", 
            "all_cells_mask_R_colored.png",
            "all_cells_mask.png", 
            "BF_mask_overlay.png", "BF_R_mask_overlay.png", 
            "L_mask_overlay.png", "R_mask_overlay.png", 
            "BF_fits_overlay.png", "BF_R_fits_overlay.png", 
            "L_fits_overlay.png", "R_fits_overlay.png"
        ]
        added_mask_files = set()
        for fname in mask_targets:
            if fname in added_mask_files: continue
            for d in base_search_dirs:
                p = os.path.join(d, fname)
                if os.path.exists(p):
                    self.combo_mask_view.addItem(fname, userData=p)
                    added_mask_files.add(fname)
                    break

        for d in base_search_dirs:
            for fname in os.listdir(d):
                fname_lower = fname.lower()
                if fname not in added_mask_files and fname_lower.endswith(('.png', '.jpg', '.tif', '.tiff')):
                    if any(k in fname_lower for k in ['mask', 'fit', 'overlay']):
                        p = os.path.join(d, fname)
                        self.combo_mask_view.addItem(fname, userData=p)
                        added_mask_files.add(fname)

        self.combo_mask_view.blockSignals(False)
        if self.combo_mask_view.count() > 0: self.on_mask_view_changed(0)

        # 5. Populate Feature Summary Tab
        feature_targets = [
            "obj_feature_summary.png", "ncc_scores.png",
            "line_profile_location_L.png", "line_profile_location_R.png", "line_profile_location.png"
        ]
        for fname in feature_targets:
            found = False
            for d in base_search_dirs:
                p = os.path.join(d, fname)
                if os.path.exists(p):
                    self.log(f"Displaying Feature Summary image: '{p}'")
                    ax = self.canvas_features.axes
                    ax.clear()
                    ax.imshow(Image.open(p))
                    ax.set_xticks([]); ax.set_yticks([])
                    self.canvas_features.draw()
                    found = True
                    break
            if found: break

        # 6. Populate Cell Inspector Tab
        self.combo_cell_select.blockSignals(True)
        self.combo_cell_select.clear()

        found_cell_dirs = []
        for d in base_search_dirs:
            subdirs = glob.glob(os.path.join(d, "cell_*"))
            for sd in subdirs:
                if os.path.isdir(sd):
                    folder_name = os.path.basename(sd)
                    if folder_name not in [c[0] for c in found_cell_dirs]:
                        found_cell_dirs.append((folder_name, sd))

        def natural_sort_key(item):
            match = re.search(r'\d+', item[0])
            return int(match.group()) if match else 0

        found_cell_dirs.sort(key=natural_sort_key)

        for folder_name, folder_path in found_cell_dirs:
            self.combo_cell_select.addItem(folder_name, userData=folder_path)

        self.combo_cell_select.blockSignals(False)
        if self.combo_cell_select.count() > 0:
            self.on_cell_folder_changed(0)