# -*- coding: utf-8 -*-
import sys
import os
import glob
import re
import time
import traceback
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QGroupBox, QFormLayout,
    QLabel, QPushButton, QFileDialog, QLineEdit, QComboBox,
    QSpinBox, QDoubleSpinBox, QScrollArea, QProgressDialog, QMessageBox, QTextEdit,
    QDialog, QCheckBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QCoreApplication

ORIGINAL_PLT_SHOW = plt.show

# Import backend modules
try:
    import fields_of_view
    FOV_MODULE_LOADED = True
except ImportError:
    try:
        from studio import fields_of_view
        FOV_MODULE_LOADED = True
    except ImportError:
        FOV_MODULE_LOADED = False

try:
    import preprocess
    from preprocess import AnalysisPipeline
    PREPROCESS_LOADED = True
except ImportError:
    try:
        from studio import preprocess
        from studio.preprocess import AnalysisPipeline
        PREPROCESS_LOADED = True
    except ImportError:
        PREPROCESS_LOADED = False


class QtLogStream:
    """Stream object that redirects stdout/stderr writes safely to a Qt log widget."""
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


class NoScrollComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)


class FOVPreprocessParams:
    """Parameters wrapper object for AnalysisPipeline."""
    def __init__(self, **kwargs):
        self.name = kwargs.get('name', 'preprocessing_results')
        self.video_path = kwargs.get('video_path', '')
        self.bf_path = kwargs.get('bf_path', '')
        self.bead_path = kwargs.get('bead_path', '')
        self.roi_file = kwargs.get('roi_file', '')
        self.roi_channel = kwargs.get('roi_channel', 'L')
        self.save_dir = kwargs.get('save_dir', '.')
        self.tmats_path = kwargs.get('tmats_path', 'transformation_matrix.npy')
        
        self.pixel_size = kwargs.get('pixel_size', 0.120)
        self.pxsize = self.pixel_size
        self.num_channels = kwargs.get('num_channels', 2)
        self.use_channel = kwargs.get('use_channel', 'L')
        self.channel = kwargs.get('channel', self.use_channel)
        self.start_channel = kwargs.get('start_channel', 'L')
        self.ALEX = kwargs.get('ALEX', False)
        self.num_frames = kwargs.get('num_frames', 0)
        self.frame_avg = kwargs.get('frame_avg', 5)
        
        self.mask_type = kwargs.get('mask_type', 'MANUAL')
        self.model_type = kwargs.get('model_type', 'unet')
        self.model = kwargs.get('model', '')
        self.model_params = kwargs.get('model_params', 0.5)
        self.threshold_param = kwargs.get('threshold_param', 0.5)
        self.dilations = kwargs.get('dilations', 1)
        self.inv_bf = kwargs.get('inv_bf', False)
        self.use_otsu = kwargs.get('use_otsu', 'False')
        self.mask_prefix = kwargs.get('mask_prefix', 'mask')
        
        self.cell_fitting = kwargs.get('cell_fitting', True)
        self.area_filter = kwargs.get('area_filter', 50)
        self.eccentricity_filter = kwargs.get('eccentricity_filter', 0.0)
        self.overwrite = kwargs.get('overwrite', False)


class MaskSelectionDialog(QDialog):
    """Popup dialog with checkboxes to select which FoV masks to edit."""
    def __init__(self, results_dirs, folder_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Fields of View Masks to Edit")
        self.setMinimumWidth(520)
        self.setMinimumHeight(380)

        self.results_dirs = results_dirs
        self.folder_name = folder_name
        self.checkboxes = []

        layout = QVBoxLayout(self)

        lbl = QLabel(
            "<b>Select which Fields of View masks you would like to edit interactively:</b><br>"
            "<small>Each selected mask will open sequentially in the editor window, "
            "followed by cell shape fitting and image regeneration.</small>"
        )
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

        # Scroll area for checkboxes
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(6)

        for res_dir in self.results_dirs:
            fov_path = os.path.dirname(res_dir)
            rel_fov = os.path.relpath(fov_path, self.folder_name)
            display_name = rel_fov if rel_fov != "." else os.path.basename(fov_path)

            chk = QCheckBox(f"FOV: {display_name}  ({os.path.basename(res_dir)})")
            chk.setChecked(True)
            chk.setProperty("results_path", res_dir)
            self.checkboxes.append(chk)
            scroll_layout.addWidget(chk)

        scroll_layout.addStretch()
        scroll_content.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)

        # Selection Control Buttons
        btn_sel_lay = QHBoxLayout()
        btn_all = QPushButton("Select All")
        btn_all.clicked.connect(self.select_all)
        btn_none = QPushButton("Deselect All")
        btn_none.clicked.connect(self.deselect_all)
        btn_sel_lay.addWidget(btn_all)
        btn_sel_lay.addWidget(btn_none)
        btn_sel_lay.addStretch()
        layout.addLayout(btn_sel_lay)

        # Dialog Action Buttons
        btn_box = QHBoxLayout()
        btn_ok = QPushButton("Start Editing Selected Masks")
        btn_ok.setStyleSheet("font-weight: bold; background-color: #28a745; color: white; padding: 6px;")
        btn_ok.clicked.connect(self.accept)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)

        btn_box.addStretch()
        btn_box.addWidget(btn_cancel)
        btn_box.addWidget(btn_ok)
        layout.addLayout(btn_box)

    def select_all(self):
        for chk in self.checkboxes:
            chk.setChecked(True)

    def deselect_all(self):
        for chk in self.checkboxes:
            chk.setChecked(False)

    def get_selected_dirs(self):
        return [chk.property("results_path") for chk in self.checkboxes if chk.isChecked()]


class FOVWorkerThread(QThread):
    """Background execution thread for fields_of_view main process."""
    progress_signal = pyqtSignal(str)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, folder_name, save_dir, num_channels, channel, tmats_path, parent=None):
        super().__init__(parent)
        self.folder_name = folder_name
        self.save_dir = save_dir
        self.num_channels = num_channels
        self.channel = channel
        self.tmats_path = tmats_path

    def run(self):
        old_stdout, old_stderr = sys.stdout, sys.stderr
        stream = QtLogStream(self.log_signal)
        sys.stdout, sys.stderr = stream, stream

        start_time = time.time()
        try:
            import matplotlib
            import matplotlib.pyplot as plt

            if not hasattr(plt, '_original_show'):
                plt._original_show = ORIGINAL_PLT_SHOW if 'ORIGINAL_PLT_SHOW' in globals() else plt.show

            matplotlib.use('Agg', force=True)
            plt.ioff()
            plt.show = lambda *args, **kwargs: None

            self.progress_signal.emit("Scanning directory and generating Fields of View summary figure...")
            print(f"[FOV WORKER] Starting processing in target folder: '{self.folder_name}'", flush=True)

            if not FOV_MODULE_LOADED:
                raise ImportError("The `fields_of_view` backend module could not be loaded.")

            os.makedirs(self.save_dir, exist_ok=True)
            
            fields_of_view.main(
                folder_name=self.folder_name,
                save_dir=self.save_dir,
                num_channels=int(self.num_channels),
                channel=self.channel,
                tmats_path=self.tmats_path if (self.tmats_path and os.path.exists(self.tmats_path)) else None
            )

            out_png = os.path.join(self.save_dir, "All_FoVs.png")
            elapsed = time.time() - start_time
            print(f"[FOV WORKER] Processing completed in {elapsed:.2f} seconds.", flush=True)

            sys.stdout, sys.stderr = old_stdout, old_stderr
            self.finished_signal.emit(out_png)

        except Exception as e:
            err_details = f"[FOV WORKER ERROR] {str(e)}\n\n{traceback.format_exc()}"
            print(err_details, flush=True)
            sys.stdout, sys.stderr = old_stdout, old_stderr
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
            sys.stdout, sys.stderr = old_stdout, old_stderr


class FieldsOfViewTab(QWidget):
    def __init__(self, parent=None, root_dir="."):
        super().__init__(parent)
        self.parent = parent
        self.root_dir = os.path.abspath(root_dir) if root_dir else os.getcwd()
        self.config_dir = self.root_dir
        self.worker_thread = None
        self.progress_dialog = None
        self.init_ui()
        if self.root_dir and os.path.exists(self.root_dir):
            self.scan_and_load_config()

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
        formatted_msg = f"[FOV TAB] {msg}"
        print(formatted_msg, flush=True)
        self.append_log(formatted_msg + "\n")

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # --- Left Control Sidebar ---
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(8, 8, 8, 8)
        sidebar_layout.setSpacing(10)

        # 1. Config Setup Group
        config_group = QGroupBox("Configuration Setup")
        config_layout = QFormLayout()
        self.txt_config_file = QLineEdit()
        self.txt_config_file.setPlaceholderText("Select config.txt file")
        btn_browse_config = QPushButton("Browse...")
        btn_browse_config.clicked.connect(self.browse_config_file)
        lay_config = QHBoxLayout()
        lay_config.addWidget(self.txt_config_file)
        lay_config.addWidget(btn_browse_config)
        config_layout.addRow("Dataset Config (config.txt):", lay_config)
        config_group.setLayout(config_layout)
        sidebar_layout.addWidget(config_group)

        # 2. Parameters Group
        fov_group = QGroupBox("Fields of View Parameters")
        fov_layout = QFormLayout()

        self.txt_folder = QLineEdit(self.root_dir)
        btn_browse_folder = QPushButton("Browse...")
        btn_browse_folder.clicked.connect(self.browse_folder)
        lay_folder = QHBoxLayout()
        lay_folder.addWidget(self.txt_folder)
        lay_folder.addWidget(btn_browse_folder)

        self.txt_save_dir = QLineEdit(self.root_dir)
        btn_browse_save = QPushButton("Browse...")
        btn_browse_save.clicked.connect(self.browse_save_dir)
        lay_save = QHBoxLayout()
        lay_save.addWidget(self.txt_save_dir)
        lay_save.addWidget(btn_browse_save)

        self.combo_num_channels = NoScrollComboBox()
        self.combo_num_channels.addItems(["1", "2"])
        self.combo_num_channels.setCurrentText("2")

        self.combo_channel = NoScrollComboBox()
        self.combo_channel.addItems(["L", "R"])
        self.combo_channel.setCurrentText("L")

        self.txt_tmats_path = QLineEdit()
        self.txt_tmats_path.setPlaceholderText("Path to transformation_matrix.npy")
        btn_browse_tmats = QPushButton("Browse...")
        btn_browse_tmats.clicked.connect(self.browse_tmats)
        lay_tmats = QHBoxLayout()
        lay_tmats.addWidget(self.txt_tmats_path)
        lay_tmats.addWidget(btn_browse_tmats)

        fov_layout.addRow("Target Folder:", lay_folder)
        fov_layout.addRow("Output Directory:", lay_save)
        fov_layout.addRow("Channels:", self.combo_num_channels)
        fov_layout.addRow("Default Channel:", self.combo_channel)
        fov_layout.addRow("Transformation Matrix:", lay_tmats)

        fov_group.setLayout(fov_layout)
        sidebar_layout.addWidget(fov_group)

        # 3. Mask Editing & Refitting Group
        edit_group = QGroupBox("Interactive Mask Editing & Fitting")
        edit_layout = QFormLayout()

        self.spin_area_filter = QSpinBox()
        self.spin_area_filter.setRange(0, 100000)
        self.spin_area_filter.setValue(50)
        self.spin_area_filter.setToolTip("Minimum cell area in pixels for shape fitting")

        self.spin_ecc_filter = QDoubleSpinBox()
        self.spin_ecc_filter.setRange(0.00, 1.00)
        self.spin_ecc_filter.setSingleStep(0.05)
        self.spin_ecc_filter.setDecimals(2)
        self.spin_ecc_filter.setValue(0.00)
        self.spin_ecc_filter.setToolTip("Minimum eccentricity filter (0.0 to 1.0)")

        self.btn_edit_masks = QPushButton("Edit Selected FOV Masks...")
        self.btn_edit_masks.setStyleSheet("font-weight: bold; background-color: #6c757d; color: white; padding: 8px;")
        self.btn_edit_masks.clicked.connect(self.edit_selected_masks)

        edit_layout.addRow("Min Area Filter (px):", self.spin_area_filter)
        edit_layout.addRow("Min Eccentricity:", self.spin_ecc_filter)
        edit_layout.addRow(self.btn_edit_masks)

        edit_group.setLayout(edit_layout)
        sidebar_layout.addWidget(edit_group)

        # 4. Execution & Log Group
        exec_group = QGroupBox("Execution")
        exec_layout = QVBoxLayout()
        self.btn_run_fov = QPushButton("Run Fields of View Analysis")
        self.btn_run_fov.setStyleSheet("font-weight: bold; background-color: #008080; color: white; padding: 8px;")
        self.btn_run_fov.clicked.connect(self.run_fov_analysis)
        exec_layout.addWidget(self.btn_run_fov)
        exec_group.setLayout(exec_layout)
        sidebar_layout.addWidget(exec_group)

        self.txt_status_log = QTextEdit()
        self.txt_status_log.setReadOnly(True)
        self.txt_status_log.setMaximumHeight(200)
        sidebar_layout.addWidget(self.txt_status_log)

        scroll_area.setWidget(sidebar)
        splitter.addWidget(scroll_area)

        # --- Right Display Panel ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.lbl_info = QLabel("Fields of View Summary Display")
        self.lbl_info.setStyleSheet("font-weight: bold; font-size: 13px; padding: 4px;")
        right_layout.addWidget(self.lbl_info)

        self.canvas_fov = MplCanvas(self)
        right_layout.addWidget(NavigationToolbar(self.canvas_fov, self))
        right_layout.addWidget(self.canvas_fov)

        splitter.addWidget(right_panel)
        splitter.setSizes([360, 940])

    def scan_and_load_config(self):
        if not self.root_dir or not os.path.exists(self.root_dir):
            return
        self.log(f"Scanning for config files in: '{self.root_dir}'")
        config_matches = glob.glob(os.path.join(self.root_dir, "**", "*config*.txt"), recursive=True)
        if config_matches:
            self.log(f"Found config file: '{config_matches[0]}'")
            self.txt_config_file.setText(config_matches[0])
            self.load_config_file(config_matches[0])

    def load_config_file(self, filepath):
        if not filepath or not os.path.exists(filepath):
            return
        self.log(f"Loading settings from config file: '{filepath}'")
        self.config_dir = os.path.dirname(os.path.abspath(filepath))
        self.txt_folder.setText(self.config_dir)
        if not self.txt_save_dir.text() or self.txt_save_dir.text() == ".":
            self.txt_save_dir.setText(self.config_dir)

        def resolve_path(p):
            p = p.strip('\'"')
            return p if os.path.isabs(p) else os.path.normpath(os.path.join(self.config_dir, p))

        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.split('#')[0].split('//')[0].strip()
                    if not line:
                        continue
                    parts = re.split(r'[:=\t]+', line, maxsplit=1)
                    if len(parts) < 2:
                        continue
                    key, val = parts[0].strip().lower().replace(' ', '_'), parts[1].strip()

                    if key in ['num_channels', 'channels']:
                        m = re.search(r"\d+", val)
                        if m and self.combo_num_channels.findText(m.group()) != -1:
                            self.combo_num_channels.setCurrentText(m.group())
                    elif key in ['channel', 'use_channel']:
                        clean_ch = val.upper()
                        if clean_ch in ['L', 'R']:
                            self.combo_channel.setCurrentText(clean_ch)
                    elif key in ['tmats_path', 'transformation_matrix', 'matrix_path']:
                        self.txt_tmats_path.setText(resolve_path(val))
                    elif key in ['save_dir', 'output_dir']:
                        if val:
                            self.txt_save_dir.setText(resolve_path(val))
                    elif key in ['area_filter', 'min_area']:
                        m = re.search(r"\d+", val)
                        if m:
                            self.spin_area_filter.setValue(int(m.group()))
                    elif key in ['eccentricity_filter', 'ecc_filter', 'min_eccentricity']:
                        try:
                            self.spin_ecc_filter.setValue(float(val))
                        except ValueError:
                            pass
        except Exception as e:
            self.log(f"Error parsing config file '{filepath}': {e}")

        expected_out = os.path.join(self.txt_save_dir.text(), "All_FoVs.png")
        if os.path.exists(expected_out):
            self.render_result_image(expected_out)

    def browse_config_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Config File", self.root_dir, "Config Files (*config*.txt *.cfg *.txt)")
        if fname:
            self.txt_config_file.setText(fname)
            self.load_config_file(fname)

    def browse_folder(self):
        dname = QFileDialog.getExistingDirectory(self, "Select Target Folder", self.txt_folder.text() or self.root_dir)
        if dname:
            self.txt_folder.setText(dname)

    def browse_save_dir(self):
        dname = QFileDialog.getExistingDirectory(self, "Select Output Directory", self.txt_save_dir.text() or self.root_dir)
        if dname:
            self.txt_save_dir.setText(dname)

    def browse_tmats(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Transformation Matrix File", self.config_dir, "Numpy Matrix (*.npy);;All Files (*)")
        if fname:
            self.txt_tmats_path.setText(fname)

    def edit_selected_masks(self):
        """Scans for results directories, opens a selection popup, and edits masks sequentially."""
        if not PREPROCESS_LOADED:
            QMessageBox.critical(self, "Module Error", "The `preprocess` module is required for editing masks but could not be loaded.")
            return

        folder_name = self.txt_folder.text()
        if not folder_name or not os.path.exists(folder_name):
            QMessageBox.warning(self, "Invalid Directory", f"The target folder does not exist:\n{folder_name}")
            return

        # Find all 'results' directories inside folder_name
        search_pattern = os.path.join(folder_name, "**", "results")
        results_dirs = glob.glob(search_pattern, recursive=True)
        results_dirs = [
            r for r in results_dirs 
            if os.path.isdir(r) and not os.path.basename(os.path.dirname(r)).startswith("BF")
        ]

        if not results_dirs:
            QMessageBox.warning(self, "No Results Found", f"No 'results' directories found under:\n{folder_name}")
            return

        def get_sort_key(s):
            nums = re.findall(r'\d+', s)
            return [int(n) for n in nums] if nums else [s]

        results_dirs = sorted(results_dirs, key=get_sort_key)

        # Popup selection dialog
        dialog = MaskSelectionDialog(results_dirs, folder_name, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        selected_dirs = dialog.get_selected_dirs()
        if not selected_dirs:
            QMessageBox.information(self, "No Selection", "No FoV mask directories were selected for editing.")
            return

        # Configure Matplotlib for interactive Qt GUI window rendering
        import matplotlib
        import matplotlib.pyplot as plt

        if hasattr(plt, '_original_show'):
            plt.show = plt._original_show
        elif 'ORIGINAL_PLT_SHOW' in globals():
            plt.show = ORIGINAL_PLT_SHOW

        matplotlib.use('QtAgg', force=True)
        plt.ion()

        num_channels = int(self.combo_num_channels.currentText())
        channel = self.combo_channel.currentText()
        tmats_path = self.txt_tmats_path.text()
        area_filter = self.spin_area_filter.value()
        eccentricity_filter = self.spin_ecc_filter.value()

        for idx, res_dir in enumerate(selected_dirs, start=1):
            fov_path = os.path.dirname(res_dir)
            rel_fov = os.path.relpath(fov_path, folder_name)
            name = rel_fov if rel_fov != "." else os.path.basename(fov_path)

            self.log(f"\n--------------------------------------------------")
            self.log(f"[{idx}/{len(selected_dirs)}] EDITING MASK FOR FOV: '{name}'")
            self.log(f"Directory: {res_dir}")
            self.log(f"Filters -> Area: {area_filter} px, Eccentricity: {eccentricity_filter:.2f}")
            self.log(f"--------------------------------------------------")

            params = FOVPreprocessParams(
                save_dir=res_dir,
                num_channels=num_channels,
                use_channel=channel,
                channel=channel,
                tmats_path=tmats_path if (tmats_path and os.path.exists(tmats_path)) else "transformation_matrix.npy",
                cell_fitting=True,
                area_filter=area_filter,
                eccentricity_filter=eccentricity_filter,
                overwrite=False
            )

            try:
                pipeline = AnalysisPipeline(params)
                edited_mask = pipeline.edit_existing_mask()

                if edited_mask is not None:
                    self.log(f"Mask edit saved for '{name}'. Running cell shape fitting...")
                    pipeline.run_fitting()
                    self.log(f"Cell shape fitting completed for '{name}'.")
                else:
                    self.log(f"Mask editing was skipped or canceled for '{name}'.")
            except Exception as e:
                err_msg = f"Error during mask edit/refit for '{name}': {e}\n{traceback.format_exc()}"
                self.log(err_msg)
                QMessageBox.critical(self, "Mask Edit Error", f"An error occurred in FOV '{name}':\n{e}")

        self.log("\n==================================================")
        self.log("MASK EDITING COMPLETE. RE-GENERATING FOV SUMMARY")
        self.log("==================================================")
        self.run_fov_analysis()

    def run_fov_analysis(self):
        folder = self.txt_folder.text()
        save_dir = self.txt_save_dir.text() or folder
        num_channels = int(self.combo_num_channels.currentText())
        channel = self.combo_channel.currentText()
        tmats_path = self.txt_tmats_path.text()

        if not folder or not os.path.exists(folder):
            QMessageBox.warning(self, "Invalid Directory", f"The target folder does not exist:\n{folder}")
            return

        self.log("\n==================================================")
        self.log("RUNNING FIELDS OF VIEW ANALYSIS")
        self.log(f"Folder: {folder}")
        self.log(f"Save Dir: {save_dir}")
        self.log(f"Channels: {num_channels}, Channel: {channel}")
        self.log(f"Transformation Matrix: {tmats_path}")
        self.log("==================================================")

        self.progress_dialog = QProgressDialog("Generating Fields of View Summary...", "Cancel", 0, 100, self)
        self.progress_dialog.show()

        self.worker_thread = FOVWorkerThread(
            folder_name=folder,
            save_dir=save_dir,
            num_channels=num_channels,
            channel=channel,
            tmats_path=tmats_path,
            parent=self
        )
        self.worker_thread.progress_signal.connect(self.on_progress)
        self.worker_thread.log_signal.connect(self.append_log)
        self.worker_thread.finished_signal.connect(self.on_finished)
        self.worker_thread.error_signal.connect(self.on_error)
        self.worker_thread.start()

    def on_progress(self, msg):
        self.log(f"[PROGRESS] {msg}")
        if self.progress_dialog:
            self.progress_dialog.setLabelText(msg)

    def on_finished(self, out_path):
        if self.progress_dialog:
            self.progress_dialog.close()
        self.log(f"[FINISHED] Successfully generated summary figure: {out_path}")
        self.render_result_image(out_path)

    def on_error(self, err_msg):
        if self.progress_dialog:
            self.progress_dialog.close()
        self.log(f"[ERROR] Task failed:\n{err_msg}")
        QMessageBox.critical(self, "Fields of View Error", f"An error occurred:\n{err_msg}")

    def render_result_image(self, img_path):
        if img_path and os.path.exists(img_path):
            self.lbl_info.setText(f"Summary Image: {img_path}")
            ax = self.canvas_fov.axes
            ax.clear()
            img = Image.open(img_path)
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            self.canvas_fov.fig.tight_layout()
            self.canvas_fov.draw()
            self.log(f"Rendered summary image '{img_path}' on canvas.")
        else:
            self.log(f"Summary image not found at '{img_path}'.")