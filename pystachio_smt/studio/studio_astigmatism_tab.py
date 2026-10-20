# =========================================================================
# TAB 5: 3D ASTIGMATISM (Fully Fixed Version)
# =========================================================================

import os
import glob
import csv
import numpy as np
import matplotlib
matplotlib.use('QtAgg')

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QSlider, QLabel, QGroupBox, QFileDialog, QCheckBox)
from PyQt6.QtCore import Qt

# Fallback popup in case studio_helpers isn't available
try:
    from studio.studio_helpers import show_popup
except ImportError:
    from PyQt6.QtWidgets import QMessageBox
    def show_popup(parent, title, message, critical=False):
        icon = QMessageBox.Icon.Critical if critical else QMessageBox.Icon.Information
        msg = QMessageBox(parent)
        msg.setIcon(icon)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.exec()


class Astigmatism3DTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        
        self.directory = ""
        self.calibration_spline = None
        self.ratio_min = 0.0
        self.ratio_max = 1.0
        self.data_store = []  
        self.plotted_tracks = []
        
        self.init_ui()

    def _create_scaled_slider_group(self, title, min_val, max_val, default_min, default_max, scale):
        layout = QVBoxLayout()
        title_label = QLabel(f"<b>{title}</b>")
        layout.addWidget(title_label)
        
        row_min = QHBoxLayout()
        sl_min = QSlider(Qt.Orientation.Horizontal)
        sl_min.setRange(int(min_val * scale), int(max_val * scale))
        sl_min.setValue(int(default_min * scale))
        lbl_min = QLabel(f"{default_min:.2f}")
        lbl_min.setFixedWidth(65)
        row_min.addWidget(QLabel("Min:"))
        row_min.addWidget(sl_min)
        row_min.addWidget(lbl_min)
        layout.addLayout(row_min)
        
        row_max = QHBoxLayout()
        sl_max = QSlider(Qt.Orientation.Horizontal)
        sl_max.setRange(int(min_val * scale), int(max_val * scale))
        sl_max.setValue(int(default_max * scale))
        lbl_max = QLabel(f"{default_max:.2f}")
        lbl_max.setFixedWidth(65)
        row_max.addWidget(QLabel("Max:"))
        row_max.addWidget(sl_max)
        row_max.addWidget(lbl_max)
        layout.addLayout(row_max)
        
        def update_labels():
            lbl_min.setText(f"{sl_min.value() / scale:.2f}")
            lbl_max.setText(f"{sl_max.value() / scale:.2f}")
                
        sl_min.valueChanged.connect(update_labels)
        sl_max.valueChanged.connect(update_labels)
        
        return layout, sl_min, sl_max, lbl_min, lbl_max

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        top_bar = QHBoxLayout()
        self.btn_calib = QPushButton("Import Astigmatism Calibration")
        self.btn_calib.clicked.connect(self.load_calibration)
        top_bar.addWidget(self.btn_calib)
        
        self.lbl_calib_status = QLabel("Calibration: Not Loaded")
        top_bar.addWidget(self.lbl_calib_status)

        self.btn_dir = QPushButton("Import Trajectories in Folder")
        self.btn_dir.clicked.connect(self.load_directory)
        top_bar.addWidget(self.btn_dir)
        
        self.chk_recursive = QCheckBox("Include Subdirectories")
        self.chk_recursive.setChecked(True)
        top_bar.addWidget(self.chk_recursive)
        main_layout.addLayout(top_bar)

        middle_layout = QHBoxLayout()
        
        self.fig = plt.figure(figsize=(8, 6))
        self.ax_3d = self.fig.add_subplot(111, projection='3d')
        self.ax_3d.set_title("3D Trajectories")
        self.ax_3d.set_xlabel("X")
        self.ax_3d.set_ylabel("Y")
        self.ax_3d.set_zlabel("Z")
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        middle_layout.addLayout(plot_layout, stretch=3)

        filter_box = QGroupBox("Filter Trajectories")
        filter_layout = QVBoxLayout(filter_box)
        
        self.chk_req_diff = QCheckBox("Require Diffusivity Estimate")
        self.chk_req_int = QCheckBox("Require Intensity Data")
        filter_layout.addWidget(self.chk_req_diff)
        filter_layout.addWidget(self.chk_req_int)
        
        # FIXED: Increased max intensity slider to 1,000,000 to prevent skipping high-intensity spots.
        int_lay, self.sl_int_min, self.sl_int_max, _, _ = self._create_scaled_slider_group("Initial Intensity", 0, 1000000, 0, 1000000, 1)
        filter_layout.addLayout(int_lay)

        diff_lay, self.sl_diff_min, self.sl_diff_max, _, _ = self._create_scaled_slider_group("Diffusivity", -1, 100, -1, 100, 100)
        filter_layout.addLayout(diff_lay)
        
        filter_layout.addStretch()
        middle_layout.addWidget(filter_box, stretch=1)
        main_layout.addLayout(middle_layout)

        bottom_bar = QHBoxLayout()
        self.btn_export = QPushButton("Write 3D Trajectories")
        self.btn_export.clicked.connect(self.export_3d_trajectories)
        bottom_bar.addWidget(self.btn_export)
        
        main_layout.addLayout(bottom_bar)

        self.chk_req_diff.stateChanged.connect(self.update_plot)
        self.chk_req_int.stateChanged.connect(self.update_plot)
        self.sl_int_min.valueChanged.connect(self.update_plot)
        self.sl_int_max.valueChanged.connect(self.update_plot)
        self.sl_diff_min.valueChanged.connect(self.update_plot)
        self.sl_diff_max.valueChanged.connect(self.update_plot)

    def load_calibration(self):
        calib_path, _ = QFileDialog.getOpenFileName(
            self, "Select Astigmatism Calibration File", "", "Text Files (*.txt *.tsv);;All Files (*)"
        )
        if not calib_path: 
            return
        
        try:
            data = np.loadtxt(calib_path)
            z_vals = data[:, 0]
            ratios = data[:, 1]
            
            sort_idx = np.argsort(ratios)
            ratios_sorted = ratios[sort_idx]
            z_vals_sorted = z_vals[sort_idx]
            
            self.ratio_min = float(np.min(ratios_sorted))
            self.ratio_max = float(np.max(ratios_sorted))
            
            self.calibration_spline = CubicSpline(ratios_sorted, z_vals_sorted)
            self.lbl_calib_status.setText("Calibration: Loaded")
            show_popup(self, "Success", f"Astigmatism calibration loaded ({len(z_vals)} points).")
            
            if self.data_store:
                self.update_plot()
                
        except Exception as e:
            show_popup(self, "Error", f"Failed to load calibration:\n{str(e)}", critical=True)

    def load_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Root Results Directory")
        if not dir_path: 
            return
        
        self.directory = dir_path
        self.data_store.clear()
        
        search_pattern = os.path.join(self.directory, "**", "*_trajectories.tsv") if self.chk_recursive.isChecked() else os.path.join(self.directory, "*_trajectories.tsv")
        traj_files = glob.glob(search_pattern, recursive=self.chk_recursive.isChecked())
        
        if not traj_files:
            show_popup(self, "No Files Found", "No trajectory files found.", critical=True)
            return
            
        loaded_count = 0

        for fpath in traj_files:
            folder_path = os.path.dirname(fpath)
            fname = os.path.basename(fpath)
            base_name = fname.replace("_trajectories.tsv", "")
            
            diff_dict, int_dict = {}, {}
            
            diff_path = os.path.join(folder_path, f"{base_name}_diff_coeff_data.tsv")
            if os.path.exists(diff_path):
                try:
                    with open(diff_path, 'r') as f:
                        for row in csv.DictReader(f, delimiter='\t'):
                            t_key = row.get("trajectory") or row.get("track_id") or row.get("particle")
                            d_key = row.get("diffusion coefficient") or row.get("diff_coeff") or row.get("D")
                            if t_key and d_key: diff_dict[int(float(t_key))] = float(d_key)
                except Exception: pass
                
            int_path = os.path.join(folder_path, f"{base_name}_intensity_data.tsv")
            if os.path.exists(int_path):
                try:
                    with open(int_path, 'r') as f:
                        lines = [float(line.strip()) for line in f if line.strip()]
                        for idx, val in enumerate(lines, start=0): int_dict[idx] = val
                except Exception: pass
            
            try:
                with open(fpath, 'r') as f:
                    # Skip files that are likely pure calibration matrices
                    first_line = f.readline()
                    f.seek(0)
                    if all(t.replace('.', '', 1).replace('-', '', 1).isdigit() for t in first_line.strip().split('\t') if t):
                        continue

                    reader = csv.DictReader(f, delimiter='\t')
                    rows_added = 0
                    
                    for row in reader:
                        try:
                            t_id_str = row.get("trajectory") or row.get("track_id") or row.get("particle")
                            frame_str = row.get("frame") or row.get("slice")
                            x_str = row.get("x") or row.get("xc")
                            y_str = row.get("y") or row.get("yc")
                            
                            if None in (t_id_str, frame_str, x_str, y_str): continue
                            
                            wx_str = row.get("widthx") or row.get("w_x") or row.get("sigma_x")
                            wy_str = row.get("widthy") or row.get("w_y") or row.get("sigma_y")
                            
                            # FIXED: Forces absolute values, handles empty strings smoothly.
                            w_x = abs(float(wx_str)) if wx_str and wx_str.strip() else 1.0
                            w_y = abs(float(wy_str)) if wy_str and wy_str.strip() else 1.0
                            
                            t_id = int(float(t_id_str))
                            self.data_store.append({
                                "original_file": fname,
                                "original_id": t_id,
                                "unique_id": f"{fname}::{t_id}",
                                "frame": int(float(frame_str)),
                                "x": float(x_str),
                                "y": float(y_str),
                                "width_x": w_x,
                                "width_y": w_y,
                                "diff_coeff": diff_dict.get(t_id),
                                "intensity": int_dict.get(t_id) or row.get("spot_intensity") # Fallback to inline intensity if available
                            })
                            rows_added += 1
                        except ValueError:
                            continue
                            
                    if rows_added > 0: loaded_count += 1
                        
            except Exception: pass

        if loaded_count > 0:
            show_popup(self, "Data Loaded", f"Successfully loaded trajectory points from {loaded_count} file(s).")
        else:
            show_popup(self, "Load Error", "Could not parse trajectory coordinates.", critical=True)

        self.update_plot()

    def update_plot(self):
        self.ax_3d.clear()
        self.ax_3d.set_title("3D Trajectories")
        self.ax_3d.set_xlabel("X")
        self.ax_3d.set_ylabel("Y")
        self.ax_3d.set_zlabel("Z")
        self.plotted_tracks.clear()
        
        if not self.data_store or self.calibration_spline is None:
            self.canvas.draw_idle()
            return
            
        req_diff = self.chk_req_diff.isChecked()
        req_int = self.chk_req_int.isChecked()
        diff_min, diff_max = sorted([self.sl_diff_min.value() / 100.0, self.sl_diff_max.value() / 100.0])
        int_min, int_max = sorted([self.sl_int_min.value(), self.sl_int_max.value()])
        
        grouped_trajs = {}
        for pt in self.data_store:
            uid = pt["unique_id"]
            if uid not in grouped_trajs:
                grouped_trajs[uid] = []
            grouped_trajs[uid].append(pt)
            
        cmap = plt.get_cmap("jet")
            
        for uid, points in grouped_trajs.items():
            first_pt = points[0]
            diff = first_pt["diff_coeff"]
            # Safely handle intensity, casting to float if it exists
            intensity = float(first_pt["intensity"]) if first_pt.get("intensity") is not None else None
            
            if req_diff and diff is None: continue
            if req_int and intensity is None: continue
            if diff is not None and not (diff_min <= diff <= diff_max): continue
            if intensity is not None and not (int_min <= intensity <= int_max): continue
            
            points = sorted(points, key=lambda x: x["frame"])
            xs = [p["x"] for p in points]
            ys = [p["y"] for p in points]
            
            zs = []
            for p in points:
                ratio = p["width_x"] / max(1e-9, p["width_y"])
                clipped_ratio = np.clip(ratio, self.ratio_min, self.ratio_max)
                z_est = float(self.calibration_spline(clipped_ratio))
                zs.append(round(z_est, 2))
                p["z"] = round(z_est, 2)
                
            color = cmap(np.random.rand())
            self.ax_3d.plot(xs, ys, zs, color=color, alpha=0.8, linewidth=1.5)
            self.plotted_tracks.extend(points)
            
        self.canvas.draw_idle()

    def export_3d_trajectories(self):
        if not self.plotted_tracks:
            show_popup(self, "Export Failed", "No trajectories currently plotted to export.", critical=True)
            return
            
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save 3D Trajectories", os.path.join(self.directory, "3D_trajectories_output.tsv"), "TSV Files (*.tsv)"
        )
        if not save_path: 
            return
        
        fieldnames = ["original_file", "original_id", "unique_id", "frame", "x", "y", "z", "width_x", "width_y", "diff_coeff", "intensity"]
        
        try:
            with open(save_path, "w", newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t', extrasaction='ignore')
                writer.writeheader()
                for pt in self.plotted_tracks:
                    writer.writerow(pt)
            show_popup(self, "Export Complete", f"Successfully wrote 3D trajectories to:\n{save_path}")
        except Exception as e:
            show_popup(self, "Export Error", f"Failed to write file:\n{str(e)}", critical=True)
