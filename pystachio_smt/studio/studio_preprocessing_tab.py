# -*- coding: utf-8 -*-
import sys
import os
import glob
import re
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

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QGroupBox, QFormLayout,
    QLabel, QPushButton, QFileDialog, QLineEdit, QCheckBox, QComboBox,
    QSpinBox, QDoubleSpinBox, QTabWidget, QScrollArea, QProgressDialog,
    QMessageBox, QTextEdit, QDialog, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont

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
except ImportError as e:
    print(f"Warning: Could not import preprocess module ({e}). Preprocessing tab operating in standalone mode.")
    PREPROCESS_LOADED = False


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
            if 'num_channels' in cfg:
                details.append(f"Channels: {cfg['num_channels']}")
            if 'localisation_precision' in cfg:
                details.append(f"Loc Prec: {cfg['localisation_precision']} μm")
            if 'roi_file' in cfg:
                details.append(f"ROI: {os.path.basename(cfg['roi_file'])}")

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
        super().__init__(self.fig)


# =========================================================================
# NATIVE PYQT MODAL DIALOG FOR INTERACTIVE DRAG & ROTATE REGISTRATION
# =========================================================================

class InteractiveDragDialog(QDialog):
    """Native PyQt Dialog for interactive 2-channel drag & rotate registration."""
    def __init__(self, img, save_dir, tmats_path="transformation_matrix.npy", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Interactive Channel Registration (Drag & Rotate)")
        self.resize(1100, 820)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        self.img = img
        self.save_dir = save_dir
        self.tmats_path = tmats_path
        self.matrix = None

        h, w = img.shape[:2]
        if w >= h and w > 1:
            half_w = w // 2
            self.ch1 = img[:, :half_w].astype(float)
            self.ch2 = img[:, half_w:].astype(float)
        else:
            self.ch1 = img.astype(float)
            self.ch2 = img.astype(float)

        self.ch1_norm = (self.ch1 - self.ch1.min()) / (self.ch1.max() - self.ch1.min() + 1e-8)
        self.ch2_norm = (self.ch2 - self.ch2.min()) / (self.ch2.max() - self.ch2.min() + 1e-8)

        self.dx = 0.0
        self.dy = 0.0
        self.angle_deg = 0.0

        self.drag_start = None
        self.initial_dxdy = (0.0, 0.0)

        self.init_ui()
        self.update_display()

    def init_ui(self):
        layout = QVBoxLayout(self)

        lbl_info = QLabel(
            "<b>Interactive Drag & Rotate Alignment:</b><br>"
            "• <b>Left-Click & Drag:</b> Move channel | • <b>Mouse Scroll Wheel / Q,E Keys:</b> Rotate<br>"
            "• <b>Arrow Keys:</b> Fine Nudge (0.5 px) | • <b>R Key:</b> Reset | • <b>Enter / D:</b> Apply Registration"
        )
        lbl_info.setWordWrap(True)
        layout.addWidget(lbl_info)

        ctrl_box = QGroupBox("Transform & Visualization Controls")
        ctrl_layout = QHBoxLayout(ctrl_box)

        ctrl_layout.addWidget(QLabel("X Shift (dx):"))
        self.spin_dx = NoScrollDoubleSpinBox()
        self.spin_dx.setRange(-2000.0, 2000.0)
        self.spin_dx.setSingleStep(0.5)
        self.spin_dx.setSuffix(" px")
        self.spin_dx.valueChanged.connect(self.on_spin_changed)
        ctrl_layout.addWidget(self.spin_dx)

        ctrl_layout.addWidget(QLabel("Y Shift (dy):"))
        self.spin_dy = NoScrollDoubleSpinBox()
        self.spin_dy.setRange(-2000.0, 2000.0)
        self.spin_dy.setSingleStep(0.5)
        self.spin_dy.setSuffix(" px")
        self.spin_dy.valueChanged.connect(self.on_spin_changed)
        ctrl_layout.addWidget(self.spin_dy)

        ctrl_layout.addWidget(QLabel("Rotation (Angle):"))
        self.spin_angle = NoScrollDoubleSpinBox()
        self.spin_angle.setRange(-180.0, 180.0)
        self.spin_angle.setSingleStep(0.1)
        self.spin_angle.setSuffix("°")
        self.spin_angle.valueChanged.connect(self.on_spin_changed)
        ctrl_layout.addWidget(self.spin_angle)

        ctrl_layout.addWidget(QLabel("Overlay Mode:"))
        self.combo_mode = NoScrollComboBox()
        self.combo_mode.addItems(["Red/Green Overlay", "Grayscale Average", "Difference (Abs Diff)"])
        self.combo_mode.currentIndexChanged.connect(lambda: self.update_display())
        ctrl_layout.addWidget(self.combo_mode)

        btn_reset = QPushButton("Reset")
        btn_reset.clicked.connect(self.reset_transform)
        ctrl_layout.addWidget(btn_reset)

        ctrl_layout.addStretch()
        layout.addWidget(ctrl_box)

        self.canvas = MplCanvas(self, width=10, height=7)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)

        btn_bar = QHBoxLayout()
        self.lbl_status = QLabel("dx: 0.00 px | dy: 0.00 px | Angle: 0.00°")
        self.lbl_status.setStyleSheet("font-weight: bold; font-family: monospace;")
        btn_bar.addWidget(self.lbl_status)
        btn_bar.addStretch()

        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_bar.addWidget(btn_cancel)

        btn_apply = QPushButton("Apply & Save Registration")
        btn_apply.setStyleSheet("font-weight: bold; background-color: #28a745; color: white; padding: 6px 14px;")
        btn_apply.clicked.connect(self.apply_and_close)
        btn_bar.addWidget(btn_apply)

        layout.addLayout(btn_bar)

    def update_display(self):
        rad = np.deg2rad(self.angle_deg)
        center = (self.ch2_norm.shape[1] / 2.0, self.ch2_norm.shape[0] / 2.0)

        t_center = ski_transform.AffineTransform(translation=-np.array(center))
        t_rot = ski_transform.AffineTransform(rotation=rad)
        t_back = ski_transform.AffineTransform(translation=np.array(center) + np.array([self.dx, self.dy]))
        
        self.total_tform = t_center + t_rot + t_back
        warped = ski_transform.warp(self.ch2_norm, self.total_tform.inverse)

        mode = self.combo_mode.currentText()
        ax = self.canvas.axes
        ax.clear()

        if mode == "Red/Green Overlay":
            rgb = np.zeros((self.ch1_norm.shape[0], self.ch1_norm.shape[1], 3), dtype=np.float32)
            rgb[:, :, 0] = warped
            rgb[:, :, 1] = self.ch1_norm
            ax.imshow(rgb)
        elif mode == "Grayscale Average":
            avg_img = (self.ch1_norm + warped) / 2.0
            ax.imshow(avg_img, cmap='gray')
        else:
            diff = np.abs(self.ch1_norm - warped)
            ax.imshow(diff, cmap='inferno')

        ax.set_title(f"dx: {self.dx:.2f} px | dy: {self.dy:.2f} px | Angle: {self.angle_deg:.2f}°")
        ax.set_axis_off()
        self.canvas.draw_idle()

        self.spin_dx.blockSignals(True)
        self.spin_dy.blockSignals(True)
        self.spin_angle.blockSignals(True)
        self.spin_dx.setValue(self.dx)
        self.spin_dy.setValue(self.dy)
        self.spin_angle.setValue(self.angle_deg)
        self.spin_dx.blockSignals(False)
        self.spin_dy.blockSignals(False)
        self.spin_angle.blockSignals(False)

        self.lbl_status.setText(f"dx: {self.dx:.2f} px | dy: {self.dy:.2f} px | Angle: {self.angle_deg:.2f}°")

    def on_spin_changed(self):
        self.dx = self.spin_dx.value()
        self.dy = self.spin_dy.value()
        self.angle_deg = self.spin_angle.value()
        self.update_display()

    def reset_transform(self):
        self.dx = 0.0
        self.dy = 0.0
        self.angle_deg = 0.0
        self.update_display()

    def on_press(self, event):
        if event.inaxes != self.canvas.axes or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        self.drag_start = (event.xdata, event.ydata)
        self.initial_dxdy = (self.dx, self.dy)

    def on_motion(self, event):
        if self.drag_start is None or event.inaxes != self.canvas.axes:
            return
        if event.xdata is None or event.ydata is None:
            return
        x0, y0 = self.drag_start
        init_dx, init_dy = self.initial_dxdy
        self.dx = init_dx + (event.xdata - x0)
        self.dy = init_dy + (event.ydata - y0)
        self.update_display()

    def on_release(self, event):
        self.drag_start = None

    def on_scroll(self, event):
        step = 0.5 if event.button == 'up' else -0.5
        self.angle_deg += step
        self.update_display()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Q:
            self.angle_deg -= 0.5
            self.update_display()
        elif key == Qt.Key.Key_E:
            self.angle_deg += 0.5
            self.update_display()
        elif key == Qt.Key.Key_Left:
            self.dx -= 0.5
            self.update_display()
        elif key == Qt.Key.Key_Right:
            self.dx += 0.5
            self.update_display()
        elif key == Qt.Key.Key_Up:
            self.dy -= 0.5
            self.update_display()
        elif key == Qt.Key.Key_Down:
            self.dy += 0.5
            self.update_display()
        elif key == Qt.Key.Key_R:
            self.reset_transform()
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_D):
            self.apply_and_close()
        else:
            super().keyPressEvent(event)

    def generate_and_save_overlay(self, tform):
        warped_ch2 = ski_transform.warp(self.ch2_norm, tform.inverse)
        
        def calc_ncc(a, b):
            a_c = a - np.mean(a)
            b_c = b - np.mean(b)
            num = np.sum(a_c * b_c)
            den = np.sqrt(np.sum(a_c**2) * np.sum(b_c**2))
            return 0.0 if den == 0 else float(num / den)

        ncc_before = calc_ncc(self.ch1_norm, self.ch2_norm)
        ncc_after = calc_ncc(self.ch1_norm, warped_ch2)

        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        axs[0].imshow((self.ch1_norm + self.ch2_norm) / 2.0, cmap='gray')
        axs[0].set_title(f'Original Images (Unaligned Overlay) - NCC = {ncc_before:.4f}')
        axs[0].set_xticks([]); axs[0].set_yticks([])

        axs[1].imshow(warped_ch2, cmap='gray')
        axs[1].set_title('Registered Moving Channel (Channel 2)')
        axs[1].set_xticks([]); axs[1].set_yticks([])

        axs[2].imshow((self.ch1_norm + warped_ch2) / 2.0, cmap='gray')
        axs[2].set_title(f'Manual Drag Alignment Overlay - NCC = {ncc_after:.4f}')
        axs[2].set_xticks([]); axs[2].set_yticks([])

        plt.tight_layout()
        os.makedirs(self.save_dir, exist_ok=True)
        plt.savefig(os.path.join(self.save_dir, "Registered_overlay_manual.png"))
        plt.savefig(os.path.join(self.save_dir, "Registered_overlay_summed_beads.png"))
        plt.close(fig)

        df_reg = pd.DataFrame([{
            'Best_Folder': 'Manual Drag Alignment',
            'Best_NCC': ncc_after,
            'Matrix_00': self.matrix[0, 0], 'Matrix_01': self.matrix[0, 1], 'Matrix_02': self.matrix[0, 2],
            'Matrix_10': self.matrix[1, 0], 'Matrix_11': self.matrix[1, 1], 'Matrix_12': self.matrix[1, 2],
            'Matrix_20': self.matrix[2, 0], 'Matrix_21': self.matrix[2, 1], 'Matrix_22': self.matrix[2, 2]
        }])
        df_reg.to_csv(os.path.join(self.save_dir, "registration.csv"), index=False)

    def apply_and_close(self):
        rad = np.deg2rad(self.angle_deg)
        center = (self.ch2_norm.shape[1] / 2.0, self.ch2_norm.shape[0] / 2.0)
        t_center = ski_transform.AffineTransform(translation=-np.array(center))
        t_rot = ski_transform.AffineTransform(rotation=rad)
        t_back = ski_transform.AffineTransform(translation=np.array(center) + np.array([self.dx, self.dy]))
        final_tform = t_center + t_rot + t_back

        self.matrix = final_tform.params.astype(np.float64)

        os.makedirs(self.save_dir, exist_ok=True)
        if os.path.isabs(self.tmats_path):
            out_path = self.tmats_path
        else:
            out_path = os.path.normpath(os.path.join(self.save_dir, self.tmats_path))

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        np.save(out_path, self.matrix)

        mat_path = os.path.splitext(out_path)[0] + ".mat"
        sio.savemat(mat_path, {
            'T_matrix': self.matrix.T,
            'tform_type': 'affine',
            'generated_by': 'PySTACHIO interactive drag registration'
        })

        self.generate_and_save_overlay(final_tform)
        self.accept()


# =========================================================================
# NATIVE PYQT MODAL DIALOG FOR POINT-PICKING MANUAL REGISTRATION
# =========================================================================

class PointPickingDialog(QDialog):
    """Native PyQt Dialog for manual point-picking pair registration."""
    def __init__(self, img, save_dir, tmats_path="transformation_matrix.npy", num_pairs=10, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Interactive Point-Picking Registration ({num_pairs} pairs target)")
        self.resize(1200, 750)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        self.img = img
        self.save_dir = save_dir
        self.tmats_path = tmats_path
        self.num_pairs = num_pairs
        self.matrix = None

        h, w = img.shape[:2]
        if w >= h and w > 1:
            self.half_w = w // 2
            self.ch1 = img[:, :self.half_w].astype(float)
            self.ch2 = img[:, self.half_w:].astype(float)
        else:
            self.half_w = w
            self.ch1 = img.astype(float)
            self.ch2 = img.astype(float)

        self.ch1_norm = (self.ch1 - self.ch1.min()) / (self.ch1.max() - self.ch1.min() + 1e-8)
        self.ch2_norm = (self.ch2 - self.ch2.min()) / (self.ch2.max() - self.ch2.min() + 1e-8)

        self.display_img = np.hstack((self.ch1_norm, self.ch2_norm))

        self.pts_ch1 = []
        self.pts_ch2 = []
        self.raw_clicks = []
        self.init_limits = True

        self.init_ui()
        self.update_display()

    def init_ui(self):
        layout = QVBoxLayout(self)

        lbl_info = QLabel(
            "<b>Point-Picking Registration Instructions:</b><br>"
            "Click corresponding bead pairs on the images:<br>"
            "• <b>Odd Click (1st, 3rd, 5th...):</b> Click bead on Left Channel (Channel 1)<br>"
            "• <b>Even Click (2nd, 4th, 6th...):</b> Click matching bead on Right Channel (Channel 2)<br>"
            "• <b>Mouse Scroll Wheel:</b> Zoom in/out | <b>Right-Click / Backspace / Z:</b> Undo last click<br>"
            "• <b>D Key / Enter / Green Button:</b> Finish & Save Registration (requires ≥ 3 pairs)"
        )
        lbl_info.setWordWrap(True)
        layout.addWidget(lbl_info)

        ctrl_bar = QHBoxLayout()
        self.lbl_status = QLabel(f"Picked 0 / {self.num_pairs} pairs (Need at least 3 pairs for Affine Matrix)")
        self.lbl_status.setStyleSheet("font-weight: bold; padding: 4px;")
        ctrl_bar.addWidget(self.lbl_status)
        ctrl_bar.addStretch()

        btn_reset_zoom = QPushButton("Reset Zoom")
        btn_reset_zoom.clicked.connect(self.reset_zoom)
        ctrl_bar.addWidget(btn_reset_zoom)

        btn_undo = QPushButton("Undo Last Point")
        btn_undo.clicked.connect(self.undo_point)
        ctrl_bar.addWidget(btn_undo)

        btn_clear = QPushButton("Clear All")
        btn_clear.clicked.connect(self.clear_all)
        ctrl_bar.addWidget(btn_clear)

        layout.addLayout(ctrl_bar)

        self.canvas = MplCanvas(self, width=12, height=6)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)

        btn_bar = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_bar.addStretch()
        btn_bar.addWidget(btn_cancel)

        self.btn_apply = QPushButton("Compute & Save Registration")
        self.btn_apply.setEnabled(False)
        self.btn_apply.setStyleSheet("font-weight: bold; background-color: #28a745; color: white; padding: 6px 14px;")
        self.btn_apply.clicked.connect(self.apply_and_close)
        btn_bar.addWidget(self.btn_apply)

        layout.addLayout(btn_bar)

    def reset_zoom(self):
        self.init_limits = True
        self.update_display()

    def on_scroll(self, event):
        if event.inaxes != self.canvas.axes:
            return
        ax = self.canvas.axes
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()

        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None:
            return

        base_scale = 1.25
        scale_factor = (1 / base_scale) if event.button == 'up' else (base_scale if event.button == 'down' else 1.0)

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        rel_x = (xdata - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0])
        rel_y = (ydata - cur_ylim[0]) / (cur_ylim[1] - cur_ylim[0])

        ax.set_xlim([xdata - new_width * rel_x, xdata + new_width * (1 - rel_x)])
        ax.set_ylim([ydata - new_height * rel_y, ydata + new_height * (1 - rel_y)])
        self.canvas.draw_idle()

    def update_display(self):
        ax = self.canvas.axes
        xlim = ax.get_xlim() if not self.init_limits else None
        ylim = ax.get_ylim() if not self.init_limits else None

        ax.clear()
        ax.imshow(self.display_img, cmap='gray')
        ax.set_axis_off()

        for idx, (p1, p2) in enumerate(zip(self.pts_ch1, self.pts_ch2)):
            x1, y1 = p1
            x2_disp = p2[0] + self.half_w
            y2 = p2[1]

            ax.plot(x1, y1, 'g+', markersize=6, markeredgewidth=1.5)
            ax.text(x1 + 2, y1 - 2, str(idx + 1), color='lime', fontweight='bold', fontsize=7)

            ax.plot(x2_disp, y2, 'r+', markersize=6, markeredgewidth=1.5)
            ax.text(x2_disp + 2, y2 - 2, str(idx + 1), color='red', fontweight='bold', fontsize=7)

        if len(self.raw_clicks) % 2 != 0:
            last_x, last_y = self.raw_clicks[-1]
            ax.plot(last_x, last_y, 'y+', markersize=7, markeredgewidth=1.5)

        pair_count = len(self.pts_ch1)
        if pair_count >= self.num_pairs:
            self.lbl_status.setText(f"Picked {pair_count} / {self.num_pairs} pairs — Target reached! Press 'D' or click green button to finish.")
        else:
            self.lbl_status.setText(f"Picked {pair_count} / {self.num_pairs} pairs (Need at least 3 pairs for Affine Matrix)")

        self.btn_apply.setEnabled(pair_count >= 3)

        if not self.init_limits and xlim is not None and ylim is not None:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        else:
            self.init_limits = False

        self.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.canvas.axes:
            return

        if event.button == 3:
            self.undo_point()
            return

        if event.button != 1 or event.xdata is None or event.ydata is None:
            return

        x, y = event.xdata, event.ydata
        self.raw_clicks.append((x, y))

        if len(self.raw_clicks) % 2 == 0:
            c1 = self.raw_clicks[-2]
            c2 = self.raw_clicks[-1]

            if c1[0] > c2[0]:
                c1, c2 = c2, c1

            p1 = [c1[0], c1[1]]
            p2 = [c2[0] - self.half_w if c2[0] >= self.half_w else c2[0], c2[1]]

            self.pts_ch1.append(p1)
            self.pts_ch2.append(p2)

        self.update_display()

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_D):
            if len(self.pts_ch1) >= 3:
                self.apply_and_close()
        elif key in (Qt.Key.Key_Z, Qt.Key.Key_Backspace):
            self.undo_point()
        elif key == Qt.Key.Key_R:
            self.clear_all()
        else:
            super().keyPressEvent(event)

    def undo_point(self):
        if self.raw_clicks:
            if len(self.raw_clicks) % 2 == 0 and self.pts_ch1:
                self.pts_ch1.pop()
                self.pts_ch2.pop()
            self.raw_clicks.pop()
            self.update_display()

    def clear_all(self):
        self.pts_ch1.clear()
        self.pts_ch2.clear()
        self.raw_clicks.clear()
        self.update_display()

    def generate_and_save_overlay(self, tform):
        warped_ch2 = ski_transform.warp(self.ch2_norm, tform.inverse)
        
        def calc_ncc(a, b):
            a_c = a - np.mean(a)
            b_c = b - np.mean(b)
            num = np.sum(a_c * b_c)
            den = np.sqrt(np.sum(a_c**2) * np.sum(b_c**2))
            return 0.0 if den == 0 else float(num / den)

        ncc_before = calc_ncc(self.ch1_norm, self.ch2_norm)
        ncc_after = calc_ncc(self.ch1_norm, warped_ch2)

        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        axs[0].imshow((self.ch1_norm + self.ch2_norm) / 2.0, cmap='gray')
        axs[0].set_title(f'Original Images (Unaligned Overlay) - NCC = {ncc_before:.4f}')
        axs[0].set_xticks([]); axs[0].set_yticks([])

        axs[1].imshow(warped_ch2, cmap='gray')
        axs[1].set_title('Registered Moving Channel (Channel 2)')
        axs[1].set_xticks([]); axs[1].set_yticks([])

        axs[2].imshow((self.ch1_norm + warped_ch2) / 2.0, cmap='gray')
        axs[2].set_title(f'Manual Registration Overlay - NCC = {ncc_after:.4f}')
        axs[2].set_xticks([]); axs[2].set_yticks([])

        plt.tight_layout()
        os.makedirs(self.save_dir, exist_ok=True)
        plt.savefig(os.path.join(self.save_dir, "Registered_overlay_manual.png"))
        plt.savefig(os.path.join(self.save_dir, "Registered_overlay_summed_beads.png"))
        plt.close(fig)

        df_reg = pd.DataFrame([{
            'Best_Folder': 'Manual Registration',
            'Best_NCC': ncc_after,
            'Matrix_00': self.matrix[0, 0], 'Matrix_01': self.matrix[0, 1], 'Matrix_02': self.matrix[0, 2],
            'Matrix_10': self.matrix[1, 0], 'Matrix_11': self.matrix[1, 1], 'Matrix_12': self.matrix[1, 2],
            'Matrix_20': self.matrix[2, 0], 'Matrix_21': self.matrix[2, 1], 'Matrix_22': self.matrix[2, 2]
        }])
        df_reg.to_csv(os.path.join(self.save_dir, "registration.csv"), index=False)

    def apply_and_close(self):
        if len(self.pts_ch1) < 3:
            QMessageBox.warning(self, "Insufficient Points", "At least 3 pairs of points are required for an Affine transform.")
            return

        pts1 = np.array(self.pts_ch1)
        pts2 = np.array(self.pts_ch2)

        tform = ski_transform.estimate_transform('affine', src=pts2, dst=pts1)
        self.matrix = tform.params.astype(np.float64)

        os.makedirs(self.save_dir, exist_ok=True)
        if os.path.isabs(self.tmats_path):
            out_path = self.tmats_path
        else:
            out_path = os.path.normpath(os.path.join(self.save_dir, self.tmats_path))

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        np.save(out_path, self.matrix)

        mat_path = os.path.splitext(out_path)[0] + ".mat"
        sio.savemat(mat_path, {
            'T_matrix': self.matrix.T,
            'tform_type': 'affine',
            'generated_by': 'Python scikit-image manual point-picking registration'
        })

        self.generate_and_save_overlay(tform)
        self.accept()


# =========================================================================
# PARAMETERS CONTAINER FOR PREPROCESSING PIPELINE
# =========================================================================

class PreprocessParams:
    """Dataclass-style container passing GUI configuration to AnalysisPipeline."""
    def __init__(self, **kwargs):
        self.name = kwargs.get('name', 'preprocessing_results')
        self.video_path = kwargs.get('video_path', '')
        self.bf_path = kwargs.get('bf_path', '')
        self.bead_path = kwargs.get('bead_path', '')
        self.roi_file = kwargs.get('roi_file', '')
        self.roi_channel = kwargs.get('roi_channel', 'L')
        self.save_dir = kwargs.get('save_dir', 'results')
        self.tmats_path = kwargs.get('tmats_path', 'transformation_matrix.npy')
        
        self.pixel_size = kwargs.get('pixel_size', 0.120)
        self.num_channels = kwargs.get('num_channels', 1)
        self.use_channel = kwargs.get('use_channel', 'L')
        self.channel = kwargs.get('use_channel', 'L')
        self.start_channel = kwargs.get('start_channel', 'L')
        self.ALEX = kwargs.get('ALEX', False)
        self.num_frames = kwargs.get('num_frames', 0)
        self.frame_avg = kwargs.get('frame_avg', 5)
        
        self.mask_type = kwargs.get('mask_type', 'AI')
        self.model_type = kwargs.get('model_type', 'unet')
        self.model = kwargs.get('model', '')
        self.model_params = kwargs.get('model_params', 0.5)
        self.threshold_param = kwargs.get('threshold_param', 0.5)
        self.inv_bf = kwargs.get('inv_bf', False)
        self.use_otsu = kwargs.get('use_otsu', 'False')
        self.mask_prefix = kwargs.get('mask_prefix', 'mask')
        
        self.cell_fitting = kwargs.get('cell_fitting', True)
        self.area_filter = kwargs.get('area_filter', 100)
        self.overwrite = kwargs.get('overwrite', True)
        self.manual_registration = kwargs.get('manual_registration', False)
        self.manual_pairs = kwargs.get('manual_pairs', 10)


# =========================================================================
# BACKGROUND WORKER THREAD FOR MODULAR PREPROCESSING TASKS
# =========================================================================

class PreprocessWorkerThread(QThread):
    """Executes full pipeline or individual preprocessing tasks in a background thread."""
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, params, task_type="full", parent=None):
        super().__init__(parent)
        self.params = params
        self.task_type = task_type

    def run(self):
        try:
            if not PREPROCESS_LOADED:
                raise ImportError("The `preprocess` module could not be loaded.")

            os.makedirs(self.params.save_dir, exist_ok=True)

            if self.task_type == "full":
                self.progress_signal.emit("Running Full Preprocessing Pipeline...")
                pipeline = AnalysisPipeline(self.params)
                pipeline.run()

            elif self.task_type == "bead":
                self.progress_signal.emit("Running Auto Bead Alignment Registration...")
                registrar = BeadRegistrar()
                tmat = registrar.calculate_transformation(
                    self.params.bead_path, self.params.save_dir, self.params.use_channel
                )
                if tmat is None:
                    raise RuntimeError("Bead registration failed to produce a matrix.")

            elif self.task_type == "video":
                self.progress_signal.emit("Extracting video channels and creating frame averages...")
                pipeline = AnalysisPipeline(self.params)
                
                L_chan, R_chan, num_frames, raw_shape = VideoProcessor.process_video(
                    self.params.video_path, 0, self.params.num_frames, self.params.roi_channel,
                    self.params.roi_file, self.params.save_dir, int(self.params.num_channels),
                    self.params.use_channel, self.params.ALEX, getattr(self.params, 'start_channel', 'L')
                )
                if L_chan is not None:
                    L_avg = np.mean(L_chan[:int(self.params.frame_avg)], axis=0).astype(np.uint16)
                    tifffile.imwrite(f"{self.params.save_dir}/L_avg.tif", L_avg, imagej=True)
                    tifffile.imwrite(f"{self.params.save_dir}/L_channel.tif", L_chan, imagej=True)
                    pipeline.save_image_with_scalebar(L_avg, f"{self.params.save_dir}/L_avg.png")
                if R_chan is not None:
                    R_avg = np.mean(R_chan[:int(self.params.frame_avg)], axis=0).astype(np.uint16)
                    tifffile.imwrite(f"{self.params.save_dir}/R_avg.tif", R_avg, imagej=True)
                    tifffile.imwrite(f"{self.params.save_dir}/R_channel.tif", R_chan, imagej=True)
                    pipeline.save_image_with_scalebar(R_avg, f"{self.params.save_dir}/R_avg.png")

            elif self.task_type == "brightfield":
                self.progress_signal.emit("Processing and aligning Brightfield image...")
                pipeline = AnalysisPipeline(self.params)
                
                if os.path.isabs(self.params.tmats_path):
                    tmat_file = self.params.tmats_path
                else:
                    tmat_file = os.path.join(self.params.save_dir, self.params.tmats_path)

                if not os.path.exists(tmat_file):
                    tmat_file = os.path.join(self.params.save_dir, "transformation_matrix.npy")

                tmats = np.load(tmat_file) if os.path.exists(tmat_file) else None
                
                l_avg_path = os.path.join(self.params.save_dir, "L_avg.tif")
                if os.path.exists(l_avg_path):
                    raw_img = tifffile.imread(l_avg_path)
                    raw_shape = (1, raw_img.shape[0], raw_img.shape[1] * 2)
                else:
                    raw_shape = (1, 512, 1024)

                try:
                    from pystackreg import StackReg
                    sr = StackReg(StackReg.AFFINE)
                except ImportError:
                    sr = None

                pipeline.process_brightfield(raw_shape, sr, tmats)

            elif self.task_type in ["segmentation", "fitting"]:
                self.progress_signal.emit(f"Executing step: {self.task_type.capitalize()}...")
                pipeline = AnalysisPipeline(self.params)
                pipeline.run()

            self.finished_signal.emit(self.params.save_dir)
        except Exception as e:
            self.error_signal.emit(f"{str(e)}\n\n{traceback.format_exc()}")


# =========================================================================
# PREPROCESSING TAB MAIN WIDGET
# =========================================================================

class PreprocessingTab(QWidget):
    def __init__(self, parent=None, root_dir="."):
        super().__init__(parent)
        self.parent = parent
        self.root_dir = os.path.abspath(root_dir) if root_dir else os.getcwd()
        self.worker_thread = None
        self.progress_dialog = None
        self.config_tmats_path = "transformation_matrix.npy"
        self.bead_view_map = {}

        self.init_ui()
        if self.root_dir and os.path.exists(self.root_dir):
            self.scan_and_load_config()

    def init_ui(self):
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

        # 1. Video & File Paths Setup
        files_group = QGroupBox("File & Directory Setup")
        files_layout = QFormLayout()
        files_layout.setVerticalSpacing(6)

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
        btn_browse_save = QPushButton("Browse...")
        btn_browse_save.clicked.connect(self.browse_save)
        lay_save = QHBoxLayout()
        lay_save.addWidget(self.txt_save_dir)
        lay_save.addWidget(btn_browse_save)

        btn_run_bf = QPushButton("Process Brightfield Image")
        btn_run_bf.clicked.connect(lambda: self.run_task("brightfield"))

        files_layout.addRow("Fluorescence Video:", lay_video)
        files_layout.addRow("Brightfield Image:", lay_bf)
        files_layout.addRow("ROI File (.roi):", lay_roi)
        files_layout.addRow("Output Directory:", lay_save)
        files_layout.addRow(btn_run_bf)
        files_group.setLayout(files_layout)
        sidebar_layout.addWidget(files_group)

        # 2. Bead Alignment Parameters Group
        bead_group = QGroupBox("Bead Alignment Parameters")
        bead_layout = QFormLayout()
        bead_layout.setVerticalSpacing(6)

        self.txt_bead_path = QLineEdit()
        self.txt_bead_path.setPlaceholderText("Bead folder prefix (e.g. /path/to/beads)")
        btn_browse_beads = QPushButton("Browse...")
        btn_browse_beads.clicked.connect(self.browse_beads)
        lay_beads = QHBoxLayout()
        lay_beads.addWidget(self.txt_bead_path)
        lay_beads.addWidget(btn_browse_beads)

        self.combo_bead_mode = NoScrollComboBox()
        self.combo_bead_mode.addItem("Auto (StackReg Rigid-Body)", userData="auto")
        self.combo_bead_mode.addItem("Manual Registration", userData="manual")

        self.spin_num_pairs = NoScrollSpinBox()
        self.spin_num_pairs.setRange(0, 100)
        self.spin_num_pairs.setValue(10)
        self.spin_num_pairs.setToolTip("0 = Interactive Drag & Rotate mode | >0 = Point-Picking Mode (Number of pairs)")

        btn_run_beads = QPushButton("Run Bead Registration")
        btn_run_beads.setStyleSheet("font-weight: bold; background-color: #2e6da4; color: white; padding: 6px;")
        btn_run_beads.clicked.connect(self.run_bead_alignment_action)

        bead_layout.addRow("Bead Alignment Prefix:", lay_beads)
        bead_layout.addRow("Alignment Mode:", self.combo_bead_mode)
        bead_layout.addRow("Registration Pairs (num_pairs):", self.spin_num_pairs)
        bead_layout.addRow(btn_run_beads)
        bead_group.setLayout(bead_layout)
        sidebar_layout.addWidget(bead_group)

        # 3. Acquisition & Channel Settings
        acq_group = QGroupBox("Acquisition & Channel Settings")
        acq_layout = QFormLayout()
        acq_layout.setVerticalSpacing(6)

        self.spin_pixel_size = NoScrollDoubleSpinBox()
        self.spin_pixel_size.setDecimals(4)
        self.spin_pixel_size.setRange(0.0001, 10.0)
        self.spin_pixel_size.setValue(0.120)
        self.spin_pixel_size.setSuffix(" μm/px")

        self.combo_num_channels = NoScrollComboBox()
        self.combo_num_channels.addItems(["1", "2"])
        self.combo_num_channels.setCurrentIndex(0)

        self.combo_use_channel = NoScrollComboBox()
        self.combo_use_channel.addItems(["L", "R"])

        self.combo_start_channel = NoScrollComboBox()
        self.combo_start_channel.addItems(["L", "R"])

        self.combo_roi_channel = NoScrollComboBox()
        self.combo_roi_channel.addItems(["L", "R"])

        self.chk_alex = QCheckBox("ALEX Excitation Mode")
        self.chk_alex.setChecked(False)

        self.spin_num_frames = NoScrollSpinBox()
        self.spin_num_frames.setRange(0, 100000)
        self.spin_num_frames.setValue(0)
        self.spin_num_frames.setToolTip("0 = Process all frames")

        self.spin_frame_avg = NoScrollSpinBox()
        self.spin_frame_avg.setRange(1, 500)
        self.spin_frame_avg.setValue(5)

        btn_run_video = QPushButton("Extract & Average Video Stack")
        btn_run_video.clicked.connect(lambda: self.run_task("video"))

        acq_layout.addRow("Pixel Size:", self.spin_pixel_size)
        acq_layout.addRow("Number of Channels:", self.combo_num_channels)
        acq_layout.addRow("Active Target Channel:", self.combo_use_channel)
        acq_layout.addRow("ALEX Start Channel:", self.combo_start_channel)
        acq_layout.addRow("ROI Calibration Channel:", self.combo_roi_channel)
        acq_layout.addRow(self.chk_alex)
        acq_layout.addRow("Max Frames to Process:", self.spin_num_frames)
        acq_layout.addRow("Frames to Average:", self.spin_frame_avg)
        acq_layout.addRow(btn_run_video)
        acq_group.setLayout(acq_layout)
        sidebar_layout.addWidget(acq_group)

        # 4. Masking & Model Configuration
        mask_group = QGroupBox("Segmentation & AI Models")
        mask_layout = QFormLayout()
        mask_layout.setVerticalSpacing(6)

        self.combo_mask_type = NoScrollComboBox()
        self.combo_mask_type.addItems(["AI", "BF", "FL_AI", "THRESHOLD", "WHOLE", "MANUAL", "file"])
        self.combo_mask_type.currentIndexChanged.connect(self.on_mask_type_changed)

        self.combo_model_type = NoScrollComboBox()
        self.combo_model_type.addItems(["unet", "keras", "pytorch", "cellpose", "omnitorch"])

        self.txt_model_path = QLineEdit()
        btn_browse_model = QPushButton("Browse...")
        btn_browse_model.clicked.connect(self.browse_model)
        lay_model = QHBoxLayout()
        lay_model.addWidget(self.txt_model_path)
        lay_model.addWidget(btn_browse_model)

        self.spin_model_param = NoScrollDoubleSpinBox()
        self.spin_model_param.setRange(0.0, 1.0)
        self.spin_model_param.setSingleStep(0.05)
        self.spin_model_param.setValue(0.5)

        self.chk_inv_bf = QCheckBox("Invert Brightfield Contrast")
        self.chk_inv_bf.setChecked(False)

        self.combo_use_otsu = NoScrollComboBox()
        self.combo_use_otsu.addItems(["False", "True", "Multi"])

        btn_run_seg = QPushButton("Run Cell Segmentation / Masking")
        btn_run_seg.clicked.connect(lambda: self.run_task("segmentation"))

        mask_layout.addRow("Masking Type:", self.combo_mask_type)
        mask_layout.addRow("Model Architecture:", self.combo_model_type)
        mask_layout.addRow("Model File / URL:", lay_model)
        mask_layout.addRow("Model Threshold / Param:", self.spin_model_param)
        mask_layout.addRow(self.chk_inv_bf)
        mask_layout.addRow("Otsu Threshold Mode:", self.combo_use_otsu)
        mask_layout.addRow(btn_run_seg)
        mask_group.setLayout(mask_layout)
        sidebar_layout.addWidget(mask_group)

        # 5. Cell Geometry & Fitting
        fit_group = QGroupBox("Cell Shape Fitting & Filtering")
        fit_layout = QFormLayout()
        fit_layout.setVerticalSpacing(6)

        self.chk_cell_fitting = QCheckBox("Fit Mathematical Capsule Shapes")
        self.chk_cell_fitting.setChecked(True)

        self.spin_area_filter = NoScrollSpinBox()
        self.spin_area_filter.setRange(0, 100000)
        self.spin_area_filter.setValue(100)
        self.spin_area_filter.setSuffix(" px")

        self.chk_overwrite = QCheckBox("Overwrite Existing Results")
        self.chk_overwrite.setChecked(True)

        btn_run_fit = QPushButton("Fit Cell Shapes & Extract Features")
        btn_run_fit.clicked.connect(lambda: self.run_task("fitting"))

        fit_layout.addRow(self.chk_cell_fitting)
        fit_layout.addRow("Min Area Filter:", self.spin_area_filter)
        fit_layout.addRow(self.chk_overwrite)
        fit_layout.addRow(btn_run_fit)
        fit_group.setLayout(fit_layout)
        sidebar_layout.addWidget(fit_group)

        # 6. Full Pipeline Execution Box
        exec_group = QGroupBox("Pipeline Execution")
        exec_layout = QVBoxLayout()

        self.btn_run_pipeline = QPushButton("Run Full Preprocessing Pipeline")
        self.btn_run_pipeline.setStyleSheet("font-weight: bold; background-color: #008080; color: white; padding: 8px;")
        self.btn_run_pipeline.clicked.connect(lambda: self.run_task("full"))

        exec_layout.addWidget(self.btn_run_pipeline)
        exec_group.setLayout(exec_layout)
        sidebar_layout.addWidget(exec_group)

        for widget in sidebar.findChildren((NoScrollSpinBox, NoScrollDoubleSpinBox, NoScrollComboBox)):
            widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.txt_status_log = QTextEdit()
        self.txt_status_log.setReadOnly(True)
        self.txt_status_log.setPlaceholderText("Pipeline execution status and debug messages will appear here...")
        self.txt_status_log.setMaximumHeight(150)
        sidebar_layout.addWidget(self.txt_status_log)

        sidebar_layout.addStretch()

        scroll_area.setWidget(sidebar)
        splitter.addWidget(scroll_area)

        # ================= RIGHT PANEL =================
        self.tabs = QTabWidget()
        splitter.addWidget(self.tabs)

        # TAB 1: Bead Alignment & Manual Registration (SCROLLABLE)
        scroll_bead_tab = QScrollArea()
        scroll_bead_tab.setWidgetResizable(True)

        self.tab_beads = QWidget()
        layout_bead_tab = QVBoxLayout(self.tab_beads)
        layout_bead_tab.setContentsMargins(10, 10, 10, 10)

        matrix_group = QGroupBox("Loaded Registration Matrix & Metrics")
        matrix_layout = QHBoxLayout(matrix_group)

        self.lbl_matrix_display = QLabel("No transformation matrix loaded.")
        self.lbl_matrix_display.setStyleSheet("font-family: monospace; font-size: 11px; background-color: #f8f9fa; border: 1px solid #ddd; padding: 6px;")
        self.lbl_matrix_display.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.lbl_ncc_display = QLabel("NCC Performance: N/A")
        self.lbl_ncc_display.setStyleSheet("font-size: 11px; background-color: #f8f9fa; border: 1px solid #ddd; padding: 6px;")

        matrix_layout.addWidget(self.lbl_matrix_display, stretch=2)
        matrix_layout.addWidget(self.lbl_ncc_display, stretch=1)
        layout_bead_tab.addWidget(matrix_group)

        # Selector for Bead Overlay / Visualization Image
        view_bar = QHBoxLayout()
        view_bar.addWidget(QLabel("<b>Select Bead Image / Chart to View:</b>"))
        self.combo_bead_view = NoScrollComboBox()
        self.combo_bead_view.currentIndexChanged.connect(self.on_bead_view_changed)
        view_bar.addWidget(self.combo_bead_view, stretch=1)
        layout_bead_tab.addLayout(view_bar)

        self.canvas_beads = MplCanvas(self, width=8, height=6)
        self.canvas_beads.setMinimumHeight(450)
        self.toolbar_beads = NavigationToolbar(self.canvas_beads, self)
        layout_bead_tab.addWidget(self.toolbar_beads)
        layout_bead_tab.addWidget(self.canvas_beads)

        # Detailed NCC Statistics Table for All Bead Images
        table_group = QGroupBox("NCC Stats for All Processed Bead Images")
        table_layout = QVBoxLayout(table_group)

        self.table_ncc_stats = QTableWidget()
        self.table_ncc_stats.setColumnCount(4)
        self.table_ncc_stats.setHorizontalHeaderLabels(["Sample / Folder", "NCC Before", "NCC After", "Improvement (Δ)"])
        self.table_ncc_stats.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_ncc_stats.setAlternatingRowColors(True)
        self.table_ncc_stats.setMinimumHeight(200)
        table_layout.addWidget(self.table_ncc_stats)

        layout_bead_tab.addWidget(table_group)

        scroll_bead_tab.setWidget(self.tab_beads)
        self.tabs.addTab(scroll_bead_tab, "Bead Registration")

        # TAB 2: Channel Preview
        self.tab_channels = QWidget()
        self.canvas_channels = MplCanvas(self)
        self.toolbar_channels = NavigationToolbar(self.canvas_channels, self)
        layout_ch = QVBoxLayout(self.tab_channels)
        layout_ch.addWidget(self.toolbar_channels)
        layout_ch.addWidget(self.canvas_channels)
        self.tabs.addTab(self.tab_channels, "Fluorescence Previews")

        # TAB 3: Brightfield & Registration
        self.tab_bf = QWidget()
        layout_bf = QVBoxLayout(self.tab_bf)

        bf_view_bar = QHBoxLayout()
        bf_view_bar.addWidget(QLabel("<b>Select Brightfield Image / Overlay View:</b>"))
        self.combo_bf_view = NoScrollComboBox()
        self.combo_bf_view.currentIndexChanged.connect(self.on_bf_view_changed)
        bf_view_bar.addWidget(self.combo_bf_view, stretch=1)
        layout_bf.addLayout(bf_view_bar)

        self.canvas_bf = MplCanvas(self)
        self.toolbar_bf = NavigationToolbar(self.canvas_bf, self)
        layout_bf.addWidget(self.toolbar_bf)
        layout_bf.addWidget(self.canvas_bf)
        self.tabs.addTab(self.tab_bf, "Brightfield & Alignment")

        # TAB 4: Segmentation & Fits
        self.tab_masks = QWidget()
        self.canvas_masks = MplCanvas(self)
        self.toolbar_masks = NavigationToolbar(self.canvas_masks, self)
        layout_m = QVBoxLayout(self.tab_masks)
        layout_m.addWidget(self.toolbar_masks)
        layout_m.addWidget(self.canvas_masks)
        self.tabs.addTab(self.tab_masks, "Masks & Outlines")

        # TAB 5: Cell Feature Summaries
        self.tab_features = QWidget()
        self.canvas_features = MplCanvas(self)
        self.toolbar_features = NavigationToolbar(self.canvas_features, self)
        layout_f = QVBoxLayout(self.tab_features)
        layout_f.addWidget(self.toolbar_features)
        layout_f.addWidget(self.canvas_features)
        self.tabs.addTab(self.tab_features, "Cell Feature Summary")

        splitter.setSizes([360, 940])

    # ================= CONFIG PARSING =================

    def parse_config_file(self, filepath):
        config = {}
        PIXEL_KEYS = {'pixel_size', 'pixel_length', 'pixelsize', 'px_size', 'pixel_sz', 'pixel', 'px'}
        FRAME_KEYS = {'frame_time', 'frame_interval', 'frametime', 'exposure_time', 'exposure', 'time_step', 'timestep', 'dt', 'frame_duration', 'frame_period'}
        ALEX_KEYS = {'alex', 'alex_flag', 'alex_mode', 'use_alex', 'is_alex'}
        PRECISION_KEYS = {'localisation_precision', 'localization_precision', 'loc_prec', 'loc_precision', 'precision', 'sigma'}
        CHANNEL_KEYS = {'num_channels', 'channels', 'n_channels', 'channel_count', 'num_channel'}
        ROI_KEYS = {'roi_file', 'roi_path', 'roi', 'roifile', 'roi_filename', 'roi_file_path', 'roipath'}

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
                    elif '	' in line:
                        key, val = line.split('	', 1)
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
                    elif key in CHANNEL_KEYS:
                        m = re.search(r"\d+", val)
                        if m:
                            config['num_channels'] = int(m.group())
                    elif key in PRECISION_KEYS:
                        m = re.search(r"[-+]?\d*\.\d+|\d+", val)
                        if m:
                            config['localisation_precision'] = float(m.group())
                    elif key in ROI_KEYS:
                        config['roi_file'] = val
                    elif key == 'tmats_path':
                        config['tmats_path'] = val
        except Exception as e:
            print(f"Error parsing config file {filepath}: {e}")
        return config

    def has_config_conflicts(self, parsed_configs):
        configs = list(parsed_configs.values())
        if len(configs) <= 1:
            return False

        keys = ['pixel_size', 'frame_time', 'alex', 'num_channels', 'localisation_precision', 'tmats_path', 'roi_file']
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

    def scan_and_load_config(self):
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
            if 'alex' in selected_cfg:
                self.chk_alex.blockSignals(True)
                self.chk_alex.setChecked(selected_cfg['alex'])
                self.chk_alex.blockSignals(False)
            if 'num_channels' in selected_cfg:
                chan_str = str(selected_cfg['num_channels'])
                idx = self.combo_num_channels.findText(chan_str)
                if idx >= 0:
                    self.combo_num_channels.blockSignals(True)
                    self.combo_num_channels.setCurrentIndex(idx)
                    self.combo_num_channels.blockSignals(False)
            if 'tmats_path' in selected_cfg:
                self.config_tmats_path = selected_cfg['tmats_path']
            if 'roi_file' in selected_cfg and selected_cfg['roi_file']:
                roi_val = selected_cfg['roi_file']
                if not os.path.isabs(roi_val):
                    candidate = os.path.join(self.root_dir, roi_val)
                    if os.path.exists(candidate):
                        roi_val = candidate
                self.txt_roi_file.setText(roi_val)

        # Fallback: auto-locate any .roi file in root_dir if not specified
        if not self.txt_roi_file.text():
            roi_matches = glob.glob(os.path.join(self.root_dir, "**", "*.roi"), recursive=True)
            if roi_matches:
                self.txt_roi_file.setText(roi_matches[0])

    # ================= BROWSER SLOTS =================

    def update_save_dir_from_video(self, video_path):
        if video_path and os.path.exists(video_path):
            video_dir = os.path.dirname(os.path.abspath(video_path))
            self.txt_save_dir.setText(os.path.join(video_dir, "results"))

    def browse_video(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Fluorescence Video Stack", self.root_dir, "TIFF Files (*.tif *.tiff)")
        if fname:
            self.txt_video_path.setText(fname)
            self.update_save_dir_from_video(fname)

    def browse_bf(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Brightfield Reference Image", self.root_dir, "TIFF Files (*.tif *.tiff)")
        if fname: self.txt_bf_path.setText(fname)

    def browse_beads(self):
        dname = QFileDialog.getExistingDirectory(self, "Select Any Bead Folder or Base Directory", self.root_dir)
        if dname:
            prefix = re.sub(r'_\d+$', '', dname)
            self.txt_bead_path.setText(prefix)

    def browse_roi(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select ImageJ ROI File", self.root_dir, "ROI Files (*.roi)")
        if fname: self.txt_roi_file.setText(fname)

    def browse_save(self):
        dname = QFileDialog.getExistingDirectory(self, "Select Output Directory", self.root_dir)
        if dname: self.txt_save_dir.setText(dname)

    def browse_model(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Segmentation Model Weights", self.root_dir, "All Files (*);;Model Files (*.pth *.pt *.h5 *.keras *.onnx)")
        if fname: self.txt_model_path.setText(fname)

    def on_mask_type_changed(self, idx):
        mask_type = self.combo_mask_type.currentText()
        is_ai = mask_type in ["AI", "BF", "FL_AI"]
        self.combo_model_type.setEnabled(is_ai)
        self.txt_model_path.setEnabled(is_ai)

    # ================= BEAD ALIGNMENT ACTION SLOT =================

    def run_bead_alignment_action(self):
        mode_key = self.combo_bead_mode.currentData()
        if mode_key == "auto":
            self.run_task("bead")
        else:
            num_pairs = self.spin_num_pairs.value()
            self.run_manual_registration(num_pairs=num_pairs)

    def run_manual_registration(self, num_pairs=10):
        params = self.build_params()
        bead_file = None

        if params.bead_path:
            search_pattern = f"{params.bead_path}_*"
            bead_folders = sorted(glob.glob(search_pattern))
            if bead_folders:
                files = glob.glob(f"{bead_folders[0]}/*.ome.tif") + glob.glob(f"{bead_folders[0]}/*.tif")
                if files:
                    bead_file = files[0]

        if not bead_file or not os.path.exists(bead_file):
            bead_file, _ = QFileDialog.getOpenFileName(
                self, "Select Image Stack for Interactive Registration", self.root_dir, "TIFF Files (*.tif *.tiff)"
            )
            if not bead_file:
                return

        self.log(f"Loading image '{bead_file}' for interactive registration...")

        try:
            img = io.imread(bead_file)
            if img.ndim == 3:
                img = np.mean(img, axis=0)

            if num_pairs <= 0:
                self.log("Launching Native Interactive Drag & Rotate Dialog...")
                dialog = InteractiveDragDialog(img=img, save_dir=params.save_dir, tmats_path=params.tmats_path, parent=self)
            else:
                self.log(f"Launching Native Point-Picking Dialog for {num_pairs} pairs...")
                dialog = PointPickingDialog(img=img, save_dir=params.save_dir, tmats_path=params.tmats_path, num_pairs=num_pairs, parent=self)

            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.log("Interactive manual registration completed successfully!")
                self.load_and_render_results(params.save_dir)
                QMessageBox.information(
                    self,
                    "Registration Saved",
                    f"Transformation matrix successfully computed and saved using tmats_path config:\n{params.tmats_path}"
                )
            else:
                self.log("Interactive manual registration was cancelled by user.")

        except Exception as e:
            self.log(f"Manual registration error: {e}")
            QMessageBox.critical(
                self,
                "Registration Error",
                f"Interactive alignment failed:\n{e}\n\n{traceback.format_exc()}"
            )

    # ================= PIPELINE EXECUTION =================

    def log(self, message):
        self.txt_status_log.append(message)

    def build_params(self):
        bead_mode_key = self.combo_bead_mode.currentData()
        is_manual = (bead_mode_key == "manual")
        tmats_path = getattr(self, 'config_tmats_path', 'transformation_matrix.npy')

        return PreprocessParams(
            name=os.path.basename(self.txt_save_dir.text()),
            video_path=self.txt_video_path.text(),
            bf_path=self.txt_bf_path.text(),
            bead_path=self.txt_bead_path.text(),
            roi_file=self.txt_roi_file.text(),
            roi_channel=self.combo_roi_channel.currentText(),
            save_dir=self.txt_save_dir.text(),
            tmats_path=tmats_path,
            pixel_size=self.spin_pixel_size.value(),
            num_channels=int(self.combo_num_channels.currentText()),
            use_channel=self.combo_use_channel.currentText(),
            start_channel=self.combo_start_channel.currentText(),
            ALEX=self.chk_alex.isChecked(),
            num_frames=self.spin_num_frames.value(),
            frame_avg=self.spin_frame_avg.value(),
            mask_type=self.combo_mask_type.currentText(),
            model_type=self.combo_model_type.currentText(),
            model=self.txt_model_path.text(),
            model_params=self.spin_model_param.value(),
            threshold_param=self.spin_model_param.value(),
            inv_bf=self.chk_inv_bf.isChecked(),
            use_otsu=self.combo_use_otsu.currentText(),
            cell_fitting=self.chk_cell_fitting.isChecked(),
            area_filter=self.spin_area_filter.value(),
            overwrite=self.chk_overwrite.isChecked(),
            manual_registration=is_manual,
            manual_pairs=self.spin_num_pairs.value()
        )

    def run_task(self, task_type: str):
        params = self.build_params()
        if task_type == "bead" and not params.bead_path:
            QMessageBox.warning(self, "Missing Input", "Please specify a valid Bead Alignment Path.")
            return

        if task_type in ["video", "full"] and (not params.video_path or not os.path.exists(params.video_path)):
            QMessageBox.warning(self, "Missing Input", "Please select a valid fluorescence video file.")
            return

        self.txt_status_log.clear()
        self.log(f"Starting task '{task_type.upper()}' for directory: {params.save_dir}")

        self.progress_dialog = QProgressDialog(f"Running Task: {task_type.capitalize()}...", "Cancel", 0, 100, self)
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setValue(10)

        self.worker_thread = PreprocessWorkerThread(params, task_type=task_type, parent=self)
        self.worker_thread.progress_signal.connect(self.log)
        self.worker_thread.finished_signal.connect(self.on_pipeline_finished)
        self.worker_thread.error_signal.connect(self.on_pipeline_error)

        self.worker_thread.start()

    def on_pipeline_finished(self, save_dir):
        if self.progress_dialog:
            self.progress_dialog.close()
        self.log("Task completed successfully!")
        QMessageBox.information(self, "Success", f"Preprocessing task complete!\nResults saved to:\n{save_dir}")
        self.load_and_render_results(save_dir)

    def on_pipeline_error(self, error_msg):
        if self.progress_dialog:
            self.progress_dialog.close()
        self.log(f"ERROR: {error_msg}")
        QMessageBox.critical(self, "Task Error", f"An error occurred during task execution:\n{error_msg}")

    # ================= RESULT VISUALIZATION & SELECTOR SLOTS =================

    def on_bead_view_changed(self, index):
        image_path = self.combo_bead_view.currentData()
        if not image_path or not os.path.exists(image_path):
            return

        ax_bead = self.canvas_beads.axes
        ax_bead.clear()

        try:
            img = Image.open(image_path)
            ax_bead.imshow(img)
            title = self.combo_bead_view.currentText()
            ax_bead.set_title(title)
            ax_bead.set_xticks([]); ax_bead.set_yticks([])
        except Exception as e:
            ax_bead.text(0.5, 0.5, f"Error rendering image:\n{e}", ha='center', va='center')

        self.canvas_beads.draw()

    def populate_ncc_stats_table(self, save_dir):
        """Populates the QTableWidget with NCC before, after, and delta stats for all bead samples."""
        self.table_ncc_stats.setRowCount(0)

        csv_ncc = os.path.join(save_dir, "ncc_scores.csv")
        csv_reg = os.path.join(save_dir, "registration.csv")

        if os.path.exists(csv_ncc):
            df_n = pd.read_csv(csv_ncc)
            if not df_n.empty and 'NCC_After' in df_n.columns:
                max_ncc_idx = df_n['NCC_After'].idxmax()
                self.table_ncc_stats.setRowCount(len(df_n))

                for row_idx, row in df_n.iterrows():
                    sample_name = str(row.get('Bead_Folder', f'Bead {row_idx + 1}'))
                    ncc_before = float(row.get('NCC_Before', 0.0))
                    ncc_after = float(row.get('NCC_After', 0.0))
                    delta = ncc_after - ncc_before

                    item_sample = QTableWidgetItem(sample_name)
                    item_before = QTableWidgetItem(f"{ncc_before:.4f}")
                    item_after = QTableWidgetItem(f"{ncc_after:.4f}")
                    item_delta = QTableWidgetItem(f"{'+' if delta >= 0 else ''}{delta:.4f}")

                    # Highlight best row in light green
                    if row_idx == max_ncc_idx:
                        highlight_color = QColor(220, 245, 220)
                        bold_font = QFont()
                        bold_font.setBold(True)
                        for item in (item_sample, item_before, item_after, item_delta):
                            item.setBackground(highlight_color)
                            item.setFont(bold_font)

                    self.table_ncc_stats.setItem(row_idx, 0, item_sample)
                    self.table_ncc_stats.setItem(row_idx, 1, item_before)
                    self.table_ncc_stats.setItem(row_idx, 2, item_after)
                    self.table_ncc_stats.setItem(row_idx, 3, item_delta)

        elif os.path.exists(csv_reg):
            df_r = pd.read_csv(csv_reg)
            if not df_r.empty and 'Best_NCC' in df_r.columns:
                self.table_ncc_stats.setRowCount(1)
                best_folder = str(df_r['Best_Folder'].values[0])
                best_ncc = float(df_r['Best_NCC'].values[0])

                item_sample = QTableWidgetItem(f"{best_folder} (Best)")
                item_before = QTableWidgetItem("N/A")
                item_after = QTableWidgetItem(f"{best_ncc:.4f}")
                item_delta = QTableWidgetItem("N/A")

                self.table_ncc_stats.setItem(0, 0, item_sample)
                self.table_ncc_stats.setItem(0, 1, item_before)
                self.table_ncc_stats.setItem(0, 2, item_after)
                self.table_ncc_stats.setItem(0, 3, item_delta)

    def get_roi_coordinates(self, roi_path, img_height, img_width, is_split_view=True):
        """Reads ImageJ .roi file and returns clamped bounding box coordinates [y1, y2, x1, x2]."""
        half_width = img_width // 2 if is_split_view else img_width
        if not roi_path or not os.path.exists(roi_path):
            return [0, img_height, 0, half_width]

        try:
            from roifile import ImagejRoi
            roi = ImagejRoi.fromfile(roi_path)
            y1, x1, y2, x2 = int(roi.top), int(roi.left), int(roi.bottom), int(roi.right)

            if is_split_view:
                if x1 >= half_width: x1 -= half_width
                if x2 > half_width:  x2 -= half_width

            y1 = max(0, min(y1, img_height))
            y2 = max(0, min(y2, img_height))
            x1 = max(0, min(x1, half_width))
            x2 = max(0, min(x2, half_width))
            return [y1, y2, x1, x2]
        except Exception as e:
            print(f"Error reading ROI file '{roi_path}': {e}")
            return [0, img_height, 0, half_width]

    def on_bf_view_changed(self, index):
        view_data = self.combo_bf_view.currentData()
        if not view_data:
            return

        ax = self.canvas_bf.axes
        ax.clear()

        try:
            if isinstance(view_data, dict) and view_data.get("type") == "roi_overlay":
                img_path = view_data.get("img_path")
                roi_path = self.txt_roi_file.text()

                if not img_path or not os.path.exists(img_path):
                    ax.text(0.5, 0.5, "Image file not found for ROI overlay.", ha='center', va='center')
                    self.canvas_bf.draw()
                    return

                try:
                    import tifffile
                    img = tifffile.imread(img_path)
                except Exception:
                    img = np.array(Image.open(img_path))

                if img.ndim == 3:
                    img = np.mean(img, axis=0)

                h, w = img.shape[:2]
                is_split_view = (w > 1500)
                y1, y2, x1, x2 = self.get_roi_coordinates(roi_path, h, w, is_split_view)

                ax.imshow(img, cmap='gray')
                rect = mplpatches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor='red', facecolor='none', linestyle='--'
                )
                ax.add_patch(rect)
                roi_name = os.path.basename(roi_path) if roi_path else "Default ROI"
                ax.set_title(f"Full Registered Brightfield Image with ROI Overlay ({roi_name})\nROI: ({x1}, {y1}) to ({x2}, {y2})")
                ax.set_xticks([]); ax.set_yticks([])

            elif isinstance(view_data, str) and os.path.exists(view_data):
                file_path = view_data
                if file_path.lower().endswith(('.tif', '.tiff')):
                    try:
                        import tifffile
                        img = tifffile.imread(file_path)
                    except Exception:
                        img = np.array(Image.open(file_path))
                    ax.imshow(img, cmap='gray')
                else:
                    img = Image.open(file_path)
                    ax.imshow(img)

                title = self.combo_bf_view.currentText()
                ax.set_title(title)
                ax.set_xticks([]); ax.set_yticks([])

            else:
                ax.text(0.5, 0.5, "Selected image view unavailable.", ha='center', va='center')

        except Exception as e:
            ax.text(0.5, 0.5, f"Error rendering Brightfield view:\n{e}", ha='center', va='center')

        self.canvas_bf.draw()

    def load_and_render_results(self, save_dir):
        if not os.path.exists(save_dir):
            return

        params = self.build_params()
        if os.path.isabs(params.tmats_path):
            tmat_path = params.tmats_path
        else:
            tmat_path = os.path.join(save_dir, params.tmats_path)

        if not os.path.exists(tmat_path):
            tmat_path = os.path.join(save_dir, "transformation_matrix.npy")

        if os.path.exists(tmat_path):
            try:
                tmat = np.load(tmat_path)
                mat_str = (
                    f"Transformation Matrix (3x3):\n"
                    f"[{tmat[0,0]:10.5f}, {tmat[0,1]:10.5f}, {tmat[0,2]:10.5f}]\n"
                    f"[{tmat[1,0]:10.5f}, {tmat[1,1]:10.5f}, {tmat[1,2]:10.5f}]\n"
                    f"[{tmat[2,0]:10.5f}, {tmat[2,1]:10.5f}, {tmat[2,2]:10.5f}]"
                )
                self.lbl_matrix_display.setText(mat_str)
            except Exception as e:
                self.lbl_matrix_display.setText(f"Error reading matrix: {e}")
        else:
            self.lbl_matrix_display.setText("No transformation matrix saved yet.")

        # Summary Header Label
        csv_reg = os.path.join(save_dir, "registration.csv")
        csv_ncc = os.path.join(save_dir, "ncc_scores.csv")
        if os.path.exists(csv_reg):
            df_r = pd.read_csv(csv_reg)
            if not df_r.empty and 'Best_NCC' in df_r.columns:
                self.lbl_ncc_display.setText(
                    f"Best Folder: {df_r['Best_Folder'].values[0]}\n"
                    f"Best NCC Score: {df_r['Best_NCC'].values[0]:.4f}"
                )
        elif os.path.exists(csv_ncc):
            df_n = pd.read_csv(csv_ncc)
            if not df_n.empty and 'NCC_After' in df_n.columns:
                best_ncc = df_n['NCC_After'].max()
                self.lbl_ncc_display.setText(
                    f"Total Folders: {len(df_n)}\n"
                    f"Max NCC Score: {best_ncc:.4f}"
                )
        else:
            self.lbl_ncc_display.setText("NCC Performance: N/A")

        # Populate Full Detailed NCC Table
        self.populate_ncc_stats_table(save_dir)

        # Discover all bead registration overlay files & charts for dropdown
        self.combo_bead_view.blockSignals(True)
        self.combo_bead_view.clear()
        self.bead_view_map.clear()

        summed_overlay = os.path.join(save_dir, "Registered_overlay_summed_beads.png")
        manual_overlay = os.path.join(save_dir, "Registered_overlay_manual.png")
        ncc_chart = os.path.join(save_dir, "ncc_scores.png")

        if os.path.exists(summed_overlay):
            self.combo_bead_view.addItem("Summed Bead Alignment Overlay", userData=summed_overlay)
        if os.path.exists(manual_overlay):
            self.combo_bead_view.addItem("Manual Alignment Overlay", userData=manual_overlay)

        # Find all individual bead overlays (e.g., Registered_overlay_beads_1.png)
        individual_overlays = sorted(glob.glob(os.path.join(save_dir, "Registered_overlay_beads_*.png")))
        for overlay_path in individual_overlays:
            file_name = os.path.basename(overlay_path)
            sample_label = file_name.replace("Registered_overlay_", "").replace(".png", "").replace("_", " ").title()
            self.combo_bead_view.addItem(f"Sample Overlay: {sample_label}", userData=overlay_path)

        if os.path.exists(ncc_chart):
            self.combo_bead_view.addItem("NCC Performance Chart", userData=ncc_chart)

        self.combo_bead_view.blockSignals(False)

        if self.combo_bead_view.count() > 0:
            self.combo_bead_view.setCurrentIndex(0)
            self.on_bead_view_changed(0)
        else:
            ax_bead = self.canvas_beads.axes
            ax_bead.clear()
            ax_bead.text(0.5, 0.5, "No bead registration visual outputs found.", ha='center', va='center')
            self.canvas_beads.draw()

        # Render Tab 2: Channel Previews
        ax1 = self.canvas_channels.axes
        ax1.clear()
        l_png = os.path.join(save_dir, "L_avg.png")
        r_png = os.path.join(save_dir, "R_avg.png")
        l_tif = os.path.join(save_dir, "L_avg.tif")
        r_tif = os.path.join(save_dir, "R_avg.tif")

        if os.path.exists(l_png):
            img = Image.open(l_png)
            ax1.imshow(img)
            ax1.set_title("Channel L Average Preview")
        elif os.path.exists(r_png):
            img = Image.open(r_png)
            ax1.imshow(img)
            ax1.set_title("Channel R Average Preview")
        elif os.path.exists(l_tif):
            img = tifffile.imread(l_tif)
            ax1.imshow(img, cmap='gray')
            ax1.set_title("Channel L Average Preview (TIF)")
        elif os.path.exists(r_tif):
            img = tifffile.imread(r_tif)
            ax1.imshow(img, cmap='gray')
            ax1.set_title("Channel R Average Preview (TIF)")
        else:
            ax1.text(0.5, 0.5, "No channel average previews found.", ha='center', va='center')
        self.canvas_channels.draw()

        # Populate Tab 3: Dedicated Brightfield & Registration Views
        self.combo_bf_view.blockSignals(True)
        self.combo_bf_view.clear()

        bf_full_tif = os.path.join(save_dir, "BF_reg_full.tif")
        bf_crop_tif = os.path.join(save_dir, "BF.tif")
        bf_crop_png = os.path.join(save_dir, "BF.png")
        bf_overlay_png = os.path.join(save_dir, "BF_Registered_overlay.png")

        # 1. Full Registered BF + ROI Overlay (Interactive rendering)
        base_bg_for_overlay = bf_full_tif if os.path.exists(bf_full_tif) else (bf_crop_tif if os.path.exists(bf_crop_tif) else None)
        if base_bg_for_overlay:
            self.combo_bf_view.addItem(
                "Full Registered BF + ROI Overlay",
                userData={"type": "roi_overlay", "img_path": base_bg_for_overlay}
            )

        # 2. BF.png — Cropped ROI with Scalebar
        if os.path.exists(bf_crop_png):
            self.combo_bf_view.addItem("BF.png — Cropped ROI with Scalebar", userData=bf_crop_png)

        # 3. BF.tif — Cropped 16-bit ROI TIF
        if os.path.exists(bf_crop_tif):
            self.combo_bf_view.addItem("BF.tif — Cropped 16-bit ROI TIF", userData=bf_crop_tif)

        # 4. BF_reg_full.tif — Full Registered 16-bit TIF
        if os.path.exists(bf_full_tif):
            self.combo_bf_view.addItem("BF_reg_full.tif — Full Registered 16-bit TIF", userData=bf_full_tif)

        # 5. BF_Registered_overlay.png — 3-Panel Comparison
        if os.path.exists(bf_overlay_png):
            self.combo_bf_view.addItem("BF_Registered_overlay.png — 3-Panel Comparison", userData=bf_overlay_png)

        self.combo_bf_view.blockSignals(False)

        if self.combo_bf_view.count() > 0:
            self.combo_bf_view.setCurrentIndex(0)
            self.on_bf_view_changed(0)
        else:
            ax2 = self.canvas_bf.axes
            ax2.clear()
            ax2.text(0.5, 0.5, "No Brightfield imagery generated.", ha='center', va='center')
            self.canvas_bf.draw()

        # Render Tab 4: Masks & Outlines
        ax3 = self.canvas_masks.axes
        ax3.clear()
        fits_png = os.path.join(save_dir, "all_fits_mask.png")
        cells_png = os.path.join(save_dir, "all_cells_mask.png")
        if os.path.exists(fits_png):
            img = Image.open(fits_png)
            ax3.imshow(img)
            ax3.set_title("Fitted Mathematical Capsule Shapes")
        elif os.path.exists(cells_png):
            img = Image.open(cells_png)
            ax3.imshow(img)
            ax3.set_title("Binary Cell Segmentation Masks")
        else:
            ax3.text(0.5, 0.5, "No cell masks found.", ha='center', va='center')
        self.canvas_masks.draw()

        # Render Tab 5: Feature Summary
        ax4 = self.canvas_features.axes
        ax4.clear()
        summary_png = os.path.join(save_dir, "obj_feature_summary.png")
        csv_path = os.path.join(save_dir, "all_objects.csv")
        if os.path.exists(summary_png):
            img = Image.open(summary_png)
            ax4.imshow(img)
            ax4.set_title("Cell Property Distributions")
        elif os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if 'area' in df.columns:
                ax4.hist(df['area'].dropna(), bins=20, color='#008080', edgecolor='black')
                ax4.set_title("Cell Area Distribution")
                ax4.set_xlabel("Area (pixels)")
                ax4.set_ylabel("Count")
        else:
            ax4.text(0.5, 0.5, "No summary feature data found.", ha='center', va='center')
        self.canvas_features.draw()