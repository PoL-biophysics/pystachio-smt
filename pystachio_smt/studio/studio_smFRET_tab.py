# =========================================================================
# TAB 2: DEDICATED MANUAL smFRET ANALYSIS TAB
# =========================================================================

import os
import copy
import csv
import tempfile
from pathlib import Path
from types import SimpleNamespace
import glob
import traceback
import numpy as np
import tifffile as tf
import cv2
import matplotlib

# Backend selection belongs to the application, not this tab.
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import RectangleSelector

from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QSlider, QLineEdit,
                             QLabel, QRadioButton, QGroupBox, QFileDialog, 
                             QButtonGroup, QGridLayout, QCheckBox,
                             QSpinBox, QComboBox, QProgressBar, QDoubleSpinBox,
                             QScrollArea, QSizePolicy, QLayout)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

import images
import spots
import parameters
import postprocessing
import trajectories
import tracking
from studio.studio_helpers import show_popup

MODULES_LOADED = True



# D: donor emission under donor excitation; F: acceptor emission under donor
# excitation; A: acceptor emission under acceptor excitation. Intensities must
# already be background-subtracted. Alpha = donor leakage F/D, gamma scales D.
# Beta=1 and direct excitation=0 are assumptions, not inferred calibrations.
def corrected_es(donor, acceptor, fret, alpha=0.0, gamma=1.0):
    """Return E_alpha_gamma and S_alpha_gamma; invalid totals are NaN.

    S is an ALEX intensity stoichiometry, not molecular copy number.
    Noise can give finite ratios outside [0,1]; these are never clipped.
    """
    alpha, gamma = float(alpha), float(gamma)
    if not np.isfinite(alpha) or alpha < 0 or not np.isfinite(gamma) or gamma <= 0:
        raise ValueError('Alpha must be finite and >= 0; gamma finite and > 0.')
    d, a, f = np.broadcast_arrays(np.asarray(donor, dtype=float),
                                  np.asarray(acceptor, dtype=float),
                                  np.asarray(fret, dtype=float))
    def ratio(n, den):
        out = np.full(den.shape, np.nan, dtype=float)
        np.divide(n, den, out=out, where=np.isfinite(n) & np.isfinite(den) & (den > 0))
        out[~np.isfinite(out)] = np.nan
        return float(out) if out.ndim == 0 else out
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        corrected_f = f - alpha*d
        total = corrected_f + gamma*d
        return ratio(corrected_f, total), ratio(total, total+a)


def _registration(matrix):
    if matrix is None:
        return None
    m = np.asarray(matrix, dtype=float)
    # Legacy main window appended a second homogeneous row to StackReg's 3x3.
    # Accept only that exact redundant-row representation, not arbitrary 4x3 data.
    if m.shape == (4, 3) and np.allclose(m[2:], [[0, 0, 1], [0, 0, 1]]):
        m = m[:3]
    if m.shape not in ((2, 3), (3, 3)) or not np.isfinite(m).all():
        raise ValueError('Registration must be a finite 2x3 or 3x3 affine matrix.')
    if m.shape == (3, 3) and not np.allclose(m[2], [0, 0, 1]):
        raise ValueError('Projective registration is unsupported; expected last row [0,0,1].')
    m = m[:2].copy()
    if abs(np.linalg.det(m[:, :2])) < 1e-12:
        raise ValueError('Registration matrix is singular.')
    return m


def _map_registered(x, y, matrix):
    m = _registration(matrix)
    return (float(x), float(y)) if m is None else tuple(cv2.invertAffineTransform(m) @ [x, y, 1.])


def _fixed_traces(frame_spots, coords):
    """Keep fixed-spot identity across channels; do not relink stationary ROIs."""
    if not frame_spots:
        return []
    values = np.asarray([s.spot_intensity for s in frame_spots], dtype=float)
    if values.shape != (len(frame_spots), len(coords)):
        raise ValueError('Photometry returned mismatched spot counts.')
    return [SimpleNamespace(id=i, intensity=values[:, i].copy(),
                            path=[list(xy) for _ in frame_spots], start_frame=0,
                            end_frame=len(frame_spots)-1, length=len(frame_spots))
            for i, xy in enumerate(coords)]


class FullAnalysisWorker(QThread):
    result_ready = pyqtSignal(object, object, object, object, object)
    progress = pyqtSignal(str)

    def __init__(self, acceptor_data, donor_data, fret_data, active_frame, track_params, registration_matrix, roi=None):
        super().__init__()
        self.acceptor_data = acceptor_data
        self.donor_data = donor_data
        self.fret_data = fret_data
        self.active_frame = active_frame
        self.track_params = copy.deepcopy(track_params)
        self.registration_matrix = _registration(registration_matrix)
        self.roi = roi

    def _map_to_donor(self, ax, ay):
        return _map_registered(ax, ay, self.registration_matrix)

    def run(self):
        try:
            self.progress.emit("Detecting spots...")
            spots_res = tracking.track_frame(self.acceptor_data[self.active_frame], self.active_frame, self.track_params)
            
            if not spots_res or len(spots_res.positions) == 0:
                self.result_ready.emit([], [], [], [], None)
                return

            max_x, max_y = self.donor_data.frame_size[0], self.donor_data.frame_size[1]
            valid_acceptor_coords, valid_donor_coords = [], []
            
            for p in spots_res.positions:
                ax, ay = float(p[0]), float(p[1])
                
                if self.roi is not None:
                    r_xmin, r_xmax, r_ymin, r_ymax = self.roi
                    if not (r_xmin <= ax <= r_xmax and r_ymin <= ay <= r_ymax): continue

                dx, dy = self._map_to_donor(ax, ay)
                    
                if not (np.isfinite(dx) and np.isfinite(dy) and np.isfinite(ax) and np.isfinite(ay)): continue
                if not (0 <= dx < max_x and 0 <= dy < max_y and 0 <= ax < max_x and 0 <= ay < max_y): continue
                    
                valid_acceptor_coords.append([ax, ay])
                valid_donor_coords.append([dx, dy])

            if not valid_acceptor_coords:
                self.result_ready.emit([], [], [], [], None)
                return

            num_frames = self.donor_data.num_frames
            channel_spots_d, channel_spots_a, channel_spots_f = [], [], []

            for f in range(num_frames):
                if f % 5 == 0 or f == num_frames - 1:
                    self.progress.emit(f"Extracting frame {f + 1}/{num_frames}...")

                for target_list, coords, data in zip(
                    [channel_spots_d, channel_spots_a, channel_spots_f],
                    [valid_donor_coords, valid_acceptor_coords, valid_acceptor_coords],
                    [self.donor_data, self.acceptor_data, self.fret_data]
                ):
                    s = spots.Spots(frame=f)
                    s.set_positions(coords)
                    s.num_spots = len(coords)
                    s.get_spot_intensities(data[f].as_image(), self.track_params)
                    target_list.append(s)

            self.progress.emit("Building trajectories...")
            trajs_d = _fixed_traces(channel_spots_d, valid_donor_coords)
            trajs_a = _fixed_traces(channel_spots_a, valid_acceptor_coords)
            trajs_f = _fixed_traces(channel_spots_f, valid_acceptor_coords)

            self.result_ready.emit(trajs_d, trajs_a, trajs_f, valid_acceptor_coords, None)

        except Exception as e:
            self.result_ready.emit([], [], [], [], str(e))


class BatchWorker(QThread):
    result_ready = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, tab, files_to_process, directory, track_params, win_size, direction, orient_norm, temp_even):
        super().__init__()
        self.tab = SimpleNamespace(
            params=copy.deepcopy(tab.params), roi=copy.deepcopy(tab.roi),
            registration_matrix=_registration(tab.registration_matrix),
            _loaded_orient_norm=orient_norm,
            active_frame=tab.active_frame, alpha=tab.alpha, gamma=tab.gamma,
            e_min=tab.e_min, e_max=tab.e_max, s_min=tab.s_min, s_max=tab.s_max,
            donor_data=None, acceptor_data=None, fret_data=None,
            coords=[], trajs_donor=[], trajs_acceptor=[], trajs_fret=[])
        for name in ('get_data', '_build_pystachio_image', '_map_to_donor',
                     '_extract_trajectories', 'get_avg_bounds', '_registration_for_channels'):
            setattr(self.tab, name, getattr(SmFretTab, name).__get__(self.tab))
        self.files_to_process = files_to_process
        self.directory = directory
        self.source_directory = directory
        self.track_params = copy.deepcopy(track_params)
        self.win_size = win_size
        self.direction = direction
        self.orient_norm = orient_norm
        self.temp_even = temp_even

    def run(self):
        try:
            with tempfile.TemporaryDirectory(dir=self.source_directory, prefix='.smfret_batch_') as stage:
                self.directory = stage
                if self._process():
                    for output in Path(stage).iterdir():
                        output.replace(Path(self.source_directory) / output.name)
                    self.result_ready.emit(self.source_directory)
        except Exception:
            self.error.emit(traceback.format_exc())
        finally:
            self.directory = self.source_directory

    def _process(self):
        try:
            with open(os.path.join(self.directory, "consolidated_smFRET_raw_all_frames.txt"), "w") as f_raw, \
                 open(os.path.join(self.directory, "consolidated_smFRET_filtered_all_frames.txt"), "w") as f_filt_all, \
                 open(os.path.join(self.directory, "consolidated_smFRET_filtered_first_frame.txt"), "w") as f_filt_first:
                 
                writers = [csv.writer(handle, delimiter='\t') for handle in (f_raw, f_filt_all, f_filt_first)]
                for writer in writers:
                    writer.writerow(('Image_Filename', 'Traj_ID', 'X_Coord', 'Y_Coord', 'Frame',
                                     'E_FRET', 'S', 'Donor_Int', 'Acc_Int', 'FRET_Int'))
                global_traj_id = 0

                for idx, fname in enumerate(self.files_to_process):
                    self.progress.emit(idx + 1)
                    
                    try:
                        self.tab.source_fname = fname
                        self.tab.params.name = str(Path(fname).with_suffix(""))
                        self.tab.get_data(fname, orient_norm=self.orient_norm, temp_even=self.temp_even)
                    except Exception as exc:
                        raise RuntimeError(f"Failed to load {fname}: {exc}") from exc
                        
                    if not self.tab.acceptor_data or self.tab.acceptor_data.num_frames == 0: continue

                    detect_frame = min(self.tab.active_frame, self.tab.acceptor_data.num_frames - 1)
                    spots_res = tracking.track_frame(self.tab.acceptor_data[detect_frame], detect_frame, self.track_params)
                    
                    if not spots_res or len(spots_res.positions) == 0: continue

                    all_coords = []
                    for p in spots_res.positions:
                        pt = [float(p[0]), float(p[1])]
                        if self.tab.roi is not None:
                            rx1, rx2, ry1, ry2 = self.tab.roi
                            if not (rx1 <= pt[0] <= rx2 and ry1 <= pt[1] <= ry2): continue
                        all_coords.append(pt)

                    if not all_coords: continue

                    self.tab.coords = all_coords
                    self.tab._extract_trajectories(self.track_params)

                    start_idx, end_idx = self.tab.get_avg_bounds(detect_frame, self.win_size, self.direction, self.tab.donor_data.num_frames)

                    for i in range(len(self.tab.trajs_donor)):
                        d_trace = self.tab.trajs_donor[i].intensity
                        a_trace = self.tab.trajs_acceptor[i].intensity
                        f_trace = self.tab.trajs_fret[i].intensity

                        d_avg = np.sum(d_trace[start_idx:end_idx])
                        a_avg = np.sum(a_trace[start_idx:end_idx])
                        fr_avg = np.sum(f_trace[start_idx:end_idx])
                        
                        e_avg, s_avg = corrected_es(d_avg, a_avg, fr_avg, self.tab.alpha, self.tab.gamma)

                        passes_filter = (self.tab.e_min <= e_avg <= self.tab.e_max) and (self.tab.s_min <= s_avg <= self.tab.s_max)
                        base_fname, x_coord, y_coord = os.path.relpath(fname, self.source_directory), self.tab.coords[i][0], self.tab.coords[i][1]
                        
                        for f_idx in range(self.tab.donor_data.num_frames):
                            d_val, a_val, fr_val = d_trace[f_idx], a_trace[f_idx], f_trace[f_idx]
                            e_val, s_val = corrected_es(d_val, a_val, fr_val, self.tab.alpha, self.tab.gamma)
                            row = (base_fname, global_traj_id, x_coord, y_coord, f_idx,
                                   e_val, s_val, d_val, a_val, fr_val)
                            writers[0].writerow(row)
                            if passes_filter:
                                writers[1].writerow(row)
                                if f_idx == 0: writers[2].writerow(row)
                        global_traj_id += 1

            with open(os.path.join(self.directory, "consolidated_smFRET_summary.txt"), "w") as f_sum:
                f_sum.write(f"Alpha={self.tab.alpha}; gamma={self.tab.gamma}; beta=1; direct excitation=0\n")
                f_sum.write("E=(F-alpha*D)/(F-alpha*D+gamma*D); S=(F-alpha*D+gamma*D)/(F-alpha*D+gamma*D+A)\n")
                f_sum.write("smFRET Batch Analysis Summary\n=============================\n")
                f_sum.write(f"Processed {len(self.files_to_process)} files.\n\nFilters Applied:\n")
                f_sum.write(f"  E_FRET: {self.tab.e_min} - {self.tab.e_max}  |  Stoich: {self.tab.s_min} - {self.tab.s_max}\n")
                f_sum.write(f"  Averaging: {self.win_size} frames ({self.direction})\n")
                f_sum.write(f"  ROI Active: {f'X=[{self.tab.roi[0]:.1f}, {self.tab.roi[1]:.1f}] Y=[{self.tab.roi[2]:.1f}, {self.tab.roi[3]:.1f}]' if self.tab.roi else 'None (Full Frame)'}\n")

            return True

        except Exception:
            self.error.emit(traceback.format_exc())


class HistogramWindow(QWidget):
    def __init__(self, parent_tab, e_vals, frame_idx, out_path):
        super().__init__()
        self.setWindowTitle(f"E_FRET Histogram - Frame {frame_idx}")
        self.resize(500, 400)
        
        layout = QVBoxLayout(self)
        self.fig = Figure(figsize=(5, 4))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        
        self.ax.hist(e_vals, bins=np.linspace(-0.2, 1.2, 29), color='green', alpha=0.7, edgecolor='black')
        self.ax.set_xlabel("FRET Efficiency (E)")
        self.ax.set_ylabel("Count")
        self.ax.set_title(f"E_FRET Distribution (Frame {frame_idx})")
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.fig.tight_layout()
        
        try:
            self.fig.savefig(out_path, dpi=300)
            show_popup(self, "Histogram Saved", f"Successfully generated and saved histogram to:\n{out_path}")
        except Exception as e:
            show_popup(self, "Error Saving Histogram", f"Could not save histogram to disk: {e}", critical=True)

    def closeEvent(self, event):
        self.canvas.deleteLater()
        event.accept()


class StoichiometryWindow(QWidget):
    def __init__(self, parent_tab):
        super().__init__()
        self.parent_tab = parent_tab
        self.snapshot = parent_tab.selection_snapshot()
        self.setWindowTitle("Select E/S bounds")
        self.resize(700, 600)
        
        self.E_scatter, self.S_scatter = [], []
        self.traj_E_avg, self.traj_S_avg = [], []
        
        win_size = self.parent_tab.avg_window_spin.value()
        direction = self.parent_tab.avg_direction_combo.currentText()
        active_f = self.parent_tab.active_frame
        max_f = self.parent_tab.donor_data.num_frames if self.parent_tab.donor_data else 0
        
        start_idx, end_idx = self.parent_tab.get_avg_bounds(active_f, win_size, direction, max_f)
        
        for i in range(len(self.parent_tab.trajs_donor)):
            d_trace = self.parent_tab.trajs_donor[i].intensity
            a_trace = self.parent_tab.trajs_acceptor[i].intensity
            f_trace = self.parent_tab.trajs_fret[i].intensity
            
            d_arr, a_arr, f_arr = np.array(d_trace), np.array(a_trace), np.array(f_trace)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                e_vals, s_vals = corrected_es(d_arr, a_arr, f_arr, parent_tab.alpha, parent_tab.gamma)
            
            self.E_scatter.extend(e_vals)
            self.S_scatter.extend(s_vals)
                
            d_avg, a_avg, fr_avg = np.sum(d_arr[start_idx:end_idx]), np.sum(a_arr[start_idx:end_idx]), np.sum(f_arr[start_idx:end_idx])
            
            e, st = corrected_es(d_avg, a_avg, fr_avg, parent_tab.alpha, parent_tab.gamma)
            self.traj_E_avg.append(e)
            self.traj_S_avg.append(st)
            
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        self.fig = Figure(figsize=(6, 5))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        
        self.ax.scatter(self.E_scatter, self.S_scatter, c='blue', alpha=0.15, s=5, edgecolors='none', label='All Frames')
        self.ax.scatter(self.traj_E_avg, self.traj_S_avg, c='red', alpha=0.9, s=30, edgecolors='k', label='Trajectory Averages')
        
        self.ax.set_xlim(-0.1, 1.1)
        self.ax.set_ylim(-0.1, 1.1)
        self.ax.set_xlabel("FRET Efficiency (E)")
        self.ax.set_ylabel("S")
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.ax.legend(loc='upper right')
        
        self.rect = plt.Rectangle((0, 0), 1, 1, fill=False, color='red', linewidth=2)
        self.ax.add_patch(self.rect)
        
        controls = QGridLayout()
        self.e_min_sl = QSlider(Qt.Orientation.Horizontal); self.e_min_sl.setRange(0, 100); self.e_min_sl.setValue(int(self.parent_tab.e_min * 100))
        self.e_max_sl = QSlider(Qt.Orientation.Horizontal); self.e_max_sl.setRange(0, 100); self.e_max_sl.setValue(int(self.parent_tab.e_max * 100))
        self.s_min_sl = QSlider(Qt.Orientation.Horizontal); self.s_min_sl.setRange(0, 100); self.s_min_sl.setValue(int(self.parent_tab.s_min * 100))
        self.s_max_sl = QSlider(Qt.Orientation.Horizontal); self.s_max_sl.setRange(0, 100); self.s_max_sl.setValue(int(self.parent_tab.s_max * 100))
        
        for sl in [self.e_min_sl, self.e_max_sl, self.s_min_sl, self.s_max_sl]:
            sl.valueChanged.connect(self.update_rect)
        
        controls.addWidget(QLabel("E min:"), 0, 0); controls.addWidget(self.e_min_sl, 0, 1)
        controls.addWidget(QLabel("E max:"), 1, 0); controls.addWidget(self.e_max_sl, 1, 1)
        controls.addWidget(QLabel("S min:"), 2, 0); controls.addWidget(self.s_min_sl, 2, 1)
        controls.addWidget(QLabel("S max:"), 3, 0); controls.addWidget(self.s_max_sl, 3, 1)
        layout.addLayout(controls)
        
        self.btn_confirm = QPushButton("Confirm selection")
        self.btn_confirm.setStyleSheet("background-color: #2b5b84; color: white; font-weight: bold;")
        self.btn_confirm.clicked.connect(self.confirm_selection)
        layout.addWidget(self.btn_confirm)
        self.update_rect()
        
    def update_rect(self):
        if self.e_min_sl.value() > self.e_max_sl.value(): self.e_min_sl.setValue(self.e_max_sl.value())
        if self.s_min_sl.value() > self.s_max_sl.value(): self.s_min_sl.setValue(self.s_max_sl.value())
        
        e_min, e_max = self.e_min_sl.value() / 100.0, self.e_max_sl.value() / 100.0
        s_min, s_max = self.s_min_sl.value() / 100.0, self.s_max_sl.value() / 100.0
        
        self.rect.set_xy((e_min, s_min))
        self.rect.set_width(e_max - e_min)
        self.rect.set_height(s_max - s_min)
        self.canvas.draw_idle()
        
    def confirm_selection(self):
        if self.snapshot != self.parent_tab.selection_snapshot():
            show_popup(self, "Selection changed", "Data or settings changed; reopen E/S selection.")
            return
        e_min, e_max = self.e_min_sl.value() / 100.0, self.e_max_sl.value() / 100.0
        s_min, s_max = self.s_min_sl.value() / 100.0, self.s_max_sl.value() / 100.0
        
        keep_indices = [i for i, (e, s) in enumerate(zip(self.traj_E_avg, self.traj_S_avg)) 
                        if e_min <= e <= e_max and s_min <= s <= s_max]
                
        if not keep_indices:
            show_popup(self, "Selection Empty", "No spots fall within this selection. Try adjusting the sliders.", critical=True)
            return
            
        pt = self.parent_tab
        pt.e_min, pt.e_max, pt.s_min, pt.s_max = e_min, e_max, s_min, s_max
        if pt._selection_backup is None:
            pt._selection_backup = (list(pt.coords), list(pt.trajs_donor), list(pt.trajs_acceptor), list(pt.trajs_fret))
        pt.coords = [pt.coords[i] for i in keep_indices]
        pt.trajs_donor = [pt.trajs_donor[i] for i in keep_indices]
        pt.trajs_acceptor = [pt.trajs_acceptor[i] for i in keep_indices]
        pt.trajs_fret = [pt.trajs_fret[i] for i in keep_indices]
        
        pt.refresh()
        self.close()

    def closeEvent(self, event):
        self.canvas.deleteLater()
        event.accept()


class SmFretTab(QWidget):
    def __init__(self, parent_suite=None):
        super().__init__()
        self.parent_suite = parent_suite
        self.coords = []
        self.donor_data = self.acceptor_data = self.fret_data = None
        self.trajs_donor, self.trajs_acceptor, self.trajs_fret = [], [], []
        self.params = parameters.Parameters()
        self.active_frame = 0
        self.registration_matrix = self.source_fname = None
        self.cb1 = self.cb2 = self.cb3 = None
        self.worker = None
        self.batch_worker = None
        
        self.e_min, self.e_max, self.s_min, self.s_max = 0.0, 1.0, 0.0, 1.0
        self.roi, self.roi_selectors = None, []
        self._press_pos = self._press_axes = None

        self.alpha, self.gamma = 0.0, 1.0
        self._selection_backup = None
        self.init_ui()

    def _map_to_donor(self, x, y):
        return _map_registered(x, y, self._registration_for_channels())

    def _registration_for_channels(self):
        """Convert physical left->right calibration into donor->acceptor."""
        matrix = _registration(self.registration_matrix)
        if matrix is None:
            return None
        normal = getattr(self, '_loaded_orient_norm', True)
        return matrix if normal else cv2.invertAffineTransform(matrix)

    def get_avg_bounds(self, current_frame, window_size, direction, max_frames):
        if direction == "Forward":
            start_idx = current_frame
            end_idx = current_frame + window_size
        elif direction == "Centered":
            start_idx = current_frame - (window_size // 2)
            end_idx = start_idx + window_size
        elif direction == "Trailing":
            start_idx = current_frame - window_size + 1
            end_idx = current_frame + 1
        else:
            start_idx, end_idx = current_frame, current_frame + window_size

        start_idx = max(0, start_idx)
        end_idx = min(max_frames, end_idx)
        
        if start_idx == end_idx and end_idx < max_frames: end_idx += 1
        elif start_idx == end_idx and start_idx > 0: start_idx -= 1
            
        return start_idx, end_idx

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        plot_layout = QVBoxLayout()
        self.fig = Figure(figsize=(14, 10))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        self.build_mosaic()
        
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Frame Index:"))
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self.on_frame_change)
        slider_layout.addWidget(self.frame_slider)
        
        self.frame_label = QLabel("0 / 0")
        slider_layout.addWidget(self.frame_label)
        plot_layout.addLayout(slider_layout)
        main_layout.addLayout(plot_layout, stretch=4)

        controls = QWidget()
        control_layout = QVBoxLayout(controls)
        control_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinAndMaxSize)
        self.btn_pick_file = QPushButton("Load TIF Stack File")
        self.btn_pick_file.clicked.connect(self.pick_file)
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self.clear_plot)
        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.refresh)
        
        control_layout.addWidget(self.btn_pick_file)
        control_layout.addWidget(self.btn_clear)
        control_layout.addWidget(self.btn_refresh)

        roi_group = QGroupBox("Region of Interest (ROI)")
        roi_layout = QVBoxLayout()
        self.lbl_roi_status = QLabel("ROI: Full Frame (No Filter)")
        self.lbl_roi_status.setStyleSheet("font-weight: bold; color: #2b5b84;")
        self.btn_reset_roi = QPushButton("Reset ROI")
        self.btn_reset_roi.setStyleSheet("background-color: #5b2b84; color: white; font-weight: bold;")
        self.btn_reset_roi.clicked.connect(self.reset_roi)
        roi_layout.addWidget(self.lbl_roi_status)
        roi_layout.addWidget(self.btn_reset_roi)
        roi_group.setLayout(roi_layout)
        control_layout.addWidget(roi_group)

        avg_group = QGroupBox("E/S averaging")
        avg_layout = QHBoxLayout()
        self.avg_window_spin = QSpinBox()
        self.avg_window_spin.setRange(1, 1000)
        self.avg_window_spin.setValue(5)
        self.avg_direction_combo = QComboBox()
        self.avg_direction_combo.addItems(["Forward", "Centered", "Trailing"])
        
        avg_layout.addWidget(QLabel("Frames to Avg:"))
        avg_layout.addWidget(self.avg_window_spin)
        avg_layout.addWidget(QLabel("Direction:"))
        avg_layout.addWidget(self.avg_direction_combo)
        avg_group.setLayout(avg_layout)
        control_layout.addWidget(avg_group)

        correction_group = QGroupBox("Corrections")
        correction_layout = QGridLayout(correction_group)
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0, 10); self.alpha_spin.setDecimals(6)
        self.alpha_spin.setSingleStep(0.01); self.alpha_spin.setValue(self.alpha)
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.000001, 100); self.gamma_spin.setDecimals(6)
        self.gamma_spin.setSingleStep(0.05); self.gamma_spin.setValue(self.gamma)
        correction_layout.addWidget(QLabel("Alpha"), 0, 0)
        correction_layout.addWidget(self.alpha_spin, 0, 1)
        correction_layout.addWidget(QLabel("Gamma"), 1, 0)
        correction_layout.addWidget(self.gamma_spin, 1, 1)
        control_layout.addWidget(correction_group)
        self.alpha_spin.valueChanged.connect(self.on_corrections_changed)
        self.gamma_spin.valueChanged.connect(self.on_corrections_changed)
        self.btn_reset_es = QPushButton("Reset E/S bounds")
        self.btn_reset_es.clicked.connect(self.reset_es_selection)

        self.btn_autodetect = QPushButton("Autodetect spots")
        self.btn_autodetect.clicked.connect(self.autodetect_spots)
        self.btn_stoic = QPushButton("Select E/S bounds")
        self.btn_stoic.clicked.connect(self.open_stoic_window)
        self.btn_hist = QPushButton("Show/save E_FRET histogram")
        self.btn_hist.clicked.connect(self.show_histogram)
        
        self.btn_batch_analyse = QPushButton("Analyse all")
        self.btn_batch_analyse.setStyleSheet("background-color: #842b5b; color: white; font-weight: bold;")
        self.btn_batch_analyse.clicked.connect(self.batch_analyse_all)

        self.batch_progress = QProgressBar()
        self.batch_progress.setValue(0)
        self.batch_progress.hide()

        self.chk_subdirs = QCheckBox("Include all subdirectories")
        self.chk_subdirs.setChecked(True)
        
        control_layout.addWidget(self.btn_autodetect)
        control_layout.addWidget(self.btn_stoic)
        control_layout.addWidget(self.btn_reset_es)
        control_layout.addWidget(self.btn_hist)
        control_layout.addWidget(self.btn_batch_analyse)
        control_layout.addWidget(self.batch_progress)
        control_layout.addWidget(self.chk_subdirs)

        temporal_group = QGroupBox("Temporal Deinterleaving (ALEX)")
        temporal_layout = QVBoxLayout()
        self.temporal_cluster = QButtonGroup(self)
        self.rb_temp_even = QRadioButton("Channel 1 / Even Frames First")
        self.rb_temp_odd = QRadioButton("Channel 2 / Odd Frames First")
        self.rb_temp_even.setChecked(True)
        self.temporal_cluster.addButton(self.rb_temp_even); self.temporal_cluster.addButton(self.rb_temp_odd)
        temporal_layout.addWidget(self.rb_temp_even); temporal_layout.addWidget(self.rb_temp_odd)
        temporal_group.setLayout(temporal_layout)
        control_layout.addWidget(temporal_group)

        spatial_group = QGroupBox("Spatial Orientation Map (smFRET)")
        spatial_layout = QVBoxLayout()
        self.spatial_cluster = QButtonGroup(self)
        self.rb_orient_norm = QRadioButton("Left: Donor | Right: FRET/Acceptor")
        self.rb_orient_inv = QRadioButton("Left: FRET/Acceptor | Right: Donor")
        self.rb_orient_norm.setChecked(True)
        self.spatial_cluster.addButton(self.rb_orient_norm); self.spatial_cluster.addButton(self.rb_orient_inv)
        spatial_layout.addWidget(self.rb_orient_norm); spatial_layout.addWidget(self.rb_orient_inv)
        spatial_group.setLayout(spatial_layout)
        control_layout.addWidget(spatial_group)
        
        self.rb_temp_even.toggled.connect(self.on_settings_changed)
        self.rb_orient_norm.toggled.connect(self.on_settings_changed)
        
        calib_group = QGroupBox("Calibration & Registration")
        calib_layout = QVBoxLayout()
        self.btn_registration = QPushButton("Calculate Registration Matrix")
        self.btn_registration.clicked.connect(self.parent_suite.run_registration if self.parent_suite else lambda: None)
        self.btn_clear_registration = QPushButton("Clear Registration Matrix")
        self.btn_clear_registration.clicked.connect(self.parent_suite.clear_registration if self.parent_suite else lambda: None)
        calib_layout.addWidget(self.btn_registration)
        calib_layout.addWidget(self.btn_clear_registration)
        calib_group.setLayout(calib_layout)
        control_layout.addWidget(calib_group)
        
        control_layout.addWidget(QLabel("Output name"))
        self.outname_box = QLineEdit("smFRET_output.dat")
        control_layout.addWidget(self.outname_box)
        self.btn_save = QPushButton("Export FRET Data")
        self.btn_save.clicked.connect(self.save_data)
        control_layout.addWidget(self.btn_save)
        
        ck_group = QGroupBox("Chung-Kennedy Filter Configuration")
        ck_layout = QVBoxLayout()
        self.ck_yes = QRadioButton("Apply Filter View")
        self.ck_no = QRadioButton("Keep Raw Trajectory View")
        self.ck_yes.setChecked(True)
        ck_layout.addWidget(self.ck_yes); ck_layout.addWidget(self.ck_no)
        ck_group.setLayout(ck_layout)
        control_layout.addWidget(ck_group)
        self.ck_yes.setToolTip("Smooth the displayed intensities only; selection and exports use unsmoothed intensities.")
        self.ck_yes.toggled.connect(self.refresh)
        
        control_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(325)
        scroll.setWidget(controls)
        self.sidebar_scroll = scroll
        main_layout.addWidget(scroll, stretch=1)

    def selection_snapshot(self):
        return (id(self.donor_data), self.active_frame, self.alpha, self.gamma,
                tuple(tuple(c) for c in self.coords), tuple(id(t) for t in self.trajs_donor),
                self.avg_window_spin.value(), self.avg_direction_combo.currentText(),
                repr(self.registration_matrix), self.roi)

    def reset_es_selection(self):
        if self._selection_backup is not None:
            self.coords, self.trajs_donor, self.trajs_acceptor, self.trajs_fret = self._selection_backup
            self._selection_backup = None
        self.e_min = self.s_min = 0.0
        self.e_max = self.s_max = 1.0
        self.refresh()

    def on_corrections_changed(self, *_args):
        self.alpha, self.gamma = self.alpha_spin.value(), self.gamma_spin.value()
        if hasattr(self, 'stoic_win'):
            self.stoic_win.close()
        self.reset_es_selection()

    def build_mosaic(self):
        for selector in self.roi_selectors:
            selector.disconnect_events()
        self.fig.clear()
        mosaic = [['image1', 'image2', 'image3'],
                  ['plot1', 'plot1', 'plot1'],
                  ['plot2', 'plot2', 'plot2'],
                  ['plot3', 'plot3', 'plot3'],
                  ['plot4', 'plot4', 'plot4']]
        self.ax_dict = self.fig.subplot_mosaic(mosaic, gridspec_kw={'height_ratios': [5, 1, 1, 1, 1]})
        self.fig.subplots_adjust(left=0.06, right=0.92, top=0.96, bottom=0.06, hspace=0.35, wspace=0.15)
        self.donor_im, self.acc_im, self.fret_im = self.ax_dict['image1'], self.ax_dict['image2'], self.ax_dict['image3']
        self.iplot_donor, self.iplot_acceptor, self.iplot_fret = self.ax_dict['plot1'], self.ax_dict['plot2'], self.ax_dict['plot3']
        self.plot_efret = self.ax_dict['plot4']
        self.cb1 = self.cb2 = self.cb3 = None
        self.apply_plot_labels()

        self.roi_selectors = []
        for ax in [self.donor_im, self.acc_im, self.fret_im]:
            rs = RectangleSelector(
                ax, self._on_roi_select, useblit=False, button=[1],
                minspanx=5, minspany=5, interactive=False,
                props=dict(facecolor='cyan', edgecolor='cyan', alpha=0.25, fill=True)
            )
            self.roi_selectors.append(rs)

        self.canvas.draw_idle()

    def _on_roi_select(self, eclick, erelease):
        if eclick is None or erelease is None or any(v is None for v in (eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata)): return
        x1, y1 = float(eclick.xdata), float(eclick.ydata)
        x2, y2 = float(erelease.xdata), float(erelease.ydata)

        if abs(x2 - x1) < 3 or abs(y2 - y1) < 3: return
        ax = eclick.inaxes

        if ax == self.donor_im and self.registration_matrix is not None:
            try:
                pts = np.array([[x1, y1, 1.0], [x2, y1, 1.0], [x2, y2, 1.0], [x1, y2, 1.0]])
                mapped_pts = np.array([np.dot(self._registration_for_channels(), p)[:2] for p in pts])
                xmin, xmax = float(np.min(mapped_pts[:, 0])), float(np.max(mapped_pts[:, 0]))
                ymin, ymax = float(np.min(mapped_pts[:, 1])), float(np.max(mapped_pts[:, 1]))
            except Exception:
                xmin, xmax, ymin, ymax = min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)
        else:
            xmin, xmax, ymin, ymax = min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)

        if self.acceptor_data is not None:
            max_x, max_y = self.acceptor_data.frame_size[0], self.acceptor_data.frame_size[1]
            xmin, xmax = max(0.0, min(xmin, max_x)), max(0.0, min(xmax, max_x))
            ymin, ymax = max(0.0, min(ymin, max_y)), max(0.0, min(ymax, max_y))

        self.reset_es_selection()
        self.roi = (xmin, xmax, ymin, ymax)
        self.lbl_roi_status.setText(f"ROI Active:\nX: [{xmin:.1f}, {xmax:.1f}]\nY: [{ymin:.1f}, {ymax:.1f}]")
        
        if self.coords:
            keep_indices = [i for i, c in enumerate(self.coords) if xmin <= c[0] <= xmax and ymin <= c[1] <= ymax]
            self.coords = [self.coords[i] for i in keep_indices]
            if len(self.trajs_donor) >= len(keep_indices):
                self.trajs_donor = [self.trajs_donor[i] for i in keep_indices]
                self.trajs_acceptor = [self.trajs_acceptor[i] for i in keep_indices]
                self.trajs_fret = [self.trajs_fret[i] for i in keep_indices]

        self.refresh()

    def reset_roi(self):
        self.roi = None
        self.lbl_roi_status.setText("ROI: Full Frame (No Filter)")
        self.refresh()

    def apply_plot_labels(self):
        self.iplot_donor.set_ylabel(r"Donor (x10$^3$)")
        self.iplot_acceptor.set_ylabel(r"Acceptor (x10$^3$)")
        self.iplot_fret.set_ylabel(r"FRET (x10$^3$)")
        self.plot_efret.set_ylabel("Corrected E_FRET")
        self.plot_efret.set_xlabel("Frame number")

    def on_settings_changed(self):
        if self.source_fname:
            if not self.get_data(self.source_fname): return
            self.clear_plot(); self.update_ui_limits(); self.draw_images()

    def pick_file(self):
        if self.worker is not None and self.worker.isRunning(): return
        fname, _ = QFileDialog.getOpenFileName(self, "Open TIF Stack", filter="TIFF Files (*.tif *.tiff *.TIF *.TIFF)")
        if not fname: return
        old_name = self.params.name
        self.params.name = str(Path(fname).with_suffix(''))
        if not self.get_data(fname):
            self.params.name = old_name
            return
        loaded = (self.donor_data, self.acceptor_data, self.fret_data)
        if self.parent_suite: self.parent_suite.reset_entire_suite()
        self.donor_data, self.acceptor_data, self.fret_data = loaded
        self.source_fname = fname
        self.params.name = str(Path(fname).with_suffix(''))
        self.roi = None
        self.lbl_roi_status.setText("ROI: Full Frame (No Filter)")
        self.clear_plot()
        self.update_ui_limits()
        self.draw_images()

    def _build_pystachio_image(self, numpy_array, fname):
        img = images.ImageData()
        img.read(fname, self.params)
        img.pixel_data = numpy_array
        img.num_frames = numpy_array.shape[0]
        img.frame_size = [numpy_array.shape[2], numpy_array.shape[1]]
        img.has_mask, img.mask_data = True, np.ones((numpy_array.shape[1], numpy_array.shape[2]))
        return img

    def get_data(self, fname, orient_norm=None, temp_even=None):
        if not MODULES_LOADED: return
        try:
            orient = self.rb_orient_norm.isChecked() if orient_norm is None else orient_norm
            even = self.rb_temp_even.isChecked() if temp_even is None else temp_even
            
            raw_stack = tf.imread(fname)
            if raw_stack.ndim != 3 or raw_stack.shape[0] < 2 or raw_stack.shape[2] % 2:
                raise ValueError("Expected (frames, height, even width) TIFF with at least two frames.")
            if raw_stack.shape[0] % 2:
                raise ValueError("Odd frame count: trim the incomplete ALEX cycle before import.")
            mid_x = raw_stack.shape[2] // 2
            
            ch_left, ch_right = raw_stack[:, :, :mid_x], raw_stack[:, :, mid_x:]
            path_donor, path_acceptor_fret = (ch_left, ch_right) if orient else (ch_right, ch_left)
            
            if even:
                donor, fret, acc = path_donor[0::2], path_acceptor_fret[0::2], path_acceptor_fret[1::2]
            else:
                donor, fret, acc = path_donor[1::2], path_acceptor_fret[1::2], path_acceptor_fret[0::2]
                
            min_len = min(len(donor), len(acc), len(fret))
            donor_data = self._build_pystachio_image(np.ascontiguousarray(donor[:min_len]), fname)
            acceptor_data = self._build_pystachio_image(np.ascontiguousarray(acc[:min_len]), fname)
            fret_data = self._build_pystachio_image(np.ascontiguousarray(fret[:min_len]), fname)
            self.donor_data, self.acceptor_data, self.fret_data = donor_data, acceptor_data, fret_data
            self._loaded_orient_norm = bool(orient)
            return True
        except Exception as e:
            if orient_norm is None: 
                show_popup(self, "Dataset Import Failure", f"Failed to slice unified TIFF stack:\n{e}", critical=True)
            else:
                raise e

    def update_ui_limits(self):
        if self.donor_data is not None:
            self.frame_slider.setMinimum(0)
            self.frame_slider.setMaximum(self.donor_data.num_frames - 1)
            self.frame_slider.setValue(0); self.frame_slider.setEnabled(True)

    def on_frame_change(self, val):
        self.active_frame = val; self.draw_images()

    def draw_images(self):
        if self.donor_data is None: return
        
        for ax in [self.donor_im, self.acc_im, self.fret_im]:
            for img in list(ax.images): img.remove()
            for col in list(ax.collections): col.remove()
            retained = [a for selector in self.roi_selectors if selector.ax is ax for a in selector.artists]
            for patch in list(ax.patches):
                if patch not in retained: patch.remove()
            for line in list(ax.lines):
                if line not in retained: line.remove()
                
        f_idx = self.active_frame
        
        im1 = self.donor_im.imshow(self.donor_data[f_idx].as_image(), cmap='Greys_r')
        self.donor_im.set_xticks([]); self.donor_im.set_yticks([]); self.donor_im.set_title("Donor Channel Stack")
        im2 = self.acc_im.imshow(self.acceptor_data[f_idx].as_image(), cmap='Greys_r')
        self.acc_im.set_xticks([]); self.acc_im.set_yticks([]); self.acc_im.set_title("Acceptor Channel Stack")
        im3 = self.fret_im.imshow(self.fret_data[f_idx].as_image(), cmap='Greys_r')
        self.fret_im.set_xticks([]); self.fret_im.set_yticks([]); self.fret_im.set_title("FRET Channel Stack")
        
        self.cb1 = self.fig.colorbar(im1, ax=self.donor_im) if self.cb1 is None else self.cb1.update_normal(im1) or self.cb1
        self.cb2 = self.fig.colorbar(im2, ax=self.acc_im) if self.cb2 is None else self.cb2.update_normal(im2) or self.cb2
        self.cb3 = self.fig.colorbar(im3, ax=self.fret_im) if self.cb3 is None else self.cb3.update_normal(im3) or self.cb3
        
        self.frame_label.setText(f"Index: {f_idx} / {self.frame_slider.maximum()}")
        max_x, max_y = self.donor_data.frame_size[0], self.donor_data.frame_size[1]
        
        for coord in self.coords:
            ax_c, ay_c = float(coord[0]), float(coord[1])
            self.acc_im.scatter(ax_c, ay_c, marker='x', color='r')
            self.fret_im.scatter(ax_c, ay_c, marker='x', color='r')
            
            dx, dy = self._map_to_donor(ax_c, ay_c)
            if 0 <= dx < max_x and 0 <= dy < max_y:
                self.donor_im.scatter(dx, dy, marker='x', color='r')

        if self.roi is not None:
            xmin, xmax, ymin, ymax = self.roi
            w, h = xmax - xmin, ymax - ymin
            
            self.acc_im.add_patch(patches.Rectangle((xmin, ymin), w, h, linewidth=1.5, edgecolor='cyan', facecolor='none', linestyle='--'))
            self.fret_im.add_patch(patches.Rectangle((xmin, ymin), w, h, linewidth=1.5, edgecolor='cyan', facecolor='none', linestyle='--'))

            if self.registration_matrix is not None:
                try:
                    M_inv = cv2.invertAffineTransform(self._registration_for_channels())
                    M_inv_homo = np.vstack([M_inv, [0, 0, 1]])
                    pts = np.array([[xmin, ymin, 1.0], [xmax, ymin, 1.0], [xmax, ymax, 1.0], [xmin, ymax, 1.0]])
                    mapped_pts = np.dot(pts, M_inv_homo.T)[:, :2]
                    self.donor_im.add_patch(patches.Polygon(mapped_pts, closed=True, fill=False, edgecolor='cyan', linestyle='--', linewidth=1.5))
                except Exception:
                    self.donor_im.add_patch(patches.Rectangle((xmin, ymin), w, h, linewidth=1.5, edgecolor='cyan', facecolor='none', linestyle='--'))
            else:
                self.donor_im.add_patch(patches.Rectangle((xmin, ymin), w, h, linewidth=1.5, edgecolor='cyan', facecolor='none', linestyle='--'))

        self.canvas.draw_idle()

    def refresh(self):
        if self.donor_data is None: return
        self.draw_images()
        self.iplot_donor.clear(); self.iplot_acceptor.clear(); self.iplot_fret.clear(); self.plot_efret.clear()
        self.apply_plot_labels()
        
        if len(self.trajs_donor) == 0:
            self.canvas.draw_idle(); return

        num_frames = self.donor_data.num_frames
        max_display = min(len(self.trajs_donor), 50)
        
        for traj in range(max_display):
            tdonor = np.array(self.trajs_donor[traj].intensity)
            tacc = np.array(self.trajs_acceptor[traj].intensity)
            tfret = np.array(self.trajs_fret[traj].intensity)

            if self.ck_yes.isChecked() and num_frames >= 5:
                try:
                    # The supplied backend leaves its final output sample unset.
                    # Pad once, then remove that extra output: retain all real frames.
                    with np.errstate(divide='ignore', invalid='ignore'):
                        candidate = [np.asarray(postprocessing.chung_kennedy_filter(
                            np.r_[x, x[-1]], 5, 1)[0][:-1], dtype=float)
                                     for x in (tdonor, tacc, tfret)]
                    if all(x.shape == (num_frames,) and np.isfinite(x).all() for x in candidate):
                        tdonor, tacc, tfret = candidate
                except (ValueError, IndexError, FloatingPointError):
                    pass  # Display-only; retain all three raw traces together.

            self.iplot_donor.plot(tdonor[:num_frames] / 10**3, alpha=0.7)
            self.iplot_acceptor.plot(tacc[:num_frames] / 10**3, alpha=0.7)
            self.iplot_fret.plot(tfret[:num_frames] / 10**3, alpha=0.7)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                e, _ = corrected_es(tdonor, tacc, tfret, self.alpha, self.gamma)
                self.plot_efret.plot(e[:num_frames], alpha=0.7)
                
        self.canvas.draw_idle()

    def clear_plot(self):
        self._selection_backup = None
        self.coords, self.trajs_donor, self.trajs_acceptor, self.trajs_fret = [], [], [], []
        self.refresh()

    def _extract_trajectories(self, track_params):
        if not self.coords: return
        max_x, max_y = self.donor_data.frame_size[0], self.donor_data.frame_size[1]
        valid_acceptor_coords, valid_donor_coords = [], []
        
        for ax, ay in self.coords:
            dx, dy = self._map_to_donor(float(ax), float(ay))
            if not (np.isfinite(dx) and np.isfinite(dy) and np.isfinite(ax) and np.isfinite(ay)): continue
            if not (0 <= dx < max_x and 0 <= dy < max_y and 0 <= ax < max_x and 0 <= ay < max_y): continue
                
            valid_acceptor_coords.append([ax, ay])
            valid_donor_coords.append([dx, dy])
            
        self.coords = valid_acceptor_coords
        if not self.coords: 
            self.trajs_donor, self.trajs_acceptor, self.trajs_fret = [], [], []
            return
            
        channel_spots_d, channel_spots_a, channel_spots_f = [], [], []
        for f in range(self.donor_data.num_frames):
            sd, sa, sf = spots.Spots(frame=f), spots.Spots(frame=f), spots.Spots(frame=f)
            
            sd.set_positions(valid_donor_coords); sd.num_spots = len(valid_donor_coords)
            sa.set_positions(self.coords); sa.num_spots = len(self.coords)
            sf.set_positions(self.coords); sf.num_spots = len(self.coords)
            
            sd.get_spot_intensities(self.donor_data[f].as_image(), track_params)
            sa.get_spot_intensities(self.acceptor_data[f].as_image(), track_params)
            sf.get_spot_intensities(self.fret_data[f].as_image(), track_params)
            
            channel_spots_d.append(sd); channel_spots_a.append(sa); channel_spots_f.append(sf)
            
        self.trajs_donor = _fixed_traces(channel_spots_d, valid_donor_coords)
        self.trajs_acceptor = _fixed_traces(channel_spots_a, self.coords)
        self.trajs_fret = _fixed_traces(channel_spots_f, self.coords)

    def on_press(self, event):
        if event.inaxes in [self.donor_im, self.acc_im, self.fret_im]:
            self._press_pos, self._press_axes = (event.xdata, event.ydata), event.inaxes
        else:
            self._press_pos = self._press_axes = None

    def on_release(self, event):
        try:
            self._on_release_impl(event)
        except Exception as exc:
            show_popup(self, "Spot extraction failed", str(exc), critical=True)

    def _on_release_impl(self, event):
        if not self.donor_data or not MODULES_LOADED or self._press_pos is None or event.inaxes != self._press_axes: return
        if event.xdata is None or event.ydata is None or None in self._press_pos: return
        mode = getattr(self.toolbar, 'mode', ''); mode = mode.value if hasattr(mode, 'value') else mode
        if mode != "": return
            
        px, py = self._press_pos
        if np.hypot(event.xdata - px, event.ydata - py) >= 3.0: return
        
        ix, iy = float(event.xdata), float(event.ydata)
        max_x, max_y = self.donor_data.frame_size[0], self.donor_data.frame_size[1]
        
        if event.inaxes == self.donor_im and self.registration_matrix is not None:
            mapped = np.dot(self._registration_for_channels(), np.array([ix, iy, 1.0]))[:2]
            ix, iy = float(mapped[0]), float(mapped[1])
            
        if not (0 <= ix < max_x and 0 <= iy < max_y): return
        
        tmp_spots = spots.Spots(frame=0)
        tmp_spots.set_positions([[ix, iy]]); tmp_spots.num_spots = 1
        tmp_spots.refine_centres(self.acceptor_data[self.active_frame], self.params)
        
        if len(tmp_spots.positions) > 0:
            cx, cy = float(tmp_spots.positions[0][0]), float(tmp_spots.positions[0][1])
            if self.roi is not None:
                rmin_x, rmax_x, rmin_y, rmax_y = self.roi
                if not (rmin_x <= cx <= rmax_x and rmin_y <= cy <= rmax_y):
                    show_popup(self, "Out of Bounds", "Selected spot lies outside the active ROI bounds.")
                    return
            self.reset_es_selection()
            self.coords.append([cx, cy])
        self._extract_trajectories(self.params)
        self.refresh()

    def autodetect_spots(self):
        if not self.donor_data or not MODULES_LOADED: return
        if self.worker is not None and self.worker.isRunning(): return
        track_params = self.parent_suite.main_analysis_tab.params if self.parent_suite and hasattr(self.parent_suite, 'main_analysis_tab') else self.params
        
        try:
            _registration(self.registration_matrix)
        except ValueError as exc:
            show_popup(self, "Registration error", str(exc), critical=True)
            return
        self._worker_snapshot = self.selection_snapshot()
        self.btn_autodetect.setEnabled(False)
        self.worker = FullAnalysisWorker(self.acceptor_data, self.donor_data, self.fret_data, self.active_frame, track_params, self._registration_for_channels(), roi=self.roi)
        self.worker.progress.connect(self.btn_autodetect.setText)
        self.worker.result_ready.connect(self._on_analysis_finished)
        self.worker.start()

    def _on_analysis_finished(self, trajs_d, trajs_a, trajs_f, coords, error):
        self.btn_autodetect.setEnabled(True); self.btn_autodetect.setText("Autodetect spots")
        if self._worker_snapshot != self.selection_snapshot(): return
        if error:
            show_popup(self, "Autodetect Error", f"Error during spot processing:\n{error}", critical=True)
            return
        if not coords:
            show_popup(self, "Autodetect", "No valid spots found in the acceptor channel (or active ROI).")
            return

        self._selection_backup = None
        if not (len(coords) == len(trajs_d) == len(trajs_a) == len(trajs_f)):
            show_popup(self, "Trajectory error", "Channel trajectory counts do not match.", critical=True)
            return
        self.coords, self.trajs_donor, self.trajs_acceptor, self.trajs_fret = coords, trajs_d, trajs_a, trajs_f
        self.refresh()

    def open_stoic_window(self):
        if not self.trajs_donor:
            show_popup(self, "Error", "No spots/trajectories available.", critical=True)
            return
        self.stoic_win = StoichiometryWindow(self); self.stoic_win.show()

    def show_histogram(self):
        if not self.trajs_donor:
            show_popup(self, "Error", "No spots available to build histogram.", critical=True)
            return
            
        win_size = self.avg_window_spin.value()
        direction = self.avg_direction_combo.currentText()
        max_f = self.donor_data.num_frames if self.donor_data else 0
        start_idx, end_idx = self.get_avg_bounds(self.active_frame, win_size, direction, max_f)
        
        e_vals = []
        for i in range(len(self.trajs_donor)):
            d_avg = np.sum(self.trajs_donor[i].intensity[start_idx:end_idx])
            fr_avg = np.sum(self.trajs_fret[i].intensity[start_idx:end_idx])
            
            if (d_avg + fr_avg) > 0:
                e_avg, _ = corrected_es(d_avg, 0., fr_avg, self.alpha, self.gamma)
                if self.e_min <= e_avg <= self.e_max:
                    e_vals.append(e_avg)
                
        if not e_vals: return
            
        rootname = str(Path(self.source_fname).parent) if self.params.name else os.path.dirname(self.source_fname)
        self.hist_win = HistogramWindow(self, e_vals, self.active_frame, os.path.join(rootname, f"EFRET_hist_frame_{self.active_frame}.png"))
        self.hist_win.show()

    def save_data(self):
        if not self.source_fname or not self.trajs_donor: return
        name = self.outname_box.text().strip()
        if not name or Path(name).name != name or name in ('.', '..'):
            show_popup(self, "Invalid filename", "Enter a filename without a directory.", critical=True)
            return
        destination = Path(self.source_fname).parent / name
        if destination.suffix.lower() in ('.tif', '.tiff'):
            show_popup(self, "Invalid filename", "Export text data with a .dat, .txt or .tsv extension.", critical=True)
            return
        try:
            with tempfile.TemporaryDirectory(dir=destination.parent, prefix='.smfret_export_') as temp:
                output = Path(temp) / name
                with output.open('w', newline='', encoding='utf-8') as handle:
                    handle.write(f'# alpha={self.alpha}; gamma={self.gamma}; beta=1; direct excitation=0\n')
                    handle.write('# E=(F-alpha*D)/(F-alpha*D+gamma*D); S=(F-alpha*D+gamma*D)/(F-alpha*D+gamma*D+A)\n')
                    handle.write('# Background-subtracted input; raw intensities unchanged; CK display-only; undefined ratios=NaN\n')
                    writer = csv.writer(handle, delimiter='\t')
                    writer.writerow(('Frame','Traj_ID','X_Coord','Y_Coord','Donor_Int','Acceptor_Int','FRET_Int','E_FRET','S'))
                    if not (len(self.coords) == len(self.trajs_donor) == len(self.trajs_acceptor) == len(self.trajs_fret)):
                        raise ValueError('Coordinate and channel trajectory counts differ.')
                    for tid, (d, a, f) in enumerate(zip(self.trajs_donor,self.trajs_acceptor,self.trajs_fret)):
                        x, y = self.coords[tid]
                        if not (len(d.intensity) == len(a.intensity) == len(f.intensity) == self.donor_data.num_frames):
                            raise ValueError('Channel trace lengths differ from the dataset.')
                        e, st = corrected_es(d.intensity,a.intensity,f.intensity,self.alpha,self.gamma)
                        for frame in range(len(e)):
                            writer.writerow((frame,tid,x,y,d.intensity[frame],a.intensity[frame],f.intensity[frame],e[frame],st[frame]))
                output.replace(destination)
            show_popup(self,"Export Success",f'Exported to: {destination}')
        except Exception as exc:
            show_popup(self,"Export Failure",str(exc),critical=True)

    def batch_analyse_all(self):
        if self.batch_worker is not None and self.batch_worker.isRunning(): return
        if not self.source_fname:
            show_popup(self, "Error", "Please load a file first to set the directory.", critical=True)
            return

        directory = os.path.dirname(self.source_fname)
        pattern = "**/*.tif*" if self.chk_subdirs.isChecked() else "*.tif*"
        tif_files = glob.glob(os.path.join(directory, pattern), recursive=self.chk_subdirs.isChecked())
        
        skip_suffixes = ("_cropped.tif", "_left.tif", "_right.tif", "_donor.tif", "_acceptor.tif", "_fret.tif", 
                         "_cropped.tiff", "_left.tiff", "_right.tiff", "_donor.tiff", "_acceptor.tiff", "_fret.tiff")
        files_to_process = [f for f in tif_files if not f.endswith(skip_suffixes)]

        if not files_to_process:
            show_popup(self, "Batch Analysis", "No valid TIF files found to process.")
            return

        try:
            _registration(self.registration_matrix)
        except ValueError as exc:
            show_popup(self, "Registration error", str(exc), critical=True)
            return
        self.orig_state = (self.source_fname, self.donor_data, self.acceptor_data, self.fret_data, list(self.coords), list(self.trajs_donor), list(self.trajs_acceptor), list(self.trajs_fret))
        self.btn_batch_analyse.setEnabled(False)
        self.btn_batch_analyse.setText("Processing...")
        
        self.batch_progress.setMaximum(len(files_to_process))
        self.batch_progress.setValue(0)
        self.batch_progress.show()
        
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        
        track_params = self.parent_suite.main_analysis_tab.params if self.parent_suite and hasattr(self.parent_suite, 'main_analysis_tab') else self.params
        win_size = self.avg_window_spin.value()
        direction = self.avg_direction_combo.currentText()
        orient_norm = self.rb_orient_norm.isChecked()
        temp_even = self.rb_temp_even.isChecked()
        
        self.batch_worker = BatchWorker(self, files_to_process, directory, track_params, win_size, direction, orient_norm, temp_even)
        self.batch_worker.progress.connect(self.batch_progress.setValue)
        self.batch_worker.result_ready.connect(self._on_batch_finished)
        self.batch_worker.error.connect(self._on_batch_error)
        self.batch_worker.start()

    def _on_batch_finished(self, directory):
        self._restore_batch_state()
        show_popup(self, "Batch Analysis Completed", f"Successfully exported consolidated files to:\n{directory}")

    def _on_batch_error(self, err_trace):
        self._restore_batch_state()
        show_popup(self, "Pipeline Failure", f"Batch tracking error:\n\n{err_trace}", critical=True)
        
    def _restore_batch_state(self):
        self.btn_batch_analyse.setEnabled(True)
        self.btn_batch_analyse.setText("Analyse all")
        self.batch_progress.hide()
        QApplication.restoreOverrideCursor()
        
        # Batch state is isolated; there is no live state to restore.
        if self.source_fname: self.params.name = str(Path(self.source_fname).with_suffix(""))
        self.refresh()

    def closeEvent(self, event):
        if any(worker is not None and worker.isRunning() for worker in (self.worker, self.batch_worker)):
            show_popup(self, "Analysis running", "Wait for analysis to finish before closing this tab.")
            event.ignore()
        else:
            event.accept()
