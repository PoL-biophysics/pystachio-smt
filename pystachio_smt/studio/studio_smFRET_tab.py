"""Manual smFRET/ALEX tab.

Intensity definitions: D = donor emission on donor excitation, F = acceptor
emission on donor excitation, A = acceptor emission on acceptor excitation.

For background-subtracted intensities, alpha subtracts donor leakage from F,
and gamma scales D. E = (F-alpha*D)/(F-alpha*D+gamma*D).
S_gamma = (F-alpha*D+gamma*D)/(F-alpha*D+gamma*D+A).
The full ALEX beta (excitation balance) factor is assumed to be 1; direct
acceptor excitation under donor illumination is not corrected here. Those
calibrations must be added separately if needed for an experiment.

Raw intensity traces and files are not changed by the corrections. Undefined
ratios are NaN, never silently assigned an efficiency of zero.
"""

import copy
import csv
import shutil
from contextlib import ExitStack
from pathlib import Path
import tempfile
import traceback

import numpy as np
import tifffile as tf
import cv2
import matplotlib

matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import RectangleSelector

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider,
    QLineEdit, QLabel, QRadioButton, QGroupBox, QFileDialog, QButtonGroup,
    QGridLayout, QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox, QProgressBar,
    QScrollArea, QSizePolicy, QLayout,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

import images
import spots
import parameters
import postprocessing
import trajectories
import tracking
from studio.studio_helpers import show_popup


# Public compatibility flag retained for the surrounding studio package.
MODULES_LOADED = True

OUTPUT_COLUMNS = (
    'Image_Filename', 'Traj_ID', 'X_Coord', 'Y_Coord', 'Frame',
    'E_FRET', 'S', 'Donor_Int', 'Acc_Int', 'FRET_Int',
)
BATCH_NAMES = (
    'consolidated_smFRET_raw_all_frames.txt',
    'consolidated_smFRET_filtered_all_frames.txt',
    'consolidated_smFRET_filtered_first_frame.txt',
)


def _ratio(numerator, denominator):
    """Compute ratios on scalars/arrays; non-positive/nonfinite totals -> NaN."""
    n, den = np.broadcast_arrays(np.asarray(numerator, dtype=float),
                                 np.asarray(denominator, dtype=float))
    result = np.full(n.shape, np.nan, dtype=float)
    valid = np.isfinite(n) & np.isfinite(den) & (den > 0)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        np.divide(n, den, out=result, where=valid)
    result[~np.isfinite(result)] = np.nan
    return float(result) if result.ndim == 0 else result


def calculate_efret(donor, fret, alpha=0.0, gamma=1.0):
    """Leakage/gamma-corrected efficiency; defaults reproduce F/(F+D)."""
    d = np.asarray(donor, dtype=float)
    f = np.asarray(fret, dtype=float)
    corrected_f = f - alpha * d
    return _ratio(corrected_f, corrected_f + gamma * d)


def calculate_stoichiometry(donor, acceptor, fret, alpha=0.0, gamma=1.0):
    """Gamma-corrected ALEX S (with beta=1 and no direct-excitation correction)."""
    d = np.asarray(donor, dtype=float)
    f = np.asarray(fret, dtype=float)
    a = np.asarray(acceptor, dtype=float)
    donor_exc_total = f - alpha * d + gamma * d
    return _ratio(donor_exc_total, donor_exc_total + a)


def average_bounds(current_frame, window_size, direction, max_frames):
    """Return a nonempty half-open averaging interval within available frames."""
    if max_frames <= 0:
        raise ValueError('Cannot average a dataset with no frames.')
    pos = max(0, min(int(current_frame), max_frames - 1))
    size = max(1, int(window_size))
    if direction == 'Centered':
        start = pos - size // 2
    elif direction == 'Trailing':
        start = pos - size + 1
    else:
        start = pos
    return max(0, start), min(max_frames, max(1, start + size))


def split_alex_stack(raw_stack, orient_norm=True, temp_even=True):
    """Validate and split a 2-channel, time-interleaved TIFF stack."""
    raw = np.asarray(raw_stack)
    if raw.ndim != 3 or raw.shape[0] < 2 or raw.shape[1] < 1 or raw.shape[2] < 2:
        raise ValueError('Expected a 3-D (frames, height, width) TIFF with at least 2 frames.')
    if raw.shape[2] % 2:
        raise ValueError('TIFF width must be even to split donor and acceptor halves.')
    mid = raw.shape[2] // 2
    left, right = raw[:, :, :mid], raw[:, :, mid:]
    donor_half, acceptor_half = (left, right) if orient_norm else (right, left)
    if temp_even:
        donor, fret, acceptor = donor_half[0::2], acceptor_half[0::2], acceptor_half[1::2]
    else:
        donor, fret, acceptor = donor_half[1::2], acceptor_half[1::2], acceptor_half[0::2]
    count = min(len(donor), len(fret), len(acceptor))
    if count == 0:
        raise ValueError('No complete donor/FRET/acceptor ALEX triplets found.')
    return tuple(np.ascontiguousarray(x[:count]) for x in (donor, acceptor, fret))


def _build_image(array, fname, image_params):
    # Preserve PySTACHIO's ImageData initialization, including its metadata.
    img = images.ImageData()
    img.read(str(fname), image_params)
    img.pixel_data = array
    img.num_frames = array.shape[0]
    img.frame_size = [array.shape[2], array.shape[1]]
    img.has_mask = True
    img.mask_data = np.ones((array.shape[1], array.shape[2]))
    return img


def load_dataset(fname, image_params, orient_norm=True, temp_even=True):
    donor, acceptor, fret = split_alex_stack(tf.imread(str(fname)), orient_norm, temp_even)
    return tuple(_build_image(x, fname, image_params) for x in (donor, acceptor, fret))


def _affine_matrix(matrix):
    if matrix is None:
        return None
    m = np.asarray(matrix, dtype=float)
    if m.shape not in ((2, 3), (3, 3)) or not np.all(np.isfinite(m)):
        raise ValueError('Registration matrix must be finite and have shape (2,3) or (3,3).')
    m = m[:2, :]
    if abs(np.linalg.det(m[:, :2])) < 1e-12:
        raise ValueError('Registration matrix cannot be inverted (singular transform).')
    return m


def map_to_donor(x, y, matrix):
    """Registration maps donor -> acceptor; invert to obtain donor coordinates."""
    m = _affine_matrix(matrix)
    if m is None:
        return float(x), float(y)
    mapped = cv2.invertAffineTransform(m) @ np.array([x, y, 1.0])
    if not np.all(np.isfinite(mapped)):
        raise ValueError('Registration mapped to nonfinite donor coordinates.')
    return float(mapped[0]), float(mapped[1])


def valid_coordinate_pairs(coords, donor_data, registration_matrix, roi=None):
    """Filter in acceptor coordinates, then validate both transformed coordinates."""
    max_x, max_y = donor_data.frame_size
    acceptor_coords, donor_coords = [], []
    _affine_matrix(registration_matrix)  # Check even when no spots are detected.
    for x, y in coords:
        x, y = float(x), float(y)
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        if roi is not None:
            xmin, xmax, ymin, ymax = roi
            if not (xmin <= x <= xmax and ymin <= y <= ymax):
                continue
        dx, dy = map_to_donor(x, y, registration_matrix)
        if 0 <= x < max_x and 0 <= y < max_y and 0 <= dx < max_x and 0 <= dy < max_y:
            acceptor_coords.append([x, y])
            donor_coords.append([dx, dy])
    return acceptor_coords, donor_coords


def extract_trajectories(donor_data, acceptor_data, fret_data, coords,
                         track_params, registration_matrix=None, roi=None,
                         interrupted=None, progress=None):
    """Extract trajectories without ever touching a tab or other GUI widget."""
    a_coords, d_coords = valid_coordinate_pairs(coords, donor_data, registration_matrix, roi)
    if not a_coords:
        return [], [], [], []
    sd_all, sa_all, sf_all = [], [], []
    for frame in range(donor_data.num_frames):
        if interrupted is not None and interrupted():
            raise InterruptedError('Analysis cancelled')
        if progress is not None and (frame % 5 == 0 or frame == donor_data.num_frames - 1):
            progress(f'Extracting frame {frame + 1}/{donor_data.num_frames}...')
        sd, sa, sf = spots.Spots(frame=frame), spots.Spots(frame=frame), spots.Spots(frame=frame)
        for spot_set, coords_frame, data in (
            (sd, d_coords, donor_data), (sa, a_coords, acceptor_data),
            (sf, a_coords, fret_data),
        ):
            spot_set.set_positions(coords_frame)
            spot_set.num_spots = len(coords_frame)
            spot_set.get_spot_intensities(data[frame].as_image(), track_params)
        sd_all.append(sd)
        sa_all.append(sa)
        sf_all.append(sf)
    return (trajectories.build_trajectories(sd_all, track_params),
            trajectories.build_trajectories(sa_all, track_params),
            trajectories.build_trajectories(sf_all, track_params), a_coords)


def selection_metrics(d_trace, a_trace, f_trace, start, end, alpha, gamma):
    """Ratio of *summed* intensities over the chosen averaging window."""
    d = np.sum(np.asarray(d_trace, dtype=float)[start:end])
    a = np.sum(np.asarray(a_trace, dtype=float)[start:end])
    f = np.sum(np.asarray(f_trace, dtype=float)[start:end])
    return (calculate_efret(d, f, alpha, gamma),
            calculate_stoichiometry(d, a, f, alpha, gamma))


class FullAnalysisWorker(QThread):
    result_ready = pyqtSignal(object, object, object, object, object)
    progress = pyqtSignal(str)

    def __init__(self, acceptor_data, donor_data, fret_data, active_frame,
                 track_params, registration_matrix, roi=None):
        super().__init__()
        self.acceptor_data = acceptor_data
        self.donor_data = donor_data
        self.fret_data = fret_data
        self.active_frame = active_frame
        self.track_params = track_params
        self.registration_matrix = registration_matrix
        self.roi = roi

    def run(self):
        try:
            self.progress.emit('Detecting spots...')
            found = tracking.track_frame(self.acceptor_data[self.active_frame],
                                         self.active_frame, self.track_params)
            coords = [] if found is None else list(found.positions)
            if not coords:
                self.result_ready.emit([], [], [], [], None)
                return
            result = extract_trajectories(
                self.donor_data, self.acceptor_data, self.fret_data, coords,
                self.track_params, self.registration_matrix, self.roi,
                interrupted=self.isInterruptionRequested, progress=self.progress.emit,
            )
            self.result_ready.emit(*result, None)
        except InterruptedError:
            self.result_ready.emit([], [], [], [], 'Spot detection cancelled.')
        except Exception as exc:
            self.result_ready.emit([], [], [], [], str(exc))


class BatchWorker(QThread):
    result_ready = pyqtSignal(object)
    error_ready = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, files_to_process, directory, image_params, track_params,
                 settings):
        super().__init__()
        self.files_to_process = list(files_to_process)
        self.directory = Path(directory)
        self.image_params = image_params
        self.track_params = track_params
        self.settings = dict(settings)  # Snapshot taken on the GUI thread.

    def run(self):
        try:
            self.result_ready.emit(self.process_files())
        except InterruptedError:
            self.error_ready.emit('Batch analysis cancelled; existing output files unchanged.')
        except Exception:
            self.error_ready.emit(traceback.format_exc())

    def process_files(self):
        s = self.settings
        failures = []
        processed = 0
        no_spots = 0
        total_molecules = 0
        directory = self.directory
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(directory), prefix='.smfret_work_') as staging:
            staging = Path(staging)
            with ExitStack() as stack:
                outputs = [stack.enter_context((staging / name).open('w', newline='', encoding='utf-8'))
                           for name in BATCH_NAMES]
                writers = [csv.writer(handle, delimiter='\t') for handle in outputs]
                for writer in writers:
                    writer.writerow(OUTPUT_COLUMNS)
                for idx, fname in enumerate(self.files_to_process):
                    if self.isInterruptionRequested():
                        raise InterruptedError()
                    try:
                        # Stage each file separately. A failed file cannot leave partially
                        # exported rows in the consolidated results.
                        with ExitStack() as per_file_stack:
                            buffers = [per_file_stack.enter_context(tempfile.SpooledTemporaryFile(
                                mode='w+', max_size=1048576, newline='', encoding='utf-8'))
                                for _ in BATCH_NAMES]
                            per_file_writers = [csv.writer(handle, delimiter='\t')
                                                for handle in buffers]
                            image_params = copy.deepcopy(self.image_params)
                            image_params.name = str(Path(fname).with_suffix(''))
                            donor, acceptor, fret = load_dataset(
                                fname, image_params, s['orient_norm'], s['temp_even'])
                            frame = min(s['active_frame'], donor.num_frames - 1)
                            found = tracking.track_frame(acceptor[frame], frame, self.track_params)
                            coords = [] if found is None else list(found.positions)
                            if not coords:
                                no_spots += 1
                                processed += 1
                                continue
                            d_trajs, a_trajs, f_trajs, coords = extract_trajectories(
                                donor, acceptor, fret, coords, self.track_params,
                                s['registration_matrix'], s['roi'],
                                interrupted=self.isInterruptionRequested)
                            if not coords:
                                no_spots += 1
                                processed += 1
                                continue
                            if not (len(coords) == len(d_trajs) == len(a_trajs) == len(f_trajs)):
                                raise ValueError('PySTACHIO returned mismatched trajectory counts.')
                            start, end = average_bounds(frame, s['win_size'], s['direction'],
                                                        donor.num_frames)
                            source = str(Path(fname).relative_to(directory))
                            for i, (td, ta, tfr) in enumerate(zip(d_trajs, a_trajs, f_trajs)):
                                if self.isInterruptionRequested():
                                    raise InterruptedError()
                                d_trace, a_trace, f_trace = (
                                    np.asarray(t.intensity, dtype=float) for t in (td, ta, tfr))
                                if min(len(d_trace), len(a_trace), len(f_trace)) < donor.num_frames:
                                    raise ValueError('Trajectory shorter than dataset.')
                                e_mean, s_mean = selection_metrics(
                                    d_trace, a_trace, f_trace, start, end, s['alpha'], s['gamma'])
                                selected = (np.isfinite(e_mean) and np.isfinite(s_mean)
                                            and s['e_min'] <= e_mean <= s['e_max']
                                            and s['s_min'] <= s_mean <= s['s_max'])
                                e_frame = calculate_efret(d_trace, f_trace, s['alpha'], s['gamma'])
                                s_frame = calculate_stoichiometry(
                                    d_trace, a_trace, f_trace, s['alpha'], s['gamma'])
                                x, y = coords[i]
                                for k in range(donor.num_frames):
                                    row = (source, total_molecules + i,
                                           f'{x:.3f}', f'{y:.3f}', k,
                                           f'{e_frame[k]:.6g}', f'{s_frame[k]:.6g}',
                                           f'{d_trace[k]:.6g}', f'{a_trace[k]:.6g}',
                                           f'{f_trace[k]:.6g}')
                                    per_file_writers[0].writerow(row)
                                    if selected:
                                        per_file_writers[1].writerow(row)
                                        if k == 0:
                                            per_file_writers[2].writerow(row)
                            if self.isInterruptionRequested():
                                raise InterruptedError()
                            for staged_file, destination in zip(buffers, outputs):
                                staged_file.seek(0)
                                shutil.copyfileobj(staged_file, destination)
                            total_molecules += len(coords)
                            processed += 1
                    except InterruptedError:
                        raise
                    except Exception as exc:
                        failures.append((str(fname), str(exc)))
                    finally:
                        self.progress.emit(idx + 1)
                # All writers flushed/closed before files are published.
            with (staging / 'consolidated_smFRET_summary.txt').open('w', encoding='utf-8') as out:
                out.write('smFRET Batch Analysis Summary\n=============================\n')
                out.write(f'Discovered: {len(self.files_to_process)}\n')
                out.write(f'Successfully processed: {processed}\n')
                out.write(f'Processed without spots: {no_spots}\n')
                out.write(f'Failed: {len(failures)}\n')
                out.write(f'Trajectories exported: {total_molecules}\n\n')
                out.write(f"E_FRET range: {s['e_min']} to {s['e_max']}\n")
                out.write(f"Stoichiometry range: {s['s_min']} to {s['s_max']}\n")
                out.write(f"Alpha: {s['alpha']}    Gamma: {s['gamma']}\n")
                out.write('E=(F-alpha*D)/(F-alpha*D+gamma*D)\n')
                out.write('S=(F-alpha*D+gamma*D)/(F-alpha*D+gamma*D+A); beta=1, no direct-excitation correction\n')
                out.write('Invalid ratios: NaN (excluded from filtering); raw intensities unchanged\n')
                out.write(f"Averaging: {s['win_size']} frames ({s['direction']})\n")
                out.write(f"ROI: {s['roi']}\n")
                out.write('Output format: tab-separated values (.txt)\n')
                out.write('Display-only Chung-Kennedy smoothing is not applied to exports.\n')
            with (staging / 'consolidated_smFRET_errors.txt').open('w', encoding='utf-8') as out:
                for fname, reason in failures:
                    out.write(f'{fname}\t{reason}\n')
            if self.isInterruptionRequested():
                raise InterruptedError()
            for name in (*BATCH_NAMES, 'consolidated_smFRET_summary.txt',
                         'consolidated_smFRET_errors.txt'):
                (staging / name).replace(directory / name)
        return {'directory': str(directory), 'processed': processed,
                'failed': len(failures), 'no_spots': no_spots,
                'molecules': total_molecules}


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
        self.setWindowTitle('Select Stoichiometry (S vs E)')
        self.resize(700, 600)
        self.snapshot = parent_tab.selection_snapshot()
        self.indices = parent_tab.roi_indices()  # Do not restrict to an older S/E selection.
        self.e_average, self.s_average = [], []
        all_e, all_s = [], []
        start, end = parent_tab.get_avg_bounds(
            parent_tab.active_frame, parent_tab.avg_window_spin.value(),
            parent_tab.avg_direction_combo.currentText(), parent_tab.donor_data.num_frames)
        for i in self.indices:
            d = np.asarray(parent_tab.all_trajs_donor[i].intensity, dtype=float)
            a = np.asarray(parent_tab.all_trajs_acceptor[i].intensity, dtype=float)
            f = np.asarray(parent_tab.all_trajs_fret[i].intensity, dtype=float)
            e = calculate_efret(d, f, parent_tab.alpha, parent_tab.gamma)
            s = calculate_stoichiometry(d, a, f, parent_tab.alpha, parent_tab.gamma)
            mask = np.isfinite(e) & np.isfinite(s)
            all_e.extend(e[mask])
            all_s.extend(s[mask])
            e_avg, s_avg = selection_metrics(d, a, f, start, end,
                                             parent_tab.alpha, parent_tab.gamma)
            self.e_average.append(e_avg)
            self.s_average.append(s_avg)
        layout = QVBoxLayout(self)
        self.fig = Figure(figsize=(6, 5))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        self.ax.scatter(all_e, all_s, c='blue', alpha=0.15, s=5,
                        edgecolors='none', label='Valid frames')
        self.ax.scatter(self.e_average, self.s_average, c='red', alpha=0.9,
                        s=30, edgecolors='k', label='Trajectory averages')
        self.ax.set_xlim(-0.1, 1.1)
        self.ax.set_ylim(-0.1, 1.1)
        self.ax.set_xlabel('Corrected FRET efficiency (E)')
        self.ax.set_ylabel('Corrected stoichiometry (S)')
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.ax.legend(loc='upper right')
        self.rect = patches.Rectangle((0, 0), 1, 1, fill=False,
                                      edgecolor='red', linewidth=2)
        self.ax.add_patch(self.rect)
        controls = QGridLayout()
        self.e_min_sl = QSlider(Qt.Orientation.Horizontal)
        self.e_max_sl = QSlider(Qt.Orientation.Horizontal)
        self.s_min_sl = QSlider(Qt.Orientation.Horizontal)
        self.s_max_sl = QSlider(Qt.Orientation.Horizontal)
        for label, slider, value, row in (
            ('E min:', self.e_min_sl, parent_tab.e_min, 0),
            ('E max:', self.e_max_sl, parent_tab.e_max, 1),
            ('S min:', self.s_min_sl, parent_tab.s_min, 2),
            ('S max:', self.s_max_sl, parent_tab.s_max, 3),
        ):
            slider.setRange(0, 100)
            slider.setValue(int(round(value * 100)))
            slider.valueChanged.connect(self.update_rect)
            controls.addWidget(QLabel(label), row, 0)
            controls.addWidget(slider, row, 1)
        layout.addLayout(controls)
        self.btn_confirm = QPushButton('Confirm selection')
        self.btn_confirm.clicked.connect(self.confirm_selection)
        layout.addWidget(self.btn_confirm)
        self.update_rect()

    def update_rect(self):
        if self.e_min_sl.value() > self.e_max_sl.value():
            self.e_min_sl.setValue(self.e_max_sl.value())
        if self.s_min_sl.value() > self.s_max_sl.value():
            self.s_min_sl.setValue(self.s_max_sl.value())
        e_min, e_max = self.e_min_sl.value()/100, self.e_max_sl.value()/100
        s_min, s_max = self.s_min_sl.value()/100, self.s_max_sl.value()/100
        self.rect.set_xy((e_min, s_min))
        self.rect.set_width(e_max-e_min)
        self.rect.set_height(s_max-s_min)
        self.canvas.draw_idle()

    def confirm_selection(self):
        tab = self.parent_tab
        if self.snapshot != tab.selection_snapshot():
            show_popup(self, 'Selection out of date',
                       'Dataset or analysis settings changed. Reopen the selection window.')
            return
        e_min, e_max = self.e_min_sl.value()/100, self.e_max_sl.value()/100
        s_min, s_max = self.s_min_sl.value()/100, self.s_max_sl.value()/100
        matches = [i for i, e, s in zip(self.indices, self.e_average, self.s_average)
                   if np.isfinite(e) and np.isfinite(s)
                   and e_min <= e <= e_max and s_min <= s <= s_max]
        if not matches:
            show_popup(self, 'Selection empty',
                       'No valid trajectories fall within this rectangle; adjust the limits.')
            return
        tab.e_min, tab.e_max, tab.s_min, tab.s_max = e_min, e_max, s_min, s_max
        tab.selection_enabled = True
        tab.apply_selection()
        tab.refresh()
        self.close()

    def closeEvent(self, event):
        self.canvas.deleteLater()
        event.accept()


class SmFretTab(QWidget):
    def __init__(self, parent_suite=None):
        super().__init__()
        self.parent_suite = parent_suite
        self.params = parameters.Parameters()
        self.source_fname = None
        self.donor_data = self.acceptor_data = self.fret_data = None
        self.registration_matrix = None
        self.active_frame = 0
        self._loaded_orient_norm = True
        self._loaded_temp_even = True
        self.coords = []                  # Visible subset, for legacy callers.
        self.trajs_donor, self.trajs_acceptor, self.trajs_fret = [], [], []
        self.all_coords = []              # Complete trajectories (never filtered destructively).
        self.all_trajs_donor, self.all_trajs_acceptor, self.all_trajs_fret = [], [], []
        self.selected_indices = []
        self.selection_enabled = False
        self.e_min, self.e_max, self.s_min, self.s_max = 0.0, 1.0, 0.0, 1.0
        self.alpha, self.gamma = 0.0, 1.0
        self.roi = None
        self.roi_selectors = []
        self._press_pos = self._press_axes = None
        self._generation = 0
        self._worker_generation = 0
        self._display_traces = []
        self.worker = None
        self.batch_worker = None
        self.init_ui()

    def get_avg_bounds(self, current_frame, window_size, direction, max_frames):
        return average_bounds(current_frame, window_size, direction, max_frames)

    def _map_to_donor(self, x, y):
        return map_to_donor(x, y, self.registration_matrix)

    def selection_snapshot(self):
        return (self._generation, self.active_frame, self.alpha, self.gamma,
                self.avg_window_spin.value(), self.avg_direction_combo.currentText(),
                self.roi)

    def roi_indices(self):
        if self.roi is None:
            return list(range(len(self.all_coords)))
        xmin, xmax, ymin, ymax = self.roi
        return [i for i, (x, y) in enumerate(self.all_coords)
                if xmin <= x <= xmax and ymin <= y <= ymax]

    def apply_selection(self):
        indices = self.roi_indices()
        if self.selection_enabled and self.donor_data is not None:
            start, end = self.get_avg_bounds(self.active_frame, self.avg_window_spin.value(),
                                             self.avg_direction_combo.currentText(),
                                             self.donor_data.num_frames)
            indices = [i for i in indices if self._passes_selection(i, start, end)]
        self.selected_indices = indices
        self.coords = [self.all_coords[i] for i in indices]
        self.trajs_donor = [self.all_trajs_donor[i] for i in indices]
        self.trajs_acceptor = [self.all_trajs_acceptor[i] for i in indices]
        self.trajs_fret = [self.all_trajs_fret[i] for i in indices]

    def _passes_selection(self, i, start, end):
        e, s = selection_metrics(self.all_trajs_donor[i].intensity,
                                 self.all_trajs_acceptor[i].intensity,
                                 self.all_trajs_fret[i].intensity, start, end,
                                 self.alpha, self.gamma)
        return (np.isfinite(e) and np.isfinite(s)
                and self.e_min <= e <= self.e_max
                and self.s_min <= s <= self.s_max)

    def set_trajectories(self, coords, donors, acceptors, frets):
        if not (len(coords) == len(donors) == len(acceptors) == len(frets)):
            raise ValueError('Coordinate and trajectory counts do not match.')
        self.all_coords = list(coords)
        self.all_trajs_donor = list(donors)
        self.all_trajs_acceptor = list(acceptors)
        self.all_trajs_fret = list(frets)
        self._generation += 1
        self.apply_selection()

    def reset_stoichiometry_filter(self):
        self.selection_enabled = False
        self.e_min = self.s_min = 0.0
        self.e_max = self.s_max = 1.0
        self.apply_selection()
        self.refresh()

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
        slider_layout.addWidget(QLabel('Frame Index:'))
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self.on_frame_change)
        slider_layout.addWidget(self.frame_slider)
        self.frame_label = QLabel('0 / 0')
        slider_layout.addWidget(self.frame_label)
        plot_layout.addLayout(slider_layout)
        main_layout.addLayout(plot_layout, stretch=4)

        # The sidebar must scroll independently of the image/trajectory plots.
        # Build its complete layout BEFORE calling setWidget(): attaching an
        # empty layout can leave the QScrollArea using a stale size hint, which
        # makes the bottom controls inaccessible on shorter displays.
        scroll = QScrollArea(self)
        self.sidebar_scroll = scroll
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        scroll.setMinimumWidth(325)
        scroll.setMinimumHeight(0)
        scroll.verticalScrollBar().setSingleStep(24)
        controls = QWidget()
        controls.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        control_layout = QVBoxLayout(controls)
        # Tell Qt to recalculate the contents' minimum height when groups are
        # added, so it activates vertical scrolling instead of clipping them.
        control_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinAndMaxSize)
        main_layout.addWidget(scroll, stretch=1)
        self.btn_pick_file = QPushButton('Load TIF Stack File')
        self.btn_pick_file.clicked.connect(self.pick_file)
        self.btn_clear = QPushButton('Clear spots')
        self.btn_clear.clicked.connect(self.clear_plot)
        self.btn_refresh = QPushButton('Refresh')
        self.btn_refresh.clicked.connect(self.refresh)
        control_layout.addWidget(self.btn_pick_file)
        control_layout.addWidget(self.btn_clear)
        control_layout.addWidget(self.btn_refresh)

        roi_group = QGroupBox('Region of Interest (ROI)')
        roi_layout = QVBoxLayout()
        self.lbl_roi_status = QLabel('ROI: Full Frame (No Filter)')
        self.lbl_roi_status.setStyleSheet('font-weight: bold; color: #2b5b84;')
        self.btn_reset_roi = QPushButton('Reset ROI')
        self.btn_reset_roi.clicked.connect(self.reset_roi)
        roi_layout.addWidget(self.lbl_roi_status)
        roi_layout.addWidget(self.btn_reset_roi)
        roi_group.setLayout(roi_layout)
        control_layout.addWidget(roi_group)

        avg_group = QGroupBox('Spot Stoichiometry Filter Settings')
        avg_layout = QGridLayout()
        self.avg_window_spin = QSpinBox()
        self.avg_window_spin.setRange(1, 1000)
        self.avg_window_spin.setValue(5)
        self.avg_direction_combo = QComboBox()
        self.avg_direction_combo.addItems(['Forward', 'Centered', 'Trailing'])
        avg_layout.addWidget(QLabel('Frames to Avg:'), 0, 0)
        avg_layout.addWidget(self.avg_window_spin, 0, 1)
        avg_layout.addWidget(QLabel('Direction:'), 1, 0)
        avg_layout.addWidget(self.avg_direction_combo, 1, 1)
        avg_group.setLayout(avg_layout)
        control_layout.addWidget(avg_group)
        self.avg_window_spin.valueChanged.connect(self.on_selection_settings_changed)
        self.avg_direction_combo.currentTextChanged.connect(self.on_selection_settings_changed)

        correction_group = QGroupBox('E / S Alpha-Gamma Corrections')
        correction_layout = QGridLayout()
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0, 10.0)
        self.alpha_spin.setDecimals(4)
        self.alpha_spin.setSingleStep(0.01)
        self.alpha_spin.setValue(self.alpha)
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.0001, 100.0)
        self.gamma_spin.setDecimals(4)
        self.gamma_spin.setSingleStep(0.05)
        self.gamma_spin.setValue(self.gamma)
        correction_layout.addWidget(QLabel('Alpha (α):'), 0, 0)
        correction_layout.addWidget(self.alpha_spin, 0, 1)
        correction_layout.addWidget(QLabel('Gamma (γ):'), 1, 0)
        correction_layout.addWidget(self.gamma_spin, 1, 1)
        correction_group.setLayout(correction_layout)
        correction_group.setToolTip('E=(F-αD)/(F-αD+γD); S=(F-αD+γD)/(F-αD+γD+A), β=1.')
        control_layout.addWidget(correction_group)
        self.alpha_spin.valueChanged.connect(self.on_fret_corrections_changed)
        self.gamma_spin.valueChanged.connect(self.on_fret_corrections_changed)

        self.btn_autodetect = QPushButton('Autodetect Spots')
        self.btn_autodetect.clicked.connect(self.autodetect_spots)
        self.btn_stoic = QPushButton('Select Stoichiometry')
        self.btn_stoic.clicked.connect(self.open_stoic_window)
        self.btn_reset_stoic = QPushButton('Reset E / S selection')
        self.btn_reset_stoic.clicked.connect(self.reset_stoichiometry_filter)
        self.btn_hist = QPushButton('Show/Save E_FRET Histogram')
        self.btn_hist.clicked.connect(self.show_histogram)
        self.btn_batch_analyse = QPushButton('Analyse all')
        self.btn_batch_analyse.clicked.connect(self.batch_analyse_all)
        self.btn_cancel_batch = QPushButton('Cancel batch')
        self.btn_cancel_batch.clicked.connect(self.cancel_batch)
        self.btn_cancel_batch.setEnabled(False)
        self.batch_progress = QProgressBar()
        self.batch_progress.hide()
        self.chk_subdirs = QCheckBox('Include all subdirectories')
        self.chk_subdirs.setChecked(True)
        for widget in (self.btn_autodetect, self.btn_stoic, self.btn_reset_stoic,
                       self.btn_hist, self.btn_batch_analyse, self.btn_cancel_batch,
                       self.batch_progress, self.chk_subdirs):
            control_layout.addWidget(widget)

        temporal_group = QGroupBox('Temporal Deinterleaving (ALEX)')
        temp_layout = QVBoxLayout()
        self.temporal_cluster = QButtonGroup(self)
        self.rb_temp_even = QRadioButton('Channel 1 / Even Frames First')
        self.rb_temp_odd = QRadioButton('Channel 2 / Odd Frames First')
        self.rb_temp_even.setChecked(True)
        for rb in (self.rb_temp_even, self.rb_temp_odd):
            self.temporal_cluster.addButton(rb)
            temp_layout.addWidget(rb)
        temporal_group.setLayout(temp_layout)
        control_layout.addWidget(temporal_group)
        spatial_group = QGroupBox('Spatial Orientation Map (smFRET)')
        spatial_layout = QVBoxLayout()
        self.spatial_cluster = QButtonGroup(self)
        self.rb_orient_norm = QRadioButton('Left: Donor | Right: FRET/Acceptor')
        self.rb_orient_inv = QRadioButton('Left: FRET/Acceptor | Right: Donor')
        self.rb_orient_norm.setChecked(True)
        for rb in (self.rb_orient_norm, self.rb_orient_inv):
            self.spatial_cluster.addButton(rb)
            spatial_layout.addWidget(rb)
        spatial_group.setLayout(spatial_layout)
        control_layout.addWidget(spatial_group)
        self.rb_temp_even.toggled.connect(self.on_settings_changed)
        self.rb_orient_norm.toggled.connect(self.on_settings_changed)

        calib_group = QGroupBox('Calibration & Registration')
        calib_layout = QVBoxLayout()
        self.btn_registration = QPushButton('Calculate Registration Matrix')
        self.btn_registration.clicked.connect(
            self.parent_suite.run_registration if self.parent_suite else lambda: None)
        self.btn_clear_registration = QPushButton('Clear Registration Matrix')
        self.btn_clear_registration.clicked.connect(
            self.parent_suite.clear_registration if self.parent_suite else lambda: None)
        calib_layout.addWidget(self.btn_registration)
        calib_layout.addWidget(self.btn_clear_registration)
        calib_group.setLayout(calib_layout)
        control_layout.addWidget(calib_group)
        control_layout.addWidget(QLabel('Output name'))
        self.outname_box = QLineEdit('smFRET_output.dat')
        control_layout.addWidget(self.outname_box)
        self.btn_save = QPushButton('Export FRET Data')
        self.btn_save.clicked.connect(self.save_data)
        control_layout.addWidget(self.btn_save)
        ck_group = QGroupBox('Chung-Kennedy Filter Configuration')
        ck_layout = QVBoxLayout()
        self.ck_yes = QRadioButton('Apply Filter View')
        self.ck_no = QRadioButton('Keep Raw Trajectory View')
        self.ck_yes.setChecked(True)
        ck_layout.addWidget(self.ck_yes)
        ck_layout.addWidget(self.ck_no)
        ck_group.setLayout(ck_layout)
        control_layout.addWidget(ck_group)
        self.ck_yes.toggled.connect(self.refresh_trajectories)
        control_layout.addStretch()
        # Attach the fully populated sidebar only now so QScrollArea observes
        # its correct minimum size and can scroll to Chung-Kennedy and exports.
        scroll.setWidget(controls)

    def on_fret_corrections_changed(self, *args):
        self.alpha = self.alpha_spin.value()
        self.gamma = self.gamma_spin.value()
        self.apply_selection()
        # Only the corrected bottom plot changes when no E/S selection is active.
        if self.selection_enabled:
            self.refresh()
        else:
            self.refresh_efret_only()

    def on_selection_settings_changed(self, *args):
        if self.selection_enabled:
            self.apply_selection()
            self.refresh()

    def build_mosaic(self):
        for selector in self.roi_selectors:
            selector.set_active(False)
        self.fig.clear()
        mosaic = [['image1', 'image2', 'image3'],
                  ['plot1', 'plot1', 'plot1'],
                  ['plot2', 'plot2', 'plot2'],
                  ['plot3', 'plot3', 'plot3'],
                  ['plot4', 'plot4', 'plot4']]
        self.ax_dict = self.fig.subplot_mosaic(
            mosaic, gridspec_kw={'height_ratios': [5, 1, 1, 1, 1]})
        self.fig.subplots_adjust(left=0.06, right=0.92, top=0.96, bottom=0.06,
                                 hspace=0.35, wspace=0.15)
        self.donor_im, self.acc_im, self.fret_im = (self.ax_dict[x] for x in
                                                   ('image1', 'image2', 'image3'))
        self.iplot_donor, self.iplot_acceptor, self.iplot_fret = (
            self.ax_dict[x] for x in ('plot1', 'plot2', 'plot3'))
        self.plot_efret = self.ax_dict['plot4']
        self.cb1 = self.cb2 = self.cb3 = None
        self.apply_plot_labels()
        self.roi_selectors = [RectangleSelector(
            ax, self._on_roi_select, useblit=False, button=[1],
            minspanx=5, minspany=5, interactive=False,
            props=dict(facecolor='cyan', edgecolor='cyan', alpha=0.25, fill=True),
        ) for ax in (self.donor_im, self.acc_im, self.fret_im)]
        self.canvas.draw_idle()

    def _on_roi_select(self, eclick, erelease):
        if eclick is None or erelease is None or self.donor_data is None:
            return
        if any(v is None for v in (eclick.xdata, erelease.xdata,
                                   eclick.ydata, erelease.ydata)):
            return
        x1, y1 = float(eclick.xdata), float(eclick.ydata)
        x2, y2 = float(erelease.xdata), float(erelease.ydata)
        if abs(x2-x1) < 3 or abs(y2-y1) < 3:
            return
        if eclick.inaxes == self.donor_im and self.registration_matrix is not None:
            try:
                matrix = _affine_matrix(self.registration_matrix)
                corners = np.array([[x1, y1, 1], [x2, y1, 1],
                                    [x2, y2, 1], [x1, y2, 1]])
                transformed = corners @ matrix.T
                xmin, xmax = transformed[:, 0].min(), transformed[:, 0].max()
                ymin, ymax = transformed[:, 1].min(), transformed[:, 1].max()
            except ValueError as exc:
                show_popup(self, 'Invalid registration', str(exc), critical=True)
                return
        else:
            xmin, xmax = sorted((x1, x2))
            ymin, ymax = sorted((y1, y2))
        max_x, max_y = self.acceptor_data.frame_size
        xmin, xmax = max(0., xmin), min(float(max_x), xmax)
        ymin, ymax = max(0., ymin), min(float(max_y), ymax)
        if xmin >= xmax or ymin >= ymax:
            return
        self.roi = (float(xmin), float(xmax), float(ymin), float(ymax))
        self._generation += 1  # Invalidate any in-flight detection with the previous ROI.
        self.lbl_roi_status.setText(
            f'ROI Active:\nX: [{xmin:.1f}, {xmax:.1f}]\nY: [{ymin:.1f}, {ymax:.1f}]')
        self.apply_selection()
        self.refresh()

    def reset_roi(self):
        self.roi = None
        self._generation += 1
        self.lbl_roi_status.setText('ROI: Full Frame (No Filter)')
        self.apply_selection()
        self.refresh()

    def apply_plot_labels(self):
        self.iplot_donor.set_ylabel(r'Donor (x10$^3$)')
        self.iplot_acceptor.set_ylabel(r'Acceptor (x10$^3$)')
        self.iplot_fret.set_ylabel(r'FRET (x10$^3$)')
        self.plot_efret.set_ylabel('Corrected E_FRET')
        self.plot_efret.set_xlabel('Frame number')

    def on_settings_changed(self, *args):
        if self.source_fname is None:
            return
        # Keep the old dataset/trajectory combination if reload fails.
        if self.get_data(self.source_fname):
            self.clear_plot()
            self.update_ui_limits()
            self.refresh()
        else:
            # Revert the controls to the settings of the still-loaded dataset.
            controls = (self.rb_orient_norm, self.rb_orient_inv,
                        self.rb_temp_even, self.rb_temp_odd)
            for control in controls:
                control.blockSignals(True)
            try:
                self.rb_orient_norm.setChecked(self._loaded_orient_norm)
                self.rb_orient_inv.setChecked(not self._loaded_orient_norm)
                self.rb_temp_even.setChecked(self._loaded_temp_even)
                self.rb_temp_odd.setChecked(not self._loaded_temp_even)
            finally:
                for control in controls:
                    control.blockSignals(False)

    def pick_file(self):
        if self.worker is not None and self.worker.isRunning():
            show_popup(self, 'Busy', 'Spot detection is still running.')
            return
        fname, _ = QFileDialog.getOpenFileName(
            self, 'Open TIF Stack', filter='TIFF Files (*.tif *.tiff)')
        if not fname:
            return
        # Import before resetting the other tabs, so a bad TIFF preserves the session.
        if not self.get_data(fname):
            return
        loaded = (self.donor_data, self.acceptor_data, self.fret_data)
        if self.parent_suite:
            self.parent_suite.reset_entire_suite()
        self.donor_data, self.acceptor_data, self.fret_data = loaded
        self.source_fname = str(fname)
        self.params.name = str(Path(fname).with_suffix(''))
        self.roi = None
        self.lbl_roi_status.setText('ROI: Full Frame (No Filter)')
        self.selection_enabled = False
        self.e_min = self.s_min = 0.0
        self.e_max = self.s_max = 1.0
        self.clear_plot()
        self.update_ui_limits()
        self.build_mosaic()
        self.refresh()

    def _build_pystachio_image(self, numpy_array, fname):
        return _build_image(numpy_array, fname, self.params)

    def get_data(self, fname, orient_norm=None, temp_even=None):
        """Transactionally replace channel objects; preserve previous state on errors.

        This method remains available to external suite callers. Explicit channel
        override parameters cause errors to propagate; interactive calls show a popup.
        """
        try:
            normal = self.rb_orient_norm.isChecked() if orient_norm is None else orient_norm
            even = self.rb_temp_even.isChecked() if temp_even is None else temp_even
            # Some PySTACHIO ImageData implementations depend on params.name.
            old_name = self.params.name
            self.params.name = str(Path(fname).with_suffix(''))
            try:
                new_data = load_dataset(fname, self.params, normal, even)
            except Exception:
                self.params.name = old_name
                raise
            self.donor_data, self.acceptor_data, self.fret_data = new_data
            self._loaded_orient_norm = normal
            self._loaded_temp_even = even
            return True
        except Exception as exc:
            if orient_norm is not None or temp_even is not None:
                raise
            show_popup(self, 'Dataset Import Failure', str(exc), critical=True)
            return False

    def update_ui_limits(self):
        if self.donor_data is None:
            self.frame_slider.setEnabled(False)
            return
        self.frame_slider.blockSignals(True)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(self.donor_data.num_frames - 1)
        self.active_frame = min(self.active_frame, self.frame_slider.maximum())
        self.frame_slider.setValue(self.active_frame)
        self.frame_slider.blockSignals(False)
        self.frame_slider.setEnabled(True)
        self.frame_label.setText(f'Index: {self.active_frame} / {self.frame_slider.maximum()}')

    def on_frame_change(self, value):
        if self.active_frame != value:
            self._generation += 1
        self.active_frame = value
        if self.selection_enabled:
            self.apply_selection()
            self.refresh()
        else:
            self.draw_images()

    def draw_images(self):
        if self.donor_data is None:
            return
        frame = min(self.active_frame, self.donor_data.num_frames - 1)
        for ax in (self.donor_im, self.acc_im, self.fret_im):
            # Do not clear the axes: RectangleSelector owns artists on them.
            # Removing those would break subsequent ROI drag interactions.
            selector_artists = [artist for selector in self.roi_selectors
                                if selector.ax is ax for artist in selector.artists]
            for image in list(ax.images):
                image.remove()
            for collection in list(ax.collections):
                if collection not in selector_artists:
                    collection.remove()
            for patch in list(ax.patches):
                if patch not in selector_artists:
                    patch.remove()
            for line in list(ax.lines):
                if line not in selector_artists:
                    line.remove()
        channels = (
            (self.donor_im, self.donor_data, 'Donor Channel Stack'),
            (self.acc_im, self.acceptor_data, 'Acceptor Channel Stack'),
            (self.fret_im, self.fret_data, 'FRET Channel Stack'),
        )
        colorbars = ('cb1', 'cb2', 'cb3')
        for (ax, data, title), cb_name in zip(channels, colorbars):
            image = ax.imshow(data[frame].as_image(), cmap='Greys_r')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title)
            cb = getattr(self, cb_name)
            if cb is None:
                setattr(self, cb_name, self.fig.colorbar(image, ax=ax))
            else:
                cb.update_normal(image)
        self.frame_label.setText(f'Index: {frame} / {self.frame_slider.maximum()}')
        max_x, max_y = self.donor_data.frame_size
        for x, y in self.coords:
            self.acc_im.scatter(x, y, marker='x', color='r')
            self.fret_im.scatter(x, y, marker='x', color='r')
            try:
                dx, dy = self._map_to_donor(x, y)
            except ValueError:
                continue  # Invalid registrations never substitute untransformed spots.
            if 0 <= dx < max_x and 0 <= dy < max_y:
                self.donor_im.scatter(dx, dy, marker='x', color='r')
        if self.roi is not None:
            xmin, xmax, ymin, ymax = self.roi
            corners = np.array([[xmin, ymin, 1], [xmax, ymin, 1],
                                [xmax, ymax, 1], [xmin, ymax, 1]])
            for ax in (self.acc_im, self.fret_im):
                ax.add_patch(patches.Polygon(corners[:, :2], closed=True,
                                             fill=False, edgecolor='cyan',
                                             linestyle='--', linewidth=1.5))
            try:
                matrix = _affine_matrix(self.registration_matrix)
                mapped = corners[:, :2] if matrix is None else (
                    corners @ cv2.invertAffineTransform(matrix).T)
                self.donor_im.add_patch(patches.Polygon(
                    mapped, closed=True, fill=False, edgecolor='cyan',
                    linestyle='--', linewidth=1.5))
            except ValueError:
                pass
        self.canvas.draw_idle()

    def _plot_display_traces(self):
        self._display_traces = []
        for td, ta, tfret in zip(self.trajs_donor[:50], self.trajs_acceptor[:50],
                                 self.trajs_fret[:50]):
            traces = [np.asarray(x.intensity, dtype=float) for x in (td, ta, tfret)]
            if self.ck_yes.isChecked():
                try:
                    traces = [np.asarray(postprocessing.chung_kennedy_filter(
                        x, 5, 1)[0][:-1], dtype=float) for x in traces]
                except Exception:
                    pass  # View-only smoothing; never alters the raw data.
            self._display_traces.append(traces)
            for ax, trace in zip((self.iplot_donor, self.iplot_acceptor,
                                  self.iplot_fret), traces):
                ax.plot(trace[:self.donor_data.num_frames]/1000, alpha=0.7)

    def refresh_trajectories(self, *args):
        if self.donor_data is None:
            return
        for ax in (self.iplot_donor, self.iplot_acceptor, self.iplot_fret):
            ax.clear()
        self.apply_plot_labels()
        self._plot_display_traces()
        self.refresh_efret_only()

    def refresh_efret_only(self, *args):
        if self.donor_data is None:
            return
        self.plot_efret.clear()
        self.plot_efret.set_ylabel('Corrected E_FRET')
        self.plot_efret.set_xlabel('Frame number')
        for d, _a, fret in self._display_traces:
            e = calculate_efret(d, fret, self.alpha, self.gamma)
            self.plot_efret.plot(e[:self.donor_data.num_frames], alpha=0.7)
        self.canvas.draw_idle()

    def refresh(self, *args):
        if self.donor_data is None:
            return
        self.draw_images()
        self.refresh_trajectories()

    def clear_plot(self):
        self._generation += 1
        self.e_min = self.s_min = 0.0
        self.e_max = self.s_max = 1.0
        self.all_coords = []
        self.all_trajs_donor = []
        self.all_trajs_acceptor = []
        self.all_trajs_fret = []
        self.selection_enabled = False
        self.apply_selection()
        self.refresh()

    def _extract_trajectories(self, track_params):
        """Retained as a compatibility entry point for other suite components."""
        coords_to_extract = self.all_coords if self.all_coords else self.coords
        d, a, f, coords = extract_trajectories(
            self.donor_data, self.acceptor_data, self.fret_data, coords_to_extract,
            track_params, self.registration_matrix, self.roi)
        self.set_trajectories(coords, d, a, f)

    def on_press(self, event):
        if event.inaxes in (self.donor_im, self.acc_im, self.fret_im) and event.xdata is not None:
            self._press_pos = (event.xdata, event.ydata)
            self._press_axes = event.inaxes
        else:
            self._press_pos = self._press_axes = None

    def on_release(self, event):
        if (self.donor_data is None or self._press_pos is None
                or event.inaxes != self._press_axes or event.xdata is None):
            return
        if self.worker is not None and self.worker.isRunning():
            return
        mode = getattr(self.toolbar, 'mode', '')
        mode = mode.value if hasattr(mode, 'value') else mode
        if mode != '':
            return
        px, py = self._press_pos
        self._press_pos = self._press_axes = None
        if np.hypot(event.xdata-px, event.ydata-py) >= 3:
            return
        x, y = float(event.xdata), float(event.ydata)
        try:
            if event.inaxes == self.donor_im and self.registration_matrix is not None:
                matrix = _affine_matrix(self.registration_matrix)
                x, y = matrix @ np.array([x, y, 1.0])
            max_x, max_y = self.acceptor_data.frame_size
            if not (0 <= x < max_x and 0 <= y < max_y):
                return
            refined = spots.Spots(frame=0)
            refined.set_positions([[x, y]])
            refined.num_spots = 1
            refined.refine_centres(self.acceptor_data[self.active_frame], self.params)
            if len(refined.positions) == 0:
                return
            x, y = map(float, refined.positions[0][:2])
            d, a, f, coords = extract_trajectories(
                self.donor_data, self.acceptor_data, self.fret_data, [[x, y]],
                self.params, self.registration_matrix, self.roi)
            if not coords:
                show_popup(self, 'Spot excluded', 'Spot outside the ROI or valid channel bounds.')
                return
            self.set_trajectories(self.all_coords + coords,
                                  self.all_trajs_donor + d,
                                  self.all_trajs_acceptor + a,
                                  self.all_trajs_fret + f)
            self.refresh()
        except Exception as exc:
            show_popup(self, 'Spot extraction failed', str(exc), critical=True)

    def autodetect_spots(self):
        if self.donor_data is None:
            return
        if self.worker is not None and self.worker.isRunning():
            return
        track_params = (self.parent_suite.main_analysis_tab.params
                        if self.parent_suite and hasattr(self.parent_suite, 'main_analysis_tab')
                        else self.params)
        try:
            registration = (None if self.registration_matrix is None
                            else _affine_matrix(self.registration_matrix).copy())
        except ValueError as exc:
            show_popup(self, 'Invalid registration', str(exc), critical=True)
            return
        try:
            params_snapshot = copy.deepcopy(track_params)
        except Exception as exc:
            show_popup(self, 'Detection setup failed', str(exc), critical=True)
            return
        self.btn_autodetect.setEnabled(False)
        self.btn_clear.setEnabled(False)
        self.btn_pick_file.setEnabled(False)
        self._worker_generation = self._generation
        self.worker = FullAnalysisWorker(
            self.acceptor_data, self.donor_data, self.fret_data, self.active_frame,
            params_snapshot, registration, roi=self.roi)
        self.worker.progress.connect(self.btn_autodetect.setText)
        self.worker.result_ready.connect(self._on_analysis_finished)
        self.worker.finished.connect(self._on_analysis_thread_finished)
        self.worker.start()

    def _on_analysis_finished(self, td, ta, tfret, coords, error):
        if self._worker_generation != self._generation:
            return  # Ignore stale completion following an external dataset change.
        if error:
            show_popup(self, 'Autodetect Error', error, critical=True)
        elif not coords:
            show_popup(self, 'Autodetect', 'No valid spots were found in the active ROI.')
        else:
            try:
                self.set_trajectories(coords, td, ta, tfret)
                self.refresh()
            except Exception as exc:
                show_popup(self, 'Trajectory error', str(exc), critical=True)

    def _on_analysis_thread_finished(self):
        self.btn_autodetect.setEnabled(True)
        self.btn_autodetect.setText('Autodetect Spots')
        self.btn_clear.setEnabled(True)
        self.btn_pick_file.setEnabled(True)

    def open_stoic_window(self):
        if not self.all_trajs_donor:
            show_popup(self, 'Error', 'No spots/trajectories available.', critical=True)
            return
        self.stoic_win = StoichiometryWindow(self)
        self.stoic_win.show()

    def show_histogram(self):
        if self.donor_data is None or not self.trajs_donor:
            show_popup(self, 'Error', 'No spots available to build histogram.', critical=True)
            return
        start, end = self.get_avg_bounds(
            self.active_frame, self.avg_window_spin.value(),
            self.avg_direction_combo.currentText(), self.donor_data.num_frames)
        e_vals = []
        for td, ta, tfret in zip(self.trajs_donor, self.trajs_acceptor, self.trajs_fret):
            e, _ = selection_metrics(td.intensity, ta.intensity, tfret.intensity,
                                     start, end, self.alpha, self.gamma)
            if np.isfinite(e) and self.e_min <= e <= self.e_max:
                e_vals.append(e)
        if not e_vals:
            show_popup(self, 'No histogram values', 'No valid E_FRET values in the selected range.')
            return
        output_dir = Path(self.source_fname).parent
        path = output_dir / f'EFRET_hist_frame_{self.active_frame}.png'
        self.hist_win = HistogramWindow(self, e_vals, self.active_frame, str(path))
        self.hist_win.show()

    def save_data(self):
        if self.source_fname is None or not self.trajs_donor:
            show_popup(self, 'Nothing to export', 'Load and analyse a dataset first.')
            return
        out_name = Path(self.outname_box.text())
        if not out_name.name or out_name.is_absolute() or out_name.name != str(out_name):
            show_popup(self, 'Invalid output filename', 'Enter a filename without a directory.', critical=True)
            return
        out_path = Path(self.source_fname).parent / out_name
        try:
            # Only replace an existing export after the entire file is written.
            with tempfile.TemporaryDirectory(dir=str(out_path.parent),
                                             prefix='.smfret_export_') as staging:
                temp_path = Path(staging) / out_name.name
                with temp_path.open('w', newline='', encoding='utf-8') as handle:
                    handle.write(f'# Source: {Path(self.source_fname).name}\n')
                    handle.write(f'# Alpha={self.alpha:.4f}, gamma={self.gamma:.4f}, beta=1\n')
                    handle.write('# E=(F-alpha*D)/(F-alpha*D+gamma*D)\n')
                    handle.write('# S=(F-alpha*D+gamma*D)/(F-alpha*D+gamma*D+A)\n')
                    handle.write('# Direct excitation uncorrected; invalid ratios are NaN.\n')
                    handle.write('# Raw intensities; Chung-Kennedy is display-only.\n')
                    handle.write('# Format: tab-separated; selected trajectories only.\n')
                    writer = csv.writer(handle, delimiter='\t')
                    writer.writerow(('Frame', 'Traj_ID', 'X_Coord', 'Y_Coord',
                                     'Donor_Int', 'Acceptor_Int', 'FRET_Int', 'E_FRET', 'S'))
                    if not (len(self.coords) == len(self.trajs_donor) ==
                            len(self.trajs_acceptor) == len(self.trajs_fret)):
                        raise ValueError('Spot coordinate and trajectory counts differ.')
                    for i, (td, ta, tfret) in enumerate(zip(
                            self.trajs_donor, self.trajs_acceptor, self.trajs_fret)):
                        x, y = self.coords[i]
                        d, a, f = (np.asarray(t.intensity, dtype=float)
                                   for t in (td, ta, tfret))
                        if min(len(d), len(a), len(f)) < self.donor_data.num_frames:
                            raise ValueError(f'Trajectory {i} is shorter than the dataset.')
                        ef = calculate_efret(d, f, self.alpha, self.gamma)
                        st = calculate_stoichiometry(d, a, f, self.alpha, self.gamma)
                        for frame in range(self.donor_data.num_frames):
                            writer.writerow((frame, i, f'{x:.3f}', f'{y:.3f}',
                                             f'{d[frame]:.6g}', f'{a[frame]:.6g}',
                                             f'{f[frame]:.6g}', f'{ef[frame]:.6g}',
                                             f'{st[frame]:.6g}'))
                temp_path.replace(out_path)
            show_popup(self, 'Export Success', f'Exported to:\n{out_path}')
        except Exception as exc:
            show_popup(self, 'Export failed', str(exc), critical=True)

    def batch_analyse_all(self):
        if self.source_fname is None:
            show_popup(self, 'Error', 'Load a TIFF file before starting batch analysis.', critical=True)
            return
        if self.batch_worker is not None and self.batch_worker.isRunning():
            return
        directory = Path(self.source_fname).parent
        paths = (directory.rglob('*') if self.chk_subdirs.isChecked() else directory.iterdir())
        excluded = ('_cropped', '_left', '_right', '_donor', '_acceptor', '_fret')
        files_to_process = sorted(
            (p for p in paths if p.is_file() and p.suffix.lower() in ('.tif', '.tiff')
             and not p.stem.lower().endswith(excluded)), key=str)
        if not files_to_process:
            show_popup(self, 'Batch Analysis', 'No valid TIFF stacks found.')
            return
        track_params = (self.parent_suite.main_analysis_tab.params
                        if self.parent_suite and hasattr(self.parent_suite, 'main_analysis_tab')
                        else self.params)
        try:
            registration = (None if self.registration_matrix is None
                            else _affine_matrix(self.registration_matrix).copy())
            settings = dict(
                win_size=self.avg_window_spin.value(),
                direction=self.avg_direction_combo.currentText(),
                orient_norm=self.rb_orient_norm.isChecked(),
                temp_even=self.rb_temp_even.isChecked(),
                active_frame=self.active_frame,
                roi=self.roi,
                registration_matrix=registration,
                alpha=self.alpha, gamma=self.gamma,
                e_min=self.e_min, e_max=self.e_max,
                s_min=self.s_min, s_max=self.s_max,
            )
            image_params = copy.deepcopy(self.params)
            track_params = copy.deepcopy(track_params)
        except Exception as exc:
            show_popup(self, 'Batch setup failed', str(exc), critical=True)
            return
        self.btn_batch_analyse.setEnabled(False)
        self.btn_batch_analyse.setText('Processing...')
        self.btn_cancel_batch.setEnabled(True)
        self.batch_progress.setMaximum(len(files_to_process))
        self.batch_progress.setValue(0)
        self.batch_progress.show()
        self.batch_worker = BatchWorker(files_to_process, directory, image_params,
                                        track_params, settings)
        self.batch_worker.progress.connect(self.batch_progress.setValue)
        self.batch_worker.result_ready.connect(self._on_batch_finished)
        self.batch_worker.error_ready.connect(self._on_batch_error)
        self.batch_worker.finished.connect(self._batch_thread_finished)
        self.batch_worker.start()

    def cancel_batch(self):
        if self.batch_worker is not None and self.batch_worker.isRunning():
            self.batch_worker.requestInterruption()
            self.btn_cancel_batch.setEnabled(False)
            self.btn_batch_analyse.setText('Cancelling...')

    def _on_batch_finished(self, result):
        details = (f"Processed {result['processed']} files; "
                   f"{result['no_spots']} without spots; "
                   f"{result['failed']} failed; "
                   f"{result['molecules']} trajectories.\n\n"
                   f"Results: {result['directory']}")
        show_popup(self, 'Batch completed', details,
                   critical=result['failed'] > 0)

    def _on_batch_error(self, message):
        show_popup(self, 'Batch stopped', message, critical=True)

    def _batch_thread_finished(self):
        self.btn_batch_analyse.setEnabled(True)
        self.btn_batch_analyse.setText('Analyse all')
        self.btn_cancel_batch.setEnabled(False)
        self.batch_progress.hide()

    def closeEvent(self, event):
        # Prevent deleting QThreads/data while either analysis is still running.
        running = [w for w in (self.worker, self.batch_worker)
                   if w is not None and w.isRunning()]
        if running:
            for worker in running:
                worker.requestInterruption()
            show_popup(self, 'Processing active',
                       'Cancellation requested. Close the tab after processing has stopped.')
            event.ignore()
        else:
            event.accept()
