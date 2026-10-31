"""Standalone PySTACHIO Studio tracking tab.

The image-processing helpers in this module belong exclusively to the tracking
workflow. They are intentionally not shared with the separate smFRET program.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import math
import multiprocessing as mp
import os
from pathlib import Path
import traceback
from uuid import uuid4

import numpy as np
import tifffile as tf
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import RectangleSelector
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QDoubleValidator, QIntValidator
from PyQt6.QtWidgets import (
    QCheckBox, QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton, QRadioButton,
    QButtonGroup, QSlider, QTabWidget, QVBoxLayout, QWidget,
)

import images
import parameters
import postprocessing
import tracking
from studio.studio_helpers import (
    export_pystachio_params, show_popup, to_imagej_compatible,
)

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrackingViewConfig:
    """Settings specific to this tracking tab's spatial/temporal processing."""

    mode: str = "single"  # single, dual or fret
    crop: tuple[int, int, int, int] | None = None  # channel-local x0,x1,y0,y1
    dual_alex: bool = False  # for dual mode, retain one excitation phase
    first_even: bool = True  # selected phase in dual; donor excitation in FRET
    donor_left: bool = True

    @property
    def deinterleaved(self) -> bool:
        return self.mode == "fret" or (self.mode == "dual" and self.dual_alex)


def _validate_stack(array: np.ndarray, axes: str | None = None) -> np.ndarray:
    """Validate a grayscale stack with shape (frames, y, x).

    Some microscope/ImageJ TIFFs label an ordinary frame sequence as ``ZYX``.
    The axes alone cannot distinguish those movies from actual Z-stacks, so
    in this *tracking* application we interpret the leading Z axis as frames,
    matching the behaviour of the original program. Explicit colour/channel
    axes are still rejected rather than silently treating channels as time.
    """
    array = np.asarray(array)
    if array.ndim == 2:
        if axes and axes != "YX":
            raise ValueError(f"Expected grayscale image axes YX; TIFF has {axes!r}.")
        array = array[np.newaxis, :, :]
    elif array.ndim == 3:
        if axes and axes not in ("TYX", "QYX", "IYX", "ZYX"):
            raise ValueError(
                f"Unsupported grayscale movie axes {axes!r}; expected "
                "(frames, y, x) data. TIFFs tagged ZYX are accepted "
                "as frame sequences, but colour/channel dimensions are not."
            )
    else:
        raise ValueError(
            f"Expected grayscale 2D or (frames,y,x) TIFF; found shape {array.shape}."
        )
    if any(d < 1 for d in array.shape):
        raise ValueError(f"Empty TIFF dimension: {array.shape}.")
    if array.dtype.kind not in "uif":
        raise ValueError(f"Unsupported image type: {array.dtype}.")
    return array


def _read_stack(path: str) -> np.ndarray:
    with tf.TiffFile(path) as tif:
        if len(tif.series) != 1:
            raise ValueError(
                "TIFF contains multiple series. Export the intended grayscale "
                "time sequence as a single TIFF series first."
            )
        series = tif.series[0]
        return _validate_stack(series.asarray(), series.axes)


def _validate_crop(crop, frame_height, frame_width):
    if crop is None:
        return None
    x0, x1, y0, y1 = crop
    if not (0 <= x0 < x1 <= frame_width and 0 <= y0 < y1 <= frame_height):
        raise ValueError(
            f"Crop {crop} is outside the valid channel area "
            f"(width={frame_width}, height={frame_height})."
        )
    if x1 - x0 < 2 or y1 - y0 < 2:
        raise ValueError("Crop must be at least 2 pixels wide and high.")
    return crop


def _prepare_stack(stack: np.ndarray, config: TrackingViewConfig) -> dict:
    """Produce this tracking program's channel stacks and frame provenance."""
    stack = _validate_stack(stack)
    n_frames, height, width = stack.shape
    if config.mode not in ("single", "dual", "fret"):
        raise ValueError(f"Unknown analysis mode: {config.mode}")

    if config.mode == "single":
        crop = _validate_crop(config.crop, height, width)
        if crop:
            x0, x1, y0, y1 = crop
            stack = stack[:, y0:y1, x0:x1]
        return {
            "mode": "single", "ch1": stack, "deinterleaved": False,
            "frame_step": 1, "raw_offsets": {"ch1": 0},
            "dropped_raw_frames": 0,
        }

    if width % 2 != 0:
        raise ValueError(
            f"Split-channel TIFF width ({width}) must be even. "
            "A split through an odd-width detector creates unequal channels."
        )
    half = width // 2
    if half < 2:
        raise ValueError("Each spatial channel must contain at least 2 columns.")
    crop = _validate_crop(config.crop, height, half)
    left, right = stack[:, :, :half], stack[:, :, half:]
    if crop:
        x0, x1, y0, y1 = crop
        left = left[:, y0:y1, x0:x1]
        right = right[:, y0:y1, x0:x1]

    if config.mode == "dual":
        phase = 0 if config.first_even else 1
        if config.dual_alex:
            a, b = left[phase::2], right[phase::2]
        else:
            a, b = left, right
        if not len(a):
            raise ValueError("No images remain in the selected ALEX excitation phase.")
        return {
            "mode": "dual", "ch1": a, "ch2": b,
            "deinterleaved": config.dual_alex,
            "frame_step": 2 if config.dual_alex else 1,
            "raw_offsets": {"ch1": phase, "ch2": phase} if config.dual_alex
                           else {"ch1": 0, "ch2": 0},
            "dropped_raw_frames": n_frames - len(a) if config.dual_alex else 0,
        }

    if n_frames < 2:
        raise ValueError("FRET/ALEX processing requires at least two camera frames.")
    donor, right_path = (left, right) if config.donor_left else (right, left)
    donor_phase = 0 if config.first_even else 1
    acceptor_phase = 1 - donor_phase
    # Every output index denotes the same 2-frame ALEX cycle, not simultaneous
    # camera exposures. For odd-length movies one unpaired frame is discarded.
    n_cycles = n_frames // 2
    if not n_cycles:
        raise ValueError("No complete ALEX cycles are present.")
    return {
        "mode": "fret",
        "ch1": donor[donor_phase::2][:n_cycles],
        "ch2": right_path[acceptor_phase::2][:n_cycles],
        "ch3": right_path[donor_phase::2][:n_cycles],
        "deinterleaved": True,
        "frame_step": 2,
        "raw_offsets": {
            "ch1": donor_phase, "ch2": acceptor_phase, "ch3": donor_phase,
        },
        "dropped_raw_frames": n_frames - 2 * n_cycles,
    }


def _processed_start_frame(config: TrackingViewConfig, raw_start: int) -> int:
    """Map a raw camera index onto a processed sequence's first usable frame.

    In FRET mode all three images of a 2-frame cycle must be available after
    raw_start, so the first usable full cycle is ceil(raw_start / 2).
    """
    if config.mode == "fret":
        return (raw_start + 1) // 2
    if config.mode == "dual" and config.dual_alex:
        phase = 0 if config.first_even else 1
        return max(0, (raw_start - phase + 1) // 2)
    return raw_start


def _channel_names(dataset: dict) -> dict[str, str]:
    if dataset["mode"] == "single":
        return {"ch1": "cropped"}
    if dataset["mode"] == "dual":
        return {"ch1": "left", "ch2": "right"}
    return {"ch1": "donor", "ch2": "acceptor", "ch3": "fret"}


def _new_results_folder(source_file: str) -> Path:
    path = Path(source_file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = path.parent / "PySTACHIO_results" / path.stem / f"{timestamp}_{uuid4().hex[:8]}"
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def _safe_number(value, field: str, *, integer=False, minimum=None, strict=False):
    try:
        result = int(value) if integer else float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a valid number.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite.")
    if minimum is not None and (result <= minimum if strict else result < minimum):
        sign = ">" if strict else "≥"
        raise ValueError(f"{field} must be {sign} {minimum}.")
    return result


def _metadata(source_file, stack, dataset, config, input_params, backend_params,
              registration_matrix, task):
    raw_start = input_params.start_frame
    return {
        "source_file": str(Path(source_file).resolve()),
        "operation": task,
        "utc_created": datetime.now(timezone.utc).isoformat(),
        "raw_shape_frames_y_x": list(stack.shape),
        "analysis_mode": config.mode,
        "crop_channel_local_x0_x1_y0_y1": list(config.crop) if config.crop else None,
        "donor_is_left": config.donor_left if config.mode == "fret" else None,
        "raw_even_frames_are_donor_or_selected": config.first_even if dataset["deinterleaved"] else None,
        "dual_alex_extracts_one_excitation_phase": config.dual_alex if config.mode == "dual" else None,
        "output_channels": _channel_names(dataset),
        "raw_frame_index_for_processed_i": {
            name: {"offset": offset, "stride": dataset["frame_step"]}
            for name, offset in dataset["raw_offsets"].items()
        },
        "raw_frames_excluded_from_export": dataset["dropped_raw_frames"],
        "raw_frames_unpaired_at_end": (dataset["dropped_raw_frames"] if config.mode == "fret" else 0),
        "processed_frame_count": len(dataset["ch1"]),
        "camera_frame_interval_seconds": input_params.frame_time,
        "tracking_frame_interval_seconds": backend_params.frame_time,
        "gui_raw_start_frame": raw_start,
        "backend_processed_start_frame": backend_params.start_frame,
        "requested_backend_ALEX": bool(input_params.ALEX),
        "used_backend_ALEX": bool(backend_params.ALEX),
        "requested_backend_colocalize": bool(input_params.colocalize),
        "used_backend_colocalize": bool(backend_params.colocalize),
        "registration_matrix_exported": registration_matrix is not None,
        "registration_applied": False,
        "registration_note": "Not applied by this tab: registration convention / postprocessing coordinate format is unspecified.",
        "tracking_parameters": {
            key: getattr(backend_params, key, None) for key in (
                "frame_time", "pixel_size", "start_frame", "calculate_isingle",
                "I_single", "snr_filter_cutoff", "max_displacement", "min_traj_len",
                "bw_threshold_tolerance", "struct_disk_radius", "subarray_halfwidth",
                "inner_mask_radius", "num_procs", "ALEX", "start_channel",
                "L_isingle", "R_isingle", "colocalize", "colocalize_distance",
                "colocalize_n_frames",
            )
        },
    }


def _execute_one(source_file, config, params_snapshot, reference_hw,
                 registration_matrix, action, report, cancelled):
    """Standalone, non-Qt processing work executed in the worker thread."""
    stack = _read_stack(source_file)
    if (reference_hw is not None and
            (config.crop is not None or registration_matrix is not None) and
            stack.shape[1:] != reference_hw):
        raise ValueError(
            f"Dimensions {stack.shape[1:]} differ from the crop reference "
            f"{reference_hw}. Refusing to apply the selected crop to a different camera geometry."
        )
    dataset = _prepare_stack(stack, config)
    tracking_params = copy.deepcopy(params_snapshot)
    tracking_params.frame_time = params_snapshot.frame_time * dataset["frame_step"]
    tracking_params.start_frame = _processed_start_frame(config, params_snapshot.start_frame)
    if config.mode != "single":
        # Each output is tracked independently. ALEX/colocalization in the
        # backend must not be applied to these spatially separated outputs,
        # whether or not the GUI also selected one temporal excitation phase.
        tracking_params.ALEX = False
        tracking_params.colocalize = False
    if action == "analyse":
        shape_yx = dataset["ch1"].shape[1:]
        needed = 2 * tracking_params.subarray_halfwidth + 1
        if min(shape_yx) < needed:
            raise ValueError(
                f"Processed image dimensions {shape_yx} are smaller than "
                f"the {needed}x{needed} subarray required by subarray_halfwidth."
            )
        remaining = len(dataset["ch1"]) - tracking_params.start_frame
        if remaining < tracking_params.min_traj_len:
            raise ValueError(
                f"Only {max(0, remaining)} processed frames remain at the selected "
                f"start frame; min_traj_length is {tracking_params.min_traj_len}."
            )
    if cancelled():
        return None
    output_folder = _new_results_folder(source_file)
    names = _channel_names(dataset)
    metadata = _metadata(source_file, stack, dataset, config, params_snapshot,
                         tracking_params, registration_matrix, action)
    metadata["completion_status"] = "in_progress"
    metadata_path = output_folder / "analysis_metadata.json"

    def write_metadata():
        metadata_path.write_text(
            json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
        )

    write_metadata()
    if registration_matrix is not None:
        np.savetxt(output_folder / "registration_matrix_UNAPPLIED.txt", registration_matrix)

    # imagej-compatible stacks saved as .tif regardless of source extension.
    try:
        for channel, label in names.items():
            if cancelled():
                metadata["completion_status"] = "cancelled"
                write_metadata()
                return None
            report(f"Exporting {label} from {Path(source_file).name}")
            base = output_folder / f"{Path(source_file).stem}_{label}"
            image = to_imagej_compatible(dataset[channel])
            tf.imwrite(
                str(base) + ".tif", image, imagej=True,
                metadata={"axes": "TYX"} if image.ndim == 3 else None,
            )
            channel_params = copy.deepcopy(tracking_params)
            channel_params.name = str(base)
            channel_params.display_figures = False
            export_pystachio_params(
                channel_params, str(base) + "_PySTACHIO_params.dat"
            )
            if action == "analyse":
                if cancelled():
                    metadata["completion_status"] = "cancelled"
                    write_metadata()
                    return None
                report(f"Tracking {label} from {Path(source_file).name}")
                tracking.track(channel_params)
                if cancelled():
                    metadata["completion_status"] = "cancelled"
                    write_metadata()
                    return None
                if not _run_gui_postprocess(channel_params, report, cancelled):
                    metadata["completion_status"] = "cancelled"
                    write_metadata()
                    return None
        metadata["completion_status"] = "complete"
        write_metadata()
    except Exception:
        metadata["completion_status"] = "failed"
        write_metadata()
        raise
    return str(output_folder)



def _gui_estimate_isingle(params, trajs, report):
    """GUI adapter: retain backend cutoff/KDE bandwidth, return a finite scalar.

    A bounded grid avoids allocating one KDE point per camera count. The
    selected mode is refined locally; no changes to backend module globals.
    """
    from scipy.stats import gaussian_kde
    from scipy.optimize import minimize_scalar
    percent = float(getattr(params, "isingle_percent_frames", 66.6666667))
    if not np.isfinite(percent) or not 0 < percent <= 100:
        raise ValueError("isingle_percent_frames must be in (0, 100].")
    fallback = float(params.I_single)
    if not np.isfinite(fallback) or fallback <= 0:
        raise ValueError("The fallback I_single must be finite and positive.")
    cutoff = int(max(t.end_frame for t in trajs) * (1 - percent / 100))
    spots = postprocessing.trajectories.to_spots(trajs)
    samples = [np.asarray(t.spot_intensity, dtype=float).ravel()
               for t in spots if t.frame >= cutoff]
    values = np.concatenate(samples) if samples else np.array([], dtype=float)
    values = values[np.isfinite(values) & (values > 1)]
    np.savetxt(params.name + "_intensity_data.tsv", values)
    reason = None
    peak = fallback
    if values.size < 2 or np.ptp(values) == 0:
        reason = "fewer than two distinct finite intensities above 1"
    else:
        try:
            kde = gaussian_kde(values, bw_method=0.1)
            x = np.linspace(0, float(values.max()), 4096)
            pdf = kde(x)
            index = int(np.argmax(pdf))
            fit = minimize_scalar(lambda v: -float(kde([v])[0]),
                                  bounds=(x[max(0, index-1)], x[min(len(x)-1, index+1)]),
                                  method="bounded")
            peak = float(fit.x) if fit.success else float(x[index])
            if not np.isfinite(peak) or peak <= 0:
                raise ValueError("KDE returned an invalid intensity")
            fig = Figure()
            ax = fig.subplots()
            ax.hist(values, bins=min(100, max(1, int(np.sqrt(values.size)))), density=True,
                    color="gray", alpha=0.5)
            ax.plot(x, pdf, color="black")
            ax.axvline(peak, color="red", linestyle="--", label=f"Isingle={peak:.6g}")
            ax.set(xlabel="Intensity (camera counts)", ylabel="Probability density")
            ax.legend()
            fig.savefig(params.name + "_intensity_plot.png", dpi=150)
        except (ValueError, np.linalg.LinAlgError) as exc:
            reason = str(exc)
            peak = fallback
    result = {"value": peak, "source": "fallback" if reason else "estimated",
              "reason": reason, "sample_count": int(values.size), "cutoff_frame": cutoff,
              "grid_points": 4096, "bandwidth": 0.1}
    Path(params.name + "_i_single_peak.dat").write_text(str(peak), encoding="utf-8")
    Path(params.name + "_isingle_status.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    report(f"Isingle = {peak:.6g}" + (f"; using specified fallback: {reason}" if reason else " (estimated)"))
    return peak


def _gui_postprocess(params, report):
    """Adapt the GUI to the supplied backend without modifying backend code."""
    params = copy.deepcopy(params)
    params.display_figures = False
    channels = ("L", "R") if params.ALEX else (None,)
    tracks = {}
    for channel in channels:
        suffix = f"_{channel}channel" if channel else ""
        trajs = postprocessing.trajectories.read_trajectories(params.name + suffix + "_trajectories.tsv")
        tracks[channel] = trajs
        if not trajs:
            report(f"No trajectories for {channel or 'whole frame'}; postprocessing skipped.")
            continue
        local = copy.deepcopy(params)
        # Unique output stems also avoid backend L/R filename collisions.
        local.name = params.name + suffix
        local.I_single = float(getattr(params, channel + "_isingle")) if channel else float(params.I_single)
        if params.calculate_isingle:
            local.I_single = _gui_estimate_isingle(local, trajs, report)
        local.calculate_isingle = False
        report(f"Postprocessing {channel or 'whole frame'} with scalar Isingle={local.I_single:.6g}")
        postprocessing.get_diffusion_coef(trajs, local)
        # Backend plot_traj_intensities references undefined 'clickmode' when CK
        # is enabled. Plot here while using its unchanged numerical CK filter.
        fig = Figure(); ax = fig.subplots()
        filtered = []
        for traj in trajs:
            values = np.asarray(traj.intensity, dtype=float)
            ax.plot(values / 1000)
            if getattr(local, "chung_kennedy", False):
                window = int(local.chung_kennedy_window)
                if window < 1 or len(values) < window:
                    report(f"Track {traj.id}: too short for Chung-Kennedy window {window}; filter skipped.")
                else:
                    filtered.append(postprocessing.chung_kennedy_filter(values, window, 1)[0][:-1])
        ax.set(xlabel="Frame number", ylabel="Intensity (camera counts x 1000)")
        fig.savefig(local.name + "_trajectory_intensities.png", dpi=150)
        if filtered:
            import csv
            with open(local.name + "_chung_kennedy_data.csv", "w", newline="") as handle:
                csv.writer(handle).writerows(filtered)
            fig = Figure(); ax = fig.subplots()
            for values in filtered: ax.plot(values)
            ax.set(xlabel="Frame number", ylabel="Filtered intensity (camera counts)")
            fig.savefig(local.name + "_CK_filtered_intensities.png", dpi=150)
        # np.float64 is scalar but retains .ndim used by the Linear method.
        postprocessing.get_stoichiometries(trajs, np.float64(local.I_single), local)
        if params.copy_number:
            postprocessing.get_copy_number(params, local.I_single, channel=channel)
        postprocessing.overtrack(params, trajs, channel=channel)
    if params.ALEX and params.colocalize and all(tracks.values()):
        postprocessing.colocalize(params, tracks["L"], tracks["R"])


def _postprocess_child(params, connection):
    """Spawned process owns pyplot; Qt's global backend stays untouched."""
    try:
        import matplotlib.pyplot as plt
        plt.switch_backend("Agg")
        _gui_postprocess(params, lambda message: connection.send(("progress", message)))
        connection.send(("done", None))
    except BaseException:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


def _run_gui_postprocess(params, report, cancelled):
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_postprocess_child, args=(params, sender))
    process.start()
    sender.close()
    try:
        while True:
            if cancelled():
                process.terminate()
                return False
            if receiver.poll(0.1):
                try:
                    kind, message = receiver.recv()
                except EOFError:
                    raise RuntimeError("Postprocessing process exited without a result")
                if kind == "progress": report(message)
                elif kind == "error": raise RuntimeError(message)
                elif kind == "done": return True
            elif not process.is_alive():
                raise RuntimeError(f"Postprocessing exited unexpectedly ({process.exitcode})")
    finally:
        if process.is_alive():
            process.join(timeout=2)
        if process.is_alive(): process.terminate()
        process.join()
        receiver.close()


class TrackingWorker(QThread):
    progress = pyqtSignal(str)
    result = pyqtSignal(object)
    file_progress = pyqtSignal(int)

    def __init__(self, sources, config, params_snapshot, reference_hw,
                 registration_matrix, action, parent=None):
        super().__init__(parent)
        self.sources = tuple(sources)
        self.config = config
        self.params_snapshot = copy.deepcopy(params_snapshot)
        self.reference_hw = reference_hw
        self.registration_matrix = (
            np.array(registration_matrix, copy=True)
            if registration_matrix is not None else None
        )
        self.action = action
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True

    def run(self):
        successes, failures = [], []
        for index, path in enumerate(self.sources, 1):
            if self._cancel_requested:
                break
            self.progress.emit(f"[{index}/{len(self.sources)}] {Path(path).name}")
            try:
                result = _execute_one(
                    path, self.config, self.params_snapshot, self.reference_hw,
                    self.registration_matrix, self.action, self.progress.emit,
                    lambda: self._cancel_requested,
                )
                if result:
                    successes.append((path, result))
            except Exception:
                details = traceback.format_exc()
                LOG.error("Processing failed for %s:\n%s", path, details)
                failures.append((path, details))
            self.file_progress.emit(index)
        self.result.emit({
            "successful": successes,
            "failed": failures,
            "cancelled": self._cancel_requested,
            "total": len(self.sources),
            "action": self.action,
        })


class TiffAnalyzerTab(QWidget):
    def __init__(self, parent_suite=None):
        super().__init__()
        self.parent_suite = parent_suite
        self.raw_stack = None
        self.source_fname = None
        self.crop_coords = None
        self.registration_matrix = None
        self.current_frame = 0
        self.params = parameters.Parameters()
        self._worker = None
        self.selectors = []
        self.ax_single = self.ax_left = self.ax_right = None
        self.ax_donor = self.ax_acceptor = self.ax_fret = None
        self.im_single = self.im_left = self.im_right = None
        self.im_donor = self.im_acceptor = self.im_fret = None
        self.cb_single = self.cb_left = self.cb_right = None
        self.cb_donor = self.cb_acceptor = self.cb_fret = None
        self._spots_artists = []
        self.init_ui()
        self.set_defaults()
        self.destroyed.connect(self._on_tab_destroyed)

    def init_ui(self):
        main = QHBoxLayout(self)
        canvas_col = QVBoxLayout()
        self.fig = Figure(figsize=(10, 8), constrained_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        canvas_col.addWidget(self.toolbar)
        canvas_col.addWidget(self.canvas)
        slider_row = QHBoxLayout()
        self.slider_title_label = QLabel("Frame:")
        slider_row.addWidget(self.slider_title_label)
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self.on_frame_change)
        slider_row.addWidget(self.frame_slider)
        self.frame_label = QLabel("0 / 0")
        slider_row.addWidget(self.frame_label)
        canvas_col.addLayout(slider_row)
        main.addLayout(canvas_col, stretch=4)

        controls = QVBoxLayout()
        self.tabs = QTabWidget()
        self.tab_display, self.tab_tracking, self.tab_alex = QWidget(), QWidget(), QWidget()
        self.tabs.addTab(self.tab_display, "Display & Layout")
        self.tabs.addTab(self.tab_tracking, "Tracking Params")
        self.tabs.addTab(self.tab_alex, "ALEX Config")
        self.setup_display_tab()
        self.setup_tracking_tab()
        self.setup_alex_tab()
        controls.addWidget(self.tabs, stretch=1)

        execution = QGroupBox("Execution Pipeline")
        row = QVBoxLayout(execution)
        self.btn_find_spots = QPushButton("Find spots (current frame)")
        self.btn_find_spots.clicked.connect(self.find_spots)
        self.btn_export_data = QPushButton("Export processed image data stacks")
        self.btn_export_data.clicked.connect(self.export_image_data)
        self.btn_run_analysis = QPushButton("Run full PySTACHIO analysis")
        self.btn_run_analysis.setStyleSheet(
            "background-color: #2b5b84; color: white; font-weight: bold; padding: 10px;"
        )
        self.btn_run_analysis.clicked.connect(self.run_pystachio)
        self.btn_apply_workflow = QPushButton("Apply workflow (all files in this directory)")
        self.btn_apply_workflow.setStyleSheet(
            "background-color: #842b5b; color: white; font-weight: bold; padding: 10px;"
        )
        self.btn_apply_workflow.clicked.connect(self.apply_workflow_to_directory)
        self.chk_subdirs = QCheckBox("Include all subdirectories")
        self.chk_subdirs.setChecked(True)
        self.chk_skip_legacy = QCheckBox("Skip legacy _left/_right/_donor/_fret TIFFs")
        self.chk_skip_legacy.setChecked(True)
        self.btn_cancel = QPushButton("Cancel after current tracking stage")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_work)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_text = QLabel("")
        self.progress_text.setWordWrap(True)
        for widget in (self.btn_find_spots, self.btn_export_data, self.btn_run_analysis,
                       self.btn_apply_workflow, self.chk_subdirs, self.chk_skip_legacy,
                       self.btn_cancel,
                       self.progress_bar, self.progress_text):
            row.addWidget(widget)
        controls.addWidget(execution)
        main.addLayout(controls, stretch=1)

    def setup_display_tab(self):
        layout = QVBoxLayout(self.tab_display)
        self.btn_load = QPushButton("Load TIF Stack File")
        self.btn_load.clicked.connect(self.load_file)
        self.btn_reset_crop = QPushButton("Reset Region of Interest (Crop)")
        self.btn_reset_crop.clicked.connect(self.reset_crop)
        layout.addWidget(self.btn_load)
        layout.addWidget(self.btn_reset_crop)

        mode_group = QGroupBox("Analysis Mode Selection")
        mode_layout = QVBoxLayout(mode_group)
        self.mode_cluster = QButtonGroup(self)
        self.rb_mode_single = QRadioButton("Single View (Full Frame)")
        self.rb_mode_dual = QRadioButton("Dual Channel (Vertical Split)")
        self.rb_mode_fret = QRadioButton("ALEX/FRET View (3 Channel Spatial/Temporal)")
        self.rb_mode_single.setChecked(True)
        for widget in (self.rb_mode_single, self.rb_mode_dual, self.rb_mode_fret):
            self.mode_cluster.addButton(widget)
            mode_layout.addWidget(widget)
            widget.toggled.connect(self.on_structural_setting_changed)
        layout.addWidget(mode_group)

        temporal = QGroupBox("Temporal deinterleaving")
        temporal_layout = QVBoxLayout(temporal)
        self.cb_alex = QCheckBox("In dual mode, extract one ALEX excitation phase")
        self.cb_alex.setEnabled(False)
        self.cb_alex.stateChanged.connect(self.on_structural_setting_changed)
        self.temporal_cluster = QButtonGroup(self)
        self.rb_temp_even = QRadioButton("Selected / donor-excitation frames are even")
        self.rb_temp_odd = QRadioButton("Selected / donor-excitation frames are odd")
        self.rb_temp_even.setChecked(True)
        for radio in (self.rb_temp_even, self.rb_temp_odd):
            self.temporal_cluster.addButton(radio)
            temporal_layout.addWidget(radio)
            radio.setEnabled(False)
            radio.toggled.connect(self.on_structural_setting_changed)
        temporal_layout.insertWidget(0, self.cb_alex)
        layout.addWidget(temporal)

        spatial = QGroupBox("Spatial Orientation (FRET)")
        spatial_layout = QVBoxLayout(spatial)
        self.spatial_cluster = QButtonGroup(self)
        self.rb_orient_norm = QRadioButton("Left: donor | Right: FRET/acceptor")
        self.rb_orient_inv = QRadioButton("Left: FRET/acceptor | Right: donor")
        self.rb_orient_norm.setChecked(True)
        for radio in (self.rb_orient_norm, self.rb_orient_inv):
            self.spatial_cluster.addButton(radio)
            spatial_layout.addWidget(radio)
            radio.setEnabled(False)
            radio.toggled.connect(self.on_structural_setting_changed)
        layout.addWidget(spatial)

        calibration = QGroupBox("Calibration & Registration")
        calibration_layout = QVBoxLayout(calibration)
        self.btn_registration = QPushButton("Calculate Registration Matrix")
        self.btn_clear_registration = QPushButton("Clear Registration Matrix")
        self.btn_registration.clicked.connect(
            self.parent_suite.run_registration if self.parent_suite else lambda: None
        )
        self.btn_clear_registration.clicked.connect(
            self.parent_suite.clear_registration if self.parent_suite else lambda: None
        )
        calibration_layout.addWidget(self.btn_registration)
        calibration_layout.addWidget(self.btn_clear_registration)
        note = QLabel("Registration is saved as calibration data; this tab does not "
                      "transform images or tracked coordinates.")
        note.setWordWrap(True)
        calibration_layout.addWidget(note)
        layout.addWidget(calibration)

        scaling = QGroupBox("Display intensity")
        scaling_layout = QFormLayout(scaling)
        self.chk_fixed_contrast = QCheckBox("Fixed intensity range")
        self.contrast_min = QLineEdit("0")
        self.contrast_max = QLineEdit("1000")
        for widget in (self.contrast_min, self.contrast_max):
            widget.setValidator(QDoubleValidator(-1e12, 1e12, 6, widget))
            widget.editingFinished.connect(self.update_display_frame_only)
        self.chk_fixed_contrast.toggled.connect(self.update_display_frame_only)
        scaling_layout.addRow(self.chk_fixed_contrast)
        scaling_layout.addRow("Minimum:", self.contrast_min)
        scaling_layout.addRow("Maximum:", self.contrast_max)
        layout.addWidget(scaling)
        layout.addStretch()

    def setup_tracking_tab(self):
        layout = QVBoxLayout(self.tab_tracking)
        grid = QGridLayout()
        fields = [
            ("num_procs", "num_procs_box", True),
            ("Camera frame interval (s)", "frame_time_box", False),
            ("Pixel size (µm)", "pixel_size_box", False),
            ("Start frame (raw 0-based)", "start_frame_box", True),
            ("I_single", "isingle_box", False),
            ("SNR cutoff", "snr_box", False),
            ("max_displacement (pixels)", "maxdisp_box", True),
            ("min_traj_length", "mintraj_box", True),
            ("bw_threshold_tolerance", "bwthresh_box", False),
            ("struct_disk_radius", "sdisk_box", True),
            ("subarray_halfwidth", "subarray_box", True),
            ("inner_mask_radius", "innermask_box", True),
        ]
        for row, (label, attr, integer) in enumerate(fields):
            widget = QLineEdit()
            if integer:
                widget.setValidator(QIntValidator(0, 2_000_000_000, widget))
            else:
                widget.setValidator(QDoubleValidator(0.0, 1e12, 9, widget))
            setattr(self, attr, widget)
            grid.addWidget(QLabel(label + ":"), row, 0)
            grid.addWidget(widget, row, 1)
        layout.addLayout(grid)
        group = QGroupBox("Calculate I_single?")
        horizontal = QHBoxLayout(group)
        self.calc_isingle_yes = QRadioButton("Yes")
        self.calc_isingle_no = QRadioButton("No")
        horizontal.addWidget(self.calc_isingle_yes)
        horizontal.addWidget(self.calc_isingle_no)
        layout.addWidget(group)
        self.btn_defaults = QPushButton("Restore Parameter Defaults")
        self.btn_defaults.clicked.connect(self.set_defaults)
        layout.addWidget(self.btn_defaults)
        layout.addStretch()

    def setup_alex_tab(self):
        layout = QVBoxLayout(self.tab_alex)
        group = QGroupBox("Enable ALEX processing in backend?")
        horizontal = QHBoxLayout(group)
        self.backend_alex_yes = QRadioButton("Yes")
        self.backend_alex_no = QRadioButton("No")
        horizontal.addWidget(self.backend_alex_yes)
        horizontal.addWidget(self.backend_alex_no)
        layout.addWidget(group)
        note = QLabel(
            "Backend ALEX applies only to an unsplit, interleaved TIFF. "
            "When this tab exports phase-separated images, ALEX and backend "
            "colocalization are disabled for those independent tracking jobs."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        form_group = QGroupBox("Backend ALEX colocalization parameters")
        form = QFormLayout(form_group)
        self.start_channel_box = QLineEdit()
        self.l_isingle_box = QLineEdit()
        self.r_isingle_box = QLineEdit()
        self.colocalize_dist_box = QLineEdit()
        self.colocalize_n_box = QLineEdit()
        for widget in (self.l_isingle_box, self.r_isingle_box):
            widget.setValidator(QDoubleValidator(0.0, 1e12, 9, widget))
        for widget in (self.colocalize_dist_box, self.colocalize_n_box):
            widget.setValidator(QIntValidator(0, 2_000_000_000, widget))
        self.colocalize_yes = QRadioButton("True")
        self.colocalize_no = QRadioButton("False")
        colocalize_widget = QWidget()
        colocalize_layout = QHBoxLayout(colocalize_widget)
        colocalize_layout.addWidget(self.colocalize_yes)
        colocalize_layout.addWidget(self.colocalize_no)
        form.addRow("Start channel (L/R):", self.start_channel_box)
        form.addRow("L_isingle:", self.l_isingle_box)
        form.addRow("R_isingle:", self.r_isingle_box)
        form.addRow("Colocalize:", colocalize_widget)
        form.addRow("Colocalize distance:", self.colocalize_dist_box)
        form.addRow("Colocalize n_frames:", self.colocalize_n_box)
        layout.addWidget(form_group)
        layout.addStretch()

    def set_defaults(self):
        defaults = {
            "num_procs_box": "1", "frame_time_box": "0.005",
            "pixel_size_box": "0.120", "start_frame_box": "0",
            "isingle_box": "120", "snr_box": "0.4", "maxdisp_box": "5",
            "mintraj_box": "3", "bwthresh_box": "1.0", "sdisk_box": "5",
            "subarray_box": "8", "innermask_box": "5",
            "start_channel_box": "L", "l_isingle_box": "10000.0",
            "r_isingle_box": "10000.0", "colocalize_dist_box": "5",
            "colocalize_n_box": "5",
        }
        for key, value in defaults.items():
            getattr(self, key).setText(value)
        self.calc_isingle_no.setChecked(True)
        self.backend_alex_no.setChecked(True)
        self.colocalize_no.setChecked(True)

    def _get_config(self):
        mode = "single" if self.rb_mode_single.isChecked() else (
            "dual" if self.rb_mode_dual.isChecked() else "fret"
        )
        return TrackingViewConfig(
            mode=mode, crop=self.crop_coords,
            dual_alex=mode == "dual" and self.cb_alex.isChecked(),
            first_even=self.rb_temp_even.isChecked(),
            donor_left=self.rb_orient_norm.isChecked(),
        )

    def fetch_params_from_ui(self):
        """Validate a complete parameter snapshot before changing self.params."""
        params = copy.deepcopy(self.params)
        numeric_fields = (
            ("num_procs", "num_procs_box", True, 1, False),
            ("frame_time", "frame_time_box", False, 0, True),
            ("pixel_size", "pixel_size_box", False, 0, True),
            ("start_frame", "start_frame_box", True, 0, False),
            ("I_single", "isingle_box", False, 0, True),
            ("snr_filter_cutoff", "snr_box", False, 0, False),
            ("max_displacement", "maxdisp_box", True, 0, True),
            ("min_traj_len", "mintraj_box", True, 1, False),
            ("bw_threshold_tolerance", "bwthresh_box", False, 0, False),
            ("struct_disk_radius", "sdisk_box", True, 0, True),
            ("subarray_halfwidth", "subarray_box", True, 0, True),
            ("inner_mask_radius", "innermask_box", True, 0, True),
            ("L_isingle", "l_isingle_box", False, 0, True),
            ("R_isingle", "r_isingle_box", False, 0, True),
            ("colocalize_distance", "colocalize_dist_box", True, 0, False),
            ("colocalize_n_frames", "colocalize_n_box", True, 1, False),
        )
        for param_name, widget_name, integer, minimum, strict in numeric_fields:
            value = _safe_number(
                getattr(self, widget_name).text().strip(),
                param_name, integer=integer, minimum=minimum, strict=strict,
            )
            setattr(params, param_name, value)
        start_channel = self.start_channel_box.text().strip().upper()
        if start_channel not in ("L", "R"):
            raise ValueError("Start channel must be L or R.")
        params.start_channel = start_channel
        params.calculate_isingle = self.calc_isingle_yes.isChecked()
        params.ALEX = self.backend_alex_yes.isChecked()
        params.colocalize = self.colocalize_yes.isChecked()
        available_cpus = os.cpu_count()
        if available_cpus is not None and params.num_procs > available_cpus:
            raise ValueError(
                f"num_procs exceeds this computer's reported {available_cpus} CPU cores."
            )
        if params.inner_mask_radius > params.subarray_halfwidth:
            raise ValueError("inner_mask_radius must not exceed subarray_halfwidth.")
        self.params = params
        return copy.deepcopy(params)

    def _reset_frame(self):
        self.current_frame = 0
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)

    def load_file(self):
        if self._worker is not None:
            return
        fname, _ = QFileDialog.getOpenFileName(
            self, "Open TIF Stack", filter="TIFF Files (*.tif *.tiff *.TIF *.TIFF)"
        )
        if not fname:
            return
        try:
            stack = _read_stack(fname)
            if self.parent_suite:
                self.parent_suite.reset_entire_suite()
            self.raw_stack = stack
            self.source_fname = fname
            self.params.name = str(Path(fname).with_suffix(""))
            self.crop_coords = None
            self._reset_frame()
            self.update_display()
        except Exception as exc:
            show_popup(self, "Loading Error", f"Failed to open file:\n{exc}", critical=True)

    def reset_crop(self):
        if self.raw_stack is not None:
            self.crop_coords = None
            self._reset_frame()
            self.update_display()

    def on_structural_setting_changed(self, *_args):
        self.crop_coords = None
        self._reset_frame()
        is_dual, is_fret = self.rb_mode_dual.isChecked(), self.rb_mode_fret.isChecked()
        self.cb_alex.setEnabled(is_dual)
        use_phase = is_fret or (is_dual and self.cb_alex.isChecked())
        self.rb_temp_even.setEnabled(use_phase)
        self.rb_temp_odd.setEnabled(use_phase)
        self.rb_orient_norm.setEnabled(is_fret)
        self.rb_orient_inv.setEnabled(is_fret)
        if hasattr(self, "backend_alex_yes"):
            for control in (self.backend_alex_yes, self.backend_alex_no,
                            self.colocalize_yes, self.colocalize_no):
                control.setEnabled(self.rb_mode_single.isChecked())
        self.update_display()

    def on_frame_change(self, value):
        self.current_frame = value
        self.update_display_frame_only()

    def process_data(self):
        if self.raw_stack is None:
            raise ValueError("Load a TIFF first.")
        return _prepare_stack(self.raw_stack, self._get_config())

    def apply_crop_callback(self, eclick, erelease):
        if self.raw_stack is None or None in (
            eclick.xdata, erelease.xdata, eclick.ydata, erelease.ydata
        ):
            return
        _, height, width = self.raw_stack.shape
        if not self.rb_mode_single.isChecked():
            width //= 2
        previous = self.crop_coords
        base_x, base_y = (previous[0], previous[2]) if previous else (0, 0)
        display_width = (previous[1] - previous[0]) if previous else width
        display_height = (previous[3] - previous[2]) if previous else height
        x0 = max(0, min(display_width, math.floor(min(eclick.xdata, erelease.xdata) + 0.5)))
        x1 = max(0, min(display_width, math.ceil(max(eclick.xdata, erelease.xdata) + 0.5)))
        y0 = max(0, min(display_height, math.floor(min(eclick.ydata, erelease.ydata) + 0.5)))
        y1 = max(0, min(display_height, math.ceil(max(eclick.ydata, erelease.ydata) + 0.5)))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return
        crop = (x0 + base_x, x1 + base_x, y0 + base_y, y1 + base_y)
        try:
            _validate_crop(crop, height, width)
        except ValueError as exc:
            show_popup(self, "Invalid crop", str(exc), critical=True)
            return
        self.crop_coords = crop
        self._reset_frame()
        self.update_display()

    def _clear_selectors(self):
        for selector in self.selectors:
            selector.set_active(False)
            selector.disconnect_events()
        self.selectors.clear()
        self._spots_artists.clear()

    def update_display(self):
        if self.raw_stack is None:
            return
        self._clear_selectors()
        self.fig.clear()
        self.ax_single = self.ax_left = self.ax_right = None
        self.ax_donor = self.ax_acceptor = self.ax_fret = None
        self.im_single = self.im_left = self.im_right = None
        self.im_donor = self.im_acceptor = self.im_fret = None
        self.cb_single = self.cb_left = self.cb_right = None
        self.cb_donor = self.cb_acceptor = self.cb_fret = None
        try:
            dataset = self.process_data()
        except ValueError as exc:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, str(exc), ha="center", va="center", wrap=True)
            ax.set_axis_off()
            self.frame_slider.setEnabled(False)
            self.frame_label.setText("No usable frames")
            self.canvas.draw_idle()
            return
        self.slider_title_label.setText(
            "ALEX cycle index:" if dataset["mode"] == "fret" else
            "Selected excitation index:" if dataset["deinterleaved"] else "Frame:"
        )
        if dataset["mode"] == "single":
            self.ax_single = self.fig.add_subplot(111)
            axes = [self.ax_single]
        elif dataset["mode"] == "dual":
            self.ax_left, self.ax_right = self.fig.subplots(1, 2)
            axes = [self.ax_left, self.ax_right]
        else:
            self.ax_donor, self.ax_acceptor, self.ax_fret = self.fig.subplots(1, 3)
            axes = [self.ax_donor, self.ax_acceptor, self.ax_fret]
        count = min(len(dataset[key]) for key in _channel_names(dataset))
        self.frame_slider.setEnabled(count > 0)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setMaximum(max(0, count - 1))
        self.current_frame = min(self.current_frame, max(0, count - 1))
        self.frame_slider.setValue(self.current_frame)
        self.frame_slider.blockSignals(False)
        for ax in axes:
            self.selectors.append(
                RectangleSelector(ax, self.apply_crop_callback, useblit=True,
                                  button=[1], minspanx=5, minspany=5,
                                  interactive=True)
            )
        self.update_display_frame_only()

    def _display_limits(self, image):
        if self.chk_fixed_contrast.isChecked():
            try:
                lower = float(self.contrast_min.text())
                upper = float(self.contrast_max.text())
                if math.isfinite(lower) and math.isfinite(upper) and lower < upper:
                    return lower, upper
            except ValueError:
                pass
        lower, upper = float(np.min(image)), float(np.max(image))
        if not math.isfinite(lower) or not math.isfinite(upper):
            return 0.0, 1.0
        return lower, upper if upper > lower else lower + 1.0

    def _render_channel(self, ax, image_handle, colorbar, img_data, title):
        lower, upper = self._display_limits(img_data)
        if image_handle is None:
            ax.set_title(title)
            image_handle = ax.imshow(img_data, cmap="Greys_r", aspect="equal",
                                     vmin=lower, vmax=upper)
            colorbar = self.fig.colorbar(image_handle, ax=ax, fraction=0.046, pad=0.04)
        else:
            image_handle.set_data(img_data)
            image_handle.set_clim(vmin=lower, vmax=upper)
            colorbar.update_normal(image_handle)
        return image_handle, colorbar

    def _clear_spot_overlays(self):
        for artist in self._spots_artists:
            try:
                artist.remove()
            except ValueError:
                pass
        self._spots_artists.clear()

    def update_display_frame_only(self, *_args):
        if self.raw_stack is None:
            return
        try:
            dataset = self.process_data()
        except ValueError:
            return
        idx = self.current_frame
        self._clear_spot_overlays()
        mapping = (
            [("ch1", "ax_single", "im_single", "cb_single", "Full frame")]
            if dataset["mode"] == "single" else
            [("ch1", "ax_left", "im_left", "cb_left", "Left channel"),
             ("ch2", "ax_right", "im_right", "cb_right", "Right channel")]
            if dataset["mode"] == "dual" else
            [("ch1", "ax_donor", "im_donor", "cb_donor", "Donor excitation / donor emission"),
             ("ch2", "ax_acceptor", "im_acceptor", "cb_acceptor", "Acceptor excitation / emission"),
             ("ch3", "ax_fret", "im_fret", "cb_fret", "Donor excitation / acceptor emission")]
        )
        for key, axis_name, image_name, colorbar_name, label in mapping:
            if idx >= len(dataset[key]):
                continue
            ax = getattr(self, axis_name)
            if ax is None:
                continue
            handle, colorbar = self._render_channel(
                ax, getattr(self, image_name), getattr(self, colorbar_name),
                dataset[key][idx], label,
            )
            setattr(self, image_name, handle)
            setattr(self, colorbar_name, colorbar)
        indices = ", ".join(
            f"{key}={offset + idx * dataset['frame_step']}"
            for key, offset in dataset["raw_offsets"].items()
        )
        self.frame_label.setText(
            f"{idx} / {self.frame_slider.maximum()} (raw: {indices})"
        )
        self.canvas.draw_idle()

    def _preview_images(self, dataset):
        """Read PySTACHIO image initialization once for multiple preview channels.

        ImageData.read may initialize additional framework state, so it cannot
        safely be replaced without an explicit in-memory PySTACHIO API.
        """
        base = images.ImageData()
        preview_params = copy.deepcopy(self.params)
        preview_params.name = os.path.splitext(self.source_fname)[0]
        base.read(self.source_fname, preview_params)
        for key in _channel_names(dataset):
            array = dataset[key]
            img = copy.copy(base)
            img.pixel_data = array
            img.num_frames = array.shape[0]
            img.frame_size = [array.shape[2], array.shape[1]]
            img.has_mask = True
            img.mask_data = np.ones(array.shape[1:], dtype=np.uint8)
            yield key, img

    def find_spots(self):
        if self.raw_stack is None or self._worker is not None:
            return
        try:
            self.fetch_params_from_ui()
            dataset = self.process_data()
            idx = self.current_frame
            if idx >= len(dataset["ch1"]):
                raise ValueError("Selected frame is outside the processed stack.")
            self._clear_spot_overlays()
            axes = (
                {"ch1": self.ax_single} if dataset["mode"] == "single" else
                {"ch1": self.ax_left, "ch2": self.ax_right} if dataset["mode"] == "dual" else
                {"ch1": self.ax_donor, "ch2": self.ax_acceptor, "ch3": self.ax_fret}
            )
            preview_params = copy.deepcopy(self.params)
            preview_params.frame_time *= dataset["frame_step"]
            preview_params.start_frame = _processed_start_frame(
                self._get_config(), self.params.start_frame
            )
            if dataset["mode"] != "single":
                preview_params.ALEX = False
                preview_params.colocalize = False
            for key, image in self._preview_images(dataset):
                result = tracking.track_frame(image[idx], idx, preview_params)
                if result is None or len(result.positions) == 0:
                    continue
                positions = np.atleast_2d(np.asarray(result.positions))
                scatter = axes[key].scatter(
                    positions[:, 0], positions[:, 1], marker="x", color="blue", s=40
                )
                self._spots_artists.append(scatter)
            self.canvas.draw_idle()
        except Exception as exc:
            LOG.exception("Spot preview failed")
            show_popup(self, "Spot Preview Error", str(exc), critical=True)

    def _begin_worker(self, sources, action):
        if self._worker is not None or not sources:
            return
        try:
            params_snapshot = self.fetch_params_from_ui()
            config = self._get_config()
            if self.raw_stack is None:
                raise ValueError("Load a TIFF first.")
            _prepare_stack(self.raw_stack, config)
            reference_hw = self.raw_stack.shape[1:]
        except Exception as exc:
            show_popup(self, "Invalid analysis settings", str(exc), critical=True)
            return
        self._worker = TrackingWorker(
            sources, config, params_snapshot, reference_hw,
            self.registration_matrix, action, self,
        )
        for selector in self.selectors:
            selector.set_active(False)
        self._worker.progress.connect(self._on_progress)
        self._worker.result.connect(self._on_worker_result)
        self._worker.file_progress.connect(self.progress_bar.setValue)
        self._worker.finished.connect(self._on_worker_finished)
        self.progress_bar.setRange(0, len(sources))
        self.progress_bar.setValue(0)
        self.progress_text.setText("Starting...")
        self.btn_cancel.setEnabled(True)
        self.tabs.setEnabled(False)
        self.btn_find_spots.setEnabled(False)
        self.btn_run_analysis.setEnabled(False)
        self.btn_export_data.setEnabled(False)
        self.btn_apply_workflow.setEnabled(False)
        self._worker.start()

    def _on_progress(self, text):
        self.progress_text.setText(text)

    def cancel_work(self):
        if self._worker:
            self._worker.cancel()
            self.btn_cancel.setEnabled(False)
            self.progress_text.setText("Cancellation requested; finishing current PySTACHIO call...")

    def _on_worker_result(self, stats):
        successful, failed = stats["successful"], stats["failed"]
        self.progress_bar.setValue(len(successful) + len(failed))
        self.progress_text.setText(
            f"Completed: {len(successful)}; Failed: {len(failed)}; "
            f"Cancelled: {'yes' if stats['cancelled'] else 'no'}"
        )
        report = [
            f"Files requested: {stats['total']}",
            f"Successful: {len(successful)}",
            f"Failed: {len(failed)}",
            f"Cancelled: {'Yes' if stats['cancelled'] else 'No'}",
        ]
        if successful:
            report.extend(["", "Output folders:"] + [folder for _, folder in successful[:4]])
            if len(successful) > 4:
                report.append(f"...and {len(successful) - 4} more.")
        if failed:
            report.extend(["", "Failures:"] + [
                f"{Path(path).name}: {details.strip().splitlines()[-1]}"
                for path, details in failed[:8]
            ])
            if len(failed) > 8:
                report.append(f"...and {len(failed) - 8} additional failures.")
            # Persist full tracebacks without interfering with any source TIFF.
            try:
                log_folder = Path(self.source_fname).parent / "PySTACHIO_results"
                log_folder.mkdir(exist_ok=True)
                log_path = log_folder / (
                    f"tracking_failures_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}.log"
                )
                log_path.write_text("\n\n".join(
                    f"{path}\n{details}" for path, details in failed
                ), encoding="utf-8")
                report.extend(["", f"Full error log: {log_path}"])
            except Exception:
                LOG.exception("Could not save failure report")
        show_popup(
            self, "Tracking task results", "\n".join(report),
            critical=bool(failed),
        )

    def _on_worker_finished(self):
        self.tabs.setEnabled(True)
        for button in (self.btn_find_spots, self.btn_run_analysis,
                       self.btn_export_data, self.btn_apply_workflow):
            button.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        for selector in self.selectors:
            selector.set_active(True)
        if self._worker:
            self._worker.deleteLater()
            self._worker = None

    def run_pystachio(self):
        if self.source_fname:
            self._begin_worker([self.source_fname], "analyse")

    def export_image_data(self):
        if self.source_fname:
            self._begin_worker([self.source_fname], "export")

    def apply_workflow_to_directory(self):
        if not self.source_fname:
            show_popup(self, "No input", "Load a reference TIFF first.", critical=True)
            return
        directory = Path(self.source_fname).parent
        paths = directory.rglob("*") if self.chk_subdirs.isChecked() else directory.iterdir()
        excluded = ("_cropped", "_left", "_right", "_donor", "_acceptor", "_fret")
        files = sorted({
            str(path) for path in paths
            if path.is_file() and path.suffix.lower() in (".tif", ".tiff")
            and "PySTACHIO_results" not in path.parts
            and (not self.chk_skip_legacy.isChecked() or
                 not path.stem.lower().endswith(excluded))
        })
        if not files:
            show_popup(self, "No TIFFs", "No source TIFF files found.")
            return
        answer = QMessageBox.question(
            self, "Start batch tracking",
            f"Process {len(files)} TIFF files? The current crop, analysis mode "
            "and parameter settings will be applied to every file. "
            "Each result will be stored in a new folder.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._begin_worker(files, "analyse")

    def _on_tab_destroyed(self, *_args):
        """Avoid destroying a Qt worker thread before its backend call exits."""
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.cancel()
            worker.wait()

    def closeEvent(self, event):
        if self._worker is not None and self._worker.isRunning():
            show_popup(
                self, "Tracking is running",
                "Cancel the task and allow the current PySTACHIO operation to finish before closing.",
            )
            event.ignore()
        else:
            self._clear_selectors()
            super().closeEvent(event)
