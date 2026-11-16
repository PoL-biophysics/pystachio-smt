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
import glob
import csv
import re
import shutil
import subprocess
from pathlib import Path
import traceback
from uuid import uuid4

import numpy as np
import tifffile as tf
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import RectangleSelector

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QSplitter,
                             QHBoxLayout, QPushButton, QSlider, QLineEdit,
                             QLabel, QRadioButton, QGroupBox, QFileDialog, 
                             QButtonGroup, QFormLayout, QMessageBox, QCheckBox,
                             QTabWidget, QGridLayout, QToolTip, QComboBox,
                             QScrollArea, QProgressBar, QLayout, QSizePolicy,)
from PyQt6.QtCore import Qt
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QDoubleValidator, QIntValidator

from pystackreg import StackReg

import images
import parameters
import postprocessing
import tracking
from studio.studio_helpers import (
    export_pystachio_params, show_popup, to_imagej_compatible,
)

LOG = logging.getLogger(__name__)

# Only the parameters in parameters.py's "New tracking and linking" section.
# name, label, default, integer, strictly positive
NEW_TRACKING_GROUPS = (
    ("Spot fitting", ()),
    ("Spot detection", (
        ("presmooth_sigma", "Presmoothing sigma", 0.0, False, False),
        ("psf_sigma", "PSF sigma", 2.0, False, True),
        ("detect_thresh_k", "Detection threshold k", 3.0, False, False),
        ("min_separation", "Minimum separation", 5, True, False),
    )),
    ("Trajectory linking", (
        ("d_01_max", "Maximum link distance", 6, True, False),
        ("Iratio_01_min", "Minimum intensity ratio", 0.5, False, False),
        ("Iratio_01_max", "Maximum intensity ratio", 3.0, False, False),
        ("max_gap", "Maximum gap (frames)", 2, True, False),
        ("gap_close_dist", "Gap closing distance", 8, True, False),
    )),
)
NEW_TRACKING_NAMES = tuple(
    field[0] for _, fields in NEW_TRACKING_GROUPS for field in fields
) + ("detect_scales", "spot_detection_model", "linking_model")


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
            "cropped": crop is not None,
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
        return {"ch1": "cropped" if dataset["cropped"] else "full_frame"}
    if dataset["mode"] == "dual":
        return {"ch1": "left", "ch2": "right"}
    return {"ch1": "donor", "ch2": "acceptor", "ch3": "fret"}


def _new_results_folder(source_file: str, output_directory=None) -> Path:
    folder = (Path(output_directory) if output_directory is not None
              else Path(source_file).parent / "PySTACHIO_analysis").resolve()
    folder.mkdir(parents=True, exist_ok=True)
    return folder

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

def _output_prefix(folder: Path, source_file: str) -> str:
    """Keep duplicate input names and repeated runs separate in a flat folder."""
    stem = Path(source_file).stem
    existing = [path.name.casefold() for path in folder.iterdir()]
    candidate, index = stem, 2
    while any(name.startswith(candidate.casefold() + "_") for name in existing):
        candidate = f"{stem}_{index}"
        index += 1
    return candidate


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
                "colocalize_n_frames", "astigmatism",
            ) + NEW_TRACKING_NAMES
        },
    }


def _execute_one(source_file, config, params_snapshot, reference_hw,
                 registration_matrix, action, report, cancelled,
                 output_directory=None):
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
    output_folder = _new_results_folder(source_file, output_directory)
    output_prefix = _output_prefix(output_folder, source_file)
    names = _channel_names(dataset)
    metadata = _metadata(source_file, stack, dataset, config, params_snapshot,
                         tracking_params, registration_matrix, action)
    metadata["completion_status"] = "in_progress"
    metadata_path = output_folder / f"{output_prefix}_analysis_metadata.json"

    def write_metadata():
        metadata_path.write_text(
            json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
        )

    write_metadata()
    if registration_matrix is not None:
        np.savetxt(output_folder / f"{output_prefix}_registration_matrix_UNAPPLIED.txt", registration_matrix)

    # imagej-compatible stacks saved as .tif regardless of source extension.
    try:
        for channel, label in names.items():
            if cancelled():
                metadata["completion_status"] = "cancelled"
                write_metadata()
                return None
            report(f"Exporting {label} from {Path(source_file).name}")
            base = output_folder / f"{output_prefix}_{label}"
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
    percent = float(getattr(params, "isingle_fraction", 66.6666667))
    if not np.isfinite(percent) or not 0 < percent <= 100:
        raise ValueError("isingle_fraction must be in (0, 100].")
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
    original_savefig = None
    try:
        import matplotlib.pyplot as plt
        plt.switch_backend("Agg")
        original_savefig = plt.savefig

        def save_output(filename, *args, **kwargs):
            # The backend's colocalization plot has a bare filename. Route
            # relative plot outputs into the same uniquely prefixed results.
            if isinstance(filename, (str, os.PathLike)) and not Path(filename).is_absolute():
                filename = str(params.name) + "_" + Path(filename).name
            return original_savefig(filename, *args, **kwargs)

        plt.savefig = save_output
        _gui_postprocess(params, lambda message: connection.send(("progress", message)))
        connection.send(("done", None))
    except BaseException:
        connection.send(("error", traceback.format_exc()))
    finally:
        if original_savefig is not None:
            plt.savefig = original_savefig
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
                 registration_matrix, action, parent=None, output_directory=None):
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
        self.output_directory = output_directory
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
                    self.output_directory,
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
            "output_directory": self.output_directory,
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
        
        # Raw config cache for preserving unedited lines/formatting when saving
        self.raw_config_lines = None
        self.original_config_dict = {}

        self.ax_single = self.ax_left = self.ax_right = self.ax_donor = self.ax_acceptor = self.ax_fret = None
        self.im_single = self.im_left = self.im_right = self.im_donor = self.im_acceptor = self.im_fret = None
        self.cb_single = self.cb_left = self.cb_right = self.cb_donor = self.cb_acceptor = self.cb_fret = None
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

        controls_widget = QWidget()
        controls = QVBoxLayout(controls_widget)
        controls.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self.tabs = QTabWidget()
        self.tabs.setMinimumHeight(340)
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
        self.btn_apply_workflow = QPushButton("Apply workflow to directory")
        self.btn_apply_workflow.setStyleSheet(
            "background-color: #842b5b; color: white; font-weight: bold; padding: 10px;"
        )
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
        self.btn_load_registration = QPushButton("Load Registration Matrix")
        load_registration = getattr(self.parent_suite, "load_registration", None)
        self.btn_load_registration.setEnabled(callable(load_registration))
        if callable(load_registration):
            self.btn_load_registration.clicked.connect(load_registration)
        self.btn_clear_registration = QPushButton("Clear Registration Matrix")
        self.btn_registration.clicked.connect(
            self.parent_suite.run_registration if self.parent_suite else lambda: None
        )
        self.btn_clear_registration.clicked.connect(
            self.parent_suite.clear_registration if self.parent_suite else lambda: None
        )
        calibration_layout.addWidget(self.btn_registration)
        calibration_layout.addWidget(self.btn_load_registration)
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
        fields = [
            ("num_procs", "num_procs_box", True),
            ("Camera frame interval (s)", "frame_time_box", False),
            ("Pixel size (µm)", "pixel_size_box", False),
            ("Start frame (raw 0-based)", "start_frame_box", True),
            ("I_single", "isingle_box", False),
            ("SNR cutoff", "snr_box", False),
            ("min_traj_length", "mintraj_box", True),
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
        self.new_tracking_boxes = {}
        self.new_tracking_models = {}
        self.model_parameter_panels = {}
        for title, fields in NEW_TRACKING_GROUPS:
            group = QGroupBox(title)
            form = QFormLayout(group)
            if title == "Spot fitting":
                self.astigmatism_buttons = QButtonGroup(self)
                astigmatism_widget = QWidget()
                astigmatism_row = QHBoxLayout(astigmatism_widget)
                astigmatism_row.setContentsMargins(0, 0, 0, 0)
                self.astigmatism_true = QRadioButton("True")
                self.astigmatism_false = QRadioButton("False")
                for button in (self.astigmatism_true, self.astigmatism_false):
                    self.astigmatism_buttons.addButton(button)
                    astigmatism_row.addWidget(button)
                form.addRow("Astigmatism:", astigmatism_widget)
                for label, attr in (("struct_disk_radius", "sdisk_box"),
                                    ("subarray_halfwidth", "subarray_box")):
                    box = QLineEdit()
                    box.setValidator(QIntValidator(0, 2_000_000_000, box))
                    setattr(self, attr, box)
                    form.addRow(label + ":", box)
            if title in ("Spot detection", "Trajectory linking"):
                name, choices = (("spot_detection_model", ("tophat", "log"))
                                 if title == "Spot detection" else
                                 ("linking_model", ("legacy", "gapped")))
                buttons = QButtonGroup(self)
                row_widget = QWidget()
                row = QHBoxLayout(row_widget)
                row.setContentsMargins(0, 0, 0, 0)
                options = {}
                for choice in choices:
                    button = QRadioButton(choice)
                    buttons.addButton(button)
                    row.addWidget(button)
                    options[choice] = button
                self.new_tracking_models[name] = options
                row_widget.setToolTip(name)
                form.addRow("Model:", row_widget)
                panels = {}
                for choice in choices:
                    panel = QWidget()
                    panel_form = QFormLayout(panel)
                    panel_form.setContentsMargins(0, 0, 0, 0)
                    form.addRow(panel)
                    panels[choice] = panel
                    panel.setVisible(options[choice].isChecked())
                    options[choice].toggled.connect(panel.setVisible)
                self.model_parameter_panels[name] = panels
                if title == "Spot detection":
                    self.bwthresh_box = QLineEdit()
                    self.bwthresh_box.setValidator(QDoubleValidator(0.0, 1e12, 9, self.bwthresh_box))
                    panels["tophat"].layout().addRow("bw_threshold_tolerance:", self.bwthresh_box)
                    form = panels["log"].layout()
                else:
                    self.maxdisp_box = QLineEdit()
                    self.maxdisp_box.setValidator(QIntValidator(0, 2_000_000_000, self.maxdisp_box))
                    panels["legacy"].layout().addRow("max_displacement (pixels):", self.maxdisp_box)
                    form = panels["gapped"].layout()
            for name, label, default, integer, positive in fields:
                box = QLineEdit()
                if integer:
                    box.setValidator(QIntValidator(0, 2_000_000_000, box))
                elif not integer:
                    box.setValidator(QDoubleValidator(0.0, 1e12, 9, box))
                box.setToolTip(name)
                self.new_tracking_boxes[name] = box
                form.addRow(label + ":", box)
            if title == "Spot detection":
                box = QLineEdit()
                box.setToolTip("detect_scales — comma-separated positive values")
                self.new_tracking_boxes["detect_scales"] = box
                form.addRow("Detection scales:", box)
            layout.addWidget(group)
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

        scroll = QScrollArea()
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        tab_layout = QVBoxLayout(self.tab_alex)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.addWidget(scroll)

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
        self.astigmatism_false.setChecked(True)
        for _, fields in NEW_TRACKING_GROUPS:
            for name, _, default, _, _ in fields:
                self.new_tracking_boxes[name].setText(str(default))
        self.new_tracking_boxes["detect_scales"].setText("0.75, 1.0, 1.3")
        self.new_tracking_models["spot_detection_model"]["log"].setChecked(True)
        self.new_tracking_models["linking_model"]["gapped"].setChecked(True)

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
        params.astigmatism = self.astigmatism_true.isChecked()
        for _, fields in NEW_TRACKING_GROUPS:
            for name, _, _, integer, positive in fields:
                setattr(params, name, _safe_number(
                    self.new_tracking_boxes[name].text().strip(), name,
                    integer=integer, minimum=0, strict=positive,
                ))
        scale_text = self.new_tracking_boxes["detect_scales"].text().strip()
        if ((scale_text.startswith("(") and scale_text.endswith(")")) or
                (scale_text.startswith("[") and scale_text.endswith("]"))):
            scale_text = scale_text[1:-1]
        params.detect_scales = tuple(
            _safe_number(part.strip(), "detect_scales", minimum=0, strict=True)
            for part in scale_text.split(",")
        )
        for name, options in self.new_tracking_models.items():
            selected = next((value for value, button in options.items()
                             if button.isChecked()), None)
            if selected is None:
                raise ValueError(f"Select a {name}.")
            setattr(params, name, selected)
        if params.Iratio_01_min > params.Iratio_01_max:
            raise ValueError("Minimum intensity ratio must not exceed maximum intensity ratio.")
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

    def _begin_worker(self, sources, action, output_directory=None):
        if self._worker is not None or not sources:
            return
        try:
            params_snapshot = self.fetch_params_from_ui()
            config = self._get_config()
            if self.raw_stack is None and output_directory is None:
                raise ValueError("Load a TIFF first.")
            reference_hw = None
            if self.raw_stack is not None:
                _prepare_stack(self.raw_stack, config)
                reference_hw = self.raw_stack.shape[1:]
        except Exception as exc:
            show_popup(self, "Invalid analysis settings", str(exc), critical=True)
            return
        self._worker = TrackingWorker(
            sources, config, params_snapshot, reference_hw,
            self.registration_matrix, action, self,
            output_directory=output_directory,
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
            folders = list(dict.fromkeys(folder for _, folder in successful))
            report.extend(["", "Output folders:"] + folders[:4])
            if len(folders) > 4:
                report.append(f"...and {len(folders) - 4} more.")
        if failed:
            report.extend(["", "Failures:"] + [
                f"{Path(path).name}: {details.strip().splitlines()[-1]}"
                for path, details in failed[:8]
            ])
            if len(failed) > 8:
                report.append(f"...and {len(failed) - 8} additional failures.")
            # Persist full tracebacks without interfering with any source TIFF.
            try:
                log_folder = _new_results_folder(
                    failed[0][0], stats.get("output_directory")
                )
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
