#!/usr/bin/env python3
"""
Automated PySTACHIO Processing Pipeline with Subfolder & Multi-Channel Config Support

Features:
  - Supports channel-specific configuration files (--config-left and --config-right) for
    independent tracking/postprocessing parameters per channel.
  - Dynamically creates stage-isolated config files (.temp_{config_stem}_{task}_config.txt) 
    to prevent PySTACHIO from running tracking on raw video stacks during preprocessing.
  - Passes local relative video filenames inside cwd=work_dir for proper PySTACHIO .mat generation.
  - Absolute path resolution for config overrides (roi_file, bead_path, model, tmats_path).
  - Unbuffered logging output for clean, chronological pipeline log tracking.
  - Flexible subset range input via '-r 8,12', '-r 8-12', '-r "(8,12)"', or '-r 30,max'.
  - Command-line task overrides via '--task' / '--tasks' (e.g. '--task track,postprocess').
  - Custom prefix matching via '-p' / '--video-prefix' and '-b' / '--bf-prefix'.
  - Clear progress indicators (e.g., Preprocessing Pair 1 of 21 [FL 2 / BF 1 | Target Range: 2-42]).
  - Centralized master bead outputs in config_dir (transformation_matrix.npy, summed_beads*, 
    Registered_overlay_*, ncc_scores.*, registration.csv).
  - Muted TensorFlow C++ noise and Python warnings.
"""

import os
import sys
import shutil
import argparse
import subprocess
import re
from pathlib import Path

# ==============================================================================
# SCRIPT PATH CONFIGURATION
# ==============================================================================
PYSTACHIO_SCRIPT = r"C:\Users\lf1017\OneDrive - University of York\Documents\GitHub\pystachio-smt\pystachio-smt\pystachio_smt\pystachio_smt.py"
FOV_SCRIPT_PATH = r"C:\Users\lf1017\OneDrive - University of York\Documents\GitHub\pystachio-smt\pystachio-smt\pystachio_smt\fields_of_view.py"


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def parse_range_arg(val: str) -> tuple[int, int | str]:
    """
    Parses range argument inputs like '8,12', '8-12', '(8,12)', '30,max', or '30-max'.
    Returns a tuple of (start_index, end_index_or_string).
    """
    cleaned = val.strip("()'\"[]{} ").lower()
    delimiter = "," if "," in cleaned else ("-" if "-" in cleaned else None)

    if delimiter:
        parts = [p.strip() for p in cleaned.split(delimiter)]
        start = int(parts[0])
        end_val = "max" if parts[1] in ["max", "none", ""] else int(parts[1])
        return start, end_val
    elif cleaned.isdigit():
        return int(cleaned), "max"
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid range format: '{val}'. Expected format '8,12', '8-12', or '30,max'."
        )


def normalize_path(path: Path | str) -> Path:
    """
    Converts a path to an absolute Path object while preserving drive letters (C:\, D:\, Y:\, Z:\).
    Avoids Path.resolve() which expands mapped drives into UNC network paths that break TensorFlow/h5py.
    """
    p = Path(path)
    return p.absolute() if not p.is_absolute() else p


def parse_config(config_path: Path) -> dict:
    """Parses key=value parameters and task strings from a PySTACHIO config file."""
    params = {}
    if not config_path or not config_path.exists():
        if config_path:
            print(f"Warning: Config file {config_path} does not exist.", flush=True)
        return params

    with open(config_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                clean_val = val.split("#")[0].strip()
                params[key.strip().lower()] = clean_val
            elif "," in line or line.lower() in ["preprocess", "track", "postprocess", "fovs"]:
                clean_val = line.split("#")[0].strip()
                params["task"] = clean_val

    return params


def prepare_step_config(config_path: Path, target_task: str) -> Path:
    """
    Creates a temporary stage config where task directives (e.g. 'preprocess,track,postprocess')
    are replaced strictly with target_task.
    Incorporates original config file stem to avoid overwrite conflicts between dual channel configs.
    """
    config_dir = normalize_path(config_path.parent)
    safe_task_name = target_task.replace(",", "_")
    temp_config_path = config_dir / f".temp_{config_path.stem}_{safe_task_name}_config.txt"

    with open(config_path, "r", encoding="utf-8") as f_in, open(temp_config_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" not in stripped:
                f_out.write(f"{target_task}\n")
            else:
                f_out.write(line)

    return temp_config_path


def purge_all_stale_matrices(config_dir: Path, root_dir: Path):
    """
    If no master transformation_matrix.npy exists in config_dir, remove all lingering
    transformation_matrix.npy files across all subdirectories to force clean bead registration.
    """
    master_tmats = config_dir / "transformation_matrix.npy"
    if master_tmats.exists():
        return

    print("\n[Pipeline Setup] Master transformation_matrix.npy not found in config directory.", flush=True)
    print("[Pipeline Setup] Purging all stray transformation_matrix.npy files across subfolders...", flush=True)

    search_paths = [config_dir, root_dir]
    if config_dir.parent.exists():
        search_paths.append(config_dir.parent)

    purged_count = 0
    for search_base in search_paths:
        for stale in search_base.rglob("transformation_matrix.npy"):
            try:
                stale.unlink()
                print(f"  - Deleted stale matrix: {stale}", flush=True)
                purged_count += 1
            except OSError as e:
                print(f"  - Warning: Could not delete {stale}: {e}", flush=True)

    if purged_count == 0:
        print("  - No stray matrices found. Ready for fresh bead registration.\n", flush=True)
    else:
        print(f"  - Successfully purged {purged_count} stray matrix file(s).\n", flush=True)


def move_bead_outputs_to_config_dir(work_dir: Path, config_dir: Path, config_params: dict = None):
    """
    Copies all bead registration artifacts (transformation_matrix.npy, Registered_overlay_*,
    summed_beads*, ncc_scores.*, registration.csv) from work_dir, results_dir,
    config_dir.parent, and bead folders into config_dir.
    """
    master_tmats = config_dir / "transformation_matrix.npy"
    results_dir = work_dir / "results"

    search_dirs = [work_dir, results_dir, config_dir.parent]

    if config_params:
        bead_path_str = config_params.get("bead_path", "").strip()
        if bead_path_str:
            bead_path = Path(bead_path_str)
            if not bead_path.is_absolute():
                bead_path = normalize_path(config_dir / bead_path)
            bead_parent = bead_path.parent
            if bead_parent.exists() and bead_parent not in search_dirs:
                search_dirs.append(bead_parent)
            for bead_folder in bead_parent.glob(f"{bead_path.name}*"):
                if bead_folder.is_dir() and bead_folder not in search_dirs:
                    search_dirs.append(bead_folder)

    bead_patterns = [
        "Registered_overlay_beads_*.png",
        "Registered_overlay_*.png",
        "registered_*.png",
        "*summed*.tif",
        "*summed*.png",
        "*bead*.png",
        "*bead*.tif",
        "transformation_matrix.npy",
        "ncc_scores.png",
        "ncc_scores.csv",
        "registration.csv"
    ]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        for pattern in bead_patterns:
            for file_path in search_dir.glob(pattern):
                if not file_path.is_file():
                    continue
                # Do not touch raw image stack files
                if "MMStack" in file_path.name or file_path.name.endswith(".ome.tif"):
                    continue

                dest_path = config_dir / file_path.name

                if file_path.name == "transformation_matrix.npy":
                    if not master_tmats.exists():
                        if file_path != master_tmats:
                            shutil.copy(str(file_path), str(master_tmats))
                            print(f"[Pipeline Sync] Saved master transformation_matrix.npy -> {master_tmats}", flush=True)
                    else:
                        if file_path != master_tmats:
                            try:
                                file_path.unlink()
                                print(f"[Pipeline Cleanup] Removed temporary local matrix: {file_path}", flush=True)
                            except OSError:
                                pass
                else:
                    if file_path != dest_path:
                        try:
                            shutil.copy(str(file_path), str(dest_path))
                            print(f"[Pipeline Sync] Copied bead artifact {file_path.name} -> {dest_path}", flush=True)
                        except OSError as e:
                            print(f"Warning: Could not copy {file_path.name}: {e}", flush=True)


def find_pair_files(root_dir: Path, i: int, video_prefix: str, bf_prefix: str | None = None) -> tuple[Path | None, Path | None]:
    """Locates FL (index i) and BF (index i-1) files across flat and nested directories using prefixes."""
    bf_num = i - 1
    effective_bf_prefix = bf_prefix if (bf_prefix and bf_prefix.strip()) else video_prefix

    # Direct folder lookup
    fl_dir = root_dir / f"{video_prefix}{i}"
    bf_dir = root_dir / f"{effective_bf_prefix}{bf_num}"

    fl_file = None
    bf_file = None

    if fl_dir.exists():
        fl_tifs = sorted([p for p in fl_dir.glob("*.tif") if "results" not in p.parts])
        if fl_tifs:
            fl_file = fl_tifs[0]

    if bf_dir.exists():
        bf_tifs = sorted([p for p in bf_dir.glob("*.tif") if "results" not in p.parts])
        if bf_tifs:
            bf_file = bf_tifs[0]

    # Recursive subfolder search fallback
    if not fl_file:
        fl_matches = [
            p for p in root_dir.rglob("*.tif") 
            if "results" not in p.parts and (f"{video_prefix}{i}" in p.name or f"{video_prefix}{i}" in str(p.parent))
        ]
        fl_matches.sort()
        if fl_matches:
            fl_file = fl_matches[0]

    if not bf_file:
        bf_matches = [
            p for p in root_dir.rglob("*.tif") 
            if "results" not in p.parts and (f"{effective_bf_prefix}{bf_num}" in p.name or f"{effective_bf_prefix}{bf_num}" in str(p.parent))
        ]
        bf_matches.sort()
        if bf_matches:
            bf_file = bf_matches[0]

    return fl_file, bf_file


def detect_max_pair(root_dir: Path, video_prefix: str) -> int:
    """Scans current and nested subdirectories for the highest pair index using video_prefix."""
    max_idx = 0
    if not video_prefix:
        return max_idx

    for folder in root_dir.rglob("*"):
        if folder.is_dir() and video_prefix in folder.name:
            suffix = folder.name.split(video_prefix)[-1]
            m = re.search(r'^\d+', suffix) or re.search(r'\d+', suffix)
            if m:
                max_idx = max(max_idx, int(m.group()))

    for file in root_dir.rglob("*.tif"):
        if "results" in file.parts:
            continue
        name = file.stem
        if video_prefix in name:
            suffix = name.split(video_prefix)[-1]
            m = re.search(r'^\d+', suffix) or re.search(r'\d+', suffix)
            if m:
                max_idx = max(max_idx, int(m.group()))

    return max_idx


# ==============================================================================
# PIPELINE STAGES
# ==============================================================================
def step_1_preprocessing(config_path: Path, video_prefix: str, bf_prefix: str | None, start: int, end: int, step: int):
    """Step 1: Batch Preprocessing"""
    print("\n==================================================", flush=True)
    print("STEP 1: PREPROCESSING", flush=True)
    print("==================================================", flush=True)

    root_dir = Path(".")
    config_params = parse_config(config_path)
    config_dir = normalize_path(config_path.parent)

    step_config = prepare_step_config(config_path, "preprocess")

    master_tmats = config_dir / "transformation_matrix.npy"

    # Clean stray matrices across subfolders if master matrix is missing
    purge_all_stale_matrices(config_dir, root_dir)

    # Initial sweep for any bead artifacts sitting in parent/bead folders
    move_bead_outputs_to_config_dir(root_dir, config_dir, config_params)

    path_keys = ["roi_file", "bead_path", "model", "model_params"]
    extra_overrides = [f"tmats_path={master_tmats}"]

    for key in path_keys:
        val = config_params.get(key, "").strip()
        if val:
            val_path = Path(val)
            if not val_path.is_absolute():
                abs_path = normalize_path(config_dir / val_path)
            else:
                abs_path = normalize_path(val_path)
            extra_overrides.append(f"{key}={abs_path}")

    pair_indices = list(range(start, end + 1, step))
    total_pairs = len(pair_indices)

    for pair_idx, i in enumerate(pair_indices, 1):
        bf_num = i - 1
        fl_file, bf_file = find_pair_files(root_dir, i, video_prefix, bf_prefix)

        if not fl_file or not bf_file:
            print(f"Skipping Pair {pair_idx}/{total_pairs} (Index {i}): No matching FL/BF pair found.", flush=True)
            continue

        work_dir = normalize_path(fl_file.parent)
        results_dir = work_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        local_tmats = work_dir / "transformation_matrix.npy"

        # Copy master transformation matrix into current field folder if available
        if master_tmats.exists() and not local_tmats.exists():
            shutil.copy(master_tmats, local_tmats)

        fl_rel = fl_file.name
        bf_rel = os.path.relpath(normalize_path(bf_file), work_dir)

        print("--------------------------------------------------", flush=True)
        print(f"Preprocessing Pair {pair_idx} of {total_pairs} [FL {i} / BF {bf_num} | Target Range: {start}-{end}]", flush=True)
        print(f"  Working Dir: {work_dir}", flush=True)
        print(f"  FL Video:    {fl_rel}", flush=True)
        print(f"  BF Video:    {bf_rel}", flush=True)
        print("--------------------------------------------------", flush=True)

        cmd = [
            sys.executable, PYSTACHIO_SCRIPT,
            "-r", str(step_config),
            "preprocess",
            f"video_path={fl_rel}",
            f"bf_path={bf_rel}",
            "save_dir=./results",
            *extra_overrides
        ]

        subprocess.run(cmd, cwd=work_dir, check=True)

        # Sweep all bead artifacts out of work_dir, results_dir, parent dir, and bead dirs into config_dir
        move_bead_outputs_to_config_dir(work_dir, config_dir, config_params)


def step_2_tracking(
    tracking_config_path: Path,
    video_prefix: str,
    start: int,
    end: int,
    step: int,
    tracking_tasks: list,
    left_config_path: Path | None = None,
    right_config_path: Path | None = None
):
    """Step 2: Tracking & Postprocessing across Subdirectories with optional Channel-Specific Configs"""
    print("\n==================================================", flush=True)
    print("STEP 2: TRACKING & POSTPROCESSING", flush=True)
    print("==================================================", flush=True)

    l_config = left_config_path if left_config_path else tracking_config_path
    r_config = right_config_path if right_config_path else tracking_config_path

    base_params = parse_config(tracking_config_path)
    l_params = parse_config(l_config) if l_config != tracking_config_path else base_params
    r_params = parse_config(r_config) if r_config != tracking_config_path else base_params

    # Determine num_channels from master or channel configs
    num_channels_base = int(base_params.get("num_channels", 1))
    num_channels_l = int(l_params.get("num_channels", 1))
    num_channels_r = int(r_params.get("num_channels", 1))

    if left_config_path and right_config_path:
        num_channels = 2
    else:
        num_channels = max(num_channels_base, num_channels_l, num_channels_r)

    channels_to_process = ["L_channel", "R_channel"] if num_channels == 2 else ["L_channel"]

    print("Tracking Configuration Overview:", flush=True)
    print(f"  * Master/Default Tracking Config: {tracking_config_path.name}", flush=True)
    if left_config_path:
        print(f"  * Left Channel Config:  {left_config_path.name}", flush=True)
    if right_config_path:
        print(f"  * Right Channel Config: {right_config_path.name}", flush=True)
    print(f"  * Total Channels to Process: {num_channels}\n", flush=True)

    all_results_dirs = sorted([normalize_path(p) for p in Path(".").rglob("results") if p.is_dir()])

    if not all_results_dirs:
        print("Warning: No 'results' directories found. Run preprocessing first.", flush=True)
        return

    target_indices = set(range(start, end + 1, step))
    results_dirs = []

    # Filter results directories to match active pair range
    for rdir in all_results_dirs:
        parent_name = rdir.parent.name
        for idx in target_indices:
            if f"{video_prefix}{idx}" in parent_name or f"ffield_{idx}" in parent_name:
                results_dirs.append(rdir)
                break

    if not results_dirs:
        print(f"Warning: No 'results' directories found matching range ({start}-{end}).", flush=True)
        return

    total_dirs = len(results_dirs)
    task_str = ",".join(tracking_tasks)

    for dir_idx, results_dir in enumerate(results_dirs, 1):
        print("--------------------------------------------------", flush=True)
        print(f"Tracking Results Directory {dir_idx} of {total_dirs}: {results_dir}", flush=True)
        print("--------------------------------------------------", flush=True)

        for channel in channels_to_process:
            channel_tif = results_dir / f"{channel}.tif"

            if not channel_tif.exists():
                print(f"Warning: {channel_tif.name} not found in {results_dir}. Skipping channel.", flush=True)
                continue

            if channel == "L_channel":
                chan_config = l_config
                chan_params = l_params
                chan_label = "Left Channel (L)"
            else:
                chan_config = r_config
                chan_params = r_params
                chan_label = "Right Channel (R)"

            step_config = prepare_step_config(chan_config, task_str)

            alex_enabled = chan_params.get("alex", "false").lower() in ["true", "1", "yes"]
            base_frame_time = float(chan_params.get("frame_time", 0.0035))

            if num_channels == 2 or alex_enabled:
                tracking_frame_time = base_frame_time * 2.0
            else:
                tracking_frame_time = base_frame_time

            print(f"--> Processing {channel}.tif ({tracking_tasks}) [{chan_label}]", flush=True)
            print(f"    Config File: {chan_config.name}", flush=True)
            print(f"    Base frame_time: {base_frame_time}s | Effective tracking frame_time: {tracking_frame_time}s | ALEX: {alex_enabled}", flush=True)

            cmd = [
                sys.executable, PYSTACHIO_SCRIPT,
                "-r", str(step_config),
                *tracking_tasks,
                f"video_path={channel}.tif",
                f"name={channel}",
                "save_dir=.",
                "ALEX=False",
                f"frame_time={tracking_frame_time}"
            ]

            subprocess.run(cmd, cwd=results_dir, check=True)


def step_3_fields_of_view(config_path: Path, tmats_path: str):
    """Step 3: Fields of View Summary Generation"""
    print("\n==================================================", flush=True)
    print("STEP 3: FIELDS OF VIEW SUMMARY", flush=True)
    print("==================================================", flush=True)

    fov_script = normalize_path(FOV_SCRIPT_PATH)
    if not fov_script.exists():
        print(f"Error: FOV script not found at {FOV_SCRIPT_PATH}", flush=True)
        return

    config_params = parse_config(config_path)
    num_channels = int(config_params.get("num_channels", 1))
    channel = config_params.get("channel", "L")

    tmats_file = normalize_path(config_path.parent / "transformation_matrix.npy")

    cmd = [
        sys.executable,
        str(fov_script),
        f"--tmats_path={tmats_file}",
        "--save_dir=.",
        f"--num_channels={num_channels}",
        f"--channel={channel}"
    ]

    print(f"Running command: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


# ==============================================================================
# MAIN CLI ENTRYPOINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Automated PySTACHIO Analysis Pipeline")

    parser.add_argument("-c", "--config", type=str, default="config_all.txt",
                        help="Path to master config file (default: config_all.txt)")
    parser.add_argument("-t", "--tracking-config", type=str, default=None,
                        help="Path to default tracking config file (defaults to master config if omitted)")
    parser.add_argument("-p", "--video-prefix", type=str, default=None,
                        help="Custom video/folder prefix (defaults to video_path parameter in config)")
    parser.add_argument("-b", "--bf-prefix", type=str, default=None,
                        help="Custom brightfield path/folder prefix (defaults to video-prefix with index i-1)")

    parser.add_argument("--config-left", "--left-config", "--tracking-config-left", type=str, default=None,
                        help="Path to left channel config file (overrides tracking config for L_channel)")
    parser.add_argument("--config-right", "--right-config", "--tracking-config-right", type=str, default=None,
                        help="Path to right channel config file (overrides tracking config for R_channel)")

    parser.add_argument("--task", "--tasks", type=str, default=None,
                        help="Override task list directive from config (e.g., 'track' or 'track,postprocess')")

    parser.add_argument("-r", "--range", type=parse_range_arg, default=None,
                        help="Field pair range subset in format '8,12', '8-12', '(8,12)', or '30,max'")

    parser.add_argument("--start", type=int, default=2, help="Starting field pair index (default: 2)")
    parser.add_argument("--end", type=int, default=None, help="Ending field pair index (default: auto-detected)")
    parser.add_argument("--step", type=int, default=2, help="Step index increment (default: 2)")

    parser.add_argument("--show-plots", action="store_true", default=None,
                        help="Display interactive plot windows (sets MPLBACKEND=TkAgg if True, Agg if False)")

    args = parser.parse_args()

    # Disable all TensorFlow C++ and Python logging/warnings across subprocesses
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["PYTHONWARNINGS"] = "ignore"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["PYTHONUNBUFFERED"] = "1"

    config_path = normalize_path(args.config)
    left_config_path = normalize_path(args.config_left) if args.config_left else None
    right_config_path = normalize_path(args.config_right) if args.config_right else None

    # Fallback to left config if default master config doesn't exist but left config was specified
    if not config_path.exists() and left_config_path and left_config_path.exists():
        config_path = left_config_path

    if not config_path.exists():
        sys.exit(f"ERROR: Configuration file '{config_path}' not found.")

    tracking_config_path = normalize_path(args.tracking_config) if args.tracking_config else config_path

    config_params = parse_config(config_path)

    raw_task_str = args.task if args.task else config_params.get("task", "preprocess,track,postprocess")
    task_list = [t.strip().lower() for t in raw_task_str.split(",")]

    run_preprocess = "preprocess" in task_list
    tracking_tasks = [t for t in task_list if t in ["track", "postprocess"]]
    run_fovs = run_preprocess or "fovs" in task_list

    config_show_plots = config_params.get("show_plots", "false").lower() in ["true", "1", "yes"]
    show_plots = args.show_plots if args.show_plots is not None else config_show_plots

    if show_plots:
        os.environ["MPLBACKEND"] = "TkAgg"
        print("Plotting Mode: Displaying figures interactively (MPLBACKEND=TkAgg)", flush=True)
    else:
        os.environ["MPLBACKEND"] = "Agg"
        print("Plotting Mode: Headless mode (MPLBACKEND=Agg)", flush=True)

    video_prefix = args.video_prefix.strip() if args.video_prefix else config_params.get("video_path", "").strip()
    if not video_prefix:
        sys.exit("ERROR: 'video_prefix' / 'video_path' is missing or empty in config and command-line options.")

    bf_prefix = args.bf_prefix.strip() if args.bf_prefix else config_params.get("bf_prefix", "").strip()

    root_dir = Path(".")

    # Auto-detect maximum pair index across dataset
    detected_max = detect_max_pair(root_dir, video_prefix)
    if detected_max == 0:
        detected_max = 42

    # Determine start and end values based on -r/--range or defaults
    if args.range:
        start_val, end_val_raw = args.range
        end_val = detected_max if end_val_raw == "max" else int(end_val_raw)
    else:
        start_val = args.start
        end_val = detected_max if args.end is None else args.end

    tmats_path = config_params.get("tmats_path", "./transformation_matrix.npy")

    print(f"Loaded Master Config: {config_path.name}", flush=True)
    if tracking_config_path != config_path:
        print(f"Loaded Default Tracking Config: {tracking_config_path.name}", flush=True)
    if left_config_path:
        print(f"Loaded Left Channel Config: {left_config_path.name}", flush=True)
    if right_config_path:
        print(f"Loaded Right Channel Config: {right_config_path.name}", flush=True)

    print(f"Using Video/Folder Prefix: '{video_prefix}'", flush=True)
    print(f"Using Brightfield Prefix:  '{bf_prefix if bf_prefix else (video_prefix + ' [i-1]')}'", flush=True)
    print(f"Target Pair Range: {start_val} to {end_val} (step={args.step}) [Max Detected: {detected_max}]", flush=True)
    print(f"Tasks Parsed: {task_list}", flush=True)
    print(f"  * Run Preprocessing: {run_preprocess}", flush=True)
    print(f"  * Run Tracking Tasks: {tracking_tasks}", flush=True)
    print(f"  * Run FOV Summary:    {run_fovs}\n", flush=True)

    try:
        if run_preprocess:
            step_1_preprocessing(config_path, video_prefix, bf_prefix, start_val, end_val, args.step)

        if tracking_tasks:
            step_2_tracking(
                tracking_config_path,
                video_prefix,
                start_val,
                end_val,
                args.step,
                tracking_tasks=tracking_tasks,
                left_config_path=left_config_path,
                right_config_path=right_config_path
            )

        if run_fovs:
            step_3_fields_of_view(config_path, tmats_path)

        print("\nPipeline execution completed successfully!", flush=True)
    finally:
        # Clean up temporary stage config files across all config directories
        config_dirs = {
            normalize_path(cfg.parent)
            for cfg in [config_path, tracking_config_path, left_config_path, right_config_path]
            if cfg is not None and cfg.exists()
        }
        for c_dir in config_dirs:
            for temp_file in c_dir.glob(".temp_*_config.txt"):
                try:
                    temp_file.unlink()
                except OSError:
                    pass


if __name__ == "__main__":
    main()