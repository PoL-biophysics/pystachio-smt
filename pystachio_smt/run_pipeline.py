#!/usr/bin/env python3
"""
Automated PySTACHIO Processing Pipeline with Subfolder & Multi-Channel Config Support

Features:
  - Safe Windows path globbing supporting wildcards across subdirectories.
  - Explicit debug logging for file pattern matching and path lookups.
  - Supports channel-specific configuration files (--config-left and --config-right) for
    independent tracking/postprocessing parameters per channel.
  - Strips inline comments from stage-isolated config files (.temp_{config_stem}_{task}_config.txt)
    so PySTACHIO can cleanly parse floats/ints/bools without KeyError crashes.
  - Passes local relative video filenames inside cwd=work_dir for proper PySTACHIO .mat generation.
  - Absolute path resolution for config overrides (roi_file, bead_path, model, tmats_path).
  - Unbuffered logging output for clean, chronological pipeline log tracking.
  - Flexible subset range input via '-r 8,12', '-r 8-12', '-r "(8,12)"', or '-r 30,max'.
  - Command-line task overrides via '--task' / '--tasks' (e.g. '--task track,postprocess').
  - Custom prefix matching via '-p' / '--video-prefix' and '-b' / '--bf-prefix'.
  - Centralized master bead outputs in config_dir (transformation_matrix.npy, summed_beads*, 
    Registered_overlay_*, ncc_scores.*, registration.csv).
  - Muted TensorFlow C++ noise and Python warnings.
"""

import os
import sys
import shutil
import argparse
import subprocess
import glob
import re
from pathlib import Path

# ==============================================================================
# SCRIPT PATH CONFIGURATION
# ==============================================================================
PYSTACHIO_SCRIPT_DEFAULT = r"C:\Users\lf1017\OneDrive - University of York\Documents\GitHub\pystachio-smt\pystachio-smt\pystachio_smt\pystachio_smt.py"
FOV_SCRIPT_PATH_DEFAULT = r"C:\Users\lf1017\OneDrive - University of York\Documents\GitHub\pystachio-smt\pystachio-smt\pystachio_smt\fields_of_view.py"


def get_script_path(default_path_str: str, filename: str) -> Path:
    """Resolves script path using default location or falling back to local directory."""
    p = Path(default_path_str)
    if p.exists():
        return p
    local_p = Path(__file__).parent / filename
    if local_p.exists():
        return local_p
    return p


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def safe_glob(pattern: str) -> list[str]:
    """
    Ensures wildcards in directory paths work properly on Windows by 
    converting backslashes to forward slashes before calling glob.
    """
    clean_pattern = str(pattern).replace('\\', '/')
    matches = glob.glob(clean_pattern)
    return matches


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
    """Parses key=value parameters and task strings from a PySTACHIO config file, stripping UTF-8 BOM and comments."""
    params = {}
    if not config_path or not config_path.exists():
        if config_path:
            print(f"Warning: Config file {config_path} does not exist.", flush=True)
        return params

    # 'utf-8-sig' automatically strips any leading \ufeff Byte Order Mark (BOM)
    with open(config_path, "r", encoding="utf-8-sig", errors="ignore") as f:
        for line in f:
            raw_content = line.replace('\r', '').split("#")[0].strip()
            if not raw_content:
                continue
            if "=" in raw_content:
                key, val = raw_content.split("=", 1)
                clean_key = key.lstrip('\ufeff').strip().lower().replace(" ", "_")
                params[clean_key] = val.strip()
            elif "," in raw_content or raw_content.lower() in ["preprocess", "track", "postprocess", "fovs"]:
                params["task"] = raw_content.lower()

    return params


def prepare_step_config(config_path: Path, target_task: str, overrides: dict = None) -> Path:
    """
    Creates a clean temporary stage config file where:
    1. target_task is explicitly written on Line 1.
    2. UTF-8 BOM markers (\ufeff) and carriage returns (\r) are completely stripped.
    3. Inline comments (# ...) are stripped so PySTACHIO parses numbers/floats properly.
    4. Task-specific and field-specific parameter overrides take precedence.
    """
    config_dir = normalize_path(config_path.parent)
    safe_task_name = target_task.replace(",", "_")
    temp_config_path = config_dir / f".temp_{config_path.stem}_{safe_task_name}_config.txt"

    overrides = overrides or {}
    clean_overrides = {
        str(k).lstrip('\ufeff').strip().lower().replace(" ", "_"): str(v).strip() 
        for k, v in overrides.items() if v is not None
    }
    written_keys = set()

    lines_to_write = [f"{target_task}\n"]
    known_tasks = {"preprocess", "track", "postprocess", "fovs", "track,postprocess", "postprocess,track"}

    if config_path.exists():
        # Open with 'utf-8-sig' to automatically purge UTF-8 BOM markers
        with open(config_path, "r", encoding="utf-8-sig", errors="ignore") as f_in:
            for line in f_in:
                # Strip carriage returns, inline comments, and leading BOM
                raw_content = line.replace('\r', '').split("#")[0].lstrip('\ufeff').strip()
                if not raw_content:
                    continue

                # Omit any standalone task directive lines in body
                if raw_content.lower() in known_tasks:
                    continue

                if "=" in raw_content:
                    key_part, val_part = raw_content.split("=", 1)
                    key_part = key_part.strip().lower().replace(" ", "_")
                    val_part = val_part.strip()

                    if key_part in clean_overrides:
                        lines_to_write.append(f"{key_part}={clean_overrides[key_part]}\n")
                        written_keys.add(key_part)
                    else:
                        lines_to_write.append(f"{key_part}={val_part}\n")
                        written_keys.add(key_part)

    # Append any new overrides not present in base config
    for k, v in clean_overrides.items():
        if k not in written_keys:
            lines_to_write.append(f"{k}={v}\n")

    with open(temp_config_path, "w", encoding="utf-8") as f_out:
        f_out.writelines(lines_to_write)

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
    Copies bead registration artifacts into config_dir without ever deleting or
    overwriting an existing master transformation_matrix.npy.
    """
    config_dir_norm = normalize_path(config_dir)
    master_tmats = config_dir_norm / "transformation_matrix.npy"
    master_tmats_key = os.path.normcase(str(master_tmats))
    results_dir = normalize_path(work_dir) / "results"

    search_dirs = [normalize_path(work_dir), results_dir, config_dir_norm.parent]

    if config_params:
        bead_path_str = config_params.get("bead_path", "").strip()
        if bead_path_str:
            bead_path = Path(bead_path_str)
            if not bead_path.is_absolute():
                bead_path = normalize_path(config_dir_norm / bead_path)
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
                if "MMStack" in file_path.name or file_path.name.endswith(".ome.tif"):
                    continue

                src_abs = normalize_path(file_path)
                dest_abs = config_dir_norm / file_path.name

                # Skip if source and destination point to the exact same file
                if os.path.normcase(str(src_abs)) == os.path.normcase(str(dest_abs)):
                    continue

                if file_path.name == "transformation_matrix.npy":
                    # Only promote to master if no master matrix exists yet; never overwrite or unlink
                    if not master_tmats.exists() and os.path.normcase(str(src_abs)) != master_tmats_key:
                        shutil.copy2(str(src_abs), str(master_tmats))
                        print(f"[Pipeline Sync] Saved master transformation_matrix.npy -> {master_tmats}", flush=True)
                else:
                    try:
                        shutil.copy2(str(src_abs), str(dest_abs))
                        print(f"[Pipeline Sync] Copied bead artifact {file_path.name} -> {dest_abs}", flush=True)
                    except OSError as e:
                        print(f"Warning: Could not copy {file_path.name}: {e}", flush=True)


def find_pair_files(
    root_dir: Path,
    i: int,
    video_prefix: str,
    bf_prefix: str | None = None,
    static_bf_path: Path | None = None
) -> tuple[Path | None, Path | None]:
    """Locates FL (index i) and BF (index i-1 or static bf_path) files across flat and nested directories using prefixes and safe_glob."""
    bf_num = i - 1
    effective_bf_prefix = bf_prefix if (bf_prefix and bf_prefix.strip()) else video_prefix

    def execute_glob(pref: str, idx: int) -> Path | None:
        if not pref:
            return None
        pref_clean = pref.replace('\\', '/')
        patterns = [
            f"{pref_clean}{idx}.tif",
            f"{pref_clean}{idx}.tiff",
            f"{pref_clean}{idx}_*.tif",
            f"{pref_clean}{idx}_*.tiff",
            f"{pref_clean}{idx}*.tif",
            f"{pref_clean}{idx}*.tiff",
            f"{pref_clean}{idx}*"
        ]
        for pat in patterns:
            matches = safe_glob(pat)
            valid = [
                Path(m) for m in matches 
                if ("results" not in Path(m).parts) and Path(m).suffix.lower() in ['.tif', '.tiff']
            ]
            if valid:
                found = sorted(valid)[0]
                print(f"[DEBUG FILE MATCH] Pattern '{pat}' -> Found: {found}", flush=True)
                return found
        return None

    print(f"[DEBUG PAIR SEARCH] --- Index {i} ---", flush=True)
    print(f"[DEBUG PAIR SEARCH] FL Prefix: '{video_prefix}' (Target Index: {i})", flush=True)
    fl_file = execute_glob(video_prefix, i)

    print(f"[DEBUG PAIR SEARCH] BF Prefix: '{effective_bf_prefix}' (Target Index: {bf_num})", flush=True)
    bf_file = execute_glob(effective_bf_prefix, bf_num)

    # Fallback search if direct safe_glob failed
    if not fl_file or not bf_file:
        v_clean = video_prefix.replace('\\', '/')
        bf_clean = effective_bf_prefix.replace('\\', '/')

        for p in root_dir.rglob("*.tif"):
            if "results" in p.parts:
                continue
            p_str = str(p).replace('\\', '/')
            p_name = p.name

            if not fl_file:
                if (f"{v_clean}{i}" in p_str) or (f"{v_clean}{i}" in p_name) or (f"ffield_{i}" in p_name):
                    fl_file = p
                    print(f"[DEBUG FILE MATCH] RGLOB Fallback FL -> Found: {fl_file}", flush=True)
            if not bf_file:
                if (f"{bf_clean}{bf_num}" in p_str) or (f"{bf_clean}{bf_num}" in p_name) or (f"ffield_{bf_num}" in p_name):
                    bf_file = p
                    print(f"[DEBUG FILE MATCH] RGLOB Fallback BF -> Found: {bf_file}", flush=True)

    # Fallback to a single static brightfield file if specified in config (e.g. ./data/brightfield.tif)
    if not bf_file and static_bf_path and static_bf_path.is_file():
        bf_file = static_bf_path
        print(f"[DEBUG FILE MATCH] Using static BF file -> {bf_file}", flush=True)

    if not fl_file:
        print(f"[DEBUG PAIR SEARCH] WARNING: FL File missing for index {i}", flush=True)
    if not bf_file:
        print(f"[DEBUG PAIR SEARCH] WARNING: BF File missing for index {bf_num} (Pair Index {i})", flush=True)

    return fl_file, bf_file


def detect_max_pair(root_dir: Path, video_prefix: str) -> int:
    """Scans current and nested subdirectories for the highest pair index using video_prefix and safe_glob."""
    max_idx = 0
    if not video_prefix:
        return max_idx

    pref_clean = video_prefix.replace('\\', '/')
    pattern = f"{pref_clean}*"
    matches = safe_glob(pattern)

    if not matches and not any(c in pref_clean for c in ['*', '?']):
        matches = [str(p) for p in root_dir.rglob("*.tif") if "results" not in p.parts]

    for m in matches:
        p = Path(m)
        if "results" in p.parts:
            continue
        name = p.name
        if "ffield_" in name:
            match = re.search(r'ffield_(\d+)', name)
            if match:
                max_idx = max(max_idx, int(match.group(1)))
        else:
            match = re.search(r'(\d+)', name)
            if match:
                max_idx = max(max_idx, int(match.group(1)))

    if max_idx == 0:
        for folder in root_dir.rglob("*"):
            if folder.is_dir() and "field" in folder.name.lower():
                m = re.search(r'\d+', folder.name)
                if m:
                    max_idx = max(max_idx, int(m.group()))

    print(f"[DEBUG MAX INDEX] Auto-detected highest pair index: {max_idx}", flush=True)
    return max_idx


# ==============================================================================
# PIPELINE STAGES
# ==============================================================================
def step_1_preprocessing(config_path: Path, video_prefix: str, bf_prefix: str | None, start: int, end: int, step: int):
    """Step 1: Batch Preprocessing using preprocess.py (with ALEX deinterleaving)"""
    print("\n==================================================", flush=True)
    print("STEP 1: PREPROCESSING (preprocess.py)", flush=True)
    print("==================================================", flush=True)

    root_dir = Path(".")
    config_params = parse_config(config_path)
    config_dir = normalize_path(config_path.parent)

    preprocess_script = get_script_path(str(Path(__file__).parent / "preprocess.py"), "preprocess.py")
    if not preprocess_script.exists():
        sys.exit(f"ERROR: preprocess.py not found at '{preprocess_script}'")

    master_tmats = config_dir / "transformation_matrix.npy"

    # Clean stray matrices across subfolders if master matrix is missing
    purge_all_stale_matrices(config_dir, root_dir)

    # Initial sweep for any bead artifacts sitting in parent/bead folders
    move_bead_outputs_to_config_dir(root_dir, config_dir, config_params)

    # Resolve optional static BF file from config if bf_path points to an actual file
    static_bf_path = None
    raw_bf_cfg = config_params.get("bf_path", "").strip()
    if raw_bf_cfg:
        cand_bf = Path(raw_bf_cfg) if Path(raw_bf_cfg).is_absolute() else normalize_path(config_dir / raw_bf_cfg)
        if cand_bf.is_file():
            static_bf_path = cand_bf

    # Build resolved path overrides for path-like settings
    path_keys = ["roi_file", "bead_path", "model", "model_params"]
    base_overrides = {}

    base_overrides["tmats_path"] = str(master_tmats)

    for key in path_keys:
        val = config_params.get(key, "").strip()
        if val:
            is_path_like = ("/" in val or "\\" in val or ":" in val or val.endswith((".dat", ".roi", ".keras", ".h5", ".pth", ".pt", ".onnx")))
            if is_path_like:
                val_path = Path(val)
                if not val_path.is_absolute():
                    abs_path = normalize_path(config_dir / val_path)
                else:
                    abs_path = normalize_path(val_path)
                base_overrides[key] = str(abs_path)
            else:
                base_overrides[key] = val

    pair_indices = list(range(start, end + 1, step))
    total_pairs = len(pair_indices)

    for pair_idx, i in enumerate(pair_indices, 1):
        bf_num = i - 1
        fl_file, bf_file = find_pair_files(root_dir, i, video_prefix, bf_prefix, static_bf_path=static_bf_path)

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

        # Merge field-specific parameter overrides with base path settings
        overrides = dict(base_overrides)
        overrides["video_path"] = fl_rel
        overrides["bf_path"] = bf_rel
        overrides["save_dir"] = "./results"
        if master_tmats.exists():
            overrides["tmats_path"] = str(master_tmats)

        merged = {**config_params, **overrides}

        cmd = [
            sys.executable,
            str(preprocess_script),
            "--video_path", str(merged.get("video_path", fl_rel)),
            "--bf_path", str(merged.get("bf_path", bf_rel)),
            "--bead_path", str(merged.get("bead_path", "")),
            "--save_dir", str(merged.get("save_dir", "./results")),
            "--use_channel", str(merged.get("channel", merged.get("use_channel", "L"))),
            "--num_channels", str(merged.get("num_channels", "1")),
            "--start_channel", str(merged.get("start_channel", "L")),
            "--frame_avg", str(merged.get("frame_avg", "5")),
            "--num_frames", str(merged.get("num_frames", "0")),
            "--ALEX", str(merged.get("alex", "False")),
            "--pixel_size", str(merged.get("pxsize", merged.get("pixel_size", "0.120"))),
            "--mask_type", str(merged.get("mask_type", "AI")),
            "--model_type", str(merged.get("model_type", "unet")),
            "--model", str(merged.get("model", "")),
            "--threshold_param", str(merged.get("threshold_param", merged.get("model_params", "0.5"))),
            "--dilations", str(merged.get("dilations", "1")),
            "--area_filter", str(merged.get("area_filter", "100")),
            "--eccentricity_filter", str(merged.get("eccentricity_filter", "0.0")),
            "--cell_fit_err", str(merged.get("cell_fit_err", "0.0")),
            "--inv_bf", str(merged.get("inv_bf", "False")),
            "--mask_prefix", str(merged.get("mask_prefix", "mask")),
            "--tmats_path", str(merged.get("tmats_path", "transformation_matrix.npy")),
            "--roi_channel", str(merged.get("roi_channel", "L")),
            "--roi_file", str(merged.get("roi_file", "")),
            "--manual_registration", str(merged.get("manual_registration", "False")),
            "--manual_pairs", str(merged.get("manual_pairs", "10")),
            "--use_otsu", str(merged.get("use_otsu", "False")),
            "--cell_fitting", str(merged.get("cell_fitting", "False")),
            "--overwrite", str(merged.get("overwrite", "True")),
        ]

        print(f"[RUN PIPELINE] Executing: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, cwd=work_dir, check=True)

        # Sweep all bead artifacts into config_dir
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

    pystachio_script = get_script_path(PYSTACHIO_SCRIPT_DEFAULT, "pystachio_smt.py")

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
            if f"{video_prefix}{idx}" in parent_name or f"field{idx}" in parent_name.lower() or f"ffield_{idx}" in parent_name:
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

            alex_enabled = chan_params.get("alex", "false").lower() in ["true", "1", "yes"]
            base_frame_time = float(chan_params.get("frame_time", 0.0035))

            if alex_enabled:
                tracking_frame_time = base_frame_time * 2.0
            else:
                tracking_frame_time = base_frame_time

            print(f"--> Processing {channel}.tif ({tracking_tasks}) [{chan_label}]", flush=True)
            print(f"    Config File: {chan_config.name}", flush=True)
            print(f"    Base frame_time: {base_frame_time}s | Effective tracking frame_time: {tracking_frame_time}s | ALEX: {alex_enabled}", flush=True)

            overrides = {
                "video_path": f"{channel}.tif",
                "name": channel,
                "save_dir": ".",
                "alex": "False",
                "frame_time": str(tracking_frame_time)
            }

            step_config = prepare_step_config(chan_config, task_str, overrides=overrides)

            cmd = [
                sys.executable,
                str(pystachio_script),
                "-r",
                str(step_config)
            ]

            print(f"[RUN PIPELINE] Executing: {' '.join(cmd)}", flush=True)
            subprocess.run(cmd, cwd=results_dir, check=True)


def step_3_fields_of_view(config_path: Path, tmats_path: str):
    """Step 3: Fields of View Summary Generation"""
    print("\n==================================================", flush=True)
    print("STEP 3: FIELDS OF VIEW SUMMARY", flush=True)
    print("==================================================", flush=True)

    fov_script = get_script_path(FOV_SCRIPT_PATH_DEFAULT, "fields_of_view.py")
    if not fov_script.exists():
        print(f"Error: FOV script not found at {FOV_SCRIPT_PATH_DEFAULT}", flush=True)
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

    # Auto-detect maximum pair index across dataset using safe_glob
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