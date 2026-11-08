#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Edward Higgins <ed.higgins@york.ac.uk>
#
# Distributed under terms of the MIT license.

""" TRAJECTORIES - Trajectory construction and manipulation module

Description:
    trajectories.py contains the Trajectory class containing information about
    spot trajectories across multiple frames, along with the routines involved
    in locating and extending the trajectories.

Contains:
    class Trajectory
    function build_trajectories
    function read_trajectories
    function write_trajectories

Author:
    Edward Higgins

Version: 0.2.0
"""

import csv
import os

import copy
import pandas as pd
import numpy as np

if __package__:
    from .spots import Spots
    from .linking import link_frames, close_gaps
else:
    from spots import Spots
    from linking import link_frames, close_gaps


class Trajectory:
    """Observed detections with explicit frame numbers; length counts observations."""
    _fields = {
        "path": "positions", "intensity": "spot_intensity",
        "bg_intensity": "bg_intensity", "snr": "snr",
        "converged": "converged", "width": "width",
        "noise": "noise", "precision": "precision",
    }

    def __init__(self, id, spots, spot_id):
        self.id = id
        self.frames = []
        self.length = 0
        self.start_frame = self.end_frame = None
        self.stoichiometry = 0
        self.linked_traj = None
        for name in self._fields:
            setattr(self, name, [])
        self.extend(spots, spot_id)

    def extend(self, spots, spot_id):
        frame = spots.frame
        if not np.isfinite(frame) or frame < 0 or int(frame) != frame:
            raise ValueError("Frame must be a nonnegative integer")
        frame = int(frame)
        if self.frames and frame <= self.end_frame:
            raise ValueError("Trajectory frames must be strictly increasing")
        values = {name: copy.deepcopy(getattr(spots, source)[spot_id])
                  for name, source in self._fields.items()}
        for name, value in values.items():
            getattr(self, name).append(value)
        self.frames.append(frame)
        self.start_frame = self.frames[0]
        self.end_frame = frame
        self.length = len(self.frames)

    def index_at_frame(self, frame):
        """Observation index, or None when the frame has no detection."""
        i = int(np.searchsorted(self.frames, frame))
        return i if i < self.length and self.frames[i] == frame else None


def build_trajectories(all_spots, params):
    """Dispatch to the original legacy builder or the assignment/gap linker."""
    if params.linking_model == "legacy":
        # The original builder assumes at least one frame.
        if len(all_spots) == 0:
            return []
        return build_trajectories_legacy(all_spots, params)
    if params.linking_model == "gapped":
        return build_trajectories_gapped(all_spots, params)
    raise ValueError(
        f"Unknown linking_model {params.linking_model!r}; expected 'legacy' or 'gapped'"
    )


def build_trajectories_legacy(all_spots, params):
    trajectories = []
    traj_num = 0

    # Create a trajectory for all spots in the first frame
    for i in range(all_spots[0].num_spots):
        trajectories.append(Trajectory(traj_num, all_spots[0], i))
        traj_num += 1

    # Construct trajectories for the rest of the frames
    for frame in range(1, len(all_spots)):
        assigned_spots = []
        for spot in range(all_spots[frame].num_spots):
            close_candidates = []
            for candidate in trajectories:
                if candidate.end_frame != frame - 1:
                    continue
                candidate_dist = np.linalg.norm(
                    all_spots[frame].positions[spot, :] - candidate.path[-1]
                )
                if candidate_dist < params.max_displacement:
                    close_candidates.append(candidate)

            if len(close_candidates) == 0:
                trajectories.append(Trajectory(traj_num, all_spots[frame], spot))
                traj_num += 1

            elif len(close_candidates) == 1:

                close_candidates[0].extend(all_spots[frame], spot)

            else:
                trajectories.append(Trajectory(traj_num, all_spots[frame], spot))
                traj_num += 1

    filtered_trajectories = list(filter(lambda x: x.length >= params.min_traj_len, trajectories))

    actual_traj_num = 0
    for traj in filtered_trajectories:
        traj.id = actual_traj_num
        actual_traj_num += 1

    return filtered_trajectories


def build_trajectories_gapped(all_spots, params):
    """Link, close gaps, filter by observed length, and build native trajectories.

    Accepts the native PySTACHIO Parameters object. max_gap is a frame-number difference,
    preserving the supplied linker's convention (2 permits one missing frame).
    """
    if not np.isfinite(params.min_traj_len) or int(params.min_traj_len) != params.min_traj_len or params.min_traj_len < 1:
        raise ValueError("min_traj_len must be a positive integer")
    all_spots = list(all_spots)
    rows = []
    seen_frames = set()
    for source_index, spots in enumerate(all_spots):
        frame = spots.frame
        if not np.isfinite(frame) or frame < 0 or int(frame) != frame:
            raise ValueError("Frame must be a nonnegative integer")
        if frame in seen_frames:
            raise ValueError("Expected one Spots object per frame")
        seen_frames.add(frame)
        for source in Trajectory._fields.values():
            if spots.num_spots and len(getattr(spots, source)) != spots.num_spots:
                raise ValueError(f"Frame {frame}: {source} is not aligned with detections")
        for i in range(spots.num_spots):
            rows.append((int(frame), *spots.positions[i], spots.spot_intensity[i], source_index, i))
    table = pd.DataFrame(rows, columns=["frame", "x", "y", "intensity", "source_index", "spot_index"])
    table = close_gaps(link_frames(table, params), params)
    trajectories = []
    for _, group in table.groupby("traj_id", sort=True):
        if len(group) < params.min_traj_len:
            continue
        trajectory = None
        for row in group.sort_values("frame").itertuples(index=False):
            source = all_spots[int(row.source_index)]
            if trajectory is None:
                trajectory = Trajectory(len(trajectories), source, int(row.spot_index))
            else:
                trajectory.extend(source, int(row.spot_index))
        trajectories.append(trajectory)
    return trajectories


def write_trajectories(trajectories, filename):
    """Write observed frames only, preserving the existing 13-column TSV schema."""
    with open(filename, "w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["trajectory", "frame", "x", "y", "spot_intensity",
                         "bg_intensity", "SNR", "converged", "widthx", "widthy",
                         "noise", "precision_x", "precision_y"])
        for traj in trajectories:
            if any(len(getattr(traj, name)) != len(traj.frames) for name in Trajectory._fields):
                raise ValueError(f"Trajectory {traj.id} has misaligned measurements")
            for i, frame in enumerate(traj.frames):
                writer.writerow([traj.id, frame, *traj.path[i], traj.intensity[i],
                                 traj.bg_intensity[i], traj.snr[i], traj.converged[i],
                                 *traj.width[i], traj.noise[i], *traj.precision[i]])


def to_spots(trajs):
    if not trajs:
        return []
    all_spots = []
    frame = 0
    done_all_frames = False
    while not done_all_frames:
        done_all_frames = True

        # Get the spots from trajs for this frame
        positions = []
        bg_intensity = []
        spot_intensity = []
        snr = []
        converged = []
        width = []
        noise = []
        precision = []
        for traj in trajs:

            if traj.end_frame > frame:
                done_all_frames = False

            if traj.start_frame > frame or traj.end_frame < frame:
                continue

            i = traj.index_at_frame(frame)
            if i is None:
                continue
            positions.append(traj.path[i][:])
            spot_intensity.append(traj.intensity[i])
            bg_intensity.append(traj.bg_intensity[i])
            snr.append(traj.snr[i])
            converged.append(traj.converged[i])
            width.append(traj.width[i]) # Lewis changed to width[i] from width
            # Lewis
            noise.append(traj.noise[i])
            precision.append(traj.precision[i])


        spots = Spots(len(positions), frame)
        spots.set_positions(np.array(positions))
        spots.spot_intensity = np.array(spot_intensity)
        spots.bg_intensity = np.array(bg_intensity)
        spots.snr = np.array(snr)
        spots.converged = np.array(converged, dtype=np.int8)
        spots.width=np.array(width).reshape(-1, 2)
        #Lewis
        spots.noise = np.array(noise)
        spots.precision = np.array(precision).reshape(-1, 2)
        
        all_spots.append(spots)

        frame += 1

    return all_spots

def read_trajectories(filename):
    if not os.path.isfile(filename):
        print(f"WARNING: No such file {filename}")
        return None
    groups = {}
    with open(filename, newline="") as stream:
        for line in csv.reader(stream, delimiter="\t"):
            if not line or line[0] == "trajectory":
                continue
            if len(line) != 13:
                raise ValueError("Expected 13 columns in trajectory TSV")
            groups.setdefault(int(line[0]), []).append(line)
    result = []
    for tid, lines in groups.items():
        trajectory = None
        for line in sorted(lines, key=lambda row: int(row[1])):
            spot = Spots(num_spots=1, frame=int(line[1]))
            spot.positions[0] = [float(line[2]), float(line[3])]
            spot.spot_intensity[0] = float(line[4])
            spot.bg_intensity[0] = float(line[5])
            spot.snr[0] = float(line[6])
            spot.converged[0] = int(line[7])
            spot.width[0] = [float(line[8]), float(line[9])]
            spot.noise[0] = float(line[10])
            spot.precision[0] = [float(line[11]), float(line[12])]
            if trajectory is None:
                trajectory = Trajectory(tid, spot, 0)
            else:
                trajectory.extend(spot, 0)
        result.append(trajectory)
    return result


def compare_trajectories(params):
    all_target_spots = []
    all_spots = []

    trajs = read_trajectories(params.name + "_trajectories.tsv")
    target_trajs = read_trajectories(params.name + "_simulated.tsv")
    if trajs is None or target_trajs is None:
        raise FileNotFoundError("Both trajectory files are required for comparison")

    frame = 0
    done_all_frames = False
    while not done_all_frames:
        done_all_frames = True

        # Get the spots from target_trajs for this frame
        frame_target_spots = []
        for traj in target_trajs:

            if traj.end_frame > frame:
                done_all_frames = False

            if traj.start_frame > frame or traj.end_frame < frame:
                continue
            i = traj.index_at_frame(frame)
            if i is not None:
                frame_target_spots.append(traj.path[i][:])

        target_spots = Spots(len(frame_target_spots), frame)
        target_spots.set_positions(np.array(frame_target_spots))
        all_target_spots.append(target_spots)

        # Get the spots from trajs for this frame
        frame_spots = []
        for traj in trajs:

            if traj.end_frame > frame:
                done_all_frames = False

            if traj.start_frame > frame or traj.end_frame < frame:
                continue

            i = traj.index_at_frame(frame)
            if i is not None:
                frame_spots.append(traj.path[i][:])

        spots = Spots(len(frame_spots), frame)
        spots.set_positions(np.array(frame_spots))
        all_spots.append(spots)

        frame += 1

    num_frames = frame

    for frame in range(num_frames):
        false_positives = 0
        false_negatives = 0
        multiples = 0
        matches = 0
        errors = []
        outside_frame = 0

        assigned_spots = []
        for spot in range(all_target_spots[frame].num_spots):

            if all_target_spots[frame].positions[spot,0] < 0 \
                      or all_target_spots[frame].positions[spot,1] < 0 \
                      or all_target_spots[frame].positions[spot,0] >= params.frame_size[0] \
                      or all_target_spots[frame].positions[spot,1] >= params.frame_size[1]:
              outside_frame  += 1
              continue

            close_candidates = []
            for candidate in range(all_spots[frame].num_spots):
                candidate_dist = np.linalg.norm(
                    all_target_spots[frame].positions[spot,:] - all_spots[frame].positions[candidate,:]
                )
                if candidate_dist < 1:
                    close_candidates.append(candidate_dist)

            if len(close_candidates) == 0:
                false_negatives += 1

            elif len(close_candidates) == 1:
                matches += 1
                errors.append(close_candidates[0])

            else:
                false_negatives += 1
                multiples += 1

        false_positives = all_spots[frame].num_spots - matches
        print(f"Frame {frame:4d}:",
              f"Error = {np.mean(errors) if errors else float('nan'):8f} pixels, ",
              f"Found = {all_spots[frame].num_spots:3d} ",
              f"Matches = {matches:3d} ",
              f"False negatives = {false_negatives:3d} ",
              f"False positives = {false_positives:3d} ",
              f"Outside = {outside_frame:3d} ")
