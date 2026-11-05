#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Edward Higgins <ed.higgins@york.ac.uk>
#
# Distributed under terms of the MIT license.

""" TRACKING - Spot tracking module

Description:
    tracking.py contains the code for the tracking task, which identifies spots
    within a set of frames and builds spot trajectories across those frames.

Contains:
    function track

Author:
    Edward Higgins
    Dom Alderson
    Jack Shepherd

Version: 1.1.0
"""

import sys

import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import spots
import trajectories
import images
from scipy.ndimage import gaussian_laplace, maximum_filter


def track(params):
    # Read in the image data
    image_data = images.ImageData()
    image_data.read(params.name + ".tif", params)

    if params.num_frames != 0:
        image_data.num_frames = params.num_frames

    if params.ALEX==True:
        imageL=np.zeros((image_data.num_frames//2,image_data.frame_size[1],image_data.frame_size[0]//2))
        imageR=np.zeros((image_data.num_frames//2,image_data.frame_size[1],image_data.frame_size[0]//2))
        if params.start_channel=='L':
            for i in range(0,image_data.num_frames-1,2):
                imageL[i//2,:,:] = image_data.pixel_data[i,:,:image_data.frame_size[0]//2]
                imageR[i//2,:,:] = image_data.pixel_data[i+1,:,image_data.frame_size[0]//2:]
        else:
            for i in range(0,image_data.num_frames-1,2):
                imageR[i//2,:,:] = image_data.pixel_data[i,:,:image_data.frame_size[0]//2]
                imageL[i//2,:,:] = image_data.pixel_data[i+1,:,image_data.frame_size[0]//2:]
        image_data.num_frames = image_data.num_frames//2

        #LHS
        image_data.pixel_data = imageL
        image_data.frame_size = [image_data.frame_size[0]//2,image_data.frame_size[1]]
        all_spots = []
        for frame in range(image_data.num_frames):
            all_spots.append(track_frame(image_data[frame], frame, params))
        trajs = trajectories.build_trajectories(all_spots, params)
        trajectories.write_trajectories(trajs, params.name +  "_Lchannel_trajectories.tsv")

        #RHS
        image_data.pixel_data = imageR
        all_spots = []
        for frame in range(image_data.num_frames):
            all_spots.append(track_frame(image_data[frame], frame, params))
        trajs = trajectories.build_trajectories(all_spots, params)
        trajectories.write_trajectories(trajs, params.name +  "_Rchannel_trajectories.tsv")

    # For each frame, detect spots
    else:
        all_spots = []
        if params.num_procs == 0:
            for frame in range(image_data.num_frames):
                all_spots.append(track_frame(image_data[frame], frame, params))

        else:
            res = [None] * image_data.num_frames
            with mp.Pool(params.num_procs) as pool:
                for frame in range(image_data.num_frames):
                    res[frame] = pool.apply_async(track_frame, (image_data[frame], frame, params))
                for frame in range(image_data.num_frames):
                    all_spots.append(res[frame].get())

        # Link the spot trajectories across the frames
        trajs = trajectories.build_trajectories(all_spots, params)
        trajectories.write_trajectories(trajs, params.name + "_trajectories.tsv")

def original_track_frame(frame_data, frame, params):
        # Find the spots in this frame
        frame_spots = spots.Spots(frame=frame)
        frame_spots.find_in_frame(frame_data.as_image()[:, :], params)
        found_spots = frame_spots.num_spots
        frame_spots.merge_coincident_candidates()

        merged_spots = frame_spots.num_spots
        # Iteratively refine the spot centres
        frame_spots.refine_centres(frame_data, params)

        frame_spots.filter_candidates(frame_data, params)

        frame_spots.get_spot_intensities(frame_data.as_image()[:,:], params)
        frame_spots.get_spot_widths(frame_data.as_image()[:,:], params)
        frame_spots.get_precision(frame_data.as_image()[:,:], params) # Lewis edit
        if params.verbose:
            print(
                f"Frame {frame:4d}: found {frame_spots.num_spots:3d} spots "
                f"({found_spots:3d} identified, "
                f"{found_spots-merged_spots:3d} merged, "
                f"{merged_spots-frame_spots.num_spots:3d} filtered)"
            )
        return frame_spots

def track_frame(frame_data, frame_idx, params):
    """Detect and localise one frame. Returns a list of spot dicts."""
    frame_spots = spots.Spots(frame=frame_idx)
    frame_spots.detect(frame_data, frame_idx, params)
    found_spots = frame_spots.num_spots
    frame_spots.merge_coincident_candidates()
    merged_spots = frame_spots.num_spots
    # Iteratively refine the spot centres
    frame_spots.refine_centres(frame_data, params)
    frame_spots.filter_candidates(frame_data, params)
    frame_spots.get_spot_intensities(frame_data.as_image()[:,:], params)
    frame_spots.get_spot_widths(frame_data.as_image()[:,:], params)
    frame_spots.get_precision(frame_data.as_image()[:,:], params) # Lewis edit
    if params.verbose:
        print(
            f"Frame {frame_idx:4d}: found {frame_spots.num_spots:3d} spots "
            f"({found_spots:3d} identified, "
            f"{found_spots-merged_spots:3d} merged, "
            f"{merged_spots-frame_spots.num_spots:3d} filtered)"
        )
    return frame_spots
