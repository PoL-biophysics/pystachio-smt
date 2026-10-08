#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.

""" PARAMETERS - Program parameters module

Description:
    parameters.py contains the Parameters class that holds all the program
    parameters, along with the default value for each parameter and routines
    for setting those parameters, and the save_parameters helper function to 
    write out params used

Contains:
    class Parameters
    function save_parameters

Author:
    Edward Higgins
    Jack Shepherd
    Lewis Frame

Version: 0.9.0
"""

import sys
import os
import datetime
from difflib import SequenceMatcher

default_parameters = {
    # Runtime parameters
    'display_figures':
        { 'description': 'Figures (Y/n)',
          'level': 'basic',
          'class': 'general',
          'default': True },    
    'num_procs':
        { 'description': 'The number of CPU processes to run with (0 is serial)',
          'level': 'basic',
          'class': 'general',
          'default': 0 },
    'tasks':
        { 'description': 'Which task(s) to perform in the run',
          'level': 'basic',
          'class': 'general',
          'default': [],
          'options': ['preprocess', 'simulate', 'track', 'postprocess', 'view', 'app'] },
    'name':
        { 'description': 'Name prefixing all files associated with this run',
          'level': 'basic',
          'class': 'general',
          'default': '' },
    'mask_file':
        { 'description': 'Filename of an image mask for filtering spots',
          'level': 'basic',
          'class': 'general',
          'default':  ''},
    'verbose':
        { 'description': 'Whether or not to display progress messages',
          'level': 'basic',
          'class': 'general',
          'default':  True},

    # Image parameters
    'frame_time':
        { 'description': 'Time per frame in seconds',
          'level': 'advanced',
          'class': 'image',
          'default': 0.005 },
    'pixel_size':
        { 'description': 'Length of a single pixel in μm',
          'level': 'advanced',
          'class': 'image',
          'default': 0.120 },
    'psf_width':
        { 'description': '?',
          'level': 'advanced',
          'class': 'image',
          'default': 0.120 },
    'start_frame':
        { 'description': 'The first frame of the image stack to analyse',
          'level': 'basic',
          'class': 'image',
          'default':  0},
    'num_frames':
        { 'description': 'Number of frames to simulate/analyse',
          'level': 'basic',
          'class': 'image',
          'default': 0 },
    'use_channel':
        { 'description': 'Flag to use only one channel of a non-ALEX acquisition',
         'level': 'basic',
         'class': 'image',
         'options': ['None', 'Left', 'Right', 'Both'],
         'default': 'None' },
    'channel_split':
        { 'description': 'If/how the frames are split spatially',
          'level': 'basic',
          'class': 'image',
          'default': 'None',
          'options': ['None', 'Vertical', 'Horizontal'] },
    'cell_mask':
        { 'description': 'Name of a black/white TIF file containing a cell mask',
          'level': 'advanced',
          'class': 'image',
          'default': ''},
    'ALEX':
        { 'description': 'Perform Alternating-Laser experiment analysis',
          'level': 'basic',
          'class': 'image',
          'default': False},
    'start_channel':
        { 'description': '?',
          'level': 'basic',
          'class': 'image',
          'default': 'L'},

    # Simulation parameters
    'num_spots':
        { 'description': 'Number of spots to simulate',
          'level': 'basic',
          'class': 'simulation',
          'default': 10 },
    'frame_size':
        { 'description': 'Size of frame to simulate ([x,y])',
          'level': 'basic',
          'class': 'simulation',
          'default': [100,100] },
    'I_single':
        { 'description': 'I_single value for simulated spots',
          'level': 'basic',
          'class': 'simulation',
          'default': 10000.0 },
    'bg_mean':
        { 'description': 'Mean of the background pixlel intensity',
          'level': 'advanced',
          'class': 'simulation',
          'default': 500.0 },
    'bg_std':
        { 'description': 'Standard deviation of the background pixel intensity',
          'level': 'advanced',
          'class': 'simulation',
          'default': 120.0 },
    'diffusion_coeff':
        { 'description': 'Diffusion coefficient of the diffusing spots',
          'level': 'basic',
          'class': 'simulation',
          'default': 1.0 },
    'spot_width':
        { 'description': 'Width of the simulated Gaussian spot',
          'level': 'advanced',
          'class': 'simulation',
          'default':  1.33},
    'max_spot_molecules':
        { 'description': 'Maximum number of dye molecules per spot',
          'level': 'advanced',
          'class': 'simulation',
          'default': 1 },
    'p_bleach_per_frame':
        { 'description': 'Probability of a spot bleaching in a given frame',
          'level': 'advanced',
          'class': 'simulation',
          'default': 0.0 },
    'photobleach':
        { 'description': 'Perform photobleaching (alias for max_spot_molecules=10, p_bleach_per_frame=0.05)',
          'level': 'basic',
          'class': 'simulation',
          'default': False },

    'photoblink':
        { 'description': 'Allow fully bleached spots to photoblink',
          'level': 'basic',
          'class': 'simulation',
          'default': True },

    'p_photoblink':
        { 'description': 'Probability that a bleached fluorophore blinks',
          'level': 'basic',
          'class': 'simulation',
          'default': 0.001 },    

    'psf_name':
    { 'description': 'Name of the PSF .npy file to use for a 3D simulation',
      'level': 'basic',
      'class': 'simulation',
      'default': None
        },

    'spherical_volume_radius':
    { 'description': 'Radius in microns of the spherical volume in which to simulate 3D diffusion',
      'level': 'basic',
      'class': 'simulation',
      'default': 1.0,
        },

    
    # Tracking parameters
    'bw_threshold_tolerance':
        { 'description': 'Threshold for generating the b/w image relative to the peak intensity',
          'level': 'advanced',
          'class': 'tracking',
          'default': 1.0 },
    'snr_filter_cutoff':
        { 'description': 'Cutoff value when filtering spots by signal/noise ratio',
          'level': 'basic',
          'class': 'tracking',
          'default': 0.4 },
    'filter_image':
        { 'description': 'Method for filtering the input image pre-analysis',
          'level': 'advanced',
          'class': 'tracking',
          'default': 'Gaussian',
          'options': ['Gaussian', 'None']},
    'max_displacement':
        { 'description': 'Maximum displacement allowed for spots between frames',
          'level': 'advanced',
          'class': 'tracking',
          'default': 5.0 },
    'struct_disk_radius':
        { 'description': 'Radius of the Disk structural element',
          'level': 'advanced',
          'class': 'tracking',
          'default': 5 },
    'min_traj_len':
        { 'description': 'Minimum number of frames needed to define a trajectory',
          'level': 'advanced',
          'class': 'tracking',
          'default': 3 },
    'subarray_halfwidth':
        { 'description': 'Halfwidth of the sub-image for analysing individual spots',
          'level': 'advanced',
          'class': 'tracking',
          'default': 8 },
    'gauss_mask_sigma':
        { 'description': 'Width of the Gaussian used for the iterative centre refinement',
          'level': 'advanced',
          'class': 'tracking',
          'default': 2.0 },
    'gauss_mask_max_iter':
        { 'description': 'Max number of iterations for the iterative centre refinement',
          'level': 'advanced',
          'class': 'tracking',
          'default': 1000 },
    'inner_mask_radius':
        { 'description': 'Radius of the mask used for calculating spot intensities',
          'level': 'advanced',
          'class': 'tracking',
          'default': 5 },
    'astigmatism':
        { 'description': 'Whether to do astigmatic imaging or rejecting fitted PSFs that have a ratio of x:y widths >=2',
          'level': 'basic',
          'class': 'tracking',
          'default': False },

    # Postprocessing parameters
    'display_figures':
        { 'description': 'Whether or not to display the figures live in MatPlotLib',
          'level': 'basic',
          'class': 'postprocessing',
          'default': True},
    'chung_kennedy_window':
        { 'description': 'Window width for Chung-Kennedy filtering',
          'level': 'basic',
          'class': 'postprocessing',
          'default': 3},
    'chung_kennedy':
        { 'description': 'Flag to specify whether or not to Chung-Kennedy filter intensity tracks',
          'level': 'basic',
          'class': 'postprocessing',
          'default': False},
    'msd_num_points':
        { 'description': 'Number of points used to calculate the mean-squared displacement',
          'level': 'basic',
          'class': 'postprocessing',
          'default': 4 },
    'stoic_method':
        { 'description': 'Method used for determining the stoichiometry of each trajectory',
          'level': 'advanced',
          'class': 'postprocessing',
          'default': 'Linear',
          'options': ['Linear', 'Mean', 'Initial', 'Max'] },
    'num_stoic_frames': {
          'level': 'advanced',
          'class': 'postprocessing',
          'description': 'Number of frames used to determine the stoichiometry',
          'default': 3 },
    'stoic_trajectory_start_within_n_frames': {
          'level': 'advanced',
          'class': 'postprocessing',
          'description': 'A trajectory must start within n frames of the start of the image acquisition to be used',
          'default': 3 },
    'calculate_isingle': {
          'level': 'advanced',
          'class': 'postprocessing',
          'description': 'Whether or not to calculate the ISingle',
          'default': True },
    'colocalize': {
          'level': 'advanced',
          'class': 'postprocessing',
          'description': '?',
          'default': False },
    'colocalize_distance': {
          'level': 'advanced',
          'class': 'postprocessing',
          'description': '?',
          'default': 5 },
    'colocalize_n_frames': {
          'level': 'advanced',
          'class': 'postprocessing',
          'description': '?',
          'default': 5 },
    'copy_number': {
          'level': 'advanced',
          'class': 'postprocessing',
          'description': '?',
          'default': False },
    'L_isingle':{
        'level' : 'intermediate',
        'class' : 'postprocessing',
        'default' : '10000',
        },
    'R_isingle':{
        'level' : 'intermediate',
        'class' : 'postprocessing',
        'default' : '10000',
        },

    # Preprocessing parameters
    'video_path':
        { 'description': 'Path to the raw video TIFF',
          'level': 'basic',
          'class': 'preprocessing',
          'default': '' },
    'channel':
        { 'description': 'Default channel to use for alignment registration (e.g., L or R)',
          'level': 'basic',
          'class': 'preprocessing',
          'default': 'L' },
    'num_channels':
        { 'description': 'Number of channels in the raw video',
          'level': 'basic',
          'class': 'preprocessing',
          'default': 1 },
    'bf_path':
        { 'description': 'Path to the brightfield image',
          'level': 'basic',
          'class': 'preprocessing',
          'default': '' },
    'bead_path':
        { 'description': 'Path to the calibration bead TIFF',
          'level': 'basic',
          'class': 'preprocessing',
          'default': '' },
    'manual_registration':
        { 'description': 'Set True to manually click beads for alignment',
          'level': 'basic',
          'class': 'preprocessing',
          'default': False },
    'manual_pairs':
        { 'description': 'Number of bead pairs to click if manual_registration is True',
          'level': 'advanced',
          'class': 'preprocessing',
          'default': 10 },
    'tmats_path':
        { 'description': 'Path to save/load transformation matrix',
          'level': 'advanced',
          'class': 'preprocessing',
          'default': './transformation_matrix.npy' },
    'model':
        { 'description': 'Path to the segmentation model file (.h5)',
          'level': 'basic',
          'class': 'preprocessing',
          'default': 'model.h5' },
    'model_type':
        { 'description': 'Type of model (e.g., unet)',
          'level': 'advanced',
          'class': 'preprocessing',
          'default': 'unet' },
    'model_params':
        { 'description': 'Parameter or threshold value for segmentation model',
          'level': 'advanced',
          'class': 'preprocessing',
          'default': '' },
    'mask_type':
        { 'description': 'Method for mask generation (AI, BF, FL_AI, THRESHOLD, WHOLE, MANUAL, or file)',
          'level': 'basic',
          'class': 'preprocessing',
          'default': 'WHOLE',
          'options': ['AI', 'BF', 'FL_AI', 'THRESHOLD', 'WHOLE', 'MANUAL', 'file'] },
    'frame_avg':
        { 'description': 'Number of frames to average for ROI/Mask generation',
          'level': 'basic',
          'class': 'preprocessing',
          'default': 5 },
    'roi_channel':
        { 'description': 'Channel to use for ROI generation (L or R)',
          'level': 'basic',
          'class': 'preprocessing',
          'default': 'L' },
    'roi_file':
        { 'description': 'Path to the ImageJ ROI zip file',
          'level': 'advanced',
          'class': 'preprocessing',
          'default': 'roi.zip' },
    'area_filter':
        { 'description': 'Minimum area size for cell filtering',
          'level': 'advanced',
          'class': 'preprocessing',
          'default': 0 },
    'inv_bf':
        { 'description': 'Invert brightfield image intensity',
          'level': 'advanced',
          'class': 'preprocessing',
          'default': False },
    'cell_fitting':
        { 'description': 'Perform precise cell boundary fitting',
          'level': 'advanced',
          'class': 'preprocessing',
          'default': False },
    'mask_prefix':
        { 'description': 'Prefix for saved mask files',
          'level': 'advanced',
          'class': 'preprocessing',
          'default': '' },
    'use_otsu':
        { 'description': 'Use Otsu thresholding instead of model segmentation (True, False, or multi)',
          'level': 'advanced',
          'class': 'preprocessing',
          'default': 'False', 
          'options': ['False', 'True', 'multi'] },
    'overwrite':
        { 'description': 'Overwrite existing output files',
          'level': 'advanced',
          'class': 'preprocessing',
          'default': True },
    'save_dir':
        { 'description': 'Directory to save output results',
          'level': 'basic',
          'class': 'preprocessing',
          'default': '' }
}


class Parameters:
    def __init__(self, initial=default_parameters):
        self._params = initial

        for param in self._params.keys():
            # Set all the values to be the default values
            self._params[param]['value'] = self._params[param]['default']

    def __getattr__(self, name):
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        else:
            try:
                return object.__getattribute__(self, "_params")[name]['value']
            except KeyError as exc:
                max_param = ''
                max_val  = 0
                for key in self._params:
                    if SequenceMatcher(None, name, key).ratio() > max_val:
                        max_param = key
                        max_val = SequenceMatcher(None, name, key).ratio()
                print(f"\nNo such key {name}. Did you mean {max_param}?\n")
                raise  exc

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            if name in self._params:
                self._params[name]['value'] = value
            else:
                object.__setattr__(self, name, value)

    def help(self, name=None, param_class=None, level='basic'):
        names = []
        if name:
            names.append(name)
        elif param_class:
            for key in self._params:
                if level != 'advanced':
                    if (self._params[key]['class'] == param_class
                      and self._params[key]['level'] == 'basic'):
                        names.append(key)
                else:
                    names.append(key)
        elif level != 'basic':
            for key in self._params:
                names.append(key)
        else:
            for key in self._params:
                if self._params[key]['level'] == 'basic':
                    names.append(key)


        for name in names:
            print()
            print(f"{name.upper()}")
            print(f"  Description: {self._params[name]['description']}")
            print(f"      Default: {self._params[name]['default']}")
            print(f"        Class: {self._params[name]['class']}")
            print(f"        Level: {self._params[name]['level']}")



    def read(self, args):
        if not args:
            return

        self.task = args.pop(0)
        self.task = self.task.split(",")
        if self.task == ['help']:
            return
        elif self.task != ['app'] and args:
            self.name = args.pop(0)

        for arg in args:
            if "=" not in arg:
                continue
            key, value = arg.split("=", 1)
            try:
                current_val = getattr(self, key)
                # Check for bool first since bool is an instance/subclass of int in Python
                if isinstance(current_val, bool):
                    setattr(self, key, value == "True")
                elif isinstance(current_val, int):
                    setattr(self, key, int(value))
                elif isinstance(current_val, float):
                    setattr(self, key, float(value))
                elif isinstance(current_val, list):
                    setattr(self, key, list(map(lambda x: int(x), value.split(","))))
                else:
                    setattr(self, key, value)
            except (AttributeError, ValueError, KeyError):
                print(f"Warning: Unknown or invalid parameter '{key}'")

            if key == "pixel_size":
                self.psf_width = 0.160 / self.pixel_size

    def param_dict(self, param_class=''):
        param_dict = {}

        if param_class:
            for k,v in self._params.items():
                if v["class"] == param_class:
                    param_dict[k] = v
        else:
            param_dict = self._params

        return param_dict


def save_parameters(params, save_dir="."):    
    # 1. Define parameter categories
    PARAM_GROUPS = {
        "General": [
            "task", "name", "overwrite", "num_frames", "pixel_size", 
            "use_channel", "num_channels", "ALEX"
        ],
        "Preprocessing": [
            "video_path", "bead_path", "bf_path", "tmats_path", "roi_file", 
            "roi_channel", "mask_type", "mask_prefix", "model", "model_type", 
            "area_filter", "cell_fitting", "inv_bf", "use_otsu", "frame_avg",
            "manual_registration", "manual_pairs"
        ],
        "Tracking": [
            "search_radius", "min_trajectory_len", "max_linking_distance", 
            "tracking_method", "detection_threshold"
        ],
        "Simulation / Postprocessing": [
            "sim_type", "postprocess_mode", "output_format"
        ]
    }

    # 2. Extract key-value dictionary safely
    if hasattr(params, '__dict__'):
        param_dict = vars(params)
    else:
        param_dict = {
            attr: getattr(params, attr) 
            for attr in dir(params) 
            if not attr.startswith('_') and not callable(getattr(params, attr))
        }

    # 3. Clean string values
    cleaned_params = {}
    for k, v in param_dict.items():
        if callable(v):
            continue
        if isinstance(v, (list, tuple, set)):
            cleaned_params[k] = ", ".join(map(str, v)) if v else "None"
        elif v is None or str(v).strip() == "":
            cleaned_params[k] = "None"
        else:
            cleaned_params[k] = str(v)

    # 4. Calculate longest key length across all keys for uniform alignment
    max_key_len = max((len(k) for k in cleaned_params.keys()), default=20)

    # 5. Track which keys have been categorized
    written_keys = set()
    filepath = os.path.join(save_dir, "parameters.txt")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("==================================================\n")
        f.write(" PySTACHIO Parameters\n")
        f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("==================================================\n")

        # Write categorized sections
        for group_name, keys in PARAM_GROUPS.items():
            # Check if any key from this group exists in the parameters
            present_keys = [k for k in keys if k in cleaned_params]
            if not present_keys:
                continue

            f.write(f"\n--- [{group_name}] ---\n")
            for k in sorted(present_keys):
                f.write(f"{k:<{max_key_len}} : {cleaned_params[k]}\n")
                written_keys.add(k)

        # Catch-all section for any parameters not explicitly in PARAM_GROUPS
        remaining_keys = set(cleaned_params.keys()) - written_keys
        if remaining_keys:
            f.write("\n--- [Other / Custom] ---\n")
            for k in sorted(remaining_keys):
                f.write(f"{k:<{max_key_len}} : {cleaned_params[k]}\n")

        f.write("\n==================================================\n")

    print(f"Parameters saved to: {filepath}", flush=True)
