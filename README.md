# PySTACHIO SMT - 2026 Updated version 

> **WARNING**
> 
> This codebase is still under active development and, as such, is not stable.
> Expect some functionality not to be complete, and interfaces to be changed
> without warning.

## Introduction
PySTACHIO SMT began life as a Python port of Adam Wollman's Single
Molecule Tools[1]. At its heart, PySTACHIO provides a utility for tracking and analysing
localisation microscopy data, providing features for spot tracking and spot
intensity & trajectory analysis. This extended second version building off our first full release [2] provides several improvements and extensions over our previous editions:

+ Improved spot detection and track linking through improvements made in the Wollman lab and released as MaCADEMIA [3]
+ Neural network integration for cell segmentation, using OmniPose or your own trained U-NET
+ An extensible tab-based GUI With a host of features out of the box:
  + Segmentation with machine learning (and if necessary manual cleanup)
  + Cropping ROIs for analysis
  + Registration
  + smFRET
  + Astigmatism microscopy analyser
  + Results explorer
+ Significant performance improvements
+ Simulation improvements:
  + 3D with astigmatism
  + Confined diffusion in a sphere or on a spherical surface
  + Confined diffusion in a sphere with an excluded volume
  + Confined diffusion in a sphere with a central volume with different diffusion coefficient
+ Batch processing via the GUI - no more writing scripts


## CLI Operation Synopsis
```{bash}
$ pystachio <task> <seed name> [<parameter>=<value>...]
```

For example:
```{bash}
$ pystachio track data/exp_data
$ pystachio simulate,track data/sim_data num_spots=10
```
(Note: The old tasks "view" and "app" have been removed. These are superseded by new standalone utilities.)

## GUI Operation Synopsis
```{bash}
$ PySTACHIO_studio.py
```
You will load into the preprocessing screen. Use the tabs at the top to navigate according to your experiment/workflow.

## Overview
PySTACHIO SMT can be used in two ways: GUIs for interactive use for both tracking and visualising results, or a
command UI for terminal-based & scripted use. Both interfaces provide the
following functionality:

### Preprocessing
This allows input experiment image data to be split into two channels and cropped to a specific ImageJ region of interest. Also includes calculation of a transformation matrix for channel registration and defining masks either from brightfield or fluorescence micrographs. 

### 

### Spot tracking
This allows input image data (either from experiment or simulation) to be
imported into the program and spots tracked across multiple frames.

### Post-processing analysis
This allows spot physical parameters such as ISingle values and diffusions
coefficients to be extracted from tracked spots' trajectories & intensities.

### Simulation of experimental data
This allows pseudo-microscopy data to be simulated based on a number of
tunable parameters, such as the number of spots, number of frames, bleaching
time, Signal-Noise ratio etc...

## GUI-Only functionality
### smFRET analysis
This allows visualisation of single molecule FRET pairs, tracking intensity through time and estimating FRET efficiency as ... Only for immobile spots - for diffusing FRET sensors do a full two-colour analysis, colocalisation, and extract intensities from linked trajectories. Only works currently for ALEX data but this will be fixed eventually.

### Click mode
Allows the user to click an approximate spot position and see intensity through time, using Chung-Kennedy filtering or not. Only for immobile spots. Very useful for plotting stepwise photobleaching traces.

### Results Viewer 
This provides a convenient GUI to visualise the outputs from the PySTACHIO, including histograms of outputs for entire experiment folders, object by object analysis, overlays of tracks onto brightfield or fluorescence images and model fitting for diffusivity distributions. 

## AI/ML segmentation

## Installation
PySTACHIO SMT uses a few fairly common libraries, listed in `requirements.txt`
In short, these are:
- *Numpy* for array handling
- *OpenCV* for image processing
- *TiffFile* for handling opening and saving images
- *Matplotlib* for plotting and some simple GUI implementations
- *tensorflow* for machine learning models

Installation of these can be performed as such:
```{bash}
$ pip install -r requirements.txt
```

After that, just make sure `pystachio` is in your `$PATH` somewhere, or just refer
to it directly using `/path/to/pystachio ...`.

## Usage

For the terminal UI, the command can be run as follows:
```{bash}
$ pystachio <tasks> <seedname> [parameters]
```

Here, `<seedname>` is used for telling the program which file(s) to read from
and write to; and `<tasks>` is a comma-separated list of the tasks you want to
be performed.  For example, if you want to track spots in a dataset and then
view the result, this can be done with `pystachio track,view ...`. 

For a given `<seedname>`, the files currently recognised are:
- `seedname.tif`: An input dataset saved as a multi-frame TIFF,
- `seedname_trajectories.tsv`: A tab-separated value list containing predicted trajectories for `seedname.tif`
- `seedname_simulated_trajectories.tsv`: A tab-separated value list containing simulated trajectories for  `seedname.tif`.

The complete list of tasks currently implemented is:
- `simulate`: Create a simulated psuedo-dataset based upon a set of parameters
- `track`: Track the spots in a given dataset and produce predicted trajectories for that dataset.
- `postprocess`: Perform postprocessing on tracked trajectories


## Contributing

Information on how to contribute to the PySTACHIO SMT project is available in
[contributing.md](doc/contributing.md). In short, the process is:
- Clone the repository,
- Create a branch for your new changes (functionality, bug fixes, etc...),
- Develop your changes in your new branch, and
- Create a pull-request to merge the branch back into `master`.

## Authors

This code has been developed by Edward Higgins, Jack Shepherd, and Lewis Frame in
collaboration with the Physics of Life group at the University of York. For more
information, contact Jack at [jack.shepherd@york.ac.uk](jack.shepherd@york.ac.uk).

## License

Released under the MIT License.

## References
[1] [Single Molecule Tools](https://awollman.github.io/single-molecule-tools/)
[2] [PySTACHIO published paper](https://www.sciencedirect.com/science/article/pii/S2001037021002944)