# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 10:50:31 2025

@author: lf1017
"""

import numpy as np
import time, os, sys
import cv2

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from cellpose_omni import io, plot, models
import omnipose
import skimage.io
import matplotlib.pyplot as plt
import torch

def main(img_obj, modeldir,save_dir):
    import torch.nn as nn

    # define parameters
    mask_threshold = -1 
    verbose = 0 
    use_gpu = False 
    transparency = True 
    rescale = None 
    flow_threshold = 0 
    resample = True 
    cluster = True
    omni = True
    model_name = 'bact_fluor_omni'
    modeldir = modeldir
    
    curr_dir = os.getcwd()

    print("Initializing CellposeModel...")
    # 1. Init without a path so we control the build
    model = models.CellposeModel(
        pretrained_model=None, 
        net_avg=False, 
        diam_mean=0, 
        nclasses=4, 
        nchan=2,    
        dim=2
    )

    # 2. NETWORK SURGERY
    # Force the output layer to match your 4-class checkpoint
    current_out_channels = model.net.output[2].out_channels
    if current_out_channels != 4:
        print(f"Detected {current_out_channels} output classes. forcing to 4 to match checkpoint...")
        in_channels = model.net.output[2].in_channels
        model.net.output[2] = nn.Conv2d(in_channels, 4, kernel_size=1)
        model.nclasses = 4

    # 3. MANUAL WEIGHT LOADING
    print(f"Loading weights manually from {modeldir}...")
    state_dict = torch.load(modeldir, map_location="cpu")
    model.net.load_state_dict(state_dict, strict=True)
    print("Model loaded successfully!")
    
    # 4. THE MONKEY PATCH
    # Prevents the library from reloading the model and breaking our surgery
    def no_op_load_model(filename, cpu=None):
        print(" Library attempted to reload model. Blocked by monkey-patch to preserve 4-class architecture.")
        return True
    model.net.load_model = no_op_load_model
    model.pretrained_model = ["dummy_path"]
    
    # 5. PREPARE IMAGE
    img = img_obj.copy()
    #img = cv2.GaussianBlur(img, (3, 3), 1.0)
    
    # (H, W) -> (1, H, W, 1) -> (1, H, W, 2)
    patch = np.reshape(img, (1, img.shape[0], img.shape[1], 1))
    patch = np.tile(patch, (1, 1, 1, 2))
    
    # Transpose to NCHW: (1, 2, H, W)
    patch_nchw = patch.transpose(0, 3, 1, 2)
    input_image = patch_nchw[0] # Get (2, H, W)
    print("Shape",input_image.shape)

    chans = [0,0] 
    
    # 6. RUN EVAL
    masks, flows, styles = model.eval(
        [input_image], 
        channels=chans,
        rescale=rescale,
        mask_threshold=mask_threshold,
        transparency=transparency,
        flow_threshold=flow_threshold,
        omni=omni, 
        resample=resample,
        verbose=verbose, 
        cluster=cluster,
        interp=True
    )
    
    # 7. ROBUST UNWRAPPING (The Fix)
    # We use a loop to peel away any layers of lists until we hit the numpy array
    maski = masks[0]
    while isinstance(maski, list):
        print(f"Drilling down mask layer... type is {type(maski)}")
        maski = maski[0]
    
    # Do the same for flows. flows[0] is the result for image 0.
    # flows[0][0] is usually the RGB visualization.
    flowi = flows[0][0]
    while isinstance(flowi, list):
         print(f"Drilling down flow layer... type is {type(flowi)}")
         flowi = flowi[0]
         
    print(f"Final Mask shape: {maski.shape}")
    print(f"Final Flow shape: {flowi.shape}")

    fig = plt.figure(figsize=(12,5))
    plot.show_segmentation(fig, img, maski, flowi, channels=chans, omni=True)
    plt.tight_layout()
    # ==========================================
    # 8. SAVE OUTPUTS
    # ==========================================
    
    # 1. Save the Analysis Mask (The "Invisible" one, good for data)
    print("Saving raw analysis mask (output_raw_mask.tif)...")
    skimage.io.imsave(f"{save_dir}/output_raw_mask_omni.tif", maski.astype(np.uint16))

    # 2. Save a VISUAL Mask (Colored, good for human eyes)
    print("Saving visual check mask (output_visual_mask.tif)...")
    # Create a color overlay: background is black, cells are random colors
    visual_mask = skimage.color.label2rgb(maski, bg_label=0)
    # Convert from float (0-1) to byte (0-255) for saving
    visual_mask = skimage.util.img_as_ubyte(visual_mask)
    skimage.io.imsave(f"{save_dir}/output_visual_mask_omni.tif", visual_mask)

    io.imsave(f"{save_dir}/output_flows_omni.tif", flowi.astype(np.float32))

    # 3. Save the Plot
    print("Saving plot visualization...")
    plt.savefig(f"{save_dir}/output_plot_visualization.png", dpi=300)
    
    print("All files saved. Check 'output_visual_mask.tif' to see the cells.")
    plt.show()
    
    return visual_mask;
    

if __name__ == "__main__":
    main(image_obj=None,model_dir="",save_dir=".")