# -*- coding: utf-8 -*-

import os
from PIL import Image  
import numpy as np  
import matplotlib.pyplot as plt  
import cv2
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from pystackreg import StackReg
import argparse
from skimage import io
import glob 

def get_sort_key(s):
    parts = s.split("_")
    numerical_value = int(parts[-1])
    return numerical_value

def main(folder_name,save_dir,num_channels,channel,tmats_path):
    
    folders = [f for f in os.listdir(folder_name) if (os.path.isdir(os.path.join(folder_name, f)) and os.path.join(f)[:2] != "BF")]
    # New code (checks for the 'results' subfolder)
    folders = [
        f for f in os.listdir(folder_name) 
        if os.path.isdir(os.path.join(folder_name, f)) 
        and f[:2] != "BF" 
        and os.path.isdir(os.path.join(folder_name, f, "results"))
    ]
    folders = sorted(folders,key=get_sort_key)
    print(folders)
    
    # Create a matplotlib figure
    if num_channels == 1:
        fig, axes = plt.subplots( len(folders),3, figsize=(15, 2.5*len(folders)))  # 2 rows, num_images columns
    if num_channels == 2:
        fig, axes = plt.subplots( len(folders),5, figsize=(15, 2.5*len(folders)))
    canvas = FigureCanvas(fig)
    blank = np.zeros((256,256))
    
    for i,folder in enumerate(folders):
        print(folder)
        # Point to the specific FOV's folder
        fov_path = os.path.join(folder_name, folder)
        results_path = os.path.join(fov_path, "results")
        
        # CHANGED: Use the full folder name instead of splitting it
        name = folder 
        
        # Check if the results folder exists (instead of cell folders)
        if os.path.exists(results_path):
            
            if num_channels == 1:
                if channel == "R":
                    img = io.imread(f"{fov_path}/results/R_avg.tif")
                    color = "Reds"
                elif channel == "L":
                    img = io.imread(f"{fov_path}/results/L_avg.tif")
                    color = "Greens"
                    
                blurred_img = cv2.GaussianBlur(img, ksize=(5,5), sigmaX=0)
                
                try:
                    mask = io.imread(f"{fov_path}/results/all_cells_mask.tif")
                except FileNotFoundError:
                    print(f"Warning: Missing mask for {name}, using blank.")
                    mask = blank.copy()
      
                try:
                    bf_img = io.imread(f"{fov_path}/results/BF.tif")
                except:
                    bf_img = blank.copy()
                
                axes[i, 0].imshow(bf_img, cmap='gray')
                axes[i, 0].axis('off')  # Turn off axes
                axes[i, 0].set_title(f'Brightfield\n{name}', fontsize=9)
                
                axes[i, 1].imshow(img,cmap=color)
                axes[i, 1].axis('off')  # Turn off axes
                axes[i, 1].set_title(f'Left\n{name}', fontsize=9)
            
                axes[i, 2].imshow(mask,cmap='gray')
                axes[i, 2].axis('off')  # Turn off axes
                axes[i, 2].set_title(f'Right\n{name}', fontsize=9)
                
            
            if num_channels == 2: 
                img_L = io.imread(f"{fov_path}/results/L_avg.tif")
                img_R = io.imread(f"{fov_path}/results/R_avg.tif")
                
                img_L = cv2.GaussianBlur(img_L, ksize=(5,5), sigmaX=0)
                img_R = cv2.GaussianBlur(img_R, ksize=(5,5), sigmaX=0)
                
                max_int = 255
                img_L = img_L - np.mean(img_L)
                img_L[img_L<0] = 0
                
                sr = StackReg(StackReg.RIGID_BODY)
                transformation_matrix = np.load(tmats_path)
                
                try:
                    mask_L = io.imread(f"{fov_path}/results/all_cells_mask.tif")
                except FileNotFoundError:
                    print(f"Warning: Missing mask for {name}, using blank.")
                    mask_L = blank.copy()
                    
                try:
                    bf_img = io.imread(f"{fov_path}/results/BF.tif")
                except:
                    bf_img = blank.copy()
                
                sr = StackReg(StackReg.RIGID_BODY)
                transformation_matrix = np.load(tmats_path)
                
                bkg = np.mean(img_R)
                img_R = sr.transform(img_R , transformation_matrix)
                img_R[img_R == 0] = bkg
                img_R = img_R - bkg
                img_R[img_R<0] = 0
                
                axes[i, 0].imshow(bf_img, cmap='gray')
                axes[i, 0].axis('off')  # Turn off axes
                axes[i, 0].set_title(f'Brightfield\n{name}', fontsize=9)
                
                axes[i, 1].imshow(img_L,cmap="Greens")
                axes[i, 1].axis('off')  # Turn off axes
                axes[i, 1].set_title(f'Left\n{name}', fontsize=9)
                
                axes[i, 2].imshow(img_R,cmap="Reds")
                axes[i, 2].axis('off')  # Turn off axes
                axes[i, 2].set_title(f'Right\n{name}', fontsize=9)
                
                img_R= img_R/np.amax(img_R) * 255
                img_L= img_L/np.amax(img_L) * 255
                
                ####
                composite_merge = np.zeros((img_L.shape[0], img_L.shape[1], 3), dtype=np.uint8)
                composite_merge[:,:,0] = img_R
                composite_merge[:,:,1] = img_L
                Image.fromarray(composite_merge).save(f"{fov_path}/results/merge_img.tif")
                ####
                
                axes[i, 3].imshow(composite_merge)
                axes[i, 3].axis('off')  # Turn off axes
                axes[i, 3].set_title(f'Merge\n{name}', fontsize=9)
            
                axes[i, 4].imshow(mask_L,cmap="gray")
                axes[i, 4].axis('off')  # Turn off axes
                axes[i, 4].set_title(f'Mask\n{name}', fontsize=9)
            
        else:
            print(i)
            if num_channels == 1:
                
                axes[i, 0].imshow(blank)
                axes[i, 0].axis('off')  # Turn off axes
                axes[i, 0].set_title(f'Brightfield\n{name}', fontsize=9)
                
                axes[i, 1].imshow(blank)
                axes[i, 1].axis('off')  # Turn off axes
                axes[i, 1].set_title(f'Left\n{name}', fontsize=9)
            
                axes[i, 2].imshow(blank)
                axes[i, 2].axis('off')  # Turn off axes
                axes[i, 2].set_title(f'Right\n{name}', fontsize=9)
                
            if num_channels == 2: 
                
                axes[i, 0].imshow(blank, cmap='gray')
                axes[i, 0].axis('off')  # Turn off axes
                axes[i, 0].set_title(f'Brightfield\n{name}', fontsize=9)
                
                axes[i, 1].imshow(blank,cmap="Greens")
                axes[i, 1].axis('off')  # Turn off axes
                axes[i, 1].set_title(f'Left\n{name}', fontsize=9)
                
                axes[i, 2].imshow(blank,cmap="Reds")
                axes[i, 2].axis('off')  # Turn off axes
                axes[i, 2].set_title(f'Right\n{name}', fontsize=9)
                
                axes[i, 3].imshow(blank)
                axes[i, 3].axis('off')  # Turn off axes
                axes[i, 3].set_title(f'Merge\n{name}', fontsize=9)
            
                axes[i, 4].imshow(blank)
                axes[i, 4].axis('off')  # Turn off axes
                axes[i, 4].set_title(f'Mask\n{name}', fontsize=9)
             
    plt.tight_layout()
    plt.savefig(f"{save_dir}/All_FoVs.png")         

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get user information from command line.")
    # Define the arguments
    parser.add_argument("--num_channels",default=2)
    parser.add_argument("--channel",default=None) # Change to always convert to upper 
    parser.add_argument("--mask_type",default="BF") 
    parser.add_argument("--video_path")
    parser.add_argument("--bf_path",default=None) 
    parser.add_argument("--tmats_path",default=None) 
    parser.add_argument("--model",default=None) 
    parser.add_argument("--pxsize",default=51e-9)
    parser.add_argument("--roi_channel") # Change to always convert to upper 
    parser.add_argument("--roi_file",default="./circle.roi")
    parser.add_argument("--save_dir",default=".")
    parser.add_argument("--area_filter",default="400")
    parser.add_argument("--inv_bf",default=False) 
    parser.add_argument("--folder",default=".")
    
    # Parse the arguments
    args = parser.parse_args()
    
    global pxsize 
    pxsize = args.pxsize
    
    with open(f'{args.save_dir}/parameters.txt', 'w') as f:
            for arg_name, arg_value in vars(args).items():
                f.write(f"{arg_name}: {arg_value}\n")
    
    # Call the main function with the parsed arguments
    main(args.folder,args.save_dir,int(args.num_channels),args.channel,args.tmats_path)