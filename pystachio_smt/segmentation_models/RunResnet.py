# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 16:03:18 2025

@author: lf1017
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np
print(np.version.version)
import os
import torch
import celldetection as cd
from scipy import ndimage 
import nd2
import shutil
from skimage import measure, morphology

import csv

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

torch.set_num_threads(1)
cv2.setNumThreads(0)
directory = "."
model = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device in use: {device}')

# Fetching the CpnResNet50 UNet model variant
model = cd.fetch_model('ginoro_cpnresnet50_UNet', check_hash=True, progress=True).to(device)    
model.eval()

for img_file in os.listdir("./Images"):
    if img_file.endswith(".nd2"):
        img_path = img_file
        print(f'Processing: {img_path}')
        folder_name = f'./{os.path.splitext(img_path)[0]}'
        
        try:
            os.makedirs(folder_name)
        except Exception:
            shutil.rmtree(folder_name)
            os.makedirs(folder_name)
        
        outfile = open(f'{folder_name}/results.tsv', 'w', newline='')
        fieldnames = ["cell_num", "eccentricity", "length_skl", "area_ski", "a", "edge"]  
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        
        input_image = nd2.imread(f'./Images/{img_path}')
        min_val = np.min(input_image)
        max_val = np.max(input_image)
        img_8bit_1channel = np.uint8(((input_image - min_val) / (max_val - min_val)) * 255)
        print(f'Image shape: {input_image.shape}')
        
        img_8bit_3channel = cv2.cvtColor(img_8bit_1channel, cv2.COLOR_GRAY2RGB)
        
        with torch.no_grad():
            x = cd.to_tensor(img_8bit_3channel, transpose=True, device=device, dtype=torch.float32)
            x = x / 255  # ensure 0..1 range
            x = x[None]  # add batch dimension: Tensor[3, h, w] -> Tensor[1, 3, h, w]
            y = model(x)
        
        print(y)
        contours = y['contours'][0]
        np.save(f'{folder_name}/celldetection_output_contours.npy', cd.asnumpy(contours))
        cd.imshow_row(x, x, figsize=(16, 9), titles=('input', 'contours'))
        cd.plot_contours(contours, contour_line_width=1)
        plt.savefig(f'{folder_name}/celldetection_output_img.png', dpi=300)
        plt.close()
        print(f'Contours detected: {contours.shape}')
        
        full_mask = np.zeros(input_image.shape, dtype=np.uint8)
        
        for index, contour in enumerate(contours):
            np_contour = cd.asnumpy(contour)
            mask = np.zeros(input_image.shape, dtype=np.uint8)
            np_contour = np_contour.reshape(-1, 1, 2).astype(np.int32)
            mask = cv2.drawContours(mask, [np_contour], -1, color=255, thickness=cv2.FILLED)
            mask = mask.astype(np.uint8)
            
            label_im, nb_labels = ndimage.label(mask) 
            print(f'Cell index {index} - Number of labels: {nb_labels}')
        
            object_features = measure.regionprops(label_im)
            
            label = 0
            label_i = object_features[label].label
            im = (label_im == label_i)
            im = np.asarray(im, dtype=np.uint8)
        
            eccentricity = getattr(object_features[label], 'eccentricity')
            area_ski = getattr(object_features[label], 'area')
            
            # Find bounding box coordinates
            rows, cols = np.where(im > 0)
            margin = 10
            min_row = max(0, np.min(rows) - margin)
            max_row = min(im.shape[0] - 1, np.max(rows) + margin)
            min_col = max(0, np.min(cols) - margin)
            max_col = min(im.shape[1] - 1, np.max(cols) + margin)
            
            if max_row == im.shape[0] - 1 or min_row == 0 or max_col == im.shape[1] - 1 or min_col == 0:
                edge = True
            else: 
                edge = False
                
            # Crop image slice
            im = im[min_row:max_row + 1, min_col:max_col + 1]
            
            plt.imshow(im, cmap='binary_r', interpolation='nearest')
            plt.close()
        
            skeleton = morphology.skeletonize(im).astype(np.float32)
            y_skl, x_skl = np.where(skeleton > 0)
           
            try:
                a, b, c = np.polyfit(x_skl, y_skl, 2)
            except Exception as e:
                print(f"Polynomial fit error: {e}. Skipping cell {index}.")
                continue
                
            # Generate points for the fitted curve
            x_fit = np.linspace(0, im.shape[1], 1000)
            
            y_fit_masked = []
            x_fit_masked = []
            for x_val in x_fit:
                y_val = a * (x_val**2) + b * x_val + c
                y_val_round = int(round(y_val))
                x_val_round = int(round(x_val))
                if 0 <= y_val_round < im.shape[0] and 0 <= x_val_round < im.shape[1] and im[y_val_round, x_val_round] > 0:
                    y_fit_masked.append(y_val)
                    x_fit_masked.append(x_val)
            
            x_fit_masked = np.array(x_fit_masked)
            y_fit_masked = np.array(y_fit_masked)
            
            if len(x_fit_masked) > 1:
                curve = np.column_stack((x_fit_masked, y_fit_masked))
                length_skl = np.sum(np.sqrt(np.sum((curve[:-1] - curve[1:])**2, axis=1)))
            else:
                length_skl = 0.0

            plt.imshow(im, cmap='binary_r', interpolation='nearest')
            plt.imshow(skeleton, cmap=plt.cm.Reds, alpha=0.5)
            plt.plot(x_fit_masked, y_fit_masked, c='blue', label='Quadratic Fit', linewidth=2)
            plt.title(f'Area: {round(area_ski, 3)}, Eccentricity: {round(eccentricity, 3)}, Length: {round(length_skl, 3)}, a: {round(a, 5)}')
            plt.savefig(f'{folder_name}/cell_{index}_skeleton.png')
            plt.close()
        
            cv2.imwrite(f'{folder_name}/cell_{index}_mask_img.tif', mask)    
        
            stats = {
                "cell_num": index,
                "eccentricity": eccentricity,
                "length_skl": length_skl,
                "area_ski": area_ski,
                "a": a,
                "edge": edge
            }
            writer.writerow(stats)
            full_mask = full_mask + mask 
        
        outfile.close()
        full_mask[full_mask > 255] = 255
        cv2.imwrite(f'{folder_name}/CpnResNet_Prediction.tif', full_mask.astype(np.uint8))