#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.

""" SPOTS - Spot characterisation and manipulation module

Description:
    spots.py contains the Spots class for characterising and manipulating data
    associated with bright spots in the image datasets provided.

Contains:
    class Spots

Author:
    Edward Higgins, Jack Shepherd

Version: 0.2.0
"""

import sys
import math
import cv2
import matplotlib.pyplot as plt
import numpy as np

from algorithms import *
from scipy.ndimage import gaussian_laplace, maximum_filter


class Spots:
    def __init__(self, num_spots=0, frame=0):
        self.num_spots = num_spots
        if num_spots > 0:
            self.positions = np.zeros([num_spots, 2])
            self.bg_intensity = np.zeros(num_spots)
            self.spot_intensity = np.zeros(num_spots)
            self.frame = frame
            self.traj_num = [-1] * self.num_spots
            self.snr = np.zeros([num_spots])
            self.laser_on_frame = 0
            self.converged = np.zeros([num_spots], dtype=np.int8)
            self.exists = True
            self.width = np.zeros((num_spots,2))
            # Lewis edits
            self.noise = np.zeros(num_spots)
            self.precision = np.zeros((num_spots,2))
        else:
            self.frame = frame
            self.exists = False

    def set_positions(self, positions):
        self.num_spots = len(positions)
        self.positions = np.zeros([self.num_spots, 2])
        self.clipping = [False] * self.num_spots
        self.bg_intensity = np.zeros(self.num_spots)
        self.spot_intensity = np.zeros(self.num_spots)
        self.centre_intensity = np.zeros(self.num_spots)
        self.width = np.zeros([self.num_spots, 2])
        self.traj_num = [-1] * self.num_spots
        self.snr = np.zeros([self.num_spots])
        self.converged = np.zeros([self.num_spots],dtype=np.int8)
        # Lewis edits
        self.noise = np.zeros(self.num_spots)
        self.precision = np.zeros((self.num_spots,2))

        for i in range(self.num_spots):
            self.positions[i, :] = positions[i]

    def find_in_frame(self, frame, params):
        img_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Get structural element (map with disk of ones in a square of 0s) [strel]
        disk_size = 2 * params.struct_disk_radius - 1
        disk_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (disk_size, disk_size)
        )

        # Optionally apply gaussian filtering to the frame
        if params.filter_image == "Gaussian":
            blurred_frame = cv2.GaussianBlur(img_frame, (3, 3), 0)
        else:
            blurred_frame = img_frame.copy()

        # Apply top-hat filtering [imtophat]
        tophatted_frame = cv2.morphologyEx(blurred_frame, cv2.MORPH_TOPHAT, disk_kernel)

        # Get b/w threshold value from the histogram
        hist_data = cv2.calcHist([tophatted_frame], [0], None, [256], [0, 256])
        hist_data[0] = 0

        peak_width, peak_location = fwhm(hist_data)
        bw_threshold = int(peak_location + params.bw_threshold_tolerance * peak_width)

        # Apply gaussian filter to the top-hatted image [fspecial, imfilter]
        blurred_tophatted_frame = cv2.GaussianBlur(tophatted_frame, (3, 3), 0)

        # Convert the filtered image to b/w [im2bw]
        bw_frame = cv2.threshold(
            blurred_tophatted_frame, bw_threshold, 255, cv2.THRESH_BINARY
        )[1]

        # "Open" the b/w image (in a morphological sense) [imopen]
        bw_opened = cv2.morphologyEx(
            bw_frame, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        )

        # Fill holes ofsize 1 pixel in the resulting image [bwmorph]
        bw_filled = cv2.morphologyEx(
            bw_opened,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )

        spot_locations = ultimate_erode(bw_filled[:, :, 0], frame)
        if np.isnan(spot_locations).any():
            raise "Found nans"
        self.set_positions(spot_locations)

    def merge_coincident_candidates(self):
        new_positions = []
        skip = []
        for i in range(self.num_spots):
            tmp_positions = [self.positions[i, :]]
            if i in skip:
                continue

            for j in range(i + 1, self.num_spots):
                if sum((self.positions[i, :] - self.positions[j, :]) ** 2) < 4:
                    skip.append(j)
                    tmp_positions.append(self.positions[j, :])

            p = [0, 0]
            for pos in tmp_positions:
                p[0] += pos[0]
                p[1] += pos[1]
            p[0] = p[0] / len(tmp_positions)
            p[1] = p[1] / len(tmp_positions)

            new_positions.append(p)

        self.set_positions(new_positions)

    def filter_candidates(self, frame, params):
        positions = []
        clipping = []
        bg_intensity = []
        spot_intensity = []
        centre_intensity = []
        width = []
        traj_num = []
        snr = []

        for i in range(self.num_spots):
            # Fliter spots that are too noisy to be useful candidates
            if self.snr[i] <= params.snr_filter_cutoff:
                continue
            
            # Filter spots too close to the edge to give good numbers
            if self.positions[i,0] < params.subarray_halfwidth \
              or self.positions[i,0] >= frame.frame_size[0] - params.subarray_halfwidth \
              or self.positions[i,1] < params.subarray_halfwidth \
              or self.positions[i,1] >= frame.frame_size[1] - params.subarray_halfwidth:
                continue
            
            # Fitler spots that are outside of any existing mask --> Lewis edit, this used to be before the edge filter spots were rounding to outside the mask dimensions
            #print(round(self.positions[i,1]),round(self.positions[i,0]))
            if frame.has_mask and frame.mask_data[round(self.positions[i,1]), round(self.positions[i,0])] == 0:
                continue

            # Filter spots that appear not to be circular (unless we are doing astigmatism)
            if params.astigmatism == False:
                if self.width[i,0] / self.width[i,1] >= 2 \
                   or self.width[i,1] / self.width[i,0] >= 2:
                    continue

            positions.append(self.positions[i, :])
            clipping.append(self.clipping[i])
            bg_intensity.append(self.bg_intensity[i])
            spot_intensity.append(self.spot_intensity[i])
            centre_intensity.append(self.centre_intensity[i])
            width.append(self.width[i, :])
            traj_num.append(self.traj_num[i])
            snr.append(self.snr[i])

        self.num_spots = len(clipping)
        self.positions = np.array(positions)
        self.clipping = np.array(clipping)
        self.bg_intensity = np.array(bg_intensity)
        self.spot_intensity = np.array(spot_intensity)
        self.centre_intensity = np.array(centre_intensity)
        self.width = np.array(width)
        self.traj_num = np.array(traj_num)
        self.snr = np.array(snr)



    def distance_from(self, other):
        distances = np.zeros([self.num_spots, other.num_spots])

        for i in range(self.num_spots):
            for j in range(other.num_spots):
                distances[i, j] = np.linalg.norm(
                    self.positions[i, :] - other.positions[j, :]
                )

        return distances

    def index_first(self):
        self.traj_num = list(range(self.num_spots))

    def link(self, prev_spots, params):
        distances = self.distance_from(prev_spots)

        assigned = []
        neighbours = np.argsort(distances[:, :], axis=1)
        paired_spots = []
        next_trajectory = max(prev_spots.traj_num) + 1
        for i in range(self.num_spots):
            for j in range(prev_spots.num_spots):
                neighbour = neighbours[i, j]
                if distances[i, neighbour] < params.max_displacement:
                    if neighbour in paired_spots:
                        continue
                    else:
                        paired_spots.append(neighbour)
                        self.traj_num[i] = prev_spots.traj_num[neighbour]
                else:
                    self.traj_num[i] = next_trajectory
                    next_trajectory += 1
                break

            if self.traj_num[i] == -1:
                sys.exit(f"Unable to find a match for spot {i}, frame {self.frame}")

    def get_spot_intensities(self, frame, params):
        for i in range(self.num_spots):
            x = round(self.positions[i, 0])
            y = round(self.positions[i, 1])
            # Create a tmp array with the centre of the spot in the centre
            tmp = frame[
                y - params.subarray_halfwidth : y + params.subarray_halfwidth+1, 
                x - params.subarray_halfwidth : x + params.subarray_halfwidth + 1
            ] 
            spotmask = np.zeros(tmp.shape)
            cv2.circle(spotmask, 
                    (params.subarray_halfwidth, params.subarray_halfwidth),
                    params.inner_mask_radius,
                    1,
                    -1
            )
            bgintensity = np.mean(tmp[spotmask == 0])
            tmp = tmp - bgintensity
            intensity = np.sum(tmp[spotmask == 1])
            if intensity == 0:
                print(f"WARNING: Zero intensity found at {[x, y]}")
            self.spot_intensity[i] = intensity
            self.bg_intensity[i] = bgintensity

    def refine_centres(self, frame, params):
        image = frame.as_image()
        # Refine the centre of each spot independently
        for i_spot in range(self.num_spots):
            r = params.subarray_halfwidth
            N = 2 * r + 1

            # Get the centre estimate, make sure the spot_region fits in the frame
            p_estimate = self.positions[i_spot, :]

            for d in (0, 1):
                if round(p_estimate[d]) < r:
                    p_estimate[d] = r
                elif round(p_estimate[d]) > frame.frame_size[d]-r-1:
                    p_estimate[d] = frame.frame_size[d] - r - 1

            # Create the sub-image
            spot_region = np.array(
                [
                    [round(p_estimate[0]) - r, round(p_estimate[0]) + r],
                    [round(p_estimate[1]) - r, round(p_estimate[1]) + r],
                ]
            ).astype(int)

            spot_pixels = image[
                spot_region[1, 0] : spot_region[1, 1] + 1,
                spot_region[0, 0] : spot_region[0, 1] + 1,
            ]

            coords = np.mgrid[
                spot_region[0, 0] : spot_region[0, 1] + 1,
                spot_region[1, 0] : spot_region[1, 1] + 1,
            ]

            Xs, Ys = np.meshgrid(
                range(spot_region[0, 0], spot_region[0, 1] + 1),
                range(spot_region[1, 0], spot_region[1, 1] + 1),
            )

            converged = False
            iteration = 0
            clipping = False
            spot_intensity = 0
            bg_intensity = 0
            snr = 0
            while not converged and iteration < params.gauss_mask_max_iter:
                iteration += 1

                # Generate the inner mask
                inner_mask = np.where(
                    (coords[0, :, :] - p_estimate[0]) ** 2
                    + (coords[1, :, :] - p_estimate[1]) ** 2
                    <= params.inner_mask_radius ** 2,
                    1,
                    0,
                )
                mask_pixels = np.sum(inner_mask)

                # Generate the Gaussian mask
                # This uses Numpy magic, it's almost as bad as the MATLAB...
                coords_sq = (
                    coords[:, :, :] - p_estimate[:, np.newaxis, np.newaxis]
                ) ** 2
                exponent = -(coords_sq[0, :, :] + coords_sq[1, :, :]) / (
                    2 * params.gauss_mask_sigma ** 2
                )
                gauss_mask = np.exp(exponent)

                if np.sum(gauss_mask) != 0:
                    gauss_mask /= np.sum(gauss_mask)

                bg_mask = 1 - inner_mask

                # Calculate the local background intensity and subtract it off the sub-image
                spot_bg = spot_pixels * bg_mask
                num_bg_spots = np.sum(bg_mask)
                bg_average = np.sum(spot_bg) / num_bg_spots

                # Calculate background corrected sub-image
                bg_corr_spot_pixels = spot_pixels - bg_average

                # Calculate revised position estimate
                spot_gaussian_product = bg_corr_spot_pixels * gauss_mask
                p_estimate_new = np.zeros(2)
                p_estimate_new[0] = np.sum(spot_gaussian_product * Xs) / np.sum(
                    spot_gaussian_product
                )
                p_estimate_new[1] = np.sum(spot_gaussian_product * Ys) / np.sum(
                    spot_gaussian_product
                )
                estimate_change = np.linalg.norm(p_estimate - p_estimate_new)

                if not np.isnan(p_estimate_new).any():
                    p_estimate = p_estimate_new
                else:
                    print("WARNING: Position estimate is NaN, falied to converge")
                    p_estimate = self.positions[i_spot,:]
                    converged = False
                    break

                spot_intensity = np.sum(bg_corr_spot_pixels * inner_mask)
                bg_std = np.std(spot_bg[bg_mask==1])

                if estimate_change < 1e-6:
                    converged = True

                # Calculate signal-noise ratio
                # Don't bother reiterating this spot if it's too low
                snr = abs(spot_intensity / (bg_std*np.sum(inner_mask)))
#EJH#                 snr = abs(spot_intensity / (bg_std*np.sum(inner_mask)))
                if snr <= params.snr_filter_cutoff:
                    break

            self.bg_intensity[i_spot] = bg_average
            self.spot_intensity[i_spot] = spot_intensity
            self.snr[i_spot] = snr
            self.converged[i_spot] = converged
            self.noise[i_spot] = bg_std # Lewis edit 
            self.positions[i_spot, :] = p_estimate
            
    def get_spot_widths(self, frame, params):
        for i in range(self.num_spots):
            x = round(self.positions[i, 0])
            y = round(self.positions[i, 1])
            # Create a tmp array with the centre of the spot in the centre
            tmp = frame[
                y - params.subarray_halfwidth : y + params.subarray_halfwidth+1, 
                x - params.subarray_halfwidth : x + params.subarray_halfwidth + 1
                ] 
            spotmask = np.zeros(tmp.shape)
            cv2.circle(spotmask, 
             (params.subarray_halfwidth, params.subarray_halfwidth),
             params.inner_mask_radius,
             1,
             -1
             )
            bgintensity = np.mean(tmp[spotmask == 0])
            tmp = tmp - bgintensity
            p, succ = fit2Dgaussian(tmp)
            if succ==1: # the fit is OK
                self.width[i,0] = p[3]
                self.width[i,1] = p[4]
            else: # something went wrong
                self.width[i,0] = params.psf_width
                self.width[i,1] = params.psf_width
                
    def get_precision(self, frame, params): # Lewis edit 
        for i in range(self.num_spots):
            width_x = self.width[i,0] * params.pixel_size * 1000
            width_y = self.width[i,1] * params.pixel_size * 1000
            
            a = params.pixel_size * 1000
            
            qe = 0.95
            gain = 1
            
            intensity = self.spot_intensity[i] 
            bg_int = self.bg_intensity[i]
            N = intensity #(intensity * gain)/qe
            b = self.noise[i]
            
            precision_x = (width_x**2 + (a**2)/12)/(N) + (8*math.pi*(width_x**4)*(b**2))/((a**2)*(N**2))
            
            precision_y = (width_x**2 + (a**2)/12)/(N) + (8*math.pi*(width_x**4)*(b**2))/((a**2)*(N**2))
            
            self.precision[i,0] = np.sqrt(precision_x)
            self.precision[i,1] = np.sqrt(precision_y)
            

#################### NEW TRACKING #########################
    def detect(self,frame_data, frame_idx, p):
        """Multi-scale Laplacian-of-Gaussian. Returns integer (y, x) candidates.

        The response is taken as the per-pixel maximum over several PSF widths, so
        spots a little narrower or wider than `psf_sigma` are still found. The
        sigma**2 factor makes responses comparable across scales.

        The threshold is set per frame from the median and MAD of the response
        rather than an absolute value, so it follows bleaching and uneven
        illumination. MAD is used because the spots themselves are the outliers and
        would inflate a standard deviation.
        """
        frame = frame_data.as_image()
        if p.presmooth_sigma > 0:
            from scipy.ndimage import gaussian_filter
            frame = gaussian_filter(frame, p.presmooth_sigma)

        resp = None
        for scale in p.detect_scales:
            sigma = p.psf_sigma * float(scale)
            r = -gaussian_laplace(frame, sigma) * sigma ** 2
            resp = r if resp is None else np.maximum(resp, r)

        med = np.median(resp)
        mad = np.median(np.abs(resp - med)) + 1e-9
        thresh = med + p.detect_thresh_k * 1.4826 * mad

        sep = int(p.min_separation)
        peaks = (maximum_filter(resp, size=sep) == resp) & (resp > thresh)
        y, x = np.nonzero(peaks)
        # Spots nearer the edge than the fitting window cannot be localised.
        b = sep
        keep = ((y >= b) & (y < frame.shape[0] - b) &
                (x >= b) & (x < frame.shape[1] - b))
        if keep.size > 0:
            pos = np.zeros((sum(keep),2))
            pos[:,1] = x[keep]
            pos[:,0] = y[keep]
            self.set_positions(pos)
