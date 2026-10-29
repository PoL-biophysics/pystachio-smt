"""
Post processing trajectories and intensities

Routines for getting isingle and diffusion coefficient from lists of intensities

v0.1 Jack W Shepherd, University of York
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
from scipy.spatial import distance_matrix
import os
import trajectories, images
import cv2 as cv2

def postprocess(params, simulated=False):
    if not params.ALEX and params.use_channel!="Both":
        trajs = []
        if simulated:
            trajs = trajectories.read_trajectories(params.name + "_simulated_trajectories.tsv")
        else:
            trajs = trajectories.read_trajectories(params.name + "_trajectories.tsv")
        spots = trajectories.to_spots(trajs)
        if params.verbose:
            print(f"Looking at {len(trajs)} trajectories across {len(spots)} frames")

        intensities = np.array([])
        # Get a cutoff frame so we use last 2/3 of data for Isingle
        lastframe = 0
        for traj in trajs:
            if traj.end_frame > lastframe:
                lastframe = traj.end_frame
        cutoff_frame = int(lastframe*(1.-params.isingle_percent_frames/100.))
        snrs = np.array([])
        for i in range(len(spots)):
            tmp_snr = spots[i].snr
            if spots[i].frame >= cutoff_frame:
                tmp = spots[i].spot_intensity
                intensities = np.concatenate((intensities,tmp))
            snrs = np.concatenate((snrs,tmp_snr))

        if params.calculate_isingle:
            try:
                calculated_isingle = get_isingle(params,intensities)
            except:
                print("Error finding Isingle - using default/specified value")
        else:
            calculated_isingle = params.I_single
#        calculated_snr = plot_snr(params,snrs)
        dc, lp = get_diffusion_coef(trajs, params)
        if simulated:
            if params.verbose:
                print(f"Simulated diffusion coefficient: {np.mean(dc)}")
                print(f"Simulated Isingle:               {calculated_isingle[0]}")
        else:
            if params.verbose:
                print(f"Tracked diffusion coefficient: {np.mean(dc)}")
            #print(f"Tracked Isingle:               {calculated_isingle[0]}")

        plot_traj_intensities(params, trajs, params.chung_kennedy)
        get_stoichiometries(trajs, calculated_isingle, params)
        if params.copy_number==True: get_copy_number(params, calculated_isingle)
        overtrack(params, trajs)

    elif params.ALEX or params.use_channel=="Both":
        Rtrajs = trajectories.read_trajectories(params.name + "_Rchannel_trajectories.tsv")
        Ltrajs = trajectories.read_trajectories(params.name + "_Lchannel_trajectories.tsv")
        Rspots = trajectories.to_spots(Rtrajs)
        Lspots = trajectories.to_spots(Ltrajs)
        Llastframe = 0
        for traj in Ltrajs:
            if traj.endframe > Llastframe:
                Llastframe = traj.end_frame
        Lcutoff_frame = int(Llastframe*params.isingle_fraction/100.)
        Rlastframe = 0
        for traj in Rtrajs:
            if traj.endframe > Rlastframe:
                Rlastframe = traj.end_frame
        Rcutoff_frame = int(Rlastframe*params.isingle_fraction/100.)

        Rintensities= np.array([])
        Rsnrs = np.array([])
        Rintensities= np.array([])
        Rsnrs = np.array([])
        Lintensities= np.array([])
        Lsnrs = np.array([])
        Lintensities= np.array([])
        Lsnrs = np.array([])         
        for i in range(len(Rspots)):
            if Rspots[i].frame >= Rcutoff_frame:
                Rintensities = np.concatenate((Rintensities,Rspots[i].spot_intensity))
            Rsnrs = np.concatenate((Rsnrs,Rspots[i].snr))
        for i in range(len(Lspots)):
            if Lspots[i].frame >= Lcutoff_frame:
                Lintensities = np.concatenate((Lintensities,Lspots[i].spot_intensity))
            Lsnrs = np.concatenate((Lsnrs,Lspots[i].snr))            
        if params.calculate_isingle:
            try:
                R_isingle = get_isingle(params,Rintensities, channel="R")
            except:
                print("Error finding Isingle RHS - using default/specified value")
                R_isingle = params.R_isingle
            try:
                L_isingle = get_isingle(params,Lintensities, channel="L")
            except:
                L_isingle = params.L_isingle
                print("Error finding Isingle LHS - using default/specified value")
        else:
            R_isingle = params.R_isingle
            L_isingle = params.L_isingle

        try:
            L_calculated_snr = plot_snr(params,Lsnrs, channel="L")
        except:
            print("Error finding LHS SNRs - you have probably not got any tracks")
        try:
            R_calculated_snr = plot_snr(params,Rsnrs, channel="R")
        except:
            print("Error finding RHS SNRs - you have probably not got any tracks")
        try:
            Ldc, Llp = get_diffusion_coef(Ltrajs, params, channel="L")
        except:
            print("Error finding LHS diffusion - you have probably not got any tracks")
        try:
            Rdc, Rlp = get_diffusion_coef(Rtrajs, params, channel="R")
        except:
            print("Error finding RHS diffusion - you have probably not got any tracks")
        try:
            plot_traj_intensities(params, Ltrajs, channel="L")
        except:
            print("Error plotting LHS intensities - you have probably not got any tracks")
        try:
            plot_traj_intensities(params, Rtrajs, channel="R")
        except:
            print("Error plotting RHS intensities - you have probably not got any tracks")
        get_stoichiometries(Ltrajs, L_isingle, params, channel="L")
        get_stoichiometries(Rtrajs, R_isingle, params, channel="R")
        
        if params.copy_number==True: 
            get_copy_number(params, Lcalculated_isingle, channel="L")
            get_copy_number(params, Rcalculated_isingle, channel="R")

        if params.colocalize: colocalize(params, Ltrajs, Rtrajs)
        overtrack(params, Ltrajs, channel="L")
        overtrack(params, Rtrajs, channel="R")

    else: sys.exit("ERROR: look do you want ALEX or not?\nSet params.ALEX=True or False")

def chung_kennedy_filter(data, window, R): #God only knows
    # Based on Adam Wollman's code, just translated
    # Originally based on some old Fortran written in 2000
    # A good Fortran programmer can write Fortran in any language....
    # ... Even 21 years later!
    N = len(data)
    extended_data = np.zeros(N+2*window)
    extended_data[window:-window]=data
    for i in range(1,window+1):
        extended_data[window-i] = data[i-1]
        extended_data[N+window+i-1] = data[N-i]  
    N_extended = extended_data.shape[0]
    wdiffx = np.zeros(N_extended)
    datamx = np.zeros((N+window,window))
    datx = np.zeros((N+window+1,window))
    for i in range(window):
        stop = N+window+i
        datamx[:,i] = extended_data[i:stop]
    wx = np.mean(datamx,axis=1)
    sx = np.std(datamx,axis=1,ddof=1)
    XP = wx[:N_extended]
    XM = wx[window+1:N_extended+window+1]
    SDP = sx[:N]
    SDM = sx[window:N_extended+window]
    DSD = SDP-SDM
    SP = SDP**2
    SM = SDM**2
    
    # Form switching functions (?)
    RSP = SP**R
    RSM = SM**R
    GM = RSP/(RSP+RSM)
    GP = RSM/(RSP+RSM)

    S = np.zeros(GP.shape)
    XX = np.zeros(GP.shape)
    for i in range(len(GP)-1):
        if GM[i]>=0 and GM[i]<=1 and GP[i]>=0 and GP[i]<=1:
            S[i] = GM[i]*SM[i] + GP[i]*SP[i]
            XX[i] = GP[i]*XP[i] + GM[i]*XM[i]
        else:
            S[i] =  SP[i]
            XX[i] = XP[i]
    # SD = np.sqrt(S)
    # SE = np.sqrt(S/window)
    # TX = (XP-XM)/(np.sqrt(2.)*SE);
    # DX = (XP-XM)
    # XPRE = XP
    return [XX] #[XX,TX,DX,SD,DSD,XPRE]

def colocalize(params, Ltrajs, Rtrajs):
    image_data = images.ImageData()
    image_data.read(params.name + '.tif', params)
    imageL=np.zeros((image_data.num_frames//2,image_data.frame_size[1],image_data.frame_size[0]//2))
    imageR=np.zeros((image_data.num_frames//2,image_data.frame_size[1],image_data.frame_size[0]//2))
    if params.start_channel=='L':
        for i in range(0,image_data.num_frames-1,2):
            imageL[i//2,:,:] = image_data.pixel_data[i,:,:image_data.frame_size[0]//2]
            imageR[i//2,:,:] = image_data.pixel_data[i+1,:,image_data.frame_size[0]//2:]
    else:
        for i in range(0,image_data.num_frames-1,2):
            imageR[i//2,:,:] = image_data.pixel_data[i,:,image_data.frame_size[0]//2:]
            imageL[i//2,:,:] = image_data.pixel_data[i+1,:,:image_data.frame_size[0]//2]
    Llinks = []
    Rlinks = []
    nlinks = []
    for i in range(params.num_frames):
        s1 = []
        s2 = []
        for traj in Ltrajs:
            if traj.start_frame<=i and traj.end_frame>=i:
                s1.append([traj.path[i-traj.start_frame],traj.width,traj.id])
        for traj in Rtrajs:
            if traj.start_frame<=i and traj.end_frame>=i:
                s2.append([traj.path[i-traj.start_frame],traj.width,traj.id])
        id1, id2 = linker(params,s1,s2)
        for j in range(len(id1)):
            found = False
            if id1[j] in Llinks:
                indices = np.where(Llinks==id1)
                for index in indices:
                    if Rlinks[index]==id2[j]:
                        nlinks[index] += 1
                        found = True
            if found == False:
                Llinks.append(id1[j])
                Rlinks.append(id2[j])
                nlinks.append(1)
    outfile = params.name + "_colocalized_trajectories.tsv"
    f = open(outfile, 'w')
    f.write("Left_traj\tRight_traj\tn_frames\n")
    Lids = []; Rids = []
    for i in range(len(Llinks)):
        Lids.append(Llinks[i][2])
        Rids.append(Rlinks[i][2])
    f.write(f'{Lids}\n{Rids}\n')
    Lids = np.array(Lids).astype(int)
    Rids = np.array(Rids).astype(int)
    for i in np.unique(Lids):
        R_colocs = Rids[Lids==i]
        for j in np.unique(R_colocs):
            if len(np.where(R_colocs==j))>=params.colocalize_n_frames:
                f.write(f"{i}\t{j}\t{len(np.where(R_colocs==j))}\n")
    f.close()
            
                        
                        
    colors = ['r','b','g','m','y','c']
    c = 0
    z, y, x = imageL.shape
    disp_frame = np.zeros((y,2*x))
    disp_frame[:,:x] = imageL[1,:,:]
    disp_frame[:,x:] = imageR[1,:,:]
    plt.imshow(disp_frame, cmap="Greys_r")
    plt.plot([x,x],[0,x], 'w--')
    plt.xticks([])
    plt.yticks([])
    plt.ylim([0,x])
    if params.display_figures:
        plt.show(block=True)
    plt.imshow(disp_frame, cmap="Greys_r")
    Lids = np.array(Lids)
    Rids = np.array(Rids)
    for traj in Ltrajs:
        if traj.id in Lids:
            plt.scatter(traj.path[0][0], traj.path[0][1], 50, marker='x', color=colors[c])
            index = np.argwhere(Lids==traj.id)[0][0]
            for t2 in Rtrajs:
                if t2.id == Rids[index]:
                    plt.scatter(t2.path[0][0]+x, t2.path[0][1], 50, marker='x', color=colors[c])
            c = (c+1) % len(colors)

    plt.yticks([])
    plt.xticks([])
    plt.plot([x,x],[0,x], 'w--')
    plt.ylim([0,x])
    plt.savefig("colocalized_spots.png", dpi=300)
    if params.display_figures:
        plt.show(block=True)
    plt.close()

def linker(params, spots1, spots2): #spots1 and spots2 are arrays going xpos, ypos, xwidth, ywidth, traj#
    s1_pos = []
    s2_pos = []
    s1_width = []
    s2_width = []
    for i in range(len(spots1[:])): 
        s1_pos.append(spots1[i][0])
        s1_width.append(spots1[i][1])
    for i in range(len(spots2[:])): 
        s2_pos.append(spots2[i][0])
        s2_width.append(spots2[i][1])
    try:
        dm = distance_matrix(s1_pos,s2_pos)
    except:
        return np.array([]), np.array([])
    dm[dm>params.colocalize_distance]=-1
    overlap = np.zeros(dm.shape)
    for i in range(len(s1_pos)):
        for j in range(len(s2_pos)):
            if dm[i,j] >= 0:
                overlap[i,j] = np.exp(-dm[i,j]**2 / (2.*((0.5*(s1_width[i][0][0]+s1_width[i][0][1])**2+(0.5*(s2_width[j][0][0]+s2_width[j][0][1])**2 )))))
    overlap[overlap<0.75]=0
    indices = np.argwhere(overlap!=0)
    id1=[];id2=[]
    for i in range(len(indices)):
        id1.append(spots1[indices[i][0]])
        id2.append(spots2[indices[i][1]])
    return np.array(id1), np.array(id2)


def straightline(x, m, c):
    return m * x + c

def get_copy_number(params, calculated_isingle, channel=None):
    image_data = images.ImageData()
    image_data.read(params)
    frame =image_data.pixel_data[spots.laser_on_frame,:,:]
    copy_nums = []
    f = open(params.name + "_copy_numbers.tsv")
    f.write("Cell\tCopy number\n")
    bg = np.mean(frame[image_data.mask==0])
    for i in range(1,np.amax(image_data.mask_data)+1):
        copy_nums.append(np.sum(frame[image_data.mask_data==i])/calculated_isingle)
        f.write(str(i)+"\t"+str(copy_nums[i-1])+"\n")
    f.close()
    # import tifffile as tiff
    # copy_nums = []
    # f = open(params.name + "_copy_numbers.tsv", 'w')
    # f.write("Cell\tCopy number\n")
    # for sim in range(1,11):
    #     image_data = tiff.imread("../data/Data4Jack/simImages/"+str(sim)+".tif")
    #     mask = tiff.imread("../data/Data4Jack/simImages/"+str(sim)+"_mask.tif")
    #     bg = np.mean(image_data[mask==0])
    #     copy_nums.append(np.sum(image_data[mask>0]-bg)/calculated_isingle)
    # for i in range(1,11):
    #     f.write(str(i)+"\t"+str(copy_nums[i-1])+"\n")
    # f.close()

def plot_snr(params,snr,channel=None):
    bandwidth=0.07
    kde = gaussian_kde(snr[snr != 0], bw_method=bandwidth)
    x = np.linspace(0, np.amax(snr), 10000)
    pdf = kde.evaluate(x)
    fig, ax1 = plt.subplots()
    ax1.hist(snr[snr != 0], bins=np.arange(0,np.amax(snr)+2,0.05), label="Raw data")
    ax2 = ax1.twinx()
    ax2.plot(x, pdf, 'k-', label="Gaussian KDF")
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,2))
    plt.legend()
    if channel=="L": 
        plt.title("Left hand channel SNR plot")
        outseed = params.name + "Lchannel_SNR"
    elif channel=="R":
        plt.title("Right hand channel SNR plot")
        outseed = params.name + "Rchannel_SNR"
    else:
        plt.title("Whole frame SNR plot")
        outseed = params.name + "_SNR"
    plt.savefig(outseed+"_plot.png", dpi=300)
    if params.display_figures:
        plt.show(block=True)
    plt.close()
    peak = x[np.where(pdf == np.amax(pdf))]
    ofile = params.name + "_data.tsv"
    f = open(ofile, 'w')
    for i in range(len(snr)):
        f.write(str(snr[i])+"\n")
    f.close()
    return peak


def get_isingle(params, intensities, channel=None):
    scale = 3
    intensities = intensities[intensities > 1]
    bandwidth = 0.1
    kde = gaussian_kde(intensities, bw_method=bandwidth)
    x = np.linspace(0, np.amax(intensities), int(np.amax(intensities)))
    pdf = kde.evaluate(x)
    peak = x[np.where(pdf == np.amax(pdf))].astype('int')
    peakval = np.amax(pdf)
    fig, ax1 = plt.subplots()
    plt.xlabel("Intensity (camera counts per pixel x$10^%i$)"%(scale))
    plt.ylabel("Number of foci")
    l1 = ax1.hist(
        intensities/10**scale,
        bins=np.arange(0, np.amax(intensities/10**scale) + 100/10**scale, 100/10**scale),
        label="Raw data", color="gray"
    )
    ax2 = ax1.twinx()
    l2 = ax2.plot(x/10**scale, pdf*10**scale, "k-", label="Gaussian KDE")
    plt.ylabel("Probability density (a.u.)")
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 2))
    l3 = ax1.plot([peak/10**scale,peak/10**scale],[0,10], 'r--', label=f"Isingle={peak}", lw=2)
    if channel=="L": 
        plt.title("Left hand channel intensity plot\nIsingle = %i"%(np.round(peak)))
        outseed = params.name + "Lchannel_intensities"
    elif channel=="R":
        plt.title("Right hand channel intensities plot\nIsingle = %i"%(np.round(peak)))
        outseed = params.name + "Rchannel_intensity"
    else:
        #plt.title("Whole frame intensities\nIsingle = %i"%(np.round(peak)))
        outseed = params.name + "_intensity"
    lns = l2+l3
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc=0)
    plt.savefig(outseed+"_plot.png", dpi=300)   
    if params.display_figures:
        plt.show(block=True)
    plt.close()
    ofile = outseed + "_data.tsv"
    f = open(ofile, 'w')
    for i in range(len(intensities)):
        f.write(str(intensities[i])+"\n")
    f.close()
    f = open(params.name+"_i_single_peak.dat", 'w')
    f.write(str(peak))
    f.close()
    
    return peak


def get_diffusion_coef(traj_list, params, channel=None):
    diffusion_coefs = []
    ids = []
    loc_precisions = []
    for traj in traj_list:
        trajectory_length = traj.length
        if trajectory_length < params.msd_num_points + 1:
            continue
        MSD = np.zeros(trajectory_length - 1)  # mean squared displacement
        n = np.zeros(
            trajectory_length - 1
        )  # used to measure number of trajectories of given length for weighting
        tau = np.zeros(trajectory_length - 1)  # times between MSDs
        track_lengths = np.zeros(trajectory_length - 1)
        x = np.array(traj.path)[:, 0] * params.pixel_size
        y = np.array(traj.path)[:, 1] * params.pixel_size
        for i in range(1, trajectory_length):
            sqd = (x[i:] - x[: trajectory_length - i]) ** 2 + (
                y[i:] - y[: trajectory_length - i]
            ) ** 2
            MSD[i - 1] = np.mean(sqd)
            track_lengths[i - 1] = len(sqd)
            tau[i - 1] = i * params.frame_time
        tau = tau[: params.msd_num_points]
        MSD = MSD[: params.msd_num_points]
        track_lengths = track_lengths[: params.msd_num_points]
        weights = track_lengths[: params.msd_num_points].astype("float32") / float(
            np.amax(track_lengths[: params.msd_num_points])
        )
        # Add in the 0 point of 2*sigma^2 at 0 tau.
        tau = np.concatenate((np.array([0]), tau))
        MSD = np.concatenate((np.array([2*0.04**2]), MSD))
        weights = np.concatenate((np.array([weights[0]]), weights))
        plt.scatter(tau, MSD)
        plt.xlabel(r"$\tau$")
        plt.ylabel("MSD ($\mu$m$^2$)")
        try:
            popt, pcov = curve_fit(straightline, tau, MSD, p0=[1, 0], sigma=weights)
            # if popt[0] > 0:
            diffusion_coefs.append(popt[0] / 4.0)
            ids.append(traj.id)
            #plt.plot(tau, straightline(tau, popt[0], popt[1])) #Uncomment if you want individual plots
            if popt[1] > 0:
                loc_precisions.append(np.sqrt(popt[1]) / 4.0)
        except:
            print("WARNING: Unable to fit curve")
        ofile = params.name+f"_trajectory{traj.id}_MSD_fit.png"
        #plt.savefig(ofile, dpi=300) #Uncomment if you want individual plots
        plt.close()
    plt.hist(diffusion_coefs)
    plt.xlabel("Diffusion coefficient ($\mu$m$^{2}$s$^{-1}$)")
    plt.ylabel("Number of foci trajectories")
    if channel=="L":
        plt.title("Left channel diffusion coefficients\nMean = %3.2f"%(np.mean(diffusion_coefs)))
        ofile = params.name+"_Lchannel_diff_coeff.png"
    elif channel=="R":
        plt.title("Right channel diffusion coefficients\nMean = %3.2f"%(np.mean(diffusion_coefs)))
        ofile = params.name+"_Rchannel_diff_coeff.png"
    else:
        ofile = params.name+"_diff_coeff.png"
    plt.savefig(ofile, dpi=300)
    if params.display_figures:
        plt.show(block=True)
    plt.close()
    if channel=="L":
        f = open(params.name + "_Lchannel_diff_coeff_data.tsv", "w")
    if channel=="R":
        f = open(params.name + "_Rchannel_diff_coeff_data.tsv", "w")
    else:
        f = open(params.name + "_diff_coeff_data.tsv", "w")
    f.write("trajectory\tdiffusion coefficient\n")
    for i in range(len(diffusion_coefs)):
        f.write(str(ids[i]) + "\t" + str(float(diffusion_coefs[i]))+"\n")
    f.close()
    f = open(params.name + "_diff_coeff_loc_precision_data.tsv", "w")
    for i in range(len(loc_precisions)):
        f.write(str(ids[i]) + "\t" + str(float(loc_precisions[i]))+"\n")
    f.close()
    return diffusion_coefs, loc_precisions


def plot_traj_intensities(params, trajs, channel=None, chung_kennedy=True):
    if params.chung_kennedy: ck_data = []
    for traj in trajs:
        t = np.array(traj.intensity)
        plt.plot(t/10**3)
        if params.chung_kennedy: ck_data.append(chung_kennedy_filter(t,params.chung_kennedy_window,1)[0][:-1])
    if params.chung_kennedy:
        if clickmode:
            ofile = params.name+"_click_mode_chung_kennedy_data.csv"
        else:
            ofile = params.name+"_chung_kennedy_data.csv"
        f = open(ofile, 'w')
        ck_data = np.array(ck_data, dtype=object)
        for ck in range(len(ck_data)): 
            f.write(str(ck_data[ck][0]))
            for j in range(len(ck_data[ck])): 
                f.write(","+str(ck_data[ck][j]))
            f.write("\n")
        f.close()
    plt.xlabel("Frame number")
    plt.ylabel("Intensity (camera counts per pixel x$10^3$)")
    if channel=="L":
        plt.title("Left channel trajectory intensity")
        ofile = params.name+"_Lchannel_trajectory_intensities.png"
    elif channel=="R":
        plt.title("Right channel trajectory intensity")
        ofile = params.name+"_Rchannel_trajectory_intensities.png"
    else:
        plt.title("Whole frame trajectory intensity")
        ofile = params.name+"_trajectory_intensities.png"
    plt.savefig(ofile, dpi=300)

    if params.display_figures:
        plt.show(block=True)
    plt.close()
    if params.chung_kennedy:
        for ck in ck_data:
            plt.plot(ck)
        plt.xlabel("Frame number")
        plt.ylabel("Intensity (camera counts per pixel x$10^3$)")
        if channel=="L":
            plt.title("Left channel Chung-Kennedy intensity")
            ofile = params.name+"_Lchannel_CK_filtered_intensities.png"
        elif channel=="R":
            plt.title("Right channel Chung-Kennedy intensity")
            ofile = params.name+"_Rchannel_CK_filtered_intensities.png"
        else:
            plt.title("Whole frame Chung-Kennedy intensity")
            ofile = params.name+"_CK_filtered_intensities.png"
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,3))
        plt.savefig(ofile, dpi=300)
        if params.display_figures:
            plt.show(block=True)
        plt.close()

def get_stoichiometries(trajs, isingle, params, channel=None):
    # Let's do the easy part first - the ones where they do not start at the start
    stoics = []
    ids = []
    startframe = 100000
    for traj in trajs:
        if traj.start_frame<startframe and traj.length>=params.num_stoic_frames: startframe=traj.start_frame
    for traj in trajs:
        if traj.length < params.num_stoic_frames:
            continue
        if traj.start_frame-startframe > 1+params.stoic_trajectory_start_within_n_frames:
            continue #stoics.append(traj.intensity[0] / isingle)
        if params.stoic_method == "Initial":
            # Initial intensity
            traj.stoichiometry = traj.intensity[0] / isingle
#            traj.stoichiometry = traj.stoichiometry[0]
        elif params.stoic_method == "Mean":
            # Mean of first N frames
            traj.stoichiometry = (
                np.mean(traj.intensity[: params.num_stoic_frames]) / isingle
                )
        elif params.stoic_method == "Max":
            # Mean of first N frames
            traj.stoichiometry = (
                np.amax(traj.intensity[: params.num_stoic_frames]) / isingle
                )
        elif params.stoic_method == "Linear":
            xdata = (
                np.arange(0, params.num_stoic_frames , dtype="float")
                # * params.frameTime
            )            
            ydata = traj.intensity[0: params.num_stoic_frames]
            popt, pcov = curve_fit(straightline, xdata, ydata)
            intercept = popt[1]
            if intercept > 0 and popt[0]<0 and startframe!=100000:
                traj.stoichiometry = (intercept + abs((traj.start_frame-startframe)*popt[0])) / isingle
                if traj.stoichiometry.ndim!=0: 
                    traj.stoichiometry = traj.stoichiometry[0]
            else:
                continue 
        else:
            print("WARNING: Unknown stoic_method. Skipping stoichiometry estimation.")
            continue
        stoics.append(traj.stoichiometry)
        ids.append(traj.id)
    stoics = np.array(stoics)
    ids = np.array(ids)

    if channel=="L":
        plt.title("Left channel stoichiometry")
        oseed = params.name+"_Lchannel_stoichiometry"
    elif channel=="R":
        plt.title("Right channel stoichiometry")
        oseed = params.name+"_Rchannel_stoichiometry"
    else:
        plt.title("Whole frame stoichiometry")
        oseed = params.name+"_stoichiometry"

    f = open(oseed + "_data.tsv", "w")
    f.write("trajectory\tstoichiometry\n")
    for i in range(len(stoics)):
        f.write(str(ids[i]) + "\t" + str(float(stoics[i]))+"\n")
    f.close()

    if stoics.size<=1:
        print("Not enough stoic data to do a KDE/further plotting")
        return stoics
    max_stoic = int(np.round(np.amax(stoics)))

    bandwidth = 0.7
    ids = ids[np.isfinite(stoics)]
    stoics = stoics[np.isfinite(stoics)]
    if stoics.shape[0]<3:
        print("WARNING: Empty stoichiometry array for KDE plot. Ignoring stoichiometries")
        return 0
    kde = gaussian_kde(stoics, bw_method=bandwidth)
    x = np.linspace(0, max_stoic, max_stoic)
    pdf = kde.evaluate(x)
    
    fig, ax1 = plt.subplots()
    l1 = ax1.hist(
        stoics,
        bins=np.arange(0, np.amax(np.round(stoics)+1), 1),
        color="gray"
    )
    ax2 = ax1.twinx()
    l2 = ax2.plot(x, pdf, "k-", label="Gaussian KDE")
    plt.ylabel("Probability density (a.u.)")
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 2))

    plt.xticks(range(0,max_stoic+1))
    plt.xlabel("Rounded stoichiometry")
    plt.ylabel("N")
    
    plt.savefig(oseed+"_histogram.png", dpi=300)
    if params.display_figures:
        plt.show(block=True)
    plt.close()
    plt.scatter(range(len(stoics)), stoics)
    plt.xlabel("Spot #")
    plt.ylabel("Raw stoichiometry")
    if channel=="L":
        plt.title("Left channel stoichiometry")
        oseed = params.name+"_Lchannel_stoichiometry"
    elif channel=="R":
        plt.title("Right channel stoichiometry")
        oseed = params.name+"_Rchannel_stoichiometry"
    else:
        plt.title("Whole frame stoichiometry")
        oseed = params.name+"_stoichiometry"
    plt.savefig(oseed+"_scatter.png", dpi=300)
    if params.display_figures:
        plt.show(block=True)
    plt.close()
    return 0

def overtrack(params, trajs, channel=None):
    image_data = images.ImageData()
    image_data.read(params.name+".tif", params)
    if channel != None:
        outfile = params.name+"_"+channel+"_overtracked_trajectory_intensities.tsv"
    else:
        outfile = params.name+"_overtracked_trajectory_intensities.tsv"
    f = open(outfile, 'w')
    f.write("Trajectory ID\tFrame\tIntensity\n")
    for traj in trajs:
        ints = traj.intensity
        final_position = traj.path[-1]
        x = round(final_position[0])
        y = round(final_position[1])
        if channel == "R":
            x += image_data.frame_size[0]//2
        final_frame = traj.end_frame
        if final_frame+11 >= params.num_frames:
            end = params.num_frames
        else:
            end = final_frame+11
        for frame in range(final_frame+1,end):
            image = image_data.pixel_data[frame,:,:]
            # Create a tmp array with the centre of the spot in the centre
            tmp = image[
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
            ints.append(intensity)
        for i in range(len(ints)):
            outstr = str(traj.id) + "\t" + str(traj.start_frame+i) + "\t" + str(ints[i]) + "\n"
            f.write(outstr)
        # Uncomment if necessary
        # plt.plot(range(traj.start_frame, end), ints, lw=1, label="Trajectory "+str(traj.id))
        # plt.title("Overtracked trajectories")
        # plt.xlabel("Frame number")
        # plt.ylabel("Intensity (a.u.)")
        # #plt.legend()
        # plt.savefig(params.name+"_overtracked_trajectory_intensities_plot.png", dpi=300)
    plt.close()
