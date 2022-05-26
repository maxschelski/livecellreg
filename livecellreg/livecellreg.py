                                                                                                                                                                                                                                                  # -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 09:22:35 2019

@author: schelskim
"""

import os
import tifffile
from skimage import io, feature
from scipy.signal import fftconvolve
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
from PIL.TiffTags import TAGS as tiff_tags_dict
from collections import OrderedDict
from skimage.filters import median
from skimage.morphology import disk
import time
import tkinter
from tkinter import filedialog
from PIL import Image, ImageSequence
import copy
import re
from scipy.ndimage import median_filter
import pandas as pd
import itertools
import math

#sccript registered two channels based on reference channel (folder name of channel)

#how to implement vectorization?
#do alignment for all images wth the same reference image together.
#create one more dimension in the vector for that
#then create one more dimension each shifted image
#in multi dimensional alignment value matrix then exclude ids where best alignment is not at the border
#continue with remaining array until no id is left


#is it possible to create an ML algorithm that detects neurites
#by using the algorithm I coded for automated analysis?
#Since it might average out random mistakes. Hm...



input_path = "E:\\TUBB\\MT-RF_manipulation\\FRB-Dync1h1m-C2-TE-CAMS3\\"
input_path = "E:\\TUBB\\MT-RF_manipulation\\LatA10-FRB-Dync1h1m-C2-TE\\"
input_path = "C:\\Users\\Maxsc\\Documents\\data_tmp\\FRB-Dync1h1m-C2-TE-CAMS3\\"

input_path = "E:\\TUBB\\MT-RF_manipulation\\KIF-C2-CAMS3\\FRB-KIF13m-C2-CAMS3\\"
input_path = "E:\\TUBB\\MT-RF_manipulation\\KIF-C2-CAMS3\\FRB-KIF13aM-C2-Control-CAMS3\\"
input_path = "E:\\TUBB\\MT-RF_manipulation\\KIF-C2-CAMS3\\FRB-KIF1aM-C2-Control-CAMS3\\"
input_path = "E:\\TUBB\\MT-RF_manipulation\\KIF-C2-CAMS3\\FRB-KIF1aM-C2-CAMS3\\"
input_path = "E:\\TUBB\\MT-RF_manipulation\\recruit-CAAX-control\\"
input_path = "E:\\TUBB\\MT-RF_manipulation\\FRB-KIFC1-N593K-C2-CAMS3\\"
input_path = "E:\\TUBB\\MT-RF_manipulation\\Dync1h1m-C2-old_neurons\\old-DIV7-FRB-Dync1h1m-C2-TE-CAMS3\\"
input_path = "E:\\TUBB\\MT-RF_manipulation\\Dync1h1m-C2-old_neurons\\old-DIV5-FRB-Dync1h1m-C2-TE-CAMS3\\"
input_path = "E:\\TUBB\\MT-RF_manipulation\\Dync1h1m-C2-old_neurons\\old-DIV4-FRB-Dync1h1m-C2-TE-CAMS3\\"
input_path = "E:\\TUBB\\MT-RF_manipulation\\Dync1h1m-C2-old_neurons\\old-DIV10-FRB-Dync1h1m-C2-TE-CAMS3\\"

# input_path = "C:\\Users\\Maxsc\\Documents\\data_tmp\\slices\\211104\\registration\\test"

# super bright spot: 220501 cell0006
# quite a bit shift: 220315
# lots of shift: 220225 cell0003, cell0001



#Choose the channel with an as much as possible non-changing intracellular localization
#signals that accumulate at a certain place or are excluded from a from a place
#over time are not going to work very well
reference_channel = "c0000"
date = "2204"
images_in_sorted_folder = False
choose_folder_manually = False
force_multipage_save = False
only_register_multipage_tiffs = True
overwrite_registered_images = False

# start the registration from the last frame
# thereby the position of all frames 
# will be similar to the position in the last frame
start_reg_from_last_img = False


# number of STD that the correlation value of an image with the last shift
# needs to be different to be considered an outlier, which leads
# to a larger range of shift values to be tried out and the reference
# image to be updated
# will be increased up to max_shift or until the difference 
# is below threshold
threshold_std_difference = 4

# minimum of correlation value difference from mean in multiples of std
# for some conditions (e.g. live cell imaging in 3D with a bit of drift)
# it's better to choose this smaller than threshold_std_difference
threshold_std_for_new_reference = 3
# alterantively to changing the threshold std for new reference
# defining how often a new reference image is choosen can help
# empirically found to be helpful for 3D stacks with more shift
nb_images_until_new_reference = np.nan

# max std difference allowed to not consider the image as outlier
outlier_std_difference = 10

# the maximum multiple of stds above the mean of an image that is allowed
# above that it will be cut to that multiple
max_multiple_stds_above = 10

# refining shift value
# maximum number of testing new shift values without improvement
explorations_without_improvement_outlier = 100#20
explorations_without_improvement = 10


# TODO FOR SPEEDING UP SCRIPT:
# - explore shift values for reining shift values differently; start with large
#   steps and once they don't lead to improvement anymore, try back and forth
#   from new position to find direction it goes in, once it cannot be improved
#   anymore, reduce step size until step size arrived at 1 and 
#   cannot be improved anymore

remove_zero_images = True


# step_size_shift needs to be at least 1
step_size_shift = 2


# if correlation value cannot be brought below the threshold_std_difference,
# this is an alternative rule to end initial optimization
# number of times that the same shift value needs to be found 
# to get the initial shift value, to finish this part
necessary_shift_count = 5

# maximum shift in pixels which is allowed to get initial shift 
# beyond this shift, the image will be replaced with a zero image
# usually before max shift is triggered, enough shifts in the same direction
# were found 
max_shift = 150#20
# if no translation could be found within the allowed max shift
# this happens e.g. if the position scan at the
# microscope ended up at the wrong cell), replace images with zero image
# if False, the untranslated image will be kept
replace_not_translated_images_with_zeros = True

# parameters for removing out of focus images
# this feature might not always work properly
# will output csv file with all frames that were removed
# check that file for some movies to make sure the feature works reliably
remove_out_of_focus_images = False
only_delete_out_of_focus_at_end = False
nb_empty_frames_instead_of_out_of_focus_frames = 0

processes = 1
process = 1
background_val = 200


folders = ["date"]
conditions = [""]
# define strings that need to be present in file name, one needs to be present
# for the file to be used
file_name_inclusions = [""]
# define strings that must not be present in file name, one of them present
# will exclude the file
file_name_exclusions = ["Cer", "wholecell"]


 
def get_normalized_image(image, image_mean):
    image_normalized = image - image_mean
    image_normalized = np.array(image_normalized.astype(int))
    return image_normalized


def get_image_properties(file_path):
    """
    Extracts order of dimensions of imageJ image and image width and height
    """
    with Image.open(file_path) as img:
        dimensions = ["image", "frames", "slices", "channels"]

        #get key that is used for imagedescription in ".tag" dict
        tiff_tags_inv_dict = {v:k for k,v in tiff_tags_dict.items()}
        tiff_tag = tiff_tags_inv_dict["ImageDescription"]
        #use create ordered dict of imagedescription
        #order in dict determines which dimension in the array is used
        #counting starts from right
        meta_data_str = img.tag[tiff_tag][0]
        meta_data = OrderedDict()
        data_values = meta_data_str.split("\n")
        data_dict = OrderedDict()
        for value in data_values:
            value_split = value.split("=")
            if len(value_split) == 2:
                if value_split[0] in dimensions:
                    data_dict[value_split[0]] = value_split[1]
            if len(value_split) > 1:
                meta_data[value_split[0]] = value_split[1]
        img_width = np.array(img).shape[1]
        img_height = np.array(img).shape[0]
        data_dict = OrderedDict(reversed(list(data_dict.items())))
    return data_dict, img_width, img_height, meta_data


def move_xy_axes_in_img_to_last_dimensions(img, img_width, img_height):
    """
    Move x and y axes of all images in stack (img) to last position in dimension list
    :param img: multi-dimensional numpy array
    :param img_width: width (length of x axis) of one image in img (stack)
    :param img_height: height (length of y axis) of one image in img (stack)
    """
    img_axes = img.shape
    #if there are more than 2 (x,y) axes
    if len(img_axes) > 2:
        #check which axes are x and y and put these axes last
        xy_axes = []
        for ax_nb, axis in enumerate(img_axes):
            if (axis == img_width) | (axis == img_height):
                xy_axes.append(ax_nb)
        for xy_nb, xy_axis in enumerate( reversed(xy_axes) ):
            img = np.moveaxis(img, xy_axis, - xy_nb - 1 )
    return img


def get_images_from_folder(folder):
    """
    Get all images in one folder.
    Each image should only contain images of the same cell and channel
    Save each plane from a multipage image (with different timepoints) in
    array
    """
    input_image_array = []
    input_image_name_array = []
    image_shape = []
    image_shapeSet = False
    image_names = os.listdir(folder)
    multi_page = False
    for input_imageName in image_names:
        if input_imageName.find(".tif") == -1:
            continue
        #go through each file and if it is a multi page
        #than go through each plane and add it separately
        input_image_name_array.append(input_imageName)
        input_image_path = os.path.join(folder, input_imageName)
        input_image_array = io.imread(input_image_path)
        single_plane = np.array(input_image_array[0])
        image_shape = single_plane.shape
    #input_image_array replaced by input_image
    return input_image_array, input_image_name_array, image_shape, multi_page 


def extract_imgs_from_multipage(path, image_name):
    """
    Extract all images from multipage ImageJ image.
    Allow image to contain several channels as well.
    For each channel add array of all images, all names, image_shape and the channel number.
    """
    image_path = os.path.join(path,image_name)

    input_image = io.imread(image_path)
    #name of channel property in data_dict, according to imageJ
    channel_prop = "channels"

    (data_dict, img_width, 
     img_height, meta_data) = get_image_properties(image_path)
    
    input_image = move_xy_axes_in_img_to_last_dimensions(input_image, 
                                                         img_width, 
                                                         img_height)
    # expand dimensions to always have z dimension included as well
    if ("slices" not in data_dict.keys()):
        input_image = np.expand_dims(input_image, -3)
    
    if channel_prop not in data_dict.keys():
        channel = 0
        # if there is only one channel, no channel attribute is in data_dict
        # just export same data but with one channel only
        image_name_channel = image_name.replace(".tif",
                                                "c"+str(channel)+".tif")
        image_shapes = [[img_height, img_width]]
        image_name_arrays = [[image_name_channel]]
        image_arrays = [input_image]
        channels = ["c{0:0=4d}".format(channel)]
        return (image_arrays, image_name_arrays, image_shapes, 
                channels, meta_data)

    #get dimension number of channel
    #convert ordered dict keys object to list to access index attribute
    channel_dim = list(data_dict.keys()).index(channel_prop)

    #create slice object to reference to entire array of only one channel
    slices = [slice(None)] * (len(data_dict) + 2)


    image_arrays = []
    image_name_arrays = []
    image_shapes= []
    channels= []

    #iterate through each channel
    for channel in range(int(data_dict[channel_prop])):

        #create slice object to get all images of one channel
        slices_channel = copy.copy(slices)
        slices_channel[channel_dim] = channel
        image_array = input_image[slices_channel]
        image_name_channel = image_name.replace(".tif",
                                                "c"+str(channel)+".tif")

        image_arrays.append(image_array)
        image_name_arrays.append([image_name_channel])
        image_shapes.append([img_height, img_width])
        channels.append("c{0:0=4d}".format(channel))

    return image_arrays, image_name_arrays, image_shapes, channels, meta_data


def delete_out_of_focus_images(image_arrays, channel, cell, background_val, 
                               deleted_frames_cols, output_folder, 
                               nb_empty_frames_instead_of_out_of_focus_frames):
    #remove frames out of focus in both channels
    #use reference_channel since it has a stronger, more consistent signal
    image = image_arrays[channel]
    #create mask of all px in 4D array above background
    image_mask = image > background_val
    #create zero matrix with 1s only at points above background
    image_thresh = np.zeros_like(image)
    image_thresh[image_mask] = 1
    #calculate the number of px above background for each timeframe
    nb_px_above_background = np.sum(image_thresh, axis=(1,2))
    #set zero values as 1 to prevent division by zero
    nb_px_above_background[nb_px_above_background == 0] = 1
    #calculate the relative change in the number of px above background
    change_px_above_background = (nb_px_above_background[1:] / 
                                  nb_px_above_background[:-1])
    #get positions at which focus was probably lost and where it was gained again
    loss_of_focus = np.where(change_px_above_background < 0.2)[0]
    gain_of_focus = np.where(change_px_above_background > 5)[0]
    #get all ranges where focus is lost
    lost_focus_ranges = []
    position = 0
    while True:
        starts_of_ranges = loss_of_focus[loss_of_focus > position]
        if len(starts_of_ranges) == 0:
            break
        #set first value in startof range (smallest)
        #as start of range for this iteration
        start_of_range = starts_of_ranges[0]
        ends_of_ranges = gain_of_focus[gain_of_focus > start_of_range]
        if len(ends_of_ranges) == 0:
            #if there is no gain of focus after its lost,
            #then focus is lost till the end of the timeseries
            lost_focus_ranges.append([start_of_range+1, 
                                      len(nb_px_above_background)])
            break
        for end_of_range in ends_of_ranges:
            #check if there is another loss of focus before the next gain of focus
            all_losses_of_focus_between = loss_of_focus[(loss_of_focus < 
                                                         ends_of_ranges[0]) & 
                                                        (loss_of_focus > 
                                                         start_of_range)]
            if len(loss_of_focus) == 0:
                break
            #if there is, multiply the changes in nb of px of
            #initial loss of focus events and all in between loss of focus events
            complete_loss_of_focus = change_px_above_background[start_of_range]
            for loss_of_focus_between in all_losses_of_focus_between:
                complete_loss_of_focus *= change_px_above_background[loss_of_focus_between]
            #check if the next gain of focus event multiplied by total loss of focus
            #and multiplied by 2 (make up for errors) is more than 1
            complete_gain_of_focus = change_px_above_background[end_of_range]
            if (2 * complete_gain_of_focus * complete_loss_of_focus) >= 1:
                break
            #if not, multiply total loss of focus by gain of focus
            else:
                complete_loss_of_focus *= end_of_range
            #then move to next gain of focus (ends_of_ranges) event and repeat

        position = end_of_range
        lost_focus_ranges.append([start_of_range+1, end_of_range+1])
    #reverse lost focus range to start removing images from the back

    lost_focus_ranges.reverse()
    all_deleted_indices = []
    #go through each range of lost focus
    for lost_focus_range in lost_focus_ranges:
        #create a list of all indices with lost focus for current range
        lost_focus_all_ind = list(range(lost_focus_range[0],
                                        lost_focus_range[1]))
        #save all indices of frames that will be deleted
        all_deleted_indices.append(lost_focus_all_ind)

        #extract first X nb of frames in frames to delete
        #X = nb_empty_frames_between_instead_of_out_of_focus_frames
        #max size of empty images has to be the number of out of focus images in current range
        nb_empty_frames_instead_of_out_of_focus_frames = min(nb_empty_frames_instead_of_out_of_focus_frames, 
                                                             len(lost_focus_all_ind))
        slices_to_replace_with_empty_frames = slice(nb_empty_frames_instead_of_out_of_focus_frames)
        indices_to_replace_with_empty_frames = lost_focus_all_ind[slices_to_replace_with_empty_frames]
        lost_focus_all_ind = np.delete(lost_focus_all_ind, 
                                       slices_to_replace_with_empty_frames)

        #delete all indices for both channels
        for nb, image_array in enumerate(image_arrays):
            image_arrays[nb] = np.delete(image_arrays[nb], 
                                        lost_focus_all_ind, axis=0)

            #then replace indices defined above with empty frames
            #(nb_empty_frames_between_instead_of_out_of_focus_frames)
            for index_to_replace_with_empty_frames in indices_to_replace_with_empty_frames:
                image_arrays[nb][index_to_replace_with_empty_frames,:,:] = np.zeros_like(image_arrays[nb][0,:,:])

        #if only out of focus images at end should be delete
        #stop after one iteration of for loop
        #thereby each video will only go until the last in focus image
        if only_delete_out_of_focus_at_end:
            break
    #flatten list with list of all indices that were deleted
    all_deleted_indices = list(itertools.chain(*all_deleted_indices))

    #save all frames that were deleted in dataframe
    deleted_frames = pd.DataFrame(columns=deleted_frames_cols)
    deleted_frames["frames"] = all_deleted_indices
    deleted_frames["date"] = date
    deleted_frames["cell"] = cell
    csv_file_path = os.path.join(output_folder, 
                                 cell.replace(".tif","") + "_df.csv")
    deleted_frames.to_csv(csv_file_path)
    return image_arrays


def calculate_correlation_value(input_image, x_shift, y_shift, z_shift, 
                                input_image_std, 
                                input_image_mean, reference_image_shifted, 
                                std_reference):
    shifted_image = np.roll(input_image,(z_shift,x_shift,y_shift),
                            axis=(0,2,1))
    shifted_image= cut_image_based_on_z_shift(shifted_image, z_shift)

    shifted_image = get_normalized_image(shifted_image, input_image_mean)
    
    image_product = np.multiply(shifted_image, reference_image_shifted)
    image_product = image_product / (std_reference * input_image_std)
    image_product_sum = np.sum(image_product)
    correlation_value = image_product_sum / (shifted_image.shape[0] * 
                                             shifted_image.shape[1] *
                                             shifted_image.shape[2])
    return correlation_value

def cut_image_based_on_z_shift(image, z_shift):
    if z_shift > 0:
        image = image[z_shift:,:,:]
    if z_shift < 0:
        image = image[:z_shift,:,:]
    return image

def get_translations(input_image_array, step_size_shift, max_shift):
    all_shifts = []
    is_first_nonzero_image = True
    last_shift = {"x":0, "y":0, "z":0}
    """
    #WORK IN PROGRESS TO VECTORIZE THE REGISTRATION
    start = time.time()
    # ref_norm = input_image_array[0] - np.mean(input_image_array[0])
    # std_ref = np.std(ref_norm)

    refs_norm = norm[:-1, :, :]
    stds = np.std(norm, axis=(1,2))
    stds_ref = stds[:-1]
    norm = norm[1:, :, :]
    stds = stds[1:]
    print(time.time() - start)

    start = time.time()
    for x_shift in range(3):
        for y_shift in range(3):
            rolled = np.roll(norm, (x_shift, y_shift), axis=(1,2))
            products = np.multiply(rolled, refs_norm)
            products = products / (stds * stds_ref)[:, None, None]
            sums = np.sum(products, axis=(1,2))
            correlation_values = sums / (input_image_array[0].shape[0] * input_image_array[0].shape[1])
    print(time.time() - start)
    """
    # if shift should be started from the back, reverse the order
    # of images (in the end reverse the shifts to be applicable to a
    # non reversed image array)
    if start_reg_from_last_img:
        input_image_array = np.flip(input_image_array, axis=0)
    else:
        all_shifts.append([0,0,0])

    # prevent too high values in input image array
    # by only allowing a maximum multiple of stds above the mean
    # this is crucial for non cytosolic signals or in general signals that
    # change their localization during the movie
    image_means = np.mean(input_image_array,axis=(2,3), keepdims=True)
    image_stds = np.std(input_image_array, axis=(2,3), keepdims=True)
    new_images = input_image_array - image_means
    new_images = new_images / image_stds
    new_images[new_images > max_multiple_stds_above] = max_multiple_stds_above
    new_images *= image_stds
    new_images += image_means
    input_image_array = new_images

    # save all correlation values with last shift
    correlation_values_last_shift = []
    counter = 0
    for frame_nb, input_image in enumerate(input_image_array):
        print("Finding translation for Frame #",frame_nb)
        # use first nonzero image as first reference image
        if is_first_nonzero_image:
            # only if the image is nonzero, set reference and continue
            # then to next frame
            if len(np.unique(input_image)) > 1:
                is_first_nonzero_image = False
                reference_image = input_image
            elif frame_nb > 0:
                all_shifts.append([0,0,0])
            continue
        
        # do not consider zero-image
        if len(np.unique(input_image)) == 1:
            all_shifts.append([0,0,0])
            continue
        
        input_image_mean = np.mean(input_image)
        std_input_image = np.std(input_image)
        
        (shift,
         shift_changes,
         best_correlation_value,
         correlation_values_last_shift,
         difference_in_stds,
         get_new_reference_image, 
         shifted_references,
         shifted_input_image_stats) = get_initial_shift_for_image(input_image, 
                                                              last_shift, 
                                                              step_size_shift,
                                                              max_shift,
                                                              reference_image, 
                                                              correlation_values_last_shift)
        
        shift = refine_shift_values(input_image, shift_changes,
                                        best_correlation_value, shift,
                                        reference_image, 
                                        difference_in_stds,
                                         shifted_references,
                                         shifted_input_image_stats)
        
            
        all_shifts.append([shift["x"], shift["y"], shift["z"]])
        #update last shift variables
        #will be used as starting point for shifting the next frame
        last_shift = shift
        
        if not np.isnan(nb_images_until_new_reference):
            if counter >= nb_images_until_new_reference:
                get_new_reference_image = True
        
        #set new reference image
        if get_new_reference_image:
            counter = 1
            reference_image = np.roll(input_image,(shift["z"], 
                                                   shift["x"],
                                                   shift["y"]),
                                      axis=(0,2,1))
            # reference_image = get_normalized_image(reference_image, 
            #                                     input_image_mean)
            std_reference = std_input_image
            # reset shifted reference images
            shifted_references = {}

        else:
            counter += 1
            
    # since applying shifts to images always starts with the first image
    # reverse the order of all shifts if registration was started at 
    # last image
    if start_reg_from_last_img:
        all_shifts.reverse()
        all_shifts.append([0,0,0])

    return all_shifts

def get_initial_shift_for_image(input_image, last_shift,
                                step_size_shift,max_shift,
                                reference_image, correlation_values_last_shift):
    
    
    # POTENTIAL ISSUES FOR LARGE SHIFTS:
    # shifting too far could cause problems due to rolling of image 
    # (stuff from one side would appear on other side)
    # also big shifts from one to the next frame would cause 
    # a linear slow down of the algorithm

    
    # also use new reference image if that threshold was reached
    # since this could also indicate that other changes in the
    # cells happened (e.g. if the shift was due to restart of imaging
    # after a few hours without imaging)
    
    shifted_references = {}
    shifted_input_image_stats = {}
    
    shifted_references = add_z_shifted_reference_images(reference_image, 
                                                            shifted_references, 
                                                            [last_shift["z"]])
    shifted_input_image_stats = add_z_shifted_input_image_stats(input_image, 
                                                                shifted_input_image_stats, 
                                                                [last_shift["z"]])
    
    shifted_reference = shifted_references[last_shift["z"]]["image"]
    std_reference = shifted_references[last_shift["z"]]["std"]
    input_image_mean = shifted_input_image_stats[last_shift["z"]]["mean"]
    std_input_image = shifted_input_image_stats[last_shift["z"]]["std"]
    correlation_value_last_shift = calculate_correlation_value(input_image, 
                                                           last_shift["x"], 
                                                           last_shift["y"], 
                                                           last_shift["z"], 
                                                           std_input_image, 
                                                           input_image_mean, 
                                                           shifted_reference, 
                                                           std_reference)
    
    std_correlation_values = np.std(correlation_values_last_shift)
    mean_correlation_values = np.mean(correlation_values_last_shift)
    
    difference_in_stds = get_std_diff_of_correlation(correlation_value_last_shift,
                                                     mean_correlation_values,
                                                     std_correlation_values)
    
    step_size_shift_tmp = {"x":step_size_shift,
                           "y":step_size_shift,
                           "z": step_size_shift}
    # make sure that step size in z is always smaller than the dimension size
    step_size_shift_tmp["z"] = get_max_step_size_shift_for_z(step_size_shift_tmp,
                                                             input_image)
    

    correlation_value_array = get_correlation_value_array(step_size_shift_tmp,  
                                                          input_image)
    

    if ((difference_in_stds > threshold_std_for_new_reference) & 
            (len(correlation_values_last_shift) > 2)):
            get_new_reference_image = True
    
    # save initial correlation values (at last shift)
    # and if correlation value gets much lower than it got before 
    # (> 5STD away from mean), this indicates that the 
    # underlying shift might be much larger; 
    # therefore increase the step soze incremently in ALL directions
    # until no improvement can be made anymore for 3 increases
    # !this will not catch large shifts in the second or third timeframe!
    # (since not enough correlation values present to calculate STD)
    good_correlation_found = False
    get_new_reference_image = False
    all_shift_changes = {}
    # initialize as zero shift is present 0 times
    # to prevent empty dicts, since zero shifts won't be counted later
    all_shift_changes["x"] = {0:0}
    all_shift_changes["y"] = {0:0}
    all_shift_changes["z"] = {0:0}
    shift = {}
    while not good_correlation_found:
        # at the beginning of each loop check whether the difference in stds
        # from the mean is smaller than the defined threshold
        if ((difference_in_stds > threshold_std_difference) & 
            (len(correlation_values_last_shift) > 2)):
            for dimension in step_size_shift_tmp:
                step_size_shift_tmp[dimension] += 1
            # make sure that step size in z 
            # is always smaller than the dimension size
            step_size_shift_tmp["z"] = get_max_step_size_shift_for_z(step_size_shift_tmp,
                                                                     input_image)
        else:
            good_correlation_found = True
        
        # for each iteration calculate a new correlation value matrix
        # since the step_size_shift is expanded for each iteration
        # the array gets bigger as well
        correlation_value_array_new = get_correlation_value_array(step_size_shift_tmp,  
                                                                  input_image)
        
        # incorporate previous (smaller) array into new (larger) array
        correlation_value_array = put_smaller_in_mid_of_larger_array(correlation_value_array,
                                                                     correlation_value_array_new)
        
        all_shifts = get_shifts_from_empty_array_positions(correlation_value_array,
                                                           last_shift["x"],
                                                           last_shift["y"],
                                                           last_shift["z"],
                                                           step_size_shift_tmp)
        all_z_shifts = [shifts[2] for shifts in all_shifts]

        shifted_references = add_z_shifted_reference_images(reference_image, 
                                                                shifted_references, 
                                                                all_z_shifts)
        shifted_input_image_stats = add_z_shifted_input_image_stats(input_image, 
                                                                    shifted_input_image_stats, 
                                                                    all_z_shifts)
        # try shifts lower and higher than the last shift in both directions
        for x_shift, y_shift, z_shift in all_shifts:
            shift["x"] = x_shift
            shift["y"] = y_shift
            shift["z"] = z_shift
            # do not try z shifts that would move out of the dimension
            if z_shift >= input_image.shape[0]:
                continue
            shifted_reference = shifted_references[z_shift]["image"]
            std_reference = shifted_references[z_shift]["std"]
            input_image_mean = shifted_input_image_stats[z_shift]["mean"]
            std_input_image = shifted_input_image_stats[z_shift]["std"]
            correlation_value = calculate_correlation_value(input_image, 
                                                           x_shift, y_shift, 
                                                           z_shift,
                                                           std_input_image, 
                                                           input_image_mean, 
                                                           shifted_reference, 
                                                           std_reference)
            array_shape = correlation_value_array.shape
            shift_relative_to_last = {}
            # if there was only one value tested, the shift is the same
            # as the last shift
            # add dimensions in order of correlation value array
            dimensions = ["x", "y", "z"]
            for dim_nb, dimension in enumerate(dimensions):
                
                if array_shape[dim_nb] > 1:
                    shift_relative_to_last[dimension] = abs_to_rel_shift(shift[dimension],
                                                                         step_size_shift_tmp[dimension],
                                                                         last_shift[dimension])
                else:
                    shift_relative_to_last[dimension] = last_shift[dimension]
                    
            correlation_value_array[shift_relative_to_last["x"], 
                                    shift_relative_to_last["y"], 
                                    shift_relative_to_last["z"]] = correlation_value
                
        #get the highest (best) correlation value
        best_correlation_value = np.max(correlation_value_array)
        
        difference_in_stds = get_std_diff_of_correlation(best_correlation_value,
                                                         mean_correlation_values,
                                                         std_correlation_values)
        
        best_correlation = np.where(correlation_value_array == 
                                    best_correlation_value)
        
        (shift_changes) = get_initial_shift_refinings(best_correlation, 
                                                       step_size_shift_tmp,
                                                       correlation_value_array.shape)
                                                       
        # increase counter for each shift change over all trials
        for dimension, shift_change in shift_changes.items():
            # do not count zero shift changes
            if shift_change == 0:
                continue
            if shift_change not in all_shift_changes[dimension]:
                all_shift_changes[dimension][shift_change] = 0
            all_shift_changes[dimension][shift_change] += 1
            
        # then get the maximum count of each shift value for all dimension
        max_shift_value_counts = [max(shift_changes.values()) 
                                  if len(shift_changes) > 0
                                  else 0
                                  for shift_changes 
                                  in all_shift_changes.values()
                                  ]
        # get the maximum count across all dimensions
        max_shift_value_count = max(max_shift_value_counts)
        # if one shift was identified often enough, use that shift change
        if max_shift_value_count == necessary_shift_count:
            good_correlation_found = True
            shift_changes = get_max_counted_shift_change(all_shift_changes)
            break
        
        # stop optimization after max_shfit is reached
        max_step_size_shift = max(list(step_size_shift_tmp.values()))
        if (max_step_size_shift*2) > max_shift:
            if difference_in_stds < outlier_std_difference:
                good_correlation_found = True
            else:
                good_correlation_found = False
            break
    
    # if the correlation was too different initially, 
    # do not add the initial correlation value
    # otherwise the correlation will be shifted by outliers
    # for outliers, add the optimized correlation value if a good correlation
    # value was found
    if not get_new_reference_image:
        correlation_values_last_shift.append(correlation_value_last_shift)
    elif good_correlation_found:
        correlation_values_last_shift.append(best_correlation_value)
        
    # TODO:
    # what if there are two shift values with 
    # exactly the same correlation values?
    # does this ever happen?
    # direction to follow up on would be wrong;
    # this would also be a problem for refining the shift values

    # set dimensions in order of shifts
    #calculate corresponding shifts by starting from lastshifts
    for dim_nb, dimension in enumerate(shift.keys()):
        if array_shape[dim_nb] > 1:
            shift[dimension] = rel_to_abs_shift(best_correlation[dim_nb][0],
                                                step_size_shift_tmp[dimension],
                                                last_shift[dimension])
        else:
            shift[dimension] = last_shift[dimension]

    
    return (shift, shift_changes,
            best_correlation_value, correlation_values_last_shift,
            difference_in_stds,
            get_new_reference_image, shifted_references,
            shifted_input_image_stats)

def get_correlation_value_array(step_size_shift_tmp,  input_image):
    image_shape = input_image.shape
    correlation_value_array = np.zeros((step_size_shift_tmp["x"]*2 + 1, 
                                        step_size_shift_tmp["y"]*2 + 1,
                                        step_size_shift_tmp["z"]*2 + 1))
    return correlation_value_array


def get_max_step_size_shift_for_z(step_size_shift_tmp, input_image):
    step_size_shift_z = min(input_image.shape[0] - 2, 
                            step_size_shift_tmp["z"])
    step_size_shift_z = max(step_size_shift_z, 0)
    return step_size_shift_z


def get_std_diff_of_correlation(correlation_value, mean_correlation_values,
                                std_correlation_values):
    difference_in_stds = (abs(correlation_value - mean_correlation_values) /
                          std_correlation_values)
    return difference_in_stds

def put_smaller_in_mid_of_larger_array(smaller_array, larger_array):
        # incorporate the previous array in the middle of the new array
        array_shape = np.array(smaller_array.shape)
        shape_difference = np.array(larger_array.shape) - array_shape
        if (shape_difference[0] == 0) & (shape_difference[1] == 0):
            return smaller_array
        
        x_slice = slice(int(shape_difference[0]/2), 
                        int(shape_difference[0]/2 + array_shape[0]))
        y_slice = slice(int(shape_difference[1]/2), 
                        int(shape_difference[1]/2 + array_shape[1]))
        z_slice = slice(int(shape_difference[2]/2), 
                        int(shape_difference[2]/2 + array_shape[2]))
        larger_array[x_slice, y_slice, z_slice] = smaller_array
        return larger_array

def add_z_shifted_reference_images(reference_image, 
                                   shifted_references, all_z_shifts):
    for z_shift in all_z_shifts:
        if z_shift in shifted_references:
            continue
        shifted_reference = cut_image_based_on_z_shift(reference_image, 
                                                       z_shift)
        image_mean = np.mean(shifted_reference)
        shifted_reference_norm = get_normalized_image(shifted_reference,
                                                      image_mean)
        shifted_references[z_shift] = {}
        shifted_references[z_shift]["image"] = shifted_reference_norm
        shifted_references[z_shift]["std"] = np.std(shifted_reference)
    return shifted_references
    
def add_z_shifted_input_image_stats(input_image, 
                                   shifted_input_image_stats, all_z_shifts):
    for z_shift in all_z_shifts:
        if z_shift in shifted_input_image_stats:
            continue
        # cut the inverse as for shifted reference 
        # since the image will first be rolled and then the 
        # image part roled "over the edge" is removed
        shifted_input_image = cut_image_based_on_z_shift(input_image, 
                                                       -z_shift)
        shifted_input_image_stats[z_shift] = {}
        shifted_input_image_stats[z_shift]["std"] = np.std(shifted_input_image)
        shifted_input_image_stats[z_shift]["mean"] = np.mean(shifted_input_image)
    return shifted_input_image_stats
    
def get_shifts_from_empty_array_positions(array, last_x_shift, last_y_shift, 
                                          last_z_shift, step_size_shift):
    # get values for all shifts where no correlation value is set
    all_shifts = np.vstack(np.where(array == 0))
    
    # to get the absolute shift values, add the last_shift values to them
    if array.shape[0] > 1:
        all_shifts[0] += (last_x_shift - step_size_shift["x"])
    if array.shape[1] > 1:
        all_shifts[1] += (last_y_shift - step_size_shift["y"])
    # make sure that the z shift is not changed if there should 
    # not be additional z shifts
    if array.shape[2] == 1:
        pass
    else:
        all_shifts[2] += (last_z_shift - step_size_shift["z"])
    
    # invert to be able to loop through the shifts
    all_shifts = all_shifts.T
    return all_shifts
    
def abs_to_rel_shift(abs_shift, step_size_shift, last_shift):
    rel_shift = abs_shift - last_shift + step_size_shift 
    return rel_shift

def rel_to_abs_shift(rel_shift, step_size_shift, last_shift):
    abs_shift = rel_shift - step_size_shift + last_shift
    return abs_shift


def get_max_counted_shift_change(all_shift_changes):
    """
    From counts of all shift changes get shift_changes the dimension with
    the highest count having the shift change with the highest count in that
    dimension. For all other dimension set shift change to zero.
    """
    max_shift_counts = {}
    for dimension in all_shift_changes.keys():
        if len(all_shift_changes[dimension]) == 0:
            max_shift_counts[dimension] = {0:0}
            continue
        shift_max_count = max(all_shift_changes[dimension], 
                              key=all_shift_changes[dimension].get)
        count_max_shift = all_shift_changes[dimension][shift_max_count]
        max_shift_counts[dimension] = {shift_max_count:count_max_shift}
    # x in the lambda function is the dimension
    # lambda extracts the maximum value count for that dimension
    max_shift_count_dimension = max(max_shift_counts, 
                                    key= lambda x: list(max_shift_counts[x].values())[0])
    # get dictionary with the shift change (not the count)
    # for dimension where the count was not the maximum count, set
    # the shift change to 0
    shift_changes = {}
    for dimension in all_shift_changes:
        if dimension == max_shift_count_dimension:
            max_shift_count_dim = max_shift_counts[dimension].keys()
            shift_changes[dimension] = list(max_shift_count_dim)[0]
            continue
        shift_changes[dimension] = 0
    return shift_changes

def refine_shift_values(input_image,  shift_changes, best_correlation_value, 
                        shift,  reference_image, 
                        difference_in_stds,
                        shifted_references, shifted_input_image_stats):
    
    refine_shift = {}
    for dimension in shift_changes:
        if shift_changes[dimension] == 0:
            refine_shift[dimension] = False
        else:
            refine_shift[dimension] = True
    
    # reformat shift changes to contain an array for each dimension
    # with shift changes that should be tested
    shift_changes = {dim:[shift] for dim, shift in shift_changes.items()}

    no_improvement_counter = {"x":0, "y":0, "z":0}
    
    # if the current image is an outlier (much worse correlation score)
    # then use a higher distance to explore
    if difference_in_stds > threshold_std_difference:
        max_without_improvement = explorations_without_improvement_outlier
    else:    
        max_without_improvement = explorations_without_improvement
        
    # if there are no multiple slices in the z dimension
    # z dimension does not need to be optimized
    if input_image.shape[0] == 1:
       no_improvement_counter["z"] = max_without_improvement
       
    # dimensions should now be ordered as they are order in the input_image
    dimensions = ["z", "x", "y"]
    a = 0
    while refine_shift["x"] | refine_shift["y"] | refine_shift["z"]:
        #save all x_shifts & y_shifts to be tested
        shifts = {}
        for dim_nb, dimension in enumerate(dimensions):
            shifts[dimension] = []
            # only add initial shift value for not refined dimensions
            if not refine_shift[dimension]:
                shifts[dimension].append(shift[dimension])
                continue
            # refining the dimension with shift change of 0 means
            # that the shift change needs to be tested
            for shift_change in shift_changes[dimension]:
                if shift_change == 0:
                    continue
                new_shift = shift[dimension] + shift_change
                # only add new shift, if it does not move out of the dimension
                if abs(new_shift) >= (input_image.shape[dim_nb] - 2): 
                    continue
                shifts[dimension].append(new_shift)
                
            if len(shifts[dimension]) == 0:
                no_improvement_counter[dimension] = max_without_improvement
                shifts[dimension].append(shift[dimension])
        best_shift = copy.copy(shift)
        
        all_shifts = list(itertools.product(shifts["x"], 
                                            shifts["y"], 
                                            shifts["z"]))

        all_shifted_references = add_z_shifted_reference_images(reference_image, 
                                                                shifted_references, 
                                                                shifts["z"])
        shifted_input_image_stats = add_z_shifted_input_image_stats(input_image, 
                                                                    shifted_input_image_stats, 
                                                                    shifts["z"])
        #test all combinations of x_shifts and y_shifts
        for x_shift_test, y_shift_test, z_shift_test in all_shifts:
            shifted_reference = all_shifted_references[z_shift_test]["image"]
            std_reference = all_shifted_references[z_shift_test]["std"]
            input_image_mean = shifted_input_image_stats[z_shift_test]["mean"]
            std_input_image = shifted_input_image_stats[z_shift_test]["std"]
            
            #don't calculate correlation that was calculated already again
            if ((x_shift_test == shift["x"]) & (y_shift_test == shift["y"]) &
                 (z_shift_test == shift["z"])):
                continue
            correlation_value_test = calculate_correlation_value(input_image, 
                                                                 x_shift_test, 
                                                                 y_shift_test, 
                                                                 z_shift_test,
                                                                 std_input_image, 
                                                                 input_image_mean, 
                                                                 shifted_reference, 
                                                                 std_reference)
            #check if the new correlation_value is larger than the best so far
            if correlation_value_test > best_correlation_value:
                #update best correlation value
                best_correlation_value = correlation_value_test
                #save shifts for best correlation value in tmp vars
                best_shift["x"] = x_shift_test
                best_shift["y"] = y_shift_test
                best_shift["z"] = z_shift_test

        # if best shift were not changed, 
        # then don't refine it further otherwise update the shifts
        # when one dimension is refined, go through the others x -> y -> z
        # always refine only one dimension at a time
        # reiterate until there was no improvement in any dimension
        dimension_order_map = {"x":"y", "y":"z", "z":"x"}
        for dimension in best_shift:
            next_dimension = dimension_order_map[dimension]
            if not refine_shift[dimension]:
                continue
            if best_shift[dimension] == shift[dimension]:
                no_improvement_counter[dimension] += 1
                if no_improvement_counter[dimension] >= max_without_improvement:
                    refine_shift[dimension] = False
                    refine_shift[next_dimension] = True
                    shift_changes[dimension] = [0]
                    shift_changes[next_dimension] = [-1, 1]
                    break
                elif no_improvement_counter[dimension] < max_without_improvement:
                    if len(shift_changes[dimension]) > 1:
                        # if there was no improvement, try larger shift changes
                        # in both directions
                        shift_changes[dimension][0] -= 1
                        shift_changes[dimension][1] += 1
                    elif shift_changes[dimension][0] < 0:
                        # otherwise increase shift changes in the same direction more
                        shift_changes[dimension][0] -= 1
                    elif shift_changes[dimension][0] > 0:
                        shift_changes[dimension][0] += 1
            else:
                # once there is one improvement, reset all counters
                for counter_dimension in no_improvement_counter:
                    no_improvement_counter[counter_dimension] = 0
                # if there are still multiple shift changes that are tested
                # set the direction now that an improvement was evident
                # if the new shift was smaller than the original shift
                # set the shift change to -1, otherwise to 1
                # shift tests are then started from the new shift
                if best_shift[dimension] < shift[dimension]:
                    shift_changes[dimension] = [-1]
                else:
                    shift_changes[dimension] = [+1]
                shift[dimension] = best_shift[dimension]
                
        # stop optimization if no improvements could be made in any dimension
        min_no_improvement = np.min(list(no_improvement_counter.values()))
        if min_no_improvement >= max_without_improvement:
            return shift
        
    return shift

def get_initial_shift_refinings(best_correlation, step_size_shift,
                                correlation_array_shape):
    """
    Check whether shift should be refined in each direction
    if best value was not +1 or -1 px, leave it as is
    if the best value was left in the matrix (lower shift)
    then the shift should be further reduced
    if it was right in the array (higher shift)
    then the shift should be further increased
    """
    shift_changes = {}
    # add dimension in order they are present in the correlation_value_array
    dimensions = ["x", "y", "z"]
    for dim_nb, dimension in enumerate(dimensions):
        # if only one value was tested for the current correlation array
        # frame shift is 0 in any case
        if (correlation_array_shape[dim_nb] == 1):
            shift_changes[dimension] = 0
        elif (best_correlation[dim_nb][0] == 0):
            #reduce x_shift
            shift_changes[dimension] = -1
        elif (best_correlation[dim_nb][0] == (2 * step_size_shift[dimension])):
            #increase x_shift
            shift_changes[dimension] = 1
        else:
            shift_changes[dimension] = 0

    return shift_changes

def translate_images(all_translations,image_array,image_name_array, 
                    outputFolder, channel, save_as_multipage=False):

    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    registered_images_array = []
    for a, image in enumerate(image_array):
        if a >= len(all_translations):
            continue
        x_shift = all_translations[a][0]
        y_shift = all_translations[a][1]
        z_shift = all_translations[a][2]
        # replace images where no translation was found with zero image
        replace_with_zero_image = False
        if np.isnan(x_shift):
            if replace_not_translated_images_with_zeros:
                replace_with_zero_image = True
            # if no translation was found, replace values with 0
            # to allow the translation by 0 (no translation)
            # instead of by nangit add 
            x_shift, y_shift, z_shift = (0,0,0)
                
        if replace_with_zero_image:
            translated_image = np.zeros_like(image)
        else:
            translated_image = np.roll(image,(z_shift,x_shift,y_shift),
                                       axis=(0,2,1))
            if x_shift > 0:
                translated_image[:,:,:x_shift] = 0
            if x_shift < 0:
                translated_image[:,:,x_shift:] = 0
            if y_shift > 0:
                translated_image[:,:y_shift,:] = 0
            if y_shift < 0:
                translated_image[:,y_shift:,:] = 0
            if z_shift > 0:
                translated_image[:z_shift,:,:] = 0
            if z_shift < 0:
                translated_image[z_shift:,:,:] = 0

        #only save single images if they should not be saved as multipage images
        if not save_as_multipage:
            io.imsave(outputFolder+image_name_array[a],translated_image)
        else:
            registered_images_array.append(translated_image)#Image.fromarray(
    #if tiff should be saved as multi page, do so instead of saveing single images
    if save_as_multipage:
        registered_images_array = np.array(registered_images_array)
        return registered_images_array


def check_whether_file_name_is_allowed(file_name, file_name_inclusions,
                                       file_name_exclusions):
    matches_one = False
    for inclusion in file_name_inclusions:
        if file_name.find(inclusion) != -1:
            matches_one = True
            break
    if not matches_one:
        return False
    matches_none = True
    for exclusion in file_name_exclusions:
        if file_name.find(exclusion) != -1:
            matches_none = False
            break
    if matches_none:
        return True
    else:
        return False

def translate_cells_in_folder(exp_iteration_folder, date, step_size_shift,
                              max_shift,
                              current_nb, file_name_inclusions, 
                              file_name_exclusions,
                              force_multipage_save = False) :
    """
    Translate all cells within a folder starting with xy shift values up until step_size_shift.
    Allow cells within folder to be either saved as tiff files in channel folder in cell folder
    or cells saved as a single multi-channel tiff
    """
    for cell in os.listdir(exp_iteration_folder):
        cell_allowed = check_whether_file_name_is_allowed(cell,
                                                          file_name_inclusions,
                                                          file_name_exclusions)
        if not cell_allowed:
            continue
        image_arrays = []
        image_name_arrays = []
        image_shapes = []
        channels = []
        cell_folder = os.path.join(exp_iteration_folder,cell)
        cell_folder_reg = os.path.join(exp_iteration_folder,cell+"_registered")
        #check if registration was done either with or without multipage save
        registration_done_no_multipage = os.path.exists(cell_folder_reg)
        cell_file_name_multipage = cell.replace(".tif","_registered.tif")
        registration_done_multipage = os.path.exists(os.path.join(exp_iteration_folder, 
                                                                  cell_file_name_multipage))
        registration_done = registration_done_no_multipage | registration_done_multipage
        if (((cell.replace("_registered","") == cell) & 
             ~(registration_done) ) | 
            overwrite_registered_images):
            output_folder = os.path.join(exp_iteration_folder,
                                         cell + "_registered")
            if not only_register_multipage_tiffs:
                if os.path.isdir(cell_folder) & (cell.find("cell0") != -1):
                    current_nb += 1
                    print("cell folder:",cell)
                    if current_nb != process:
                        continue
                    for channel in os.listdir(cell_folder):
                        channel_folder = os.path.join(cell_folder,
                                                     channel)
                        if not os.path.isdir(channel_folder):
                            continue
                        (image_array, image_name_array, 
                         image_shape, multi_page) = get_images_from_folder(channel_folder)
                        image_arrays.append(image_array)
                        image_name_arrays.append(image_name_array)
                        image_shapes.append(image_shape)
                        channels.append(channel)

            if (cell.find(".tif") != -1) & (len(image_arrays) == 0):
                output_folder = exp_iteration_folder
                current_nb += 1
                print("tiffile:",cell)
                if (current_nb == process):
                    #if item in exp_iteration_folder is not a folder
                    #and is tiff or tif file, then process as multipage tiff
                    (image_arrays, 
                     image_name_arrays, 
                     image_shapes, 
                     channels, 
                     meta_data) = extract_imgs_from_multipage(exp_iteration_folder, 
                                                              cell)

                    multi_page = True

        if (current_nb % processes) == 0:
            current_nb = 0
        #independent of whether cell data is saved in folder structure
        #or in multi-channel tiff, get translations from reference channel
        #and then translate all images accordingly
        if len(image_arrays) <= 0:
            continue
        if len(image_arrays[0].shape) <= 2:
            continue
        if remove_zero_images:
            zero_frames_channels = []
            #remove zero images
            #by selecting all timeframes comprised of a non zero image
            for nb, channel in enumerate(channels):
                zero_frames_channels.append(np.all(image_arrays[nb] > 0, 
                                                   axis=(-3,-2,-1)))
            # make sure to only have one set of zero frames for all channels
            # otherwise channels will have different number of frames
            # remove all frames with a zero frame in at least one channel
            final_zero_frames = zero_frames_channels[0]
            for zero_frames in zero_frames_channels[1:]:
                final_zero_frames = final_zero_frames & zero_frames
                
            for nb, channel in enumerate(channels):
                image_arrays[nb] = image_arrays[nb][final_zero_frames,:,:,:]
            
        if remove_out_of_focus_images:
            for nb, channel in enumerate(channels):
                if channel != reference_channel:
                    continue
                image_arrays = delete_out_of_focus_images(image_arrays, 
                                                          nb, cell, 
                                                          background_val, 
                                                          deleted_frames_cols, 
                                                          output_folder, 
                                                          nb_empty_frames_instead_of_out_of_focus_frames)

        for nb, channel in enumerate(channels):
            if channel != reference_channel:
                continue
            all_translations = get_translations(image_arrays[nb], 
                                              step_size_shift,
                                              max_shift)


        registered_images_array_all_channels = []
        if multi_page:
            output_folder_channel = exp_iteration_folder

        for nb, channel in enumerate(channels):
                if force_multipage_save:
                    multi_page = True

                if not multi_page:
                    output_folder_channel = os.path.join(output_folder,channel)
                    if not os.path.exists(output_folder_channel):
                        os.makedirs(output_folder_channel)
                registered_images_array = translate_images(all_translations, 
                                                          image_arrays[nb], 
                                                          image_name_arrays[nb], 
                                                          output_folder_channel, 
                                                          channel, 
                                                          save_as_multipage = multi_page)
                if type(registered_images_array) != type(None):
                    registered_images_array_all_channels.append(registered_images_array)

        if len(registered_images_array_all_channels) <= 0:
            return current_nb
        #transform list into numpy array by stacking into new dimension
        registered_images_array_all_channels = np.stack(registered_images_array_all_channels)
        # then add empty axes for Z dimension, if z is not present already
        # at second 
        if len(registered_images_array_all_channels.shape) < 5:
            #axes need to be arranged as TZCYX to be displayed as correct Hyperstack in ImageJ
            #first move axes for time to first position
            registered_images_array_all_channels = np.moveaxis(registered_images_array_all_channels, 
                                                               1,0)
            registered_images_array_all_channels = np.expand_dims(registered_images_array_all_channels, 1)
        else:
            
            #axes need to be arranged as TZCYX to be displayed as correct Hyperstack in ImageJ
            #first move axes for time to first position
            registered_images_array_all_channels = np.moveaxis(registered_images_array_all_channels, 
                                                                1,0)
            registered_images_array_all_channels = np.moveaxis(registered_images_array_all_channels, 
                                                                2,1)
        #add all ImageJ meta_data from the non registered file as well
        image_path = os.path.join(output_folder, 
                                  cell.replace(".tif","_registered.tif"))
        io.imsave(image_path , 
                  registered_images_array_all_channels, 
                  imagej=True,
                  plugin="tifffile", metadata = meta_data)
    return current_nb

def goDeeper(props,this_path, a, max_a, conditions, folders, step_size_shift,
             max_shift,
             current_nb, file_name_inclusions, file_name_exclusions, 
             force_multipag_save):
    a += 1
    for newFolder in os.listdir(this_path):
        if conditions[a-1] != "":
            if newFolder.find(conditions[a-1]) != -1:
                useFolder = True
            else:
                useFolder = False
        else:
            useFolder = True
        if not useFolder:
            continue
        path_new = os.path.join(this_path, newFolder)
        props_new = copy.copy(props)
        props_new[folders[a-1]] = newFolder
        if a == (max_a):
            if os.path.isdir(path_new):
                current_nb = translate_cells_in_folder(path_new, 
                                                       props_new["date"], 
                                                       step_size_shift, 
                                                       max_shift,
                                                       current_nb,
                                                       file_name_inclusions, 
                                                       file_name_exclusions,
                                                       force_multipage_save = force_multipage_save)
        else:
            if os.path.isdir(path_new):
                current_nb = goDeeper(props_new, path_new, a, max_a,
                                      conditions,folders, step_size_shift,
                                      max_shift,
                                     current_nb, 
                                     file_name_inclusions, file_name_exclusions,
                                     force_multipag_save)
                

def traverseFolders(input_path, folders, conditions, step_size_shift,max_shift,
                    current_nb, file_name_inclusions, file_name_exclusions, 
                    force_multipag_save,
                    ):
    props = {}
    nb = 0
    goDeeper(props,input_path,0,len(folders),conditions,folders, 
             step_size_shift,max_shift,
             current_nb, file_name_inclusions, file_name_exclusions, 
             force_multipag_save)
    



current_nb = 0

deleted_frames_cols = ("date", "cell", "frame")

if choose_folder_manually:
    print("choose folder now...")
    root = tkinter.Tk()
    root.withdraw()

    collection_folder = filedialog.askdirectory(initialdir = input_path)
    collection_folder = os.path.abspath(collection_folder)

    if os.path.exists(collection_folder):
        if os.path.isdir(collection_folder):
            collection_folder = collection_folder

            current_nb = 0
            translate_cells_in_folder(collection_folder, "unknown", 
                                   step_size_shift,max_shift, current_nb, 
                                   file_name_inclusions, file_name_exclusions,
                                   force_multipage_save = force_multipage_save)

else:
    traverseFolders(input_path, folders,conditions, step_size_shift,max_shift,
                    current_nb, file_name_inclusions, file_name_exclusions,
                    force_multipage_save)