import os
import sys
import napari
import numpy as np
from tqdm import tqdm
from skimage import morphology
from tifffile import imread, imwrite
from skimage.filters import threshold_otsu
from skimage.morphology import binary_erosion
from skimage.segmentation import clear_border
from skimage.morphology import white_tophat
from voldist_tools.basictools import wipe_layers, remove_large_objects, return_points, find_label_density, get_cube, apply_cube,get_long_axis

'''
This script iterates through a folder expecting each subdirectory to contain a set of images:
1) a cellpose segmentation file (.npy)
2) propidium iodide staining file (.tif)
3) anti-tubulin staining file (.tif)
4) DoG-filtered anti-tubulin staining file (.tif)

It opens the segmentation file and filters it of large and small objects. For each remaining object
(presumably corresponding to a cell), it creates a cube of the same size as the cell. It then applies
the cube to the staining files and saves the resulting images to a new folder. It applies an otsu 
threshold to the DoG filtered staining file, filters it of large and small labels, and saves the 
resulting images to the data cubes folder.

The data cubes folder can then be iterated through to specify which labels correspond to spindles and
which labels are nonspecific. This information will be used to train a random forest classifier so that
we can automate the process for the remaining data cubes.
'''


if __name__ == '__main__':
    viewer = napari.Viewer()

    analysis_dir = '/Volumes/bigData/wholeMount_volDist/220712_Fix_Emb_Flvw_Chn1GAP_PI_aTub647_Processed/N2V_Denoised/0_Analysis_01'
    subdirs = [d for d in os.listdir(analysis_dir) if os.path.isdir(os.path.join(analysis_dir, d)) and 'Exp' in d or 'Cntrl' in d]

    # load shared filters:
    shared_filters_text = np.loadtxt(os.path.join(sys.path[0], 'shared_filters.txt'), delimiter=',', dtype=str).tolist()
    shared_filters = {k: float(v) for k, v in shared_filters_text}

    # create save directory
    data_save_dir = os.path.join(analysis_dir, '0_data_cubes_TopHat-DoG_mask_otsu')
    if not os.path.exists(data_save_dir):
        os.mkdir(data_save_dir)

    for subdir in tqdm(subdirs):

        # make a save directory for this embryo
        emb_save_dir = os.path.join(data_save_dir, subdir)
        if not os.path.exists(emb_save_dir):
            os.makedirs(emb_save_dir)

        # define file paths and load data
        emb_type, emb_num = subdir.split('_')
        segmentations_name = f'220712_Fix_Emb_Flvw_Chn1GAP_PI_aTub647_{emb_type}_{emb_num}-Z01_PI_16bit_scaleZ_sbdl2_16bit_seg.npy'
        dog_tub_name = f'220712_Fix_Emb_Flvw_Chn1GAP_PI_aTub647_{emb_type}_{emb_num}-Z01_Tub_16bit_scaleZ_sbdl2_16bit.tif'
        raw_tub_name = f'220712_Fix_Emb_Flvw_Chn1GAP_PI_aTub647_{emb_type}_{emb_num}-Z01_Tub_16bit_scaleZ.tif'
        pi_name = f'220712_Fix_Emb_Flvw_Chn1GAP_PI_aTub647_{emb_type}_{emb_num}-Z01_PI_16bit_scaleZ_sbdl2_16bit.tif'

        data_load_dir = os.path.join(analysis_dir, subdir)
        masks = np.load(os.path.join(data_load_dir, segmentations_name), allow_pickle=True).item()['masks']
        tub_dog = imread(os.path.join(data_load_dir, dog_tub_name))
        tub_raw = imread(os.path.join(data_load_dir, raw_tub_name))
        pi = imread(os.path.join(data_load_dir, pi_name))
        tub_dog_th = white_tophat(tub_dog, footprint=morphology.ball(5))

        # coarsly filter the masks of poor segmentations
        minimum_size = shared_filters['minimum_size']
        maximum_size = shared_filters['maximum_size']

        filtered_masks = clear_border(masks)
        filtered_masks = morphology.remove_small_objects(filtered_masks, min_size=minimum_size, connectivity=1)
        filtered_masks = remove_large_objects(filtered_masks, max_size=maximum_size)
        remaining_labels = [label for label in np.unique(filtered_masks) if label != 0]

        # Establish the remaining labels for the embryo and loop through each cell
        final_labels = [label for label in np.unique(filtered_masks) if label != 0]
        for curr_mask_id in final_labels:
            wipe_layers(viewer)

            # isolate the current mask as bolean array
            curr_mask = filtered_masks == curr_mask_id

            # get the coordinate of the bounding cube for the current mask ID. Apply it to the labels and images
            cube_dims = get_cube(filtered_masks, curr_mask_id)
            cubed_label = apply_cube(curr_mask, cube_dims)
            cubed_tub_dog_th = apply_cube(tub_dog_th, cube_dims)
            cubed_tub_raw = apply_cube(tub_raw, cube_dims)
            cubed_PI = apply_cube(pi, cube_dims)

            # get the mask coordinates, centroid, and long axis
            mask_coords = np.column_stack(np.where(cubed_label == True))
            cell_centroid = mask_coords.mean(axis=0)
            cell_long_vect, cell_long_line = get_long_axis(cubed_label)

            # get the tubulin signal from the remaining region and define an Otsu threshold
            remaining_tub = np.zeros(shape=cubed_label.shape)
            remaining_tub[cubed_label] = cubed_tub_dog_th[cubed_label]
            remaining_vals = cubed_tub_dog_th[cubed_label].ravel()
            thresh_val = threshold_otsu(remaining_vals)
            thresh_mask = morphology.label(remaining_tub > thresh_val)

            # get number of remaining labels
            num_tub_labels_b4_filter = [label for label in np.unique(thresh_mask) if label != 0]

            if len(num_tub_labels_b4_filter) == 0:
                continue

            # filter and labels smaller than the mimum and maximum expected label sizes
            min_thrsh_size = shared_filters['min_thrsh_size']
            max_thrsh_size = shared_filters['max_thrsh_size']
            '''
            if len(num_tub_labels_b4_filter) > 1:
                thresh_mask = morphology.remove_small_objects(thresh_mask, min_size=min_thrsh_size, connectivity=1)
                thresh_mask = remove_large_objects(thresh_mask, max_size=max_thrsh_size)
            '''
            # get the number of labels after filtering
            remaining_labels = [label for label in np.unique(thresh_mask) if label != 0]

            if len(remaining_labels) == 0:
                continue

            # define mask save directory
            mask_save_dir = os.path.join(emb_save_dir, f'cell_{curr_mask_id}')
            if not os.path.exists(mask_save_dir):
                os.mkdir(mask_save_dir)

            # populate the viewer 
            viewer.add_labels(cubed_label, name='curr_mask_cube', blending='additive')
            viewer.add_image(cubed_tub_dog_th, name='cubed_tub_dog_th', blending='additive', visible=False)
            viewer.add_image(cubed_tub_raw, name='curr_tub_raw_cube', blending='additive', visible=False)
            viewer.add_image(cubed_PI, name='curr_PI_cube', blending='additive', visible=False)
            viewer.add_labels(thresh_mask, name='thresh_mask', blending='additive')

            images_and_layers = ['curr_mask_cube',
                                'cubed_tub_dog_th',
                                'curr_tub_raw_cube',
                                'curr_PI_cube',
                                'thresh_mask']

            # save the tif compatible layers as tifs
            for item in images_and_layers:
                viewer.layers[item].save(os.path.join(mask_save_dir, item + '.tif'))
            
        print(f'finished with embryo {subdir}')
    
    napari.run()