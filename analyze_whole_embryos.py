import os
import vg
import sys
import joblib
import napari
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import spatial
from numpy import random
from tifffile import imread, imwrite
from skimage.measure import regionprops_table
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage import morphology
from voldist_tools.basictools import remove_large_objects, findVec, wipe_layers
from skimage.morphology import white_tophat

# paths
base_dir = '/Volumes/bigData/wholeMount_volDist/220726-0805_Fix_Emb_Flvw_Chn1GAP_PI_aTub647_Processed/N2V_Denoised/Analysis02' 
subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and 'Cntrl' in d or 'Exp' in d]
embryo_paths = [os.path.join(base_dir, subdir) for subdir in subdirs]

dataframelist = []

for emb_path in embryo_paths:
    embryo_name = emb_path.split('/')[-1]
    print(f'starting {embryo_name}')
    group = embryo_name.split('_')[0]
    emb = embryo_name.split('_')[1]
    embryo_files = [f for f in os.listdir(emb_path) if not f.startswith('.') and f.endswith('.tif') or f.endswith('.npy')]
    mask_name = [f for f in embryo_files if f.endswith('.npy')][0]
    mask_path = os.path.join(emb_path, mask_name)

    # load the masks and find the embryo centroid
    masks = np.load(mask_path, allow_pickle=True).item()['masks']
    masks_fused = masks.astype('bool') 
    masks_fused_coords = np.column_stack(np.where(masks_fused))
    masks_fused_centroid = masks_fused_coords.mean(axis=0)

    # create save directory
    save_dir = os.path.join(base_dir, 'vectors_and_masks')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # load the tubulin and filter with white tophat
    tub_name = [f for f in embryo_files if '_Tub_' in f and '_sbdl2' in f][0]
    tub_path = os.path.join(emb_path, tub_name) 
    tub_raw = imread(tub_path)
    ''' white tophat takes 3 minutes, so I'm going to skip it for now'''
    #tub = tub_raw
    tub = white_tophat(tub_raw, footprint=morphology.ball(5))
    #tub = imread('/Volumes/bigData/wholeMount_volDist/220712_Fix_Emb_Flvw_Chn1GAP_PI_aTub647_Processed/N2V_Denoised/0_Analysis_01/vectors_and_masks/220712_Fix_Emb_Flvw_Chn1GAP_PI_aTub647_Cntrl_E01-Z01_Tub_16bit_scaleZ_sbdl2_16bit.tif_white-tophat.tif')
    imwrite(os.path.join(save_dir, f'{tub_name}_white-tophat.tif'), tub)

    # load shared filters:
    shared_filters_text = np.loadtxt(os.path.join(sys.path[0], 'shared_filters.txt'), delimiter=',', dtype=str).tolist()
    shared_filters = {k: float(v) for k, v in shared_filters_text}

    # filter masks of small and large labels
    minimum_size = shared_filters['minimum_size']
    maximum_size = shared_filters['maximum_size']

    filtered_masks = clear_border(masks)
    filtered_masks = morphology.remove_small_objects(filtered_masks, min_size=minimum_size, connectivity=1)
    filtered_masks = remove_large_objects(filtered_masks, max_size=maximum_size)
    cell_labels = [l for l in np.unique(filtered_masks) if l != 0]

    # load the cell and spindle classifiers
    spindle_classifier_path = '/Users/bementmbp/Desktop/Scripts/volume-distribution/classifiers/spindle_classifier_3classes.joblib'
    spindle_classifier = joblib.load(spindle_classifier_path)
    cell_classifier_path = '/Users/bementmbp/Desktop/Scripts/volume-distribution/classifiers/cell_classifier.joblib'
    cell_classifier = joblib.load(cell_classifier_path)

    for mask_id in tqdm(cell_labels):
        print(f'analyzing mask {mask_id}')
        # boolean mask of cell mask coordinates
        cell_mask = filtered_masks == mask_id
        cell_mask_coords = np.column_stack(np.where(cell_mask))
        cell_mask_centroid = cell_mask_coords.mean(axis=0)

        # zeros array to fill with masked tubulin signal
        tub_mask = np.zeros(tub.shape)

        # fill the tubulin mask with the masked tubulin signal
        tub_mask[cell_mask] = tub[cell_mask]
        tub_mask_vals = tub[cell_mask].ravel()
        thresh_val = threshold_otsu(tub_mask_vals)

        # cc labeling of tub threshold
        labeled_tub_mask = morphology.label(tub_mask > thresh_val)

        # see if it's necessary to filter or not
        num_tub_labels_b4_filter = [l for l in np.unique(labeled_tub_mask) if l != 0]
        if not len(num_tub_labels_b4_filter) > 1:
            print('No labels detected')
            continue
        
        # filter and labels smaller than the mimum and maximum expected label sizes
        min_thrsh_size = 200#shared_filters['min_thrsh_size']
        max_thrsh_size = 5000#shared_filters['max_thrsh_size']
        labeled_tub_mask = morphology.remove_small_objects(labeled_tub_mask, min_size=min_thrsh_size, connectivity=1)
        labeled_tub_mask = remove_large_objects(labeled_tub_mask, max_size=max_thrsh_size)

        # move on if no labels remain
        tub_labels = [l for l in np.unique(labeled_tub_mask) if l != 0]
        if not len(tub_labels) > 1:
            print('small and large filtering removed remaining labels')
            continue

        # calculating remaining label properties
        tub_label_props = regionprops_table(labeled_tub_mask, cache = True, properties=('area',
                                                                                        'axis_major_length',
                                                                                        'axis_minor_length',
                                                                                        'solidity',
                                                                                        'extent',
                                                                                        'label'))
        tub_label_props_df = pd.DataFrame(tub_label_props)

        # add custom measurements to the df
        tub_label_props_df['aspect_ratio'] = tub_label_props_df['axis_major_length'] / tub_label_props_df['axis_minor_length']
        for thresh_label in tub_labels:
            label_coords = np.column_stack(np.where(tub_labels == thresh_label))
            label_centroid = label_coords.mean(axis=0)
            dist = spatial.distance.euclidean(cell_mask_centroid, label_centroid)
            tub_label_props_df.loc[tub_label_props_df['label'] == thresh_label, 'dist_to_cell'] = dist

            # ask classifier to classify label props as spindle or not
            curr_label_stats = tub_label_props_df.loc[tub_label_props_df['label'] == thresh_label]
            curr_label_stats = curr_label_stats.drop(columns=['label'])
            curr_label_vals = curr_label_stats.values
            
            # remove the label if the classifier doesn't think it's a spindle
            spindle_prediction = spindle_classifier.predict(curr_label_vals)[0]
            print(f'spindle prediction for label {thresh_label}: {spindle_prediction}')
            if not spindle_prediction == 'spindle':
                labeled_tub_mask[labeled_tub_mask == thresh_label] = 0
        
        # move on if no labels remain
        tub_labels = [l for l in np.unique(labeled_tub_mask) if l != 0]
        if len(tub_labels) == 0:
            print('RF filtering removed remaining labels')
            continue
        elif len(tub_labels) > 1:
            print('more than one label remaining')
            continue
        else:
            print('one label remaining')
            tub_label = tub_labels[0]

        # filter label props df down to the remaining label
        tub_label_props = tub_label_props_df.loc[tub_label_props_df['label'] == tub_label]

        # get the cell mask properties
        cell_mask_props = regionprops_table(cell_mask.astype(np.uint8), cache=True, properties=('area',
                                                                                                'axis_major_length',
                                                                                                'axis_minor_length',
                                                                                                'solidity',
                                                                                                'extent'
                                                                                                ))
        cell_mask_props_df = pd.DataFrame(cell_mask_props)

        # classify the cell properties as a cell or not
        cell_props_vals = cell_mask_props_df.values
        cell_prediction = cell_classifier.predict(cell_props_vals)
        if cell_prediction[0] == 'bad':
            print('bad cell segmentation')
            continue
        else:
            print('good segmentation!')

        # map "spindle" and "cell" onto the dataframes to distinguish after merging
        tub_label_props.columns = tub_label_props.columns.map(lambda x: 'spindle_' + x)
        tub_label_props_dict = tub_label_props.to_dict('records')[0]
        cell_mask_props_df.columns = cell_mask_props_df.columns.map(lambda x: 'cell_' + x)
        cell_mask_props_dict = cell_mask_props_df.to_dict('records')[0]

        # record additional relevant metrics
        additional_label_props = {}
        additional_label_props['group'] = group
        additional_label_props['emb'] = emb
        additional_label_props['cell_id'] = mask_id

        # geometry and vector measurements

        # get the principle components of the spindle mask. downsample if there are too many points
        spindle_mask = labeled_tub_mask == tub_label
        spindle_coords = np.column_stack(np.where(spindle_mask))

        if spindle_coords.shape[0] > 1000:
            ds_val = int(spindle_coords.shape[0] / 1000)
            random.shuffle(spindle_coords)
            spindle_coords_ds = spindle_coords[::ds_val]
        else:
            spindle_coords_ds = spindle_coords

        spindle_centroid = np.mean(spindle_coords_ds, axis=0)
        spindle_pc0, spindle_pc1, spindle_pc2 = vg.principal_components(spindle_coords_ds)

        # get the principle components of the cell mask. downsample if there are too many points
        if cell_mask_coords.shape[0] > 1000:
            ds_val = int(cell_mask_coords.shape[0] / 1000)
            random.shuffle(cell_mask_coords)
            cell_mask_coords_ds = cell_mask_coords[::ds_val]

        cell_centroid = np.mean(cell_mask_coords_ds, axis=0)
        cell_pc0, cell_pc1, cell_pc2 = vg.principal_components(cell_mask_coords_ds)

        # get the angle between the long axis of the spindle and short axis of the cell:
        spindle_long_short_angle = vg.angle(spindle_pc0, cell_pc2)
        spindle_long_short_angle = np.min([np.abs(0-spindle_long_short_angle), 
                                           np.abs(180-spindle_long_short_angle)])

        # get the angle between the long axis of the spindle and the long axis of the cell:
        spindle_long_long_angle = vg.angle(spindle_pc0, cell_pc0)
        spindle_long_long_angle = np.min([np.abs(0-spindle_long_long_angle),
                                            np.abs(180-spindle_long_long_angle)])

        # get the vector between the spindle/cell centroid and the embryo centroid
        spindle_central_vect = findVec(masks_fused_centroid, spindle_centroid)
        cell_central_vect = findVec(masks_fused_centroid, cell_centroid)
        spindle_central_long_angle = vg.angle(spindle_central_vect, spindle_pc0)
        spindle_central_long_angle = np.min([np.abs(0-spindle_central_long_angle),  
                                             np.abs(180-spindle_central_long_angle)])
        cell_central_long_angle = vg.angle(cell_central_vect, cell_pc0)
        cell_central_long_angle = np.min([np.abs(0-cell_central_long_angle),
                                          np.abs(180-cell_central_long_angle)])
        
        cell_central_short_angle = vg.angle(cell_central_vect, cell_pc2)
        cell_central_short_angle = np.min([np.abs(0-cell_central_short_angle),
                                           np.abs(180-cell_central_short_angle)])
        
        additional_label_props['spindle_long_short_angle'] = spindle_long_short_angle
        additional_label_props['spindle_long_long_angle'] = spindle_long_long_angle
        additional_label_props['spindle_central_long_angle'] = spindle_central_long_angle   
        additional_label_props['cell_central_long_angle'] = cell_central_long_angle
        additional_label_props['cell_central_short_angle'] = cell_central_short_angle

        additional_label_props.update(tub_label_props_dict)
        additional_label_props.update(cell_mask_props_dict)
        dataframelist.append(additional_label_props)
        
        np.savetxt(os.path.join(save_dir, f'{embryo_name}_{mask_id}_spindle_pco.txt'), spindle_pc0)
        np.savetxt(os.path.join(save_dir, f'{embryo_name}_{mask_id}_cell_pco.txt'), cell_pc0)
        np.savetxt(os.path.join(save_dir, f'{embryo_name}_{mask_id}_cell_pc2.txt'), cell_pc2)
        np.savetxt(os.path.join(save_dir, f'{embryo_name}_{mask_id}_spindle_central_vect.txt'), spindle_central_vect)
        np.savetxt(os.path.join(save_dir, f'{embryo_name}_{mask_id}_cell_central_vect.txt'), cell_central_vect)
        np.savetxt(os.path.join(save_dir, f'{embryo_name}_{mask_id}_spindle_centroid.txt'), spindle_centroid)
        np.savetxt(os.path.join(save_dir, f'{embryo_name}_{mask_id}_cell_centroid.txt'), cell_centroid)
        np.savetxt(os.path.join(save_dir, f'{embryo_name}_{mask_id}_cell_coords.txt'), cell_mask_coords)
        np.savetxt(os.path.join(save_dir, f'{embryo_name}_{mask_id}_spindle_coords.txt'), spindle_coords)

summary_df= pd.DataFrame(dataframelist)
summary_df.to_csv(os.path.join(base_dir, 'summary.csv'), index=False)









