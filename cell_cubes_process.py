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
from skimage.measure import regionprops_table
from voldist_tools.basictools import open_with_napari, find_label_density, calculate_cell_properties, calculate_label_properties, get_long_axis

if __name__ == '__main__':

    viewer = napari.Viewer()

    # define directories
    main_dir = '/Volumes/bigData/wholeMount_volDist/220712_Fix_Emb_Flvw_Chn1GAP_PI_aTub647_Processed/N2V_Denoised/0_Analysis_01/0_data_cubes_TopHat-DoG_mask_otsu'
    save_dir = os.path.join(main_dir, 'processed_cubes')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cube_paths = [] 
    subdirs = [s for s in os.listdir(main_dir) if not s.startswith('.') and not 'label' in s and not 'processed' in s]
    for subdir in subdirs:
        cell_nums = [s for s in os.listdir(os.path.join(main_dir, subdir)) if not s.startswith('.') and not '.csv' in s and not '.ipynb' in s]
        for cell_num in cell_nums:
            folder_path = os.path.join(main_dir, subdir, cell_num)
            cube_paths.append(folder_path)

    # load the cell and spindle classifiers
    spindle_classifier_path = '/Users/bementmbp/Desktop/Scripts/volume-distribution/classifiers/spindle_classifier_3classes.joblib'
    spindle_classifier = joblib.load(spindle_classifier_path)
    cell_classifier_path = '/Users/bementmbp/Desktop/Scripts/volume-distribution/classifiers/cell_classifier.joblib'
    cell_classifier = joblib.load(cell_classifier_path)

    ########################################################################################################################
    cell_label_props_list = []
    spindle_label_props_list = []
    spatial_props_list = []

    cellwise_classification = []

    for cube_path in tqdm(cube_paths):
        
        open_with_napari(file_path = cube_path, viewer_name = viewer)
        cube_name = "_".join(cube_path.rsplit('/', 2)[1:])
        print(f'Now viewing {cube_name}')

        # get the properties of the spindle labels
        all_labels = viewer.layers['labels'].data
        label_nums, props_df = calculate_label_properties(napari_viewer_name = viewer)
        if len(label_nums) == 0:
            print(f'No spindle labels found in {cube_name}')
            continue

        # ask classifier to classify label props as spindle or not
        num_spindles = 0
        num_trash = 0
        num_halves = 0
        for label_num in label_nums:
            stats = props_df.loc[props_df['label'] == label_num]
            stats = stats.drop(columns=['label'])
            vals = stats.values

            # remove the label if the classifier doesn't think it's a spindle
            spindle_prediction = spindle_classifier.predict(vals)
            print(f'spindle prediction for label {label_num} of embryo: {spindle_prediction[0]}')
            if not spindle_prediction[0] == 'spindle':
                all_labels[all_labels == label_num] = 0
            if spindle_prediction[0] == 'spindle':
                num_spindles += 1
            elif spindle_prediction[0] == 'trash':
                num_trash += 1
            elif spindle_prediction[0] == 'half':
                num_halves += 1
        
        cellwise_classification.append({'cube_name': cube_name, 
                                        'num_spindles': num_spindles, 
                                        'num_trash': num_trash, 
                                        'num_halves': num_halves})

        # save cellwise_classification as csv
        '''
        cellwise_classification_results = pd.read_csv(os.path.join(save_dir, 'cellwise_classification.csv'))
        cellwise_classification_results = pd.concat([cellwise_classification_results, pd.DataFrame(cellwise_classification)])
        cellwise_classification_results.to_csv(os.path.join(save_dir, 'cellwise_classification.csv'), index=False)
        '''

        # get the remaining label numbers, if more or less than 1 label remains continue to next cell cube
        remaining_label_nums = [l for l in np.unique(all_labels) if l != 0]
        if len(remaining_label_nums) == 0:
            print('No remaining labels')
            continue

        elif len(remaining_label_nums) > 1:
            print('Multiple remaining labels')
            continue

        print('Single remaining label')

        # get the properties of the spindle label from the original dataframe
        spindle_label_num = remaining_label_nums[0]
        spindle_props_df = props_df.loc[props_df['label'] == spindle_label_num]

        # get the cell label properties and see if the classifier thinks it's good or bad
        cell_props_df = calculate_cell_properties(napari_viewer_name = viewer)
        cell_props_vals = cell_props_df.values
        cell_prediction = cell_classifier.predict(cell_props_vals)
        if cell_prediction[0] == 'bad':
            print('bad cell segmentation')
            continue

        # final sanity check:
        if not spindle_props_df.shape[0] == 1 and cell_props_df.shape[0] == 1:
            print('multiple spindle drops or multiple cell labels remaining!')
            continue
        print('good cell segmentation!')

        # create a save path for the cube:
        cube_save_path = os.path.join(save_dir, cube_name)
        if not os.path.exists(cube_save_path):
            os.makedirs(cube_save_path)

        # get the principle components of the spindle mask. downsample if there are too many points
        spindle_mask = all_labels == spindle_label_num
        spindle_coords = np.column_stack(np.where(spindle_mask))

        if spindle_coords.shape[0] > 1000:
            ds_val = int(spindle_coords.shape[0] / 1000)
            random.shuffle(spindle_coords)
            spindle_coords = spindle_coords[::ds_val]

        spindle_centroid = np.mean(spindle_coords, axis=0)
        spindle_pc0, spindle_pc1, spindle_pc2 = vg.principal_components(spindle_coords)

        # get the long axis of the cell mask and the angle between spindle and cell vectors
        cell_mask = viewer.layers['cell mask'].data.astype('bool')
        cell_coords = np.column_stack(np.where(cell_mask))

        if cell_coords.shape[0] > 1000:
            ds_val = int(cell_coords.shape[0] / 1000)
            random.shuffle(cell_coords)
            cell_coords = cell_coords[::ds_val]

        cell_centroid = np.mean(cell_coords, axis=0)
        cell_pc0, cell_pc1, cell_pc2 = vg.principal_components(cell_coords)

        # merge the cell and spindle label properties
        spindle_props_df.columns = spindle_props_df.columns.map(lambda x: 'spindle_' + x)
        cell_props_df.columns = cell_props_df.columns.map(lambda x: 'cell_' + x)
        label_props_df = pd.concat([spindle_props_df.reset_index(drop=True), cell_props_df.reset_index(drop=True)], axis=1)

        cube_name_splits = cube_name.split('_')
        group = cube_name_splits[0]
        emb = cube_name_splits[1]
        cell = "".join(cube_name_splits[2:])

        label_props_df['group'] = group
        label_props_df['emb'] = emb
        label_props_df['cell'] = cell
        label_props_df['spindle_pc0-0'] = spindle_pc0[0]
        label_props_df['spindle_pc0-1'] = spindle_pc0[1]
        label_props_df['spindle_pc0-2'] = spindle_pc0[2]
        label_props_df['spindle_pc1-0'] = spindle_pc1[0]
        label_props_df['spindle_pc1-1'] = spindle_pc1[1]
        label_props_df['spindle_pc1-2'] = spindle_pc1[2]
        label_props_df['spindle_pc2-0'] = spindle_pc2[0]
        label_props_df['spindle_pc2-1'] = spindle_pc2[1]
        label_props_df['spindle_pc2-2'] = spindle_pc2[2]
        label_props_df['cell_pc0-0'] = cell_pc0[0]
        label_props_df['cell_pc0-1'] = cell_pc0[1]
        label_props_df['cell_pc0-2'] = cell_pc0[2]
        label_props_df['cell_pc1-0'] = cell_pc1[0]
        label_props_df['cell_pc1-1'] = cell_pc1[1]
        label_props_df['cell_pc1-2'] = cell_pc1[2]
        label_props_df['cell_pc2-0'] = cell_pc2[0]
        label_props_df['cell_pc2-1'] = cell_pc2[1]
        label_props_df['cell_pc2-2'] = cell_pc2[2]

        # save the tif compatible layers as tifs
        images_and_layers = ['cell mask',
                            'tub',
                            'PI',
                            'labels'
                            ]
        for item in images_and_layers:
            viewer.layers[item].save(os.path.join(cube_save_path, item + '.tif'))

        label_props_df.to_csv(f'/Users/bementmbp/Desktop/wtf/{cube_name}.csv', index=False)
        # opens the results csv in the save dir
        results = pd.read_csv(os.path.join(save_dir, 'results.csv'))
        # concatenate the new results to the existing results
        results = pd.concat([results, label_props_df])
        results.to_csv(os.path.join(save_dir, 'results.csv'), index=False)

    napari.run()