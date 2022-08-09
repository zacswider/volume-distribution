import os
import napari
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from tifffile import imread

'''
Opens a napari viewer...
'''

if __name__ == '__main__':
    viewer = napari.Viewer(title = 'this is the cube viewer')



    main_dir = '/Volumes/bigData/wholeMount_volDist/220712_Fix_Emb_Flvw_Chn1GAP_PI_aTub647_Processed/N2V_Denoised/0_Analysis_01'
    group = 'Exp'
    emb = 'E09'
    mask_dir = os.path.join(main_dir, f'{group}_{emb}')
    mask_name = [f for f in os.listdir(mask_dir) if f.endswith('.npy') and not f.startswith('.')][0]
    mask_path = os.path.join(mask_dir, mask_name)

    saved_output_dir = os.path.join(main_dir, 'vectors_and_masks')
    tub_name = [f for f in os.listdir(saved_output_dir) if not f.startswith('.') and f.endswith('.tif') and group in f and emb in f][0]
    tub_path = os.path.join(saved_output_dir, tub_name)

    cell_nums = [f.split('_')[2] for f in os.listdir(saved_output_dir) if not f.startswith('.') and f.endswith('.txt') and group in f and emb in f]
    cell_nums = np.unique(cell_nums).tolist()

    masks = np.load(mask_path, allow_pickle=True).item()['masks']
    masks_fused = masks.astype('bool') 
    masks_fused_coords = np.column_stack(np.where(masks_fused))
    masks_fused_centroid = masks_fused_coords.mean(axis=0)

    tub = imread(tub_path)

    viewer.add_labels(masks_fused, name='embryo', blending = 'additive', opacity=0.5)
    viewer.add_points(masks_fused_centroid, name='embryo centroid', blending = 'additive')
    viewer.add_image(tub, name='tub', blending = 'additive')


    spindle_mask = np.zeros(tub.shape, dtype='uint16')
    for cell in tqdm(cell_nums):
        spindle_coords = pd.read_csv(os.path.join(saved_output_dir, f'{group}_{emb}_{cell}_spindle_coords.txt'), sep=" ").to_numpy().astype('int')

        if spindle_coords.shape[0] > 1000:
            np.random.shuffle(spindle_coords)
            ds = spindle_coords.shape[0] // 1000
            spindle_coords = spindle_coords[::ds]

        for i in range(spindle_coords.shape[0]):
            spindle_mask[spindle_coords[i,0], spindle_coords[i,1], spindle_coords[i,2]] = 2

        spindle_pc0 = np.loadtxt(os.path.join(saved_output_dir,f'{group}_{emb}_{cell}_spindle_pco.txt'))
        spindle_centroid = spindle_coords.mean(axis=0)
        line_length = 50
        spindle_pc0_points = spindle_pc0 * np.mgrid[-line_length:line_length:2j][:, np.newaxis]
        spindle_pc0_points += spindle_centroid
        viewer.add_shapes(spindle_pc0_points, shape_type='line', name=f'{cell}_pc0', edge_color='green', opacity=0.35, blending='additive')

    viewer.add_labels(spindle_mask, name='spindle mask', blending = 'additive')

    napari.run()








            