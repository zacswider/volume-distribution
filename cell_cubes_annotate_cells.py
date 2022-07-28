import os
import napari
import random
import numpy as np
from voldist_tools.basictools import wipe_layers, open_with_napari, calculate_cell_properties

'''
Opens a napari viewer. The key press "n" will open a random cell cube (generated by cell_cubes_create).
If a given segmentation is a spindle, the key press 'q' will open a magicgui widget asking for the label
identity of the spindle. It then measured the properties of all labels in the cell cube (area, major axis
length, minor axis length, label id, distance to cell mask centroid, and label density), annotates the
appropriate label as a spindle, and saves the data as a csv file.
'''

if __name__ == '__main__':

    def annotate_labels(good_cell: bool):

        cell_props_df = calculate_cell_properties(napari_viewer_name = viewer)
        print('calculating cell mask properties...')
        if good_cell:
            cell_props_df['class'] = 'good'
            print(f'saving {cube_name}.csv as good cell = {good_cell}...')
        if not good_cell:
            cell_props_df['class'] = 'bad'
            print(f'saving {cube_name}.csv as good cell = {good_cell}...')

        cell_props_df.to_csv(os.path.join(save_dir, f'{cube_name}.csv'), index=False)
        print('done!')
        


    main_dir = '/Volumes/bigData/wholeMount_volDist/220712_Fix_Emb_Flvw_Chn1GAP_PI_aTub647_Processed/N2V_Denoised/0_Analysis_01/0_data_cubes'
    save_dir = os.path.join(main_dir, 'cell_label_properties')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cube_paths = [] 
    subdirs = [s for s in os.listdir(main_dir) if not s.startswith('.') and not 'label_properties' in s]
    for subdir in subdirs:
        cell_nums = [s for s in os.listdir(os.path.join(main_dir, subdir)) if not s.startswith('.')]
        for cell_num in cell_nums:
            folder_path = os.path.join(main_dir, subdir, cell_num)
            cube_paths.append(folder_path)

    random.shuffle(cube_paths)

    viewer = napari.Viewer(title = 'this is the cube viewer')

    @viewer.bind_key('n')
    def next_cube(viewer):
        wipe_layers(viewer)
        # define next_path variable as global
        next_path = cube_paths.pop(0)
        open_with_napari(next_path, viewer, view_pi = True)
        global cube_name
        cube_name = "_".join(next_path.rsplit('/', 2)[1:])
        print(f'Now viewing {cube_name}')

    @viewer.bind_key('g')
    def annotate_good_cell(viewer):
        annotate_labels(good_cell=True)

    @viewer.bind_key('t')
    def annotate_bad_cell(viewer):
        annotate_labels(good_cell=False)

    napari.run()








            