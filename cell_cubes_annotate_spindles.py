import os
import napari
import random
import numpy as np
from voldist_tools.basictools import wipe_layers, open_with_napari, calculate_label_properties

'''
Opens a napari viewer. The key press "n" will open a random cell cube (generated by cell_cubes_create).
If a given segmentation is a spindle, the key press 'q' will open a magicgui widget asking for the label
identity of the spindle. It then measured the properties of all labels in the cell cube (area, major axis
length, minor axis length, label id, distance to cell mask centroid, and label density), annotates the
appropriate label as a spindle, and saves the data as a csv file.
'''

if __name__ == '__main__':

    def annotate_labels(spindle_ID: str):

        label_props_df = calculate_label_properties(napari_viewer_name = viewer)
        all_label_nums = label_props_df['label'].tolist()

        # sanity checks up on input:
        try:
            spindle_ID = int(spindle_ID)
        except ValueError:
            print('spindle label number must be an integer')
            return
        if spindle_ID not in np.unique(all_label_nums) and spindle_ID != 0:
            print(f'label {spindle_ID} not found')
            print('please enter 0 if no spindle threshold is detected')
            return
        if spindle_ID == 0:
            spindle_ID = None
            print('No spindle threshold specified. Categorizing all labels as non-spindle.')
        else:
            print(f'calculating properties for spindle {spindle_ID}')

        # make a new column named "spindle" and assign to 1 if the label value is 1 otherwise 0
        label_props_df['spindle'] = (label_props_df['label'] == spindle_ID).astype(int)
        # drop the label column and save the df
        label_props_df.drop(columns=['label'], inplace=True)
        label_props_df.to_csv(os.path.join(save_dir, f'{cube_name}.csv'), index=False)
        print(f'saved {cube_name}.csv')


    main_dir = '/Volumes/bigData/wholeMount_volDist/220712_Fix_Emb_Flvw_Chn1GAP_PI_aTub647_Processed/N2V_Denoised/0_Analysis_01/0_data_cubes'
    save_dir = os.path.join(main_dir, 'label_properties')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cube_paths = [] 
    subdirs = [s for s in os.listdir(main_dir) if not s.startswith('.') and not 'label' in s and not 'processed' in s]
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
        open_with_napari(next_path, viewer)
        global cube_name
        cube_name = "_".join(next_path.rsplit('/', 2)[1:])
        print(f'Now viewing {cube_name}')

    @viewer.bind_key('q')
    def get_input(viewer):
        from magicgui.widgets import request_values
        values = request_values(name=dict(annotation=str, label='spindle label:'))
        annotate_labels(spindle_ID = values['name'])

    napari.run()








            