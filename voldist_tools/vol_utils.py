from genericpath import isfile
import os
import vg
import numpy as np
from pathlib import Path
from skimage.segmentation import clear_border
from skimage import morphology
from skimage.measure import regionprops_table
from skimage.filters import threshold_otsu
import joblib
from tqdm import tqdm
import pandas as pd
from tifffile import imread, imwrite
from scipy import spatial
import time
import napari
np.seterr(invalid='ignore')

def remove_large_objects(labels_array: np.ndarray, max_size: int) -> np.ndarray:
    ''' 
    Remove all objects in a mask above a specific threshold
    '''
    out = np.copy(labels_array)
    component_sizes = np.bincount(labels_array.ravel()) 
    too_big = component_sizes > max_size
    too_big_mask = too_big[labels_array]
    out[too_big_mask] = 0
    return out

def wipe_layers(viewer) -> None:
    '''
    Delete all layers in the viewer objected
    '''
    layers = viewer.layers
    while len(layers) > 0:
        layers.remove(layers[0])

def find_vector(pt1,pt2):
    ''' 
    Calculate the vector between two points
    '''
    vect = [0 for c in pt1]
    for dim, coord in enumerate(pt1):
        deltacoord = pt2[dim]-coord
        vect[dim] = deltacoord
    return np.array(vect)

def get_cube(source: np.ndarray, label_num: int) -> np.ndarray:
    '''
    Return a cube of the label in a mask
    ---
    Parameters:
    source: np.ndarray the mask containing the label
    label_num: int the label number of the label you want isolate
    ---
    Returns:
    cube: np.ndarray the cube of the label
    '''
    label_points = np.column_stack(np.where(source == label_num))
    x = label_points.T[0]
    y = label_points.T[1]
    z = label_points.T[2]
    x_min = np.min(x) - 1
    x_max = np.max(x) + 2
    y_min = np.min(y) - 1
    y_max = np.max(y) + 2
    z_min = np.min(z) - 1
    z_max = np.max(z) + 2
    return x_min, x_max, y_min, y_max, z_min, z_max

def apply_cube(source: np.ndarray, cube: tuple) -> np.ndarray:
    '''
    Crop an ndArray with a cube
    ---
    Parameters:
    source: np.ndarray the array to crop
    cube: tuple containing the x_min, x_max, y_min, y_max, z_min, z_max
    ---
    Returns:
    out: np.ndarray array with the cube applied
    '''
    x_min, x_max, y_min, y_max, z_min, z_max = cube
    out = source[x_min:x_max, y_min:y_max, z_min:z_max]
    return out

def get_principle_components(mask: np.ndarray):
    '''
    Returns the centroid and principle components of a boolean mask
    ---
    Parameters:
    mask: np.ndarray boolean mask to get the principle components of
    ---
    Returns:
    centroid: np.ndarray the centroid of the mask
    pc0: np.ndarray the "long axis" of the max
    pc1: np.ndarray the "medium axis" of the max
    pc2: np.ndarray the "short axis" of the max
    '''
    # get the principle components of the mask. downsample if there are too many points
    coords = np.column_stack(np.where(mask))
    if coords.shape[0] > 1000:
        ds_val = int(coords.shape[0] / 1000)
        np.random.shuffle(coords)
        coords = coords[::ds_val]
        
    centroid = np.mean(coords, axis=0) 
    pc0, pc1, pc2 = vg.principal_components(coords)
    return centroid, pc0, pc1, pc2

def line_from_vect(v, c, length = 50):
    '''
    return a line given a vector v and centroid c
    '''
    v_points = v * np.mgrid[-length:length:2j][:, np.newaxis]
    v_points += c
    return v_points

class EmbryoSeg():
    def __init__(self, path_to_data: str, 
                       min_cell_size = 250000, 
                       max_cell_size = 1000000, 
                       min_spind_size = 200, 
                       max_spind_size = 5000,
                       wthball = 5):

        # sanity checks upon object creation
        _npy_files = [f for f in os.listdir(path_to_data) if not f.startswith('.') and f.endswith('.npy')]

        if not len(_npy_files) == 1:
            error_message = ("Did not find expected number of files in the Embryo Seg directory \n" + 
            "Expecting a single cellpose .npy file per Embryo Seg directory.")
            raise ValueError(error_message)

        _tif_files = [f for f in os.listdir(path_to_data) if not f.startswith('.') and f.endswith('.tif')]
        _raw_tub_files = [f for f in _tif_files if '_Tub_' in f and not '_sbdl2_' in f]
        _DoG_tub_files = [f for f in _tif_files if '_Tub_' in f and '_sbdl2_' in f]
        _raw_pi_files = [f for f in _tif_files if '_PI_' in f and not '_sbdl2_' in f and not '_Tub_' in f]
        _DoG_pi_files = [f for f in _tif_files if '_PI_' in f and '_sbdl2_' in f and not '_Tub_' in f]

        if any(len(_files) != 1 for _files in [_raw_tub_files, _DoG_tub_files, _raw_pi_files, _DoG_pi_files]):
            error_message = ("Did not find expected files in the Embryo Seg directory \n" + 
            "Expecting raw and DoG filtered PI and a-Tub stacks, 4 total files expected.")
            raise ValueError(error_message)
        
        self.min_cell_size = min_cell_size
        self.max_cell_size = max_cell_size
        self.min_spind_size = min_spind_size
        self.max_spind_size = max_spind_size
        self.wthball = wthball

        self.data_path = Path(path_to_data)
        self.segmentation_path = Path(self.data_path / _npy_files[0])
        self.raw_tub_path = Path(self.data_path / _raw_tub_files[0])
        self.DoG_tub_path = Path(self.data_path / _DoG_tub_files[0])
        self.raw_pi_path = Path(self.data_path / _raw_pi_files[0])
        self.DoG_pi_path = Path(self.data_path / _DoG_pi_files[0])
        self.vect_save_path = Path(self.data_path / 'coords_and_vectors')
        self.embryo_name = str(path_to_data).split('/')[-1]
        self.group = self.embryo_name.split('_')[0]
        self.embryo_num = self.embryo_name.split('_')[1]
        self.vect_save_path.mkdir(parents=False, exist_ok=True)
        self.parent_dir = Path( __file__ ).parent.absolute().parent
        self.emb_cent_save = self.vect_save_path / (self.embryo_name + '_emb_cent.txt')
        self.filt_mask_save = self.vect_save_path / (self.embryo_name + '_filtered_cell_masks.tif')
        self.spind_mask_save = self.vect_save_path / (self.embryo_name + '_filtered_spindle_masks.tif')
        self.whitetophat_save = self.vect_save_path / (self.embryo_name + f'_tub_whitetophat_{self.wthball}.tif')
        self.properties_save = self.vect_save_path / (self.embryo_name + '_properties.csv')
        
    def find_emb_centroid(self, overwrite = False) -> np.ndarray:
        '''
        Load the Cellpose segmentation file and find the centroid of all masked regions (essentially, the centroid of the embryo).
        Save the centroid coordinates as a .txt file
        Parameters
        ----------
        overwrite : bool whether to overwrite previously calculated properties file

        Returns
        -------
        self.emb_centroid : numpy array specifying the embryo centroid
        '''
        
        if os.path.isfile(self.emb_cent_save) and not overwrite:
            self.emb_centroid = np.loadtxt(self.emb_cent_save)
        else:
            bool_masks = np.load(self.segmentation_path, allow_pickle=True).item()['masks'].astype('bool') 
            bool_masks_coords = np.column_stack(np.where(bool_masks))
            self.emb_centroid = bool_masks_coords.mean(axis=0)
            np.savetxt(self.emb_cent_save, self.emb_centroid)
        return self.emb_centroid
        
    def filter_cell_labels(self, overwrite = False) -> np.ndarray:
        '''
        Filters the cell labels with size thresholds and random forest classifier trained
        to recognize cell shapes.
        Parameters
        ----------
        overwrite : bool whether to overwrite previously calculated properties file

        Returns
        -------
        self.filtered_cell_masks : numpy array of filtered cell masks
        '''
        if os.path.isfile(self.filt_mask_save) and not overwrite:
            self.filtered_cell_masks = imread(self.filt_mask_save)
            return self.filtered_cell_masks

        print('filtering cell masks...')
        raw_masks = np.load(self.segmentation_path, allow_pickle=True).item()['masks']
        print('finished loading raw cell masks')
        filt_masks = clear_border(raw_masks)
        print('finished clearing borders')
        filt_masks = morphology.remove_small_objects(filt_masks, min_size=self.min_cell_size, connectivity=1)
        print('finished removing small labels')
        filt_masks = remove_large_objects(filt_masks, max_size=self.max_cell_size)
        print('finished removing large labels')
        cell_labels_before_rf = [l for l in np.unique(filt_masks) if l != 0]
        cell_classifier_path = self.parent_dir / ('classifiers/cell_classifier.joblib')
        cell_classifier = joblib.load(cell_classifier_path)
        print('Using RF to filter remaining bad masks')
        with tqdm(total = len(cell_labels_before_rf)) as pbar:
            pbar.set_description('Using RF to filter remaining bad masks')
            for cell_label in cell_labels_before_rf:
                cell_mask = filt_masks == cell_label
                cell_mask_props = regionprops_table(cell_mask.astype(np.uint8), cache=True, properties=('area',
                                                                                                        'axis_major_length',
                                                                                                        'axis_minor_length',
                                                                                                        'solidity',
                                                                                                        'extent'
                                                                                                        ))
                cell_props_vals = pd.DataFrame(cell_mask_props).values
                cell_prediction = cell_classifier.predict(cell_props_vals)
                if cell_prediction[0] == 'bad':
                    filt_masks[cell_mask] = 0
                pbar.update(1)
    
        self.filtered_cell_masks = filt_masks
        imwrite(self.filt_mask_save, self.filtered_cell_masks)
        
        return self.filtered_cell_masks

    def filter_spindle_labels(self, overwrite = False) -> np.ndarray:
        '''
        Calculates a cell-specific otsu threshold for the DoG and white tophat filtered tubulin arrays,
        then filters the corresponding labels with size thresholds and random forest classifier trained
        to recognize spindle shapes.
                Parameters
        ----------
        overwrite : bool whether to overwrite previously calculated properties file

        Returns
        -------
        self.filtered_spindle_masks : numpy array of filtered spindle masks
        '''
        
        # check for pre-existing filtered spindle labels array
        if os.path.isfile(self.spind_mask_save) and not overwrite:
            print('previously calculated spindle masks found. Loading now...')
            self.filtered_tub_masks = imread(self.spind_mask_save)
            print('finished loading spindle masks')
            return self.filtered_tub_masks

        # check for pre-existing filtered mask labels array
        if not os.path.isfile(self.filt_mask_save):
            print('no filtered labels found, filtering cell labels now...')
            self.filter_cell_labels()
        else:
            print('filtered cell masks found.')
            self.filtered_cell_masks = imread(self.filt_mask_save)
            print('finished loading filtered cell masks.')

        # check for pre-existing whit tophat filtered tub array
        if not os.path.isfile(self.whitetophat_save):
            print('no white tophat filtered tub file detected. Calculating white tophat now...')
            raw_tub = imread(self.raw_tub_path)
            self.whitetophat = morphology.white_tophat(raw_tub, footprint=morphology.ball(5))
            imwrite(self.whitetophat_save, self.whitetophat)
            print('finished calculating white tophat filtered tub.')
        else:
            print('previously calculate white tophat filtered tub found. Loading now...')
            self.whitetophat = imread(self.whitetophat_save)
            print('finished loading white tophat filtered tubulin file. ')
            
        cell_label_nums = [i for i in np.unique(self.filtered_cell_masks) if not i == 0]
        self.filtered_tub_masks = np.zeros(self.whitetophat.shape)
        # first spindle label
        spind_mask_val = 1
        spindle_classifier_path = self.parent_dir / ('classifiers/spindle_classifier_3classes.joblib')
        spindle_classifier = joblib.load(spindle_classifier_path)
        with tqdm(total = len(cell_label_nums)) as pbar:
            pbar.set_description('Calculating tub thresholds')
            for cell_label in cell_label_nums:
                
                curr_tub_mask = np.zeros(self.whitetophat.shape)
                cell_mask = self.filtered_cell_masks == cell_label
                cell_mask_coords = np.column_stack(np.where(cell_mask))
                cell_mask_centroid = cell_mask_coords.mean(axis=0)
                # erode the mask to eliminate some cortical signal
                eroded_mask = morphology.binary_erosion(cell_mask, footprint=np.ones((3, 3, 3)))
                for i in range(5):
                    eroded_mask = morphology.binary_erosion(eroded_mask)
                
                curr_tub_mask[eroded_mask] = self.whitetophat[eroded_mask]
                tub_mask_vals = self.whitetophat[eroded_mask].ravel()
                thresh_val = threshold_otsu(tub_mask_vals)
                curr_tub_labels = morphology.label(curr_tub_mask > thresh_val)
                curr_tub_labels = morphology.remove_small_objects(curr_tub_labels, min_size=self.min_spind_size, connectivity=1)
                curr_tub_labels = remove_large_objects(curr_tub_labels, self.max_spind_size)
                curr_remaining_labels = [i for i in np.unique(curr_tub_labels) if not i == 0]
                if len(curr_remaining_labels) == 0:
                    pbar.update(1)
                    continue
                if len(curr_remaining_labels) > 1:
                    curr_tub_labels = morphology.remove_small_objects(curr_tub_labels, min_size=self.min_spind_size, connectivity=1)
                    curr_tub_labels = remove_large_objects(curr_tub_labels, self.max_spind_size)
                    curr_remaining_labels = [i for i in np.unique(curr_tub_labels) if not i == 0]
                    if len(curr_remaining_labels) == 0:
                        return
                tub_label_props = regionprops_table(curr_tub_labels, cache = True, properties=('area',
                                                                                                'axis_major_length',
                                                                                                'axis_minor_length',
                                                                                                'solidity',
                                                                                                'extent',
                                                                                                'label'))
                tub_label_props_df = pd.DataFrame(tub_label_props)
                tub_label_props_df['aspect_ratio'] = tub_label_props_df['axis_major_length'] / tub_label_props_df['axis_minor_length']

                for thresh_label in curr_remaining_labels:
                    # add distance between cell centroid and spindle centroid to the props table
                    label_mask = curr_tub_labels == thresh_label
                    label_coords = np.column_stack(np.where(label_mask))
                    label_centroid = label_coords.mean(axis=0)
                    dist = spatial.distance.euclidean(cell_mask_centroid, label_centroid)
                    tub_label_props_df.loc[tub_label_props_df['label'] == thresh_label, 'dist_to_cell'] = dist

                    # ask classifier to classify label props as spindle or not
                    curr_label_stats = tub_label_props_df.loc[tub_label_props_df['label'] == thresh_label]
                    curr_label_stats = curr_label_stats.drop(columns=['label'])
                    curr_label_vals = curr_label_stats.values
                    spindle_prediction = spindle_classifier.predict(curr_label_vals)[0]
                    if not spindle_prediction == 'spindle':
                        curr_tub_labels[label_mask] = 0
                    
                final_thresh_labels = [i for i in np.unique(curr_tub_labels) if not i == 0]
                if not len(final_thresh_labels) == 1:
                    pbar.update(1)
                    continue
                final_spindle_mask = curr_tub_labels == final_thresh_labels[0]
                self.filtered_tub_masks[final_spindle_mask] = spind_mask_val
                spind_mask_val += 1
                pbar.update(1)

        imwrite(self.spind_mask_save, self.filtered_tub_masks)
        return self.filtered_tub_masks
        
    def calculate_properties(self, overwrite = False) -> pd.DataFrame:
        '''
        Uses the previously filtered spindle and cell masks to calculate individual properties (shape, size, etc.) 
        as well as geometric properties (orientation of y relative to x, etc.) for each cell.
        Parameters
        ----------
        overwrite : bool whether to overwrite previously calculated properties file

        Returns
        -------
        self.embryo_properties pandas.DataFrame with properties for each cell
        '''

        if os.path.isfile(self.properties_save) and not overwrite:
            print('previously calculated properties found. Loading now.')
            self.embryo_properties = pd.read_csv(self.properties_save)
            print('finished loading properties file. ')
            return self.embryo_properties

        # load embryo centroid
        if not os.path.isfile(self.emb_cent_save):
            print('no pre-calculated embryo centroid found, calculating centroid now...')
            self.emb_centroid = self.find_emb_centroid()
            print('finished calculating embryo centroid.')
        else: 
            print('pre-calculated embryo centroid found, loading now...')
            self.emb_centroid = np.loadtxt(self.emb_cent_save)
            print('finished loading embryo centroid')

        # load filtered cell mask array
        if not os.path.isfile(self.filt_mask_save):
            print('no pre-filtered cell masks found, filtering cell masks now...')
            self.filtered_cell_masks = self.filter_cell_labels()
            print('finished filtering cell masks')
        else:
            print('pre-filtered cell masks found, loading now...')
            self.filtered_cell_masks = imread(self.filt_mask_save)
            print('finished loading cell masks')
        
        # load filtered spindle array:
        if not os.path.isfile(self.spind_mask_save):
            print('no pre-filtered spindle masks found, filtering spindle masks now...')
            self.filtered_tub_masks = self.filter_spindle_labels()
            print('finished filtering spindle masks')
        else:
            print('pre-filtered spindle masks found, loading now...')
            self.filtered_tub_masks = imread(self.spind_mask_save)
            print('finished loading spindle masks')

        embryo_props = []

        cell_label_nums = [i for i in np.unique(self.filtered_cell_masks) if not i == 0]
        
        for cell_label in tqdm(cell_label_nums):
            print(f'calculating properties for cell {cell_label}')

            cell_mask = self.filtered_cell_masks == cell_label
            
            d1 = regionprops_table(cell_mask.astype(np.uint8), cache=True, properties=('area',
                                                                                      'axis_major_length',
                                                                                      'axis_minor_length',
                                                                                      'solidity',
                                                                                      'extent'
                                                                                      ))
            
            # annotate column names 
            cell_mask_props = {}
            for k, v in d1.items():
                cell_mask_props[f'cell_{k}'] = v[0]
            cell_mask_props['cell_label'] = cell_label
                                                                                                
            cell_centroid, cell_pc0, cell_pc1, cell_pc2 = get_principle_components(cell_mask)
            
            # calculate the geometric orientation of the cell relative the embryo centroid
            def get_angle(v1, v2): return np.min([np.abs(0 - vg.angle(v1, v2)), np.abs(180 - vg.angle(v1, v2))])
            v0 = find_vector(self.emb_centroid, cell_centroid)
            v0_pc0 = get_angle(v0, cell_pc0)
            v0_pc1 = get_angle(v0, cell_pc1)
            v0_pc2 = get_angle(v0, cell_pc2)

            # saving...
            np.savetxt(self.vect_save_path / (self.embryo_name + f'_cell{cell_label}_centroid.txt'), cell_centroid)
            np.savetxt(self.vect_save_path / (self.embryo_name + f'_cell{cell_label}_pc0.txt'), cell_pc0)
            np.savetxt(self.vect_save_path / (self.embryo_name + f'_cell{cell_label}_pc1.txt'), cell_pc1)
            np.savetxt(self.vect_save_path / (self.embryo_name + f'_cell{cell_label}_pc2.txt'), cell_pc2)
            np.savetxt(self.vect_save_path / (self.embryo_name + f'_cell{cell_label}_v0.txt'), v0)

            cell_mask_props['cell_to_emb_dist'] = spatial.distance.euclidean(self.emb_centroid, cell_centroid)
            cell_mask_props['cell_to_emb_pc0_angle'] = v0_pc0
            cell_mask_props['cell_to_emb_pc1_angle'] = v0_pc1
            cell_mask_props['cell_to_emb_pc2_angle'] = v0_pc2

            # see if there is a spindle mask within the cell
            spindle_labels_found = [i for i in np.unique(self.filtered_tub_masks[cell_mask]) if not i== 0]
            if not len(spindle_labels_found) == 1:
                embryo_props.append(cell_mask_props)
                continue

            print('spindle found')
            spindle_label = spindle_labels_found[0]
            spindle_mask = self.filtered_tub_masks == spindle_label
            cell_mask_props['has_spindle'] = True
            d2 = regionprops_table(spindle_mask.astype(np.uint8), cache = True, properties=('area',
                                                                                            'axis_major_length',
                                                                                            'axis_minor_length',
                                                                                            'solidity',
                                                                                            'extent',
                                                                                            'label'
                                                                                            ))
            # annotate column names
            for k, v in d2.items():
                cell_mask_props[f'spindle_{k}'] = v[0]
            
            spindle_centroid, spindle_pc0, spindle_pc1, spindle_pc2 = get_principle_components(spindle_mask)

            # calculate the geometric orientation of the spindle relative the cell centroid
            cell_pc0_to_spindle_pc0 = get_angle(cell_pc0, spindle_pc0)
            cell_pc2_to_spindle_pc0 = get_angle(cell_pc2, spindle_pc0)

            np.savetxt(self.vect_save_path / (self.embryo_name + f'_cell{cell_label}_spindle_centroid.txt'), spindle_centroid)
            np.savetxt(self.vect_save_path / (self.embryo_name + f'_cell{cell_label}_spindle_pc0.txt'), spindle_pc0)

            cell_mask_props['cell_to_spindle_dist'] = spatial.distance.euclidean(cell_centroid, spindle_centroid)
            cell_mask_props['cell_to_spindle_pc0_angle'] = cell_pc0_to_spindle_pc0
            cell_mask_props['cell_to_spindle_pc2_angle'] = cell_pc2_to_spindle_pc0

            embryo_props.append(cell_mask_props)
        
        self.embryo_properties = pd.DataFrame(embryo_props)
        self.embryo_properties.to_csv(self.vect_save_path / (self.embryo_name + '_properties.csv'))
        return self.embryo_properties

    def manual_annotation(self):
        ''' 
        Open a napari view to interactively visualize cropped cubes for each cell. Key binding allows 
        user to manually specify whether a cell is appropriately thresholded or not.
        '''
        
        viewer = napari.Viewer(title = 'Press "t" to specify trash. Press "n" to proceed to next cube', ndisplay = 3)
        
        # calculating label numbers from files is much faster than calculating from the image
        def get_label_numbers(mystring: str): return mystring.split('_')[2].split('cell')[-1]
        file_names = [f for f in os.listdir(self.vect_save_path) if not f.startswith('.') and f.endswith('.txt')]
        nums = [int(i) for i in map(get_label_numbers, file_names) if i.isnumeric()]
        unique_labels = np.unique(nums).tolist()

        print(f'{self.embryo_name}_cell{unique_labels[0]}_spindle_pc0.txt')

        # only include cell labels with associated spindles
        labels_with_spindles = [i for i in unique_labels if os.path.isfile(self.vect_save_path / f'{self.embryo_name}_cell{i}_spindle_pc0.txt')]

        # load the arrays
        print('loading data arrays...')
        filtered_cell_masks = imread(self.vect_save_path / (self.embryo_name + '_filtered_cell_masks.tif')).astype('uint16')
        filtered_spind_masks = imread(self.vect_save_path / (self.embryo_name + '_filtered_spindle_masks.tif')).astype('uint16')
        dog_tub = imread(self.DoG_tub_path)
        dog_pi = imread(self.DoG_pi_path)
        print('done loading arrays.')

        # list to store the bad numbers
        bad_cell_labels = []

        def load_label_cube(viewer, label_num: int) -> None:
            wipe_layers(viewer)
            cube_coords = get_cube(filtered_cell_masks, label_num)
            pi_cube = apply_cube(dog_pi, cube_coords)
            cell_mask_cube = apply_cube(filtered_cell_masks, cube_coords)
            tub_cube = apply_cube(dog_tub, cube_coords)
            spind_mask_cube = apply_cube(filtered_spind_masks, cube_coords)

            viewer.add_image(pi_cube, name = f'pi_cube_{label_num}', blending = 'additive', visible = False)
            viewer.add_labels(cell_mask_cube, name=f'cell_mask_cube_{label_num}', blending = 'additive', opacity = 0.5, visible = True)
            viewer.add_image(tub_cube, name = f'tub_cube_{label_num}', blending = 'additive', visible = True)
            viewer.add_labels(spind_mask_cube.astype('bool'), name = f'spind_mask_cube_{label_num}', blending = 'additive', opacity = 0.5, visible = True)
            global curr_label
            curr_label = label_num
            print(f'label {label_num} loaded')
        
        try:
            first_label = labels_with_spindles.pop(0)
            print(f'starting with label {first_label}')
        except IndexError:
            print('no labels with spindles found')
            return
        
        load_label_cube(viewer, first_label)
        
        @viewer.bind_key('t')
        def mark_as_bad(viewer):
            print(f'marking cell {curr_label} as bad')
            bad_cell_labels.append(int(curr_label))
            print(f'{len(bad_cell_labels)} cells marked as bad')
            np.savetxt(self.vect_save_path / (self.embryo_name + f'_bad_cells.txt'), bad_cell_labels, fmt='%i', delimiter=',')
        
        @viewer.bind_key('n')
        def next_cube(viewer):
            try:
                next_label = labels_with_spindles.pop(0)
                print(f'now loading label {next_label}')
            except IndexError:
                print('no more labels with spindles found')
                return
            load_label_cube(viewer, next_label)

        napari.run()

    def interact(self, show_curr_cell = False, cell_opacity = 0.35, spindle_opacity = 0.35):
        '''
        Open a napari viewer to interactively visualize the position of the cell and spindle 
        in the context of the whole embryo..
        Parameters
        ----------
        show_curr_cell : bool whether to highlight the current cell in viewer (time cost ~3 seconds)
        cell_opacity : float the opacity of the cell masks in the viewer
        spindle_opacity : float the opacity of the spindle masks in the viewer
        '''

        if not any([os.path.isfile(f) for f in [self.segmentation_path,
                                                self.filt_mask_save,
                                                self.spind_mask_save]]):
            print('One or more segmentations file not found. Run calculate_properties() first.')

        viewer = napari.Viewer(title = 'Click a mask to see its contents', ndisplay = 3)

        print('loading filtered cell masks...')
        filtered_cell_masks = imread(self.filt_mask_save).astype('uint16')
        print('loading filtered spindle masks...')
        filtered_tub_masks = imread(self.spind_mask_save).astype('uint16')
        print('loading embryo centroid...')
        embryo_centroid = np.loadtxt(self.emb_cent_save)
        print('opening viewer...')
        layer = viewer.add_labels(filtered_cell_masks, name ='filtered cell masks', blending='additive', opacity=cell_opacity)
        viewer.add_labels(filtered_tub_masks, name ='filtered spindle masks', blending='additive', opacity=spindle_opacity)
        viewer.add_points(embryo_centroid, name ='embryo centroid', blending = 'additive', size = 5)
    
        # calculating label numbers from files is much faster than calculating from the image
        def get_label_numbers(mystring: str): return mystring.split('_')[2].split('cell')[-1]
        file_names = [f for f in os.listdir(self.vect_save_path) if not f.startswith('.') and f.endswith('.txt')]
        nums = [int(i) for i in map(get_label_numbers, file_names) if i.isnumeric()]
        unique_labels = np.unique(nums)

        # fake a call to unique labels so we can just update the layers going forward
        # NOTE updating shapes in 3D is currently broken, so we will delete and reassign these shapes instead
        first_cell = unique_labels[0]
        print(f'first cell is {first_cell}')
        cell_centroid = np.loadtxt(self.vect_save_path / (self.embryo_name + f'_cell{first_cell}_centroid.txt'))
        cell_pc0 = np.loadtxt(self.vect_save_path / (self.embryo_name + f'_cell{first_cell}_pc0.txt'))
        cell_pc1 = np.loadtxt(self.vect_save_path / (self.embryo_name + f'_cell{first_cell}_pc1.txt'))
        cell_pc2 = np.loadtxt(self.vect_save_path / (self.embryo_name + f'_cell{first_cell}_pc2.txt'))
        v0 = np.loadtxt(self.vect_save_path / (self.embryo_name + f'_cell{first_cell}_v0.txt'))

        spindle_centroid_path = self.vect_save_path / (self.embryo_name + f'_cell{first_cell}_spindle_centroid.txt')
        spindle_pc0_path = self.vect_save_path / (self.embryo_name + f'_cell{first_cell}_spindle_pc0.txt')
        # not all cells have an adequately thresholded spindle within them.
        if os.path.isfile(spindle_centroid_path) and os.path.isfile(spindle_pc0_path):
            spindle_centroid = np.loadtxt(spindle_centroid_path)
            spindle_pc0 = np.loadtxt(spindle_pc0_path)
            spindle_long = line_from_vect(spindle_pc0, spindle_centroid)
            viewer.add_shapes(spindle_long, shape_type='line', name='spindle_pc0', edge_color='magenta', opacity=0.35, blending='additive')
        else:
            print(f'no spindle found in cell {first_cell}')
            spindle_long = line_from_vect([0,0,0], [0,0,0])
            viewer.add_shapes(spindle_long, shape_type='line', name='spindle_pc0', edge_color='magenta', opacity=0.35, blending='additive')

        cell_long = line_from_vect(cell_pc0, cell_centroid)
        cell_mid = line_from_vect(cell_pc1, cell_centroid)
        cell_short = line_from_vect(cell_pc2, cell_centroid)
        
        central_vec = line_from_vect(v0, embryo_centroid, length = 2)
        viewer.add_shapes(cell_long, shape_type='line', name='cell_pc0', edge_color='green', opacity=0.35, blending='additive')
        viewer.add_shapes(cell_mid, shape_type='line', name='cell_pc1', edge_color='blue', opacity=0.35, blending='additive')
        viewer.add_shapes(cell_short, shape_type='line', name='cell_pc2', edge_color='red', opacity=0.35, blending='additive')
        viewer.add_shapes(central_vec, shape_type='line', name='v0', edge_color='white', opacity=0.25, blending='additive')

        if show_curr_cell:
            curr_cell = np.zeros(filtered_cell_masks.shape, dtype='uint16')
            curr_cell_mask = filtered_cell_masks == first_cell
            curr_cell[curr_cell_mask] = first_cell
            curr_cell_layer = viewer.add_labels(curr_cell, name ='curr_cell', blending='additive', opacity=0.55)

        viewer.layers.selection.active = viewer.layers['filtered cell masks']


        def load_files(viewer, label_num: str) -> None:
            '''
            Given a cell label number, load the corresponding vectors and add them to the viewer.
            NOTE: creating a "current cell" layer is rate limiting (~2-3 seconds).
            '''
            # sanity checks up on input:
            try:
                label_num = int(label_num)
            except ValueError:
                print('spindle label number must be an integer')
                return

            # load the data
            cell_centroid = np.loadtxt(self.vect_save_path / (self.embryo_name + f'_cell{label_num}_centroid.txt'))
            cell_pc0 = np.loadtxt(self.vect_save_path / (self.embryo_name + f'_cell{label_num}_pc0.txt'))
            cell_pc1 = np.loadtxt(self.vect_save_path / (self.embryo_name + f'_cell{label_num}_pc1.txt'))
            cell_pc2 = np.loadtxt(self.vect_save_path / (self.embryo_name + f'_cell{label_num}_pc2.txt'))
            v0 = np.loadtxt(self.vect_save_path / (self.embryo_name + f'_cell{label_num}_v0.txt'))

            spindle_centroid_path = self.vect_save_path / (self.embryo_name + f'_cell{label_num}_spindle_centroid.txt')
            spindle_pc0_path = self.vect_save_path / (self.embryo_name + f'_cell{label_num}_spindle_pc0.txt')

            # check for spindles
            if os.path.isfile(spindle_centroid_path) and os.path.isfile(spindle_pc0_path):
                spindle_centroid = np.loadtxt(spindle_centroid_path)
                spindle_pc0 = np.loadtxt(spindle_pc0_path)
                spindle_long = line_from_vect(spindle_pc0, spindle_centroid)
                viewer.layers.remove(viewer.layers['spindle_pc0'])
                viewer.add_shapes(spindle_long, shape_type='line', name='spindle_pc0', edge_color='magenta', opacity=0.35, blending='additive')
            else:
                print(f'no spindle found in cell {label_num}')
                viewer.layers['spindle_pc0'].visible = False

            # create the new lines from new vectors
            cell_long = line_from_vect(cell_pc0, cell_centroid)
            cell_mid = line_from_vect(cell_pc1, cell_centroid)
            cell_short = line_from_vect(cell_pc2, cell_centroid)
            central_vec = line_from_vect(v0, embryo_centroid, length = 2)

            # updating shapes in 3D is currently broken in napari, so we will delete and reassign
            viewer.layers.remove(viewer.layers['cell_pc0'])
            viewer.add_shapes(cell_long, shape_type='line', name='cell_pc0', edge_color='green', opacity=0.35, blending='additive')
            viewer.layers.remove(viewer.layers['cell_pc1'])
            viewer.add_shapes(cell_mid, shape_type='line', name='cell_pc1', edge_color='blue', opacity=0.35, blending='additive')
            viewer.layers.remove(viewer.layers['cell_pc2'])
            viewer.add_shapes(cell_short, shape_type='line', name='cell_pc2', edge_color='red', opacity=0.35, blending='additive')
            viewer.layers.remove(viewer.layers['v0'])
            viewer.add_shapes(central_vec, shape_type='line', name='v0', edge_color='white', opacity=0.25, blending='additive')

            if show_curr_cell:
                curr_cell = np.zeros(filtered_cell_masks.shape, dtype='uint16')
                curr_cell_mask = filtered_cell_masks == label_num
                curr_cell[curr_cell_mask] = label_num
                curr_cell_layer.data = curr_cell

            viewer.layers.selection.active = viewer.layers['filtered cell masks']
            print(f'now viewing cell {label_num}')

        @layer.mouse_drag_callbacks.append
        def update_layer(layer, event):
            if len(viewer.dims.displayed) == 2:
                mask_num = layer.get_value(event.position)
                print(f'clicked on mask {mask_num}')
                if not mask_num == 0:
                    load_files(viewer, label_num = mask_num)
                    print('viewing cell', mask_num)
                else:
                    print('click a mask to see its contents')
    
        @viewer.bind_key('v')
        def get_input(viewer):
            from magicgui.widgets import request_values
            values = request_values(name=dict(annotation=str, label='spindle label:'))
            if values == None:
                print('please enter a value')
            else:
                load_files(viewer, label_num = values['name'])

        napari.run()









                






            


        




