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

def find_vector(pt1,pt2):
    ''' 
    Calculate the vector between two points
    '''
    vect = [0 for c in pt1]
    for dim, coord in enumerate(pt1):
        deltacoord = pt2[dim]-coord
        vect[dim] = deltacoord
    return np.array(vect)

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
        

    def find_emb_centroid(self, overwrite = False):
        '''
        Load the Cellpose segmentation file and find the centroid of all masked regions (essentially, the centroid of the embryo).
        Save the centroid coordinates as a .txt file
        '''
        
        if os.path.isfile(self.emb_cent_save) and not overwrite:
            self.emb_centroid = np.loadtxt(self.emb_cent_save)
        else:
            bool_masks = np.load(self.segmentation_path, allow_pickle=True).item()['masks'].astype('bool') 
            bool_masks_coords = np.column_stack(np.where(bool_masks))
            self.emb_centroid = bool_masks_coords.mean(axis=0)
            np.savetxt(self.emb_cent_save, self.emb_centroid)
        return self.emb_centroid
        
    def filter_cell_labels(self, overwrite = False):

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

    def filter_spindle_labels(self, overwrite = False):
        
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
        
    def calculate_properties(self):

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

        ''' 
        Iterate through the cell labels: -----
        1) calculate the cell properties: -----
            a) the region props defaults    -----
            b) the centroid location        -----
            c) distance from cell to embryo centroid    -----
            d) The pc0, pc1, pc2 vectors    -----
            e) the vector from embryo centroid to cell centroid (v0)
            f) the angle between pc0 and v0
            g) the angle between pc2 and v0
        2) add the distance, and the two angles to the properties dictionary  
        3) save the pco, pc1, and pc2 vectors and centroid as <group>_E<num>_cell_pcs and <group>_E<num>_cell_centroid
        4) see if there is a spindle mask within the cell
        5) if so:
            a) calculate the region props defaults
            b) calculate the distance from the cell to spindle centroid
            c) calculate the pc0, pc1, and pc2 vectors
            d) add the distance to the region properties dict
            e) save the pco, pc1, and pc2 vectors as <group>_E<num>_spindle_pcs
        6) if not, continue to the next cell
        '''

        cell_label_nums = [i for i in np.unique(self.filtered_cell_masks) if not i == 0]
        for cell_label in cell_label_nums:
            cell_mask = self.filtered_cell_masks == cell_label
            cell_mask_props = regionprops_table(cell_mask.astype(np.uint8), cache=True, properties=('area',
                                                                                                    'axis_major_length',
                                                                                                    'axis_minor_length',
                                                                                                    'solidity',
                                                                                                    'extent'
                                                                                                    ))

            # get the principle components of the cell mask. downsample if there are too many points
            cell_coords = np.column_stack(np.where(cell_mask))
            if cell_coords.shape[0] > 1000:
                ds_val = int(cell_coords.shape[0] / 1000)
                np.random.shuffle(cell_coords)
                cell_mask_coords_ds = cell_coords[::ds_val]
            cell_centroid = np.mean(cell_mask_coords_ds, axis=0)
            cell_pc0, cell_pc1, cell_pc2 = vg.principal_components(cell_mask_coords_ds)

            np.savetxt(self.vect_save_path / (self.embryo_name + f'_cell{cell_label}_centroid.txt'), cell_centroid)
            np.savetxt(self.vect_save_path / (self.embryo_name + f'_cell{cell_label}_pc0.txt'), cell_pc0)
            np.savetxt(self.vect_save_path / (self.embryo_name + f'_cell{cell_label}_pc1.txt'), cell_pc1)
            np.savetxt(self.vect_save_path / (self.embryo_name + f'_cell{cell_label}_pc2.txt'), cell_pc2)

            '''
            Up above also calculate the vector from the cell centroid to the embryo centroid and save to txt
            '''


            '''
            Down below also add the angle between cell axes and v0
            '''
            cell_mask_props['cell_to_emb_dist'] = spatial.distance.euclidean(self.emb_centroid, cell_centroid)




                






            


        




