import os
import vg
import time
import joblib
import napari
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import spatial
from skimage import filters
from skimage import morphology
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from tifffile import imread, imwrite
from skimage.morphology import label
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.morphology import binary_erosion
from skimage.measure import regionprops, regionprops_table
from sklearn.ensemble import RandomForestClassifier
from qtpy.QtWidgets import QApplication, QPushButton
from contextlib import contextmanager



def wipe_layers(viewer_name: str) -> None:
    '''
    Delete all layers in the viewer objected
    '''
    layers = viewer_name.layers
    while len(layers) > 0:
        layers.remove(layers[0])

def remove_large_objects(labels_array: np.ndarray, max_size: int) -> np.ndarray:
    ''' 
    Remove all objects in a mask above a specific threshold
    '''
    out = np.copy(labels_array)
    component_sizes = np.bincount(labels_array.ravel()) # count the number of pixels in different labels
    too_big = component_sizes > max_size
    too_big_mask = too_big[labels_array]
    out[too_big_mask] = 0
    return out

def return_points(labels_array: np.ndarray, label_ID: int) -> np.ndarray:
    '''
    Return the points in a mask that belong to a specific label
    ---
    Parameters:
    labels_array: np.ndarray an ndArray of labels
    label_ID: int the label ID of the label whos points you want to calculate
    ---
    Returns:
    points: np.ndarray an ndArray of shape (n,3) where n is the number of points in the label
    and dim1 is the x,y,z coordinates of the points
    '''
    points = np.column_stack(np.where(labels_array == label_ID))
    return points

def find_label_density(label_points: np.ndarray) -> float:
    '''
    Calculate the bounding box for a point cloud and return the density of points in the bounding box
    ---
    Parameters:
    label_points: np.ndarray the array point coordinates for a given label
    ---
    Returns:
    np.nan if the label is 0, or if the label has no length
    density (float) the number of points in the label divided by the volume of the bounding box
    '''

    x = label_points.T[0]
    y = label_points.T[1]
    z = label_points.T[2]
    num_points = len(x)
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)
    z_min = np.min(z)
    z_max = np.max(z)
    # add 1 to prevent division by 0
    x_range = (x_max - x_min) + 1
    y_range = (y_max - y_min) + 1
    z_range = (z_max - z_min) + 1
    vol = x_range * y_range * z_range
    density = num_points / vol
    return density

def print_label_props(source: np.ndarray, label_num: int) -> None:
    '''
    Print the properties of a label in a mask
    ---
    Parameters:
    source: np.ndarray the mask containing the label
    label_num: int the label number of the label you want to print the properties of
    ---
    Returns:
    None
    '''
    label_points = return_points(source, label_num)
    density = find_label_density(label_points)
    size = label_points.shape[0]
    print(f'Label {label_num} has:')
    print(f'{size:,} points.')
    print(f'density of {round(density,4):,}')

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
    label_points = return_points(source, label_num)
    x = label_points.T[0]
    y = label_points.T[1]
    z = label_points.T[2]
    x_min = np.min(x) - 1
    x_max = np.max(x) + 2
    y_min = np.min(y) - 1
    y_max = np.max(y) + 2
    z_min = np.min(z) - 1
    z_max = np.max(z) + 2
    #cube = source[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
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

def get_long_axis(cubed_label: np.ndarray, line_length = 75):
    '''
    Get the longest axis of an cubed_label
    ---
    Parameters:
    cubed_label: np.ndarray the cubed_label to get the longest axis of
    ---
    Returns:
    linepts: np.ndarray the points of the longest axis
    '''
    if cubed_label.dtype == 'bool':
        coords = np.column_stack(np.where(cubed_label == True))
    else:
        label_identify = [i for i in np.unique(cubed_label) if i != 0][0]
        coords = np.column_stack(np.where(cubed_label == label_identify))
    if coords.shape[0] > 1000:
        sampling_interval = coords.shape[0] // 1000
    else:
        sampling_interval = 1    

    np.random.shuffle(coords)
    subsampled = coords[::sampling_interval]
    datamean = subsampled.mean(axis=0)
    uu, dd, vv = np.linalg.svd(subsampled - datamean)
    linepts = vv[0] * np.mgrid[-line_length:line_length:2j][:, np.newaxis]
    linepts += datamean
    return vv[0], linepts

def view_saved_files(file_path: str) -> None:
    ''' 
    Fxn for visualizing saved output files.
    '''
    dedicated_file_viewer = napari.Viewer()
    contents = [c for c in os.listdir(file_path) if not c.startswith('.')]
    for content in contents:
        if content.endswith('.tif'):
            if 'tub' in content or 'PI' in content:
                dedicated_file_viewer.add_image(imread(os.path.join(file_path, content)), name=content.split('.')[0], blending='additive', visible=False)
            else:
                dedicated_file_viewer.add_labels(imread(os.path.join(file_path, content)), name=content.split('.')[0], blending='additive')
        elif content.endswith('.txt'):
            nums = np.loadtxt(os.path.join(file_path, content))
            if nums.ndim == 1:
                dedicated_file_viewer.add_points(nums, name=content.split('.')[0], face_color='white', blending='additive')
            elif nums.ndim == 2:
                dedicated_file_viewer.add_shapes(nums, shape_type='line', name=content.split('.')[0], edge_color='white', blending='additive')
        else:
            print(f'file "{content}" not imported to viewer')

def open_with_napari(file_path: str, viewer_name) -> None:
    '''
    Open contents of a file with napari
    ---
    Parameters:
    file_path: str the path to the file to open
    ---
    Returns:
    None
    '''
    wipe_layers(viewer_name)
    cell_mask_path = os.path.join(file_path, 'curr_mask_cube.tif')
    tub_path = os.path.join(file_path, 'curr_tub_cube.tif')
    labels_path = os.path.join(file_path, 'thresh_mask.tif')
    viewer_name.add_labels(imread(cell_mask_path), name = 'cell mask', blending='additive', opacity = 0.25)
    viewer_name.add_image(imread(tub_path), name = 'tub', blending='additive')
    viewer_name.add_labels(imread(labels_path), name = 'labels', blending='additive', opacity = 0.75)

def calculate_label_properties(spindle_label_number: int) -> None:
    print(f'calculating properties for spindle {spindle_label_number}')

from contextlib import contextmanager

@contextmanager
def event_hook_removed():
    """Context manager to temporarily remove the PyQt5 input hook"""
    from qtpy import QtCore

    if hasattr(QtCore, 'pyqtRemoveInputHook'):
        QtCore.pyqtRemoveInputHook()
    try:
        yield
    finally:
        if hasattr(QtCore, 'pyqtRestoreInputHook'):
            QtCore.pyqtRestoreInputHook()

app = QApplication([])
btn = QPushButton("Get input")

@btn.clicked.connect
def _get_input():
    from magicgui.widgets import request_values

    values = request_values(
        name=dict(annotation=str, label="Enter your name:")
    )
    print(values)

main_dir = '/Volumes/bigData/wholeMount_volDist/220712_Fix_Emb_Flvw_Chn1GAP_PI_aTub647_Processed/N2V_Denoised/0_Analysis_01/0_data_cubes'
cube_paths = [] 
subdirs = [s for s in os.listdir(main_dir) if not s.startswith('.')]
for subdir in subdirs:
    cell_nums = [s for s in os.listdir(os.path.join(main_dir, subdir)) if not s.startswith('.')]
    for cell_num in cell_nums:
        folder_path = os.path.join(main_dir, subdir, cell_num)
        cube_paths.append(folder_path)

viewer = napari.Viewer(title = 'this is the cube viewer')

@viewer.bind_key('n')
def next_cube(viewer):
    wipe_layers(viewer)
    next_path = cube_paths.pop(0)
    open_with_napari(next_path, viewer)
    cube_name = "_".join(next_path.rsplit('/', 2)[1:])
    print(f'Now viewing {cube_name}')
    

btn.show()
app.exec_()

napari.run()








            