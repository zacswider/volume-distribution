import os
import napari
import numpy as np
import pandas as pd
from scipy import spatial
from tifffile import imread
from skimage.morphology import remove_small_objects
from skimage.measure import regionprops_table

def wipe_layers(viewer_name) -> None:
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
    component_sizes = np.bincount(labels_array.ravel()) 
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
    vv[0]: vector of the longest axis
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

def view_saved_files(file_path: str, viewer_name = None) -> None:
    ''' 
    Fxn for visualizing saved output files.
    '''
    if not viewer_name:
        dedicated_file_viewer = napari.Viewer(title='dedicated file viewer')
    else:
        dedicated_file_viewer = viewer_name

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

def open_with_napari(file_path: str, viewer_name, view_pi = False) -> None:
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
    tub_path = os.path.join(file_path, 'cubed_tub_dog_th.tif')
    labels_path = os.path.join(file_path, 'thresh_mask.tif')
    pi_path = os.path.join(file_path, 'curr_PI_cube.tif')
    if view_pi:
        viewer_name.add_labels(imread(cell_mask_path), name = 'cell mask', blending='additive', opacity = 0.125)
        viewer_name.add_image(imread(pi_path), name = 'PI', blending='additive')
    else:
        all_labels = imread(labels_path)
        all_labels = remove_small_objects(all_labels, min_size=200)
        all_labels = remove_large_objects(all_labels, max_size=5000)
        
        viewer_name.add_labels(imread(cell_mask_path), name = 'cell mask', blending='additive', opacity = 0.25)
        viewer_name.add_image(imread(tub_path), name = 'tub', blending='additive')
        viewer_name.add_image(imread(pi_path), name = 'PI', blending='additive', visible = False)
        viewer_name.add_labels(all_labels, name = 'labels', blending='additive', opacity = 0.75)

def calculate_label_properties(napari_viewer_name) -> pd.DataFrame:
    '''
    Get the label properties from the napari viewer
    ---
    Parameters:
    napari_viewer_name: variable name of the napari viewer class
    ---
    Returns:
    label_nums: np.ndarray the label numbers
    props_df: pd.DataFrame describing the area, major axis length, 
    minor axis length, label ID, distance to cell label centroid, 
    and label density.
    '''
    # get the mask coordinates and centroid
    cell_mask = napari_viewer_name.layers['cell mask'].data.astype('bool')
    mask_coords = np.column_stack(np.where(cell_mask == True))
    cell_centroid = mask_coords.mean(axis=0)

    # get the label array and label numbers
    all_labels = napari_viewer_name.layers['labels'].data
    label_nums = [l for l in np.unique(all_labels) if l != 0]
    if len(label_nums) == 0:
        return [], None
    props = regionprops_table(all_labels, cache = True, properties=('area',
                                                      'axis_major_length',
                                                      'axis_minor_length',
                                                      'solidity',
                                                      'extent',
                                                      'label'))
    props_df = pd.DataFrame(props)

    for label_num in label_nums:
        label_coords = np.column_stack(np.where(all_labels == label_num))
        label_centroid = label_coords.mean(axis=0)
        dist = spatial.distance.euclidean(cell_centroid, label_centroid)
        props_df.loc[props_df['label'] == label_num, 'dist_to_cell'] = dist
    
    props_df['aspect_ratio'] = props_df['axis_major_length'] / props_df['axis_minor_length']
    
    return label_nums, props_df

def calculate_cell_properties(napari_viewer_name) -> pd.DataFrame:
    '''
    Get the properties of the cell mask from the napari viewer
    ---
    Parameters:
    napari_viewer_name: variable name of the napari viewer class
    ---
    Returns:
    props_df: pd.DataFrame describing the properties of the active cell mask
    '''
    # get the mask coordinates and centroid
    cell_mask = napari_viewer_name.layers['cell mask'].data.astype('bool')
    cell_mask_props = regionprops_table(cell_mask.astype(np.uint8), cache=True, properties=('area',
                                                                                            'axis_major_length',
                                                                                            'axis_minor_length',
                                                                                            'solidity',
                                                                                            'extent'
                                                                                            ))
    cell_mask_props_df = pd.DataFrame(cell_mask_props)
    
    return cell_mask_props_df

def multiDimenDist(point1,point2):
   #find the difference between the two points, its really the same as below
   deltaVals = [point2[dimension]-point1[dimension] for dimension in range(len(point1))]
   runningSquared = 0
   #because the pythagarom theorm works for any dimension we can just use that
   for coOrd in deltaVals:
       runningSquared += coOrd**2
   return runningSquared**(1/2)
def findVec(point1,point2,unitSphere = False):
  #setting unitSphere to True will make the vector scaled down to a sphere with a radius one, instead of it's orginal length
  finalVector = [0 for coOrd in point1]
  for dimension, coOrd in enumerate(point1):
      #finding total differnce for that co-ordinate(x,y,z...)
      deltaCoOrd = point2[dimension]-coOrd
      #adding total difference
      finalVector[dimension] = deltaCoOrd
  if unitSphere:
      totalDist = multiDimenDist(point1,point2)
      unitVector =[]
      for dimen in finalVector:
          unitVector.append( dimen/totalDist)
      return unitVector
  else:
      return np.array(finalVector)








