# Volume Distribution

#### This repository contains a collection of packages and pipelines for viewing and fully automating the volumetric segmentations of *X. laevis* embryos. For this project I was specifically interested in quantifying the spatial location of the mitotic spindle within the cell volume as well as its alignment with the long axis of the cell. This readme describes some of the methods that I used to a) segment 3D cell volumes, b) filter poor quality segmentations, and c) measure the spatial properties of the cell and spindle.

### Upstream processing

The images from this project were acquired at low resolution on an Olympuc Fv1000 confocal, denoised using Noise2Void, and segmented in three dimensions using a custom [Cellpose](https://github.com/mouseland/cellpose) model trained with ground-truth annotations of XY, XZ, and YZ slices from 3D datasets:

<img src="https://github.com/zacswider/README_Images/blob/main/cellpose_XY_combined.png" width="500">

### Quality control

Convolutional neural networks like cellpose can be trained to *very* high accuracy. However, 3D segmentation is especially challenging, especial for data like these, where the Z resolution is ~15x lower than the XY resolution.

<img src="https://github.com/zacswider/README_Images/blob/main/embryo_orthoviews.png" width="500">

Given this limitation, segmentations can be imperfect and should be checked for accuracy. However, manual correction is low-throughput and inappropriate for large datasets (imagine hundreds or thousands of stacks). To safeguard against inappropriate segmentations we can implement low effort filters like minimum and maximum segmentation sizes that encompass the expected distribution of our cells. But what about labels that are within the right size range, but look like this? (spoiler, this is a *bad* segmentation) 

<img src="https://github.com/zacswider/README_Images/blob/main/example_bad_seg.png" width="500">

To eliminate appropriately sized but inappropriately shaped labels we can take a more sophisticated filtering approach, like training a random forest classifier to discriminate between "good" and "bad" segmentations. Here I use the n-dimensional viewer [napari](https://napari.org/stable/) to quickly scrub through cell segmentations which can easily be classified as "good" or "bad" with a key press. Once the ground truth class has been specified, the spatial properties of the label (e.g., long/short axis length and label density/solidity) are used to train the classifier. 

<img src="https://github.com/zacswider/README_Images/blob/main/training_rf_model.gif" width="500">

After filtering, we are left with only cell shapes that match the data we used to train the classifier . Since I'm only interested in these relatively small epithelial cells, I trained the classifier against everything else.

<img src="https://github.com/zacswider/README_Images/blob/main/rotating_masks.gif" width="500">

### Isolating spindles

In principle, we could also train a segmentation model to specifically threshold spindle shapes, but in this case classic thresholding worked well. In order to avoid applying a blanket threshold value to the entire image, where bright and dim cells would be segmented unequally, we isolate individual cell masks, run a 3D tophat filter to increase contast, and calculate cell-specific threshold values. 

<img src="https://github.com/zacswider/README_Images/blob/main/spindle_thresh.gif" width="500">

Similar to our approach to cell segmentations, we can filter the non-specific regions with a random forest classifier trained to recognize spindle shapes.

###  Measuring cell properties

Given isolated masks, we can easily measure 3D label properties with built-in python libraries like [scikit-image](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops), or take custom approaches like in the example below. Here we use principle component analysis to quantify interesting metrics like the degree of alignment between the long axis of the spindle and the long axis of the cell.

<img src="https://github.com/zacswider/README_Images/blob/main/mask_alignment.gif" width="500">

Please check back soon, I am in the process of updating this repository.








