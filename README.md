# Volume Distribution

#### This repository contains a collection of packages and pipelines for viewing and automating the volumetric segmentations of *X. laevis* embryos. For this project I was specifically interested in quantifying the spatial location and orientation of the mitotic spindle with respect to the cell volume and orientation. This readme describes some of the methods that I used to a) segment 3D cell volumes, b) filter poor quality segmentations, and c) measure the spatial properties of the cell and spindle.

### Upstream processing

The images from this project were acquired at low resolution on an Olympus Fv1000 confocal, denoised using Noise2Void, and segmented in three dimensions using a custom [Cellpose](https://github.com/mouseland/cellpose) model trained with ground-truth annotations of XY, XZ, and YZ slices from 3D datasets.

###### The image below shows a raw XY slice on the left with human annotated cells on the right: 
<img src="https://github.com/zacswider/README_Images/blob/main/cellpose_XY_combined.png" width="650">

###### The image below shows a 3D projection of PI and anti-tubulin staining on the left and the cellpose-calculated cell labels on the right: 
<img src="https://github.com/zacswider/README_Images/blob/main/napari_cube_combined-small.png" width="800">

### Quality control

3D segmentation is challenging, especial for data like these where the Z resolution is ~15x lower than the XY resolution. Given this limitation, segmentations can be imperfect and should be checked for accuracy. However, manual correction is low-throughput and inappropriate for large datasets (imagine hundreds or thousands of stacks). To safeguard against inappropriate segmentations we can implement low effort filters like minimum and maximum segmentation sizes that encompass the expected distribution of our cells. But what about labels that are within the right size range (such as the one below) but completely inaccurate? 

###### This image below shows a mask that passes the size filters, but is complete garbage
<img src="https://github.com/zacswider/README_Images/blob/main/example_bad_seg.png" width="500">

To eliminate appropriately sized but inappropriately shaped labels we can take a slightly slower but more sophisticated filtering approach. Here I train a random forest classifier to discriminate between good and bad segmentations. The classifier learns uses the 3 dimensional properties (size, shape, density, etc) of labels with known classes (good or bad) to decide whether new labels are either good or bad. Generating training data for a random classifier is made easy with open source tools like [napari](https://napari.org/stable/), a multi-dimensional image viewer that can be paired with 3D measurement tools like [scikit-image](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops) and machine learning libraries like [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html). 

###### In the example below I use napari to to quickly scrub through cell segmentations which can be classified as "good" or "bad" with a key press. Once the class has been specified, the spatial properties of the label (e.g., long/short axis length and label density/solidity) are measured and used to train the classifier. 
<img src="https://github.com/zacswider/README_Images/blob/main/test_rf_downsample_gifski_xsmall.gif" width="500">

After filtering, we are left with only cell shapes that match the data we used to train the classifier . Since I'm only interested in these relatively small epithelial cells, I trained the classifier against everything else. 

###### The example below shows filtered labeled on the right, and the corresponding masked immunofluorescence on the left
<img src="https://github.com/zacswider/README_Images/blob/main/rotating_masks_small.gif" width="500">

### Isolating spindles

In principle, we could also train a segmentation model to specifically threshold spindle shapes, but in this case classic thresholding worked well. In order to avoid applying a blanket threshold value to the entire image, where bright and dim cells would be segmented unequally, we isolate individual cell masks, run a 3D tophat filter to increase contast, and calculate cell-specific threshold values. 

###### The example below shows raw anti-tubulin immunofluorescence and the correponding thresholded regions.
<img src="https://github.com/zacswider/README_Images/blob/main/spindle_thresh_small.gif" width="500">

Similar to our approach to cell segmentations, we end up with a mix of accurate and inaccurate segmentations. Once again, we can filter the non-specific regions with a random forest classifier trained to recognize spindle shapes.


###### Below left: all labels following thresholding and connected component labeling. Below right: all labels after filtering with a random forest classifier trained to recognize spindle shapes.
<img src="https://github.com/zacswider/README_Images/blob/main/spindle_rf_filtering_3panel.png" width="1000">

In aggregate, this approach produces high quality segmentations from the majority of cells in a given embryo:

###### Example of cell and spindle thresholding
<img src="https://github.com/zacswider/README_Images/blob/main/segmentation.png" width="1000">

###  Measuring cell properties

Given isolated masks, we can easily measure 3D label properties with built-in python libraries like [scikit-image](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops). In this case I am more interested in measuring the relationships between shapes, such as distance and relative alignment.

###### Here we measure the distance between the spindle and the center of the cell, as well as the degree of alignment between the long axis of the spindle and the long axis of the cell.
<img src="https://github.com/zacswider/README_Images/blob/main/spindle%20aligment.png" width="1000">

Proof of principle: automated analysis of 350 cells from 15 different embryos reveals that mitotic spindles follow [Hertwig's Rule](https://en.wikipedia.org/wiki/Hertwig_rule) and align perpendicular to the short axis of the cell. Deviations from this rule tend to be found in cells with a small aspect ratio (i.e., more spherical cells), where pressure to align is not as strong. 

###### On average, mitotic spindles in the _X. laevis_ epithelium tend to align perpendicular to the short axis of the cell. This trend is strongest when cell shape deviates from spherical.

<img src="https://github.com/zacswider/README_Images/blob/main/spindle_alignment_graph.png" width="350">

We can make similar measurements of tissue-level organization by measuring the alignment of cell volumes relative to the geometric center of the embryo:

###### Cells tend to align with their long axis along the plane of the epithelium (perpendicular to a vector towards the center of the embryo), however, this morphology can be disturbed under certain conditions:

<img src="https://github.com/zacswider/README_Images/blob/main/cell_alignment.png" width="800">



