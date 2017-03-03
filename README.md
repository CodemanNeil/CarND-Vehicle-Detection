#Vehicle Detection

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./output_images/hog_features.png
[image3]: ./output_images/subsampling_windows.png
[image4a]: ./output_images/sliding_windows1.png
[image4b]: ./output_images/sliding_windows2.png
[image4c]: ./output_images/sliding_windows3.png
[image5a]: ./output_images/hog_subsampling1.png
[image5b]: ./output_images/hog_subsampling2.png
[image5c]: ./output_images/hog_subsampling3.png
[image5d]: ./output_images/hog_subsampling4.png
[image5e]: ./output_images/hog_subsampling5.png
[image5f]: ./output_images/hog_subsampling6.png
[image6]: ./examples/labels_map.png
[image7]: ./output_images/labeled_cars.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook.  Specifically, it's located in the `get_hog_features` method.  In the code cell below the title "Get Data", there is code that reads in all the filenames of the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters.  I was able to achieve decent results with only the first channel ('Y') when in YCrCb color space, but still found ALL channels to be superior in minimizing false positives.  Increasing the orientation from 9 to 18 also improved my accuracy.  I tried increasing the number of cells per block to 3 for HOG normalization, but saw no improvement and an increase in computation time.  Eventually I settled on the following:

```python
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 18  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
```

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using `LinearSVC` and a C value of 1.  I tried several different values for C (0.1,1,10,100), but found the default value of 1 to be most accurate.  The code for this can be found in the cell block under the title "Data Extraction & Fit".  I was able to achieve >99% accuracy on the validation set.  

I tried unsuccessfully to mitigate any issues from the data set containing time series data.  My attempts to manually split the GTI data (containing some time series data) resulted in worse performance by the SVM despite varied C parameters.  I may try to come back to this at a later time.  

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I initially created a naive window search using window sizes of (128,128),(96,96), and (64,64).  These windows slided across the entire horizontal axis, but only a portion of the vertical axis.  The corresponding part of the vertical axis for each of the window sizes was (2/5,4/5),(2/5,3/4),(2/5,2/3).  To clarify, using the 128*128 window as an example, that window scanned the image from 2/5 of the way down the image to 4/5 of the way down the image, across the entire horizontal axis.  This achieved good results but was quite slow because every window required a call to get the HOG features. The code for this can be found in the code cell under the title "Naive Window Search".

After this naive approach, I use HOG subsampling to speed up the implementation.  This meant that HOG was called once on the image (cropped to just the portion we were interested in), and then that result was subsampled at different scales, corresponding to different window sizes being slid across the image.  The HOG features extracted this way were combined with the spatial and color hist features.  The code for this can be found in the code cell under the title "Hog Subsampling".  Here's an image of the windows searched using subsampling:

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Tuning the feature extraction parameters mentioned above, along with the hog subsampling scales to [4/3,3/2,2], gave me good results with minimal false positives. Here are some example images:

![alt text][image4a]
![alt text][image4b]
![alt text][image4c]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

In the method `find_heatmap_and_drawn_images` under "Hog Subsampling" a heatmap and image with all overlapping windows of positive results is returned.  This is then aggregated across the last 4 images, and an average heatmap is created.  This is then tresholded, and the `labels` method is used in the `draw_labeled_boxes` method to create a final box drawn around each car.  By aggregating and tresholding, we can remove the vast number of false positives.

### Here are six frames and their corresponding heatmaps:

![alt text][image5a]
![alt text][image5b]
![alt text][image5c]
![alt text][image5d]
![alt text][image5e]
![alt text][image5f]


### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I was unable to mitigate any issues that may have arised due to the time series nature of some of the training data.  Tuning the parameters also proved to be extremely slow.  

I've noticed that the pipeline tends to have some false positives with guard rail in shadow, which could cause some issues.  Also, since it uses the last 4 images to create an average heatmap to judge, it can tend to trail the actual car it's tracking.  This is usually not a big deal for cars traveling in the same direction as they don't move quickly relative to our own car.  However, for cars travelling in the opposite direction, the box lags behind the vehicle when detected, and can appear to be a false positive at first glance.

Also, since I'm restricting the search to only a slice of the image, if a car resides outside of that slice (if the camera is mounted at a different angle or we're driving uphill or downhill), a car may not be detected. 

To be more robust, we would want to use more varied scaled windows and have those windows cover a larger portion of the image.  To counter the increase in false positives we'd see if we used many more windows, we would also need to collect more training samples to improve the accuracy of the classifier.  This may also be a good application for a CNN, though I haven't explored this.

