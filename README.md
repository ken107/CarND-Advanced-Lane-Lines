## Writeup Template

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort.png "Undistorted"
[image2]: ./output_images/undistort2.png "Road Undistorted"
[image3]: ./output_images/binary_combo.png "Binary Example"
[image4]: ./output_images/warp.png "Warp Example"
[image7]: ./output_images/binary_warped.png "Binary Warped"
[image8]: ./output_images/windows.png "Windows Visualized"
[image5]: ./output_images/color_fit_lines.png "Fit Visual"
[image6]: ./output_images/example_output.png "Output"
[video1]: https://youtu.be/1XX50QZzn74 "Video"

### Code Organization

My code consists of the following files:

* _main.py_: defines the image processing pipeline, loads and process the video
* _region_of_interest.py_: clipping functions
* _undistort.py_: camera calibration functions
* _perspective.py_: perspective calculation & warping
* _threshold.py_: sobel & threshold functions
* _window.py_: lane points detection using window convolution
* _curve.py_: curve fitting and radius calculation & drawing

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in _undistort.py_, function `calc_undistort()`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I save the coefficients to a file so that I don't have to recalibrate it next time.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

I applied distortion correction to the image _test_images/test5.jpg_ and obtained this result:

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code for this step is found in _main.py_ line 26-32, and in _threshold.py_.

First I apply a region_of_interest mask to remove scenery and keep only the road.  Then using only the red channel, I normalize it and apply CLAHE to enhance contrast between the lane and the road, making it easier to detect edges.  Then I: 1) threshold the contrast-enhanced image, 2) threshold the Sobel gradient along the x direction, and combine the results to get my binary image.  As example, when applied to _test_images/test5.jpg_ I get the following result:

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for this step is found in _main.py_ line 14,34 and in _perspective.py_.

The `calc_warp()` function takes in the image dimension and the vanishing point, and auto-calculate the 4 points on the road surface.  These 4 points are then warped into the 4 corners of the image.  The two additional parameters controls how much will be visible horizontally & vertically in the resulting warped image.

I verified that my perspective transform was working as expected by drawing two vanishing lines onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

When applied to the binary threshold image from step 2, I get the following:

![alt text][image7]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code for this step is found in _main.py_ line 36-41, _window.py_, and _curve.py_ line 5.

From the warped binary image in step 3, I apply the window-template convolution method to identify the lane lines.  First I generate a window template that looks like 123454321 that encourages centering the lane in the window.  Then I convolve the template with the histogram of the bottom half of the image.  The resulting left and right peaks are my starting search positions.  Then for each layer starting from the bottom, I convolve the template with the histogram of the layer, the resulting peaks are added to my lane_points array.  Actually instead of convolving with the whole layer, I convolve with only a section ±margin around the peaks found in the previous layer (since we expect the lane to be contiguous).

![alt text][image8]

Using `cv2.polyfit`, I fit a second degree polynomial to the detected lane points.  The result looks like:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code for this step is in `main.py` line 44 and `curve.py` lines 16.

While `ym_per_pix` needs to be specified, I calculate `xm_per_pix` automatically by knowing that lanes are 3.7m apart in California.  Then using these two scalars, I calculate how much the vehicle is off center by considering the difference between the detected lane center and the center of the image.  This assumes the camera is center-mounted, and that the perspective warp preserves horizontal positioning.

To calculate lane curvature, I convert lane points to world coordinates (meter), perform a polyfit, and use the resulting poly coefficients in the curvature radius formula.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The code for this step is at _main.py_ line 42.

I use the inverse perspective warp to warp the drawn lane image back into road-perspective, and combine it with the original image. Here is the result:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://youtu.be/1XX50QZzn74)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main problem I had is filtering out the lane lines.  I didn't use the S channel in HLS because I noticed that in the challenge video the saturation of the lane markings are not high enough to be thresholded.  In the harder_challenge video, some parts had the lane marking completely washed out by the bright sun, and I have no idea how to handle that.  That aside, in the end I decided that the best way is to rely primarily on Sobel edge detection, since that anyway is the way we "see" the lanes.  I came up with the idea of using CLAHE on the road surface to enhance the edges to help the edge detection.  But even with that, the thresholds had to be carefully chosen to avoid detecting too much road noise.  And instead of doing this on the grayscale image, I decided to do it on just the red channel, since both Yellow and White lane lines have strong red components.  Nonetheless in the end it required a very fine balance of hyperparameter settings to detect lanes correctly just in the project video.  In the challenge video my pipeline misdetects the center divider and road noise because those sometimes have stronger contrast/edges than the current lane.  I did not have time to come up with solution for this.

I had a few ideas, such as performing a histogram on the Sobel gradients to filter out noise, but it didn't work out very well.  And using an increasing margin for upper layers when doing window detection.
