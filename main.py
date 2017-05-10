import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from undistort import calc_undistort
from region_of_interest import clip
from perspective import calc_warp
from threshold import sobel, threshold
from window import find_lane_points, draw_window
from curve import draw_lane, print_curve_stats

undistort = calc_undistort(calibration_image_glob='camera_cal/calibration*.jpg', pattern_size=(9,6), input_image_size=(1280,720), save_file='save/undistort.p')
warp_forward, warp_backward = calc_warp(image_size=(1280,720), vanishing_point=(639,419), percent_visible=.9, width_compression=.7)
clahe = cv2.createCLAHE(tileGridSize=(30,30), clipLimit=2)
lane_pts = None


def to_rgb(gray):
    gray = np.uint8(255.0 * gray / np.max(gray))
    return np.tile(gray[:,:,None], (1,1,3))

def process(img):
    global lane_pts
    undist = undistort(img)
    clipped = clip(undist)
    red = clipped[:,:,0]
    red = cv2.normalize(red, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    contrast_enhanced = clahe.apply(red)
    high_red = threshold(contrast_enhanced, thresh=(210,255))
    edges = threshold(sobel(contrast_enhanced, orient='x', ksize=7), thresh=(0.15,1))
    combined = cv2.bitwise_or(high_red, edges)
    #return to_rgb(combined)
    binary_warped = warp_forward(combined)
    #return to_rgb(binary_warped)
    lane_pts = find_lane_points(binary_warped, window_size=(100,60), prev_pts=lane_pts, margin=100, thresh=3000)
    lane_warped = np.zeros(img.shape, img.dtype)
    draw_lane(lane_warped, lane_pts, (0,255,0))
    #for pt in lane_pts[0]: draw_window(lane_warped, pt, (100,60), (255,0,0))
    #for pt in lane_pts[1]: draw_window(lane_warped, pt, (100,60), (0,0,255))
    #return cv2.addWeighted(to_rgb(binary_warped), .5, lane_warped, .5, 0)
    lane_img = warp_backward(lane_warped)
    final = cv2.addWeighted(undist, 1, lane_img, .3, 0)
    print_curve_stats(final, lane_pts, lane_width_m=3.7, ym_per_pix=30.0/720, color=(255,255,255))
    return final



### IMAGE
img = mpimg.imread('test_images/test5.jpg')
#clip1 = VideoFileClip("project_video.mp4")
#img = clip1.get_frame(40.5)
#plt.imsave('test_images/challenge_straight.jpg', img)
result = process(img)
#img2 = clip1.get_frame(40.7)
#result2 = process(img2)
plt.imshow(result)
plt.show()


### VIDEO
#clip1 = VideoFileClip("project_video.mp4")
#clip2 = clip1.fl_image(process)
#clip2.write_videofile("R:/project_video_birds_eye.mp4", audio=False)
