import cv2
import numpy as np


def draw_lane(img, lane_pts, color):
    left_pts, right_pts = lane_pts
    left_fit = np.polyfit(left_pts[:,1], left_pts[:,0], 2)
    right_fit = np.polyfit(right_pts[:,1], right_pts[:,0], 2)
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    for y, left_x, right_x in zip(ploty, left_fitx, right_fitx):
        img[y, left_x:right_x] = color


def print_curve_stats(img, lane_pts, lane_width_m, ym_per_pix, color):
    #calculate xm_per_pix
    left_pt, right_pt = lane_pts[0][0], lane_pts[1][0]
    xm_per_pix = lane_width_m / (right_pt[0] - left_pt[0])

    #distance from center (assuming image center is car center -- true if camera mounted at center and if perspective warp is symmetrical)
    lane_center_x = (left_pt[0] + right_pt[0]) / 2
    off_center_x = (img.shape[1] / 2) - lane_center_x
    off_center_m = off_center_x * xm_per_pix
    cv2.putText(img, 'Off Center: {:.2f}m'.format(off_center_m), (10,110), cv2.FONT_HERSHEY_PLAIN, 2, color)

    #curve radius
    left_pts = lane_pts[0] * (xm_per_pix, ym_per_pix)
    right_pts = lane_pts[1] * (xm_per_pix, ym_per_pix)
    left_fit = np.polyfit(left_pts[:,1], left_pts[:,0], 2)
    right_fit = np.polyfit(right_pts[:,1], right_pts[:,0], 2)
    y_eval = left_pts[0,1]
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    cv2.putText(img, 'Left Radius: {:.0f}m'.format(left_curverad), (10,50), cv2.FONT_HERSHEY_PLAIN, 2, color)
    cv2.putText(img, 'Right Radius: {:.0f}m'.format(right_curverad), (10,80), cv2.FONT_HERSHEY_PLAIN, 2, color)
