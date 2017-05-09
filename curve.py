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

def print_curve_stats(img, lane_pts, xm_per_pix, ym_per_pix, color):
    left_pts = lane_pts[0] * (xm_per_pix, ym_per_pix)
    right_pts = lane_pts[1] * (xm_per_pix, ym_per_pix)
    left_fit = np.polyfit(left_pts[:,1], left_pts[:,0], 2)
    right_fit = np.polyfit(right_pts[:,1], right_pts[:,0], 2)
    y_eval = left_pts[0,1]
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    cv2.putText(img, 'Left: {:.2f}'.format(left_curverad), (10,50), cv2.FONT_HERSHEY_PLAIN, 2, color)
    cv2.putText(img, 'Right: {:.2f}'.format(right_curverad), (10,80), cv2.FONT_HERSHEY_PLAIN, 2, color)
    off = (left_pts[0,0] + right_pts[0,0]) / 2 - (img.shape[1] * xm_per_pix)
    cv2.putText(img, 'Off: {:.2f}'.format(off), (10,110), cv2.FONT_HERSHEY_PLAIN, 2, color)
