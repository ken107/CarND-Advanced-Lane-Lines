import numpy as np
import cv2

def make_window_template(window_width, window_height):
    wing = np.arange(window_width//2)
    span = np.concatenate([wing, wing[::-1]])
    return span

def find_lane_points(binary_warped, window_size, prev_pts, margin, thresh):
    height, width = binary_warped.shape
    window_width, window_height = window_size
    window_template = make_window_template(window_width, window_height)

    #find starting peaks
    if prev_pts is None:
        histogram = np.sum(binary_warped[height//2:], axis=0)
        left_hist, right_hist = np.split(histogram, 2)
        left_conv, right_conv = np.convolve(left_hist, window_template, 'valid'), np.convolve(right_hist, window_template, 'valid')
        left_peak, right_peak = np.argmax(left_conv) + window_width//2, np.argmax(right_conv) + width//2 + window_width//2
        print('start peaks', left_peak, right_peak)
    else:
        left_peak, right_peak = prev_pts[0][0,0], prev_pts[1][0,0]

    #find lane points
    left_pts = []
    right_pts = []
    num_layers = height // window_height
    for i in range(num_layers):
        #get layer hist
        end = height - i * window_height
        start = end - window_height
        layer = binary_warped[start:end]
        layer_hist = np.sum(layer, axis=0)

        #find left peak
        search_window = (np.max([0, left_peak-margin]), np.min([width, left_peak+margin]))
        left_conv = np.convolve(layer_hist[search_window[0]:search_window[1]], window_template, 'valid')
        new_left_peak = np.argmax(left_conv)
        if left_conv[new_left_peak] >= thresh:
            left_peak = new_left_peak + search_window[0] + window_width//2
            left_pt = (left_peak, start + window_height//2)
            left_pts.append(left_pt)

        #find right peak
        search_window = (np.max([0, right_peak-margin]), np.min([width, right_peak+margin]))
        right_conv = np.convolve(layer_hist[search_window[0]:search_window[1]], window_template, 'valid')
        new_right_peak = np.argmax(right_conv)
        if right_conv[new_right_peak] >= thresh:
            right_peak = new_right_peak + search_window[0] + window_width//2
            right_pt = (right_peak, start + window_height//2)
            right_pts.append(right_pt)

    return (np.array(left_pts), np.array(right_pts))

def draw_window(img, center, size, color):
    cv2.rectangle(img, (center[0]-size[0]//2, center[1]-size[1]//2), (center[0]+size[0]//2, center[1]+size[1]//2), color, 3)

def draw_lane(img, lane_pts, color):
    left_pts, right_pts = lane_pts
    left_fit = np.polyfit(left_pts[:,1], left_pts[:,0], 2)
    right_fit = np.polyfit(right_pts[:,1], right_pts[:,0], 2)
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    for y, left_x, right_x in zip(ploty, left_fitx, right_fitx):
        img[y, left_x:right_x] = color

def print_stats(img, lane_pts, xm_per_pix, ym_per_pix, color):
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
