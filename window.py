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
        left_conv = np.convolve(layer_hist[search_window[0]:search_window[1]], window_template, 'full')
        left_conv_peak = np.argmax(left_conv)
        if left_conv[left_conv_peak] >= thresh:
            left_peak = left_conv_peak - (window_width-1) + search_window[0] + window_width//2
            left_pt = (left_peak, start + window_height//2)
            left_pts.append(left_pt)

        #find right peak
        search_window = (np.max([0, right_peak-margin]), np.min([width, right_peak+margin]))
        right_conv = np.convolve(layer_hist[search_window[0]:search_window[1]], window_template, 'full')
        right_conv_peak = np.argmax(right_conv)
        if right_conv[right_conv_peak] >= thresh:
            right_peak = right_conv_peak - (window_width-1) + search_window[0] + window_width//2
            right_pt = (right_peak, start + window_height//2)
            right_pts.append(right_pt)

    return (np.array(left_pts), np.array(right_pts))

def draw_window(img, center, size, color):
    cv2.rectangle(img, (center[0]-size[0]//2, center[1]-size[1]//2), (center[0]+size[0]//2, center[1]+size[1]//2), color, 3)
