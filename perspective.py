import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def calc_warp(image_size, vanishing_point, percent_visible, width_compression):
    top_left, top_right, bottom_right, bottom_left = np.float32([
        (0, 0),
        (image_size[0], 0),
        (image_size[0], image_size[1]),
        (0, image_size[1])
    ])
    src = np.float32([
        bottom_left + (vanishing_point - bottom_left) * percent_visible,
        bottom_right + (vanishing_point - bottom_right) * percent_visible,
        bottom_right,
        bottom_left
    ])
    width = top_right[0] - top_left[0]
    compressed_width = width * width_compression
    offset = ((width - compressed_width) / 2, 0)
    dst = np.float32([
        top_left + offset,
        top_right - offset,
        bottom_right - offset,
        bottom_left + offset
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return lambda img: cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR),\
        lambda img: cv2.warpPerspective(img, Minv, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)




def main():
    forward, backward = calc_warp((1280,720), (639,419), .9, .75)
    img = mpimg.imread('test_images/straight_lines1.jpg')
    img2 = mpimg.imread('test_images/straight_lines2.jpg')
    warped = forward(img)
    warped2 = forward(img2)
    plt.subplot(211); plt.imshow(warped)
    plt.subplot(212); plt.imshow(warped2)
    plt.show()

if __name__ == '__main__':
    main()
