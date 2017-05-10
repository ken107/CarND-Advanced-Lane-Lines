import os
import pickle
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def calc_undistort(calibration_image_glob, pattern_size, input_image_size, save_file):
    if os.path.isfile(save_file):
        with open(save_file, 'rb') as f:
            mat, coef = pickle.load(f)
    else:
        objp = np.zeros((np.prod(pattern_size),3), np.float32)
        objp[:,:-1] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape((-1,2))
        imgpoints = []
        objpoints = []
        for path in glob.glob(calibration_image_glob):
            gray = cv2.imread(path, 0)
            ret, corners = cv2.findChessboardCorners(gray, pattern_size)
            if ret == True:
                imgpoints.append(corners.squeeze())
                objpoints.append(objp)
        ret, mat, coef, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, input_image_size, None, None)
        with open(saveFile, 'wb') as f:
            pickle.dump([mat, coef], f)
    return lambda img: cv2.undistort(img, mat, coef)





def main():
    img = mpimg.imread('camera_cal/calibration2.jpg')
    undistort = calc_undistort('camera_cal/calibration*.jpg', (9,6), (img.shape[1], img.shape[0]), 'save/undistort.p')
    result = undistort(img)
    plt.subplot(121); plt.imshow(img)
    plt.subplot(122); plt.imshow(result)
    plt.show()

if __name__ == '__main__':
    main()
