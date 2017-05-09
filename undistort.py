import os
import pickle
import glob
import cv2
import numpy as np

imageGlob = 'camera_cal/calibration*.jpg'
patternWh = (9,6)
saveFile = 'save/undistort.p'

def calc_undistort(imageWh):
    if os.path.isfile(saveFile):
        with open(saveFile, 'rb') as f:
            mat, coef = pickle.load(f)
    else:
        objp = np.zeros((np.prod(patternWh),3), np.float32)
        objp[:,:-1] = np.mgrid[0:patternWh[0], 0:patternWh[1]].T.reshape((-1,2))
        imgpoints = []
        objpoints = []
        for path in glob.glob(imageGlob):
            gray = cv2.imread(path, 0)
            ret, corners = cv2.findChessboardCorners(gray, patternWh)
            if ret == True:
                imgpoints.append(corners.squeeze())
                objpoints.append(objp)
        ret, mat, coef, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imageWh, None, None)
        with open(saveFile, 'wb') as f:
            pickle.dump([mat, coef], f)
    return lambda img: cv2.undistort(img, mat, coef)





def main():
    gray = cv2.imread('camera_cal/calibration10.jpg', 0)
    undistort = calc_undistort(gray.shape[::-1])
    result = undistort(gray)
    cv2.imshow('', result)
    cv2.waitKey()

if __name__ == '__main__':
    main()
