import cv2
import numpy as np

def sobel(gray, orient, ksize):
    if orient == 'x':
        output = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize))
    elif orient == 'y':
        output = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize))
    elif orient == 'mag':
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        output = np.sqrt(np.square(sobelx) + np.square(sobely))
    elif orient == 'dir':
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        output = np.arctan2(np.abs(sobely), np.abs(sobelx))
    return output/(np.pi/2) if orient == 'dir' else output/np.max(output)

def threshold(gray, thresh):
    output = np.zeros(gray.shape)
    output[(gray >= thresh[0]) & (gray <= thresh[1])] = 1
    return output
