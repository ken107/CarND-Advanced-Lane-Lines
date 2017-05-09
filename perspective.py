import cv2
import numpy as np

src = np.float32([(605,444), (675,444), (1040,677), (266,677)])
dst = np.float32([(366,0), (940,0), (940,720), (366,720)])
#src = np.float32([(593,487), (733,487), (1097,692), (355,692)])
#dst = np.float32([(366,0), (940,0), (940,720), (366,720)])

def calc_perspective():
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return lambda img: cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR),\
        lambda img: cv2.warpPerspective(img, Minv, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)




def main():
    forward, backward = calc_perspective()
    img = cv2.imread('test_images/straight_lines1.jpg')
    for i in range(0,4):
        cv2.line(img, tuple(src[i]), tuple(src[(i+1)%4]), (255,0,0), 1)
    result = forward(img)
    cv2.imshow('', img)
    cv2.waitKey()
    cv2.imshow('', result)
    cv2.waitKey()

if __name__ == '__main__':
    main()
