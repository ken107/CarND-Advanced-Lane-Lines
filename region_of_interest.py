import numpy as np
import cv2

clip_mask = None
vertices = [(0,520), (560,440), (720,440), (1280,520), (1280,720), (0,720)]


def clip(img):
    global clip_mask
    if (clip_mask is None):
        clip_mask = make_clip_mask(img, vertices)
    return cv2.bitwise_and(img, clip_mask)


def make_clip_mask(img, vertices):
  mask = np.zeros_like(img)
  if len(img.shape) > 2:
      channel_count = img.shape[2]
      ignore_mask_color = (255,) * channel_count
  else:
      ignore_mask_color = 255
  cv2.fillPoly(mask, np.array([vertices]), ignore_mask_color)
  return mask
