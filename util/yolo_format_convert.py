import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.io import imread, imsave, imshow

def points_to_mask(points, img_shape):
    mask = np.zeros(img_shape, dtype=np.uint8)
    points = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)
    return mask

def yolo_format_points_to_mask(points, img_shape):
  """Yolo dataset has this format: <class-index> <x1> <y1> <x2> <y2>
   ... <xn> <yn>, we need to convert to the points taken by opencv.

   Args:
    points read from txt file, list(str), this is in image view
    img_shape typle (x,y,c)

   Returns:
    points in list [[x1, y1], [x2, y2], ...]
  """
  # change to float
  points = np.array(points).astype(float)

  # reshape to coordinates
  points = points.reshape(-1,2)

  xy_length = np.array([img_shape[1], img_shape[0]])
  points = points*xy_length
  mask = points_to_mask(points, img_shape[:2])
  return mask