import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.io import imread, imsave, imshow
import sys
sys.path.append('/content/utils')
from yolo_convert import yolo_format_points_to_mask

work_dir = "/content/work/KateData_yolo/"

def evaluate_accuracy(res, img, img_name):
    label_dir = work_dir + "labels/"
    mask_dir = work_dir + "masks/"
    deepcell_mask = np.load(mask_dir + img_name + ".npy")
    deepcell_mask = np.squeeze(deepcell_mask).astype(bool)
    # Open the file in read mode
    with open(label_dir + img_name + ".txt", 'r') as label_file:
        # Read and print each line in the file
        for line in label_file:
            l = line.strip().split()
            # class type: 
                #   4: cell_active_div, 5: cell_non_div, 6: cell_sick_apop, 7: micronuclei
            cls, points = int(l[0]), l[1:]
            if cls < 4 or cls > 7:
              continue
            labeled_mask = yolo_format_points_to_mask(points, img.shape).astype(bool)
            intersect = np.logical_and(deepcell_mask, labeled_mask)
            cover = np.sum(intersect) / np.sum(labeled_mask)
            res[cls][0] += 1
            if cover > 0.5:
              res[cls][1] += 1 
              res[cls][2] += cover
                    
def main():
    img_dir = work_dir + "images/"
    label_dir = work_dir + "labels/"
    mask_dir = work_dir + "masks/"

    res = {}
    for i in range(4,8):
      # record each class [total, matched, coverage]
      res[i] = [0,0,0]

    for file in os.listdir(img_dir):
        img_name = file.split(".")[0]
        img = imread(img_dir+file)
        img_shape = img.shape
        evaluate_accuracy(res, img, img_name)
    for i in range(4,8):
      # record each class [total, matched, coverage]
      res[i][2] /= res[i][1] if res[i][1] != 0 else 1
    print(res)
