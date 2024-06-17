import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.io import imread, imsave, imshow

work_dir = "/home/y3229wan/scratch/work/KateData_yolo/"

def points_to_mask(points, img_shape):
    mask = np.zeros(img_shape, dtype=np.uint8)
    points = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)
    return mask

def main():
    img_dir = work_dir + "images/"
    label_dir = work_dir + "labels/"
    mask_dir = work_dir + "masks/"

    for file in os.listdir(img_dir):
        img_name = file.split(".")[0]
        img = imread(img_dir+file)
        img_shape = img.shape
        print(img_shape)

        # Open the file in read mode
        with open(label_dir + img_name + ".txt", 'r') as label_file:
            # Read and print each line in the file
            for line in label_file:
                l = line.strip().split()
                cls, points = l[0], l[1:]

                # class type: 
                #   4: cell_active_div, 5: cell_non_div, 6: cell_sick_apop, 7: micronuclei
                if cls == 4:
                    gt_mask = points_to_mask(points, img_shape[:-1]) 
                    labeled_mask = np.load(mask_dir + img_name + ".npy")
                    
