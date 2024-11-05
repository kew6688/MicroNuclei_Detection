"""
Evaluate the performance of deep cell's pretrained segmentation model
"""

import copy
import os

import imageio
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread, imsave, imshow
from skimage.transform import resize as imresize
from skimage.color import rgb2gray
import os

from deepcell.applications import NuclearSegmentation, CellTracking
from deepcell_tracking.trk_io import load_trks

from yolo_format_convert import yolo_format_points_to_mask

# path variables
home_dir = ""
img_dir = home_dir + "KateData_yolo/images/"
label_dir = home_dir + "KateData_yolo/labels/"
mask_dir = home_dir + "KateData_yolo/masks/"
# os.mkdir(mask_dir)

os.environ.update({"DEEPCELL_ACCESS_TOKEN": "nCDMTpLD.XlCU4Emtpd3FCubSnZqmrCoQo7wzGHuz"})


def generate_mask():
    """generate segmentation mask from pretrained deepcell model

    Args: 
        None

    Returns: 
        None
    """
    app = NuclearSegmentation()
    print('Training Resolution:', app.model_mpp, 'microns per pixel')

    for file in os.listdir(img_dir):
        file_name = file.split(".")[0]
        img = np.expand_dims(imread(img_dir + file), axis=0)
        img = rgb2gray(img)
        img = np.expand_dims(img, axis=3)
        print(img.dtype, img.shape)  
        y_pred = app.predict(img)
        print(y_pred.shape)
        np.save(mask_dir+file_name+".npy", y_pred)


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

              # save examples for dividing and mn
              if cls == 4:
                np.save("/content/gdrive/MyDrive/PMCC/mislabels/dividing/" + img_name + ".npy", labeled_mask)
              if cls == 7:
                np.save("/content/gdrive/MyDrive/PMCC/mislabels/mn/" + img_name + ".npy", labeled_mask)

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

if __name__ == "__main__":
    generate_mask()