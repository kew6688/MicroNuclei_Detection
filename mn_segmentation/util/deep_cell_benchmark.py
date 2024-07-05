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
    
if __name__ == "__main__":
    generate_mask()