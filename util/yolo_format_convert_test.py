import copy
import os

import imageio
import matplotlib
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

file = "10 Gy_GFP-H2B_A1_1_2023y06m24d_18h17m"
img = imread("Data/images/" + file + ".png")
mask = np.load("Data/masks/" + file + ".npy")
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(img[330:380, 630:680])
ax[0].axis('off')
ax[0].set_title('Raw')
img_shape = img.shape
with open("Data/labels/" + file + ".txt", 'r') as label_file:
    # Read and print each line in the file
    for line in label_file:
        l = line.strip().split()
        cls, points = l[0], l[1:]
        if cls == "7":
          continue
        points = np.array(points).astype(float)
        points = points.reshape(-1,2)
        xy_length = np.array([img_shape[1], img_shape[0]])
        points = points*xy_length
        print(points)
        mask = points_to_mask(points, img_shape[:2])
        print(mask.shape)
        img_covered = copy.deepcopy(img)
        # print(np.where(mask == 255))
        img_covered[mask != 255, :] = [0, 0, 0]
        ax[1].imshow(img_covered[330:380, 630:680])
        ax[1].set_title('Segmented')
        ax[1].axis('off')
        break