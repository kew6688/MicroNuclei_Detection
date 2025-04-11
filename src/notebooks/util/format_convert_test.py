import copy
import os

import imageio
import matplotlib
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

def mask_test():
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
    
def fillpolygon_full_test():
  # Test display one filled polygon on the original image

  data_dir = 'data/'
  img_path = "/content/data/images/10 Gy_GFP-H2B_A1_1_2023y06m24d_18h17m.png"

  # Define an array of endpoints of Hexagon
  points = np.array([[220, 120], [130, 200], [130, 300],
                    [220, 380], [310, 300], [310, 200]])

  label_path = "/content/data/labels/10 Gy_GFP-H2B_A1_1_2023y06m24d_18h17m.txt"

  _, img = load_img(img_path)
  points = load_polys(label_path)[0]
  print(points)
  x,y = img.shape[1], img.shape[0]
  xy_length = np.array([img.shape[1], img.shape[0]])
  points = points*xy_length
  points = points.astype(np.int32)
  # points = change_origin(np.array([0,0]), points)
  # print(points)
  # buf = ["1 " + " ".join(map(str, points.flatten())), "1 " + " ".join(map(str, points.flatten()))]
  # with open('points.txt',"w") as f:
  #     f.write("\n".join(buf))
  # np.savetxt('points.txt', points.flatten(), fmt="%f", newline=" ")
  display(img, points)

def fillpolygon_crop_test():
  # Test display one filled polygon on the cropped image
  img_path = "/content/dest/images/10 Gy_GFP-H2B_A1_1_2023y06m24d_18h17m_0.png"
  label_path = "/content/dest/labels/10 Gy_GFP-H2B_A1_1_2023y06m24d_18h17m_0.txt"

  _, img = load_img(img_path)
  points = load_polys(label_path, 1)
  print(points)

  x,y = img.shape[1], img.shape[0]
  xy_length = np.array([img.shape[1], img.shape[0]])
  points = points*xy_length
  points = points.astype(np.int32)
  display(img, points)