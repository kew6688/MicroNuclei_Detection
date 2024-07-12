import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.io import imread, imsave, imshow
from PIL import Image

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
   
def mask_to_points(mask):
  contours, _ = cv2.findContours(a, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
  polygons = []

  for obj in contours:
      coords = []

      for point in obj:
          coords.append(int(point[0][0]))
          coords.append(int(point[0][1]))

      polygons.append(coords)

def display_contour(img, contours):
  print(contours[0].squeeze().shape, img.shape)
  plt.imshow(img)
  poly = contours[0].squeeze()
  np.append(poly, poly[0])
  x,y = zip(*poly)
  plt.plot(x,y)
  # plt.contour(contours[0].squeeze())



def load_img(img_path):
  """
  Read an image and convert it to RGB format.

  Args:
    img_path (str): Path to the image file.

  Returns:
    numpy.ndarray: RGB image array from cv2.imread.
  """

  img = Image.open(img_path)
  img_arr = np.array(img)
  return img, img_arr

def load_polys(label_path, target_clas = 7):
  """
  Read a label txt file and convert it to numpy array.

  Args:
    label_path (str): Path to the label text file.

  Returns:
    numpy.ndarray: polygon array that each polygon contains an array of coordinates ([[x,y],...]).
  """

  poly_arr = []
  with open(label_path, 'r') as label_file:
    # Read and print each line in the file
    for line in label_file:
      l = line.strip().split()
      # class type:
          #   4: cell_active_div, 5: cell_non_div, 6: cell_sick_apop, 7: micronuclei
      cls, points = int(l[0]), l[1:]

      # current focus on mn
      if cls != target_clas:
        continue

      points = np.array(points, dtype=np.float64).reshape((-1, 2))
      poly_arr.append(points)
  return poly_arr

def convert_poly_points(xy_length, points):
  """
  Convert points from relative location to pixel location

  Args:
    xy_length: np array [x,y], the original image size
    points: np array (n,2), float
  
  Returns:
    points: np array (n,2), int32
  """
  points = points*xy_length
  points = points.astype(np.int32)
  return points

def change_origin(origin, wnd_size, points):
  """
  Crop an image based on the given points. Change the points relative coordinates regards to the origin.
  yolo format with relative coordinates: (x - origin_x) / 224, (y - origin_y) / 224
  the window size and origin changed.

  Args:
    origin: array of [x,y], the left top corner of the window
    wnd_size: int or array of [x,y]
  Returns:
    pts: array (n,2), float relative position
  """
  pts = (points - origin) / wnd_size
  return pts

def convert_ROI_dataset(data_dir, dest_dir):
  """
  convert the dataset from full image size (1040 x 1408) to cropped image size (224 x 224). This operation aim to
  reduce the unlabelled part in the full image, increase the accuracy.
  From dataset data_dir, read images and labels in yolo format. Convert the labels into pixel integer format. 
  Then crop the image in size 224 x 224 centered with micronuclei and write all the micronucleus in the image with new relative
  coordinates.

  Args:
    data_dir: str
    dest_dir: str, where to save the output dataset
  
  Returns:
  
  """
  image_dir = data_dir + 'images/'
  label_dir = data_dir + 'labels/'

  img_cnt = 0
  mn_cnt = 0

  for file in os.listdir(image_dir):
    img_name = file.split('.')[0]

    # read image
    img, img_arr = load_img(image_dir + file)

    # read polygons
    polys = load_polys(label_dir + img_name + '.txt')

    # points to pixel integer
    xy_length = np.array([img_arr.shape[1], img_arr.shape[0]])
    for i,poly in enumerate(polys):
      points = convert_poly_points(xy_length, poly)

      # crop image, save to dest
      x,y = points[0]
      w,h = 112, 112
      img2 = img.crop((x-w,y-h,x+w,y+h))
      img2.save(dest_dir + "images/" + img_name + '_' + str(i) + '.png')
      img_cnt += 1

      # write label
      new_pts = change_origin(np.array([x-w,y-h]), 224, points)
      write_buffer = []
      write_buffer.append("1 " + " ".join(map(str, new_pts.flatten())))

      # check if any other mn in the window
      for j,poly in enumerate(polys):
        if j == i: continue
        pt = poly[0]*xy_length
        if x-w < pt[0] < x+w and y-h < pt[1] < y+h:
          new_pts = change_origin(np.array([x-w,y-h]), 224, poly)
          write_buffer.append("1 " + " ".join(map(str, new_pts.flatten())))

      with open(dest_dir + "labels/" + img_name + '_' + str(i) + '.txt', 'w') as the_file:
        the_file.write("\n".join(write_buffer))
        mn_cnt += len(write_buffer)

  print("image cnt: {}, mn cnt: {}".format(img_cnt, mn_cnt))

def display(img, points):
    """
    display the img with filled polygon by points. Test purpose.
    """
    cv2.fillPoly(img, pts=[points], color=(255, 0, 255))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img)
