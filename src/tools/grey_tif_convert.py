'''
Utilities to convert random grey tif images to model desired input format.
The grey scaled images should be moved towards zero. Normalized to RGB with 8 bits values.
'''
from PIL import Image
import numpy as np
from multiprocessing import Pool
import os

image_path = "folder/to/images"
dest_path = "folder/to/save"

def process_image(file_name):
  img = Image.open(os.path.join(image_path, file_name))
  img_arr = np.array(img.convert("L"))

  img_arr -= np.min(img_arr)
  img = Image.fromarray(img_arr.astype(np.uint8))

  img.convert("RGB").save(os.path.join(dest_path, file_name))

def process_folder():
  with Pool(processes=4) as pool:
    pool.imap(process_image, os.listdir(image_path))

if __name__ == "__main__":
  process_folder()