'''
This file is used to convert a folder of json-mini label format 
from label studio and convert to individual masks

# input is a folder with images folder and some json-mini,
# output one processed json for all images
# brush_info is a list of brush
concate_json(folder_path)


# input is the folder with images and json
# output to a mask folder with all masks where each obj in instance id
# this can handle mixed brush or instance brush
def save_mask(json_file_path, mask_path):

'''
import argparse
import os
import json


# input is a folder with images folder and some json-mini,
# output one processed json for all images
# brush_info is a list of brush
def concate_json(folder_path):

  out_json = []

  # read all json file
  for file in os.listdir(folder_path):
    if file.endswith('.json'):
      # Read JSON file
      with open(folder_path+file, 'r') as file:
        data = json.load(file)

        # read each annotation in json
        for anno in data:
          obj = {}
          obj['image'] = anno['image'].split('/')[-1].replace("%20", " ")
          obj['id'] = anno['id']

          annotation = {}
          if 'brush' in anno:
            annotation['brush_info'] = anno['brush']
          if 'bbox' in anno:
            annotation['bbox'] = anno['bbox']
          # TODO: polygon info and crop_box

          obj['annotation'] = annotation
          out_json.append(obj)

  print(out_json[0])

  # Define the path where you want to save the JSON file
  json_file_path = folder_path

  # Open the file in write mode and use json.dump to write the data
  with open(json_file_path+"data.json", 'w') as json_file:
      json.dump(out_json, json_file, indent=4)


# copied from label studio github
# https://stackoverflow.com/questions/74339154/how-to-convert-rle-format-of-label-studio-to-black-and-white-image-masks

from typing import List
import numpy as np
import cv2

class InputStream:
    def __init__(self, data):
        self.data = data
        self.i = 0

    def read(self, size):
        out = self.data[self.i:self.i + size]
        self.i += size
        return int(out, 2)


def access_bit(data, num):
    """ from bytes array to bits by num position"""
    base = int(num // 8)
    shift = 7 - int(num % 8)
    return (data[base] & (1 << shift)) >> shift


def bytes2bit(data):
    """ get bit string from bytes data"""
    return ''.join([str(access_bit(data, i)) for i in range(len(data) * 8)])


def rle_to_mask(rle: List[int], height: int, width: int) -> np.array:
    """
    Converts rle to image mask
    Args:
        rle: your long rle
        height: original_height
        width: original_width

    Returns: np.array
    """

    rle_input = InputStream(bytes2bit(rle))

    num = rle_input.read(32)
    word_size = rle_input.read(5) + 1
    rle_sizes = [rle_input.read(4) + 1 for _ in range(4)]
    print('RLE params:', num, 'values,', word_size, 'word_size,', rle_sizes, 'rle_sizes')

    i = 0
    out = np.zeros(num, dtype=np.uint8)
    while i < num:
        x = rle_input.read(1)
        j = i + 1 + rle_input.read(rle_sizes[rle_input.read(2)])
        if x:
            val = rle_input.read(word_size)
            out[i:j] = val
            i = j
        else:
            while i < j:
                val = rle_input.read(word_size)
                out[i] = val
                i += 1

    image = np.reshape(out, [height, width, 4])[:, :, 3]
    return image

# input is the folder with images and json
# output to a mask folder with all masks where each obj in instance id
def save_mask(json_file_path, mask_path):
    # save the masks to data/masks/
    with open(json_file_path, 'r') as file:
      data = json.load(file)

      for anno in data:
        print(anno['image'])
        print(anno['id'])

        if not anno['annotation'] or 'brush_info' not in anno['annotation']: continue

        empty_mask = np.zeros((224,224))
        cnt = 0
        for i,brush in enumerate(anno['annotation']['brush_info']):
          rle = brush['rle']
          mask = rle_to_mask(rle, 224, 224)

          # Convet the mask into binary
          binary_mask = (mask > 0).astype(np.uint8)

          # Find the connected components in the image
          num_labels, labels_im = cv2.connectedComponents(binary_mask)

          # print(mask.shape)
          # print(mask.dtype)
          # print(np.unique(mask))
          for label in range(1, num_labels):  # Start from 1 to skip the background
            individual_mask = (labels_im == label).astype(np.uint8)
            if individual_mask.sum() < 5: continue
            empty_mask[individual_mask>0] = cnt+1
            cnt += 1

        # Check if the directory exists
        if not os.path.exists(mask_path):
            # Create the directory
            os.makedirs(mask_path)
        # Define the path where you want to save the array
        file_path = mask_path+'{}.npy'.format(anno['image'].split('.')[0])

        # Save the array to a .npy file
        np.save(file_path, empty_mask)

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.1])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=1)
    ax.imshow(mask_image)


def main(args):
    folder_path = "/content/mnMask-2/"
    concate_json(folder_path)

    mask_path = "/content/mnMask-2/masks/"
    json_file_path = "/content/mnMask-2/data.json"
    save_mask(json_file_path, mask_path)

    # masks = np.load("/content/mnMask-2/masks/GFP-H2B_A4_1_2023y07m01d_11h02m-1.npy")
    # im = Image.open("/content/mnMask-2/images/GFP-H2B_A4_1_2023y07m01d_11h02m-1.png")
    # plt.imshow(im)
    # show_mask(masks>0,plt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="process vague mask input to better segment mask")
    
    # Define command-line arguments
    parser.add_argument("image_dir", type=str, help="Path to the image file")
    parser.add_argument("mask_dir", type=str, help="Path to the original mask file")
    parser.add_argument("batch_sz", type=str, help="batch size for processing")
    parser.add_argument("output_path", type=str, help="Path to the folder save output")

    # Parse arguments
    args = parser.parse_args()

    # Run the main function
    main(args)