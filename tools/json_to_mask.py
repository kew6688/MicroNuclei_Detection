import os
import json

def concate_json(folder_path):

  out_json = []

  # read all json file
  for file in os.listdir():
    if file.endswith('.json'):
      # Read JSON file
      with open(file, 'r') as file:
        data = json.load(file)

        # read each annotation in json
        for anno in data:
          obj = {}
          obj['image'] = anno['image'].split('/')[-1].replace("%20", " ")
          obj['id'] = anno['id']

          annotation = {}
          if 'brush' in anno:
            annotation['brush_info'] = anno['brush'][0]
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

def save_mask(json_file_path, mask_path)
    # save the masks to data/masks/
    with open(json_file_path, 'r') as file:
      data = json.load(file)

      for anno in data:
        print(anno['image'])
        print(anno['id'])

        if not anno['annotation'] or 'brush_info' not in anno['annotation']: continue

        brush = anno['annotation']['brush_info']
        rle = brush['rle']
        height = brush['original_height']
        width = brush['original_width']
        mask = rle_to_mask(rle, height, width)
        print(mask.shape)
        print(mask.dtype)

        # Define the path where you want to save the array
        file_path = mask_path+'{}.npy'.format(anno['image'].split('.')[0])

        # Save the array to a .npy file
        np.save(file_path, mask > 0)


import argparse

def main(args):
    output, output_name = refine_images(args.image_dir, args.mask_dir, args.batch_sz)

    for i in range(len(output)):
        np.save(os.path.join(args.output_path,output_name[i]), output[i])

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