import cv2
import os
import numpy as np
import shutil


class Converter():
    """
    Convert the masks in binary numpy array to text polygon input as expected by YOLO

    Args:
        source_dir: The folder of dataset in numpy format
        mask_dir: The folder name of the mask
        dest_dir: The destination folder for the dataset
    """
    def __init__(self, source_dir, mask_dir, dest_dir):
       self.source_dir = source_dir
       self.mask_dir = mask_dir
       self.dest_dir = dest_dir

    def format_convert(self):
        val = 300
        cnt = 0

        save_dir = "train/"

        # go through mask folder
        for file in os.listdir(self.mask_dir):
            
            # save last 100 to validation
            if len(os.listdir(self.mask_dir)) - cnt <= val:
                save_dir = "val/"

            polys = []

            # load masks in shape (n,224,224)
            masks = np.load(mask_dir + file)

            for mask in masks:
                # convert to int
                mask = mask.astype(np.uint8).squeeze()

                # get contour of each shape
                contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                # yolo relative position
                polys.append(contours[0].squeeze() / 224)

            # copy image to folder
            source_file = source_dir + "images/" + file.split('.')[0] + ".png"
            destination_file = destination_dir + "images/" + save_dir + file.split('.')[0] + ".png"
            shutil.copy(source_file, destination_file)

            # write all polygons into txt file
            with open(os.path.join(destination_dir + "labels/" + save_dir + file.split('.')[0] + ".txt"), 'w') as f:
                for poly in polys:
                    sl = map(str, poly.flatten())
                    f.writelines("0 " + " ".join(sl) + "\n")
            cnt += 1

if __name__ == "__main__":
    # Define the source and destination directories
    source_dir = '/content/mnMask_v2/'
    destination_dir = '/content/mnSegYolo_v2/'
    mask_dir = os.path.join(source_dir, "final_masks/")
    
    cvt = Converter(source_dir, mask_dir, destination_dir)
    cvt.format_convert()