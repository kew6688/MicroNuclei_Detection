import json
import os 
from PIL import Image
from collections import defaultdict

class CropImg:
    """
    Cut the image into different pieces centered by MN and nuclei. 
    Seperate output into two groups (only nuclei or contained MN)

    Args:
        dir (str): directory for the original dataset, in COCO format
        crop_dir (str): directory for the target dataset, 
            the architecture will be crop_dir/micronuclei, crop_dir/nuclei
        label (str): json file for all the annotations
    """
    def __init__(self, dir, crop_dir, label) -> None:
            self.dir = dir
            self.crop_dir = crop_dir
            self.label = label
        
    def show_data(self):
        mc = len(os.listdir(self.crop_dir+"micronuclei"))
        nc = len(os.listdir(self.crop_dir+"nuclei"))
        print(f"generate micronuclei data {mc}")
        print(f"generate nuclei data {nc}")

        # Opening JSON file
        # f = open(self.label)
        
        # returns JSON object as 
        # a dictionary
        # data = json.load(f)

        # print(data.keys()) 
        # dict_keys(['images', 'categories', 'annotations', 'info'])


    def _overlap(self, nuc_bbox, mn_bbox):
        """
        check if the nuc_bbox contain a mn

        Args:
            nuc_bbox (tuple): coordinates for the nuclei crop image
            mn_bbox (list of tuples): the list of mn bbox in the current image

        Raises:

        Returns:
            (bool): if the nuc_bbox contain a mn
        """
        for bbox in mn_bbox:
            cx, cy = bbox[0] + 112, bbox[1] + 112
            if nuc_bbox[0] < cx < nuc_bbox[2] and nuc_bbox[1] < cy < nuc_bbox[3]:
                return True
        return False

    
    def crop(self):
        """
        run the crop algorithm for the Kate Dataset

        Args:

        Raises:

        Returns:
        """
         # Opening JSON file
        f = open(self.label)
        
        # returns JSON object as a dictionary
        data = json.load(f)

        # map image id to image path
        id_img = {}
        for img in data["images"]:
            id_img[img['id']] = img['file_name']

        s = id_img[0].split('/')[-1]
        im = Image.open(self.dir+s)
        cur_img_id = 0

        micro_nuclei_lst = []
        nuclei_lst = []
        cnt = 0
        save_id = 1

        for data_anno in data['annotations']:
            img_id = data_anno['image_id']
            # if move to next image, save all bounding box from last image
            if img_id != cur_img_id:
                cnt += 1
                print(f"processed image {cnt}")
                
                for bbox in micro_nuclei_lst:
                    im_crop = im.crop(bbox)
                    im_crop.save(self.crop_dir + f"micronuclei/{save_id}.png","png")
                    save_id += 1
                
                for bbox in nuclei_lst:
                    if self._overlap(bbox, micro_nuclei_lst):
                        im_crop = im.crop(bbox)
                        im_crop.save(self.crop_dir + f"micronuclei/{save_id}.png","png")
                        save_id += 1
                        continue
                    im_crop = im.crop(bbox)
                    im_crop.save(self.crop_dir + f"nuclei/{save_id}.png","png")
                    save_id += 1
                
                micro_nuclei_lst = []
                nuclei_lst = []
                cur_img_id = img_id
                im = Image.open(self.dir + id_img[img_id].split('/')[-1])
            
            # temp save bbox into lst
            x,y,w,h = data_anno['bbox']
            cx, cy = int(x+w/2), int(y+h/2)
            bbox = (cx-112, cy-112, cx+112, cy+112)
            if data_anno["category_id"] == 7:
                micro_nuclei_lst.append(bbox)
            elif data_anno['category_id'] == 5:
                nuclei_lst.append(bbox)
        
        for bbox in micro_nuclei_lst:
            im_crop = im.crop(bbox)
            im_crop.save(self.crop_dir + f"micronuclei/{save_id}.png","png")
            save_id += 1
        
        for bbox in nuclei_lst:
            if self._overlap(bbox, micro_nuclei_lst):
                im_crop = im.crop(bbox)
                im_crop.save(self.crop_dir + f"micronuclei/{save_id}.png","png")
                save_id += 1
                continue
            im_crop = im.crop(bbox)
            im_crop.save(self.crop_dir + f"nuclei/{save_id}.png","png")
            save_id += 1

        print(save_id)



if __name__ == "__main__":
    dir = '/home/y3229wan/scratch/KateData/images/'
    crop_dir = '/home/y3229wan/scratch/'
    # crop_dir = '/home/y3229wan/projects/def-sushant/y3229wan/mn-project/Data/KateData/cropped_images/'
    label = '/home/y3229wan/scratch/KateData/result.json'

    cropimg = CropImg(dir, crop_dir, label)
    cropimg.crop()
    cropimg.show_data()

"""
2359
generate micronuclei data 976
generate nuclei data 1382
"""