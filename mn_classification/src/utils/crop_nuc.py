import json
import os 
from PIL import Image
from collections import defaultdict
from MN.mn_classification.main import *
import numpy as np

# allow truncated image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

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

    def crop_ROI(self, model_path=None, processed_img_path=None):
        '''Generate smaller image for better annotation. pre-predict the image if contain mn by using pre-trained ResNet on NucRec dataset,
            to mimic the process of ROI (Region of Interest, https://www.nature.com/articles/s41586-023-06157-7#Sec60)

        Args:
            model_path: pre trained model that used to classify the proposed image
            processed_img_path: the text file location to save processed images
        
        Returns:
            None
            (print count of generated image, saved to self.crop_dir)
        '''
        model = MNClassifier()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using {device} device")
        model.to(device)

        preprocess = v2.Compose([
            # v2.Resize(size = (224,224)),
            # v2.RandomHorizontalFlip(p=0.5),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # crop image into 6 rows, 5 cols
        box_w, box_h = 224, 224
        x,y = 200, 204
        cnt = 0
        im = None

        seen_img = set()

        for file_cnt,file in enumerate(sorted(os.listdir(self.dir))):
            try:
                im = Image.open(self.dir+file)
            except:
                continue

            # add this image into processed set
            seen_img.add(file)
            
            for i in range(30):
                # tile image
                cur_x, cur_y = x * (i//5), y * (i%5)
                box = (cur_x, cur_y, cur_x + box_w, cur_y + box_h)
                img2 = im.crop(box)

                # preprocess the input
                input_arr = np.array(img2)
                # print(input_arr.shape)
                input_tensor = preprocess(input_arr)

                # display preprocessed test image
                # plt.imshow(torch.permute(input_tensor, (1,2,0)))
                # plt.savefig(f"test_transformed.png")

                input_batch = input_tensor.unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(input_batch)
                    if output.argmax(1) == 0:
                        img2.save(self.crop_dir + file.split('.')[0] + '-' + str(i) + '.png')
                        cnt += 1

            if cnt > 3000: 
                print(f"process {file_cnt} files to generate 3000 cropped image")
                break
        print(cnt)

        # write processed images into a file, so that we won't generate repeat image next time. 
        # use a+ to append new lines
        with open(processed_img_path, 'a+') as f:
            for line in seen_img:
                f.write(f"{line}\n")



if __name__ == "__main__":
    dir = '20230621_MCF10A_pngs/'
    os.mkdir('MCF10A_cropped/')
    crop_dir = 'MCF10A_cropped/'
    label = '/home/y3229wan/scratch/KateData/result.json'
    model_path = '/home/y3229wan/projects/def-sushant/y3229wan/mn-project/MN/output/MNClassifier_best.pt'

    processed_img_path = '/home/y3229wan/projects/def-sushant/y3229wan/mn-project/Data/MCF10A_processed_images.txt'

    cropimg = CropImg(dir, crop_dir, label)
    # cropimg.crop()
    # cropimg.show_data()
    cropimg.crop_ROI(model_path, processed_img_path)

"""
2359
generate micronuclei data 976
generate nuclei data 1382
"""