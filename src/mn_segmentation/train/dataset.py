import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import v2 as T

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

def seperate_semantic_mask(mask):
  '''
  Convert mask that contains all obj into individual mask that
    contain each obj, from semantic seg to instance seg.

  Args:
    im: mask from the dataset

  Returns:
    [im]: list of masks
  '''

  # Convet the mask into binary
  binary_mask = (mask > 0).astype(np.uint8)

  # Find the connected components in the image
  num_labels, labels_im = cv2.connectedComponents(binary_mask)

  # Map component labels to individual masks
  individual_masks = []
  for label in range(1, num_labels):  # Start from 1 to skip the background
    individual_mask = (labels_im == label).astype(np.uint8)
    if individual_mask.sum() < 3: continue
    individual_masks.append(torch.from_numpy(individual_mask))

  # Return the list of individual masks
  return individual_masks


class mnMaskDataset(torch.utils.data.Dataset):
    '''
    Load raw mask dataset, the mask is semantic brush label.

    Args:
        root: root directory of the dataset
    '''
        
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))

    def __getitem__(self, idx):
        # load images and masks
        file_name = self.masks[idx].split(".")[0]
        mask_name = file_name + ".npy"
        img_name = file_name + ".png"
        # print(img_name)
        # print(mask_name)

        img_path = os.path.join(self.root, "images", img_name)
        mask_path = os.path.join(self.root, "masks", mask_name)
        img = read_image(img_path)
        mask = np.load(mask_path)

        # split the color-encoded mask into a set
        # of binary masks
        masks = seperate_semantic_mask(mask)
        if not masks: print(file_name)
        masks = torch.stack(masks)

        # instances are encoded as different colors
        # obj_ids = torch.unique(mask)
        # first id is the background, so remove it
        # obj_ids = obj_ids[1:]
        num_objs = len(masks)
        # print(obj_ids)

        # get bounding box coordinates for each mask
        boxes = masks_to_boxes(masks)
        for box in boxes:
          if box[0] == box[2] or box[1] == box[3]:
            box[2] += 1
            box[3] += 1

        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.masks)
    
class mnMaskFinalDataset(torch.utils.data.Dataset):
    '''
    Load refined mask dataset

    Args:
        root: root directory of the dataset
    '''

    def __init__(self, root, image_folder_name="images", mask_folder_name="final_masks", transforms=None):
        
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = os.listdir(os.path.join(root, "final_masks"))

    def __getitem__(self, idx):
        # load images and masks
        file_name = self.masks[idx].split(".")[0]
        mask_name = file_name + ".npy"
        img_name = file_name + ".png"
        # print(img_name)
        # print(mask_name)

        img_path = os.path.join(self.root, "images", img_name)
        mask_path = os.path.join(self.root, "final_masks", mask_name)
        img = read_image(img_path)
        masks = np.load(mask_path).squeeze()
        if masks.ndim == 2:
          masks = np.expand_dims(masks, axis=0)

        masks = masks.astype(np.uint8)
        masks = torch.from_numpy(masks)

        # instances are encoded as different colors
        # obj_ids = torch.unique(mask)
        # first id is the background, so remove it
        # obj_ids = obj_ids[1:]
        num_objs = len(masks)
        # print(obj_ids)

        # get bounding box coordinates for each mask
        boxes = masks_to_boxes(masks)
        for box in boxes:
          if box[0] == box[2] or box[1] == box[3]:
            box[2] += 1
            box[3] += 1

        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.masks)