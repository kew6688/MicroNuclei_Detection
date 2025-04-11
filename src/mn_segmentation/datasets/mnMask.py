import os
import torch
import numpy as np

from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import v2 as T

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

def get_transform_grey(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.Grayscale(num_output_channels=1))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

def get_transform_jitter(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.ColorJitter(brightness=.3, contrast=.3, saturation=.3, hue=.5))
        transforms.append(T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.)))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

def get_transform_test(train,brightness=0,contrast=0,saturation=0,hue=0,blur=0):
    transforms = []
    if train:
        # transforms.append(T.RandomHorizontalFlip(0.5))
        # transforms.append(T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue))
        transforms.append(T.GaussianBlur(kernel_size=(5, 9), sigma=blur))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

class mnMaskDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
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