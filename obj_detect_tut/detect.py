import matplotlib.pyplot as plt
from torchvision.io import read_image
import os
import torch

# image = read_image("data/PennFudanPed/PNGImages/FudanPed00046.png")
# mask = read_image("data/PennFudanPed/PedMasks/FudanPed00046_mask.png")

from PIL import Image

image = Image.open("/home/y3229wan/projects/def-sushant/y3229wan/mn-project/obj_detect_tut/data/PennFudanPed/PNGImages/FudanPed00046.png")
image.save("img.png")
print("hi")

# plt.figure(figsize=(16, 8))
# plt.subplot(121)
# plt.title("Image")
# plt.imshow(image.permute(1, 2, 0))
# plt.subplot(122)
# plt.title("Mask")
# plt.imshow(mask.permute(1, 2, 0))

# from torchvision.io import read_image
# from torchvision.ops.boxes import masks_to_boxes
# from torchvision import tv_tensors
# from torchvision.transforms.v2 import functional as F


# class PennFudanDataset(torch.utils.data.Dataset):
#     def __init__(self, root, transforms):
#         self.root = root
#         self.transforms = transforms
#         # load all image files, sorting them to
#         # ensure that they are aligned
#         self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
#         self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

#     def __getitem__(self, idx):
#         # load images and masks
#         img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
#         mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
#         img = read_image(img_path)
#         mask = read_image(mask_path)
#         # instances are encoded as different colors
#         obj_ids = torch.unique(mask)
#         # first id is the background, so remove it
#         obj_ids = obj_ids[1:]
#         num_objs = len(obj_ids)

#         # split the color-encoded mask into a set
#         # of binary masks
#         masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

#         # get bounding box coordinates for each mask
#         boxes = masks_to_boxes(masks)

#         # there is only one class
#         labels = torch.ones((num_objs,), dtype=torch.int64)

#         image_id = idx
#         area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
#         # suppose all instances are not crowd
#         iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

#         # Wrap sample and targets into torchvision tv_tensors:
#         img = tv_tensors.Image(img)

#         target = {}
#         target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
#         target["masks"] = tv_tensors.Mask(masks)
#         target["labels"] = labels
#         target["image_id"] = image_id
#         target["area"] = area
#         target["iscrowd"] = iscrowd

#         if self.transforms is not None:
#             img, target = self.transforms(img, target)

#         return img, target

#     def __len__(self):
#         return len(self.imgs)


# import torchvision
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# # load a model pre-trained on COCO
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

# # replace the classifier with a new one, that has
# # num_classes which is user-defined
# num_classes = 2  # 1 class (person) + background
# # get number of input features for the classifier
# in_features = model.roi_heads.box_predictor.cls_score.in_features
# # replace the pre-trained head with a new one
# model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)