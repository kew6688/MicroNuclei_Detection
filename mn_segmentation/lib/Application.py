import json
import os
from PIL import Image
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
from torchvision.transforms import v2 as T
from torchvision.ops import nms
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms.functional import pil_to_tensor

import mn_segmentation.lib.cluster as cluster

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model

def get_device():
  if torch.cuda.is_available():
    return torch.device('cuda')
  else:
    return torch.device('cpu')

def load_weight(model, path):
  model.load_state_dict(torch.load(path))

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

def countImage(image_path, model, device='cpu'):
  im = Image.open(image_path)
  mn_cnt = 0
  nuc_cnt = 0
  for i in range(35):
    # skip footer
    if i in [4,9,29]: continue

    # tile image
    wnd_sz = 224
    cur_x, cur_y = wnd_sz * (i//5), wnd_sz * (i%5)
    box = (cur_x, cur_y, cur_x + wnd_sz, cur_y + wnd_sz)

    image = pil_to_tensor(im.crop(box))
    eval_transform = get_transform(train=False)

    model.eval()
    with torch.no_grad():
        x = eval_transform(image)
        # convert RGBA -> RGB and move to device
        x = x[:3, ...].to(device)
        predictions = model([x, ])
        pred = predictions[0]

    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    image = image[:3, ...]

    ind = nms(pred["boxes"], pred["scores"], 0.2)
    # print(ind)
    pred_labels = [f"pedestrian: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
    pred_boxes = pred["boxes"].long()
    mn_cnt += len(pred_boxes[ind][pred["scores"][ind]>0.6])
  return mn_cnt

class Application:
  # object that takes a model and manage predictions
  def __init__(self, weight=None, model=None, device=None):
    if not model:
      num_class = 2
      self.model = get_model_instance_segmentation(num_class)
      load_weight(self.model, weight)
    else:
      self.model = model

    if not device:
      self.device = get_device()
    else:
      self.device = device

    self.model.to(self.device)

  def predict_image(self, image, resolveApop=True, conf=0.5):
    im = Image.open(image)
    mn_cnt = 0
    for i in range(35):
      # skip footer
      if i in [4,9,29]: continue

      # tile image
      wnd_sz = 224
      cur_x, cur_y = wnd_sz * (i//5), wnd_sz * (i%5)
      box = (cur_x, cur_y, cur_x + wnd_sz, cur_y + wnd_sz)

      image = pil_to_tensor(im.crop(box))
      pred = self._predict(image)

      pred_boxes,_,_ = self._post_process(pred, conf)

      if resolveApop:
        mn_cnt += cluster.resolveApop(pred_boxes)
      else:
        mn_cnt += len(pred_boxes)
    return mn_cnt
  
  def predict_image_info(self, image, conf=0.4):
    im = Image.open(image)
    output = {"coord":[], "area":[], "bbox":[]}
    for i in range(35):
      # skip footer
      if i in [4,9,29]: continue

      # tile image
      wnd_sz = 224
      cur_x, cur_y = wnd_sz * (i//5), wnd_sz * (i%5)
      box = (cur_x, cur_y, cur_x + wnd_sz, cur_y + wnd_sz)

      image = pil_to_tensor(im.crop(box))
      pred = self._predict(image)

      pred_boxes, pred_masks,_ = self._post_process(pred, conf)
      pred_boxes[:, [0,2]] += cur_x
      pred_boxes[:, [1,3]] += cur_y
      area = pred_masks.sum(1).sum(1).sum(1)
      # print(area.shape)
      
      output["bbox"] += pred_boxes.cpu().numpy().tolist()
      output["coord"] += cluster.boxToCenters(pred_boxes).tolist()
      output["area"] += area.cpu().numpy().tolist()
    return output

  def _predict(self, image):
    eval_transform = get_transform(train=False)
    self.model.eval()
    with torch.no_grad():
        x = eval_transform(image)
        # convert RGBA -> RGB and move to device
        x = x[:3, ...].to(self.device)
        predictions = self.model([x, ])
        pred = predictions[0]
    return pred
  
  def predict_image_mask(self, image, conf=0.7, footer=True):
    im = Image.open(image)
    image_height, image_width = np.array(im).shape[:2]
    # Create an empty array of the same size as the image to hold the masks
    output_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    mn_id = 1
    for i in range(35):
      # skip footer
      if footer and i in [4,9,29]: continue

      # tile image
      wnd_sz = 224
      cur_x, cur_y = wnd_sz * (i//5), wnd_sz * (i%5)
      box = (cur_x, cur_y, cur_x + wnd_sz, cur_y + wnd_sz)

      image = pil_to_tensor(im.crop(box))
      pred = self._predict(image)

      pred_boxes, pred_masks,_ = self._post_process(pred, conf)
      for i in range(pred_masks.shape[0]):
        m = (pred_masks[i] > conf).cpu().numpy()
        output_mask[cur_y: cur_y+wnd_sz, cur_x: cur_x+wnd_sz][m] = mn_id
        mn_id += 1
    return output_mask

  def _post_process(self, pred, conf=0.4):
    ind = nms(pred["boxes"], pred["scores"], 0.2)
    # print(ind)
    pred_boxes = pred["boxes"].long()
    return pred_boxes[ind][pred["scores"][ind]>conf], pred["masks"][ind][pred["scores"][ind]>conf], pred["scores"][ind][pred["scores"][ind]>conf]

  def _tile_input(self, image_path, wnd_sz = 224):
    '''
    should accept varies image size, batch input
    '''
    l = []
    im = Image.open(image_path)
    for i in range(35):
      # skip footer
      if i in [4,9,29]: continue

      # tile image
      cur_x, cur_y = wnd_sz * (i//5), wnd_sz * (i%5)
      box = (cur_x, cur_y, cur_x + wnd_sz, cur_y + wnd_sz)

      image = pil_to_tensor(im.crop(box))
      l.append(image)
    return l


  def _untile_output(self, image_path):
    pass
