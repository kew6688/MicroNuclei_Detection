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
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

import mn_segmentation
from mn_segmentation.lib import cluster
from mn_segmentation.models.mask_rcnn import maskrcnn_resnet50_fpn, get_mn_model, MaskRCNN

def get_device():
  if torch.cuda.is_available():
    return torch.device('cuda')
  elif torch.backends.mps.is_available():
    return torch.device("mps")
  else:
    return torch.device('cpu')

def load_weight(model, path):
  model.load_state_dict(torch.load(path, map_location=get_device()))

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

def countImage(image_path, model, device=get_device()):
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
  '''
  Object that takes a model or weight and manage predictions.

  Examples:

    >>> from mn_segmentation.lib.Application import Application
    >>> app = Application(checkpoint_path)
    >>> cnt = app.predict_image_count(image)
    >>> print(cnt)

    predict_image_mask():
    predict_image_info():
    predict_image_count():
    predict_display():

  Args:

    weight (str): the checkpoint path to load the pre-tained weight for the model
    model (MaskRCNN): directly take a model for prediction
    device (str): the device for computation
    mode (str): the usage of the app, expacting color of input
  '''
  
  def __init__(self, 
               weight: str = None, 
               model: MaskRCNN = None, 
               device: str = None, 
               mode: str = None):
    if not model:
      self.model = get_mn_model()

      # Checkpoints are downloaded by the script provided in checkpoints folder by default
      if weight == None:
        weight = os.path.join(mn_segmentation.__path__[0], "../checkpoints/maskrcnn-resnet50fpn.pt")

      load_weight(self.model, weight)

    else:
      self.model = model

    if not device:
      self.device = get_device()
    else:
      self.device = device

    self.model.to(self.device)
    
    self.mode = mode

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
  
  def _post_process(self, pred, conf=0.4, bbox_nms_thresh=0.2):
    ind = nms(pred["boxes"], pred["scores"], bbox_nms_thresh)
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

  def predict_image_count(self, image_path, resolveApop=True, conf=0.7, footer_skip=False):
    im = Image.open(image_path)
    mn_cnt = 0

    # crop image to model input size 224x224
    wnd_sz = 224
    height, width = np.array(im).shape[:2]

    # calculate how many rows and cols to cover the image
    for i in range(height // wnd_sz + 1):
      for j in range(width // wnd_sz + 1):
        # skip footer in Kate's images, row and col is 5x7
        if footer_skip and i*7+j in [28,29,33,34]: continue

        # tile image
        cur_x, cur_y = wnd_sz * j, wnd_sz * i
        box = (cur_x, cur_y, cur_x + wnd_sz, cur_y + wnd_sz)

        image = pil_to_tensor(im.crop(box))
        pred = self._predict(image)

        pred_boxes,_,_ = self._post_process(pred, conf)

        if resolveApop:
          mn_cnt += cluster.resolveApop(pred_boxes)
        else:
          mn_cnt += len(pred_boxes)
    return mn_cnt
  
  def predict_image_info(self, image_path, conf=0.7, footer_skip=False):
    im = Image.open(image_path)
    output = {"coord":[], "area":[], "bbox":[]}

    # crop image to model input size 224x224
    wnd_sz = 224
    height, width = np.array(im).shape[:2]

    # calculate how many rows and cols to cover the image, contain last row and col
    for i in range(height // wnd_sz + 1):
      for j in range(width // wnd_sz + 1):
        # skip footer in Kate's images, row and col is 5x7
        if footer_skip and i*7+j in [28,29,33,34]: continue

        # tile image
        cur_x, cur_y = wnd_sz * j, wnd_sz * i
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
  
  def predict_image_mask(self, image_path, conf=0.7, bbox_nms_thresh=0.2):
    im = Image.open(image_path)
    image_height, image_width = np.array(im).shape[:2]

    # Create an empty array of the same size as the image to hold the masks
    output_mask = np.zeros((image_height, image_width), dtype=np.uint8)

    mn_id = 1

    # crop image to model input size 224x224
    wnd_sz = 224
    height, width = np.array(im).shape[:2]

    # calculate how many rows and cols to cover the image, 
    # does not contain last row and col
    for i in range(height // wnd_sz):
      for j in range(width // wnd_sz):

        # tile image
        wnd_sz = 224
        cur_x, cur_y = wnd_sz * j, wnd_sz * i
        box = (cur_x, cur_y, cur_x + wnd_sz, cur_y + wnd_sz)

        image = pil_to_tensor(im.crop(box))
        pred = self._predict(image)

        _, pred_masks,_ = self._post_process(pred, conf,bbox_nms_thresh)
        pred_masks = pred_masks.cpu().numpy().squeeze(1)
        for mask_i in range(pred_masks.shape[0]):
          m = (pred_masks[mask_i] > conf)
          output_mask[cur_y: cur_y+wnd_sz, cur_x: cur_x+wnd_sz][m] = mn_id
          mn_id += 1
    return output_mask

  def predict_display(self, image_path: str, point: list[int, int], conf: int=0.7, bbox_nms_thresh: int=0.2) -> None:
    '''
    Display a part of image with predictions, include boxes, masks, and scores.

    Args:
      
      image_path,
      point: left corner of the window,
      conf,
      bbox_nms_thresh
    '''
    # load image
    im = Image.open(image_path)

    # tile image
    wnd_sz = 224
    cur_x, cur_y = point
    box = (cur_x, cur_y, cur_x + wnd_sz, cur_y + wnd_sz)
    image = pil_to_tensor(im.crop(box))
    pred = self._predict(image)
    pred_boxes,pred_masks,pred_scores = self._post_process(pred, conf=conf, bbox_nms_thresh=bbox_nms_thresh)
    pred_masks = pred_masks.cpu()
    pred_scores = pred_scores.cpu().numpy()

    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    image = image[:3, ...]
    pred_labels = [f"mn: {score:.3f}" for score in pred_scores]
    pred_boxes = pred_boxes.long()

    # draw the bounding boxes
    output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

    masks = (pred_masks > 0.7).squeeze(1)
    output_image = draw_segmentation_masks(output_image, masks, alpha=0.7, colors="blue")

    return output_image
