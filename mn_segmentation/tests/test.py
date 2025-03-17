import os
from PIL import Image
import numpy as np
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import time

from mn_segmentation.lib.Application import Application

from .evaluator import Evaluator

def evaluate_mn_dataset(model, dataset_path, evaluator, nms_iou=0.2, conf=0.4, mask_conf=0.7, ap_iou=0.5, mode=None):
  for file in tqdm(os.listdir(os.path.join(dataset_path,'final_masks'))[-100:]):
    # get final pred, may need customize model predict function
    im = Image.open(os.path.join(dataset_path,'images',file[:-4]+".png"))
    if mode == 'grey':
       im = im.convert('L')
    image = pil_to_tensor(im)
    pred = model._predict(image)
    pred_boxes,pred_masks,pred_scores = model._post_process(pred, conf)
    pred_boxes = pred_boxes.cpu().numpy()
    pred_masks = pred_masks.cpu().numpy()
    pred_masks = (pred_masks > mask_conf).squeeze(1)       # shape [n,w,h]
    pred_scores = pred_scores.cpu().numpy()           # shape [n]
    if pred_masks.ndim == 2:
      pred_masks = np.expand_dims(pred_masks, axis=0)
    # print(pred_masks.shape)
    # print(pred_scores.shape)

    # get GT mask
    gt_masks = np.load(os.path.join(dataset_path,'final_masks',file))  # shape [n,w,h]
    # obj_ids = np.unique(gt_masks)[1:]
    # gt_masks = (gt_masks == obj_ids[:, None, None])
    gt_masks.squeeze()
    if gt_masks.ndim == 2:
      gt_masks = np.expand_dims(gt_masks, axis=0)

    # compare and update
    recall = evaluator.update(pred_masks, pred_scores, gt_masks, ap_iou)

    # uncomment this can give bad cases that cause recall drop, missing mn
    # if recall < 0.5:
    #   print(recall,file)

  evaluator.finalize()
  # evaluator.draw_pr_curve()
  return

def test_rcnn():
    model = Application("/content/RCNN.pt")

    ap = []
    p = []
    f1 = []
    for i in range(10):
        print(f"============== conf: {i/10} =============")
        evaluator = Evaluator()
        evaluate_mn_dataset(model, "/content/mnMask/data", evaluator, conf=i/10, ap_iou=0.25)
        ap.append(evaluator.map)
        p.append(evaluator.precision)
        f1.append(evaluator.f1)

        # if i == 7:
        #   evaluator.draw_pr_curve()
