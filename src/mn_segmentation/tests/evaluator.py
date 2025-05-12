import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm
import argparse
import time
from mn_segmentation.tests.iou import IoUcreator 

def get_mask_center(mask):
    """
    Computes the center (centroid) of a binary mask.

    Parameters:
        mask (np.ndarray): 2D numpy array of the binary mask (bool or 0/1).

    Returns:
        (float, float): (row_center, col_center)
    """
    indices = np.argwhere(mask)  # Get coordinates of non-zero pixels
    if indices.size == 0:
        return None  # or raise ValueError("Mask is empty.")

    center = indices.mean(axis=0)  # Mean along rows gives centroid
    return tuple(center)

class Evaluator:
  def __init__(self, save=False, iou_method="Standard"):
      self.pred_list = []
      self.objects = 0
      self.predictions = 0
      self.TP = 0
      self.FP = 0
      self.FN = 0
      self.precision = 0
      self.recall = 0
      self.f1 = 0
      self.map = 0

      self.save = save
      self.mn_lst = []
      self.IoU = IoUcreator(iou_method).create()

  def update(self, pred_masks, pred_scores, gt_masks, ap_iou=0.5):

    self.objects += gt_masks.shape[0]
    self.predictions += pred_masks.shape[0]

    t_cnt = 0
    for i in range(pred_masks.shape[0]):
      conf = pred_scores[i]
      res = False
      overlap = 0
      save_iou = 0
      save_gt = 0
      save_shift = 0
      for j in range(gt_masks.shape[0]):
        if gt_masks[j].sum()>1000 or gt_masks[j].sum()<5: continue
        intersection = np.logical_and(pred_masks[i], gt_masks[j]).sum()
        union = np.logical_or(pred_masks[i], gt_masks[j]).sum()
        iou = self.IoU(pred_masks[i], gt_masks[j])

        if intersection > overlap:
          overlap = intersection
          res = True if iou > ap_iou else False
          save_iou = max(save_iou, iou)
          save_gt = gt_masks[j].sum()

          gt_center = get_mask_center(gt_masks[j])
          pred_center = get_mask_center(pred_masks[i])
          save_shift = np.sqrt((gt_center[0] - pred_center[0])**2 + (gt_center[1] - pred_center[1])**2)

      if res:
        self.TP += 1
        t_cnt += 1
      else:
        self.FP += 1
      self.pred_list.append((conf, res))

      # save the positive predictions' size and iou
      if self.save and res:
        self.mn_lst.append({"pred_size":pred_masks[i].sum().item(), "label_size":save_gt.item(), "shift":save_shift.item(), "iou":save_iou.item()})

    return t_cnt/(gt_masks.shape[0]-1) if (gt_masks.shape[0]-1) > 0 else 1

  def finalize(self):
    self.FN = self.objects - self.TP
    self.pred_list.sort(key=lambda x: x[0], reverse=True)
    correct = 0
    for i in range(len(self.pred_list)):
      if self.pred_list[i][1]:
        correct += 1
      self.map += correct / (i + 1)
    self.map /= self.predictions if self.predictions != 0 else -1
    print(f"mAP: {self.map}")

    self.precision = self.TP / (self.TP + self.FP) if self.TP + self.FP != 0 else -1
    self.recall = self.TP / (self.TP + self.FN) if self.TP + self.FN != 0 else -1
    self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall) if self.precision + self.recall != 0 else -1
    print(f"precision: {self.precision}")
    print(f"recall: {self.recall}")
    print(f"f1: {self.f1}")

    return self.precision, self.recall, self.f1

  def draw_pr_curve(self):
    p = []
    r = []
    correct = 0
    for i in range(len(self.pred_list)):
      if self.pred_list[i][1]:
        correct += 1
      p.append(correct / (i + 1))
      r.append(correct / self.objects)
    plt.plot(r, p)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

    print(f"Average Precision: {sum(p)/len(p)}")

  def save_szIoU(self):
    return self.mn_lst
  
# evaluate pipeline
def evaluate_mn_dataset(model, dataset_path, evaluator, img_folder='images', mask_folder='test_masks', mod="all", nms_iou=0.2, conf=0.4, mask_conf=0.7, ap_iou=0.5):
  for file in tqdm(os.listdir(os.path.join(dataset_path,mask_folder))[-300:]):
    # get final pred, may need customize model predict function
    im = Image.open(os.path.join(dataset_path,img_folder,file[:-4]+".png"))
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
    gt_masks = np.load(os.path.join(dataset_path,mask_folder,file))  # shape [n,w,h]
    # obj_ids = np.unique(gt_masks)[1:]
    # gt_masks = (gt_masks == obj_ids[:, None, None])
    gt_masks.squeeze()
    if gt_masks.ndim == 2:
      gt_masks = np.expand_dims(gt_masks, axis=0)

    if mod=="easy" and len(gt_masks) > 3:
      continue
    if mod=="hard" and len(gt_masks) <= 3:
      continue

    # compare and update
    recall = evaluator.update(pred_masks, pred_scores, gt_masks, ap_iou)

    # uncomment this can give bad cases that cause recall drop, missing mn
    # if recall < 0.5:
    #   print(recall,file)

  evaluator.finalize()
  # evaluator.draw_pr_curve()
  return