import numpy as np
import matplotlib.pyplot as plt
from iou import IoUcreaor 

class Evaluator:
  def __init__(self, save=False):
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
      self.IoU = IoUcreaor()

  def update(self, pred_masks, pred_scores, gt_masks, ap_iou=0.5, iou_method="Standard"):

    self.objects += gt_masks.shape[0]
    self.predictions += pred_masks.shape[0]

    t_cnt = 0
    for i in range(pred_masks.shape[0]):
      conf = pred_scores[i]
      res = False
      overlap = 0
      save_iou = 0
      for j in range(gt_masks.shape[0]):
        if gt_masks[j].sum()>1000 or gt_masks[j].sum()<5: continue
        intersection = np.logical_and(pred_masks[i], gt_masks[j]).sum()
        union = np.logical_or(pred_masks[i], gt_masks[j]).sum()
        iou = self.IoU(iou_method)(pred_masks[i], gt_masks[i])
        
        if intersection > overlap:
          overlap = intersection
          res = True if iou > ap_iou else False
          save_iou = max(save_iou, iou)

      if res:
        self.TP += 1
        t_cnt += 1
      else:
        self.FP += 1
      self.pred_list.append((conf, res))

      # save the positive predictions' size and iou
      if self.save and res:
        self.mn_lst.append({"size":pred_masks[i].sum(), "iou":save_iou})

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