import matplotlib.pyplot as plt

class Evaluator:
  def __init__(self):
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
      
  def update(self, pred_masks, pred_scores, gt_masks, ap_iou):
    self.objects += gt_masks.shape[0]
    self.predictions += pred_masks.shape[0]
    for i in range(pred_masks.shape[0]):
      conf = pred_scores[i]
      res = False
      overlap = 0
      for j in range(gt_masks.shape[0]):
        intersection = (pred_masks[i]>0 & gt_masks[j]>0).sum()
        union = (pred_masks[i]>0 | gt_masks[j]>0).sum()
        iou = intersection / union
        if (pred_masks[i]>0 & gt_masks[j]>0).sum() > overlap:
          overlap = (pred_masks[i]>0 & gt_masks[j]>0).sum()
          res = True if iou > ap_iou else False
      if res:
        self.TP += 1
      else:
        self.FP += 1
    self.pred_list.append((conf, res))
  
  def finalize(self):
    self.FN = self.objects - self.TP
    self.pred_list.sort(key=lambda x: x[0], reverse=True)
    correct = 0
    for i in range(len(self.pred_list)):
      if self.pred_list[i][1]:
        correct += 1
      self.map += correct / (i + 1)
    self.map /= self.predictions
    print(f"mAP: {self.map}")

    self.precision = self.TP / (self.TP + self.FP)
    self.recall = self.TP / (self.TP + self.FN)
    self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)
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