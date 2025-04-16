import numpy as np
import math

class IoUcreator:
    def __init__(self, s):
        self.s = s
    
    def create(self):
        if self.s == "Standard":
            return IoU()
        if self.s == "Scale":
            return ScaleIoU()
        
class IoU:
    def __init__(self):
        pass
    def __call__(self, pred_mask, gt_mask):
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        iou = intersection / union
        return iou

class ScaleIoU(IoU):
    def __init__(self, gamma=0.5, kappa=64):
        self.gamma = gamma
        self.kappa = kappa

    def __call__(self, pred_mask, gt_mask):
        iou = super().__call__(pred_mask, gt_mask)

        avg_sz = (pred_mask.sum() + gt_mask.sum()) / 2
        p = 1.0 - self.gamma * math.exp(- np.sqrt(avg_sz) / self.kappa)
        
        return iou**p
    
