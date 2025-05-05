import os
from PIL import Image
import numpy as np
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

def evaluate_mn_dataset(model, dataset_path, evaluator, nms_iou=0.2, conf=0.4, ap_iou=0.5):
  for file in tqdm(os.listdir(os.path.join(dataset_path,'label_masks'))):
    # get final pred, may need customize model predict function
    im = Image.open(os.path.join(dataset_path,'images',file))
    image = pil_to_tensor(im)
    pred = model._predict(image)
    pred_boxes,pred_masks,pred_scores = model._post_process(pred, conf)
    pred_boxes = pred_boxes.cpu().numpy()
    pred_masks = pred_masks.cpu().numpy()       # shape [n,w,h]
    pred_scores = pred_scores.cpu().numpy()      # shape [n]
    print(pred_masks.shape)
    print(pred_scores.shape)

    # get GT mask
    gt_masks = np.load(os.path.join(dataset_path,'label_masks',file))  # shape [n,w,h]
    print(gt_masks.shape)

    # compare and update
    evaluator.update(pred_masks, pred_scores, gt_masks, ap_iou)
  
  evaluator.finalize()
  evaluator.draw_pr_curve()
  return