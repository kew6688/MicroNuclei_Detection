# -*- coding: utf-8 -*-
"""Model Evaluation
# Load dataset
"""

import json
import os

data = json.load(open('/content/mnMask/data/result.json'))

len(os.listdir('/content/mnMask/data/masks'))

"""# Load model

### mask RCNN
"""

from mn_segmentation.lib.Application import Application
import importlib
import mn_segmentation.lib.Application # Make sure that this is the correct path to your Application module
importlib.reload(mn_segmentation.lib.Application)

import os
from PIL import Image
import numpy as np
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm
import matplotlib.pyplot as plt

dataset_path = "/content/mnMask/data"

# miss mn labeled case: "10 Gy_GFP-H2B_A2_3_2023y06m24d_20h17m_5.npy"

file = "GFP-H2B_A1_1_2023y07m03d_01h02m-1.npy"
model = Application("/content/RCNN.pt")

im = Image.open(os.path.join(dataset_path,'images',file[:-4]+".png"))
image = pil_to_tensor(im)
pred = model._predict(image)
pred_boxes,pred_masks,pred_scores = model._post_process(pred, 0.7)
pred_boxes = pred_boxes.cpu().numpy()
pred_masks = np.squeeze(pred_masks.cpu().numpy())       # shape [n,w,h]
pred_scores = pred_scores.cpu().numpy()      # shape [n]
if pred_masks.ndim == 2:
  pred_masks = np.expand_dims(pred_masks, axis=0)
print(pred_masks.shape)
print(pred_scores.shape)

# get GT mask
gt_masks = np.load(os.path.join(dataset_path,'label_masks',file))  # shape [n,w,h]
obj_ids = np.unique(gt_masks)
gt_masks = (gt_masks == obj_ids[:, None, None])
print(gt_masks.shape)

!rm "/content/mnMask/data/label_masks/Plate 1_GFP-H2B_F17_1_2022y12m06d_07h30m-0.npy"

from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.io import read_image
import torch

image = read_image(os.path.join(dataset_path,'images',file[:-4]+".png"))
image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
image = image[:3, ...]
output_image = draw_segmentation_masks(image, torch.from_numpy(pred_masks>0.3), alpha=0.5, colors="blue")

plt.figure(figsize=(12, 12))
plt.imshow(output_image.permute(1, 2, 0))

im

plt.imshow(gt_masks[0],alpha=0.5)

plt.imshow(im)
plt.imshow(gt_masks[1],alpha=0.5)
# np.unique(gt_masks[0])
gt_masks[1].sum()

cnt = 0
for i in range(pred_masks.shape[0]):
  res=False
  overlap=0
  for j in range(gt_masks.shape[0]):
    # if gt_masks[j].sum() > 1000: continue
    intersection = np.logical_and(pred_masks[i]>0.8, gt_masks[j]>0).sum()
    union = np.logical_or(pred_masks[i]>0.8, gt_masks[j]>0).sum()
    iou = intersection / union
    print(intersection,union,iou)
    if intersection > overlap:
      overlap = intersection
      res = True if iou > 0.5 else False
  if res:
    cnt += 1
cnt/(gt_masks.shape[0]-1)

"""## Eveluate"""

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
    self.objects += gt_masks.shape[0]-1
    self.predictions += pred_masks.shape[0]
    t_cnt = 0
    for i in range(pred_masks.shape[0]):
      conf = pred_scores[i]
      res = False
      overlap = 0
      for j in range(gt_masks.shape[0]):
        if gt_masks[j].sum()>1000 or gt_masks[j].sum()<5: continue
        intersection = np.logical_and(pred_masks[i], gt_masks[j]).sum()
        union = np.logical_or(pred_masks[i], gt_masks[j]).sum()
        iou = intersection / union
        if intersection > overlap:
          overlap = intersection
          res = True if iou > ap_iou else False
      if res:
        self.TP += 1
        t_cnt += 1
      else:
        self.FP += 1
      self.pred_list.append((conf, res))
    return t_cnt/(gt_masks.shape[0]-1) if (gt_masks.shape[0]-1) > 0 else 1

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

    print(f"Average Precision: {sum(p)/len(p)}")

import os
from PIL import Image
import numpy as np
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm
import matplotlib.pyplot as plt

def evaluate_mn_dataset(model, dataset_path, evaluator, nms_iou=0.2, conf=0.4, ap_iou=0.5):
  for file in tqdm(os.listdir(os.path.join(dataset_path,'label_masks'))[-200:]):
    # get final pred, may need customize model predict function
    im = Image.open(os.path.join(dataset_path,'images',file[:-4]+".png"))
    image = pil_to_tensor(im)
    pred = model._predict(image)
    pred_boxes,pred_masks,pred_scores = model._post_process(pred, conf)
    pred_boxes = pred_boxes.cpu().numpy()
    pred_masks = pred_masks.cpu().numpy()
    pred_masks = (pred_masks > conf).squeeze(1)       # shape [n,w,h]
    pred_scores = pred_scores.cpu().numpy()           # shape [n]
    if pred_masks.ndim == 2:
      pred_masks = np.expand_dims(pred_masks, axis=0)
    # print(pred_masks.shape)
    # print(pred_scores.shape)

    # get GT mask
    gt_masks = np.load(os.path.join(dataset_path,'label_masks',file))  # shape [n,w,h]
    obj_ids = np.unique(gt_masks)
    gt_masks = (gt_masks == obj_ids[:, None, None])
    # print(gt_masks.shape)

    # compare and update
    recall = evaluator.update(pred_masks, pred_scores, gt_masks, ap_iou)

    # uncomment this can give bad cases that cause recall drop, missing mn
    # if recall < 0.5:
    #   print(recall,file)

  evaluator.finalize()
  # evaluator.draw_pr_curve()
  return

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

plt.plot(np.arange(10)/10, ap)
plt.plot(np.arange(10)/10, p)
plt.plot(np.arange(10)/10, f1)
plt.legend(['AP', 'Precision', 'F1'])
plt.xlabel('Confidence')
plt.ylabel('Metrics')
plt.title('Metrics vs Confidence')
plt.show()

import json

# Example lists

# Create a dictionary from the lists
data = {
    "ap": ap,
    "p": p,
    "f1": f1
}

# Write the dictionary to a JSON file
with open("maskrcnn_resnet50.json", "w") as json_file:
    json.dump(data, json_file, indent=4)  # indent=4 for pretty printing

"""# Nuc"""

import torch
import torchvision
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())
import sys
import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

np.random.seed(3)

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    ax.imshow(img)

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2,
    points_per_side=64,
    points_per_batch=128,
    pred_iou_thresh=0.7,
    stability_score_thresh=0.92,
    stability_score_offset=0.7,
    min_mask_region_area=25
)

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from PIL import Image


def load_img(img_path):
  """
  Read an image and convert it to RGB format.

  Args:
    img_path (str): Path to the image file.

  Returns:
    Img object: Image object from PIL.
    numpy.ndarray: RGB image array from Image.open.
  """

  img = Image.open(img_path)
  img_arr = np.array(img)
  return img, img_arr

def load_polys(label_path, target_clas = 7, xy_length=(224,224)):
  """
  Read a label txt file and convert it to numpy array.

  Args:
    label_path (str): Path to the label text file.

  Returns:
    numpy.ndarray: polygon array that each polygon contains an array of coordinates ([[x,y],...]).
  """

  poly_arr = []
  with open(label_path, 'r') as label_file:
    # Read and print each line in the file
    for line in label_file:
      l = line.strip().split()
      # class type:
          #   4: cell_active_div, 5: cell_non_div, 6: cell_sick_apop, 7: micronuclei
      cls, points = int(l[0]), l[1:]

      # current focus on mn
      if cls != target_clas:
        continue

      points = np.array(points, dtype=np.float64).reshape((-1, 2))
      points = convert_poly_points(xy_length, points)
      poly_arr.append(points)
  return poly_arr

def convert_poly_points(xy_length, points):
  points = points*xy_length
  points = points.astype(np.int32)
  return points

def change_origin(origin, wnd_size, points):
  """
  Crop an image based on the given points. Change the points relative coordinates regards to the origin.
  yolo format with relative coordinates: (x - origin_x) / 224, (y - origin_y) / 224
  the window size and origin changed.

  Args:
    origin: array of [x,y]
    wnd_size: int or array of [x,y]
  Returns:

  """
  pts = (points - origin) / wnd_size
  return pts

def convert_ROI_dataset(data_dir, dest_dir):
  image_dir = data_dir + 'images/'
  label_dir = data_dir + 'labels/'

  img_cnt = 0
  mn_cnt = 0

  for file in os.listdir(image_dir):
    img_name = file.split('.')[0]

    # read image
    img, img_arr = load_img(image_dir + file)

    # read polygons
    polys = load_polys(label_dir + img_name + '.txt')

    # points to pixel integer
    xy_length = np.array([img_arr.shape[1], img_arr.shape[0]])
    for i,poly in enumerate(polys):
      points = convert_poly_points(xy_length, poly)

      # crop image, save to dest
      x,y = points[0]
      w,h = 112, 112
      img2 = img.crop((x-w,y-h,x+w,y+h))
      img2.save(dest_dir + "images/" + img_name + '_' + str(i) + '.png')
      img_cnt += 1

      # write label
      new_pts = change_origin(np.array([x-w,y-h]), 224, points)
      write_buffer = []
      write_buffer.append("1 " + " ".join(map(str, new_pts.flatten())))

      # check if any other mn in the window
      for j,poly in enumerate(polys):
        if j == i: continue
        pt = poly[0]*xy_length
        if x-w < pt[0] < x+w and y-h < pt[1] < y+h:
          new_pts = change_origin(np.array([x-w,y-h]), 224, poly)
          write_buffer.append("1 " + " ".join(map(str, new_pts.flatten())))

      with open(dest_dir + "labels/" + img_name + '_' + str(i) + '.txt', 'w') as the_file:
        the_file.write("\n".join(write_buffer))
        mn_cnt += len(write_buffer)

  print("image cnt: {}, mn cnt: {}".format(img_cnt, mn_cnt))

def display(img, points):
    cv2.fillPoly(img, pts=[points], color=(255, 0, 255))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img)

def evaluate_nuc_dataset(model, dataset_path, evaluator, nms_iou=0.2, conf=0.4, ap_iou=0.5):
  for file in tqdm(os.listdir(os.path.join(dataset_path,'labels'))):
    # get final pred, may need customize model predict function
    image = Image.open(os.path.join(dataset_path,'images',file[:-4]+".png"))
    image = np.array(image.convert("RGB"))
    masks = mask_generator.generate(image)
    # print(len(masks))
    # print(masks[0]['segmentation'].shape)
    pred_masks = []
    pred_scores = []
    for mask in masks:
      if mask['bbox'][3] > 900 and mask['bbox'][2] > 900:
        continue
      pred_masks.append(mask['segmentation'])
      pred_scores.append(mask['predicted_iou'])

    # get GT mask
    points = load_polys(os.path.join(dataset_path,'labels',file), target_clas = 5, xy_length=(1408,1040))
    gt_masks = []
    for p in points:
      im = np.zeros((1040,1408),dtype=np.uint8)
      cv2.fillPoly( im, points, 255 )
      gt_masks.append(im)

    # compare and update
    evaluator.update(np.array(pred_masks), np.array(pred_scores), np.array(gt_masks), ap_iou)

  evaluator.finalize()
  evaluator.draw_pr_curve()
  return

points = load_polys("/content/MicroNuclei_Detection/uncrop/labels/10 Gy_GFP-H2B_A1_1_2023y06m24d_18h17m.txt", target_clas = 5, xy_length=(1408,1040))
im = np.zeros((1040,1408),dtype=np.uint8)
plt.imshow(cv2.fillPoly( im, points, 255 ))

img = Image.open("/content/MicroNuclei_Detection/uncrop/images/10 Gy_GFP-H2B_A1_1_2023y06m24d_18h17m.png")
img = np.array(img.convert("RGB"))
masks = mask_generator.generate(img)
plt.figure(figsize=(10, 10))
plt.imshow(img)
show_anns(masks)
plt.axis('off')
plt.show()

print(len(masks))
plt.imshow(masks[0]['segmentation'])

evaluator = Evaluator()
evaluate_nuc_dataset(mask_generator, "/content/uncrop", evaluator, conf=0.4, ap_iou=0.5)

from google.colab import runtime
runtime.unassign()

"""# condition"""

from PIL import Image, ImageEnhance
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img_path = '/content/MCF10AIDH2WT-DMSO-IR-CENPA-1.tif'  # Replace with your image path
image = Image.open(img_path)

# Step 1: Increase Brightness
enhancer = ImageEnhance.Brightness(image)
bright_image = enhancer.enhance(1.5)  # Adjust the factor as needed

# Step 2: Increase Contrast
enhancer = ImageEnhance.Contrast(bright_image)
contrast_image = enhancer.enhance(1.5)  # Adjust the factor as needed

# Convert to OpenCV format (BGR) for color adjustments
image_cv = cv2.cvtColor(np.array(contrast_image), cv2.COLOR_RGB2BGR)

# Step 3: Enhance Blue Tone
# Split the image into B, G, R channels
b, g, r = cv2.split(image_cv)
b = cv2.add(b, 50)  # Increase blue channel, adjust 50 based on desired intensity
enhanced_image_cv = cv2.merge((b, g, r))

# Convert back to RGB for displaying with PIL or plt
final_image = cv2.cvtColor(enhanced_image_cv, cv2.COLOR_BGR2RGB)
final_image_pil = Image.fromarray(final_image)

# Display the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)

plt.subplot(1, 2, 2)
plt.title("Enhanced Image")
plt.imshow(final_image_pil)
plt.show()

img_path = '/content/MCF10AIDH2WT-DMSO-IR-CENPA-1.tif'  # Replace with your image path
image = Image.open(img_path).convert('I')

l,h = np.min(np.array(image)), np.max(np.array(image))
im = np.array(image)
# im = (im / h)*(255)
img_array = np.array(image)
min_val = 170
max_val = 2337
normalized_img = ((img_array - min_val) / (max_val - min_val) * 65535).astype(np.uint16)
# plt.imshow(normalized_img)
print(l,h)
normalized_img_8bit = (normalized_img / 256).astype(np.uint8)

# Display the results
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title("Original Image")
# plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
# plt.title("Enhanced Image")
plt.imshow(normalized_img_8bit, cmap='gray')
plt.show()

im[0][0]

!apt install imagemagick

!identify -verbose MCF10AIDH2WT-DMSO-IR-CENPA-1.tif

!convert MCF10AIDH2WT-DMSO-IR-CENPA-1.tif -fill "gray(1)" +opaque black indices.pgm

!convert indices.pgm -normalize result.jpg