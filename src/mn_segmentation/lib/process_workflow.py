# -*- coding: utf-8 -*-
"""Process Workflow.ipynb
# Load Image
"""

import json
import os
from PIL import Image
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load an image using PIL
image_path = '/content/Ulises_counts/Day 6 count/VID3664_A5_1_2024y08m06d_11h44m.jpg'
image = Image.open(image_path)
box = (0,0,1400,950)
image = image.crop(box)

# Create a figure and axis
fig, ax = plt.subplots(1)

# Display the image
ax.imshow(image)

# Define a sequence of boxes, each box is defined by (x, y, width, height)
boxes = []

# box_w, box_h = 470, 350
# x,y = 470, 350

box_w, box_h = 224, 224
x,y = 224, 224

for i in range(35):
  # tile image
  cur_x, cur_y = x * (i//5), y * (i%5)
  box = (cur_x, cur_y, box_w, box_h)
  boxes.append(box)

# Draw each box on the image
for box in boxes:
    rect = patches.Rectangle(
        (box[0], box[1]),  # (x, y)
        box[2],  # width
        box[3],  # height
        linewidth=2,
        edgecolor='r',
        facecolor='none'
    )
    ax.add_patch(rect)

# Show the final image with boxes
plt.show()

image_path = '/content/MicroNuclei_Detection/Ulises_counts/Day 6 count/VID3664_A4_1_2024y08m06d_11h44m.jpg'
box_w, box_h = 224, 224
x,y = 224, 224
im = Image.open(image_path)
# shuffle id
array = np.arange(1, 31)
np.random.shuffle(array)
print(array)
for i in range(30):
  if i in [4,9,29]: continue
  # tile image
  cur_x, cur_y = x * (i//5), y * (i%5)
  box = (cur_x, cur_y, cur_x + box_w, cur_y + box_h)
  img2 = im.crop(box)
  img2.save(f'tile_{array[i]}.png')


"""# Predict

## SAM

### Load model
"""

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

mask_generator = SAM2AutomaticMaskGenerator(sam2)

image_paths = ['images/GFP-H2B_A1_1_2023y07m02d_11h02m.png', 'images/GFP-H2B_A4_1_2023y07m02d_11h02m.png','images/GFP-H2B_B1_1_2023y07m02d_11h02m.png',
              'images/GFP-H2B_B4_1_2023y07m02d_11h02m.png','images/GFP-H2B_C1_1_2023y07m02d_11h02m.png','images/GFP-H2B_C4_1_2023y07m02d_11h02m.png']
for image_path in image_paths:
  im = Image.open(image_path)
  mn_cnt = 0
  nuc_cnt = 0
  for i in range(30):
    # tile image
    cur_x, cur_y = x * (i//5), y * (i%5)
    box = (cur_x, cur_y, cur_x + box_w, cur_y + box_h)
    img2 = im.crop(box)
    img2 = np.array(img2.convert("RGB"))
    masks = mask_generator.generate(img2)
    for ann in masks:
      m = ann['segmentation']
      if np.sum(m) > 30:
        nuc_cnt += 1
      else:
        mn_cnt += 1
  print(image_path, mn_cnt, nuc_cnt)

image_path = "tile_10.png"
img2 = Image.open(image_path)
img2 = np.array(img2.convert("RGB"))
masks = mask_generator.generate(img2)

plt.figure(figsize=(10, 10))
plt.imshow(img2)
show_anns(masks)
plt.axis('off')
plt.show()

plt.figure(figsize=(10, 10))
plt.imshow(image)
for ann in masks:
  m = ann['segmentation']
  if np.sum(m) > 30: continue
  show_anns([ann])
plt.axis('off')
plt.show()

"""## RCNN"""
import matplotlib.pyplot as plt
import os
import torch

from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.transforms import v2 as T
from torchvision.ops import nms

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# load a model pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

# replace the classifier with a new one, that has
# num_classes which is user-defined
num_classes = 2  # 1 class (person) + background
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


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

# get the model using our helper function
model = get_model_instance_segmentation(num_classes)

# move model to the right device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.load_state_dict(torch.load("RCNN.pt"))

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

image = read_image("tile_1.png")
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
pred_labels = [f"pedestrian: {score:.3f}" for lSabel, score in zip(pred["labels"], pred["scores"])]
pred_boxes = pred["boxes"].long()
output_image = draw_bounding_boxes(image, pred_boxes, colors="red")

masks = (pred["masks"] > 0.3).squeeze(1)
output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")


plt.figure(figsize=(12, 12))
plt.imshow(output_image.permute(1, 2, 0))

type(pred)

pred

image = read_image("tile_10.png")
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
pred_labels = [f"pedestrian: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
pred_boxes = pred["boxes"].long()
output_image = draw_bounding_boxes(image, pred_boxes[ind][pred["scores"][ind]>0.4], colors="red")

masks = (pred["masks"] > 0.6).squeeze(1)
output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")


plt.figure(figsize=(12, 12))
plt.imshow(output_image.permute(1, 2, 0))

image_paths = ['images/GFP-H2B_A1_1_2023y07m02d_11h02m.png', 'images/GFP-H2B_A4_1_2023y07m02d_11h02m.png','images/GFP-H2B_B1_1_2023y07m02d_11h02m.png',
              'images/GFP-H2B_B4_1_2023y07m02d_11h02m.png','images/GFP-H2B_C1_1_2023y07m02d_11h02m.png','images/GFP-H2B_C4_1_2023y07m02d_11h02m.png']

for image_path in image_paths:
  im = Image.open(image_path)
  mn_cnt = 0
  nuc_cnt = 0
  for i in range(30):
    # tile image
    wnd_sz = 224
    cur_x, cur_y = wnd_sz * (i//5), wnd_sz * (i%5)
    box = (cur_x, cur_y, cur_x + wnd_sz, cur_y + wnd_sz)
    # print(box)
    img2 = im.crop(box)
    img2.save(f'tile.png')

    image = read_image("tile.png")
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
  print(image_path, mn_cnt)

"""### utils"""

import matplotlib.pyplot as plt
import os
import torch

from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.transforms import v2 as T
from torchvision.ops import nms

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms.functional import pil_to_tensor

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

def load_weight(model, path):
  model.load_state_dict(torch.load(path))

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

def countImage(image_path, model):
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

"""# Evaluate"""

import importlib
import Application  # Import the module initially
import cluster

importlib.reload(Application)
importlib.reload(cluster)

from torchvision.ops import nms

app = Application.Application()

image = read_image("tile_20.png")

pred = app._predict(image)

image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
image = image[:3, ...]

ind = nms(pred["boxes"], pred["scores"], 0.2)
pred_labels = [f"pedestrian: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
pred_boxes = pred["boxes"].long()
output_image = draw_bounding_boxes(image, pred_boxes[ind][pred["scores"][ind]>0.1], colors="red")

# masks = (pred["masks"] > 0.6).squeeze(1)
# output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")


plt.figure(figsize=(12, 12))
plt.imshow(output_image.permute(1, 2, 0))

box = pred_boxes[ind][pred["scores"][ind]>0.1]
print(pred["scores"][ind])
cluster.resolveApop(box)

# predict all the images and write into data frame

import os

app = Application("/content/RCNN.pt")

pred = [[],[]]
pred_no_apop = [[],[]]
pred_8 = [[],[]]
pred_6 = [[],[]]

folders = ["/content/Ulises_counts/Day 3 count", "/content/Ulises_counts/Day 6 count"]
for i in range(2):
  folder = folders[i]
  image_paths = sorted(os.listdir(folder))

  for image_path in image_paths:
     if image_path[:2] == "._": continue
     cnt = app.predict_image(os.path.join(folder,image_path),True,0.7)
     pred[i].append(cnt)
    #  cnt = app.predict_image(os.path.join(folder,image_path),True,0.6)
    #  pred_6[i].append(cnt)
    #  cnt = app.predict_image(os.path.join(folder,image_path))
    #  pred_8[i].append(cnt)

import pandas as pd

csv_path = "/content/Ulises_counts/MNcounts_nadmod.csv"

df = pd.read_csv(csv_path)

df['3 days pred (0.7 score)'] = pred[0]
df['6 days pred (0.7 score)'] = pred[1]

df.to_csv('ulises_output_0.7.csv', index=False)
df

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Assuming df is your DataFrame
# Prepare data
X = np.concatenate((df.iloc[:, 1].values.reshape(-1, 1), df.iloc[:, 2].values.reshape(-1, 1)))
Y = np.concatenate((df.iloc[:, 4].values.reshape(-1, 1), df.iloc[:, 5].values.reshape(-1, 1)))

# Fit the linear regression model
linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

# Get slope and intercept
print(f"The regression slope is {linear_regressor.coef_}, intercept is {linear_regressor.intercept_}")

# Calculate 95% confidence interval
X_with_const = sm.add_constant(X)  # Add intercept term
model = sm.OLS(Y, X_with_const)
results = model.fit()

# Generate predictions and confidence intervals
predictions = results.get_prediction(X_with_const)
confidence_interval = predictions.conf_int(alpha=0.05)
lower_bound = confidence_interval[:, 0]
upper_bound = confidence_interval[:, 1]

# Separate data for day3 and day6
X1 = df.iloc[:, 1].values.reshape(-1, 1)
Y1 = df.iloc[:, 4].values.reshape(-1, 1)
X2 = df.iloc[:, 2].values.reshape(-1, 1)
Y2 = df.iloc[:, 5].values.reshape(-1, 1)

# Plot data points and regression line
plt.scatter(X1, Y1, color='blue', label="day3")
plt.scatter(X2, Y2, color='yellow', label="day6")
plt.plot(X, Y_pred, color='red', label="regression line")

# Plot the 95% confidence interval as a shaded area
plt.fill_between(X.flatten(), lower_bound, upper_bound, color='red', alpha=0.2, label="95% CI")

# Add labels, legend, and display plot
plt.plot(X, X, color='blue', label="standard")
plt.xlabel("All count")
plt.ylabel("All predict")
plt.legend()
plt.show()



# @title 3 days pred vs 3 days Count

from matplotlib import pyplot as plt
# df.plot(kind='scatter', x='3 days pred', y='3 days count', s=32, alpha=.8)
sns.lmplot(x='3 days pred', y='3 days count',data=df,fit_reg=True)
plt.gca().spines[['top', 'right',]].set_visible(False)

!cp /content/drive/MyDrive/PMCC/Analysis/ulises_output_0.7.csv /content/ulises_output_0.7.csv

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

csv_path = "/content/ulises_output.csv"
df = pd.read_csv(csv_path)
df

from sklearn.linear_model import LinearRegression

X = df.iloc[:, 1].values.reshape(-1, 1)  # iloc[:, 1] is the column of X
Y = df.iloc[:, 4].values.reshape(-1, 1)  # df.iloc[:, 4] is the column of Y
Y2 = df.iloc[:, 6].values.reshape(-1, 1)
Y3 = df.iloc[:, 8].values.reshape(-1, 1)

linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)
print(f"0.4: The regression slope is {linear_regressor.coef_}, intercept is {linear_regressor.intercept_}")

linear_regressor = LinearRegression()
linear_regressor.fit(X, Y2)
Y_pred2 = linear_regressor.predict(X)
print(f"0.6 The regression slope is {linear_regressor.coef_}, intercept is {linear_regressor.intercept_}")

linear_regressor = LinearRegression()
linear_regressor.fit(X, Y3)
Y_pred3 = linear_regressor.predict(X)
print(f"0.8: The regression slope is {linear_regressor.coef_}, intercept is {linear_regressor.intercept_}")

plt.scatter(X, Y2)
plt.plot(X, Y_pred, color='red', label="regression @0.4")
plt.plot(X, Y_pred2, color='yellow', label="regression without apop @0.6")
plt.plot(X, Y_pred3, color='green', label="regression without apop @0.8")
plt.plot(X, X, color='blue', label="standard")

plt.xlabel("3 days count")
plt.ylabel("3 days predict")
plt.legend()
plt.show()

X = np.concatenate((df.iloc[:, 1].values.reshape(-1, 1),df.iloc[:, 2].values.reshape(-1, 1)))  # iloc[:, 1] is the column of X
Y = np.concatenate((df.iloc[:, 4].values.reshape(-1, 1),df.iloc[:, 5].values.reshape(-1, 1)))

linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)
print(f"The regression slope is {linear_regressor.coef_}, intercept is {linear_regressor.intercept_}")
r = np.corrcoef(X.reshape(-1), Y.reshape(-1))
print(r)

X1 = df.iloc[:, 1].values.reshape(-1, 1)  # iloc[:, 1] is the column of X
Y1 = df.iloc[:, 4].values.reshape(-1, 1)  # df.iloc[:, 4] is the column of Y
X2 = df.iloc[:, 2].values.reshape(-1, 1)  # iloc[:, 1] is the column of X
Y2 = df.iloc[:, 5].values.reshape(-1, 1)  # df.iloc[:, 4] is the column of Y

plt.scatter(X1, Y1,cmap='blue',label="day3")
plt.scatter(X2, Y2,cmap='yellow',label="day6")
plt.plot(X, Y_pred, color='red', label="regression")
plt.plot(X, X, color='blue', label="standard")
plt.xlabel("All count")
plt.ylabel("All predict")
plt.legend()
plt.show()



X = df.iloc[:, 2].values.reshape(-1, 1)  # iloc[:, 1] is the column of X
Y = df.iloc[:, 5].values.reshape(-1, 1)  # df.iloc[:, 4] is the column of Y
Y2 = df.iloc[:, 7].values.reshape(-1, 1)
Y3 = df.iloc[:, 9].values.reshape(-1, 1)

linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)
print(f"0.4: The regression slope is {linear_regressor.coef_}, intercept is {linear_regressor.intercept_}")

linear_regressor = LinearRegression()
linear_regressor.fit(X, Y2)
Y_pred2 = linear_regressor.predict(X)
print(f"0.6 The regression slope is {linear_regressor.coef_}, intercept is {linear_regressor.intercept_}")

linear_regressor = LinearRegression()
linear_regressor.fit(X, Y3)
Y_pred3 = linear_regressor.predict(X)
print(f"0.8: The regression slope is {linear_regressor.coef_}, intercept is {linear_regressor.intercept_}")

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red', label="regression @0.4")
plt.plot(X, Y_pred2, color='yellow', label="regression without apop @0.6")
plt.plot(X, Y_pred3, color='green', label="regression without apop @0.8")
plt.plot(X, X, color='blue', label="standard")

plt.xlabel("6 days count")
plt.ylabel("6 days predict")
plt.legend()
plt.show()

"""## Oliv data


ID of parent nuc to assign
dividing rate different by drug

generate loc, size of nuc
same for mn

```
[
    {
        "image": ...,
        "nuclei": nuc_info,
        "micronuclei": mn_info
    }
]

For each info:

{
      "coord": [[x1, y1],...],
      "area": [x,...],
      "bbox": [...],
      "parent": [id,...]       # mn only
}

```
"""
"""### Import lib

Clone from github repo.

Install project and import module.
"""

import importlib
import Application  # Import the module initially
import cluster

importlib.reload(Application)
importlib.reload(cluster)

from mn_segmentation.lib.Application import Application
import importlib
import mn_segmentation.lib.Application # Make sure that this is the correct path to your Application module
importlib.reload(mn_segmentation.lib.Application)

"""Import Sam2"""

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

mask_generator = SAM2AutomaticMaskGenerator(sam2)

import copy
import os

import imageio
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import zipfile

from skimage.io import imread, imsave, imshow
from skimage.transform import resize as imresize
from skimage.color import rgb2gray

import cv2

from deepcell.applications import NuclearSegmentation, CellTracking
from deepcell_tracking.trk_io import load_trks

os.environ.update({"DEEPCELL_ACCESS_TOKEN": "6ReEfZXk.klUcA2BWefkB8kD8ijF8rb7lSy7yBSem"})
app = NuclearSegmentation().from_version("1.1")

print('Training Resolution:', app.model_mpp, 'microns per pixel')

"""Test"""

import time

image_path = '/content/Olivieri_panel_titration_GFP-H2B/Plate 1_GFP-H2B_C10_1_2024y02m22d_09h41m.tif'
img2 = Image.open(image_path)
img2 = np.array(img2.convert("RGB"))

start = time.time()
masks = mask_generator.generate(img2)
end = time.time()
print(end - start)

m = masks
plt.figure(figsize=(10, 10))
plt.imshow(img2)
show_anns(masks)
plt.axis('off')
plt.show()

image_path = '/content/Olivieri_panel_titration_GFP-H2B/Plate 1_GFP-H2B_C10_1_2024y02m22d_09h41m.tif'

image = Image.open(image_path)
box = (0,0,1400,950)
img2 = image.crop(box)
img2 = np.array(img2.convert("RGB"))

mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2,
    points_per_side=64,
    points_per_batch=128,
    pred_iou_thresh=0.7,
    stability_score_thresh=0.92,
    stability_score_offset=0.7,
    min_mask_region_area=25
)
start = time.time()
masks = mask_generator.generate(img2)
end = time.time()
print(end - start)

m = masks
plt.figure(figsize=(10, 10))
plt.imshow(img2)
show_anns(masks)
plt.axis('off')
plt.show()
plt.savefig("test_full.png")

masks

from torchvision.ops import nms
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

app = Application("/content/RCNN.pt")

image = read_image("/content/MicroNuclei_Detection/tile_2.png")

pred = app._predict(image)

image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
image = image[:3, ...]

ind = nms(pred["boxes"], pred["scores"], 0.2)
pred_labels = [f"pedestrian: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
pred_boxes = pred["boxes"].long()
output_image = draw_bounding_boxes(image, pred_boxes[ind][pred["scores"][ind]>0.1], colors="red")

# masks = (pred["masks"] > 0.6).squeeze(1)
# output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")


plt.figure(figsize=(12, 12))
plt.imshow(output_image.permute(1, 2, 0))

m = pred["masks"][ind][pred["scores"][ind]>0.4]
pred["masks"].sum(1).sum(1).sum(1).shape

"""### Utils

Generate a json file with all the processed nuclei and mn information stored in the following format.

```
[image_info], size n == number of images:

[
    {
        "image": image_name,    # str
        "nuclei": nuc_info,
        "micronuclei": mn_info
    }
]

For each info:

{
      "coord": [[x1, y1],...],   `# list of center coordinates
      "area": [x,...],        # list of mask area
      "bbox": [...],         # list of bounding box,
                        for mn bbox: (xmin, ymin, xmax, ymax)
                        for nuc bbox: (x, y, w, h)
      "parent": [id,...]       # assigned parent nuclei, mn only
}

```
"""

# get mn_info
def get_mn_info(image_path, model):
  return model.predict_image_info(image_path)

# get nuc_info
def get_nuc_info(image_path, model):
  img = Image.open(image_path)
  box = (0,0,1400,950)
  img = img.crop(box)
  img = np.array(img.convert("RGB"))
  masks = model.generate(img)

  output = {"coord":[], "area":[], "bbox":[]}
  for ann in masks[1:]:
    if ann['area'] > 30:
      x,y,w,h = ann['bbox']
      output["coord"].append([x+w//2, y+h//2])
      output["area"].append(ann['area'])
      output["bbox"].append(ann['bbox'])
  return output

# get image_info
def get_image_info(image_path, nuc_model, mn_model):
  image_name = image_path.split("/")[-1]
  nuc_info = get_nuc_info(image_path, nuc_model)
  mn_info = get_mn_info(image_path, mn_model)
  return {
      "image": image_name,
      "nuclei": nuc_info,
      "micronuclei": mn_info
  }

image_path = '/content/Ulises_counts/Day 6 count/VID3664_A5_1_2024y08m06d_11h44m.jpg'
info = get_image_info(image_path, mask_generator, app)
with open("sample.json", "w") as outfile:
    json.dump([info], outfile)

folder = "/content/Olivieri_panel_titration_GFP-H2B/"
image_paths = os.listdir(folder)[1]
nc = get_mn_info(folder+image_paths, app)
nc

img = Image.open(folder+image_paths)
fig, ax = plt.subplots(1)
ax.imshow(img)
plt.show()

"""### Run"""

import os
import pandas as pd
from tqdm import tqdm
import json

folder = "/content/Olivieri_panel_titration_GFP-H2B"
image_paths = os.listdir(folder)
print(len(image_paths))
cnt = 0
for image_name in tqdm(image_paths):
  if image_name[:2] == "._": continue
  if not ("C" in image_name or "D" in image_name or "M" in image_name or "N" in image_name):
    cnt += 1
cnt

# predict all the images and write into data frame

app = Application("/content/sample_data/RCNN.pt")

mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2,
    points_per_side=64,
    points_per_batch=128,
    pred_iou_thresh=0.7,
    stability_score_thresh=0.92,
    stability_score_offset=0.7,
    min_mask_region_area=25
)

folder = "/content/MicroNuclei_Detection/katecompare"
image_paths = os.listdir(folder)

img = []
pred = []

for image_name in tqdm(image_paths):
  if image_name[:2] == "._": continue
  # if ("C" in image_name or "D" in image_name or "M" in image_name or "N" in image_name):
  #   continue

  image_path = os.path.join(folder,image_name)
  # start = time.time()
  info = get_image_info(image_path, mask_generator, app)
  # print(time.time() - start)

  pred.append(info)

with open("rep1_part.json", "w") as outfile:
  json.dump(pred, outfile)

# df = pd.DataFrame({"image":img,"cnt":pred})
# df.to_csv('oliv_out.csv', index=False)


"""### parent assign"""


from tqdm import tqdm

def assign_parent_nuc(nuc_coord, mn_coord):
  '''
  assign parent nuc to mn

  Parameter:
    nuc_pos: list of coords [[x,y],...]

  Return:
    ind: index of parent nuc for each mn
  '''
  ind = []
  for x1,y1 in mn_coord:
    min_dist = float('inf')
    min_ind = -1
    for i, c in enumerate(nuc_coord):
      x2,y2 = c
      dist = (x1-x2)**2 + (y1-y2)**2
      if dist < min_dist:
        min_dist = dist
        min_ind = i
    ind.append(min_ind)
  return ind

def add_parents(data):
  for i in tqdm(range(len(data))):
    nuc_coord = data[i]['nuclei']['coord']
    mn_coord = data[i]['micronuclei']['coord']
    ind = assign_parent_nuc(nuc_coord, mn_coord)
    data[i]['micronuclei']['parent'] = ind

import json

file = "work/MCF10A"
data = json.load(open(file+".json"))
add_parents(data)

# check if done
print('parent' in data[0]['micronuclei'])

with open(file+"_out.json", "w") as outfile:
  json.dump(data, outfile)

"""## Compare Kate"""

from tqdm import tqdm
import os

app = Application("/content/RCNN.pt")
pred = [[],[],[]]

for file in tqdm(os.listdir('/content/katecompare')):
  if file[:2] == "._": continue
  pred[0].append(file)
  pred[1].append(app.predict_image(os.path.join('/content/katecompare',file), True, 0.4))
  pred[2].append(app.predict_image(os.path.join('/content/katecompare',file), False, 0.4))

import pandas as pd

# Sample data
data = {
    'image': pred[0],  # image names
    'cnt_with_postprocess': pred[1],              # counts with post-processing
    'cnt_without_postprocess': pred[2]           # counts without post-processing
}

# Create DataFrame
df = pd.DataFrame(data)

# Display DataFrame
print(df)

df.to_csv('kate_output.csv', index=False)