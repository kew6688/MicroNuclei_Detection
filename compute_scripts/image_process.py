import os
import argparse
import numpy as np
import json
from PIL import Image
import time

import torch
import torchvision
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from mn_segmentation.lib.Application import Application

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

checkpoint = "/home/y3229wan/projects/def-sushant/y3229wan/mn-project/sam/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2 = build_sam2(model_cfg, checkpoint, device=device, apply_postprocessing=False)

# get mn_info
def get_mn_info(image_path, model):
  return model.predict_image_info(image_path) # TODO: change footer

# get nuc_info
def get_nuc_info(image_path, model):
  img = Image.open(image_path)
  box = (0,0,1400,950) # TODO: change window size
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
def get_image_info(image_path, nuc_model, mn_model, mode="ALL"):
    image_name = image_path.split("/")[-1]

    nuc_info = get_nuc_info(image_path, nuc_model) if mode=="ALL" or mode=="NUC" else None
    mn_info = get_mn_info(image_path, mn_model) if mode=="ALL" or mode=="MN" else None

    return {
        "image": image_name,
        "nuclei": nuc_info,
        "micronuclei": mn_info
    }

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
  for i in range(len(data)):
    nuc_coord = data[i]['nuclei']['coord']
    mn_coord = data[i]['micronuclei']['coord']
    ind = assign_parent_nuc(nuc_coord, mn_coord)
    data[i]['micronuclei']['parent'] = ind

def run(folder, dst, mode="ALL"):
    # predict all the images and write into data frame

    # mn seg model
    app = Application("MaskRCNN-resnet50FPN/maskrcnn-resnet50fpn.pt")

    # nuc seg model
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=64,
        points_per_batch=128,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.92,
        stability_score_offset=0.7,
        min_mask_region_area=25
    )

    pred = []
    image_paths = os.listdir(folder)
    print(f"The number of images in source folder is {len(image_paths)}")

    cnt = 0
    t = 0

    for image_name in image_paths:
        if image_name[:2] == "._": continue

        image_path = os.path.join(folder,image_name)

        start = time.time()
        info = get_image_info(image_path, mask_generator, app, mode=mode)
        t += time.time() - start
        cnt += 1

        pred.append(info)

    print(f"Average processing time per image is {t/cnt}")

    # assign mn parent
    if mode=="ALL":
        add_parents(pred)

    with open(dst, "w") as outfile:
        json.dump(pred, outfile)

if __name__ == "__main__":
    # TODO: add resolution
    parser = argparse.ArgumentParser(description='Process images for detections.')
    parser.add_argument('--src', required=True, help='Source directory containing TIFF images.')
    parser.add_argument('--dst', required=True, help='Destination json file name for PNG images.')
    parser.add_argument('--mode', required=True, help='process mode, MN to get micronuclei json, NUC to get nuclei json, ALL to get all')

    args = parser.parse_args()
    source_folder = args.src
    target_json = args.dst
    mode = args.mode
    
    if mode!="DEBUG":
        run(folder=source_folder, dst=target_json, mode=mode)

