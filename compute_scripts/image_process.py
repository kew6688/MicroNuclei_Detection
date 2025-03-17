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
from mn_segmentation.lib.image_encode import mask2rle
from mn_segmentation.lib.assign_parent import add_parents

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

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


# get mn_info
def get_mn_info(image_path, model, conf=0.7):
  return model.predict_image_info(image_path, conf=conf) 

# get nuc_info
def get_nuc_info(image_path, model, nuc_thresh):
  img = Image.open(image_path)
  img = np.array(img.convert("RGB"))
  masks = model.generate(img)

  output = {"coord":[], "area":[], "bbox":[], "score":[], "mask":[]}
  output["height"] = img.shape[0]
  output["width"] = img.shape[1]

  cur = 0
  for i in range(len(masks)):
    if masks[i]["area"] > cur:
      bg = i 
      cur = masks[i]["area"]

  for i,ann in enumerate(masks):
    if ann['area'] > nuc_thresh and i != bg:
      x,y,w,h = ann['bbox']
      output["coord"].append([x+w//2, y+h//2])
      output["area"].append(ann['area'])
      output["bbox"].append(ann['bbox'])
      output["score"].append(ann['stability_score'])
      output["mask"].append(mask2rle(ann['segmentation'].astype(int)))
  return output

# get image_info
def get_image_info(image_path, nuc_model, mn_model, mode="ALL", conf=0.7):
    image_name = image_path.split("/")[-1]

    mn_info = get_mn_info(image_path, mn_model, conf=conf) if mode=="ALL" or mode=="MN" else None
    if mn_info:
       # set the threshhold for nuclei size as the largest micronuclei, at least 100
       nuc_thresh = max(100, max(mn_info["area"]))
    else:
       nuc_thresh = 100

    nuc_info = get_nuc_info(image_path, nuc_model, nuc_thresh) if mode=="ALL" or mode=="NUC" else None

    return {
        "image": image_name,
        "nuclei": nuc_info,
        "micronuclei": mn_info
    }

def run(folder, dst, parent, conf, mode="ALL"):
    '''
    Predict all the images and write into data frame
    '''
    
    if not conf or not isinstance(conf, float):
      conf = 0.7

    # mn seg model
    app = Application("./MicroNuclei_Detection/checkpoints/maskrcnn-resnet50fpn.pt")

    # nuc seg model
    checkpoint = "./sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    sam2 = build_sam2(model_cfg, checkpoint, device=device, apply_postprocessing=False)
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
        info = get_image_info(image_path, mask_generator, app, mode=mode, conf=conf)
        t += time.time() - start
        cnt += 1

        pred.append(info)

    print(f"Average processing time per image is {t/cnt}")

    # assign mn parent
    if mode=="ALL":
        add_parents(pred, parent)

    with open(dst, "w") as outfile:
        json.dump(pred, outfile)

if __name__ == "__main__":
    # TODO: add resolution
    parser = argparse.ArgumentParser(description='Process images for detections.')
    parser.add_argument('-s', '--src', required=True, help='Source directory containing TIFF images.')
    parser.add_argument('-d', '--dst', required=True, help='Destination json file name for PNG images.')
    parser.add_argument('-mod', '--mode', required=True, help='process mode, MN to get micronuclei json, NUC to get nuclei json, ALL to get all')
    parser.add_argument('-c', '--conf', required=False, help='confidence threshold for micronuclei detection, e.g. --conf 0.7 (0.7 by default)')
    parser.add_argument('-o', '--out', required=False, help='Output format is contained mask (full) or only box (short), e.g. -o full/short (full by default)')
    parser.add_argument('-p', '--parent', required=False, help='Parent assign method, use closest center or edge to find nearest parent nuclei (edge by default)')

    args = parser.parse_args()
    source_folder = args.src
    target_json = args.dst
    mode = args.mode
    conf = args.conf if "conf" in args else 0.7
    par = args.parent if "parent" in args else "edge"
    
    if mode!="DEBUG":
        run(folder=source_folder, dst=target_json, mode=mode, conf=conf, parent=par)

