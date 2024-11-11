# refine the dataset

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torchvision.ops import masks_to_boxes

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

def load_input(img_path, label_mask_path):
    '''
    Generate input for sam predictor. Each image path will return one numpy image 
      and a np array of boxes input
    
    Args:
      img_path, str: path to image 
      label_mask_path, str: path to label mask

    Returns:
      img: numpy array
      boxes: numpy array of boxes input in shape [n,4] 

    '''
    img = Image.open(img_path)
    img = np.array(img.convert("RGB"))

    label = np.load(label_mask_path)
    mask = torch.from_numpy(label)
    # We get the unique colors, as these would be the object ids.
    obj_ids = torch.unique(mask)

    # first id is the background, so remove it.
    obj_ids = obj_ids[1:]

    # split the color-encoded mask into a set of boolean masks.
    # Note that this snippet would work as well if the masks were float values instead of ints.
    masks = mask == obj_ids[:, None, None]

    boxes = masks_to_boxes(masks).numpy()
    # print(boxes.size())
    # print(boxes.numpy())

    return img, boxes

def refine_images(img_dir, label_mask_dir, batch_sz):
    # get all masks name
    label_mask_lst = sorted(os.listdir(label_mask_dir))

    img_batch = []
    boxes_batch = []
    output_name = []
    for label_mask in label_mask_lst:
        img_name = label_mask.split('.')[0] + '.png'
        img_path = os.path.join(img_dir, img_name)
        label_mask_path = os.path.join(label_mask_dir, label_mask)

        img, boxes = load_input(img_path, label_mask_path)
        if boxes.shape[0] == 0:
            continue
        img_batch.append(img)
        boxes_batch.append(boxes)
        output_name.append(label_mask)

    sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

    output = []
    for i in tqdm(range(0, len(img_batch), batch_sz)):
        predictor = SAM2ImagePredictor(sam2_model)
        predictor.set_image_batch(img_batch[i:i+batch_sz])
        masks_batch, scores_batch, _ = predictor.predict_batch(
            None,
            None, 
            box_batch=boxes_batch[i:i+batch_sz], 
            multimask_output=False
        )
        output += masks_batch

        # clean ram
        del predictor
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    
    return output,output_name


import argparse

def main(args):
    output, output_name = refine_images(args.image_dir, args.mask_dir, args.batch_sz)

    for i in range(len(output)):
        np.save(os.path.join(args.output_path,output_name[i]), output[i])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="process vague mask input to better segment mask")
    
    # Define command-line arguments
    parser.add_argument("image_dir", type=str, help="Path to the image file")
    parser.add_argument("mask_dir", type=str, help="Path to the original mask file")
    parser.add_argument("batch_sz", type=str, help="batch size for processing")
    parser.add_argument("output_path", type=str, help="Path to the folder save output")

    # Parse arguments
    args = parser.parse_args()

    # Run the main function
    main(args)