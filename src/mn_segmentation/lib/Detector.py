import torch
from PIL import Image

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