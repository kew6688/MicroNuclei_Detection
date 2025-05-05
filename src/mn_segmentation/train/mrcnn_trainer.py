from dataset import get_transform, mnMaskDatasetFinal
import os
import torch
import numpy as np
import utils

from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import v2 as T
from engine import train_one_epoch, evaluate

from mn_segmentation.datasets.mnMask import mnMaskDataset
from mn_segmentation.tests.evaluator import Evaluator, evaluate_mn_dataset
from mn_segmentation.lib.Application import Application

def trainer_mrcnn(name, model, epoch, dataset_path, cp_path, transform, data_loader=None, optimizer=None, batch_sz=8, num_worker=0):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if not os.path.exists(cp_path):
        os.makedirs(cp_path)

    # our dataset has two classes only - background and mn
    num_classes = 2

    if data_loader is None:
      # use our dataset and defined transformations
      dataset = mnMaskDataset(dataset_path, transform(train=True))
      dataset_test = mnMaskDataset(dataset_path, transform(train=False))

      # split the dataset in train and test set
      # indices = torch.randperm(len(dataset)).tolist()
      # indices = list(range(len(dataset))
      # dataset = torch.utils.data.Subset(dataset, indices[:-300])
      # dataset_test = torch.utils.data.Subset(dataset_test, indices[-300:])

      # define training and validation data loaders
      data_loader = torch.utils.data.DataLoader(
          dataset,
          batch_size=batch_sz,
          shuffle=True,
          collate_fn=utils.collate_fn,
          num_workers=num_worker
      )

      data_loader_test = torch.utils.data.DataLoader(
          dataset_test,
          batch_size=4,
          shuffle=False,
          collate_fn=utils.collate_fn
      )

    # get the model using our helper function
    model = model

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)

    # optimizer = torch.optim.SGD(
    #     params,
    #     lr=0.005,
    #     momentum=0.9,
    #     weight_decay=0.0005
    # )

    # and a learning rate scheduler
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer,
    #     step_size=3,
    #     gamma=0.1
    # )

    num_epochs = epoch

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        # lr_scheduler.step()
        # evaluate on the test dataset
        # evaluate(model, data_loader_test, device=device)

        if (epoch+1)%10 == 0:
          print(f"++++++++++++ Save model {name}-{epoch+1} +++++++++++++++++")
          path = os.path.join(cp_path, f'{name}-{epoch+1}.pt')
          torch.save(model.state_dict(), path)

          app = Application(model=model, device=torch.device('cuda'))
          for i in range(7,8):
            print(f"**********************************************")
            print(f"**** conf: {i/10} ************************")
            print("**********************************************")
            evaluator = Evaluator()
            for j in range(5,6):
              print(f"*********** mask_conf: {j/10} ************")
              evaluate_mn_dataset(app, dataset_path, evaluator, conf=i/10, mask_conf=j/10, ap_iou=0.5)
    print("That's it!")