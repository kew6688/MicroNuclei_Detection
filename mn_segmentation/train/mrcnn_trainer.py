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

def trainer_mrcnn(model, epoch, data_loader=None, optimizer=None):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and mn
    num_classes = 2
    # use our dataset and defined transformations
    dataset = mnMaskDatasetFinal('mnMask/data', get_transform(train=True))
    dataset_test = mnMaskDatasetFinal('mnMask/data', get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-100])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-100:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=utils.collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=16,
        shuffle=False,
        collate_fn=utils.collate_fn
    )

    # get the model using our helper function
    model = model

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(params)

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
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")