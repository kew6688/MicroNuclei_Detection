from data_loader import *
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from torchvision.models import resnet50, ResNet50_Weights, resnet101
import time
from train import train
from test import test

# resnet on NucRec
def exp1():
    # TODO: Define transform
    transform = v2.Compose([
        transforms.Lambda(lambda x: x[:3,:,:]),
        # make the image gray
        # v2.Grayscale(3),
        # or make the gray image to RGB
        transforms.Lambda(lambda x: x.expand(3,-1,-1)),
        # v2.RandomResizedCrop(size=(224, 224), antialias=True),
        transforms.Resize(size = (224,224)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    training_data = NucRecDataset(transform)

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    # test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # print model architecture
    model = resnet101(weights='IMAGENET1K_V1')
    print(model)

    # cross valid model
    total_size = len(training_data)
    k_fold = 5
    for fold in range(k_fold):
        train_dataset, valid_dataset = random_split(training_data, (0.7, 0.3))

        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

        # Initialize model
        # weights = ResNet50_Weights.DEFAULT
        # model = resnet50(weights=weights)
        model = resnet101(weights='IMAGENET1K_V1')

        # Set model to eval mode
        model.eval()
        # print(model)
        model.to(device)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())

        # Train the model
        print(f"============= fold {fold} Start! =============")
        start = time.time()
        epochs = 20
        for t in range(epochs):
            # print(f"Epoch {t+1}\n-------------------------------")
            train(train_dataloader, model, loss_fn, optimizer, device)
        test(train_dataloader, model, loss_fn, device)
        test(valid_dataloader, model, loss_fn, device)
        # torch.save(model.state_dict(), "/home/y3229wan/projects/def-sushant/y3229wan/mn-project/MN_Classification/output/resnet101")
        print(f"spend {(time.time() - start)/60} seconds")
        print(f"============= fold {fold} Done! ============= \n")


    

if __name__ == "__main__":
    exp1()