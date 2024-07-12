from MN.mn_classification.data_loader import *
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from torchvision.models import resnet50, ResNet50_Weights, resnet101
import time
from MN.mn_classification.train import train
from MN.mn_classification.test import test
from MN.models.clas_model import MNClassifier

# resnet on NucRec
def exp1():
    """
    Run pretrained model on NucRec dataset
    """
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
        # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    mn_path = "/home/y3229wan/scratch/NucRec/NucReg Dataset/Micronuclie cells"
    nuc_path = "/home/y3229wan/scratch/NucRec/NucReg Dataset/Normal Cells"

    training_data = NucRecDataset(mn_path, nuc_path, transform)

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
    # model = resnet101(weights='IMAGENET1K_V1')
    model = MNClassifier()
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
        # model = resnet101(weights='IMAGENET1K_V1')
        model = MNClassifier()

        # Set model to eval mode
        model.eval()
        # print(model)
        model.to(device)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())

        # Train the model
        print(f"============= fold {fold} Start! =============")
        start = time.time()
        epochs = 40
        for t in range(epochs):
            # print(f"Epoch {t+1}\n-------------------------------")
            train(train_dataloader, model, loss_fn, optimizer, device)
        print("Test on training set:")
        test(train_dataloader, model, loss_fn, device)
        print("Test on validation set:")
        test(valid_dataloader, model, loss_fn, device)
        torch.save(model.state_dict(), f"/home/y3229wan/projects/def-sushant/y3229wan/mn-project/MN/models/MNclassifier_{fold}")
        print(f"spend {(time.time() - start)/60} seconds")
        print(f"============= fold {fold} Done! ============= \n")


def exp2():
    """
    Test classification accuracy on the Kate dataset, model is pretrained by NucRec
    """
    # model = resnet101()
    # model.load_state_dict(torch.load("/home/y3229wan/projects/def-sushant/y3229wan/mn-project/MN/models/resnet101"))
    model = MNClassifier()
    model.load_state_dict(torch.load("/home/y3229wan/projects/def-sushant/y3229wan/mn-project/MN/models/MNclassifier_4"))
    model.eval()

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

    mn_path = "/home/y3229wan/scratch/micronuclei"
    nuc_path = "/home/y3229wan/scratch/nuclei"

    training_data = NucRecDataset(mn_path, nuc_path, transform)

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    model.to(device)
    print(model)

    test(train_dataloader, model, loss_fn, device)

    # cross valid model
    total_size = len(training_data)
    k_fold = 1
    for fold in range(k_fold):
        train_dataset, valid_dataset = random_split(training_data, (0.9, 0.1))

        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

        # Initialize model
        # weights = ResNet50_Weights.DEFAULT
        # model = resnet50(weights=weights)
        # model = torch.load("/home/y3229wan/projects/def-sushant/y3229wan/mn-project/MN/models/resnet101")

        model.to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())

        # Train the model
        print(f"============= fold {fold} Start! =============")
        start = time.time()
        epochs = 40
        for t in range(epochs):
            # print(f"Epoch {t+1}\n-------------------------------")
            train(train_dataloader, model, loss_fn, optimizer, device)
        test(train_dataloader, model, loss_fn, device)
        test(valid_dataloader, model, loss_fn, device)
        print(f"spend {(time.time() - start)/60} seconds")
        print(f"============= fold {fold} Done! ============= \n")
    
    torch.save(model.state_dict(), "/home/y3229wan/projects/def-sushant/y3229wan/mn-project/MN/output/best.pt")

def val():
    """
    Test classification accuracy on the Kate dataset, model is pretrained by NucRec
    """
    # model = resnet101()
    # model.load_state_dict(torch.load("/home/y3229wan/projects/def-sushant/y3229wan/mn-project/MN/models/resnet101"))
    model = MNClassifier()
    model.load_state_dict(torch.load("/home/y3229wan/projects/def-sushant/y3229wan/mn-project/MN/output/best.pt"))
    model.eval()

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    mn_path = "/home/y3229wan/scratch/micronuclei"
    nuc_path = "/home/y3229wan/scratch/nuclei"

    training_data = NucRecDataset(mn_path, nuc_path, transform)

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    model.to(device)
    print(model)

    test(train_dataloader, model, loss_fn, device)


if __name__ == "__main__":
    # exp1()
    # exp2()
    val()