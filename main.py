from data_loader import *
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from torchvision.models import resnet50, ResNet50_Weights
import time
from train import train
from test import test


def main():
    # TODO: Define transform
    transforms = v2.Compose([
        transforms.Lambda(lambda x: x[:3,:,:]),
        v2.Grayscale(3),
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    training_data = NucRecDataset(transforms)

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    # test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    # Display image and label.
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(torch.permute(img, (1,2,0)))
    plt.savefig("test.png")
    print(f"Label: {label}")


    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # Initialize model
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)

    # Set model to eval mode
    model.eval()
    print(model)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    start = time.time()
    epochs = 1
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(train_dataloader, model, loss_fn)
    torch.save(model.state_dict(), "output/resnet50")
    print(f"spend {(time.time() - start)/60} seconds")
    print("Done!")

if __name__ == "__main__":
    main()