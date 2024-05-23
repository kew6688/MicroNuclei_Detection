import os
from os import listdir
from os.path import isfile, join
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torchvision.transforms import v2

class NucRecDataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        # read files in the directory, this path is hard coding
        self.img_labels = []
        mn_path = "/home/y3229wan/projects/def-sushant/y3229wan/mn-project/Data/NucRec/NucReg Dataset/Micronuclie cells"
        nuc_path = "/home/y3229wan/projects/def-sushant/y3229wan/mn-project/Data/NucRec/NucReg Dataset/Normal Cells"
        for f in listdir(mn_path):
            self.img_labels.append((join(mn_path,f), 0))
        for f in listdir(nuc_path):
            self.img_labels.append((join(nuc_path,f), 1))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels[idx][0]
        image = read_image(img_path)
        label = self.img_labels[idx][1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

if __name__ == "__main__":

    H, W = 32, 32
    img = torch.randint(0, 256, size=(3, H, W), dtype=torch.uint8)

    transforms = v2.Compose([
        transforms.Lambda(lambda x: x[:3,:,:]),
        v2.Grayscale(),
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = NucRecDataset(transforms)
    
    s = set()
    cnt = 0
    for img,label in dataset:
        img = img[:3,:,:]
        s.add(img.shape[0])
        if img.shape[0] == 4:
            print(img[:,100,100])
            break
    print(s)