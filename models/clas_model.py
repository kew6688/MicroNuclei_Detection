import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights, resnet101
from collections import OrderedDict

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class MNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2048, 1000),
            nn.ReLU(),
            nn.Linear(1000, 2)
        )
        self.model = resnet101(weights='IMAGENET1K_V1')
        classifier = nn.Sequential(OrderedDict([
                ('fc', self.linear_relu_stack)
            ]))
        self.model.fc = classifier

    def forward(self, x):
        x = self.model(x)
        return x