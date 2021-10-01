import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F



class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.head = models.resnet50(pretrained=True)
        # Remove classification layers so that we are able to add our own CNN layers
        self.head.requires_grad = False
        self.head.fc = nn.Sequential(
                                nn.Linear(2048, 1024, bias=True),
                                nn.BatchNorm1d(1024),
                                nn.ReLU(),
                                nn.Dropout(0.25))
        self.fc1 = nn.Sequential(
                                nn.Linear(1024, 512, bias=True),
                                nn.BatchNorm1d(512),
                                nn.ReLU(),
                                nn.Dropout(0.50),
                                nn.Linear(512, 5, bias=True),
                                nn.ReLU()
        )
        self.meta_fc1 = nn.Linear(21, 16, bias=True)
        self.meta_relu1 = nn.ReLU()
        self.bilinear = nn.Bilinear(16, 1024, 1024)

    def forward(self, image, metadata):
        metadata = self.meta_fc1(metadata)
        metadata = self.meta_relu1(metadata)
        image = self.head(image)
        combined = self.bilinear(metadata, image)
        combined = self.meta_relu1(combined)
        return self.fc1(combined)

    def num_flat_features(self, x):
        """
        https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py
        """
        size = x.size()[1:]  # get all dimensions except for batch size
        features = 1
        for s in size:
            features *= s
        return features