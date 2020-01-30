import torch
import time
import copy
import torch.nn as nn
import numpy as np
from torchvision import models
import torch.optim as optim
from data_process import datasets


# operating device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# dataloaders
dataset = datasets('data/split-garbage-dataset', 299, 16)
image_datasets = dataset[0]
dataloaders = dataset[1]
dataset_sizes = dataset[2]

# defining classifiction layers
model = models.inception_v3(pretrained=True)

for param in model.children():
    param.requires_grad = False # freeze all transfered weights

model.AuxLogits.fc = nn.Sequential(
                     nn.Linear(768, 256, bias=True),
                     nn.BatchNorm1d(256),
                     nn.Dropout(0.3),
                     nn.ReLU(),
                     nn.Linear(256, 6))
model.fc = nn.Sequential(
           nn.Linear(2048, 256, bias=True),
           nn.BatchNorm1d(256),
           nn.Dropout(0.3),
           nn.ReLU(),
           nn.Linear(256, 128, bias=True),
           nn.BatchNorm1d(128),
           nn.Dropout(0.3),
           nn.ReLU(),
           nn.Linear(128, 6))
