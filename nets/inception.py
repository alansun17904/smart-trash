import torch
import time
import copy
import torch.nn as nn
import numpy as np
from torchvision import models
import torch.optim as optim
from data_process import data_dir, data_transforms, image_datasets, dataloaders
from data_process import dataset_sizes, device


model = models.inception_v3(pretrained=True)
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
