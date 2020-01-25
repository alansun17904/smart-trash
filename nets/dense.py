import os
import torch
import time
import copy
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import models, transforms, datasets
import torch.optim as optim


model = models.densenet201(pretrained=True)
model.classifier = nn.Sequential(
                      nn.Linear(1920, 256, bias=True),
                      nn.BatchNorm1d(num_features=256),
                      nn.ReLU(),
                      nn.Dropout(0.6),
                      nn.Linear(256, 6),
                      nn.Softmax(dim=0))
print(model.classifier)

c = 0
dense = next(model.children())
for param in dense:
    print(param)
    c += 1

print(c)
