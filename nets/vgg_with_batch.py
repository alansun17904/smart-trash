import torch.nn as nn
from torchvision import models


model = models.vgg16_bn(pretrained=True)
model.classifier[6] = nn.Sequential(
                      nn.Linear(512 * 7 * 7, 256, bias=True),
                      nn.BatchNorm1d(num_features=4096),
                      nn.ReLU(),
                      nn.Dropout(0.5),
                      nn.Linear(256, 6),
                      nn.Softmax(dim=0))


count = 0
vgg = next(model.children())
for param in vgg:
    if count <= 39:
        print(param)
        param.requires_grad = False
    count += 1
