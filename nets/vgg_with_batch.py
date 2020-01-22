import torch.nn as nn
from torchvision import models


model = models.vgg16_bn(pretrained=True)
model.classifier[6] = nn.Sequential(
                      nn.Linear(512 * 7 * 7, 4096, bias=True),
                      nn.BatchNorm1d(num_features=4096),
                      nn.ReLU(),
                      nn.Dropout(0.5),
                      nn.Linear(4096, 265),
                      nn.BatchNorm1d(num_features=265),
                      nn.ReLU(),
                      nn.Dropout(0.5),
                      nn.Linear(256, 5),
                      nn.Softmax(dim=0))


# count = 0
# for param in model.parameters():
#     count += 1

# total_params = sum(p.numel() for p in model.parameters())
# print(total_params)
