"""
VGG16 (Batch Normalization)
Transfer learning with VGG16 weights, and minimal adjument to the first
3 layers. This is our best performing model with 93.29% accuracy.
"""


import torch
import time
from datetime import datetime
import copy
import torch.nn as nn
import numpy as np
from torchvision import models
import torch.optim as optim
from data_process import datasets


# operating device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# dataloaders
dataset = datasets('data/split-random', 224, 16)
image_datasets = dataset[0]
dataloaders = dataset[1]
dataset_sizes = dataset[2]

# defining classification layers
model = models.vgg16_bn(pretrained=True)

for param in model.children():
    param.requires_grad = False

model.classifier[6] = nn.Sequential(
                      nn.Linear(4096, 1024, bias=True),
                      nn.BatchNorm1d(1024),
                      nn.Dropout(.25),
                      nn.ReLU(),
                      nn.Linear(1024, 512, bias=True),
                      nn.BatchNorm1d(512),
                      nn.Dropout(.5),
                      nn.ReLU(),
                      nn.Linear(512, 4, bias=True),
                      nn.LogSoftmax(dim=0))

# freezing layers that do not need training
# c = 0
# vgg = next(model.children())
# for param in vgg:
#     if c <= 39:
#         if hasattr(param, 'weight') and hasattr(param, 'bias'):
#             param.weight.requires_grad = False
#             param.bias.requires_grad = False
#         param.requires_grad = False
#
#     c += 1

# try to load model onto the gpu if the training mode is gpu
try:
    model.cuda()
except AssertionError:
    pass

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

since = time.time()
best_acc = 0.
best_model_wts = copy.deepcopy(model.state_dict())

for epoch in range(200):
    print('Epoch: {}{}Best Val: {:.4f}'.format(epoch + 1, '-' * 10, best_acc))
    for phase in ['train', 'valid']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        run_loss = 0.
        correct = 0

        for inputs, labels in dataloaders[phase]:
            # training on gpu
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the optimizer
            optimizer.zero_grad()

            # forward feeding
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # add to stats
            run_loss += loss.item() * inputs.size(0)
            correct += torch.sum(preds == labels.data)

        epoch_loss = run_loss / dataset_sizes[phase]
        epoch_acc = correct.double() / dataset_sizes[phase]
        print('{} Loss: {:.4f}, Acc: {:.4f}'.format(phase.title(),
              epoch_loss, epoch_acc))

        if phase == 'valid' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60,
                                                    time_elapsed % 60))
print('Best val acc: {:4f}'.format(best_acc))
date = datetime.now()
timestamp = date.strftime('%Y%m%d-%H:%M')
torch.save(best_model_wts, f'nets/config/recycle_vgg{timestamp}.pth')
print(f'Model saved in nets/config/recycle_vgg{timestamp}.pth')
