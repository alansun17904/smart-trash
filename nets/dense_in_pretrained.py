"""
Dense-Net 121
Instead of using the weights directly from trained from ImageNet,
we used the same architecture as dense net. However, we performed
'pretraining'. As our dataset is quite different from ImageNet clasess.

Pretraining Epoch -- 10
!Untested
"""


import os
import torch
import time
import copy
from datetime import datetime
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from torchvision import datasets as ds
from data_process import datasets
import torch.optim as optim


# define training GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transforms = transforms.Compose([
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])])
trainset = ds.ImageNet(root='data/', split='train', download=True)
trainloader = torch.utils.data.DataLoader(trainset, train=True, batch_size=128,
                                          shuffle=True, num_workers=4)

model = models.densenet121(pretrained=False)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
pretrain = 10

# pretraining
for epoch in range(pretrain):
    for data in trainloader:
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
print('Finished Pretraining...')

# Freeze all layers
for param in model.children():
    param.requires_grad = False

# Reset the size of classification layer
model.classifier = nn.Sequential(
                      nn.Linear(1024, 6, bias=True),
                      # nn.BatchNorm1d(num_features=256),
                      # nn.ReLU(),
                      # nn.Dropout(0.6),
                      # nn.Linear(256, 6, bias=True),
                      nn.Softmax(dim=0))

# Generate Trash Net Training Data
image_datasets = datasets('data/split-garbage-dataset', 224, 8)[0]
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=0.0001, momentum=0.9)
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.
for epoch in range(200):
    print('Epoch: {} {} {:.4f}'.format(epoch, '*' * 10, best_acc))
    for phase in ['train', 'valid']:
        running_loss = 0.
        num_correct = 0
        for inputs, labels in image_datasets[phase]:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward feezing and backpropagation
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            if phase == 'train':
                loss.backward()
                optimizer.step()

            # add to stats
            running_loss += loss.item()
            num_correct += torch.sum(preds==labels.data)

        epoch_loss = run_loss / len(dataset_sizes[phase])
        epoch_acc = correct.double() / len(dataset_sizes[phase])
        print('{} Loss: {:.4f}, Acc: {:.4f}'.format(phase.title(),
              epoch_loss, epoch_acc))

        if phase == 'valid' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

print('Training Complete')
print('Best Validation Accuracy: {:.4f}'.format(best_acc))

# Saving model with time stamp
date = datetime.now()
timestamp = date.strftime('%Y%m%d-%H:%M')
torch.save(best_model_wts, f'nets/config/recycle_vgg{timestamp}.pth')
print(f'Model Saved: nets/config/recycle_vgg{timestamp}.pth')
