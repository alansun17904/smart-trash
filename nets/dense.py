import os
import torch
import time
import copy
import torch.nn as nn
import numpy as np
from torchvision import models
from data_process import datasets
import torch.optim as optim


def shutdown(model, num):
    """
    freezes the first `num` amount of layers in dense net
    """
    dense = next(model.children())
    threshold = len(dense) - num
    c = 0
    for param in dense:
        c += 1
        if c < threshold:
            param.requires_grad = False
    return model

# operating device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.densenet201(pretrained=True)
model.classifier = nn.Sequential(
                      nn.Linear(1920, 256, bias=True),
                      nn.BatchNorm1d(num_features=256),
                      nn.ReLU(),
                      nn.Dropout(0.6),
                      nn.Linear(256, 6),
                      nn.Softmax(dim=0))


dataset = datasets('data/split-garbage-dataset', 224, 32)
image_datasets = dataset[0]
dataloaders = dataset[1]
dataset_sizes = dataset[2]

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
    if epoch == 15:
        model = shutdown(model, 5)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    elif epoch == 20:
        model = shutdown(model, 11)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    print(f'Epoch: {epoch + 1}\n{"-" * 10}')
    print(f'Best Val: {best_acc}')
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
torch.save(best_model_wts, 'nets/config/recycle_dense.pth')
print('Model saved in nets/config/recycle_dense.pth')
