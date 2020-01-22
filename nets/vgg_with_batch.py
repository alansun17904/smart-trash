import os
import torch
import time
import copy
import torch.nn as nn
import numpy as np
from torchvision import models
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


data_dir = 'nets/split-garbage-dataset'

# Data Augmentation and Normalization for Training
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                  transform=data_transforms[x])
                  for x in ['train', 'test', 'valid']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
               shuffle=True, num_workers=4) for x in ['train', 'test', 'valid']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test', 'valid']}
class_names = image_datasets['train'].classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# def imshow(inp, title=None):
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.pause(10)  # pause a bit so that plots are updated
#
# inputs, classes = next(iter(dataloaders['train']))
# out = torchvision.utils.make_grid(inputs)
# imshow(out, title=[class_names[x] for x in classes])

model = models.vgg16_bn(pretrained=True)
model.classifier[6] = nn.Sequential(
                      nn.Linear(4096, 256, bias=True),
                      nn.BatchNorm1d(num_features=256),
                      nn.ReLU(),
                      nn.Dropout(0.6),
                      nn.Linear(256, 6),
                      nn.Softmax(dim=0))


count = 0
vgg = next(model.children())
for param in vgg:
    if count <= 39:
        param.requires_grad = False
    count += 1


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), eps=0.5)

since = time.time()
best_acc = 0.
best_model_wts = copy.deepcopy(model.state_dict())

for epoch in range(15):
    print(f'Epoch: {epoch + 1}\n{"-" * 10}')
    for phase in ['train', 'valid']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        run_loss = 0.
        correct = 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            lables = labels.to(device)

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
            print(correct)

        epoch_loss = run_loss / dataset_sizes[phase]
        epoch_acc = correct.double() / dataset_sizes[phase]
        print('{} Loss: {:.4f}, Acc: {:.4f}'.format(phase.title(),
              epoch_loss, epoch_acc))

        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60,
                                                    time_elapsed % 60))
print('Best val acc: {:4f}'.format(best_acc))
