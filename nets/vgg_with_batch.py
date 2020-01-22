import os
import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


data_dir = 'nets/split-garbage-dataset'

# Data Augmentation and Normalization for Training
data_transforms = {
    'train': transforms.Compose([
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

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
               shuffle=True, num_workers=4) for x in ['train', 'test', 'valid']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test', 'valid']}
class_names = image_datasets['train'].classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(class_names)

model = models.vgg16_bn(pretrained=True)
model.classifier[6] = nn.Sequential(
                      nn.Linear(512 * 7 * 7, 256, bias=True),
                      nn.BatchNorm1d(num_features=4096),
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
