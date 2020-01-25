import os
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms


data_dir = 'data/split-garbage-dataset'

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
