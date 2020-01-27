import os
import torch
import torchvision
import torchvision.datasets as ds
import torchvision.transforms as transforms


# Data Augmentation and Normalization for Training
def generate_transforms(data_dir, input_size):
    return {
        'train': transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize(300),
            transforms.RandomResizedCrop(input_size),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize(300),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize(300),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }


def datasets(data_dir, input_size, batch_size):
    image_datasets = {x: ds.ImageFolder(os.path.join(data_dir, x),
                         transform=generate_transforms(data_dir, input_size)[x])
                         for x in ['train', 'test', 'valid']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                      batch_size=batch_size, shuffle=True, num_workers=4)
                      for x in ['train', 'test', 'valid']}
    dataset_sizes = {x: len(image_datasets[x])
                        for x in ['train', 'test', 'valid']}
    class_names = image_datasets['train'].classes
    return (image_datasets, dataloaders, dataset_sizes, class_names)
