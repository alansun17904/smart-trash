import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F


def Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 image input channel, 6 output channels, 3x3 convo
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # linear transformation y = mx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        # 5 classes: landfill, plastic, paper, cans, tetrapack
        self.fc3 = nn.Linear(84, 5)
