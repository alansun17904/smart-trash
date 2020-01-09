import torch
import torch.nn as nn
import torch.functional as F


def Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 image input channel, 6 output channels, 3x3 convo
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # linear transformation y = mx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # from 6 x 6 image dimension
        self.fc2 = nn.Linear(120, 84)
        # 5 classes: landfill, plastic, paper, cans, tetrapack
        self.fc3 = nn.Linear(84, 5)

    def forward(self):
        # Max pooling over a 2 x 2 window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=2)  # 2d matrix
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
