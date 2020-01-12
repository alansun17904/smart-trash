import torch
import torch.nn as nn
<<<<<<< HEAD
import torch.optim as optim
import torch.nn.functional as F
=======
import torch.functional as F
>>>>>>> 1350a4d007f79a349e028e99e18dfb60b56a8b0e


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # IMAGES ARE 512X384
        # 1 image input channel, 6 output channels, 3x3 convo
        self.conv1 = nn.Conv2d(3, 6, 10, 2)
        self.pool1 = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(6, 8, 10, 2)
        self.pool2 = nn.MaxPool2d(4, 4)
        self.conv3 = nn.Conv2d(8, 8, 10, 2)
        # linear transformation y = mx + b
<<<<<<< HEAD
        self.fc1 = nn.Linear(8 * 52 * 36, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 64)
        # 5 classes: landfill, plastic, paper, cans, tetrapack
        self.fc6 = nn.Linear(64, 5)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = x.view(-1, 8 * 52 * 36)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        m = nn.Softmax(dim=0)
        return m(self.fc(6))
=======
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
>>>>>>> 1350a4d007f79a349e028e99e18dfb60b56a8b0e
