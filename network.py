import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


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


if __name__ == '__main__':
    net = Net()
    print(net)

