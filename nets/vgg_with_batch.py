import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import load_state_dict_from_url


model_urls = {
    'vgg16-bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19-bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}


# configurations from:
# https://arxiv.org/pdf/1409.1556.pdf
# make each layer a tuple and have 'F' represent frozen layer
cfgs = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
          512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M',
          512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGG(nn.modules):
    def __init__(self, features, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential([
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm2d(4096),  # was nn.ReLU()
            nn.Linear(4096, 256),
            nn.Dropout(0.6),
            nn.Linear(256, 5)
        ])
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(1, x)
        x = self.classifier(x)
        return F.softmax(x, dim=0)

    def _initialize_weights(self):
        """
        Generates random weights based on whether layer is convolution,
        batch, or linear.
        https://pytorch.org/docs/stable/_modules/torchvision/models/vgg.html
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
