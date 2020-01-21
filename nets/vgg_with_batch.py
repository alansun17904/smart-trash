import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url


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
    'D': [(64, 'F'), (64, 'F'), ('M', ''), (128, 'F'), (128, 'F'),
          ('M', ''), (256, ''), (256, ''), (256, ''), ('M', ''),
          (512, ''), (512, ''), (512, ''), ('M', ''),
          (512, ''), (512, ''), (512, ''), ('M', '')],
    'E': [(64, 'F'), (64, 'F'), ('M', ''), (128, 'F'), (128, 'F'),
          ('M', ''), (256, 'F'), (256, 'F'), (256, 'F'), (256, 'F'),
          ('M', ''), (512, ''), (512, ''), (512, ''), (512, ''),
          ('M', ''), (512, ''), (512, ''), (512, ''), (512, ''), ('M', '')]
}


class VGG(nn.Module):
    def __init__(self, features, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 256),
            nn.Dropout(0.6),
            nn.ReLU(inplace=True),
            nn.Linear(256, 5)
        )
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

    @staticmethod
    def make_layers(cfg, batch_norm=False):
        layers = []
        in_channels = 3  # RGB
        for v in cfg:
            # (layer type, freeze)
            if v[0] == 'M':
                layers += [nn.MaxPool2d(2, 2)]  # kernel size: 2, stride: 2
            else:
                conv2d = nn.Conv2d(in_channels, v[0], kernel_size=3, padding=1)
                if v[1] == 'F':
                    conv2d.weight.requires_grad = False
                    conv2d.bias.requires_grad = False
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v[0]), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v[0]
        return nn.Sequential(*layers)


if __name__ == '__main__':
    net = VGG(VGG.make_layers(cfgs['D'], batch_norm=True), init_weights=True)
    print(net)
