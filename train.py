import torch.nn as nn
import torch.optim as optim
from network import Net


optimizer = optim.SGD(net.parameters(), lr=0.05)
