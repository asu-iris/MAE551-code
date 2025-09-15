import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html



linear = nn.Linear(5, 3)

x1 = torch.randn(10, 5)   # batch of 10
y1 = linear(x1)

x2 = torch.randn(5)       # just a single vector
y2 = linear(x2)

for param in linear.parameters():
    print(param.shape)