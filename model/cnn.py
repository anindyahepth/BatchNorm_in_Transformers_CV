import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.transforms.v2 as transforms_v2
import torchvision
import torchvision.datasets as dsets
import torch.optim as optim
import torch.nn.init as init

class CNN_DO(nn.Module):
    def __init__(self):
        super(CNN_DO, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.drop2= nn.Dropout(p=0.5)
        self.conv3 = nn.Conv2d(32,64, kernel_size=5)
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.drop3= nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(3*3*64, 256)
        self.actfc = nn.ReLU()
        self.dropfc = nn.Dropout(p=0.5) 
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.pool2(self.act2(x))
        x = self.drop2(x)
        x = self.conv3(x)
        x = self.pool3(self.act3(x))
        x = self.drop3(x)
        z = x.view(-1,3*3*64)
        x = z
        x = self.actfc(self.fc1(x))
        x = self.dropfc(x)
        x = self.fc2(x)
        return x, z

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.norm1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.norm2 = nn.BatchNorm2d(32)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(32,64, kernel_size=5)
        self.norm3 = nn.BatchNorm2d(64)
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(3*3*64, 256)
        self.normfc = nn.BatchNorm1d(256)
        self.actfc = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.norm1(self.conv1(x))
        x = self.act1(x)
        x = self.norm2(self.conv2(x))
        x = self.pool2(self.act2(x))
        x = self.norm3(self.conv3(x))
        x = self.pool3(self.act3(x))
        x = x.view(-1,3*3*64)
        x = self.normfc(self.fc1(x))
        x = self.actfc(x)
        x = self.fc2(x)
        return x


 
