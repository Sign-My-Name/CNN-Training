import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# DEVICE = torch.devide('cuda' if torch.cuda.is_available() else 'cpu')


class CNN(nn.Module):
    def __init__(self, num_classes):
        print('hello world')
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=9, padding=3)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=9, padding=3)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(246016, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.maxpool1(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.maxpool2(out)

        print(out.shape)
        out = out.reshape(out.size(0), -1)
        print(out.shape)

        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)

        return out


