import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1)
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1)
        self.conv8 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1)
        self.conv9 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=3)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1)
        self.conv11 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1)
        #nn.lin1 = nn.Linear(in_features=)
        #nn.lin2 = nn.Linear(in_features=)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.maxpool(x)
        x = self.conv11(x)
        x = self.conv6(x)
        x = self.conv10(x)
        x = self.conv7(x)
        x = self.maxpool(x)
        return x
