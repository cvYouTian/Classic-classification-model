import cv2
import torch
import math
from pathlib import Path
from PIL import Image
from torchvision import transforms
import numpy as np
from torch import nn
from torchsummary import summary
import matplotlib.pyplot as plt


class ResBlock(nn.Module):
    def __init__(self, in_channel=None, out_channel=None, ex_channel=False):
        super(ResBlock, self).__init__()
        self.ex_channel = ex_channel
        self.conv_non = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(3, 3), stride=(1, 1), padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=(3, 3), stride=(1, 1), padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.conv_with = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(3, 3), stride=(2, 2), padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=(3, 3), stride=(1, 1), padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        if self.ex_channel:
            x1 = self.conv_with(x)
            out = x1 + self.projection(identity)
            out = self.relu(out)
        else:
            x1 = self.conv_non(x)
            out = x1 + identity
            out = self.relu(out)

        return out


# [224, 224, 3]
class ResNet34(nn.Module):
    def __init__(self, num_class=5):
        super(ResNet34, self).__init__()
        # [112, 112, 64]

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        # [56, 56, 64]
        self.maxpooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # [56, 56, 64]
        self.conv2_x = nn.Sequential(
            ResBlock(in_channel=64, out_channel=64),
            ResBlock(in_channel=64, out_channel=64),
            ResBlock(in_channel=64, out_channel=64)
        )
        # [28, 28, 128]
        self.conv3_x = nn.Sequential(
            ResBlock(in_channel=64, out_channel=128, ex_channel=True),
            ResBlock(in_channel=128, out_channel=128),
            ResBlock(in_channel=128, out_channel=128),
            ResBlock(in_channel=128, out_channel=128)
        )
        # [14, 14, 256]
        self.conv4_x = nn.Sequential(
            ResBlock(in_channel=128, out_channel=256, ex_channel=True),
            ResBlock(in_channel=256, out_channel=256),
            ResBlock(in_channel=256, out_channel=256),
            ResBlock(in_channel=256, out_channel=256),
            ResBlock(in_channel=256, out_channel=256),
            ResBlock(in_channel=256, out_channel=256)
        )
        # [7, 7, 512]
        self.conv5_x = nn.Sequential(
            ResBlock(in_channel=256, out_channel=512, ex_channel=True),
            ResBlock(in_channel=512, out_channel=512),
            ResBlock(in_channel=512, out_channel=512)
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=num_class)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.maxpooling(x1)
        x3 = self.conv2_x(x2)
        x4 = self.conv3_x(x3)
        x5 = self.conv4_x(x4)
        x6 = self.conv5_x(x5)

        x7 = self.gap(x6)
        x8 = torch.flatten(x7, 1)
        out = self.fc(x8)
        logits = torch.sigmoid(out)

        # return out
        return logits


if __name__ == '__main__':
    image_path = "/home/youtian/Pictures/Screenshots/bird.png"
    net = ResNet34(num_class=5)
    net.to("cuda:0")
    net.eval()
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor()
    ])
    # image = Image.open(image_path)
    image = cv2.imread(image_path)
    image = Image.fromarray(image)
    image = transform(image)

    batch_input = image.unsqueeze(0).to("cuda:0")

    summary(net, input_size=(3, 224, 224))

