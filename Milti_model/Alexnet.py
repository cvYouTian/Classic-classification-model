# -*- coding: UTF-8 -*-


import torch.nn as nn
import torch


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weight=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # [48, 55, 55]
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            # [48, 27, 27]
            nn.MaxPool2d(kernel_size=3, stride=2),
            # [128, 27, 27]
            nn.Conv2d(in_channels=48, out_channels=128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # [128, 13, 13]
            nn.MaxPool2d(kernel_size=3, stride=2),
            # [192, 13, 13]
            nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # [192, 13, 13]
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # [128, 13, 13]
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # [128, 6, 6]
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.flatten = nn.Flatten()
        self.dense = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128*6*6, 2048),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weight:
            self._initialize_weight()

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

    def _initialize_weight(self):
        for m in self.modules():
            # 如果m是卷积类
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # 如果m是线性类
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


