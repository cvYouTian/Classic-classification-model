# -*- coding: UTF-8 -*-
import torch
from torch import nn


class Vgg16(nn.Module):
    def __init__(self, num_class=5, init_weight=False):
        super(Vgg16, self).__init__()
        self.feature = nn.Sequential(
            # [224, 224, 64]
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(True),
            # [224, 224, 64]
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(True),
            # [112, 112, 64]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [112, 112, 128]
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(True),
            # [56, 56, 128]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [56, 56, 256]
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(True),
            # [28, 28, 256]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [28, 28, 512]
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(True),
            # [14, 14, 512]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [14, 14, 512]
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(True),
            # [7, 7, 512]
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=7*7*512, out_features=4096),
            nn.Dropout(0.5),
            nn.ReLU(True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.Dropout(0.5),
            nn.ReLU(True),
            nn.Linear(in_features=4096, out_features=num_class),
        )
        if init_weight:
            self._initialize_weights()

    def forward(self, x):
        x = self.feature(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        # 这里都没有使用softmax层来做是因为在计算损失的时候就已经集成了softmax函数了
        # x = torch.softmax(x, dim=-1)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    net = Vgg16(num_class=5)
    print(net)






