import torch
from torch import nn


class GoogLenet(nn.Module):
    def __init__(self, num_classes=5, aux=True):
        super(GoogLenet, self).__init__()
        self.aux = aux
        self.feature = nn.Sequential(
            # [112, 112, 64]
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=3),
            nn.ReLU(True),
            # [56, 56, 64]
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, ceil_mode=True),
            # [56, 56, 64]
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(True),
            # [56, 56, 192]
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(3,3), stride=(1, 1), padding=1),
            nn.ReLU(True),
            # [28, 28, 192]
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, ceil_mode=True),
        )
        # [28, 28, 256]
        self.inception3a = Inception(in_ch=192, ch11=64, ch33pre=96, ch33=128, ch55pre=16, ch55=32, chmaxla=32)
        # [28, 28, 480]
        self.inception3b = Inception(in_ch=256, ch11=128, ch33pre=128, ch33=192, ch55pre=32, ch55=96, chmaxla=64)
        # [14, 14, 480]
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=2, ceil_mode=True)
        # [14, 14, 512]
        self.inception4a = Inception(in_ch=480, ch11=192, ch33pre=96, ch33=208, ch55pre=16, ch55=48, chmaxla=64)
        # [14, 14, 512]
        self.inception4b = Inception(in_ch=512, ch11=160, ch33pre=112, ch33=224, ch55pre=24, ch55=64, chmaxla=64)
        # [14, 14, 512]
        self.inception4c = Inception(in_ch=512, ch11=128, ch33pre=128, ch33=256, ch55pre=24, ch55=64, chmaxla=64)
        # [14, 14, 528]
        self.inception4d = Inception(in_ch=512, ch11=112, ch33pre=144, ch33=288, ch55pre=32, ch55=64, chmaxla=64)
        # [14, 14, 832]
        self.inception4e = Inception(in_ch=528, ch11=256, ch33pre=160, ch33=320, ch55pre=32, ch55=128, chmaxla=128)
        # [7, 7, 832]
        self.maxpool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, ceil_mode=True)
        # [7, 7, 832]
        self.inception5a = Inception(in_ch=832, ch11=256, ch33pre=160, ch33=320, ch55pre=32, ch55=128, chmaxla=128)
        # [7, 7, 1024]
        self.inception5b = Inception(in_ch=832, ch11=384, ch33pre=192, ch33=384, ch55pre=48, ch55=128, chmaxla=128)

        # self.avgpool = nn.AvgPool2d(kernel_size=(7, 7), stride=(1, 1))
        # self.fc = nn.Linear(in_features=1024, out_features=num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

        self.aux1 = Aux(in_ch=512, num_classes=num_classes)
        self.aux2 = Aux(in_ch=528, num_classes=num_classes)


    def forward(self, x):
        x = self.feature(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool(x)
        x = self.inception4a(x)

        if self.aux and self.training:
            aux1 = self.aux1(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        if self.aux and self.training:
            aux2 = self.aux2(x)
        x = self.inception4e(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)

        if self.aux and self.training:
            return x, aux1, aux2

        return x


class Inception(nn.Module):
    def __init__(self, in_ch, ch11, ch33pre, ch33, ch55pre, ch55, chmaxla):
        super(Inception, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=ch11, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=ch33pre, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(True),
            nn.Conv2d(in_channels=ch33pre, out_channels=ch33, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=ch55pre, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(True),
            nn.Conv2d(in_channels=ch55pre, out_channels=ch55, kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.ReLU(True),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Conv2d(in_channels=in_ch, out_channels=chmaxla, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(True),
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        output = torch.cat([x1, x2, x3, x4], dim=1)
        return output


class Aux(nn.Module):
    def __init__(self, in_ch, num_classes=5):
        super(Aux, self).__init__()
        # [14, 14, 512] -> [4, 4, 512]
        self.averagePool = nn.AvgPool2d(kernel_size=(5, 5), stride=3)
        # [4, 4, 128]
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=128, kernel_size=(1, 1), stride=(1, 1))
        self.relu = nn.ReLU(True)
        self.fc = nn.Sequential(
            nn.Linear(in_features=4 * 4 * 128, out_features=1024),
            nn.ReLU(True),
            nn.Dropout(0.7),
            nn.Linear(in_features=1024, out_features=num_classes)
        )

    def forward(self, x):
        x = self.averagePool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = torch.flatten(x, 1)
        output = self.fc(x)

        return output


if __name__ == '__main__':
    net = GoogLenet(num_classes=5, aux=False)
    print(net)