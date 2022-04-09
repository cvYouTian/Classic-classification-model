import torch
from torch import nn
import torch.nn.functional as F


class GoogleNet(nn.Module):
    def __init__(self, num_classes=5, aux_logits=True, init_weights=False):
        super(GoogleNet, self).__init__()
        self.aux_logits = aux_logits

        self.feature = nn.Sequential(
            # [112, 112, 64]
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=3),
            nn.ReLU(True),
            # [56, 56, 64]
            nn.MaxPool2d(kernel_size=3, stride=(2, 2), ceil_mode=True),
            # [56, 56, 64]
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(True),
            # [56, 56, 192]
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(True),
            # [28, 28, 192]
            nn.MaxPool2d(kernel_size=3, stride=(2, 2), ceil_mode=True)
        )
            # [28, 28, 256]
        self.incep3a = Inception(192, 64, 96, 128, 16, 32, 32)
        # [28, 28, 480]
        self.incep3b = Inception(256, 128, 128, 192, 32, 96, 64)
        # [14, 14, 480]
        self.maxpooling1 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, ceil_mode=True)
        # [14, 14, 512]
        self.incep4a = Inception(480, 192, 96, 208, 16, 48, 64)
        if aux_logits:
            self.aux1 = InceptionAux(in_ch=512, num_classes=num_classes)
            self.aux2 = InceptionAux(in_ch=528, num_classes=num_classes)
        # [14, 14, 512]
        self.incep4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.incep4c = Inception(512, 128, 128, 256, 24, 64, 64)
        # [14, 14, 528]
        self.incep4d = Inception(512, 112, 144, 288, 32, 64, 64)
        # [14, 14, 832]
        self.incep4e = Inception(528, 256, 160, 320, 32, 128, 128)
        # [7, 7, 832]
        self.maxpooling2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), ceil_mode=True)
        self.incep5a = Inception(832, 256, 160, 320, 32, 128, 128)
        # [7, 7, 1024]
        self.incep5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.classifer = nn.Sequential(
            # 就是一个简单的全局平均池化来替代flatten
            nn.AvgPool2d(kernel_size=(7, 7), stride=1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(in_features=1024, out_features=num_classes)
        )

    def forward(self, x):
        x = self.feature(x)
        # [28, 28, 256]
        x = self.incep3a(x)
        # [28, 28, 480]
        x = self.incep3b(x)
        # [14, 14, 480]
        x = self.maxpooling1(x)
        # [14, 14, 512]
        x = self.incep4a(x)
        if self.training and self.aux_logits:
            aux1 = self.aux1(x)
        # [14, 14, 512]
        x = self.incep4b(x)
        x = self.incep4c(x)
        # [14, 14, 528]
        x = self.incep4d(x)
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)
        # [14, 14, 832]
        x = self.incep4e(x)
        # [7, 7, 832]
        x = self.maxpooling2(x)
        x = self.incep5a(x)
        # [7, 7, 1024]
        x = self.incep5b(x)

        logits= self.classifer(x)

        if self.training and self.aux_logits:
            return logits, aux1, aux2

        return logits


class Inception(nn.Module):
    def __init__(self, in_ch, ch11, ch33pre, ch33, ch55pre, ch55, chmaxla):
        super(Inception, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=ch11, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=ch33pre, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(True),
            nn.Conv2d(in_channels=ch33pre, out_channels=ch33, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=ch55pre, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(True),
            nn.Conv2d(in_channels=ch55pre, out_channels=ch55, kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.ReLU(True)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=(1, 1), padding=1),
            nn.Conv2d(in_channels=in_ch, out_channels=chmaxla, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(True)
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        outputs = torch.cat([x1, x2, x3, x4], dim=1)
        return outputs


class InceptionAux(nn.Module):
    # aux1 = 14, 14, 512  aux2 = 14, 14, 528
    def __init__(self, in_ch, num_classes=5):
        super(InceptionAux, self).__init__()
        # [4, 4, nn]
        self.averagePool = nn.AvgPool2d(kernel_size=(5, 5), stride=(3, 3))
        #
        self.con1 = nn.Conv2d(in_channels=in_ch, out_channels=128, kernel_size=(1, 1), stride=(1, 1))

        # flatten
        self.fc = nn.Linear(in_features=4*4*128, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=num_classes)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.averagePool(x)
        # 最后一层卷积千万不要做relu
        x = self.con1(x)
        x = self.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.relu(x)
        x = F.dropout(x, 0.7, training=self.training)
        logits = self.fc2(x)

        return logits


if __name__ == '__main__':
    weight = r"../ML_pro/googlenet_test/googlenetNet.pth"
    net = GoogleNet(num_classes=5, aux_logits=True, init_weights=False)
    # print(net)
    del_weight =[]
    weights = torch.load(weight, map_location="cuda:0")
    for key, value in weights.items():
        if "aux1" == key.strip().split('.')[0] or "aux2" == key.strip().split('.')[0]:
            del_weight.append(key)
    for i in del_weight:
        del weights[i]


    net.load_state_dict(weights, strict=False)










