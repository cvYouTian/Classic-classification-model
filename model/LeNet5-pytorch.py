import torch
from torchsummary import summary
import torch.nn as nn


class lenet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 6, 5, 1)
        self.s2= nn.MaxPool2d(2, 2)
        self.c3 = nn.Conv2d(6, 16, 5, 1)
        self.s4 = nn.MaxPool2d(2, 2)
        self.f5 = nn.Linear(400, 120)
        self.f6 = nn.Linear(120, 84)
        self.f7 = nn.Linear(84, 10)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.c1(x)
        x = self.s2(x)
        x = self.c3(x)
        x = self.s4(x)
        x = self.flatten(x)
        x = self.f5(x)
        x = self.f6(x)
        logits = self.f7(x)

        return logits

net = lenet5().cuda()
# data = torch.randn([1, 3, 640, 640]).cuda()
summary(net, input_size=(3, 32, 32))


