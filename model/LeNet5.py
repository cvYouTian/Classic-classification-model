import torch
import torch.nn as nn
from typing import Union
from utils import auto_padding




class SubSampling(nn.Module):
    """
    sub_sampling for LeNet's paper.(Gradient-based learning applied to document recognition)

    Args:
        filter_size (int | tuple | list): its shape is (w,h).neighborhood in the corresponding feature map.
        stride (int | tuple | list):

    """

    def __init__(self, filter_size: Union[int, tuple, list], stride: Union[int, tuple, list]):

        super(SubSampling, self).__init__()
        self.coefficient = torch.empty(0)
        self.stride = stride
        self.weight_w, self.weight_h = filter_size if isinstance(filter_size, int) else filter_size[0]

    def forward(self, x):
        # BCHW
        _, c, h, w = x.shape
        out_h = (h - self.weight_h) // self.stride + 1
        out_w = (w - self.weight_w) // self.stride + 1

        for n in range(c):
            for i in range(out_h):
                for j in range(out_w):
                    s_h = i
                    e_h = i + self.weight_h
                    s_w = j
                    e_w = j + self.weight_w
                    res = self.coefficient*torch.add(x[:,n,s_h:e_h,s_w, e_w])










class Lenet5(nn.Module):
    def __init__(self):
        super(Lenet5, self).__init__()
        # [b, 1, 32, 32]-> [1, 6, 28, 28]
        self.C1 = nn.Conv2d(1, 6, 5, 1, 0)
        ...




import torch
import torch.nn as nn


class MyMaxPool2D(nn.Module):
    def __init__(self, kernel_size=(2, 2), stride=2):
        super(MyMaxPool2D, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.w_height = kernel_size[0]
        self.w_width = kernel_size[1]


    def forward(self, x):
        in_height = x.size(0)
        in_width = x.size(1)

        out_height = int((in_height - self.w_height) / self.stride) + 1
        out_width = int((in_width - self.w_width) / self.stride) + 1

        out = torch.zeros((out_height, out_width))

        for i in range(out_height):
            for j in range(out_width):
                start_i = i * self.stride
                start_j = j * self.stride
                end_i = start_i + self.w_height
                end_j = start_j + self.w_width
                out[i, j] = torch.max(x[start_i: end_i, start_j: end_j])
        return out



if __name__ == "__main__":
    # myMaxPool2d
    print("="*10 + "MyMaxPool2D" + "="*10)
    x = torch.randn((6,8), requires_grad=True)
    mypool = MyMaxPool2D()
    y = mypool(x)
    c = torch.mean(y)
    c.backward()

    print(x.size(), x.dtype)
    print(y.size())
    print(x.grad)

    # nn.MaxPool2d
    print("=" * 10 + "nn.MaxPool2d" + "=" * 10)
    x2 = x.detach().view(1,1,6,8)
    x2.requires_grad=True
    mypool = nn.MaxPool2d((2,2),2)
    y2 = mypool(x2)
    c2 = torch.mean(y2)
    c2.backward()

    print(x2.size(), x2.dtype)
    print(y2.size())
    print(x2.grad)
