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
        padding (int | tuple | list):
        coefficient (torch.Tensor): a trainable coefficient
        bias (torch.Tensor): a trainable bias

    """

    def __init__(self,
                 filter_size: Union[int, tuple, list],
                 stride: Union[int, tuple, list],
                 padding: Union[int, tuple, list],
                 dilation: Union[int, tuple, list]):

        super(SubSampling, self).__init__()
        self.stride = stride
        self.padding = [padding, padding] if isinstance(padding, int) else padding
        self.dilation = [dilation, dilation] if isinstance(dilation, int) else dilation
        self.weight = [filter_size, filter_size] if isinstance(filter_size, int) else filter_size
        self.coefficient = nn.Parameter(torch.ones(1, requires_grad=True))
        self.bias = nn.Parameter(torch.ones(1, requires_grad=True))

    def forward(self, x):
        # BCHW
        b, c, h, w = x.shape
        out_h = (h + 2 * self.padding[0] - self.dilation[0] * (self.weight[0] - 1) - 1) // self.stride + 1
        out_w = (w + 2 * self.padding[1] - self.dilation[1] * (self.weight[1] - 1) - 1) // self.stride + 1
        out_feat = torch.zeros([b, c, out_h, out_w], dtype=x.dtype, device=x.device)
        for m in range(b):
            for n in range(c):
                for i in range(out_h):
                    for j in range(out_w):
                        start_width = j * self.stride
                        end_width = start_width + self.stride
                        start_height = i * self.stride
                        end_height = start_height + self.stride
                        # this subsample for lenet5
                        out_feat[m, n, i, j] = self.coefficient*torch.sum(
                            x[m, n, start_height:end_height, start_width:end_width]) + self.bias

        return out_feat

class C3Convlution(nn.Module):

    def __init__(self,
                 custom_feat: list,
                 in_chanel: int,
                 out_chanel: int,
                 kernel_size: Union[int, tuple],
                 stride: Union[int, tuple, list],
                 padding: Union[int, tuple, list]):
        super(C3Convlution, self).__init__()
        self.custom_feat = custom_feat
        self.in_chanel = in_chanel
        self.out_chanel = out_chanel
        self.kernel_size = [kernel_size, kernel_size] if isinstance(kernel_size, int) else kernel_size
        self.stride = [stride, stride] if isinstance(stride, int) else stride
        self.padding = [padding, padding] if isinstance(padding, int) else padding

    def forward(self, x):
        b, c, h, w = x.shape
        feat_container = list()
        feat_idx = dict()
        for i in range(b):
            for idx, j in enumerate(range(c)):
                feat_idx[idx] = x[i, j, :, :]
                if idx == len(c) - 1:
                    feat_container.append(feat_idx)

        for feat in feat_container:
            for i in self.custom_feat:
                ...


class Lenet5(nn.Module):
    def __init__(self):
        super(Lenet5, self).__init__()
        # [b, 1, 32, 32]-> [1, 6, 28, 28]
        self.C1 = nn.Conv2d(1, 6, 5, 1, 0)
        self.S2 = SubSampling(2, 2, 0, 1)
        self.C3 = nn.Conv2d(6, 16, 5, )



if __name__ == "__main__":

    beta = torch.randn([5, 3, 40, 40])
    mypooling = SubSampling(2, 2, 0, 1)
    maxp = nn.MaxPool2d(2, 2)

    a = mypooling(beta)
    b = maxp(beta)
    print(a.eq(b))
    # print(a.shape, b.shape)
    # print(a, b)

    # print(a)
    #
    # optimizer = torch.optim.SGD([{"params": net.parameters()},
    #                        {"params": my_module.parameters()}],
    #                         lr=base_lr, momentum=momentum, weight_decay=weight_decay)
