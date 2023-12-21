import torch
import torch.nn as nn
from typing import Union
from utils import auto_padding


class C3Convlution(nn.Module):
    """
    C3's the layer of the convolution in LeNet paper.(Gradient-based learning applied to document recognition)

    Args:
        custom_feat(dict):
        in_chanel(int):
        out_chanel（int）：
        kernel_size（int| tuple）:
        stride(int| tuple| list):
        padding(int | tuple | list): not used in LeNet
    """
    def __init__(self,
                 custom_feat: dict,
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
