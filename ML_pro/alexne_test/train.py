# -*- coding: UTF-8 -*-

import os
import sys
import json

import torch
from torch import nn, optim
# 导入图片处理工具
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


from Milti_model import Alexnet


def main():
    device =