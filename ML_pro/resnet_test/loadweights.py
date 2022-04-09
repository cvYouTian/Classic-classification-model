# -*- coding: UTF-8 -*-
import os
import torch
from torch import nn
from Milti_model.ResNet import resnet34

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_weight_path = "./resnet34-333f7ec4.pth"
    assert os.path.exists(model_weight_path), "file {} not exists".format(model_weight_path)


    # op1, 重写全连接层
    # net = resnet34()
    # 因为咱们的模型是1000类的和预训练的参数完全符合，所以strict用默认的true就行。
    # net.load_state_dict(torch.load(model_weight_path, map_location=device))
    # 下面的是重写了1全链接层
    # in_channel = net.fc.in_features
    # net.fc = nn.Linear(in_channel, 5)


    # op2
    net = resnet34(num_classes=5)
    # torch.load是加载训练好的模型参数
    pre_weights = torch.load(model_weight_path, map_location=device)
    # 这里是对加载好的预训练参数做处理， 删掉里fc的所有参数
    del_key = []
    for key, _ in pre_weights.items():
        if "fc" in key:
            del_key.append(key)

    for key in del_key:
        del pre_weights[key]
    # load_state_dict()是将咱们的模型net用load加载好的数据填充， 因为上面得参数处理使得咱们的模型多了全连接层， 故strict要设为False
    missing_keys, unexpected_keys =  net.load_state_dict(pre_weights, strict=False)
    # 将列表用换行符分割打印
    print("[missing_keys]:", *missing_keys, sep="\n")
    print("[unexpected_keys]:", *unexpected_keys, sep="\n")


if __name__ == '__main__':
    main()
