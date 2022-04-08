# -*- coding: UTF-8 -*-

import os
import json
import torch
import torchvision
from Milti_model.ResNet import resnet34


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("useing {} device".format(device))

    data_transform = {
        "train": torch.transforms.Compose([torch.transforms.RandomResizedCrop(224),
                                     torch.transforms.RandomHorizontalFlip(),
                                     torch.transforms.ToTensor(),
                                     torch.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": torch.transforms.Compose([torch.transforms.Resize(256),
                                   torch.transforms.CenterCrop(224),
                                   torch.transforms.ToTensor(),
                                   torch.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    img_path = os.path.join(data_root, "flower_data")
    train_path = os.path.join(img_path, "train")
    val_path = os.path.join(img_path, "val")
    assert os.path.exists(train_path), "{} not find".format(train_path)
    assert os.path.exists(val_path), "{} not find".format(val_path)
    # 这是做好的训练集
    train_dataset = torchvision.datasets.ImageFolder(train_path, transform=data_transform["train"])
    train_num = len(train_dataset)

    flower_list = train_dataset.class_to_idx
    cla_dict = dict((value, key) for key, value in flower_list.items())

    json_str = json.dumps(cla_dict, indent=4)
    with open("class_indices.json", 'w') as json_file:
        json_file.write(json_str)

    batch_size = 16

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print("Use {} datsloader workers ".format(nw))




