# -*- coding: UTF-8 -*-
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from ResNet34 import ResNet34


if __name__ == '__main__':
    image_path = Path("/home/youtian/Documents/pro/pyCode/Classic-classification-model/2.jpg")
    # image_path = Path("/home/youtian/Documents/pro/pyCode/Classic-classification-model/1.jpg")
    # image_path = Path("/home/youtian/Documents/pro/pyCode/Classic-classification-model/1.jpg")
    # image_path = Path("/home/youtian/Documents/pro/pyCode/Classic-classification-model/1.jpg")
    # image_path = Path("/home/youtian/Documents/pro/pyCode/Classic-classification-model/1.jpg")
    img = Image.open(image_path)
    img.show()

    trans = transforms.Compose([transforms.Resize(224),
                                transforms.ToTensor()])

    img = trans(img)
    img = img.unsqueeze(0)
    checkpoint = torch.load("/home/youtian/Documents/pro/pyCode/Classic-classification-model/ResNet/runs/Resnet34_80.pt")

    net = ResNet34(num_class=5)
    net.eval()
    net.load_state_dict(checkpoint)
    out = net(img)

    _, preds = out.max(1)
    print(torch.sigmoid(out))
    classes = {0: "daisy", 1: "dandelion", 2: "rose", 3: "sunflower", 4: "tulip"}
    print(classes[preds.item()])