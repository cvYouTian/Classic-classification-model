# -*- coding: UTF-8 -*-
import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from Milti_model.GoogLeNet import GoogleNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load image
    img_path = "./tulip.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    # model = GoogleNet(num_classes=5, aux_logits=False, init_weights=False).to(device)
    weight ="./googlenetNet.pth"
    model = GoogleNet(num_classes=5, aux_logits=True, init_weights=False).to(device)
    # print(net)
    del_weight = []
    weights = torch.load(weight, map_location=device)
    for key, value in weights.items():
        if "aux1" == key.strip().split('.')[0] or "aux2" == key.strip().split('.')[0]:
            del_weight.append(key)
    for i in del_weight:
        del weights[i]

    model.load_state_dict(weights, strict=False)
    # load model weights
    # weights_path = "./googlenetNet.pth"
    # assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    # model.load_state_dict(torch.load(weights_path, map_location=device))

    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()