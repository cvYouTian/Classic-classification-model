import time
from typing import Union
from pathlib import Path
import torch
import torch.nn as nn
from dataset import Flower
from model.ResNet34 import ResNet34
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-classes", type=int, default=5, help="number of class")
    parser.add_argument("-epoch", type=int, default=150, help="iter epoch")
    parser.add_argument("-lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("-gpu", type=bool, default=True, help="use gpu or not")
    parser.add_argument("-batch_size", type=int, default=64, help="batch size for dataloader")
    parser.add_argument("-warm", type=int, default=3, help="warm up training phase")
    parser.add_argument("-resume", type=str, default=False, help="classfier net")
    args = parser.parse_args()

    net = ResNet34(args.classes).to("cuda:0") if args.gpu else ResNet34(args.classes).to("cpu")

    train_data = DataLoader(Flower(root_path="../flowers", mode="train"), shuffle=True,
                      batch_size=args.batch_size, num_workers=8)

    val_data = DataLoader(Flower(root_path="../flowers", mode="val"), shuffle=True,
                      batch_size=args.batch_size // 8, num_workers=1)

    # 检查有多少个step
    # print(len(data))
    # 将每个step中的数据和标签拿出来
    # for i in data:
    #     print(i)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # steps = len(train_data)

    # writer = SummaryWriter(log_dir="./logs")

    # input_tensor = torch.Tensor(1,3,32,32)
    # if args.gpu:
    #     input_tensor = input_tensor.cuda()
    # writer.add_graph(net, input_tensor)

    ckpt_save = Path("./") / "runs"
    ckpt_save.mkdir(parents=True, exist_ok=True)
    ckpt_file = ckpt_save / "Resnet34_{epoch}.pt"

    best_acc = 0.0

    for epoch in range(1, args.epoch+1):
        start = time.time()
        net.train()

        for step, (images, labels) in enumerate(train_data):
            images, labels = (images.to("cuda:0"), labels.to("cuda:0")) if args.gpu else (images, labels)
            optimizer.zero_grad()
            output = net(images)
            loss = loss_function(output, labels)
            loss.backward()
            optimizer.step()

            print(f"Training Epoch: {epoch}\t loss:{loss.item():0.4f}")
        finish = time.time()
        print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))