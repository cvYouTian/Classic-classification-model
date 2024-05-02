""" 数据集的存放格式
- project
    - flowers
        -daisy
        -dandelion
        -rose
        -sunflower
        -tuilp
    - ResNet
        -dataset.py
"""

from pathlib import Path
from typing import Union, Tuple
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class Flower(Dataset):
    def __init__(self, root_path: Union[str, Path], mode: str):
        """
        root_path[str|Path]: flower数据集文件的路径
        mode[str]: 训练集还是验证集
        """
        super().__init__()
        self.root_path = root_path if isinstance(root_path, Path) else Path(root_path)
        self.mode = mode

        # 这里使用一些基本的数据增强的方案
        self.transform = {
            "train": transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()]),
            "val": transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])
        }

        self.imgs = list(self.root_path.rglob("*.jpg"))

        # {label:[img1,img2,...]}
        self.data = dict()
        self.classes = {"daisy": torch.tensor(0),
                        "dandelion": torch.tensor(1),
                        "rose": torch.tensor(2),
                        "sunflower": torch.tensor(3),
                        "tulip": torch.tensor(4)}

        if not self.root_path.exists():
            raise FileNotFoundError

        try:
            for label in self.root_path.iterdir():
                self.data[label.name] = list()
                for img in label.iterdir():
                    self.data[label.name].append(str(img))
        except Exception as e:
            print(e)

    def __len__(self) -> int:

        return len(self.imgs)

    def __getitem__(self, item) -> Tuple[torch.Tensor, str]:
        img_path = str(self.imgs[item])
        img = Image.open(img_path)
        # 输入transform之前一定要使用Pillow的读取图像
        img = self.transform[self.mode](img)
        label = self._get_label(img_path)

        return img, label

    def _get_label(self, img_path) -> Union[str, None]:
        # 根据图像路径找到对应的标签
        for label, imgs in self.data.items():
            if img_path in (str(img) for img in imgs):
                return self.classes[label]
        return None


if __name__ == '__main__':
    DT = Flower(root_path="/home/youtian/Documents/pro/pyCode/Classic-classification-model/flowers", mode="train")
    a = DT[3]
    b = len(DT)
    print(a, b)