from typing import Union
from pathlib import Path
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-backbone", type=str, required=True, help="classfier net")
    parser.add_argument("-gpu", type=bool, default=False, help="use gpu or not")
    parser.add_argument("-batchsize", type=int, default=1, help="batch size for dataloader")
    parser.add_argument("-warm", type=int, default=3, help="warm up training phase")
    parser.add_argument("-resume", type=str, default=False, help="classfier net")
    args = parser.parse_args()

