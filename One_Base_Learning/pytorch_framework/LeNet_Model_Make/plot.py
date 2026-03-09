from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import numpy as np

import os


def download_dataset_func():
    """
    数据集下载函数
    """
    try:
        if os.path.exists("./data"):
            print("数据集已经下载")
        else:
            print("正在下载数据集")

        train_data = FashionMNIST(
            root="./data",
            train=True,
            transform=transforms.Compose(
                [transforms.Resize(size=224), transforms.ToTensor()]
            ),
            download=True,
        )
        return train_data

    except Exception as e:
        raise e


def train_load_func():
    try:
        train_data = download_dataset_func()
        train_loader = Data.DataLoader(
            dataset=train_data, batch_size=64, shuffle=True, num_workers=0
        )
        print(train_loader)
    except Exception as e:
        raise e


if __name__ == "__main__":
    download_dataset_func()
