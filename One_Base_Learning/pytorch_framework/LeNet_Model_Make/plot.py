from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt

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
        b_x, b_y = 0, 0
        for step, (b_x, b_y) in enumerate(train_loader):
            if step > 0:
                break
        batch_x = b_x.squeeze().numpy()  # type:ignore
        batch_y = b_y.numpy()  # type:ignore
        class_label = train_data.classes
        print(class_label, len(class_label))
        return batch_x, batch_y, class_label
    except Exception as e:
        raise e


def draw_func():
    batch_x, batch_y, class_label = train_load_func()
    plt.figure(figsize=(12, 5))
    for ii in np.arange(len(batch_y)):
        plt.subplot(4, 16, ii + 1)
        plt.imshow(batch_x[ii, :, :], cmap=plt.cm.gray)  # type:ignore
        plt.title(class_label[batch_y[ii]], size=10)
        plt.axis("off")
        plt.subplots_adjust(wspace=0.05)
    plt.show()


if __name__ == "__main__":
    download_dataset_func()
    train_load_func()
    draw_func()
