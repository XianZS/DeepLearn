from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from lenet_model_setup import LeNet
import copy
import time


def train_val_data_process_func():
    als_data = FashionMNIST(
        root="./data",
        train=True,
        transform=transforms.Compose(
            [transforms.Resize(size=28), transforms.ToTensor()]
        ),
        download=True,
    )
    train_data, val_data = Data.random_split(
        als_data, [round(0.8 * len(als_data)), round(0.2 * len(als_data))]
    )
    train_dataloader = Data.DataLoader(
        dataset=train_data, batch_size=128, shuffle=True, num_workers=8
    )

    val_dataloader = Data.DataLoader(
        dataset=val_data, batch_size=128, shuffle=True, num_workers=8
    )
    return train_dataloader, val_dataloader


# train_dataloader, val_dataloader = train_val_data_process_func()


def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    # 设定训练所使用到的设备-device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 使用Adam优化器，学习率为0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 损失函数-交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 将模型放置到训练设备之中
    model = model.to(device)
    # 复制当前模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    # 初始化精确度
    best_acc = 0.0
    # 训练集损失值列表
    train_loss_all = []
    # 验证集损失值列表
    val_loss_all = []
    # 训练集准确度列表
    train_acc_all = []
    # 验证集准确度列表
    val_acc_all = []

    # 得到当前时间
    since = time.time()


if __name__ == "__main__":
    train_dataloader, val_dataloader = train_val_data_process_func()
