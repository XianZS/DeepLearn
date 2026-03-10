import enum
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
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        # 初始化参数
        # 训练集损失函数
        train_loss = 0.0
        # 训练集精确度
        train_corrects = 0
        # 验证集损失函数
        val_loss = 0.0
        # 验证集精确度
        val_corrects = 0
        # 训练集样本数量
        train_num = 0
        # 验证集样本数量
        val_num = 0

        for step, (b_x, b_y) in enumerate(train_dataloader):
            # 将数据放置到训练设备之中
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            # 将模型状态设置为“训练模式”
            model.train()
            # 将数据放置到模型之中，进行前向传播，得到输出output
            output = model(b_x)
            # 查找每一行最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            # 计算每一个batch的损失函数，将模型输出和标签合并在一起，计算损失函数
            loss = criterion(output, b_y)
            # 将梯度初始化为0，防止梯度累积
            optimizer.zero_grad()
            # 反向传播计算
            loss.backward()
            # 利用梯度下降法更新模型参数
            optimizer.step()
            # 对损失函数进行累加处理
            train_loss += loss.item() * b_x.size(0)
            # 如果预测正确，则准确度加1
            train_corrects += torch.sum(pre_lab == b_y.data)
            # 训练次数累加求和
            train_num += b_x.size(0)

        for step, (b_x, b_y) in enumerate(val_dataloader):
            # 将数据放置到验证设备之中
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            # 将模型状态设置为“评估模式”
            model.eval()
            # 将数据放置到模型之中，进行前向传播，得到输出output
            output = model(b_x)
            # 查找每一行最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            # 计算每一个batch的损失函数，将模型输出和标签合并在一起，计算损失函数
            loss = criterion(output, b_y)
            # # 将梯度初始化为0，防止梯度累积
            # optimizer.zero_grad()
            # # 反向传播计算
            # loss.backward()
            # # 利用梯度下降法更新模型参数
            # optimizer.step()
            # 对损失函数进行累加处理
            val_loss += loss.item() * b_x.size(0)
            # 如果预测正确，则准确度加1
            val_corrects += torch.sum(pre_lab == b_y.data)
            # 验证次数累加求和
            val_num += b_x.size(0)


if __name__ == "__main__":
    train_dataloader, val_dataloader = train_val_data_process_func()
