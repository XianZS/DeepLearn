# `LeNet`-5：从设计原理到代码实现的全面解析

## 一、代码解读

> 卷积运算：可以改变数据的维度，1个特征图变为6个特征图；
>
> 池化运算：不可以改变数据的维度；

```Python
# 导入PyTorch核心库，提供张量运算、自动求导、设备管理等基础功能
import torch

# 导入PyTorch神经网络模块，包含构建CNN所需的所有层（卷积、池化、全连接、激活函数等）
from torch import nn

# 导入torchsummary的summary函数，用于可视化神经网络的结构、每层输出维度和参数总量
from torchsummary import summary


class LeNet(nn.Module):
    """
    LeNet-5 卷积神经网络模型（经典版本适配）
    说明：
    1. LeNet-5是Yann LeCun在1998年提出的经典CNN架构，最初用于MNIST手写数字识别
    2. 原始LeNet-5输入为32x32灰度图，此处适配了28x28的MNIST常用输入尺寸（通过padding=2补偿）
    3. 继承nn.Module后，可复用PyTorch的网络层、参数优化、设备迁移等核心功能
    网络结构（适配28x28输入）：
    输入(1,28,28) → 卷积层C1 → Sigmoid激活 → 平均池化S2 → 卷积层C3 → Sigmoid激活 → 平均池化S4 → 展平 → 全连接F5 → 全连接F6 → 全连接F7（输出）
    """

    def __init__(self):
        """
        初始化函数：定义LeNet-5的所有网络层
        核心逻辑：卷积层提取空间特征 → 激活函数引入非线性 → 池化层降维+抗过拟合 → 全连接层分类
        """
        # 调用父类nn.Module的初始化方法，必须执行，否则无法使用nn.Module的核心功能
        super(LeNet, self).__init__()

        # ===================== 卷积层+池化层（特征提取部分） =====================
        # 【第一层：卷积层C1】
        # in_channels=1：输入为单通道灰度图（MNIST数据集特征）
        # out_channels=6：输出6个特征图（使用6个不同的卷积核）
        # kernel_size=5：卷积核尺寸为5x5（LeNet-5原始设计）
        # padding=2：填充2圈0，补偿输入尺寸从32x32→28x28的差异，保证卷积后尺寸不变（28+2*2-5+1=28），此处步幅为1
        # 输出尺寸=【（输入尺寸+2*填充圈数-卷积核尺寸）/步幅】+1
        # 输入维度：(1, 28, 28)
        # 输出维度：(6, 28, 28)
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)

        # 激活函数：Sigmoid（LeNet-5原始设计，替代ReLU）
        # 作用：引入非线性，让网络能拟合复杂特征（无激活则所有层都是线性变换，等价于单层）
        self.sig = nn.Sigmoid()

        # 【第二层：平均池化层S2】
        # kernel_size=2：池化核尺寸为2x2
        # stride=2：步幅为2（池化核每次移动2个像素）
        # 作用：降维（尺寸减半）、保留关键特征、减少计算量、提升鲁棒性
        # 输出维度：(6, 14, 14)（28/2=14）
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 【第三层：卷积层C3】
        # in_channels=6：输入为S2输出的6个特征图
        # out_channels=16：输出16个特征图（LeNet-5原始设计）
        # kernel_size=5：卷积核尺寸5x5
        # padding=0：无填充（原始设计）
        # 输出维度：(16, 10, 10)（14-5+1=10）
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0)

        # 【第四层：平均池化层S4】
        # 池化核2x2，步幅2，与S2参数一致
        # 输出维度：(16, 5, 5)（10/2=5）
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 【第五层：展平层】
        # 作用：将4维张量（batch_size, channels, height, width）转为2维张量（batch_size, channels*height*width）
        # 适配全连接层的输入格式（全连接层仅接受一维特征向量）
        self.flatten = nn.Flatten()

        # ===================== 全连接层（分类部分） =====================
        # 【第六层：全连接层F5】
        # in_features=400：展平后的特征数（16通道 × 5高度 × 5宽度 = 400）
        # out_features=120：隐藏层神经元数量（LeNet-5原始设计）
        # 输出维度：(batch_size, 120)
        self.f5 = nn.Linear(in_features=400, out_features=120)

        # 【第七层：全连接层F6】
        # in_features=120：输入为F5的输出
        # out_features=84：隐藏层神经元数量（LeNet-5原始设计）
        # 输出维度：(batch_size, 84)
        self.f6 = nn.Linear(in_features=120, out_features=84)

        # 【第八层：全连接层F7（输出层）】
        # in_features=84：输入为F6的输出
        # out_features=10：输出维度对应MNIST的10个数字类别（0-9）
        # 输出维度：(batch_size, 10)（每个维度代表对应类别的预测得分）
        self.f7 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        """
        前向传播函数：定义数据在网络中的流动路径（核心，决定网络的计算逻辑）
        参数：
            x (torch.Tensor)：输入张量，形状为(batch_size, 1, 28, 28)
                - batch_size：批次大小（一次输入的样本数）
                - 1：通道数（灰度图）
                - 28,28：输入图像的高和宽
        返回：
            x (torch.Tensor)：输出张量，形状为(batch_size, 10)，对应10类的预测得分
        """
        # 第一步：C1卷积 → Sigmoid激活
        # 输入(bs,1,28,28) → 卷积后(bs,6,28,28) → 激活后维度不变
        x = self.sig(self.c1(x))

        # 第二步：S2平均池化
        # 输入(bs,6,28,28) → 池化后(bs,6,14,14)（尺寸减半）
        x = self.s2(x)

        # 第三步：C3卷积 → Sigmoid激活
        # 输入(bs,6,14,14) → 卷积后(bs,16,10,10) → 激活后维度不变
        x = self.sig(self.c3(x))

        # 第四步：S4平均池化
        # 输入(bs,16,10,10) → 池化后(bs,16,5,5)（尺寸减半）
        x = self.s4(x)

        # 第五步：展平
        # 输入(bs,16,5,5) → 展平后(bs,400)（16*5*5=400）
        x = self.flatten(x)

        # 第六步：F5全连接
        # 输入(bs,400) → 输出(bs,120)
        x = self.f5(x)

        # 第七步：F6全连接
        # 输入(bs,120) → 输出(bs,84)
        x = self.f6(x)

        # 第八步：F7输出层
        # 输入(bs,84) → 输出(bs,10)（最终预测得分）
        x = self.f7(x)

        return x


def main():
    """
    主函数：测试LeNet模型的设备适配性，并可视化模型结构和参数
    """
    # 1. 配置计算设备：优先使用GPU（CUDA）加速，无GPU则使用CPU
    # torch.cuda.is_available()：检查当前环境是否支持CUDA（NVIDIA显卡+驱动）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的计算设备：{device}")

    # 2. 实例化LeNet模型，并将模型参数迁移到指定设备（GPU/CPU）
    # model.to(device)：必须执行，否则模型和数据可能不在同一设备，导致计算错误
    model = LeNet().to(device=device)

    # 3. 可视化模型结构：打印每层的输出形状、参数数量、总参数等
    # summary参数说明：
    #   model：待可视化的模型
    #   (1, 28, 28)：单样本输入形状（通道数1，高28，宽28），与MNIST数据集匹配
    print("\nLeNet-5模型结构与参数汇总：")
    summary_output = summary(model, (1, 28, 28))
    print(summary_output)


# 程序入口：仅当直接运行该脚本时，执行main函数（避免被导入时执行）
if __name__ == "__main__":
    main()
```

## 二、总结

1. **LeNet-5核心结构**：特征提取（2层卷积+2层平均池化）+ 分类（3层全连接），适配28x28 MNIST输入时通过`padding=2`补偿原始32x32的设计差异。

2. **维度变化逻辑**：输入(1,28,28) → 卷积+池化后(16,5,5) → 展平为400维 → 全连接最终输出10维（对应0-9数字分类）。

3. **关键细节**：`nn.Module`是PyTorch所有网络的基类，必须调用`super().__init__()`；`model.to(device)`需保证模型和输入数据在同一设备；`torchsummary.summary`可快速验证模型结构和参数合理性。