# 导入pytorch
import torch

# 导入层
from torch import nn

# 展示模型参数
from torchsummary import summary


class LeNet(nn.Module):
    """
    让LeNet继承nn.Module之后，那么当前类就可以使用torch里面的所有层
    """

    def __init__(self):
        """
        卷积层——》激活函数——》池化层——》卷积层——》激活函数——》池化层
        """
        super(LeNet, self).__init__()

        # 【第一层】定义卷积层
        # 输入通道数=1；输出通道数（卷积核的数量）=6；卷积核的大小=5；
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)

        # 定义激活函数
        self.sig = nn.Sigmoid()

        # 【第二层】定义池化层
        # 池化核的大小=2；步幅=2；
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 【第三层】定义卷积层
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0)

        # 【第四层】定义池化层
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 【第五层】定义平展层
        self.flatten = nn.Flatten()

        # 【第6层、第7层、第8层】全连接层
        self.f5 = nn.Linear(in_features=400, out_features=120)
        self.f6 = nn.Linear(in_features=120, out_features=84)
        self.f7 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        """
        前向传播函数
        x:代表输入
        假设`data=池化层返回的数据`，那么当data传给卷积层时，
        卷积层返回的结果需要先传递给‘激活函数’，
        然后才能二次传送给池化层。
        """
        x = self.sig(self.c1(x))
        x = self.s2(x)
        x = self.sig(self.c3(x))
        x = self.s4(x)
        x = self.flatten(x)
        x = self.f5(x)
        x = self.f6(x)
        x = self.f7(x)
        return x


def main():
    # 创建设备device
    # 查看cuda是否激活，未激活情况下使用cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # 实例化模型，并且链接模型和设备
    model = LeNet().to(device=device)
    print(summary(model, (1, 28, 28)))


if __name__ == "__main__":
    main()
