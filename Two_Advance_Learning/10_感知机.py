"""
1、感知机基础概念:
    o=func(<w,x>+b)
    感知机的输出 1/-1，可以视为二分类问题
    二分类问题：-1或者1
    vs.回归输出实数
    vs.softmax回归输出函数
2、训练感知机
    等价于使用批量大小为1的梯度下降，并且使用如下损失函数：
    func(y,x,w)=max(0,-y<w,x>)
3、收敛定理
    假设数据在半径为r内，余量ρ分类两类
    y(x^T+b)>=ρ
    对于||w||²+b²<=1
    感知机确保可以在(r²+1)/p²步之后收敛
4、感知机的缺点
    感知机不能拟合XOR函数，它只能产生线性分割面
5、输入
    作为输出
6、激活函数
    （1）sigmoid激活函数：将任意x投影到（0，1）区间之中。
    （2）tanh激活函数：将任意x投影到（-1，1）区间之中。
    （3）relu激活函数：relu(x)=max(x,0)。
"""

# === 多层感知机的实现 ===
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
num_inputs, num_outputs, num_hiddens = 784, 10, 256
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True))
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True))
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
params = [W1, b1, W2, b2]


# 实现relu激活函数
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)


# 实现我们的模型
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X @ W1 + b1)
    return H @ W2 + b2


loss = nn.CrossEntropyLoss()

# 多层感知机的训练过程与softmax回归的训练过程完成相同
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch13(net, train_iter, test_iter, loss, num_epochs, updater)
