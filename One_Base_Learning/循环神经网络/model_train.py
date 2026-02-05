"""
>>> 基于LSTM神经网络的黄金价格预测
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import tensorflow.keras


plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 加载数据集
data = pd.read_csv("./LBMA-GOLD.csv", index_col=[0])
print(data)

# 设置训练集的长度
training_len = 1256 * 200

# 获取训练集数据
training_set = data.iloc[:training_len, [0]]  # type:ignore


# 获取测试集数据
test_set = data.iloc[training_len:, [0]]  # type:ignore

# 数据进行归一化处理
sc = MinMaxScaler(feature_range=(0, 1))
train_set_sc = sc.fit_transform(training_set)
test_set_sc = sc.fit_transform(test_set)

# 设置一个列表，来划分特征和标签
x_train = list()  # 训练集特征
y_train = list()  # 训练集标签

# 测试机特征和标签
x_test = list()
y_test = list()

# 以连续五天作为一个训练周期，进行训练操作
import numpy


if __name__ == "__main__":
    pass
