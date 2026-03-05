"""
# 空气质量预测模型
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.common import random_state
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.layers import Dense
import keras

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

"""
    先导入数据
    【回归任务】：
        回归任务我需要对标签进行归一化，需要对输出值进行反归一化
"""
data_set = pd.read_csv("./data.csv")

# 先对数据进行归一化
sc = MinMaxScaler(feature_range=(0, 1))
scaled = sc.fit_transform(data_set)
print(type(scaled))
print(scaled)
print("=" * 30)
# 将归一化之后的数据转换为dataframe格式
data_set_sc = pd.DataFrame(scaled)
print(data_set_sc)
print("=" * 30)

# 将数据集之中的标签和特征找出来
X = data_set_sc.iloc[:, :-1]  # 取数据
Y = data_set_sc.iloc[:, -1]  # 取特征
# 划分数据集 训练集80% 测试集20%
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# 利用keras搭建神经网络模型
model = keras.Sequential()
model.add(Dense(10, activation="relu"))  # 第一个隐藏层
model.add(Dense(10, activation="relu"))  # 第二个隐藏层
model.add(Dense(1))  # 最后一个隐藏层：回归值

# 编译神经网络模型
model.compile(loss="mse", optimizer="SGD")

# 进行模型的训练
history = model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=24,
    verbose=2,
    validation_data=(x_test, y_test),
)
model.save("model.h5")

plt.plot(history.history["loss"], label="train 训练集")
plt.plot(history.history["val_loss"], label="val 验证集")
plt.title("全连接神经网络loss值图")
plt.show()


if __name__ == "__main__":
    pass
