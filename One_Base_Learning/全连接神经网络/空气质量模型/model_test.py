import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.models import load_model  # type:ignore
import keras

# 得分计算
from math import sqrt
from numpy import concatenate
from sklearn.metrics import mean_absolute_error, mean_squared_error


plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

data_set = pd.read_csv("./data.csv")
# 对数据进行归一化操作
sc = MinMaxScaler(feature_range=(0, 1))
data_set_sc = sc.fit_transform(data_set)
# 将训练好的数据转换为dataframe格式，方便处理
data_set_sc_df = pd.DataFrame(data_set_sc)

# 获取训练数据
X = data_set_sc_df.iloc[:, :-1]
# 获取训练数据对应的标签
Y = data_set_sc_df.iloc[:, -1]

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.05, random_state=42
)

# 加载模型对象
model = load_model("model.h5")

# 用训练好的模型进行预测
yhat_pred = model.predict(x_test)
print(yhat_pred)
print("原AQI的数据范围为(0~200)之间，现在预测的范围都是(0~1)之间。")
print("===" * 30)

# === 【预测值反归一化】 ===
# 操作对象是x_test,y_pred，将y_pred的数据尺度扩大到原有的数据尺度
# 进行预测值的反归一化:【数据尺度的扩大】：将预测值(0~1)的尺度，扩大到原有的数据尺度
# 现在的数据尺度是0~1之间，原有的数据尺度为0~200之间。
inv_yhat_pred = concatenate(
    (x_test, yhat_pred), axis=1
)  # 将x_test和我们的预测值yhat_pred（0~1）拼接在一起
inv_yhat_pred = sc.inverse_transform(
    inv_yhat_pred
)  # 进行反归一化，也就是将预测值yhat_pred扩大到原有的数据尺度
print(inv_yhat_pred)
# 然后去掉最后一列，将反归一化之后的结果，去掉AIO最后一列
prediction = inv_yhat_pred[:, -1]
print(prediction)
print(len(prediction))


# === 【真实值反归一化】 ===
# 操作对象是x_test,y_test，将y_test的数据尺度扩大到原有的数据尺度
# 将y_test进行维度转换
y_test = np.array(y_test)
y_test = np.reshape(y_test, (y_test.shape[0], 1))
# print(y_test.shape)
# 反向缩放真实值
inv_y = concatenate((x_test, y_test), axis=1)
inv_y = sc.inverse_transform(inv_y)
real = inv_y[:, -1]
print(real)


# 量化模型效果
rmse = sqrt(mean_squared_error(real, prediction))  # type:ignore
mape = np.mean(np.abs(real - prediction) / prediction)
print(rmse)
print(mape)


# 画图 真实值和预测值之间的对比图
plt.plot(prediction, label="预测值")
plt.plot(real, label="真实值")
plt.title("全连接神经网络-空气质量预测")
plt.legend()
plt.show()


if __name__ == "__main__":
    pass
