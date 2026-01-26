import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense
from tensorflow.python.keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.models import load_model

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

data_set = pd.read_csv('breast_cancer_data.csv')
# print(data_set)
X, Y = data_set.iloc[:, :-1], data_set["target"]
# print(X, Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=7)
# print(x_train, x_test, y_train, y_test)
# y_train_one = to_categorical(y_train)
y_test_one = to_categorical(y_test)

sc = MinMaxScaler(feature_range=(0, 1))
# x_train_sc = sc.fit_transform(x_train)
x_test_sc = sc.fit_transform(x_test)
# 导入上一步已经训练好的模型，用于测试，模型名为`model.h5`
model = load_model("model.h5")
# 输入测试的特征数据，x_test_sc，进行预测
y_pred = model.predict(x_test_sc)
# print(y_pred)  # [[良性概率 恶性概率],[良性概率 恶性概率],...]

y_pred_np = np.argmax(y_pred, axis=1)  # [max(良性概率，恶行概率),max(良性概率，恶性概率),...]
# print(y_pred_np)
y_pred_np_name = np.where(y_pred_np == 0, "良性", "恶性")
# print(y_pred_np_name)
# 打印模型的精确度和召回率
res = classification_report(y_test, y_pred_np, labels=[0, 1], target_names=["良性", "恶性"])
print(res)
