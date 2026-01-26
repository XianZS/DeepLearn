"""
    神经网络分类任务说明：
    神经网络在进行分类时，需要明确知道有多少个类别，以便确定输出层的神经元数量。
    本代码实现了一个基于Keras的全连接神经网络，用于乳腺癌分类任务。
"""

# 导入必要的库
import numpy as np  # 用于数值计算，处理数组和矩阵操作，是深度学习的基础库
import pandas as pd  # 用于数据处理和分析，读取和操作数据集
import matplotlib.pyplot as plt  # 用于数据可视化，绘制图表和曲线
from sklearn.preprocessing import MinMaxScaler  # 用于数据归一化，将特征缩放到0-1范围内
from sklearn.model_selection import train_test_split  # 用于将数据集分割为训练集和测试集

# 导入Keras深度学习库
import keras  # Keras是一个高层神经网络API，用于快速构建和训练深度学习模型
from keras.layers import Dense  # Dense是全连接层类，每个神经元与前一层所有神经元相连，用于提取特征
from keras.utils.np_utils import to_categorical  # 用于将整数标签转换为独热编码格式，适用于多分类问题
from sklearn.metrics import classification_report  # 用于生成分类模型的详细评估报告，包含精确率、召回率、F1分数等指标

# 设置Matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

"""
    数据预处理阶段：
    1. 读取数据集
    2. 分离特征与标签
    3. 划分训练集和测试集
    4. 标签独热编码
    5. 特征归一化
"""
# 读取乳腺癌数据集
data_set = pd.read_csv("../../线性回归和逻辑回归模型/breast_cancer_data.csv")
print(data_set)  # 打印数据集基本信息，了解数据结构

# 提取特征数据：所有行，除了最后一列
X = data_set.iloc[:, :-1]
# print(X)  # 可取消注释查看特征数据

# 提取标签数据：只保留"target"列（最后一列）
Y = data_set["target"]

# 划分训练集和测试集：测试集占20%，训练集占80%，随机种子为42以保证结果可复现
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 将整数标签转换为独热编码格式：2表示有两个类别
y_train_one = to_categorical(y_train, 2)  # 训练集标签的独热编码
y_test_one = to_categorical(y_test, 2)  # 测试集标签的独热编码
print(y_train.head())
print(y_train_one[:5])
# 特征归一化：将特征值缩放到0-1范围内，避免不同特征量纲差异影响模型训练
sc = MinMaxScaler(feature_range=(0, 1))  # 创建MinMaxScaler实例，指定缩放范围
x_train = sc.fit_transform(x_train)  # 对训练集进行拟合并转换
x_test = sc.fit_transform(x_test)  # 使用训练集的拟合结果对测试集进行转换

"""
    模型构建阶段：
    1. 创建Sequential模型
    2. 添加隐藏层
    3. 添加输出层
    4. 编译模型
"""
# 实例化Keras的Sequential模型，用于按层顺序构建神经网络
model = keras.Sequential()

# 添加第一个隐藏层：10个神经元，ReLU激活函数
model.add(Dense(
    units=10,  # 神经元数量
    activation="relu"  # 激活函数，引入非线性
))

# 添加第二个隐藏层：10个神经元，ReLU激活函数
model.add(Dense(
    units=10,
    activation="relu"
))

# 添加输出层：2个神经元（对应2个类别），Softmax激活函数（用于多分类概率输出）
model.add(Dense(
    units=2,
    activation="softmax"
))

# 编译模型：配置损失函数、优化器和评估指标
model.compile(
    loss="categorical_crossentropy",  # 分类交叉熵损失函数，适用于多分类问题
    optimizer="SGD",  # 随机梯度下降优化器
    metrics=["accuracy"]  # 评估指标：准确率
)

"""
    模型训练阶段：
    1. 使用训练集训练模型
    2. 验证集评估模型性能
    3. 保存训练历史和模型
"""
# 训练模型并记录训练历史
history = model.fit(
    x_train, y_train_one,  # 训练数据和标签
    epochs=100,  # 训练轮数
    batch_size=64,  # 批次大小：每次更新权重使用的样本数
    verbose=2,  # 日志显示模式：2表示每个epoch输出一行
    validation_data=(x_test, y_test_one)  # 验证集数据和标签
)
model.save("model.h5")  # 保存训练好的模型到文件
print(type(history))
# <class 'tensorflow.python.keras.callbacks.History'>
print(type(history.history))
# <class 'dict'>
print(history.history.keys())
# dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
"""
    模型评估与可视化阶段：
    1. 绘制训练和验证的损失曲线
    2. 绘制训练和验证的准确率曲线
"""
# 绘制训练集和验证集的损失曲线
plt.plot(history.history['loss'], label='训练集loss')  # 训练集损失
plt.plot(history.history['val_loss'], label='验证集loss')  # 验证集损失
plt.xlabel('迭代次数')  # X轴标签
plt.ylabel('loss值')  # Y轴标签
plt.legend()  # 显示图例
plt.show()  # 显示图像
plt.savefig("loss.png")  # 保存损失曲线图像


# 绘制训练集和验证集的准确率曲线
plt.plot(history.history['accuracy'], label='训练集准确率')  # 训练集准确率
plt.plot(history.history['val_accuracy'], label='验证集准确率')  # 验证集准确率
plt.xlabel('迭代次数')  # X轴标签
plt.ylabel('准确率')  # Y轴标签
plt.legend()  # 显示图例
plt.show()  # 显示图像
plt.savefig("accuracy.png")  # 保存准确率曲线图像
