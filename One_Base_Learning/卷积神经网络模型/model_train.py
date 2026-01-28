# 夹杂（IN） 划痕（SC） 压入氧化皮（PS） 裂纹（CR） 麻点（RS） 板块（PA）
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
import keras


plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

""" 数据预处理步骤 """

# 读取训练集数据
data_train = "./data/train/"
data_train = pathlib.Path(data_train)
# 读取验证集数据
data_val = "./data/val/"
data_val = pathlib.Path(data_val)
# print(str(data_val))
# <class 'pathlib.WindowsPath'>
# 给数据类别放置到列表数据之中
CLASS_NAMES = np.array(["Cr", "In", "Pa", "PS", "Rs", "Sc"])
# 设置图片大小和批次数
BATCH_SIZE = 64
IMG_HEIGHT = 32
IMG_WIDTH = 32

# 对数据进行归一化处理
# 归一化处理器
image_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

# 训练集生成器
train_data_gen = image_generator.flow_from_directory(
    directory=str(data_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    classes=list(CLASS_NAMES),
)

# 验证集生成
val_data_gen = image_generator.flow_from_directory(
    directory=str(data_val),
    batch_size=BATCH_SIZE,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    classes=list(CLASS_NAMES),
)


""" 利用keras搭建神经网络模型 """
# 实例化神经网络模型
model = keras.Sequential()
# 添加第一层 卷积层
model.add(Conv2D(filters=6, kernel_size=5, input_shape=(32, 32, 3), activation="relu"))  # type:ignore
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=16, kernel_size=5, activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=120, kernel_size=5, activation="relu"))
model.add(Flatten())
model.add(Dense(84, activation="relu"))
model.add(Dense(6, activation="softmax"))

""" 编译神经网络模型 """
# 编译
model.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy"])
history = model.fit(train_data_gen, validation_data=val_data_gen, epochs=100)

# 保存模型
model.save("model.h5")

""" 绘制图像 """
# loss 与 val_loss 图像
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("CNN神经网络loss值")
plt.legend()
plt.show()

# 准确率图像
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("CNN神经网络accuracy值")
plt.legend()
plt.show()
