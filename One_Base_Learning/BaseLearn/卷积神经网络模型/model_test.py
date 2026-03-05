import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import cv2


plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 给数据类别放置到列表数据之中
CLASS_NAMES = np.array(["Cr", "In", "Pa", "PS", "Rs", "Sc"])
# 设置图片大小和批次数
BATCH_SIZE = 64
IMG_HEIGHT = 32
IMG_WIDTH = 32


# 加载模型
model = load_model("model.h5")

# 数据读取与处理
src = cv2.imread("./data/val/Pa/Pa_13.bmp")  # type:ignore
src = cv2.resize(src, (32, 32))  # type:ignore
src.astype("int32")
src = src / 255


# 扩充数据的维度
test_img = tf.expand_dims(src, 0)
# print(test_img.shape)   # (1, 32, 32, 3)

# 预测
preds = model.predict(test_img)
print(preds[0])
# [2.3423086e-04 6.2171804e-24 9.9976450e-01 1.3200427e-06 6.0849293e-20 9.3314722e-13]


print(f"模型预测的结果为:{CLASS_NAMES[np.argmax(preds[0])]}，概率为:{np.max(preds[0])}")
# 模型预测的结果为:Pa，概率为:0.9997645020484924

if __name__ == "__main__":
    pass
