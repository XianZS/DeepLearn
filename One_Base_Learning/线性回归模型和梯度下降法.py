"""
    # 实战内容
    线性回归，使用梯度下降法对小明的成绩进行求解。
    # 实战步骤
    第一步：
        数据，学习时间与成绩得分之间的关系。
    第二步：
        模型。
    第三步：
        损失函数。
    第四步：
        梯度求导。
    第五步：
        利用梯度更新参数。
    第六步：
        设置训练轮次。
    第七步：
        使用更新之后的参数，进行推理预测。
"""
# [第一步]，定义数据
x_data = [1, 2, 3]
y_data = [2, 4, 6]
# 初始化参数w
w = 4


# [第二步]，定义线性回归模型

def forward(x):
    return x * w


# [第三步]，定义损失函数
def cost(xs, ys):
    cost_value = 0
    for x, y in zip(x_data, y_data):
        # 计算x对应的预测值y_pred
        y_pred = forward(x)
        # 计算损失cost_value
        cost_value += (y_pred - y) ** 2
    # 返回损失平均数值
    return cost_value / max(len(x_data), len(y_data))


# [第四步]，定义计算梯度的公式
def gradient(xs, ys):
    """
        主要是利用第三步之中的损失函数，来不断更新梯度。
    """
    # 初始化梯度
    gradient_value = 0
    for x, y in zip(x_data, y_data):
        gradient_value += 2 * x * (x * w - y)
    return gradient_value / max(len(x_data), len(y_data))


# [第五步]，利用梯度进行更新参数
for index in range(100):
    # 得到第index次的损失值
    cost_val = cost(x_data, y_data)
    # 计算梯度
    grad_val = gradient(x_data, y_data)
    # 参数更新
    w = w - 0.01 * grad_val
    print(f"训练轮次[{index}]===损失：{cost_val},梯度：{grad_val}，w：{w}")

print(f"训练4小时，得分为{forward(4)}")
# 训练4小时，得分为8.000444482757587
