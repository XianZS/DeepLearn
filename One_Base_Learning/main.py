"""
    使用梯度下降法，求解线性回归方程
"""
# 第一步：自定义数据集合
x_data = [10, 20, 30, 40]
y_data = [100, 200, 300, 400]
# 初始化参数w
w = 20


# 第二步：定义线性回归模型
def forward(x):
    # y = w * x + b
    return w * x


# 第三步：定义损失值函数
def cost_function(x_data, y_data):
    """
        需要不断拟合w，并不知道拟合方向对不对，真实情况下w可能越来越远离期待值。
        所以需要一个参数，来衡量w是不是越来越趋近于咱们的期待值。
        损失值：loss，loss越小，越接近，loss越大，就需要调整学习率α。
        因为针对当前轮次，咱们会得到一个w_now,w_now*x就会得到一个y_pred的预期值，（y_pred-y）**2，累加，这就是损失值。
    """
    # 定义损失值的初值
    cost_value = 0
    for x, y in zip(x_data, y_data):
        # 根据当前轮的w_now，以及x计算y的预期值y_pred
        y_pred = forward(x)
        cost_value += (y - y_pred) ** 2
    # 返回平均损失值
    return cost_value / max(len(x_data), len(y_data))


# 第四步：定义梯度计算公式
def gradient(x_data, y_data):
    """
        定义一个梯度公式，不断更新迭代w。
    """
    now_grad = 0
    for x, y in zip(x_data, y_data):
        now_grad += 2 * x * (x * w - y)
    return now_grad / max(len(x_data), len(y_data))


# 第五步：利用梯度进行更新迭代数据
for index in range(50):
    # 先得到第index轮的损失值
    index_cost_value = cost_function(x_data, y_data)
    # 接下来得到第index轮的梯度
    index_grad_value = gradient(x_data, y_data)
    # 利用梯度进行参数更新，更新w
    w = w - 0.001 * index_grad_value
    print(f"[{index}]:cost-[{index_cost_value}],grad-[{index_grad_value}],w-[{w}]")

print(f"如果投资500元，那么收入为{forward(500)}")
