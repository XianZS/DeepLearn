# 导入所需的库
import numpy as np  # 用于科学计算，尤其是多维数组操作
import pandas as pd  # 用于数据分析和处理，特别是DataFrame的操作
from sklearn.model_selection import train_test_split  # 用于将数据集划分为训练集和测试集
from sklearn.preprocessing import MinMaxScaler  # 用于数据归一化，将特征缩放到指定范围
from sklearn.linear_model import LogisticRegression  # 导入逻辑回归模型
from sklearn.metrics import classification_report  # 用于生成分类报告，评估模型性能

"""
    第一步：读取数据
    使用pandas的read_csv函数从CSV文件中加载数据。
    CSV文件应与脚本位于同一目录，或者提供正确的文件路径。
"""
# 从CSV文件'./breast_cancer_data.csv'中读取数据到pandas DataFrame
data_set = pd.read_csv("./breast_cancer_data.csv")
# print(data_set) # 可以取消注释此行来查看加载的数据集

"""
    第二步：数据预处理和数据集划分
    - 将数据集分为特征（X）和目标（Y）。
    - 将整个数据集划分为训练集和测试集。
    - 对特征数据进行归一化处理，以消除不同特征之间的量纲影响，加速模型收敛。
"""
# 读取特征数据X：iloc[:, :-1] 表示选取所有行，以及除了最后一列之外的所有列作为特征
X = data_set.iloc[:, :-1]
print("特征数据X的前5行：\n", X.head()) # 打印特征数据的前几行，方便查看

# 读取目标变量Y：选取名为"target"的列作为目标变量
Y = data_set["target"]
print("\n目标变量Y的前5行：\n", Y.head()) # 打印目标变量的前几行，方便查看

# 使用train_test_split函数将数据集划分为训练集和测试集
# test_size=0.2 表示测试集占总数据的20%，剩余80%为训练集
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42) # 添加random_state保证每次划分结果一致

# 数据归一化处理
# 实例化MinMaxScaler对象，将数据缩放到0~1之间
sc = MinMaxScaler(feature_range=(0, 1))
# 使用训练集数据拟合（fit）归一化模型，并对训练集进行转换（transform）
x_train = sc.fit_transform(x_train)
# 使用之前拟合好的归一化模型对测试集进行转换（transform）。
# 注意：这里只使用transform，而不是fit_transform，以避免测试集信息泄露到训练过程中。
x_test = sc.transform(x_test)
# print(x_train, x_test) # 可以取消注释此行来查看归一化后的训练集和测试集数据

"""
    第三步：模型搭建
    - 实例化逻辑回归模型。
    - 使用训练数据对模型进行训练。
"""
# 实例化逻辑回归模型
lr = LogisticRegression()
# 使用训练集数据(x_train, y_train)训练逻辑回归模型
lr.fit(x_train, y_train)

# 打印模型训练后的参数
# lr.coef_ 存储模型的系数（权重w），对应每个特征的重要性
print(f"\n模型系数w: {lr.coef_}")
# lr.intercept_ 存储模型的截距（偏置b）
print(f"模型截距b: {lr.intercept_}")

"""
    第四步：模型测试
    - 使用训练好的模型对测试集进行预测。
    - 打印预测结果与真实结果的对比，并标记出预测错误的样本。
    - 打印模型预测的概率。
"""
# 使用训练好的模型对测试集x_test进行预测，得到预测的类别标签
pre_result = lr.predict(x_test)

print("\n模型预测结果与真实结果对比：")
# 遍历预测结果和真实结果，对比并打印
for i, (pre_y, y) in enumerate(zip(pre_result, y_test)):
    if pre_y == y:
        # print(f"样本 {i}: 预测值={pre_y}, 真实值={y}") # 预测正确时可以不打印，减少输出
        pass
    else:
        print(f"样本 {i}: 预测值={pre_y}, 真实值={y} (错误)") # 仅打印预测错误的样本

# 打印预测结果的概率：predict_proba返回每个样本属于每个类别的概率
pre_result_proba = lr.predict_proba(x_test)
print("\n模型预测概率（前5行）：\n", pre_result_proba[:5]) # 打印前5个样本的预测概率

# 获取恶性肿瘤（类别1）的概率
pre_bad = pre_result_proba[:, 1]
# 获取良性肿瘤（类别0）的概率
pre_good = pre_result_proba[:, 0]
# print("\n预测的类别标签：", pre_result) # 再次打印预测的类别标签

"""
    第五步：模型评估
    使用classification_report函数生成详细的分类报告，评估模型性能。
    - precision：精确率，代表模型预测为正类的样本中，实际为正类的比例。计算公式：TP / (TP + FP)
    - recall：召回率，代表实际为正类的样本中，模型预测为正类的比例。计算公式：TP / (TP + FN)
    - f1-score：F1分数，是精确率和召回率的调和平均值，综合衡量模型的性能。计算公式：2 * (precision * recall) / (precision + recall)
    - support：支持度，代表每个类别在测试集中出现的样本数量。
    - y_test：代表实际的类别标签，0通常代表良性肿瘤，1代表恶性肿瘤。
    - pre_result：代表模型预测的类别标签。
    - labels：指定要评估的类别标签列表，例如 [0, 1]。
    - target_names：为类别标签提供可读性更好的名称，例如 ["良性", "恶性"]。
    classification_report(测试集的真实结果, 模型预测的结果, 标签列表, 标签列表别名=["良性", "恶性"])
"""
print("\n模型分类报告：")
# 生成分类报告
report = classification_report(
    y_test,  # 真实的类别标签
    pre_result,  # 模型预测的类别标签
    labels=[0, 1],  # 指定要评估的类别标签
    target_names=["良性", "恶性"],  # 为类别标签指定名称
)
print(report) # 打印分类报告