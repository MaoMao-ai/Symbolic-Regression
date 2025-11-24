from pysr import PySRRegressor
import numpy as np

# 生成数据
X = np.random.rand(100, 1)
y = 3 * X[:, 0]**2 + 2 * X[:, 0] + 1 + 0.1 * np.random.randn(100)

# 训练 PySR
model = PySRRegressor(
    niterations=40,  # 迭代次数
    binary_operators=["+", "*", "-", "/"],  # 允许的二元运算符
    unary_operators=["sin", "cos", "exp", "log"],  # 允许的一元运算符
)
   
# 训练模型
model.fit(X, y)

# 输出发现的数学表达式
print(model)
