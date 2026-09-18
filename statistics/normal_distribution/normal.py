import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 1. 生成横轴数据：从 -4 到 4，取 500 个点
x = np.linspace(-4, 4, 500)

# 2. 算每个点对应的概率密度（标准正态：μ=0, σ=1）
y = stats.norm.pdf(x, loc=0, scale=1)

# 3. 画图
plt.figure(figsize=(8, 4))
plt.plot(x, y, color='steelblue', linewidth=2, label='standard normal N(0,1)')
plt.title('Standard Normal Distribution')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()