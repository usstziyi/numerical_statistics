import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

x = np.linspace(-4, 4, 1000)
y = stats.norm.pdf(x, 0, 1)

plt.figure(figsize=(9, 5))
plt.plot(x, y, color='black', linewidth=2)

# 用不同颜色填充三段区域
for k, color in zip([1, 2, 3], ['#4c72b0', '#55a868', '#c44e52']):
    mask = (x >= -k) & (x <= k)
    plt.fill_between(x, y, where=mask, color=color, alpha=0.3,
                     label=f'μ±{k}σ ≈ {stats.norm.cdf(k)-stats.norm.cdf(-k):.1%}')

plt.title('正态分布的三段常用区间')
plt.xlabel('x')
plt.ylabel('概率密度')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()