import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

x = np.linspace(-8, 8, 500)

plt.figure(figsize=(10, 5))

# `subplot()` 的作用是 激活 某一块，
# 之后所有绘图命令（`plot` 、`title` 、`xlabel` 、`legend` 、`grid` ）都作用在这块激活的区域上，
# 直到下一次调用`subplot()` 切换。

# 固定 σ=1，改 μ
plt.subplot(1, 2, 1)
for mu in [-2, 0, 2]:
    plt.plot(x, stats.norm.pdf(x, mu, 1), label=f'μ={mu}, σ=1')
plt.title('change μ')
plt.xlabel('x')
plt.ylabel('density')
plt.legend()
plt.grid(alpha=0.3)

# 固定 μ=0，改 σ
plt.subplot(1, 2, 2)
for sigma in [0.5, 1, 2]:
    plt.plot(x, stats.norm.pdf(x, 0, sigma), label=f'μ=0, σ={sigma}')
plt.title('change σ')
plt.xlabel('x')
plt.ylabel('density')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()