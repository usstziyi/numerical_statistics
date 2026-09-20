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

# ============================================================
# t 分布 vs 标准正态：自由度越小，尾巴越厚
# ============================================================
xs = np.linspace(-5, 5, 1000)
dfs = [1, 3, 10, 30]
colors = ['#c44e52', '#dd8452', '#55a868', '#4c72b0']

plt.figure(figsize=(11, 4.5))

# 左：全貌对比——t 分布更矮更胖
plt.subplot(1, 2, 1)
plt.plot(xs, stats.norm.pdf(xs, 0, 1), 'k--', linewidth=2, label='N(0, 1)')
for df, c in zip(dfs, colors):
    plt.plot(xs, stats.t.pdf(xs, df), color=c, linewidth=2, label=f't(df={df})')
plt.title('t vs normal: overall shape')
plt.xlabel('x')
plt.ylabel('density')
plt.legend()
plt.grid(alpha=0.3)

# 右：只放大右尾，用对数纵轴——正态尾部塌得极快，t 分布仍保留可观概率
tail = np.linspace(2, 4.5, 500)
plt.subplot(1, 2, 2)
plt.plot(tail, stats.norm.pdf(tail, 0, 1), 'k--', linewidth=2, label='N(0, 1)')
for df, c in zip(dfs, colors):
    plt.plot(tail, stats.t.pdf(tail, df), color=c, linewidth=2, label=f't(df={df})')
plt.yscale('log')
plt.title('right tail (log scale)')
plt.xlabel('x')
plt.ylabel('density')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()