import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================
# 图 1：标准化前 vs 标准化后（三个 μ、σ 都不同的分布）
# ============================================================
# 每组：(μ, σ, 颜色, 线宽——线宽递减是为了让重合的曲线都看得见)
params = [
    (0.0, 1.0, '#4c72b0', 6.0),
    (-2.0, 0.8, '#55a868', 3.0),
    (3.0, 1.5, '#c44e52', 1.2),
]

x = np.linspace(-6, 8, 600)
plt.figure(figsize=(11, 4.5))

# 左：标准化之前，横轴是原始值
plt.subplot(1, 2, 1)
for mu, sigma, c, lw in params:
    plt.plot(x, stats.norm.pdf(x, mu, sigma),
             color=c, linewidth=2, label=f'N(μ={mu}, σ={sigma})')
plt.title('before: raw scale')
plt.xlabel('x')
plt.ylabel('density')
plt.legend()
plt.grid(alpha=0.3)

# 右：标准化之后，横轴换成 z
plt.subplot(1, 2, 2)
for mu, sigma, c, lw in params:
    z = (x - mu) / sigma                               # z = (x - μ) / σ
    plt.plot(z, sigma * stats.norm.pdf(x, mu, sigma),  # 密度变换要乘 σ
             color=c, linewidth=lw, alpha=0.8,
             label=f'N(μ={mu}, σ={sigma})')
plt.title('after: z-score scale')
plt.xlabel('z')
plt.ylabel('density')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# 图 2：身高例子，标注 182 cm 换算成 z = 1.5 的过程
# ============================================================
mu, sigma, x0 = 173, 6, 182
z0 = (x0 - mu) / sigma          # 1.5

xs = np.linspace(mu - 4*sigma, mu + 4*sigma, 600)
zs = np.linspace(-4, 4, 600)

plt.figure(figsize=(11, 4.5))

# 左：原始尺度，标出 μ 与 x0
plt.subplot(1, 2, 1)
plt.plot(xs, stats.norm.pdf(xs, mu, sigma), color='steelblue', linewidth=2)
plt.axvline(mu, color='black', linestyle='--', alpha=0.6)
plt.axvline(x0, color='gray', linestyle=':')
plt.annotate(f'x={x0}\nz={z0:.1f}σ', xy=(x0, 0.005),
             xytext=(x0 + 4, 0.03), arrowprops=dict(arrowstyle='->'))
plt.title(f'before: N(μ={mu}, σ={sigma})')
plt.xlabel('height (cm)')
plt.ylabel('density')
plt.grid(alpha=0.3)

# 右：z 尺度，同一个点落到 1.5
plt.subplot(1, 2, 2)
plt.plot(zs, stats.norm.pdf(zs, 0, 1), color='steelblue', linewidth=2)
plt.axvline(0, color='black', linestyle='--', alpha=0.6)
plt.axvline(z0, color='gray', linestyle=':')
plt.annotate(f'z={z0:.1f}', xy=(z0, 0.02), xytext=(z0 + 0.5, 0.12),
             arrowprops=dict(arrowstyle='->'))
plt.title('after: N(0, 1)')
plt.xlabel('z')
plt.ylabel('density')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
