import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 场景：全年级身高服从 N(173, 6)，σ 已知。抽到一个学生 182 cm，他算不算显著偏高？
# H0：μ = 173    他来自平均身高 173 的普通总体，182 只是抽样波动
# H1：μ ≠ 173    他来自均值不同的总体，确实偏高

np.random.seed(42)
mu, sigma = 173, 6
x = 182
alpha = 0.05

# scipy 没有 σ 已知的 z 检验函数，只能停在分布函数这一层：
# sf(x, loc, scale) 直接给出"大于 x"的面积，比 1 - cdf 在极端尾部精度更稳
z = (x - mu) / sigma
p_right = stats.norm.sf(x, loc=mu, scale=sigma)
p_z = 2 * stats.norm.sf(abs(z))

# σ 已知时 μ 的 95% 置信区间就是单个观测 ± 1.96σ，区间宽得一眼能看出结论
ci_low, ci_high = stats.norm.interval(0.95, loc=x, scale=sigma)

print(f"z = ({x} - {mu}) / {sigma} = {z:.2f}")
print(f"右侧尾巴 P(X > {x}) = {p_right:.4f}")
print(f"双侧 p-value = 2 * {p_right:.4f} = {p_z:.4f}")
print(f"μ 的 95% 置信区间 = [{ci_low:.2f}, {ci_high:.2f}] cm")

if p_z < alpha:
    print(f"p = {p_z:.4f} < {alpha}  ->  拒绝 H0：{x} 显著偏离 {mu}")
else:
    print(f"p = {p_z:.4f} >= {alpha}  ->  不拒绝 H0：{x} 算正常波动")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
xs = np.linspace(-4, 4, 1000)
ax.plot(xs, stats.norm.pdf(xs), color='#4c72b0', linewidth=2, label='N(0, 1)')
ax.fill_between(xs, stats.norm.pdf(xs), where=(np.abs(xs) > abs(z)),
                color='#c44e52', alpha=0.4, label='两侧尾巴 = p')
for k in (-1, 1):
    ax.axvline(k * abs(z), color='gray', linestyle=':', alpha=0.8)
ax.text(0.03, 0.95, f"z = {z:.2f}\np_z = {p_z:.4f}", transform=ax.transAxes,
        va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))
ax.set_title('H0 成立时 z 的分布')
ax.set_xlabel('z')
ax.set_ylabel('概率密度')
ax.legend(loc='upper right')
ax.grid(alpha=0.3)

# 右：换回原始尺度看这个区间 —— 它横跨 23 cm，当然盖得住 173
ax = axes[1]
ax.hlines(0.5, ci_low, ci_high, color='#4c72b0', linewidth=8, alpha=0.55)
ax.plot([x], [0.5], 'o', color='#4c72b0', markersize=10)
ax.axvline(mu, color='#c44e52', linestyle='--', linewidth=2)
ax.text(mu - 0.6, 0.78, f'μ = {mu}', color='#c44e52', ha='right', fontsize=10)
ax.text(ci_low, 0.24, f'{ci_low:.2f}', ha='center', fontsize=10)
ax.text(ci_high, 0.24, f'{ci_high:.2f}', ha='center', fontsize=10)
ax.text(x, 0.66, f'观测值 {x}', ha='center', fontsize=10)
ax.text(0.03, 0.95, f"μ 的 95% 置信区间\n[{ci_low:.2f}, {ci_high:.2f}]，包含 {mu}",
        transform=ax.transAxes, va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))
ax.set_ylim(0, 1.15)
ax.set_xlim(ci_low - 4, ci_high + 4)
ax.set_yticks([])
ax.set_title('单个观测给出的 μ 区间')
ax.set_xlabel('身高 (cm)')
ax.grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.show()