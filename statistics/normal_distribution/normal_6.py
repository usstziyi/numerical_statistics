import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats

# 中文标签需要指定字体，否则会显示成方块
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
# df = 9 的 t 分布
# 对应样本量 n = 10（df = n - 1），即 pvalue/students.py 里的场景
# ============================================================
df = 9
x = np.linspace(-5, 5, 1000)

t_pdf = stats.t.pdf(x, df)
norm_pdf = stats.norm.pdf(x, 0, 1)

# 95% 双侧临界值：t 比 z 更大，t 的区间更宽
t_crit = stats.t.ppf(0.975, df)
z_crit = stats.norm.ppf(0.975)
# `ppf` = p ercent p oint f unction，中文叫分位函数，它是`cdf` 的 反函数
# `ppf` 返回的就是 临界值critical value
# cdf(x) → 给一个 x，返回 P(X ≤ x)          「值 → 概率」
# ppf(q) → 给一个累积概率 q，返回对应的 x      「概率 → 值」

fig, ax = plt.subplots(figsize=(9, 5))

# 曲线：t(9) 与标准正态做参照
ax.plot(x, t_pdf, color='#c44e52', linewidth=2.5, label=f't(df={df})')
ax.plot(x, norm_pdf, color='black', linewidth=1.5, linestyle='--', label='N(0, 1)')

# 中间 95% 与两侧各 2.5%
ax.fill_between(x, t_pdf, where=(np.abs(x) <= t_crit), color='#55a868', alpha=0.3,
                label='中间 95%')
ax.fill_between(x, t_pdf, where=(np.abs(x) > t_crit), color='#c44e52', alpha=0.3,
                label='两侧各 2.5%')

# 临界值竖线
for k in (-1, 1):
    ax.axvline(k * t_crit, color='#c44e52', linestyle=':', alpha=0.8)
    ax.axvline(k * z_crit, color='gray', linestyle=':', alpha=0.8)
ax.axvline(0, color='gray', linestyle='--', alpha=0.5)

# 关键数值直接放在左上角，避免箭头位置随字体变动
ax.text(0.02, 0.97,
        f"95% 双侧临界值\n"
        f"t(df=9): ±{t_crit:.4f}\n"
        f"z: ±{z_crit:.4f}\n"
        f"t 区间宽出 {(t_crit / z_crit - 1):.2%}",
        transform=ax.transAxes, va='top', ha='left', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))

ax.set_title(f't 分布（df = {df}）与标准正态对比')
ax.set_xlabel('t')
ax.set_ylabel('概率密度')
ax.set_ylim(0, t_pdf.max() * 1.15)
ax.legend(loc='upper right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# 关键数值
# ============================================================
print(f"自由度 df = {df}")
print(f"均值 = 0（t 分布恒以 0 为中心）")
print(f"方差 = df/(df-2) = {df / (df - 2):.4f}，正态为 1")
print(f"95% 双侧临界值 t = ±{t_crit:.4f}，正态 z = ±{z_crit:.4f}")
print(f"P(|t| > 2) = {2 * (1 - stats.t.cdf(2, df)):.4f}，"
      f"对应 P(|z| > 2) = {2 * (1 - stats.norm.cdf(2)):.4f}")
