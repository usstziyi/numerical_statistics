import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 中文标签需要指定字体，否则会显示成方块
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 场景：新药 II 期临床试验，看新药是不是真能降血压。
# 试验组 30 人吃新药，对照组 30 人吃安慰剂，
# 指标 = 服药 8 周后收缩压的下降值（mmHg），降得越多越好。
#
# H0：μ_药 = μ_安慰剂   新药没用，两组差异只是抽样运气
# H1：μ_药 ≠ μ_安慰剂   新药有效
#
# 和 p_value.py 的区别：
#   那里只有 1 个观测值，比的是"这个值离总体均值 μ 多远"；
#   这里有两组样本，比的是"两个均值之差"，所以要自己算标准误。
# ============================================================
np.random.seed(42)
n1, n2 = 30, 30
drug = np.random.normal(loc=14, scale=6, size=n1)     # 新药组，真值降 14
placebo = np.random.normal(loc=8, scale=6, size=n2)   # 安慰剂组，真值降 8
# 注意对照组不是 0：安慰剂效应是真实存在的，所以两组都降，只是降多降少的差别

# 第 1 步：两组各自的均值和标准差，以及要检验的差值
m1, m2 = drug.mean(), placebo.mean()
s1, s2 = drug.std(ddof=1), placebo.std(ddof=1)
diff = m1 - m2

# 第 2 步：合并标准差（pooled SD）。两组方差差不多时，合起来估一个共同 σ。
# 分子是两个离差平方和相加，分母是两个自由度相加 (n1-1) + (n2-1)
sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))

# 第 3 步：标准误。均值之差的标准误 = sp * sqrt(1/n1 + 1/n2)
# 两组各自都有抽样波动，所以比单个均值的标准误还要再大一点
se = sp * np.sqrt(1 / n1 + 1 / n2)

# 第 4 步：t 统计量 = 观测到的差值 / 这个差值在 H0 下的波动尺度
# 除以 se 就是在问：这点差异，相当于几个标准误？
t_stat = diff / se
df = n1 + n2 - 2      # 两组各用掉 1 个自由度去估均值

# 第 5 步：双侧 p-value = t 分布两侧尾巴的面积
p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

print(f"新药组：均值 = {m1:.2f}，标准差 = {s1:.2f}")
print(f"安慰剂组：均值 = {m2:.2f}，标准差 = {s2:.2f}")
print(f"差值 = {diff:.2f} mmHg")
print(f"合并标准差 sp = {sp:.2f}，标准误 se = {se:.2f}")
print(f"t = {diff:.2f} / {se:.2f} = {t_stat:.3f}，df = {n1} + {n2} - 2 = {df}")
print(f"双侧 p-value = {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print(f"p = {p_value:.4f} < {alpha}  ->  拒绝 H0：新药与安慰剂的差异显著")
else:
    print(f"p = {p_value:.4f} >= {alpha}  ->  不拒绝 H0：没有足够证据说明新药有效")

# ============================================================
# 用 scipy 现成函数核对手算结果。
# ttest_ind 默认就是双侧 + 合并方差（equal_var=True），正好对应上面
# ============================================================
t_scipy, p_scipy = stats.ttest_ind(drug, placebo)
print(f"\nscipy 核对：t = {t_scipy:.3f}，p = {p_scipy:.4f}")

# ============================================================
# p 值小不等于药效好。p 只回答"这点差异有多难用运气解释"，
# 而"差异本身有多大"要看效应量 Cohen's d = 差值 / 合并标准差。
# 样本量越大 p 越容易变小，但 d 不会跟着变 —— 这就是审评要看 d 的原因。
# ============================================================
d = diff / sp
print(f"效应量 Cohen's d = {diff:.2f} / {sp:.2f} = {d:.2f}"
      f"   # 0.2 小 / 0.5 中 / 0.8 大")

# ============================================================
# 画出来看看
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# 左：两组数据各自的分布，看差异有多大、重叠有多少
ax = axes[0]
bins = np.linspace(-8, 32, 17)
ax.hist(drug, bins=bins, density=True, color='#4c72b0', alpha=0.55,
        edgecolor='white', label=f'新药组 n={n1}')
ax.hist(placebo, bins=bins, density=True, color='#c44e52', alpha=0.55,
        edgecolor='white', label=f'安慰剂组 n={n2}')
ax.axvline(m1, color='#4c72b0', linestyle='--', linewidth=2)
ax.axvline(m2, color='#c44e52', linestyle='--', linewidth=2)
ax.text(0.03, 0.95,
        f"新药组均值 = {m1:.1f}\n安慰剂组均值 = {m2:.1f}\n差值 = {diff:.1f} mmHg",
        transform=ax.transAxes, va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))
ax.set_title('两组各自的血压下降值')
ax.set_xlabel('收缩压下降值 (mmHg)')
ax.set_ylabel('概率密度')
ax.legend(loc='upper right')
ax.grid(alpha=0.3)

# 右：H0 成立时 t 统计量该长什么样，观测值落在哪
# 注：t 落在这么偏的位置，两侧尾巴已经小到图上几乎看不见了 ——
# 这正是 p 很小的直观含义，具体数值看文本框
ax = axes[1]
xs = np.linspace(-5, 5, 1000)
ax.plot(xs, stats.t.pdf(xs, df), color='#4c72b0', linewidth=2,
        label=f't(df={df})')
ax.fill_between(xs, stats.t.pdf(xs, df), where=(np.abs(xs) > abs(t_stat)),
                color='#c44e52', alpha=0.4)
for k in (-1, 1):
    ax.axvline(k * abs(t_stat), color='gray', linestyle=':', alpha=0.8)
ax.text(0.03, 0.95, f"t = {t_stat:.2f}, df = {df}\np = {p_value:.4f}",
        transform=ax.transAxes, va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))
ax.set_title('H0 成立时 t 的分布')
ax.set_xlabel('t')
ax.set_ylabel('概率密度')
ax.legend(loc='upper right')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
