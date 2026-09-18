import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 中文标签需要指定字体，否则会显示成方块
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 场景：全年级身高服从 N(173, 6)。抽到一个学生 182 cm，
# 他是不是"显著偏高"？
# H0：μ = 173    ← 他来自平均身高 173 的总体，就是个普通抽样
# H1：μ ≠ 173    ← 他来自一个均值更高的总体，确实偏高
# 假设 μ = 173（他来自普通总体），那么随机抽一个人，身高达到 182 及以上 或者 164 及以下（双侧）的概率是 13.36%。
# 被假设的是" 总体 "，被计算的是" 抽样的极端程度 "，观测值本身从头到尾都是固定的。
#
# p-value 的定义：在 H0（身高=173 的总体）成立的前提下，
# 观测到 182 这么极端甚至更极端结果的总概率。
# 它越小，说明用H0推出当下结果的概率越小，越倾向拒绝 H0。
# 当p=0时，说明H0推不出当下的结果，说明H0是错误的。
# ============================================================
mu, sigma = 173, 6    # 全年级 μ、σ 已知 → 用 z-score
x = 182               # 抽到的学生
alpha = 0.05          # 显著性水平

# 第 1 步：标准化，把 182 放到"距均值几个标准差"的尺度上
z = (x - mu) / sigma

# 第 2 步：z 左侧的累积概率（曲线下 z 左边的面积，就是 cdf）
p_left = stats.norm.cdf(z)

# 第 3 步：右侧尾巴 = 比 182 更高（更极端）的概率
p_right = 1 - p_left

# 第 4 步：双侧 p-value = 左右两条尾巴之和。
# 因为事先不预测"偏高还是偏低"，所以两边极端都要算进去
p_z = 2 * p_right

print(f"z = ({x} - {mu}) / {sigma} = {z:.2f}")
print(f"P(Z <= {z:.2f}) = {p_left:.4f}       # 左侧面积 = cdf")
print(f"右侧尾巴 = {p_right:.4f}")
print(f"双侧 p-value = 2 * {p_right:.4f} = {p_z:.4f}")

if p_z < alpha:
    print(f"p = {p_z:.4f} < {alpha}  ->  拒绝 H0：182 显著偏高")
else:
    print(f"p = {p_z:.4f} >= {alpha}  ->  不拒绝 H0：182 算正常波动")

# ============================================================
# 若 σ 未知，只能用样本标准差 s 估 → 不再服从 N(0,1)，
# 而要查 t 分布（df = n - 1，尾部更厚，p 略大）。
# 公式形状一样，只是把 norm 换成 t、加一个自由度。
# ============================================================
n = 30
mu = 173.0
s = 6.0      # 样本标准差。故意取 s = σ = 6，好让差别只来自分布本身

# 第 1 步：标准化（分母从 σ 换成 s，统计量从 z 改叫 t）
t_stat = (x - mu) / s # t-statistic(t统计量)

# 第 2 步：t_stat 左侧的累积概率（cdf 换成 t 的，df = n - 1）
t_left = stats.t.cdf(t_stat, df=n - 1)

# 第 3 步：右侧尾巴
t_right = 1 - t_left

# 第 4 步：双侧 p-value，和上面 z 的四步一一对应
p_t = 2 * t_right

print()
print(f"若 σ 未知，改用样本 s = {s}（df = n - 1 = {n - 1}）：")
print(f"  t = ({x} - {mu}) / {s} = {t_stat:.2f}   # 数值和 z 一样")
print(f"  P(T <= {t_stat:.2f}) = {t_left:.4f}，右侧尾巴 = {t_right:.4f}")
print(f"  双侧 p-value = 2 * {t_right:.4f} = {p_t:.4f}")
print(f"  比上面的 {p_z:.4f} 大，因为 t 尾部更厚 —— 和 z_score.py 是同一个道理")

# ============================================================
# 把两种口径画出来：两侧尾巴的面积就是各自的 p-value。
# 两个面板共用横轴范围，但分布不同 —— 尾巴胖瘦的差别一眼可见。
# ============================================================
xs = np.linspace(-4, 4, 1000)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

# 左：z 口径，分母是已知的 σ，统计量服从标准正态
ax = axes[0]
ax.plot(xs, stats.norm.pdf(xs), color='#4c72b0', linewidth=2, label='N(0, 1)')
ax.fill_between(xs, stats.norm.pdf(xs), where=(np.abs(xs) > abs(z)),
                color='#c44e52', alpha=0.4, label='两侧尾巴')
for k in (-1, 1):
    ax.axvline(k * abs(z), color='gray', linestyle=':', alpha=0.8)
ax.text(0.03, 0.95, f"z = {z:.2f}\np_z = {p_z:.4f}", transform=ax.transAxes,
        va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))
ax.set_title('z 口径：σ 已知 → N(0, 1)')
ax.set_xlabel('z')
ax.set_ylabel('概率密度')
ax.legend(loc='upper right')
ax.grid(alpha=0.3)

# 右：t 口径，分母是估出来的 s，统计量服从 t(df = n - 1)
ax = axes[1]
ax.plot(xs, stats.t.pdf(xs, df=n - 1), color='#c44e52', linewidth=2,
        label=f't(df={n - 1})')
# 把标准正态叠上来做参照，两条曲线的错位只在尾部
ax.plot(xs, stats.norm.pdf(xs), color='gray', linestyle='--', linewidth=1.5,
        label='N(0, 1)（参照）')
ax.fill_between(xs, stats.t.pdf(xs, df=n - 1), where=(np.abs(xs) > abs(t_stat)),
                color='#c44e52', alpha=0.4, label='两侧尾巴')
for k in (-1, 1):
    ax.axvline(k * abs(t_stat), color='gray', linestyle=':', alpha=0.8)
ax.text(0.03, 0.95, f"t = {t_stat:.2f}, df = {n - 1}\np_t = {p_t:.4f}",
        transform=ax.transAxes, va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))
ax.set_title(f't 口径：σ 用 s 估 → t(df={n - 1})')
ax.set_xlabel('t')
ax.legend(loc='upper right')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()