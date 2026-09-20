import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 场景：σ 未知，只能用样本标准差 s 去估。抽 30 个学生，检验班级平均身高是否等于全校 173。
# H0：μ = 173    班级平均身高与全校一致，样本均值的偏离只是抽样波动
# H1：μ ≠ 173    班级平均身高与全校不同，确实偏高或偏低

np.random.seed(42)
n = 30
mu0 = 173.0
heights = np.random.normal(loc=182, scale=6, size=n)

res = stats.ttest_1samp(
        heights,         # 待检验的样本：抽到的 30 个学生身高
        popmean=mu0,     # H0 设定的总体均值（全校平均身高）
        alternative='two-sided'  # 两尾检验（默认）
)
# df 不在解包结果里（元组只有 statistic / pvalue），作为属性从 res 上取
t_stat, p_value, df = res.statistic, res.pvalue, res.df

xbar = heights.mean()
s = heights.std(ddof=1)
se = s / np.sqrt(n)
ci_low, ci_high = res.confidence_interval(confidence_level=0.95)

print(f"样本：n = {n}，均值 = {xbar:.2f}，标准差 s = {s:.2f}")
print(f"标准误 se = s / √n = {s:.2f} / {np.sqrt(n):.2f} = {se:.2f}")
print(f"t = ({xbar:.2f} - {mu0}) / {se:.2f} = {t_stat:.3f}，df = {n} - 1 = {df:.0f}")
print(f"双侧 p-value = {p_value:.3e}")
print(f"均值 95% 置信区间 = [{ci_low:.2f}, {ci_high:.2f}] cm")

alpha = 0.05
if p_value < alpha:
    print(f"p = {p_value:.3e} < {alpha}  ->  拒绝 H0：班级平均身高显著不同于 {mu0}")
else:
    print(f"p = {p_value:.3e} >= {alpha}  ->  不拒绝 H0：没有足够证据说明班级平均身高不等于 {mu0}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# 左：样本落在哪，离 H0 说的 173 有多远
ax = axes[0]
ax.hist(heights, bins=10, density=True, color='#4c72b0', alpha=0.55,
        edgecolor='white', label=f'样本 n={n}')
ax.axvline(xbar, color='#4c72b0', linestyle='--', linewidth=2, label=f'样本均值 {xbar:.1f}')
ax.axvline(mu0, color='#c44e52', linestyle='--', linewidth=2, label=f'H0：μ = {mu0:.0f}')
ax.text(0.03, 0.95,
        f"样本均值 = {xbar:.2f}\nH0 均值 = {mu0:.0f}\n差距 = {xbar - mu0:.2f} cm",
        transform=ax.transAxes, va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))
ax.set_title('班级样本的身高分布')
ax.set_xlabel('身高 (cm)')
ax.set_ylabel('概率密度')
ax.legend(loc='upper right', fontsize=8)
ax.grid(alpha=0.3)

# 右：H0 成立时 t 该长什么样。观测 t 落在这么偏的位置，
# 两侧尾巴在图上已经小到看不见 —— 这正是 p 极小的直观含义
ax = axes[1]
xs = np.linspace(-5, 5, 1000)
ax.plot(xs, stats.t.pdf(xs, df), color='#4c72b0', linewidth=2, label=f't(df={df:.0f})')
ax.fill_between(xs, stats.t.pdf(xs, df), where=(np.abs(xs) > abs(t_stat)),
                color='#c44e52', alpha=0.4)
for k in (-1, 1):
    ax.axvline(k * abs(t_stat), color='gray', linestyle=':', alpha=0.8)
ax.text(0.03, 0.95, f"t = {t_stat:.2f}, df = {df:.0f}\np = {p_value:.3e}",
        transform=ax.transAxes, va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))
ax.set_title('H0 成立时 t 的分布')
ax.set_xlabel('t')
ax.set_ylabel('概率密度')
ax.legend(loc='upper right')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()