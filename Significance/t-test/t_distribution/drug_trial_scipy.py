import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)
n1, n2 = 30, 30
drug = np.random.normal(loc=14, scale=6, size=n1)
placebo = np.random.normal(loc=8, scale=6, size=n2)

res = stats.ttest_ind(
        drug,           # 第一个样本数据（新药组）
        placebo,        # 第二个样本数据（安慰剂组）
        equal_var=True,  # 是否假设两组方差相等（True 表示使用独立样本 t 检验的合并方差公式）
        alternative='two-sided'  # 两尾检验（默认）：只问两组有没有差异，不预设新药更好的方向
)
t_stat, p_value, df = res.statistic, res.pvalue, res.df

m1, m2 = drug.mean(), placebo.mean()
s1, s2 = drug.std(ddof=1), placebo.std(ddof=1)
diff = m1 - m2
sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
se = sp * np.sqrt(1 / n1 + 1 / n2)
d = diff / sp
ci_low, ci_high = res.confidence_interval(confidence_level=0.95)

print(f"新药组：均值 = {m1:.2f}，标准差 = {s1:.2f}")
print(f"安慰剂组：均值 = {m2:.2f}，标准差 = {s2:.2f}")
print(f"差值 = {diff:.2f} mmHg")
print(f"合并标准差 sp = {sp:.2f}，标准误 se = {se:.2f}")
print(f"t = {diff:.2f} / {se:.2f} = {t_stat:.3f}，df = {df:.0f}")
print(f"双侧 p-value = {p_value:.4f}")
print(f"差值 95% 置信区间 = [{ci_low:.2f}, {ci_high:.2f}] mmHg")
print(f"效应量 Cohen's d = {diff:.2f} / {sp:.2f} = {d:.2f}")

alpha = 0.05
if p_value < alpha:
    print(f"p = {p_value:.4f} < {alpha}  ->  拒绝 H0：新药与安慰剂的差异显著")
else:
    print(f"p = {p_value:.4f} >= {alpha}  ->  不拒绝 H0：没有足够证据说明新药有效")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

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

ax = axes[1]
xs = np.linspace(-5, 5, 1000)
ax.plot(xs, stats.t.pdf(xs, df), color='#4c72b0', linewidth=2,
        label=f't(df={df:.0f})')
ax.fill_between(xs, stats.t.pdf(xs, df), where=(np.abs(xs) > abs(t_stat)),
                color='#c44e52', alpha=0.4)
for k in (-1, 1):
    ax.axvline(k * abs(t_stat), color='gray', linestyle=':', alpha=0.8)
ax.text(0.03, 0.95, f"t = {t_stat:.2f}, df = {df:.0f}\np = {p_value:.4f}",
        transform=ax.transAxes, va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))
ax.set_title('H0 成立时 t 的分布')
ax.set_xlabel('t')
ax.set_ylabel('概率密度')
ax.legend(loc='upper right')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()