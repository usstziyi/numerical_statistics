from scipy.stats import ttest_ind, shapiro, mannwhitneyu, levene
import numpy as np

# 两组独立样本
a = [0.71, 0.73, 0.69, 0.75, 0.72]
b = [0.68, 0.70, 0.67, 0.72, 0.69]

n1, n2 = len(a), len(b)
print(f"样本量: A = {n1}, B = {n2}")

# ============================================================
# 1. 描述性统计
# ============================================================
print("\n=== 描述性统计 ===")
print(f"A: mean = {np.mean(a):.4f}, sd = {np.std(a, ddof=1):.4f}, "
      f"median = {np.median(a):.4f}")
print(f"B: mean = {np.mean(b):.4f}, sd = {np.std(b, ddof=1):.4f}, "
      f"median = {np.median(b):.4f}")

# ============================================================
# 2. 正态性检验（仅作参考，不作为切换开关）
# ============================================================
print("\n=== Shapiro-Wilk 正态性检验（仅供参考）===")
for name, data in [("A", a), ("B", b)]:
    stat, p = shapiro(data)
    print(f"{name}: W = {stat:.4f}, p = {p:.3e}")

if n1 < 10 or n2 < 10:
    print("警告：样本量过小，Shapiro-Wilk 结果不可靠，不建议据此选择检验方法。")

# ============================================================
# 3. 方差齐性检验（仅作参考）
# ============================================================
print("\n=== Levene 方差齐性检验（仅供参考）===")
lev_stat, lev_p = levene(a, b)
print(f"Levene: statistic = {lev_stat:.4f}, p = {lev_p:.3e}")

# ============================================================
# 4. 主检验：默认使用 Welch t 检验
# ============================================================
# 理由：Welch t 检验不要求方差齐性，对正态性偏离也较稳健。
# 如果研究设计明确要求 Student t 检验，且方差齐、正态，
# 可改用 equal_var=True。

print("\n=== Welch's independent t-test ===")
result = ttest_ind(a, b, equal_var=False)
print(f"t-statistic: {result.statistic:.4f}")
print(f"p-value: {result.pvalue:.4f}")
print(f"df: {result.df:.2f}")

# 均值差与 95% 置信区间
mean_diff = np.mean(a) - np.mean(b)
se_diff = np.sqrt(np.var(a, ddof=1)/n1 + np.var(b, ddof=1)/n2)
# Welch 自由度
df_welch = (np.var(a, ddof=1)/n1 + np.var(b, ddof=1)/n2)**2 / (
    (np.var(a, ddof=1)/n1)**2/(n1-1) + (np.var(b, ddof=1)/n2)**2/(n2-1)
)
from scipy.stats import t as t_dist
t_crit = t_dist.ppf(0.975, df_welch)
ci_low = mean_diff - t_crit * se_diff
ci_high = mean_diff + t_crit * se_diff
print(f"均值差 (A - B): {mean_diff:.4f}")
print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

# Cohen's d（用合并标准差）
pooled_sd = np.sqrt(((n1-1)*np.var(a, ddof=1) + (n2-1)*np.var(b, ddof=1)) / (n1+n2-2))
cohens_d = mean_diff / pooled_sd
print(f"Cohen's d: {cohens_d:.4f}")

# ============================================================
# 5. 非参数检验（作为稳健性参考）
# ============================================================
print("\n=== Mann-Whitney U test（稳健性参考）===")
u_result = mannwhitneyu(a, b, alternative="two-sided")
print(f"U-statistic: {u_result.statistic:.4f}")
print(f"p-value: {u_result.pvalue:.4f}")