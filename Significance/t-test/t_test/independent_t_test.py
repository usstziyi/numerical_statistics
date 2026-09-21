from scipy.stats import ttest_ind, shapiro, mannwhitneyu

"""
independent t-test：独立样本 t 检验

用于比较两组相互独立样本的均值是否存在显著差异。

如果数据明显不满足正态性，可以考虑：
Mann-Whitney U test（Wilcoxon rank-sum test）
"""

# 两组独立样本
# 例如：A 组和 B 组是不同的受试者，不存在一一配对关系

a = [0.71, 0.73, 0.69, 0.75, 0.72]
b = [0.68, 0.70, 0.67, 0.72, 0.69]


# ============================================================
# 1. Shapiro-Wilk 正态性检验
# ============================================================

shapiro_a = shapiro(a)
shapiro_b = shapiro(b)

print("=== Shapiro-Wilk 正态性检验 ===")

print(f"A: statistic = {shapiro_a.statistic:.4f}, "
      f"p-value = {shapiro_a.pvalue:.9f}")

print(f"B: statistic = {shapiro_b.statistic:.4f}, "
      f"p-value = {shapiro_b.pvalue:.9f}")


# ============================================================
# 2. 根据正态性选择统计检验
# ============================================================

if shapiro_a.pvalue > 0.05 and shapiro_b.pvalue > 0.05:

    print("\n两组样本没有显著偏离正态分布")
    print("使用 Welch's independent t-test")

    result = ttest_ind(
        a,
        b,
        equal_var=False
    )

    print("\n=== Welch's t-test ===")
    print(f"t-statistic: {result.statistic:.4f}")
    print(f"p-value: {result.pvalue:.9f}")

else:

    print("\n至少一组样本显著偏离正态分布")
    print("使用 Mann-Whitney U test")

    result = mannwhitneyu(
        a,
        b,
        alternative="two-sided"
    )

    print("\n=== Mann-Whitney U test ===")
    print(f"U-statistic: {result.statistic:.4f}")
    print(f"p-value: {result.pvalue:.9f}")