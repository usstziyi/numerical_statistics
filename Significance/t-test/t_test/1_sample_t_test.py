import numpy as np
from scipy.stats import ttest_1samp, shapiro, wilcoxon

"""
1samp：单样本 t 检验

用于检验一组样本是否与某个给定总体均值存在显著差异。

如果数据明显不满足正态性，可以考虑：
Wilcoxon signed-rank test
"""

scores = np.array([0.71, 0.74, 0.69, 0.76, 0.72])

# 假设总体值
popmean = 0.5


# ============================================================
# 1. Shapiro-Wilk 正态性检验
# ============================================================

shapiro_result = shapiro(scores)

print("=== Shapiro-Wilk 正态性检验 ===")
print(f"statistic: {shapiro_result.statistic:.4f}")
print(f"p-value: {shapiro_result.pvalue:.9f}")


# ============================================================
# 2. 根据正态性选择统计检验
# ============================================================

if shapiro_result.pvalue > 0.05:

    print("\n样本没有显著偏离正态分布")
    print("使用单样本 t 检验")

    result = ttest_1samp(
        scores,
        popmean=popmean,
        alternative="two-sided"
    )

    print("\n=== One-sample t-test ===")
    print(f"t-statistic: {result.statistic:.4f}")
    print(f"p-value: {result.pvalue:.9f}")

else:

    print("\n样本显著偏离正态分布")
    print("使用 Wilcoxon signed-rank test")

    # 将假设总体值转换为“差值为 0”的问题
    diff = scores - popmean

    result = wilcoxon(
        diff,
        alternative="two-sided"
    )

    print("\n=== Wilcoxon signed-rank test ===")
    print(f"W-statistic: {result.statistic:.4f}")
    print(f"p-value: {result.pvalue:.9f}")