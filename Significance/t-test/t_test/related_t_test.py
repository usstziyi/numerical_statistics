import numpy as np
from scipy.stats import ttest_rel, shapiro, wilcoxon

"""
related t-test：相关样本 t 检验 / 配对样本 t 检验
Wilcoxon signed-rank test：Wilcoxon 符号秩检验

假设同样 5 个 EEG 被试分别测试模型 A 和模型 B：
"""

model_a = np.array([0.71, 0.73, 0.69, 0.75, 0.72])
model_b = np.array([0.68, 0.70, 0.67, 0.72, 0.69])


# ============================================================
# 1. 检查“配对差值”的正态性
# ============================================================

diff = model_a - model_b

shapiro_result = shapiro(diff)

print("=== Shapiro-Wilk 正态性检验 ===")
print(f"statistic: {shapiro_result.statistic:.4f}")
print(f"p-value: {shapiro_result.pvalue:.9f}")


# ============================================================
# 2. 根据正态性选择检验
# ============================================================

if shapiro_result.pvalue > 0.05:

    print("\n差值没有显著偏离正态分布")
    print("使用配对样本 t 检验")

    result = ttest_rel(model_a, model_b)

    print("\n=== Paired t-test ===")
    print(f"t-statistic: {result.statistic:.9f}")
    print(f"p-value: {result.pvalue:.9f}")

else:

    print("\n差值显著偏离正态分布")
    print("使用 Wilcoxon signed-rank test")

    result = wilcoxon(model_a, model_b)

    print("\n=== Wilcoxon signed-rank test ===")
    print(f"statistic: {result.statistic:.9f}")
    print(f"p-value: {result.pvalue:.9f}")