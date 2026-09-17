import numpy as np
from scipy.stats import wilcoxon

# 模拟配对样本：同一批受试者治疗前后
np.random.seed(42)

# 治疗前评分
before = np.random.normal(loc=70, scale=8, size=25)

# 治疗后评分：整体下降约 5 分
after = before - np.random.normal(loc=5, scale=4, size=25)

# 进行 Wilcoxon 符号秩检验
# alternative 可选 'two-sided'、'less'、'greater'
stat, p_value = wilcoxon(before, after, alternative='two-sided')

print("=== Wilcoxon 符号秩检验（配对样本） ===")
print(f"治疗前中位数: {np.median(before):.2f}")
print(f"治疗后中位数: {np.median(after):.2f}")
print(f"差值中位数: {np.median(before - after):.2f}")
print(f"W 统计量: {stat:.4f}")
print(f"p 值: {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print("结论：拒绝原假设，治疗前后存在显著差异。")
else:
    print("结论：不能拒绝原假设，治疗前后无显著差异。")