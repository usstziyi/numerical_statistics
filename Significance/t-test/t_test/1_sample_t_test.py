from scipy.stats import ttest_1samp,shapiro

"""
1samp: 单样本 t 检验
"""

scores = [0.71, 0.74, 0.69, 0.76, 0.72]

# 检查样本是否符合正态分布
print(shapiro(scores))
if shapiro(scores).pvalue > 0.05:
    print("样本符合正态分布")
else:
    print("样本不符合正态分布")




result = ttest_1samp(
    scores,
    popmean=0.5,
    alternative="two-sided"
)

print(f"t-statistic: {result.statistic:.4f}") # t 统计量
print(f"p-value: {result.pvalue:.9f}") # p 值