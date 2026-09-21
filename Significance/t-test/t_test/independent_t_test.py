from scipy.stats import ttest_ind,shapiro

"""
independent t-test：独立样本 t 检验
用于比较两组相互独立样本的均值是否存在显著差异
H₀: 两组均值相等；p < 0.05 时拒绝 H₀
"""

# 两组独立样本（如两组不同受试者的测量值），样本间无配对关系
a = [0.71, 0.73, 0.69, 0.75, 0.72]
b = [0.68, 0.70, 0.67, 0.72, 0.69]

# 检查样本是否符合正态分布
print(shapiro(a))
print(shapiro(b))
if shapiro(a).pvalue > 0.05 and shapiro(b).pvalue > 0.05:
    print("样本符合正态分布")
else:
    print("样本不符合正态分布")



result = ttest_ind(
    a,
    b,
    equal_var=False  # True: Student's t 检验（假设两组总体方差相等）；False: Welch's t 检验（更稳健，推荐）
)

print(f"t-statistic: {result.statistic:.4f}") # t 统计量：两组均值差 / 合并标准误
print(f"p-value: {result.pvalue:.9f}") # p 值：双侧，越小表示两组均值差异越显著
