import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt

# 模拟两组独立样本
np.random.seed(42)

# 组 A：来自均值为 50 的分布
group_a = np.random.normal(loc=50, scale=10, size=20)

# 组 B：来自均值为 58 的分布（存在差异）
group_b = np.random.normal(loc=58, scale=10, size=22)

# 将两组数据叠加绘制在同一张图上，便于直接比较分布
# density=True 使纵轴为密度，样本量不同也可公平比较
plt.hist(group_a, bins=10, density=True, alpha=0.6, label='group_a (n=20)')
plt.hist(group_b, bins=10, density=True, alpha=0.6, label='group_b (n=22)')
plt.xlabel('value')
plt.ylabel('density')
plt.title('Distribution of group_a and group_b (n=20, n=22)')
plt.legend()
plt.show()




# 进行 Wilcoxon 秩和检验（Mann-Whitney U）
# alternative 可选 'two-sided'、'less'、'greater'
stat, p_value = mannwhitneyu(group_a, group_b, alternative='two-sided')

print("=== Wilcoxon 秩和检验（独立样本） ===")
print(f"组 A 中位数: {np.median(group_a):.2f}")
print(f"组 B 中位数: {np.median(group_b):.2f}")
print(f"U 统计量: {stat:.4f}")
print(f"p 值: {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print("结论：拒绝原假设，两组分布存在显著差异。")
else:
    print("结论：不能拒绝原假设，两组分布无显著差异。")