import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns

# 模拟两组独立样本
np.random.seed(42)

# 组 A：来自均值为 50 的分布
group_a = np.random.normal(loc=50, scale=10, size=100)

# 组 B：来自均值为 58 的分布（存在差异）
group_b = np.random.normal(loc=58, scale=10, size=100)

# 使用 seaborn 绘制两组分布对比（直方图 + KDE 平滑密度曲线）
sns.set_theme()

# 合并为长表格式 DataFrame，供 seaborn 按 hue 分组
df = pd.DataFrame({
    'value': np.concatenate([group_a, group_b]),
    'group': ['group_a'] * len(group_a) + ['group_b'] * len(group_b),
})

sns.histplot(data=df, 
             x='value', 
             hue='group',
             element='step',      # 描边式直方图，重叠更清晰
             kde=True,            # 叠加 KDE 密度曲线
             stat='density',      # 与 density=True 等价
             common_norm=False,   # 两组各自归一化，样本量不同也公平
             alpha=0.6)
plt.xlabel('value')
plt.ylabel('density')
plt.title('Distribution of group_a and group_b (loc=50, loc=58)') 
plt.savefig('output/mannwhitneyu_distribution.png', dpi=300, bbox_inches='tight')
# plt.show()




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