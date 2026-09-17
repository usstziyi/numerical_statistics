import numpy as np
import os
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt

# 模拟配对样本：同一批受试者治疗前后
np.random.seed(42)

# 治疗前评分
before = np.random.normal(loc=70, scale=8, size=100)

# 治疗后评分：整体下降约 5 分
after = before - np.random.normal(loc=5, scale=4, size=100)

# 绘制治疗前后评分分布对比
# density=True 使纵轴为密度，配对样本量相同也可公平比较
plt.hist(before, bins=10, density=True, alpha=0.6, label='Before Treatment (loc=70)')
plt.hist(after, bins=10, density=True, alpha=0.6, label='After Treatment (loc=65)')
plt.xlabel('Score')
plt.ylabel('Density')
plt.title('Distribution of Before and After Treatment (n=100)')
plt.legend()

# 保存图片到脚本同目录
save_path = "output/wilcoxon_before_after.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"图已保存至: {save_path}")
# plt.show()



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