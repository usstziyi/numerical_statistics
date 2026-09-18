import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体，防止中文显示为方块
# Windows 用户一般用 SimHei，Mac 用户一般用 Arial Unicode MS 或 PingFang SC
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'PingFang SC']
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# ==========================================
# 第一部分：数据准备与 z 分数计算
# ==========================================
np.random.seed(42)  # 固定随机种子，保证每次运行结果一致

# 模拟两组数据：语文成绩 和 数学成绩
# 语文：平均 70，标准差 5
# 数学：平均 90，标准差 15
chinese_scores = np.random.normal(loc=70, scale=5, size=200)
math_scores = np.random.normal(loc=90, scale=15, size=200)

# 手动实现 z 分数公式：z = (x - mu) / sigma
def calculate_z_scores(data):
    mu = np.mean(data)
    sigma = np.std(data)
    return (data - mu) / sigma

chinese_z = calculate_z_scores(chinese_scores)
math_z = calculate_z_scores(math_scores)

# 模拟一个学生：小明
xiaoming_chinese = 82
xiaoming_math = 95

# 计算小明的 z 分数
xiaoming_chinese_z = (xiaoming_chinese - np.mean(chinese_scores)) / np.std(chinese_scores)
xiaoming_math_z = (xiaoming_math - np.mean(math_scores)) / np.std(math_scores)

print("=== 小明的成绩分析 ===")
print(f"语文: 原始分 {xiaoming_chinese}, z分数 {xiaoming_chinese_z:.2f}")
print(f"数学: 原始分 {xiaoming_math}, z分数 {xiaoming_math_z:.2f}")
if xiaoming_chinese_z > xiaoming_math_z:
    print("结论: 小明的语文相对表现更好！")
else:
    print("结论: 小明的数学相对表现更好！")

# ==========================================
# 第二部分：Matplotlib 可视化
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('标准化比较 (z分数) Demo', fontsize=16)

# 1. 语文原始成绩分布
axes[0, 0].hist(chinese_scores, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 0].axvline(np.mean(chinese_scores), color='red', linestyle='dashed', linewidth=2, label=f'均值 μ={np.mean(chinese_scores):.1f}')
axes[0, 0].axvline(xiaoming_chinese, color='green', linestyle='solid', linewidth=2, label=f'小明 {xiaoming_chinese}')
axes[0, 0].set_title('语文原始成绩分布')
axes[0, 0].set_xlabel('分数')
axes[0, 0].set_ylabel('人数')
axes[0, 0].legend()

# 2. 数学原始成绩分布
axes[0, 1].hist(math_scores, bins=20, color='lightcoral', edgecolor='black', alpha=0.7)
axes[0, 1].axvline(np.mean(math_scores), color='red', linestyle='dashed', linewidth=2, label=f'均值 μ={np.mean(math_scores):.1f}')
axes[0, 1].axvline(xiaoming_math, color='green', linestyle='solid', linewidth=2, label=f'小明 {xiaoming_math}')
axes[0, 1].set_title('数学原始成绩分布')
axes[0, 1].set_xlabel('分数')
axes[0, 1].set_ylabel('人数')
axes[0, 1].legend()

# 3. 语文 z 分数分布
axes[1, 0].hist(chinese_z, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
axes[1, 0].axvline(0, color='red', linestyle='dashed', linewidth=2, label='均值 μ=0')
axes[1, 0].axvline(xiaoming_chinese_z, color='green', linestyle='solid', linewidth=2, label=f'小明 z={xiaoming_chinese_z:.2f}')
axes[1, 0].set_title('语文 z 分数分布 (标准化后)')
axes[1, 0].set_xlabel('z 分数')
axes[1, 0].set_ylabel('人数')
axes[1, 0].legend()

# 4. 数学 z 分数分布
axes[1, 1].hist(math_z, bins=20, color='lightcoral', edgecolor='black', alpha=0.7)
axes[1, 1].axvline(0, color='red', linestyle='dashed', linewidth=2, label='均值 μ=0')
axes[1, 1].axvline(xiaoming_math_z, color='green', linestyle='solid', linewidth=2, label=f'小明 z={xiaoming_math_z:.2f}')
axes[1, 1].set_title('数学 z 分数分布 (标准化后)')
axes[1, 1].set_xlabel('z 分数')
axes[1, 1].set_ylabel('人数')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

# ==========================================
# 第三部分：异常值检测演示 (额外的绘图)
# ==========================================
# 生成一组包含异常值的数据
data_with_outliers = np.random.normal(loc=50, scale=10, size=100)
data_with_outliers = np.append(data_with_outliers, [100, 5, 95]) # 加入几个极端的异常值

z_scores_outliers = calculate_z_scores(data_with_outliers)

plt.figure(figsize=(10, 5))
plt.scatter(range(len(data_with_outliers)), data_with_outliers, c='blue', label='正常数据')
# 找出 z 分数绝对值大于 3 的点
outlier_indices = np.where(np.abs(z_scores_outliers) > 3)[0]
plt.scatter(outlier_indices, data_with_outliers[outlier_indices], c='red', s=100, label='异常值 (|z| > 3)')
plt.axhline(np.mean(data_with_outliers), color='green', linestyle='--', label='均值')
plt.title('利用 z 分数检测异常值 (离群点)')
plt.xlabel('数据索引')
plt.ylabel('数值')
plt.legend()
plt.show()