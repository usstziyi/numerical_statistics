import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================
# 第 1 步：准备数据（全班男同学身高，单位 cm）
# ============================================================
# 实际中这是你测量得到的真实数据；这里模拟 40 个男生身高
np.random.seed(42)
heights = np.random.normal(loc=173, scale=6, size=40).round(1)

print("身高数据：")
print(heights)
print(f"\n样本量 n = {len(heights)}")
print(f"样本均值 x̄ = {heights.mean():.2f} cm")
print(f"样本标准差 s = {heights.std(ddof=1):.2f} cm")

# ============================================================
# 第 2 步：画直方图，看看数据形状
# ============================================================
plt.figure(figsize=(9, 5))

# density=True：让直方图的面积归一化为 1，才能和概率密度曲线对比
plt.hist(heights, bins=8, density=True, color='skyblue',
         edgecolor='white', alpha=0.8, label='actual height histogram')


# ============================================================
# 第 3 步：用样本均值和标准差拟合正态分布
# ============================================================
mu = heights.mean()          # 用样本均值估计 μ
sigma = heights.std(ddof=1)  # 用样本标准差估计 σ

# ============================================================
# 第 4 步：画理论正态曲线
# ============================================================
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 300)
y = stats.norm.pdf(x, mu, sigma)   # 正态分布概率密度Probability Density Function

plt.plot(x, y, 'r-', linewidth=2,
         label=f'fit normal normal distribution N(μ={mu:.1f}, σ={sigma:.1f})')

# 标注均值和 ±1σ、±2σ 区间
plt.axvline(mu, color='black', linestyle='--', alpha=0.6)
plt.axvline(mu - sigma, color='gray', linestyle=':', alpha=0.6)
plt.axvline(mu + sigma, color='gray', linestyle=':', alpha=0.6)

plt.title('all students height distribution and normal fit')
plt.xlabel('height (cm)')
plt.ylabel('density')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()



# ============================================================
# 第 5 步：用拟合结果回答实际问题
# ============================================================
print("\n--- 用拟合的正态分布做推断 ---")

# (1) 身高在 170~180 之间的比例
# CDF = Cumulative Distribution Function， 累积分布函数
p_170_180 = stats.norm.cdf(180, mu, sigma) - stats.norm.cdf(170, mu, sigma)
print(f"P(170 < 身高 < 180) = {p_170_180:.2%}")

# (2) 身高超过 185 的比例
p_above_185 = 1 - stats.norm.cdf(185, mu, sigma)
print(f"P(身高 > 185) = {p_above_185:.2%}")

# (3) 68% / 95% 区间
print(f"μ ± 1σ = [{mu-sigma:.1f}, {mu+sigma:.1f}] cm ≈ 68.3%")
print(f"μ ± 2σ = [{mu-2*sigma:.1f}, {mu+2*sigma:.1f}] cm ≈ 95.4%")