import numpy as np
from scipy import stats

# 场景：某班学生考试，检验班级平均分是否等于全校平均分 70
# H0（原假设）：班级平均分 = 70
# H1（备择假设）：班级平均分 ≠ 70

scores = [72, 68, 75, 71, 69, 74, 70, 73, 67, 71]  # 全班成绩
mu0 = 70        # H0 下的总体均值（全校平均分）
alpha = 0.05    # 显著性水平

# 这 1 分的差距，是 班级水平真的不一样 ，还是 抽样运气 造成的正常波动？

# 单样本 t 检验
t_stat, p_value = stats.ttest_1samp(
    a=scores,  # 待检验的样本数据（班级学生考试成绩）
    popmean=mu0,  # 原假设H0设定的总体均值（全校平均分数）
    alternative='two-sided'  # 两尾检验（默认）
)

print(f"班级平均分 = {np.mean(scores):.2f}")
print(f"t 统计量 = {t_stat:.4f}")
print(f"p 值 = {p_value:.4f}")

if p_value < alpha:
    print(f"p < {alpha} → 拒绝 H0，班级平均分与全校平均分有显著差异")
else:
    print(f"p >= {alpha} → 不拒绝 H0，没有足够证据说班级平均分不等于全校平均分")