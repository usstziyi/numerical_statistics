import numpy as np
from scipy import stats

# ==========================
# 模拟数据
# 两个模型在同一批样本上的句子级评分（如 METEOR/COMET 分数）
# ==========================

rng = np.random.default_rng(42)
n = 100

scores_model_a = rng.normal(0.75, 0.05, n)
# 模型 B：基础水平与 A 相同，但对每个样本有独立的微小波动（配对数据）
scores_model_b = scores_model_a + rng.normal(0.01, 0.03, n)


# ==========================
# Example 1
# 配对 t 检验（ttest_rel）
# 适用：同一批样本上两个模型各得一个分数
# H0：两模型平均分之差为 0
# ==========================

result = stats.ttest_rel(scores_model_a, scores_model_b)

print("Example 1 配对 t 检验")
print(result)
print(f"t 统计量 = {result.statistic:.4f}")
print(f"p 值     = {result.pvalue:.4g}")
print(f"95% 置信区间 (A-B) = {result.confidence_interval(confidence_level=0.95)}")

if result.pvalue < 0.05:
    print("结论：p < 0.05，拒绝 H0，两模型平均分差异显著")
else:
    print("结论：p >= 0.05，不能拒绝 H0，没有足够证据说两模型平均分不同")


# ==========================
# Example 2
# 独立 t 检验（ttest_ind）
# 适用：两组不同样本上的分数（如不同测试集/不同系统）
# 默认 Student's t 检验（假设两总体方差相同）；
# equal_var=False 时为 Welch's t 检验（方差不等时更稳健，一般推荐）
# ==========================

scores_sys_x = rng.normal(0.72, 0.08, 80)
scores_sys_y = rng.normal(0.70, 0.08, 120)

result = stats.ttest_ind(scores_sys_x, scores_sys_y, equal_var=False)

print("\nExample 2 独立 t 检验（Welch）")
print(result)
print(f"t 统计量 = {result.statistic:.4f}")
print(f"p 值     = {result.pvalue:.4g}")

if result.pvalue < 0.05:
    print("结论：p < 0.05，两组平均分差异显著")
else:
    print("结论：p >= 0.05，不能拒绝 H0，两组平均分差异不显著")


# ==========================
# 注意事项
# 1. t 检验假设数据（配对情形下是"差值"）近似服从正态分布。
#    MT 评测的句子级分数往往偏态、有界，此时配对 Wilcoxon 符号秩检验
#    （evaluate.load("wilcoxon") 或 stats.wilcoxon）更稳健。
# 2. 比较两个 MT 系统时，惯例做法还有 paired bootstrap resampling。
# 3. p 值只回答"差异是否显著"，效应大小请看均值差与置信区间。
# ==========================
