import numpy as np
from scipy.stats import ttest_1samp, shapiro, wilcoxon

"""
evaluate 重构版：单样本显著性检验

把"正态性检验 → 选择检验 → 执行 → 汇总结论"封装成 evaluate 函数，
主流程只负责准备数据与展示结果。

选择逻辑：
- 数据近似正态      → 单样本 t 检验
- 数据明显偏离正态  → Wilcoxon signed-rank test
"""

ALPHA = 0.05


def evaluate_normality(data):

    """Shapiro-Wilk 正态性检验"""

    result = shapiro(data)

    return {
        "is_normal": result.pvalue > ALPHA,
        "statistic": result.statistic,
        "pvalue": result.pvalue,
    }


def evaluate_1_sample(data, popmean, alternative="two-sided"):

    """
    单样本显著性检验：
    先检验正态性，再据此选择参数检验（t 检验）
    或非参数检验（Wilcoxon signed-rank），返回汇总字典。
    """

    normality = evaluate_normality(data)

    if normality["is_normal"]:

        test_name = "One-sample t-test"
        statistic_name = "t-statistic"

        result = ttest_1samp(
            data,
            popmean=popmean,
            alternative=alternative
        )

    else:

        test_name = "Wilcoxon signed-rank test"
        statistic_name = "W-statistic"

        # 将假设总体值转换为"差值为 0"的问题
        diff = data - popmean

        result = wilcoxon(
            diff,
            alternative=alternative
        )

    return {
        "normality": normality,
        "test_name": test_name,
        "statistic_name": statistic_name,
        "statistic": result.statistic,
        "pvalue": result.pvalue,
        "significant": result.pvalue < ALPHA,
    }


def print_report(report):

    """按统一格式输出检验报告"""

    normality = report["normality"]

    print("=== Shapiro-Wilk 正态性检验 ===")
    print(f"statistic: {normality['statistic']:.4f}")
    print(f"p-value: {normality['pvalue']:.9f}")

    if normality["is_normal"]:
        print("样本没有显著偏离正态分布")
    else:
        print("样本显著偏离正态分布")

    print(f"\n=== {report['test_name']} ===")
    print(f"{report['statistic_name']}: {report['statistic']:.4f}")
    print(f"p-value: {report['pvalue']:.9f}")

    if report["significant"]:
        print("结论：拒绝原假设，样本与总体均值存在显著差异")
    else:
        print("结论：不能拒绝原假设，样本与总体均值无显著差异")


if __name__ == "__main__":

    scores = np.array([0.71, 0.74, 0.69, 0.76, 0.72])

    # 假设总体值
    popmean = 0.5

    report = evaluate_1_sample(scores, popmean)

    print_report(report)

    print(report)
