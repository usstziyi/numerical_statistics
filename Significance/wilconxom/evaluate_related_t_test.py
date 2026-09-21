import numpy as np
from scipy.stats import ttest_rel, shapiro, wilcoxon

"""
evaluate 重构版：相关样本（配对）显著性检验

把"配对差值正态性检验 → 选择检验 → 执行 → 汇总结论"
封装成 evaluate 函数，主流程只负责准备数据与展示结果。

选择逻辑：
- 差值近似正态      → 配对样本 t 检验
- 差值明显偏离正态  → Wilcoxon signed-rank test

场景：同样 5 个 EEG 被试分别测试模型 A 和模型 B
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


def evaluate_related(data_a, data_b, alternative="two-sided"):

    """
    相关样本（配对）显著性检验：
    先检验配对差值的正态性，再据此选择
    配对 t 检验或 Wilcoxon signed-rank，返回汇总字典。
    """

    # 检查"配对差值"的正态性
    diff = data_a - data_b
    normality = evaluate_normality(diff)

    if normality["is_normal"]:

        test_name = "Paired t-test"
        statistic_name = "t-statistic"

        result = ttest_rel(
            data_a,
            data_b,
            alternative=alternative
        )

    else:

        test_name = "Wilcoxon signed-rank test"
        statistic_name = "W-statistic"

        result = wilcoxon(
            data_a,
            data_b,
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

    print("=== Shapiro-Wilk 正态性检验（配对差值）===")
    print(f"statistic: {normality['statistic']:.4f}")
    print(f"p-value: {normality['pvalue']:.9f}")

    if normality["is_normal"]:
        print("差值没有显著偏离正态分布")
    else:
        print("差值显著偏离正态分布")

    print(f"\n=== {report['test_name']} ===")
    print(f"{report['statistic_name']}: {report['statistic']:.9f}")
    print(f"p-value: {report['pvalue']:.9f}")

    if report["significant"]:
        print("结论：拒绝原假设，两组配对数据存在显著差异")
    else:
        print("结论：不能拒绝原假设，两组配对数据无显著差异")


if __name__ == "__main__":

    model_a = np.array([0.71, 0.73, 0.69, 0.75, 0.72])
    model_b = np.array([0.68, 0.70, 0.67, 0.72, 0.69])

    report = evaluate_related(model_a, model_b)

    print_report(report)
