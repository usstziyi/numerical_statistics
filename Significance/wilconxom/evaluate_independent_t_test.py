import numpy as np
from scipy.stats import ttest_ind, shapiro, mannwhitneyu, levene, t as t_dist

"""
evaluate 重构版：两组独立样本显著性检验

把"描述统计 → 参考性检查 → 主检验 → 效应量 → 稳健性参考"
封装成 evaluate 函数，主流程只负责准备数据与展示结果。

与配对版本不同：这里不按正态性切换主检验，
默认始终使用 Welch t 检验（不要求方差齐性，对正态性偏离较稳健），
正态性与方差齐性检验仅作参考，另附 Mann-Whitney U 作稳健性参考。
"""

ALPHA = 0.05


def evaluate_normality(data):

    """Shapiro-Wilk 正态性检验（仅供参考，不作为切换开关）"""

    result = shapiro(data)

    return {
        "statistic": result.statistic,
        "pvalue": result.pvalue,
    }


def evaluate_independent(a, b, alternative="two-sided", equal_var=False):

    """
    两组独立样本显著性检验：
    主检验为 Welch/Student t 检验（equal_var 控制），
    附均值差与 95% CI、Cohen's d，以及 Mann-Whitney U 稳健性参考。
    """

    n1, n2 = len(a), len(b)

    a, b = np.asarray(a), np.asarray(b)

    # ---- 参考性检查 ----
    normality = {
        name: evaluate_normality(data)
        for name, data in [("A", a), ("B", b)]
    }

    lev_stat, lev_p = levene(a, b)
    levene_result = {"statistic": lev_stat, "pvalue": lev_p}

    # ---- 主检验：Welch t 检验 ----
    result = ttest_ind(a, b, alternative=alternative, equal_var=equal_var)

    mean_diff = np.mean(a) - np.mean(b)
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
    se_diff = np.sqrt(var_a / n1 + var_b / n2)

    # Welch 自由度与 95% 置信区间
    df = (var_a / n1 + var_b / n2) ** 2 / (
        (var_a / n1) ** 2 / (n1 - 1) + (var_b / n2) ** 2 / (n2 - 1)
    )
    t_crit = t_dist.ppf(0.975, df)
    ci = (mean_diff - t_crit * se_diff, mean_diff + t_crit * se_diff)

    # Cohen's d（用合并标准差）
    pooled_sd = np.sqrt(((n1 - 1) * var_a + (n2 - 1) * var_b) / (n1 + n2 - 2))
    cohens_d = mean_diff / pooled_sd

    # ---- 非参数检验（稳健性参考）----
    u_result = mannwhitneyu(a, b, alternative=alternative)

    return {
        "sample_sizes": (n1, n2),
        "normality": normality,
        "levene": levene_result,
        "test_name": "Welch's independent t-test" if not equal_var
                     else "Student's independent t-test",
        "statistic": result.statistic,
        "pvalue": result.pvalue,
        "df": result.df,
        "mean_diff": mean_diff,
        "ci": ci,
        "cohens_d": cohens_d,
        "robustness": {
            "test_name": "Mann-Whitney U test",
            "statistic": u_result.statistic,
            "pvalue": u_result.pvalue,
        },
        "significant": result.pvalue < ALPHA,
    }


def print_report(report):

    """按统一格式输出检验报告"""

    n1, n2 = report["sample_sizes"]
    print(f"样本量: A = {n1}, B = {n2}")

    print("\n=== Shapiro-Wilk 正态性检验（仅供参考）===")
    for name, res in report["normality"].items():
        print(f"{name}: W = {res['statistic']:.4f}, p = {res['pvalue']:.3e}")

    if n1 < 10 or n2 < 10:
        print("警告：样本量过小，Shapiro-Wilk 结果不可靠，不建议据此选择检验方法。")

    print("\n=== Levene 方差齐性检验（仅供参考）===")
    print(f"Levene: statistic = {report['levene']['statistic']:.4f}, "
          f"p = {report['levene']['pvalue']:.3e}")

    print(f"\n=== {report['test_name']} ===")
    print(f"t-statistic: {report['statistic']:.4f}")
    print(f"p-value: {report['pvalue']:.4f}")
    print(f"df: {report['df']:.2f}")
    print(f"均值差 (A - B): {report['mean_diff']:.4f}")
    print(f"95% CI: [{report['ci'][0]:.4f}, {report['ci'][1]:.4f}]")
    print(f"Cohen's d: {report['cohens_d']:.4f}")

    if report["significant"]:
        print("结论：拒绝原假设，两组独立样本存在显著差异")
    else:
        print("结论：不能拒绝原假设，两组独立样本无显著差异")

    print(f"\n=== {report['robustness']['test_name']}（稳健性参考）===")
    print(f"U-statistic: {report['robustness']['statistic']:.4f}")
    print(f"p-value: {report['robustness']['pvalue']:.4f}")


if __name__ == "__main__":

    # 两组独立样本
    a = [0.71, 0.73, 0.69, 0.75, 0.72]
    b = [0.68, 0.70, 0.67, 0.72, 0.69]

    report = evaluate_independent(a, b)

    print_report(report)
