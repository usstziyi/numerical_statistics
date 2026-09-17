import numpy as np
from scipy.stats import rankdata, norm

def wilcoxon_rank_sum_manual(a, b):
    """手动实现 Wilcoxon 秩和检验（Mann-Whitney U）"""
    n1, n2 = len(a), len(b)
    combined = np.concatenate([a, b])
    ranks = rankdata(combined)
    R1 = ranks[:n1].sum()
    U1 = R1 - n1 * (n1 + 1) / 2
    U2 = n1 * n2 - U1
    U = min(U1, U2)
    # 正态近似
    mu = n1 * n2 / 2
    sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (U - mu) / sigma
    p = 2 * (1 - norm.cdf(abs(z)))
    return U, p

def wilcoxon_signed_rank_manual(before, after):
    """手动实现 Wilcoxon 符号秩检验"""
    diff = before - after
    diff = diff[diff != 0]
    abs_diff = np.abs(diff)
    ranks = rankdata(abs_diff)
    W_plus = ranks[diff > 0].sum()
    W_minus = ranks[diff < 0].sum()
    W = min(W_plus, W_minus)
    n = len(diff)
    mu = n * (n + 1) / 4
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (W - mu) / sigma
    p = 2 * (1 - norm.cdf(abs(z)))
    return W, p

# 测试
np.random.seed(42)

# 秩和检验：两组独立样本
group_a = np.random.normal(loc=50, scale=10, size=100)
group_b = np.random.normal(loc=58, scale=10, size=100)

# 符号秩检验：配对前后样本
before = np.random.normal(loc=50, scale=10, size=100)
after = before + np.random.normal(loc=3, scale=5, size=100)  # 处理后均值略有提升

U, p1 = wilcoxon_rank_sum_manual(group_a, group_b)
W, p2 = wilcoxon_signed_rank_manual(before, after)
print(f"手动秩和检验: U={U:.4f}, p={p1:.4f}")
print(f"手动符号秩检验: W={W:.4f}, p={p2:.4f}")