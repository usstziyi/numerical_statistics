import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# 中文标签需要指定字体，否则会显示成方块
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 第 1 步：全班 40 位同学的成绩
# 现实中这是考完试拿到手的原始分数，这里用 N(70, 10) 模拟，
# 固定随机种子 42，保证每次跑出来的成绩都一样
# ============================================================
np.random.seed(42)
scores = np.clip(np.round(np.random.normal(loc=70, scale=10, size=40)), 0, 100).astype(int)

# 小红是 1 号同学，这次考了 60 分
scores[0] = 60

n = len(scores)
mu = 70       # 理论均值：整个年级的平均分
sigma = 10    # 理论标准差
x = 60        # 小红的成绩

print(f"全班成绩：{scores.tolist()}")
print(f"样本量 n = {n}，样本均值 x̄ = {scores.mean():.2f}，"
      f"样本标准差 s = {scores.std(ddof=1):.2f}")

# ============================================================
# 第 2 步：同一个 60 分，换三种参照系各算一遍
# 参照系不同答案就不同，这不是谁算错了，而是回答的问题不同
# ============================================================
rest = scores[1:]     # 剔除小红自己，避免用她自己的分数给她定标尺

refs = [
    # (口径, 均值, 标准差, df, 说明)
    # df = None 表示用已知参数，走标准正态（真 z-score 的定义）
    # df = n-1 是自由度，跟着"算出分母 s 的那批数据"走，而非小红
    ("A 理论", mu, sigma, None, "整个年级，μ、σ 已知"),
    ("B 样本", scores.mean(), scores.std(ddof=1), n - 1, "本班 40 人，含小红自己"),
    ("C 留一", rest.mean(), rest.std(ddof=1), len(rest) - 1, "本班 39 人，剔除小红自己"),
]

print("z = (x - 参照均值) / 参照标准差，P 由累积分布函数算出")
results = []
for name, m, s, df, desc in refs:
    z_ref = (x - m) / s
    if df is None:
        # 参数已知时 (x - μ)/σ 严格服从标准正态，这是 z-score 的定义
        # 求的是 标准正态分布中 z_ref 左侧的累积概率 ，也就是曲线下、`z_ref` 左边的面积
        p_ref = stats.norm.cdf(z_ref) # p_ref = P(Z <= z_ref)
        how = "norm.cdf"
    else:
        # 用样本的 x̄、s 代替真参数时，(x - x̄)/s 服从 t 分布而非正态
        p_ref = stats.t.cdf(z_ref, df) # p_ref = P(T <= z_ref)
        how = f"t.cdf(df={df})"
    results.append((name, m, s, z_ref, p_ref, desc))
    print(f"{name}  μ={m:6.2f}, σ={s:5.2f}  z={z_ref:+.3f}  "
          f"低于小红 {p_ref:6.2%}  高于小红 {1 - p_ref:6.2%}  "
          f"{how:<13} # {desc}")

# 画图仍以理论口径为准
z = results[0][3]
p_lower = results[0][4] # p_ref = P(Z <= z_ref)

# ============================================================
# 第 3 步：拿全班 40 人的真实成绩验证，看哪套参照系更贴合
# ============================================================
n_below = (scores < x).sum()
n_above = (scores > x).sum()
n_equal = (scores == x).sum() - 1     # 减去小红自己
obs = n_below / n

print(f"\n实际观测：{n_below}/{n} = {obs:.2%} 低于小红")
for name, m, s, z_ref, p_ref, desc in results:
    print(f"  {name} 预测 {p_ref:7.2%}，比实测 {'高' if p_ref > obs else '低'} {abs(p_ref - obs):.2%}")

# 把理论口径与实测之间的差距拆成两部分
p_theory, p_sample = results[0][4], results[1][4]
print(f"\n偏差分解（理论 {p_theory:.2%} vs 实测 {obs:.2%}，共差 {obs - p_theory:+.2%}）：")
print(f"  ① 参照系不匹配（μ、σ 换成 x̄、s）：{p_sample - p_theory:+.2%}")
print(f"  ② 小样本抽样波动（预测 vs 实测）：{obs - p_sample:+.2%}")
print("  ②远小于①，说明这个 demo 里的偏差主要来自参照系，不是随机运气")

# 名次 = 比她高的人数 + 1
print(f"\n小红班内排名: 第 {n_above + 1} 名 / 共 {n} 人")
if n_equal > 0:
    print(f"与小红同分: {n_equal} 人")   # 同分的人和小红名次相同

# 样本只有 40 个人，频率和理论概率有偏差是正常的
print(f"\n样本均值 {scores.mean():.2f} vs 理论 μ = {mu}"
      f"（差 {scores.mean() - mu:+.2f}）")
print(f"样本标准差 {scores.std(ddof=1):.2f} vs 理论 σ = {sigma}"
      f"（差 {scores.std(ddof=1) - sigma:+.2f}）")

# df = 39 时 t 与正态已经很接近，样本越小差距越明显（见 normal_6.py）
print(f"\n注：df={n - 1} 时 t 与正态差别很小，理论口径下 "
      f"norm.cdf = {stats.norm.cdf(z):.2%}，t.cdf = {stats.t.cdf(z, n - 1):.2%}")

# ============================================================
# 第 4 步：画出来看看
# ============================================================
plt.figure(figsize=(9, 5))

# density=True：让直方图面积归一化为 1，才能和概率密度曲线叠加对比
plt.hist(scores, bins=8, density=True, color='skyblue', edgecolor='white',
         alpha=0.8, label=f'全班 {n} 人成绩')

xs = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
plt.plot(xs, stats.norm.pdf(xs, mu, sigma), 'r-', linewidth=2,
         label=f'理论分布 N(μ={mu}, σ={sigma})')

# 用全班数据拟合的曲线（口径 B），它比理论曲线更偏左更矮，
# 两条曲线的错位就是"参照系不匹配"在图上长什么样
s_mean, s_std = scores.mean(), scores.std(ddof=1) # 贝塞尔校正
plt.plot(xs, stats.norm.pdf(xs, s_mean, s_std), color='gray', linestyle='--',
         linewidth=1.5, label=f'本班拟合 N(均值={s_mean:.1f}, s={s_std:.1f})')

plt.axvline(mu, color='black', linestyle='--', alpha=0.6, label=f'μ = {mu}')
plt.axvline(s_mean, color='gray', linestyle=':', alpha=0.8,
            label=f'本班均值 = {s_mean:.1f}')
plt.axvline(x, color='green', linewidth=2, label=f'小红 {x} 分 (z = {z:.2f})')

# 把小红左侧的面积涂出来，直观对应 p_lower（这里用理论口径）
plt.fill_between(xs, stats.norm.pdf(xs, mu, sigma), where=(xs <= x),
                 color='green', alpha=0.15)
plt.annotate(f'理论 {p_lower:.2%}', xy=(x, 0.01), xytext=(mu - 3.3 * sigma, 0.06),
             arrowprops=dict(arrowstyle='->'), color='green')

plt.title(f'小红（{x} 分）在全班成绩中的位置')
plt.xlabel('成绩')
plt.ylabel('概率密度')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
