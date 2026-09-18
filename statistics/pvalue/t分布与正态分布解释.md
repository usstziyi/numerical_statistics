# t分布与正态分布解释

> **Workspace:** d:\AI\CodeWork\numerical_statistics

---

**User**

给我解释一下什么是t分布，什么是正态分布

**Assistant**

我先看了你工程里的两个文件：[normal.py](file:///d:/AI/CodeWork/numerical_statistics/statistics/normal_distribution/normal.py) 和 [students.py](file:///d:/AI/CodeWork/numerical_statistics/statistics/pvalue/students.py)，下面结合它们讲。

## 一、正态分布（Normal Distribution）

`stats.norm.pdf(x, loc=mu, scale=sigma)` 里画的就是它，记作 X ~ N(μ, σ²)。

密度函数：

```
f(x) = 1/(σ·√(2π)) · exp( -(x-μ)² / (2σ²) )
```

三个关键性质：

1. **两个参数决定一切**：μ 控制位置（左右平移），σ 控制胖瘦（越大越矮越宽）。这正是 [normal_1.py](file:///d:/AI/CodeWork/numerical_statistics/statistics/normal_distribution/normal_1.py#L14-L31) 左右两张子图分别在演示的：左图固定 σ 改 μ，右图固定 μ 改 σ。
2. **完全对称、钟形**，均值 = 中位数 = 众数 = μ，曲线下面积恒为 1。
3. **标准化**：任何正态都能平移压缩成标准正态 Z = (X-μ)/σ ~ N(0,1)，也就是 [normal.py](file:///d:/AI/CodeWork/numerical_statistics/statistics/normal_distribution/normal.py#L9) 画的那条。常用的 68%–95%–99.7% 就是它的 ±1σ、±2σ、±3σ 面积。

它为什么重要？**中心极限定理**：不管原始数据什么分布，只要样本量够大，样本均值 x̄ 就近似服从正态。所以它是整个推断统计的地基。

关键点：正态里的 μ 和 σ 被当作**已知的真实值**，不依赖你有多少样本。

## 二、t 分布（Student's t Distribution）

它不是一个"新数据"的分布，而是**统计量的分布**——具体说，是"σ 未知、用样本标准差 s 代替"之后，那个统计量的分布。

构造方式：

```
若 Z ~ N(0,1)，V ~ χ²(ν) 且相互独立，
则 T = Z / √(V/ν) ~ t(ν)
```

`ν` 叫**自由度**（degrees of freedom），单样本均值检验里 ν = n-1。

实际用的形式就是你 [students.py](file:///d:/AI/CodeWork/numerical_statistics/statistics/pvalue/students.py#L15-L19) 里 `ttest_1samp` 内部算的那个：

```
t = (x̄ - μ0) / (s / √n)      df = n - 1
```

**为什么要有它？** 如果 σ 已知，上面的分母就是真实的 σ/√n，x̄ 严格服从正态，用 Z 检验即可。但现实中 σ 也是从样本里估的，**s 本身带随机误差**，这层额外的不确定性让统计量变得更分散——于是尾巴变厚、山峰变矮。这正是 t 分布的形态来源。

特性：

- 同样关于 0 对称、钟形，但**尾部比标准正态厚**；
- 自由度越小，尾巴越厚（样本越少越"不确定"）；
- **ν → ∞ 时 t 分布收敛到 N(0,1)**。经验上 ν > 30 左右两者已几乎重合；
- 因此 t 的临界值总是比 Z 大，例如 95% 单侧：Z = 1.645，而 t(10) = 1.812 → **置信区间更宽、结论更保守**。

## 三、对比与选用

| | 正态分布 N(μ,σ²) | t 分布 t(ν) |
|---|---|---|
| 描述对象 | 数据 X 本身 | 统计量（如 x̄ 的标准化形式） |
| σ | 已知（真值） | 未知，用样本 s 估计 |
| 参数 | μ, σ | 自由度 ν = n-1 |
| 尾部 | 标准厚度 | 更厚，ν 越小时越厚 |
| 使用场景 | 大样本、σ 已知、描述连续变量 | 小样本均值检验/置信区间、回归系数检验 |
| 关系 | ν→∞ 时的极限 | 是正态在"σ 未知"下的修正版 |

一句话：**正态是"知道全部参数"的理想世界，t 是"方差也得估"的现实世界**。样本量小时必须用 t；样本量大时两者结果基本没差别，这也是为什么很多教材说"n > 30 可以用 Z 近似 t"。

你 [students.py](file:///d:/AI/CodeWork/numerical_statistics/statistics/pvalue/students.py) 那个例子只有 10 个样本（df = 9），所以必须走 t 分布——如果用 Z 计算，p 值会被人为压小，容易把"抽样运气"误判成"真实差异"。

需要的话，我可以在 [normal_1.py](file:///d:/AI/CodeWork/numerical_statistics/statistics/normal_distribution/normal_1.py) 的基础上加一张 `stats.t.pdf` 与标准正态的叠合图，直观看到尾部厚度差异。

---

