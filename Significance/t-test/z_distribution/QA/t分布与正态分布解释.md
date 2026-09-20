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

**User**

好的，加一张叠合图

**Assistant**

文件里已经有 `stats` 导入，我在 [normal_1.py](file:///d:/AI/CodeWork/numerical_statistics/statistics/normal_distribution/normal_1.py) 末尾追加一张新图。标签沿用该文件的英文风格，避免中文在 matplotlib 里显示成方块。

已追加到 [normal_1.py](file:///d:/AI/CodeWork/numerical_statistics/statistics/normal_distribution/normal_1.py#L36-L72)，运行通过（exit 0，只有 Agg 后端的 show 警告）。

图分两栏：

- **左（全貌）**：`N(0,1)` 黑色虚线 + df = 1/3/10/30 四条 t 曲线。能看到 t 的峰顶明显低于 0.3989，且 df 越小越矮越宽，df=30 已经几乎贴上正态。
- **右（右尾放大 + 对数纵轴）**：这是"厚尾"真正的证据。线性坐标下尾部差异会被压平到看不见，取 log 后差距一目了然。

用你的 venv 实算了几个点，右尾的密度比：

| x | N(0,1) | t(30) | t(10) | t(3) | t(1) |
|---|---|---|---|---|---|
| 3 | 4.43e-3 | 6.78e-3 | 1.14e-2 | 2.30e-2 | 3.18e-2 |
| 4 | 1.34e-4 | 5.25e-4 | 2.03e-3 | 9.16e-3 | 1.87e-2 |

x = 4 处，t(1) 的尾部密度是正态的 **140 倍左右**。对应到实操：同样落在 ±2 之外的观测，在正态假设下"几乎不可能"，在 t 假设下却很平常——这就是小样本必须用 t 的直观原因，也解释了你 [students.py](file:///d:/AI/CodeWork/numerical_statistics/statistics/pvalue/students.py#L22) 里 df=9 时 t 统计量换成 Z 会让 p 值被低估。

---

**User**

z分布和正态分布是什么关系

**Assistant**

## 先纠正一点命名

统计学里并没有一个独立的、叫 "z 分布" 的分布家族。**"z 分布"就是标准正态分布 N(0,1) 的口语叫法**——因为查表时用的变量符号是 z，所以老教材、软件菜单里常写成 "z 分布 / z 表"。

所以严格讲：**z 分布 ⊂ 正态分布**，它是正态分布的一个特例，不是并列关系。

## 三个层次的含义

| 说法 | 实际指什么 | 你代码里的位置 |
|---|---|---|
| z 分数（z-score） | 一个**数值**，标准化的结果 | [normal_3.py](file:///d:/AI/CodeWork/numerical_statistics/statistics/normal_distribution/normal_3.py#L22-L25)、[normal_5.py](file:///d:/AI/CodeWork/numerical_statistics/statistics/normal_distribution/normal_5.py#L49) 的 z = 1.5 |
| z 分布 | 标准正态分布 N(0,1) 这个**概率分布** | [normal.py](file:///d:/AI/CodeWork/numerical_statistics/statistics/normal_distribution/normal.py#L9) 画的那条曲线 |
| z 统计量 / z 检验 | 样本均值的**抽样分布**，σ 已知时服从 N(0,1) | 你 [students.py](file:///d:/AI/CodeWork/numerical_statistics/statistics/pvalue/students.py) 的对位概念（那里用的是 t） |

日常说"z 分布"，九成是在说第二层：**标准正态分布**。

## 核心关系：z 变换是"归一化通道"

一般正态 → 标准正态，靠一个线性变换：

```
z = (x - μ) / σ        X ~ N(μ, σ²)  ⟹  Z ~ N(0, 1)
```

这条通道的性质：

1. **满射且一一对应**，只做平移和缩放，不丢任何信息，只是换了把尺子。这正是 [normal_5.py](file:///d:/AI/CodeWork/numerical_statistics/statistics/normal_distribution/normal_5.py#L19-L40) 左右两栏在演示的：左边三条形状各异的 N(μ,σ)，右边全部落成同一条 N(0,1)。
2. **反向也能还原**：x = μ + zσ。
3. **所以全世界的正态只需一张表**。这就是"标准正态分布表"存在的全部理由——不管原始数据是身高、分数还是误差，查同一张表就够了。
4. z 的物理含义是"**距离均值几个标准差**"，因此它天生无量纲、可跨指标比较。[normal_3.py](file:///d:/AI/CodeWork/numerical_statistics/statistics/normal_distribution/normal_3.py#L38-L44) 里用 z 比较语文和数学，就是在用这个性质；而 [normal_2.py](file:///d:/AI/CodeWork/numerical_statistics/statistics/normal_distribution/normal_2.py) 填的 68%–95%–99.7%，本质上是 z 尺度上的固定区间。

一句话概括变换链：

```
任意正态 N(μ, σ²)  --z 变换-->  N(0, 1)  --查表/算 CDF-->  概率
```

## 和 t 分布的关系（接上一轮）

两者是同一件事在"σ 是否已知"下的两个分支：

```
X ~ N(μ, σ²)，样本均值 x̄ 的标准化

σ 已知  →  z = (x̄ - μ) / (σ/√n)  ~  N(0,1)      ← z 分布
σ 未知  →  t = (x̄ - μ) / (s/√n)  ~  t(n-1)      ← t 分布
```

注意这里有个容易混的点：**z 分数针对单个观测值 x，z 统计量针对样本均值 x̄**，分母多了个 √n（标准误）。

- σ 已知，或 n 很大（CLT 生效、s 已经估得很准）→ 用 z，即 N(0,1)；
- σ 未知且样本小 → 必须用 t，因为 s 自身带误差，尾部更厚、临界值更大。

你 [students.py](file:///d:/AI/CodeWork/numerical_statistics/statistics/pvalue/students.py) 那 10 个样本、σ 全校未知，走的就是下面那条支路；如果哪天你确认全校分数的标准差 σ=5 是已知真值，同一份数据就该换成 `stats.norm` 算——用 t 反而会得到偏保守（p 值偏大）的结论。

这就是为什么自由度 ν→∞ 时 t 分布收敛到 N(0,1)：样本足够多时 s→σ，"σ 未知"这层不确定性消失了，t 就退化成 z。

---

