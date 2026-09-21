不一定。**你的 EEG2Text 并不是因为是 EEG2Text 就“应该用 Wilcoxon”**，关键取决于你拿什么作为统计样本，以及两模型之间是不是配对数据。

假设你比较两个 EEG2Text 模型：

\[
A=\text{你的模型},\qquad B=\text{baseline}
\]

而且是在**同一批被试**上计算 BLEU、BERTScore、CIDEr 等指标。比如：

| Subject | A 的 BERTScore | B 的 BERTScore |
|---|---:|---:|
| S1 | 0.72 | 0.68 |
| S2 | 0.69 | 0.67 |
| S3 | 0.75 | 0.70 |
| ... | ... | ... |

这种情况下首先确定的是：

\[
\boxed{\text{这是配对设计}}
\]

因为：

\[
A_i \leftrightarrow B_i
\]

来自同一个被试。

接下来计算差值：

\[
d_i=A_i-B_i
\]

然后看这组：

\[
d_1,d_2,\dots,d_n
\]

的分布。

如果差值大致呈正态、没有严重异常值，那么用：

\[
\boxed{\text{paired t-test}}
\]

也就是：

```python
from scipy.stats import ttest_rel

t, p = ttest_rel(model_a, model_b)
```

如果差值明显非正态，尤其是**被试数较少 + 偏态明显 + 有异常值**，那么 Wilcoxon signed-rank 就很合适：

\[
\boxed{\text{Wilcoxon signed-rank test}}
\]

```python
from scipy.stats import wilcoxon

stat, p = wilcoxon(model_a, model_b)
```

所以你可以把决策理解为：

\[
\boxed{
\text{同一批被试}
\rightarrow
\text{配对数据}
\rightarrow
\begin{cases}
\text{差值近似正态} & \text{paired t-test}\\
\text{差值明显非正态} & \text{Wilcoxon}
\end{cases}
}
\]

这里有一个对 EEG2Text 特别重要的问题：**不要把所有 trial / sentence 直接当成独立样本。**

比如你有：

\[
20\text{ 个被试}\times100\text{ 个 trial}=2000\text{ 个样本}
\]

不能简单地做：

```python
wilcoxon(
    2000个模型A结果,
    2000个模型B结果
)
```

然后把 \(n=2000\) 当成 2000 个独立实验单位。因为同一被试内部的 trial 往往相关：

\[
x_{i1},x_{i2},\ldots,x_{i100}
\]

这会产生 **pseudoreplication（伪重复）**，导致 p-value 可能异常小。

对于典型 EEG2Text 实验，我更建议你先对**每个被试**计算一个指标：

\[
M_{A,i}
\]

和

\[
M_{B,i}
\]

然后：

\[
d_i=M_{A,i}-M_{B,i}
\]

最终统计样本量是**被试数量**，比如：

\[
n=20
\]

而不是 trial 数量 \(2000\)。

例如：

```python
import numpy as np
from scipy.stats import shapiro, ttest_rel, wilcoxon

model_a = np.array([
    0.72, 0.69, 0.75, 0.71, 0.74,
    0.68, 0.73, 0.76, 0.70, 0.72
])

model_b = np.array([
    0.68, 0.67, 0.70, 0.69, 0.71,
    0.66, 0.70, 0.72, 0.69, 0.68
])

diff = model_a - model_b

# 检查“差值”的正态性
print(shapiro(diff))

# 配对 t 检验
print(ttest_rel(model_a, model_b))

# Wilcoxon
print(wilcoxon(model_a, model_b))
```

注意 Shapiro-Wilk 检查的是：

```python
diff
```

而不是分别检查：

```python
model_a
model_b
```

这一点对配对 t 检验非常重要。

还有一个你做 EEG2Text 时容易遇到的特殊情况：如果你最后只有：

```text
Model A BLEU = 32.6
Model B BLEU = 30.1
```

也就是每个模型只有一个**整个测试集的 corpus-level BLEU**，那么：

\[
\boxed{\text{不能直接做 t-test，也不能直接做 Wilcoxon}}
\]

因为你只有两个数字，没有一组统计样本。

这种情况下通常需要构造重复测量单位，例如：

\[
\text{subject-level BLEU}
\]

或者通过：

- bootstrap
- permutation test
- repeated runs / random seeds

得到一个分布再进行统计比较。

而且 BLEU、CIDEr 这类指标还有一个特点：它们的分布通常不一定很好地满足正态假设。尤其当 EEG2Text 数据集的被试数量只有十几个、二十几个时，**Wilcoxon signed-rank 在论文里是很常见且合理的选择**。

所以针对你的 EEG2Text，我会建议这样的统计流程：

\[
\boxed{
\text{同一批 subjects 测 A、B}
}
\]

↓

\[
\boxed{
\text{每个 subject 分别计算 BLEU / BERTScore / CIDEr / ...}
}
\]

↓

计算：

\[
d_i=A_i-B_i
\]

↓

观察 Q-Q plot、异常值和差值分布。

然后：

\[
\boxed{
\begin{array}{ll}
\text{差值近似正态} & \rightarrow \text{paired t-test}\\[4pt]
\text{明显偏态/异常值，小样本} & \rightarrow \text{Wilcoxon signed-rank}
\end{array}
}
\]

如果你的 EEG2Text 最终是**十几到几十个被试，每个被试都有一个模型指标**，那么我会把 **paired t-test / Wilcoxon signed-rank** 作为主要二选一，而绝对不是 independent t-test / Mann–Whitney U。

另外，如果你同时比较 BLEU、chrF、ROUGE-L、BERTScore、CIDEr、METEOR 等很多指标，每个都做一次显著性检验，还要考虑**多重比较校正**，比如 Holm 或 Benjamini–Hochberg，否则连续做很多 \(p<0.05\) 检验会提高假阳性率。