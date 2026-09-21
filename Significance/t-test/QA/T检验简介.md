T 检验（t-test）本质上是在问：

> **观察到的“均值差异”，到底大到足以认为不是随机波动造成的吗？**

它是一类经典的**参数假设检验**，主要用于比较均值。你在 EEG、NLP 模型评估、不同算法实验中都会经常遇到，比如比较两个模型在多个被试上的 BLEU、BERTScore、分类准确率是否存在统计学差异。

---

## 1. T 检验最核心的思想

假设我们比较两个模型 A 和 B：

\[
A=[0.71,0.73,0.69,0.75,0.72]
\]

\[
B=[0.68,0.70,0.67,0.72,0.69]
\]

肉眼看起来 A 的平均分更高，但问题是：

> 这个差异是真正稳定存在，还是刚好这几个样本随机造成的？

所以建立两个假设：

\[
H_0:\mu_A=\mu_B
\]

称为**零假设**，即没有均值差异。

另一个是：

\[
H_1:\mu_A\neq\mu_B
\]

称为备择假设。

T 检验最后计算一个统计量：

\[
t=\frac{\text{观察到的均值差异}}
{\text{均值差异的标准误}}
\]

因此可以直观理解成：

\[
\boxed{
t=
\frac{\text{信号}}
{\text{噪声}}
}
\]

如果两个均值差异很大，而数据波动很小，那么 \(t\) 就会很大。

然后根据 t 分布计算：

\[
p\text{-value}
\]

如果例如：

\[
p<0.05
\]

通常就拒绝：

\[
H_0
\]

认为均值差异达到了统计显著。

但注意：

> **p < 0.05 不等于差异很大，也不等于 A 一定比 B “好很多”。**

它只是在当前模型假设下说明：如果真实均值没有差异，那么观察到当前这么极端结果的概率比较小。

---

# 2. T 检验主要有三种

## 一、单样本 T 检验

问题：

> 一个样本的均值是否与某个已知值不同？

例如某个 EEG 模型在 20 个被试上的准确率：

\[
[0.71,0.74,\dots]
\]

你想检验：

\[
\mu > 0.5
\]

也就是是否显著高于随机分类水平。

统计量：

\[
t=
\frac{\bar{x}-\mu_0}
{s/\sqrt n}
\]

其中：

- \(\bar{x}\)：样本均值
- \(\mu_0\)：假设均值
- \(s\)：样本标准差
- \(n\)：样本数

Python：

```python
from scipy.stats import ttest_1samp

scores = [0.71, 0.74, 0.69, 0.76, 0.72]

result = ttest_1samp(
    scores,
    popmean=0.5
)

print(result.statistic)
print(result.pvalue)
```

---

# 3. 独立样本 T 检验

用于：

> 比较两个**相互独立的群体**的均值。

例如：

模型 A 用 20 个被试：

\[
A_1,A_2,\dots,A_{20}
\]

模型 B 用另外 20 个被试：

\[
B_1,B_2,\dots,B_{20}
\]

两组人不是同一批。

零假设：

\[
H_0:\mu_A=\mu_B
\]

经典 Student t-test 的形式大致是：

\[
t=
\frac{\bar X_1-\bar X_2}
{s_p
\sqrt{
\frac1{n_1}+\frac1{n_2}
}}
\]

其中 \(s_p\) 是两组的合并标准差。

Python：

```python
from scipy.stats import ttest_ind

a = [0.71, 0.73, 0.69, 0.75, 0.72]
b = [0.68, 0.70, 0.67, 0.72, 0.69]

result = ttest_ind(
    a,
    b,
    equal_var=True
)

print(result.statistic)
print(result.pvalue)
```

---

# 4. 配对样本 T 检验

这个在你的实验场景中特别重要。

假设同样 10 个 EEG 被试分别测试模型 A 和模型 B：

| 被试 | A | B |
|---|---:|---:|
| 1 | 0.73 | 0.69 |
| 2 | 0.72 | 0.70 |
| 3 | 0.78 | 0.74 |
| ... | ... | ... |

这里 A 和 B 不是独立的。

因为：

\[
A_1 \leftrightarrow B_1
\]

来自同一个人。

所以应该先计算每一对的差值：

\[
d_i=A_i-B_i
\]

然后实际上做的是：

\[
H_0:\mu_d=0
\]

统计量：

\[
t=
\frac{\bar d}
{s_d/\sqrt n}
\]

所以：

\[
\boxed{
\text{配对 t 检验}
=
\text{对差值进行单样本 t 检验}
}
\]

Python：

```python
from scipy.stats import ttest_rel

model_a = [0.71, 0.73, 0.69, 0.75, 0.72]
model_b = [0.68, 0.70, 0.67, 0.72, 0.69]

result = ttest_rel(model_a, model_b)

print(result.statistic)
print(result.pvalue)
```

对于机器学习实验，比如：

> 同一组测试样本 / 同一组被试 / 同一组随机种子，分别跑模型 A 和 B

通常应该优先考虑**配对检验**。

---

# 5. 做 T 检验有什么前提？

这是最重要的部分。

不同 t 检验的前提稍有不同，但主要有下面几个。

## ① 数据应该是连续型或近似连续型

T 检验研究的是：

\[
\text{均值}
\]

所以通常适合：

- EEG amplitude
- reaction time
- BLEU score
- BERTScore
- accuracy
- loss
- 血压
- 身高

这种数值变量。

不适合直接拿：

```text
男 / 女
A类 / B类
正确 / 错误
```

这样的分类标签做均值 t 检验。

当然，accuracy 本身虽然来自 0/1 结果，但如果你比较的是“每个被试得到的 accuracy”，那这些 accuracy 可以作为连续数值处理。

---

# 6. ② 样本之间应该独立

这是一个非常重要的假设：

\[
X_i \perp X_j
\]

意思是一个观测值不能偷偷影响另一个观测值。

例如：

20 个不同被试的测量值，在合理实验设计下可以认为相互独立。

但如果你有：

```text
被试1：1000个 EEG epoch
被试2：1000个 EEG epoch
```

然后把：

\[
2000
\]

个 epoch 全当成 2000 个独立样本做 t 检验，就很可能有问题。

因为同一个被试内部的 epoch：

\[
x_{1,1},x_{1,2},\dots
\]

通常是相关的。

这就是典型的：

> **伪重复（pseudoreplication）**

EEG 研究中尤其要注意。

---

# 7. ③ 数据需要近似正态分布

这里特别容易被误解。

### 单样本 T 检验

要求：

\[
X
\]

大致服从正态分布。

### 独立样本 T 检验

通常认为两组数据分别近似正态。

### 配对 T 检验

重点来了：

并不是要求 A 和 B 各自都正态。

而是要求：

\[
D=A-B
\]

这个**差值分布**近似正态。

所以配对 t 检验真正检查的是：

```python
diff = model_a - model_b
```

是否存在严重非正态性。

可以用：

```python
from scipy.stats import shapiro

shapiro(diff)
```

进行 Shapiro-Wilk 正态性检验。

---

# 8. 但是 T 检验真的必须严格正态吗？

并不是。

这是 t 检验一个很重要的性质。

如果样本量比较大，根据**中心极限定理**：

\[
\bar X
\]

的抽样分布会逐渐接近正态，因此 t 检验对适度的非正态其实比较鲁棒。

粗略来看：

- \(n<20\)：比较关注正态性和异常值
- \(n\approx 30+\)：通常对轻度非正态比较稳健
- 更大的 \(n\)：一般更稳健

但这不是绝对规则。

真正危险的情况通常是：

\[
\boxed{\text{严重偏态 + 极端异常值 + 小样本}}
\]

而不是“Shapiro-Wilk p < 0.05 就绝对不能用 t-test”。

---

# 9. ④ 独立样本 Student T 检验还有一个条件：方差齐性

经典独立样本 t 检验要求：

\[
\sigma_1^2=\sigma_2^2
\]

即两个总体方差相同。

例如：

\[
Var(A)\approx Var(B)
\]

可以用 Levene 检验：

```python
from scipy.stats import levene

stat, p = levene(a, b)

print(p)
```

零假设为：

\[
H_0:\sigma_A^2=\sigma_B^2
\]

如果：

\[
p>0.05
\]

通常没有充分证据认为方差不同。

---

# 10. Welch T 检验解决方差不齐问题

实际研究里，我通常更建议你认识 Welch t-test。

SciPy：

```python
ttest_ind(
    a,
    b,
    equal_var=False
)
```

就是：

\[
\boxed{\text{Welch's t-test}}
\]

它不要求：

\[
\sigma_A^2=\sigma_B^2
\]

所以：

| 方法 | 方差齐性要求 |
|---|---|
| Student t-test | 要求 |
| Welch t-test | 不要求 |

而且当真实方差相等时，Welch 通常也不会损失太多性能。

因此现在很多统计实践中：

\[
\boxed{
\text{独立两组比较可以默认优先考虑 Welch t-test}
}
\]

而不是一定先做 Levene，再决定该用哪个。

---

# 11. 配对 T 检验不要求两组方差相等

例如：

```python
ttest_rel(A, B)
```

它关心的不是：

\[
Var(A)=Var(B)
\]

而是差值：

\[
D=A-B
\]

所以核心要求是：

\[
D_1,D_2,\dots,D_n
\]

相互独立，并且差值总体没有严重偏离正态。

---

# 12. 异常值也是一个重要条件

例如：

```text
A =
0.71
0.72
0.73
0.70
5.91
```

最后一个极端值会严重影响：

\[
\bar x
\]

和：

\[
s
\]

因为 t 检验是基于均值和标准差的，因此：

\[
\boxed{\text{t 检验对异常值比较敏感}}
\]

所以正式实验中最好先画：

- histogram
- boxplot
- Q-Q plot

看看数据。

---

# 13. 三种 T 检验的前提总结

| T 检验 | 场景 | 核心假设 |
|---|---|---|
| 单样本 t | 一组 vs 固定值 | 独立、总体/数据近似正态 |
| 独立样本 Student t | 两个独立组 | 独立、近似正态、方差齐 |
| Welch t | 两个独立组 | 独立、近似正态，不要求方差齐 |
| 配对 t | 同一对象两次测量 | 配对正确、各对之间独立、**差值近似正态** |

其中最值得记住的是：

\[
\boxed{
\text{独立性比“是否完美正态”更加重要}
}
\]

---

# 14. 如果不满足正态性怎么办？

常见对应关系是：

| 参数检验 | 常见非参数替代 |
|---|---|
| 单样本 t-test | Wilcoxon signed-rank |
| 独立样本 t-test | Mann–Whitney U |
| 配对 t-test | Wilcoxon signed-rank |

例如配对数据：

```python
from scipy.stats import wilcoxon

wilcoxon(model_a, model_b)
```

不过不能简单理解成：

> 非正态 → 必须用非参数检验。

因为 t 检验在很多情况下相当稳健，而且非参数检验检验的统计对象也并不总是“两个均值是否相等”。

---

# 15. 放到你的 EEG / 模型实验中理解

假设你的 EEG2Text 有两个模型：

```text
Transformer
CNN + Transformer
```

对同样 20 个被试分别计算 BERTScore：

\[
A_1,\dots,A_{20}
\]

\[
B_1,\dots,B_{20}
\]

因为：

\[
A_i,B_i
\]

来自同一个被试，所以应该考虑：

\[
D_i=A_i-B_i
\]

然后检验：

\[
H_0:\mu_D=0
\]

这就是：

```python
from scipy.stats import ttest_rel

t, p = ttest_rel(A, B)
```

而不是把 20 + 20 个值当成两组完全无关的观测。

---

## 最后把 T 检验压缩成一句话

\[
\boxed{
t
=
\frac{\text{均值差异}}
{\text{这个均值差异本身的随机不确定性}}
}
\]

而决定能不能合理使用 t 检验，主要看四件事：

\[
\boxed{
\text{数据类型}
+
\text{独立性}
+
\text{正态性}
+
\text{是否需要方差齐性}
}
\]

其中对于你以后做 **EEG / NLP 模型性能比较**，我尤其建议记住：**同一批被试比较两个模型 → 通常是配对设计；配对 t 检验检查的是“差值”的分布，而不是分别检查 A、B 两组是否正态。**