Wilcoxon 检验里的 \(p\) 和 t 检验里的 \(p\) **叫法完全一样**，都是：

\[
\boxed{p\text{-value，P值}}
\]

也都是用来判断**统计显著性（statistical significance）**的。

所以论文里可以写：

> Wilcoxon signed-rank test showed a statistically significant difference (\(W=12.0,\ p=0.018\)).

中文就是：

> Wilcoxon 符号秩检验显示两组之间存在统计学显著差异（\(W=12.0,\ p=0.018\)）。

有一点术语要特别纠正：

\[
\boxed{p\text{ 值} \neq \text{显著性水平}}
\]

“显著性水平”严格来说是你事先设定的：

\[
\alpha=0.05
\]

然后拿 \(p\) 和 \(\alpha\) 比较：

\[
p<\alpha
\]

就称结果“具有统计显著性”。

所以无论是 t 检验：

\[
t=2.31,\quad p=0.032
\]

还是 Wilcoxon：

\[
W=8,\quad p=0.021
\]

都可以说：

\[
\boxed{\text{结果达到统计显著性}}
\]

但最好不要把 \(p=0.021\) 本身叫“显著性水平 0.021”。

你可以记成：

| 概念 | 英文 | 例子 |
|---|---|---|
| P 值 | p-value | \(p=0.021\) |
| 显著性水平 | significance level | \(\alpha=0.05\) |
| 统计显著 | statistically significant | \(p<0.05\) |

所以对于你的 EEG2Text 论文，t-test 和 Wilcoxon 最终都可以统一报告为：**test statistic + p-value + 是否 statistically significant**。