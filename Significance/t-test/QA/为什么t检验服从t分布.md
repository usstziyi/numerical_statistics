因为 **t 检验里的统计量，恰好可以写成“标准正态变量 ÷ 一个独立的卡方变量平方根”**，而这种比值按定义就服从 **Student t 分布**。

以最经典的**单样本 t 检验**为例。假设

\[
X_1,\dots,X_n \overset{iid}{\sim} N(\mu,\sigma^2)
\]

我们想检验

\[
H_0:\mu=\mu_0
\]

如果总体标准差 \(\sigma\) 是已知的，那么样本均值满足

\[
\bar X\sim N\left(\mu,\frac{\sigma^2}{n}\right)
\]

所以在 \(H_0\) 成立时，

\[
Z=\frac{\bar X-\mu_0}{\sigma/\sqrt n}\sim N(0,1)
\]

这其实就是 **Z 检验**。

问题是实际中通常不知道 \(\sigma\)，只能用样本标准差

\[
S=\sqrt{
\frac{1}{n-1}
\sum_{i=1}^n(X_i-\bar X)^2
}
\]

来估计它。

于是我们把上面的 \(\sigma\) 换成 \(S\)：

\[
T=
\frac{\bar X-\mu_0}{S/\sqrt n}
\]

关键就在这里：**因为 \(S\) 自己也是一个随机变量，所以 \(T\) 不再服从标准正态分布。**

对于正态总体，有一个非常重要的结论：

\[
\frac{(n-1)S^2}{\sigma^2}
\sim \chi^2_{n-1}
\]

同时还有另一个特别关键的性质：

\[
\bar X \quad\text{与}\quad S^2
\]

是**相互独立的**。

现在把 \(T\) 改写一下：

\[
T=
\frac{
\dfrac{\bar X-\mu_0}{\sigma/\sqrt n}
}{
\dfrac{S}{\sigma}
}
\]

上面的分子我们已经知道：

\[
Z=
\frac{\bar X-\mu_0}{\sigma/\sqrt n}
\sim N(0,1)
\]

再来看分母。因为

\[
U=\frac{(n-1)S^2}{\sigma^2}
\sim\chi^2_{n-1}
\]

所以

\[
\frac{S^2}{\sigma^2}
=
\frac{U}{n-1}
\]

因此

\[
\frac{S}{\sigma}
=
\sqrt{\frac{U}{n-1}}
\]

代回去：

\[
\boxed{
T=
\frac{Z}{
\sqrt{U/(n-1)}
}
}
\]

其中

\[
Z\sim N(0,1)
\]

\[
U\sim\chi^2_{n-1}
\]

并且 \(Z\) 和 \(U\) 相互独立。

而 **t 分布的数学定义**正是：

\[
\boxed{
T=\frac{Z}{\sqrt{U/\nu}}
\sim t_\nu
}
\]

其中

\[
Z\sim N(0,1),\qquad
U\sim\chi^2_\nu
\]

二者独立。

所以这里直接得到

\[
\boxed{
\frac{\bar X-\mu_0}{S/\sqrt n}
\sim t_{n-1}
}
\]

这就是“为什么 t 检验里的 \(t\) 统计量服从 t 分布”。

---

最重要的直觉其实是：

如果我们知道真实的 \(\sigma\)：

\[
\frac{\bar X-\mu}{\sigma/\sqrt n}
\]

服从标准正态分布。

但是我们不知道 \(\sigma\)，只能用随机的 \(S\) 去估计：

\[
\frac{\bar X-\mu}{S/\sqrt n}
\]

所以分母也开始“抖动”。

这种额外的不确定性使得分布的两边尾巴变厚，于是：

\[
\text{Normal}
\quad\longrightarrow\quad
\text{t distribution}
\]

这也是为什么 t 分布相比标准正态分布**尾部更厚**：它不仅包含了样本均值的随机性，还包含了“用 \(S\) 估计 \(\sigma\)”带来的额外随机性。

而随着样本量 \(n\) 越来越大，

\[
S\to\sigma
\]

估计越来越稳定，因此

\[
t_{n-1}\to N(0,1)
\]

所以自由度很大时，t 分布看起来就几乎和标准正态分布一样。

一句话总结：

\[
\boxed{
\text{t 分布}
=
\frac{\text{标准正态}}
{\sqrt{\text{独立卡方}/\text{自由度}}}
}
\]

而 t 检验的统计量恰好能够化成这个形式，因此它服从 t 分布。