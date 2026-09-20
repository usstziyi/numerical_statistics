是的，**chrF 的核心就是计算字符级 n-gram 的 Precision（精确率）和 Recall（召回率），然后通过 F-score 融合**。

它和 BLEU 最大的区别就在这里：

- **BLEU：主要看 Precision（模型生成的内容有多少正确）**
- **chrF：同时看 Precision 和 Recall（生成得准不准，以及覆盖了多少参考信息）**

---

## 1. chrF 的基本流程

假设：

预测（hypothesis）：

```
猫在垫子上。
```

参考（reference）：

```
猫坐在垫子上。
```

首先拆成字符 n-gram。

例如 2-gram：

预测：

```
猫在
在垫
垫子
子上
上。
```

参考：

```
猫坐
坐在
在垫
垫子
子上
上。
```

匹配：

```
在垫
垫子
子上
上。
```

共有：

\[
4
\]

个匹配。

---

## 2. Precision（精确率）

问题：

> 我生成出来的字符，有多少是参考中存在的？

公式：

\[
P_n=
\frac{
\text{匹配的n-gram数量}
}
{
\text{预测中的n-gram数量}
}
\]


例如 2-gram：

预测有：

\[
5
\]

个：

```
猫在
在垫
垫子
子上
上。
```

匹配：

\[
4
\]


所以：

\[
P_2=\frac45=0.8
\]


---

## 3. Recall（召回率）

问题：

> 参考答案中的信息，我覆盖了多少？

公式：

\[
R_n=
\frac{
\text{匹配的n-gram数量}
}
{
\text{参考中的n-gram数量}
}
\]


参考：

```
猫坐
坐在
在垫
垫子
子上
上。
```

也是 6 个。

匹配：

4 个。


所以：

\[
R_2=\frac46=0.667
\]


---

## 4. 然后计算 F-score

普通 F1：

\[
F_1=
2\frac{PR}{P+R}
\]


但是 chrF 默认不是 F1，而是：

\[
F_\beta
\]


其中：

\[
\beta=2
\]


所以：

\[
F_2=
(1+\beta^2)
\frac{PR}
{\beta^2P+R}
\]


代入：

\[
F_2=
5
\frac{PR}{4P+R}
\]


---

## 为什么 chrF 用 β=2？

因为机器翻译里：

> 漏译（missing information）通常比多译一些内容更严重。

例如：

参考：

```
猫坐在垫子上。
```

生成：

```
猫在垫子上。
```

虽然缺少：

```
坐
```

但是整体意思还在。

chrF 希望 Recall 更重要：

\[
\beta=2
\]


表示：

\[
\text{Recall权重} > \text{Precision权重}
\]

---

## 5. 多阶 n-gram 怎么合并？

默认：

```python
CHRF(char_order=6)
```

会计算：

\[
n=1,2,3,4,5,6
\]


得到：

\[
P_1,P_2,...,P_6
\]

和：

\[
R_1,R_2,...,R_6
\]


然后平均：

\[
P=\frac{1}{6}\sum_{n=1}^{6}P_n
\]


\[
R=\frac{1}{6}\sum_{n=1}^{6}R_n
\]


最后：

\[
chrF=F_2(P,R)
\]


---

## 和 BLEU 对比

| | BLEU | chrF |
|-|-|-|
|基本单位|词 token|字符|
|主要统计|Precision|Precision + Recall|
|n-gram|1~4|1~6|
|惩罚机制|长度惩罚 BP|天然 Recall 控制|
|核心公式|几何平均 Precision|F-score|
|中文适应性|依赖分词|较好|

所以一句话总结：

\[
\boxed{
\text{chrF = 字符级 n-gram 的 Precision + Recall，再计算 }F_2\text{ 分数}
}
\]

其中默认：

\[
\boxed{
\text{chrF2 = char 1-6 gram + }\beta=2
}
\]