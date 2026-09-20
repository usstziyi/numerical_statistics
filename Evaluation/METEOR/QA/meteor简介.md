**METEOR（Metric for Evaluation of Translation with Explicit ORdering）** 是一种用于评价**机器翻译（Machine Translation, MT）和文本生成质量**的自动评价指标。它和 BLEU、chrF 类似，都是比较：

> **模型生成文本（candidate / hypothesis）**  
> 和  
> **人工参考文本（reference）**

之间的相似程度。:chatgpt-content-reference{index="0"}

它最初提出的目的，就是解决 BLEU 的几个问题：

- BLEU 只看 **精确 n-gram 匹配**
- BLEU 更偏向 precision（准确率）
- BLEU 对词形变化、同义词不敏感
- BLEU 通常更适合语料级评价，不太适合单句评价

METEOR 引入了：
1. **词对齐（alignment）**
2. **Recall（召回率）**
3. **词形变化匹配**
4. **同义词匹配**
5. **顺序惩罚**

因此通常和人工评价相关性更好。:chatgpt-content-reference{index="1"}


---

# 1. METEOR核心思想

一句话：

> **看生成句子里面有多少内容和参考答案一致，同时考虑词序是否自然。**

例如：

Reference：

> The cat is sitting on the mat.

模型输出：

> The cat sits on the mat.


BLEU：

```
The
cat
on
the
mat
```

很多词匹配。

但是：

```
sitting
sits
```

不是完全一样。

BLEU认为：

```
sitting != sits
```

而 METEOR 可以认为：

```
sitting ≈ sits
```

因为词根都是：

```
sit
```

这就是 stemming（词干匹配）。:chatgpt-content-reference{index="2"}


---

# 2. METEOR计算流程

整体公式：

\[
METEOR=F_{mean}\times(1-Penalty)
\]

其中：

\[
F_{mean}
\]

衡量词匹配程度。

\[
Penalty
\]

惩罚词序混乱。 :chatgpt-content-reference{index="3"}




---

# 3. 第一步：词对齐 Matching

METEOR不是直接统计n-gram，而是先建立：

```
candidate
    |
    | word alignment
    |
reference
```

匹配规则一般有：

## (1) Exact match 精确匹配

例如：

candidate:

```
cat
```

reference:

```
cat
```

匹配。


---

## (2) Stem match 词干匹配

例如：

```
playing
played
plays
```

都会归到：

```
play
```

所以：

```
played ≈ playing
```

---

## (3) Synonym match 同义词

例如：

```
good
```

和：

```
excellent
```

可以匹配。

METEOR支持同义词/释义匹配机制。:chatgpt-content-reference{index="4"}


---

# 4. 第二步：计算 Precision 和 Recall

假设：

Reference:

```
The cat is on the mat
```

长度：

6 words


Candidate:

```
The cat sits on mat
```

长度：

5 words


匹配：

```
The
cat
on
mat
```

共：

4个


---

## Precision

生成文本里面，有多少是正确的：

\[
P=
\frac{
matched
}{
candidate\ length
}
\]


所以：

\[
P=\frac45=0.8
\]


---

## Recall

参考答案里面，有多少被覆盖：

\[
R=
\frac{
matched
}{
reference\ length
}
\]


所以：

\[
R=\frac46=0.667
\]


---

# 5. METEOR为什么强调Recall？

BLEU：

主要关注：

> 你生成的东西是不是参考答案里面出现过？

偏 Precision。


例如：

Reference:

```
I like machine learning
```

模型：

```
I like
```

BLEU可能不错。

但是人觉得：

> 少了一半信息。


METEOR认为：

Recall很重要：

```
参考答案的信息，你覆盖多少？
```

所以：

\[
F_{mean}
=
\frac{10PR}{R+9P}
\]


这里 Recall 权重大约是 Precision 的9倍。:chatgpt-content-reference{index="5"}


---

# 6. 第三个部分：词序惩罚 Penalty

只匹配词还不够。

例如：

Reference:

```
I love machine learning
```

输出：

```
learning machine love I
```


词全部存在：

```
I
love
machine
learning
```

Precision:

100%

Recall:

100%


但是语义顺序完全乱。


所以 METEOR增加：

## Fragmentation penalty

看匹配词是否连续。


---

正常：

```
I love | machine learning
```

两个连续块：

```
chunks=2
```


乱序：

```
learning | I | machine | love
```

chunks增加。


chunks越多：

Penalty越大。


最终：

\[
METEOR=
F_{mean}(1-Penalty)
\]


---

# 7. METEOR 和 BLEU 区别

|指标|BLEU|METEOR|
|-|-|-|
|基本单位|n-gram|word alignment|
|主要关注|Precision|Precision+Recall|
|同义词|❌|✅|
|词形变化|❌|✅|
|词序|n-gram间接考虑|显式惩罚|
|评价粒度|语料级强|句子级较强|
|适合LLM生成|一般|比BLEU更接近人工评价| :chatgpt-content-reference{index="6"}




---

# 8. 和你前面学习的 BLEU / chrF 对比

可以这样理解：

## BLEU

问：

> "你生成的词，有多少和标准答案连续匹配？"

关注：

```
surface overlap
```

---

## chrF

问：

> "字符层面像不像？"

例如中文：

```
猫在垫子上
猫坐在垫子上
```

虽然多了一个：

```
坐
```

但是字符 n-gram 大量重合。

---

## METEOR

问：

> "你的意思是不是一样？即使表达方式不同？"


例如：

Reference:

```
The car is fast.
```

Output:

```
The automobile is quick.
```

BLEU：

```
The
is
```

匹配少。


METEOR：

```
car ≈ automobile
fast ≈ quick
```

匹配更多。


---

# 9. 在LLM时代 METEOR还有用吗？

现在 LLM 评价常见：

|任务|常用指标|
|-|-|
|机器翻译|BLEU, chrF, COMET, METEOR|
|摘要|ROUGE, BERTScore|
|图像描述|CIDEr, SPICE|
|LLM生成|GPT-based judge, BERTScore, BLEURT, COMET|

METEOR现在不像 BLEU、COMET 那么主流，但仍然经常出现在：
- 机器翻译论文
- 老benchmark
- 文本生成比较实验

---

结合你之前研究的 **EEG→Text**：

如果模型从 EEG 生成：

```
真实文本:
A dog is running in the park.

模型:
A puppy runs in a garden.
```

那么：

- BLEU：可能低（词不同）
- chrF：中等（字符相似）
- METEOR：较高（dog≈puppy, running≈runs）
- COMET：可能更高（语义模型判断）

所以 EEG2Text 论文通常会同时报告：

```
BLEU-1/2/3/4
METEOR
CIDEr
ROUGE
BERTScore
```

因为它们分别衡量：
- 字面一致性
- 局部词匹配
- 语义一致性

METEOR正好处在 **BLEU → 语义指标** 的过渡位置。