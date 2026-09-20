`Unbabel/wmt22-comet-da` 本质上是一个 **COMET评价模型**，它不是一个从零设计的Transformer，而是：

> **XLM-RoBERTa（预训练语言模型） + COMET回归头（Regression Head）**

整体结构如下：:chatgpt-content-reference{index="0"}

```
                source sentence
                      |
                      |
                MT translation
                      |
                      |
              reference translation
                      |
                      |
        --------------------------------
        |                              |
        |       XLM-R Encoder          |
        |   (Transformer Encoder)      |
        |                              |
        --------------------------------
              |          |          |
              |          |          |
          h_src      h_mt       h_ref
              |          |          |
              ----------------------
                       |
                 COMET Pooling
                       |
              Sentence Representation
                       |
                 Regression Head
                       |
                 COMET Score
```

---

# 1. 输入是什么？

`wmt22-comet-da` 是 **reference-based metric**。

所以输入三个句子：

```python
{
    "src": "我喜欢猫",
    "mt": "I like cats",
    "ref": "I love cats"
}
```

分别表示：

|字段|含义|
|-|-|
|src|原文|
|mt|机器翻译结果|
|ref|人工参考译文|

---

# 2. Backbone：XLM-R

COMET没有自己训练一个语言模型，而是使用：

```
XLM-RoBERTa-large
```

作为encoder。:chatgpt-content-reference{index="1"}

结构：

```
Token IDs
    |
Embedding
    |
Transformer Encoder × 24
    |
Hidden States
```

XLM-R large:

```
Layers:       24
Hidden size:  1024
Attention heads: 16
Parameters:   ~550M
```

输出：

假设：

```
"The cat is on the mat"
```

token数量：

```
N = 8
```

得到：

```
8 × 1024
```

隐藏表示：

\[
H=
[
h_1,h_2,...,h_8
]
\]


---

# 3. 三路输入共享同一个XLM-R

不是三个模型。

而是：

```
             XLM-R
              |
    -----------------------
    |          |          |
   src        mt        ref
```

参数共享：

\[
Encoder(src)
\]

\[
Encoder(mt)
\]

\[
Encoder(ref)
\]


得到：

\[
e_s,e_m,e_r
\]


例如：

```
source embedding

[0.21,
 0.35,
 ...
 0.72]



translation embedding

[0.18,
 0.40,
 ...
 0.65]



reference embedding

[0.20,
 0.38,
 ...
 0.70]
```

---

# 4. COMET如何融合三个embedding？

核心思想：

比较：

```
机器翻译
vs
参考译文

同时考虑：

原文
```

通常构造：

\[
x=[e_s;e_m;e_r;
|e_m-e_r|;
|e_m-e_s|;
e_m\odot e_r]
\]


也就是：

拼接：

```
source vector

+

MT vector

+

reference vector

+

差异信息
```

例如：

```
e_mt
 |
 |
 |------ difference ------|
                          |
                       e_ref
```


这样模型可以学习：

- 翻译有没有漏信息
- 语义是否一致
- 表达是否自然

---

# 5. Regression Head

最后不是softmax分类。

而是：

```
Linear
 |
ReLU
 |
Dropout
 |
Linear
 |
score
```

输出：

\[
y\in [0,1]
\]


例如：

```
0.91
```

表示：

> 预测人工评价质量很高


---

# 6. 为什么叫 DA？

`wmt22-comet-da`

里面：

```
DA
```

表示：

**Direct Assessment**

即：

人工直接打分训练。

训练数据：

```
source
translation
reference

        |
        |
        v

human quality score
```

例如：

人工：

```
90/100
70/100
30/100
```

模型学习：

\[
COMET(x)\approx HumanScore
\]


---

# 7. PyTorch结构大概类似

简化版：

```python
class COMET(nn.Module):

    def __init__(self):

        self.encoder = XLMRobertaModel()

        self.regressor = nn.Sequential(
            nn.Linear(1024*6, 3072),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(3072,1)
        )


    def forward(src, mt, ref):

        e_src = self.encoder(src)
        e_mt  = self.encoder(mt)
        e_ref = self.encoder(ref)


        x = torch.cat(
            [
             e_src,
             e_mt,
             e_ref,
             abs(e_mt-e_ref),
             abs(e_mt-e_src),
             e_mt*e_ref
            ],
            dim=-1
        )


        score = self.regressor(x)

        return score
```

---

# 8. 和BLEU/chrF最大的区别

BLEU：

```
文本
 |
n-gram匹配
 |
score
```

没有理解语义。

---

COMET：

```
文本
 |
XLM-R
 |
语义embedding
 |
神经网络
 |
预测人工评分
```

所以：

例如：

参考：

```
The cat is on the mat.
```

模型：

```
A cat sits on the carpet.
```

BLEU：

可能低。

COMET：

可能高。

因为：

```
cat ≈ cat
mat ≈ carpet
sits ≈ is on
```

语义接近。

---

对于你做 **EEG→Text**，COMET的结构其实很有启发：

你未来可以设计：

```
EEG Encoder
      |
      v
EEG embedding

        +
        
LLM/Text embedding

        |
        v

Semantic Alignment Head

        |
        v

COMET-like semantic score
```

也就是把 EEG 信号映射到语言模型空间，再用类似 COMET 的语义评价方式评估生成文本。这个方向和目前 EEG-to-language 的趋势比较接近。