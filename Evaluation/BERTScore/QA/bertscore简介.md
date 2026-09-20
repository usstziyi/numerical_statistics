BERTScore 是 NLP（尤其是机器翻译、文本生成、LLM 评价）里非常重要的**语义级评价指标**。它和你前面学习的 **BLEU、chrF、COMET、CIDEr** 属于同一类任务：评价模型生成文本质量。

不过它的思想和 BLEU/chrF 完全不同：

- **BLEU / chrF：比较字符串是否相似（字面匹配）**
- **BERTScore：比较语义是否相似（embedding 匹配）**
- **COMET：进一步学习一个模型预测人工评分**

---

# 1. 为什么需要 BERTScore？

先回顾 BLEU 的问题。

例子：

参考答案：

> The cat is sitting on the mat.

模型输出：

> A cat sits on a carpet.

人类觉得：

✅ 意思几乎一样。


但是 BLEU：

```
The cat is sitting on the mat
A cat sits on a carpet
```

逐词比较：

|词|是否匹配|
|-|-|
|cat|✓|
|on|✗|
|the|✗|
|mat|✗|
|sitting|✗|
|is|✗|

很多词不同。

BLEU 会认为：

> 相似度一般


但是语义：

```
cat ≈ cat
sitting ≈ sits
mat ≈ carpet
```

其实高度相关。

所以提出：

> 不比较词，而比较词的语义向量。


这就是 BERTScore。

---

# 2. BERTScore核心思想

一句话：

> 使用 BERT 把每个 token 映射成 embedding，然后计算候选句和参考句 token embedding 的最大余弦相似度。


流程：

```
Reference sentence

The cat is on the mat

        |
        v

     BERT Encoder

        |
        v

token embeddings


[0.21,0.33,...]
[0.56,0.12,...]
...


Candidate sentence

A cat sits on carpet

        |
        v

     BERT Encoder

        |
        v

token embeddings


[0.18,0.31,...]
...
```


然后比较：

```
candidate token
        |
        |
        v
找 reference 中最相似的token
```


---

# 3. 数学公式

假设：

参考：

\[
x=(x_1,x_2,...,x_m)
\]


生成：

\[
y=(y_1,y_2,...,y_n)
\]


BERT 得到：

\[
e(x_i)
\]

和

\[
e(y_j)
\]


每个 token 是一个向量：

例如：

```
cat

[0.12,
 0.56,
 -0.21,
 ...
 768维]
```


---

## Precision

生成文本中的每个 token：

找参考文本中最相似的 token。


公式：

\[
P=
\frac1{|y|}
\sum_{y_j}
\max_i
cos(e(y_j),e(x_i))
\]


意思：

```
每个生成词
↓
找参考里最像它的词
↓
平均
```


---

## Recall

反过来：

参考里的每个 token：

找生成文本中最像它的。


\[
R=
\frac1{|x|}
\sum_{x_i}
\max_j
cos(e(x_i),e(y_j))
\]


---

## F1

最终：

\[
BERTScore=
\frac{2PR}{P+R}
\]


和传统 F1 完全一样。


---

# 4. 手算一个简单例子

参考：

```
猫 坐 在 垫子 上
```


生成：

```
猫 蹲 在 地毯 上
```


假设 BERT embedding 相似度：

|生成token|最相似参考token|score|
|-|-|-|
|猫|猫|1.0|
|蹲|坐|0.85|
|在|在|1.0|
|地毯|垫子|0.8|
|上|上|1.0|


Precision:

\[
P=
(1+0.85+1+0.8+1)/5
\]


\[
P=0.93
\]


Recall：

参考：

|参考token|最大匹配|
|-|-|
|猫|1|
|坐|0.85|
|在|1|
|垫子|0.8|
|上|1|


所以：

\[
R=0.93
\]


F1：

\[
BERTScore=0.93
\]


虽然：

```
坐 != 蹲
垫子 != 地毯
```

但是语义接近。


---

# 5. BERTScore 和 BLEU 对比

|指标|核心|粒度|
|-|-|-|
|BLEU|n-gram精确匹配|字面|
|chrF|字符n-gram|字面|
|ROUGE|召回n-gram|字面|
|BERTScore|embedding相似|语义|
|COMET|学习人工评分|语义+质量预测|


---

# 6. BERTScore 使用什么 BERT？

不是固定 BERT。


论文：

> BERTScore: Evaluating Text Generation with BERT


可以选择：

英文：

```
bert-base-uncased
roberta-large
microsoft/deberta-xlarge
```


中文：

常用：

```
bert-base-chinese
hfl/chinese-roberta-wwm-ext
```

例如：

```python
from bert_score import score


P, R, F1 = score(
    cands,
    refs,
    lang="zh"
)

print(F1)
```


内部：

大概：

```
text
 |
tokenizer
 |
BERT
 |
hidden states
 |
cosine similarity
 |
max matching
 |
P/R/F1
```


---

# 7. 为什么 BERTScore 需要下载模型？

和 COMET 类似。

因为：

BLEU：

```
字符串算法
```

不需要模型。


BERTScore：

需要：

```
token
 |
BERT
 |
embedding
```


所以需要：

```
HuggingFace model
```

例如：

```
bert-base-chinese
```

第一次运行：

```
Downloading model...
```

以后缓存。


---

# 8. BERTScore 输出什么？

代码：

```python
from bert_score import score


candidates=[
"猫坐在垫子上"
]

references=[
"猫蹲在地毯上"
]


P,R,F1=score(
    candidates,
    references,
    lang="zh"
)


print(P)
print(R)
print(F1)
```


输出类似：

```
P tensor([0.94])

R tensor([0.95])

F1 tensor([0.95])
```


含义：

|值|含义|
|-|-|
|P|生成文本的信息是否都能在参考找到|
|R|参考信息是否被生成覆盖|
|F1|综合语义相似度|

---

# 9. BERTScore 有什么缺点？

## (1) 依赖模型

例如：

BERT 对：

```
苹果
```

可能理解：

```
水果
```

也可能：

```
Apple公司
```


上下文影响。


---

## (2) 不评价事实正确性

例：

参考：

```
北京是中国首都
```


生成：

```
上海是中国首都
```


BERTScore：

可能仍然很高。

因为：

```
北京 ≈ 上海
中国
首都
```


语义接近。


但是事实错误。


所以 LLM 评价通常组合：

```
BLEU
chrF
BERTScore
COMET
人工评价
```


---

# 10. EEG2Text 为什么可以用 BERTScore？

结合你的 EEG→Text 方向：

输入：

```
EEG signal
    |
EEG Encoder
    |
LLM Decoder
    |
"the dog is running"
```


参考：

```
"a dog runs"
```


BLEU:

可能低。


BERTScore:

高。


因为：

```
dog ≈ dog
running ≈ runs
```


所以 EEG2Text 论文常见：

|指标|作用|
|-|-|
|BLEU|词汇准确|
|ROUGE|覆盖|
|CIDEr|描述一致|
|METEOR|词形+语义|
|BERTScore|语义一致|
|COMET|人工质量预测|


---

# 11. 和 COMET 的关系（重点）

可以理解成：

```
BLEU
 |
 |  字符匹配
 v

BERTScore
 |
 |  embedding匹配
 v

COMET
 |
 |  神经网络学习评价
 v

Human score
```


COMET 内部其实也大量使用：

```
XLM-R encoder
+
regression head
```

所以：

BERTScore ≈ 手工设计的 embedding similarity

COMET ≈ 学习出来的评价器


---

# 12. 一句话总结

> **BERTScore 使用 BERT 的上下文 embedding，对生成文本和参考文本的 token 进行最大语义匹配，再计算 Precision、Recall 和 F1，因此比 BLEU/chrF 更能评价“意思是否一致”。**

对于你的 **EEG2Text 项目**，推荐评价组合：

```
BLEU-1~4
chrF++
METEOR
ROUGE-L
BERTScore
CIDEr
COMET
```

其中：

- BLEU/chrF → 看词汇准确
- BERTScore → 看语义是否表达正确
- CIDEr → 看生成描述是否像人工描述
- COMET → 接近人工评价

这套基本就是目前 EEG-to-text / image captioning / vision-language 论文常用评价体系。