COMET（**Crosslingual Optimized Metric for Evaluation of Translation）** 是目前机器翻译（MT）和 LLM 翻译评价中非常重要的**神经网络评价指标**。

它和 BLEU、chrF 最大的区别：

| 指标 | 类型 | 核心思想 |
|-|-|-|
| BLEU | n-gram匹配 | 生成文本和参考文本词重合多少 |
| chrF | 字符n-gram匹配 | 字符层面的相似度 |
| COMET | 神经网络 | 学习“人类评价标准”预测质量 |

简单理解：

> BLEU/chrF 问："你的句子和参考答案像不像？"  
> COMET 问："你的句子表达的意思是否正确，像不像人翻译？"

---

# 1. 设计一个机器翻译场景

假设我们训练一个：

**英文 → 中文翻译模型**

任务：

输入：

```
Source:
The cat is sitting on the mat.
```

模型输出：

```
Hypothesis:
猫坐在垫子上。
```

人工参考：

```
Reference:
猫正在垫子上坐着。
```

---

我们比较两个模型：

## 模型A

```
猫坐在垫子上。
```

## 模型B

```
猫在地毯下面。
```

---

传统指标：

BLEU可能：

```
模型A BLEU = 0.75
模型B BLEU = 0.45
```

因为：

模型B和参考词不同：

```
垫子
地毯
```

但是：

COMET更关注语义：

模型B：

```
cat
mat
```

变成：

```
猫
地毯下面
```

意思错了。

所以：

```
COMET(A)=0.85

COMET(B)=0.25
```

---

# 2. COMET输入是什么？

COMET不是只输入：

```
hypothesis
```

而是三个东西：

```
(source, hypothesis, reference)
```

例如：

```
Source:
The cat is sitting on the mat.


Hypothesis:
猫坐在垫子上。


Reference:
猫正在垫子上坐着。
```

结构：

```
             Source
                |
                |
                v

        ┌─────────────┐
        │  Encoder    │
        │  XLM-R      │
        └─────────────┘


Hypothesis -------->


Reference --------->


              |
              v

        Regression Head

              |
              v

          COMET score
```

---

# 3. 安装COMET

目前主流库：

```
unbabel-comet
```

安装：

```bash
pip install unbabel-comet
```

---

# 4. 下载COMET模型

COMET有多个版本：

常见：

|模型|特点|
|-|-|
|wmt20-comet-da|经典|
|wmt22-comet-da|较新|
|Unbabel/wmt22-comet-da|论文常用|
|comet-kiwi|无reference版本|

---

下载：

```python
from comet import download_model

model_path = download_model(
    "Unbabel/wmt22-comet-da"
)

print(model_path)
```

第一次会下载：

```
XLM-R encoder
+
COMET regression head
```

大约几百MB。

---

# 5. 完整代码示例

## 场景

三句话：

```python
sources = [
    "The cat is sitting on the mat.",
    "I love artificial intelligence.",
    "The weather is very good today."
]


hypotheses = [
    "猫坐在垫子上。",
    "我喜欢人工智能。",
    "今天的天气很好。"
]


references = [
    "猫正在垫子上坐着。",
    "我热爱人工智能。",
    "今天天气非常好。"
]
```

---

## 计算COMET

```python
from comet import download_model, load_from_checkpoint


# 下载模型
model_path = download_model(
    "Unbabel/wmt22-comet-da"
)


# 加载
model = load_from_checkpoint(model_path)


# 构造数据
data = []

for src, hyp, ref in zip(
    sources,
    hypotheses,
    references
):
    data.append(
        {
            "src": src,
            "mt": hyp,
            "ref": ref
        }
    )


# 推理

output = model.predict(
    data,
    batch_size=8,
    gpus=1
)


print(output)
```

输出类似：

```
Prediction(
 scores=[
 0.87,
 0.92,
 0.84
 ],
 system_score=0.876
)
```

---

# 6. 查看每句话得分

```python
for i,score in enumerate(output.scores):

    print(
        sources[i]
    )

    print(
        "COMET:",
        score
    )

    print()
```

输出：

```
The cat is sitting on the mat.

COMET:
0.87


I love artificial intelligence.

COMET:
0.92


The weather is very good today.

COMET:
0.84
```

---

# 7. 和BLEU对比

例如：

```python
from sacrebleu.metrics import BLEU


bleu = BLEU(tokenize="zh")


score = bleu.corpus_score(
    hypotheses,
    [references]
)


print(score.score)
```

可能：

```
BLEU = 65.4
```

COMET：

```
0.876
```

注意：

二者范围不同：

|指标|范围|
|-|-|
|BLEU|0-100|
|chrF|0-100|
|COMET|-1~1左右|

---

# 8. COMET为什么能理解语义？

核心：

它使用预训练语言模型：

例如：

```
XLM-R
```

把句子编码成向量：

例如：

```
Source:

The cat is sitting on the mat


embedding:

[0.12,
 0.35,
 -0.21,
 ...]
```


然后比较：

```
Source embedding

      +
      
Hypothesis embedding

      +

Reference embedding
```

输入：

```
[x_src,
 x_mt,
 x_ref]
```

经过：

```
Regression Network
```

预测：

```
human score
```

训练目标：

人工评分：

```
5分
4分
3分
...
```

模型学习：

```
文本关系
        ↓
质量分数
```

---

# 9. COMET和BLEU最大的区别案例

参考：

```
Reference:

我喜欢吃苹果。
```


模型1:

```
我喜欢苹果。
```


模型2:

```
我讨厌吃苹果。
```


BLEU:

可能：

```
模型1:
90


模型2:
80
```


因为：

```
我
苹果
吃
```

大量重合。


但是COMET：

模型1：

```
0.90
```

模型2：

```
0.30
```

因为：

```
喜欢
↓
讨厌
```

语义完全相反。

---

# 10. 在LLM时代COMET的使用

现在论文中常见：

机器翻译：

```
BLEU
+
chrF
+
COMET
```

LLM生成评价：

更多使用：

|指标|用途|
|-|-|
|BLEU|格式/词匹配|
|chrF|字符级匹配|
|COMET|语义质量|
|BERTScore|语义相似|
|BLEURT|学习型评价|
|GPT-as-a-Judge|开放生成评价|

---

如果你后续研究 **EEG→Text**，COMET也很有参考价值：

例如：

EEG模型生成：

```
预测文本:
the dog is running
```

真实：

```
reference:
a dog runs in the park
```

BLEU可能偏低，因为词不同。

但COMET可能认为：

```
dog + running
```

语义正确，给较高分。

因此 EEG-to-text 论文通常会同时报告：

```
BLEU-1/2/3/4
ROUGE
METEOR
CIDEr
BERTScore
COMET
```

COMET非常适合衡量**脑信号解码后语言生成的语义正确性**。