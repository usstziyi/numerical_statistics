# Explain BERTScore lang Parameter

> **Workspace:** d:\AI\CodeWork\numerical_statistics

---

**User**

`d:\AI\CodeWork\numerical_statistics/Evaluation/BERTScore/bert_score_en.py#L53-57`解释一下lang这个参数

**Assistant**

# `lang` 参数的作用

`lang` 指定被评文本的语言，**核心作用是自动选择合适的预训练模型来计算 token 嵌入**。

## 工作机制

BERTScore 不直接算 n-gram 匹配，而是把句子里的每个 token 用预训练语言模型编码成向量，再算余弦相似度。不同语言需要不同的模型——英文 BERT 对中文基本无效（分词、词表都不匹配）。

当你不显式传 `model_type` 时，`lang` 通过 `bert_score.lang2model` 映射到默认模型：

```python
"en"      -> "roberta-large"          # 英语
"zh"      -> "bert-base-chinese"      # 中文
"tr"      -> "dbmdz/bert-base-turkish-cased"
其他语言   -> "bert-base-multilingual-cased"  # 兜底多语言模型
```

## 两个关键点

**1. 与 `model_type` 的关系**：二者必须至少提供一个，否则报错。显式传了 `model_type` 后 `lang` 就不再决定用哪个模型：

```python
# 等价写法：显式指定模型
results = bertscore.compute(
    predictions=predictions,
    references=references,
    model_type="roberta-large",  # 此时 lang 可以省略
)
```

**2. 为什么这里源文是中文，却传 `lang="en"`**：正如你文件注释里写的，BERTScore 只用 `predictions`（模型英文译文）+ `references`（人工英文参考），**source 根本不参与计算**。被评分的文本全是英文，所以 `lang="en"` 是正确的。如果你评的是中→中任务，就该传 `lang="zh"`。

## 类比 COMET

和 COMET 显式绑定 `wmt22-comet-da` 一个固定的多语言模型不同，BERTScore 把“用哪个模型”暴露成了参数，`lang` 只是帮你挑默认值的便捷入口。这也意味着 BERTScore 的分数跨语言不可直接比较——英语用 roberta-large、中文用 bert-base-chinese，打分的“尺子”本身就不一样。

---

**User**

"en"      -> "roberta-large"          # 英语
"zh"      -> "bert-base-chinese"      # 中文
"tr"      -> "dbmdz/bert-base-turkish-cased"
其他语言   -> "bert-base-multilingual-cased"  # 兜底多语言模型

第四个需要怎么传

**Assistant**

不需要传特殊关键字——不存在 `lang="multi"` 这种写法。`lang2model` 在 bert_score 源码里是个 `defaultdict`，只有 4 个语言被特殊指定了模型，**其余任何语言代码都会自动落到 `bert-base-multilingual-cased`**。

## 传法

直接传真实的 ISO 语言代码即可，比如日语、法语、俄语：

```python
results = bertscore.compute(
    predictions=predictions,
    references=references,
    lang="ja"   # 不在映射表里 → 自动使用 bert-base-multilingual-cased
)
```

## 背后的机制

bert_score 的 `utils.py` 里本质是这样：

```python
lang2model = defaultdict(lambda: "bert-base-multilingual-cased")
lang2model.update({
    "en": "roberta-large",
    "en-sci": "scibert-scivocab-uncased",
    "zh": "bert-base-chinese",
    "tr": "dbmdz/bert-base-turkish-cased",
})
```

`defaultdict` 的特性：查表命中就用指定模型，**查不到就用默认值**，不会报错。

## 两个注意点

**1. 传错了会静默兜底**。比如你手滑写成 `lang="japanese"` 或 `lang="jp"`（正确代码是 `ja`），一样不报错，照样用多语言模型跑完——运气好结果没差别，但机制上你是“碰巧”对的。写错特殊语言如 `"en"` → `"en"` 才会出问题。

**2. 想显式控制就直接给 `model_type`**，跳过 lang 映射，还能换更强的模型：

```python
results = bertscore.compute(
    predictions=predictions,
    references=references,
    model_type="bert-base-multilingual-cased",  # 或 xlm-roberta-large 等
)
```

实际做跨语言评测时，很多人不用这个兜底的多语言 BERT-base，而是显式传 `model_type="xlm-roberta-large"`，效果通常更好。

---

**User**

默认是哪个

**Assistant**

分两层回答“默认”：

## 1. `lang` 本身没有默认值

不传 `lang` 且不传 `model_type` 会直接报错：

```python
results = bertscore.compute(
    predictions=predictions,
    references=references,
    # 什么都没传 → ValueError: Either `lang` or `model_type` must be specified
)
```

`lang` 的参数签名是 `lang=None`，但这个 `None` 不是“默认用某语言”，而是“必须二选一提供 `lang` 或 `model_type`”的占位。

## 2. 兜底模型默认是 `bert-base-multilingual-cased`

上一条说的 `defaultdict` 兜底——凡是没被特殊映射的语言代码（`fr`、`ja`、`de`、`ru`……），都用这个多语言 BERT-base。

总结成一张表：

| 传参方式 | 使用的模型 |
|---|---|
| `lang="en"` | roberta-large |
| `lang="zh"` | bert-base-chinese |
| `lang="tr"` | dbmdz/bert-base-turkish-cased |
| `lang="ja"`（或其他未映射代码） | bert-base-multilingual-cased（兜底） |
| `lang` 和 `model_type` 都不传 | 报错 |
| 只传 `model_type="xxx"` | 你指定的 xxx |

另外你的文件能跑通，说明 `bert_score` 包已经装好了（`evaluate` 的 bertscore 指标是运行时 `import bert_score` 的，不在 site-packages 的 evaluate 目录里）。

---

**User**

model_type这个参数传什么，模型名称还是类型

**Assistant**

传的是**模型名称**（HuggingFace 的模型标识符），不是 "bert"/"roberta" 这种类型名。参数名有点误导。

## 内部机制

`model_type` 会被原样传给 transformers：

```python
tokenizer = AutoTokenizer.from_pretrained(model_type)
model = AutoModel.from_pretrained(model_type)
```

所以任何 `from_pretrained()` 接受的东西都行。

## 三种合法传法

```python
# 1. Hub 模型 ID
model_type="roberta-large"

# 2. 带命名空间的 Hub ID
model_type="xlm-roberta-large"
model_type="microsoft/deberta-xlarge-mnli"

# 3. 本地模型目录（离线场景）
model_type=r"D:\models\roberta-large"
```

## 配套参数 `num_layers`

BERTScore 不是用模型最后一层，而是用**特定层的隐状态**（中间层对语义相似度更有效）。常见模型的推荐层数内置在 `model2layers` 表里：

| model_type | 默认层数 |
|---|---|
| roberta-large | 17 |
| bert-base-chinese | 12 |
| xlm-roberta-large | 20 |

用**表里没有的模型**（比如自己微调过的）时必须手动指定，否则报错：

```python
results = bertscore.compute(
    predictions=predictions,
    references=references,
    model_type="my-finetuned-roberta",
    num_layers=17,          # 不指定会报错
)
```

## 优先级

`model_type` 和 `lang` 都传时，`model_type` 优先，`lang` 被忽略。你可以在输出里验证：`results["hashcode"]` 会把实际用的模型和层数编码进去，比如 `roberta-large_L17`——这就是“本次打分用的尺子”的指纹，跨实验比较分数时先核对它。

---

