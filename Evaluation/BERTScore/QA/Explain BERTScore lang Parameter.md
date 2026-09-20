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

**User**

`& d:/AI/CodeWork/numerical_statistics/.venv/Scripts/python.exe d:/AI/CodeWork/numerical_statistics/Evaluation/BERTScore/bert_score_en.py`

**Assistant**

这个报错是我上一条说错了——`bert_score` 并没有装好。`load("bertscore")` 会从 Hub 下载指标脚本（这一步成功了），但脚本运行时 `import bert_score` 失败：这个依赖需要你自己在环境里装。项目用的是 uv，直接：

uv 缓存目录在沙箱限制外，需要绕过沙箱重试。

`bert-score==0.3.13` 装好了。现在运行脚本验证（首次运行会从 HF 下载 `roberta-large`，约 1.4GB）：

Python 进程正在跑（首次下载模型中）。等它结束后看输出。

---

**User**

`& d:/AI/CodeWork/numerical_statistics/.venv/Scripts/python.exe d:/AI/CodeWork/numerical_statistics/Evaluation/BERTScore/bert_score_en.py`

**Assistant**

脚本跑通了。结果印证了前面聊的内容：

```
Hashcode: roberta-large_L17_no-idf_version=0.3.12(hug_trans=4.57.6)
```

`lang="en"` → `roberta-large`、第 17 层隐状态，都和之前说的对上了。

## 输出里那条警告无害

```
Some weights of RobertaModel were not initialized ... ['pooler.dense.bias', 'pooler.dense.weight']
```

原因：BERTScore 只用各 token 的**隐状态**（hidden states）算相似度，完全不碰句子级的 pooler 层。transformers 加载时发现 checkpoint 里没有 pooler 权重就提示"新初始化了这几个参数"。这个参数根本不会被用到，可以忽略——和之前 COMET 那两条警告性质一样，都属于"机制上无害"。

## 结果解读

| 句对 | F1 | 对应差异 |
|---|---|---|
| The cat sits on the mat. vs is sitting | 0.978 | 进行时差异，语态不同 |
| very nice vs very good | 0.988 | nice/good 语义近 |
| child vs kid | 0.993 | 同义词，几乎无损 |

平均 F1 ≈ 0.987，符合预期——三对译文都语义等价，BERTScore 这类语义指标对同义替换（child/kid、nice/good）几乎不扣分，这正是它比 BLEU 更适合评翻译质量的地方。BLEU 会因为 n-gram 不匹配明显扣分，BERTScore 嵌入空间里 child 和 kid 的向量本来就很近。

---

**User**

Hashcode: roberta-large_L17_no-idf_version=0.3.12(hug_trans=4.57.6)
这个信息有什么用

**Assistant**

它是本次打分配置的**指纹**——记录“用哪把尺子量的分”。源码在 [utils.py](file:///d:/AI/CodeWork/numerical_statistics/.venv/Lib/site-packages/bert_score/utils.py#L665-L675)：

```python
def get_hash(model, num_layers, idf, rescale_with_baseline, ...):
    msg = "{}_L{}{}_version={}(hug_trans={})".format(
        model, num_layers, "_idf" if idf else "_no-idf", __version__, trans_version
    )
```

## 逐段拆解

| 片段 | 含义 |
|---|---|
| `roberta-large` | 用的模型 |
| `L17` | 取第 17 层隐状态 |
| `no-idf` | 没开 IDF 加权（`idf=True` 时会变成 `_idf`） |
| `version=0.3.12` | bert_score 库版本 |
| `hug_trans=4.57.6` | transformers 版本 |

## 三个实际用途

**1. 判断分数可不可比（最重要）**。BERTScore 分数没有绝对意义——换个模型、换个层、升级 transformers，分数分布都会移动。两次实验的 F1 想直接比较，先看 hashcode 是否一致：一致才可比，不一致要先换算到相同配置。

**2. 复现与报告**。论文或实验记录里报 BERTScore 时附上 hashcode，别人就知道你的分数是什么配置下算的，不会猜。

**3. 关联 rescale baseline**。包里自带一组按 `(语言, 模型)` 索引的基线文件（如 [rescale_baseline/en/roberta-large.tsv](file:///d:/AI/CodeWork/numerical_statistics/.venv/Lib/site-packages/bert_score/rescale_baseline/en/roberta-large.tsv)）。`rescale_with_baseline=True` 时按这个映射找基线做换算，把分数变成“相对基线系统好多少”，跨配置可比性才真正成立。

## 一个现成的活例子

你装的是 `bert-score==0.3.13`，但 hashcode 显示 `version=0.3.12`——因为 [__init__.py](file:///d:/AI/CodeWork/numerical_statistics/.venv/Lib/site-packages/bert_score/__init__.py#L1) 第一行还是 `__version__ = "0.3.12"`，作者发版时忘了改字符串。这正是 hashcode 存在的意义：**包名版本号靠不住，指纹才靠得住**。

---

**User**

`d:\AI\CodeWork\numerical_statistics/Evaluation/BERTScore/bert_score_zh.py#L57-59`什么是F1分数

**Assistant**

F1 是 Precision 和 Recall 的**调和平均**，但在 BERTScore 里这两个量的含义和分类任务里不太一样，先看图。

上图是 BERTScore 里 P 和 R 的真实含义（和分类任务里数 TP/FP 不同，这里是**软匹配**）：

- **Recall**：参考译文的每个词，去生成译文里找最相似的那个，取平均 → “参考被覆盖了多少”，**罚漏译**
- **Precision**：生成译文的每个词，去参考译文里找最相似的，取平均 → “生成的内容有多少站得住”，**罚多余、罚编造**

用你 [bert_score_zh.py](file:///d:/AI/CodeWork/numerical_statistics/Evaluation/BERTScore/bert_score_zh.py#L31-L43) 的第一句就很好理解：生成是“猫坐在垫子上。”，参考是“**一只**猫坐在垫子上。”——参考里多出的“一只”在生成里找不到匹配，所以这句的 Recall 会略低于 Precision。

而 F1 就是把这两者合成一个数，但用的是**调和平均**而不是普通平均——这是关键设计。

上表的第三行是 F1 存在的理由：一个漏译严重的译文，算术平均还有 0.50“看着还行”，但 F1 只有 0.18，直接暴露问题。

## 调和平均的两个关键性质

```python
F1 = 2 · P · R / (P + R)
```

1. **短板决定上限**：P 和 R 必须都高，F1 才高。任何一边趋近 0，F1 就跟着趋近 0（P=0.9、R=0.1 时 F1 只有 0.18）。
2. **永远 ≤ 算术平均**：均衡时二者相等，越失衡差距越大。所以 F1 天然偏向“两边都好”的译文。

## 实践中怎么用这三个数

- **报成绩看 F1**（通常再对整个测试集求平均，即你脚本里的 `mean_f1`）
- **诊断时分开看 P 和 R**：
  - R 明显低 → 模型**漏译**（参考里的内容没翻出来）
  - P 明显低 → 模型**冗余/幻觉**（生成了参考里没有依据的内容）

这和之前聊的 COMET 是同一个思路：COMET 只给一个综合分，BERTScore 多给了一步 P/R 分解，定位错误类型更方便。

---

**User**

这里面是严格意义上的精确率和召回率吗

**Assistant**

不是。BERTScore 的 P/R 只是**借用了这两个名字**，是经典定义的“软化推广”，不是严格意义上的精确率/召回率。

## 严格定义（分类/信息检索）

严格 P/R 建立在**二值判定**上，需要明确数出 TP、FP、FN：

```python
Precision = TP / (TP + FP)   # 判为正的里面，多少真的是正
Recall    = TP / (TP + FN)   # 真正的正里面，多少被抓出来了
```

每个样本要么匹配要么不匹配，是离散计数。

## BERTScore 的版本

没有任何“判定”环节，是连续的贪心匹配：

```python
P = mean( 对生成句每个词，取与参考句所有词的 最大余弦相似度 )
R = mean( 对参考句每个词，取与生成句所有词的 最大余弦相似度 )
```

| 维度 | 严格 P/R | BERTScore P/R |
|---|---|---|
| 判定方式 | 二值（匹配/不匹配） | 连续相似度 ∈ [0,1] |
| 计数 | TP/FP/FN 硬计数 | 相似度加权平均 |
| 匹配约束 | 一一对应 | 贪心、可多对一（同一个词可被反复匹配，无 BLEU 式截断） |
| 同义词 | child/kid 算 0 分 | 算接近 1 的分数 |

## 为什么不算乱起名

做个思想实验：把余弦相似度换成“完全相同=1，否则=0”，BERTScore 的 P/R 就退化成**词袋重叠率**——P = 生成词有多少比例出现在参考里，R = 参考词有多少比例被覆盖。这正好和严格 P/R 在词匹配场景下的语义对齐。所以可以说 BERTScore 的 P/R 是严格定义在**软匹配上的连续推广**：方向一致（P 管生成质量，R 管覆盖程度），但不再是离散计数。

顺带一提，这种“宽松用法”在 MT 评测里很普遍——ROUGE 的 P/R 同样是重叠率而非分类计数。读指标文档时看到 P/R，先看它底层是“数数”还是“算相似度”。

---

