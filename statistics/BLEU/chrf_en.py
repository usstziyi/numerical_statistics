from sacrebleu.metrics import CHRF

"""
chrF: character n-gram F-score
字符级 n-gram F 分数
语料级 chrF（corpus-level chrF）
"""

# 系统输出（模型生成的译文）
sys = [
    "The dog bit the man.",
    "It wasn't surprising.",
    "The cat is on the mat.",
]

# 参考译文，每个句子可以有多个参考
refs = [
    # 第 1 套参考译文
    [
        "The dog bit the man.",
        "It was not unexpected.",
        "There is a cat on the mat.",
    ],

    # 第 2 套参考译文
    [
        "The dog had bit the man.",
        "No one was surprised.",
        "A cat is on the mat.",
    ],
]

# chrF 默认参数即论文推荐配置：
# char_order=6 统计 1~6 元字符 n-gram
# word_order=0 表示纯 chrF；改为 2 即为 chrF++（额外统计 1~2 元词 n-gram）
# beta=2 表示召回率的权重是精确率的 4 倍
chrf = CHRF(char_order=6, word_order=0, beta=2)
# 计算语料级 chrF 分数
score = chrf.corpus_score(sys, refs)

print(score)
print("chrF 分数:", score.score)
print("字符 n-gram 阶数:", score.char_order)
print("词 n-gram 阶数:", score.word_order)
print("beta（召回权重）:", score.beta)
