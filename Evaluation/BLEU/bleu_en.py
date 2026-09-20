from sacrebleu.metrics import BLEU

"""
Bilingual Evaluation Understudy
双语评估替代指标
语料级 BLEU（corpus-level BLEU）
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

# 英文默认使用 13a 分词器,1~4-gram
bleu = BLEU(tokenize='13a')
# 计算语料级 BLEU 分数
score = bleu.corpus_score(sys, refs)


print(score)
print("BLEU 分数:", score.score)
print("各 n-gram 精度:", score.precisions)
print("长度惩罚 BP:", score.bp)
print("系统长度 / 参考长度:", score.sys_len, "/", score.ref_len)