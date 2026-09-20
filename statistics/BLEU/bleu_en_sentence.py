from sacrebleu.metrics import BLEU

"""
Bilingual Evaluation Understudy
双语评估替代指标
句子级 BLEU（sentence-level BLEU）
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
# 句子级 BLEU 建议开启 effective_order=True：
# 短句往往凑不满 4-gram，开启后会按实际可得的最高阶 n-gram 计算，避免分数被强制拉成 0
bleu = BLEU(tokenize='13a', effective_order=True)

# 逐句计算 BLEU：sentence_score(单条系统输出, 该句对应的多套参考)
for i, hyp in enumerate(sys):
    # 收集第 i 句在所有参考译文中的对应句子，作为该句的参考列表
    sent_refs = [ref_set[i] for ref_set in refs]
    score = bleu.sentence_score(hyp, sent_refs)

    print(f"第 {i + 1} 句译文:", hyp)
    print("  参考译文:", sent_refs)
    print("  BLEU 分数:", score.score)
    print("  各 n-gram 精度:", score.precisions)
    print("  长度惩罚 BP:", score.bp) # bp = math.exp( 1 - ref_len / sys_len) # 译文偏短，惩罚越大
    print("  系统长度 / 参考长度:", score.sys_len, "/", score.ref_len)
    print()
