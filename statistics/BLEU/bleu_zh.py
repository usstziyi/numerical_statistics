from sacrebleu.metrics import BLEU
"""
Bilingual Evaluation Understudy
双语评估替代指标
语料级 BLEU（corpus-level BLEU）
"""

# 系统输出（模型生成的中文译文）
sys = [
    "狗咬了那个人。",
    "这并不令人惊讶。",
    "猫在垫子上。",
]

# 参考译文
refs = [
    ["狗咬了那个人。", "这并不意外。", "垫子上有一只猫。"],
    ["那只狗咬了那个人。", "没有人感到惊讶。", "猫蹲在垫子上。"],
]
# 中文必须指定 tokenize='zh'
bleu = BLEU(tokenize='zh')
score = bleu.corpus_score(sys, refs)

print(score)
print("BLEU 分数:", score.score)
print("各 n-gram 精度:", score.precisions)
print("长度惩罚 BP:", score.bp)
print("系统长度 / 参考长度:", score.sys_len, "/", score.ref_len)