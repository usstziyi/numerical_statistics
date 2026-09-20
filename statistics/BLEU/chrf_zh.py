from sacrebleu.metrics import CHRF

"""
chrF: character n-gram F-score
字符级 n-gram F 分数
语料级 chrF（corpus-level chrF）
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

# chrF 直接基于字符计算，中日韩等无空格语言无需指定分词器
# 默认参数即论文推荐配置：
# char_order=6 统计 1~6 元字符 n-gram
# word_order=0 表示纯 chrF；改为 2 即为 chrF++（额外统计 1~2 元词 n-gram）
# beta=2 表示召回率的权重是精确率的 4 倍
chrf = CHRF(char_order=6, word_order=0, beta=2)
score = chrf.corpus_score(sys, refs)

print(score)
print("chrF 分数:", score.score)
print("字符 n-gram 阶数:", score.char_order)
print("词 n-gram 阶数:", score.word_order)
print("beta（召回权重）:", score.beta)
