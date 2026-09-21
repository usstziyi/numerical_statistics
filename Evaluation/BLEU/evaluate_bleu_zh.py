import evaluate

bleu = evaluate.load("sacrebleu")

# 系统输出（模型生成的中文译文）
predictions = [
    "狗咬了那个人。",
    "这并不令人惊讶。",
    "猫在垫子上。",
]

# 参考译文：内层列表 = 该句子的所有参考译文（与 predictions 一一对应）
references = [
    # 第 1 句的参考译文
    ["狗咬了那个人。", "那只狗咬了那个人。"],

    # 第 2 句的参考译文
    ["这并不意外。", "没有人感到惊讶。"],

    # 第 3 句的参考译文
    ["垫子上有一只猫。", "猫蹲在垫子上。"],
]

results = bleu.compute(
    predictions=predictions,
    references=references,
    tokenize="zh"
)

for key, value in results.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")