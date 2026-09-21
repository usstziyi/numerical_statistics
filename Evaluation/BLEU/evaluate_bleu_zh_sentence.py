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

for i, (prediction, reference) in enumerate(
    zip(predictions, references)
):
    result = bleu.compute(
        predictions=[prediction],
        references=[reference],
        tokenize="zh",
        # 句级 BLEU 建议开启：截断到译文实际拥有的最高 n-gram 阶（totals==0 的阶），
        # 避免短句（<4 token）把不存在的阶的 0 精度计入几何平均；译文 ≥4 token 时开关无差别
        use_effective_order=True
    )

    print(f"第 {i + 1} 句:")
    print(f"prediction: {prediction}")
    print(f"BLEU: {result['score']:.4f}")
    print(f"counts: {result['counts']}")
    print(f"totals: {result['totals']}")
    print(f"precisions: {result['precisions']}")
    print(f"bp: {result['bp']:.4f}")
    print(f"sys_len: {result['sys_len']}")
    print(f"ref_len: {result['ref_len']}")
    print()