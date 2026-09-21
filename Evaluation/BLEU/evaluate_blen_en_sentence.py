import evaluate

bleu = evaluate.load("sacrebleu")

predictions = [
    "The dog bit the man.",
    "It wasn't surprising.",
    "The cat is on the mat.",
]

references = [
    [
        "The dog bit the man.",
        "The dog had bit the man.",
    ],
    [
        "It was not unexpected.",
        "No one was surprised.",
    ],
    [
        "There is a cat on the mat.",
        "A cat is on the mat.",
    ],
]

for i, (prediction, reference) in enumerate(
    zip(predictions, references)
):
    result = bleu.compute(
        predictions=[prediction],
        references=[reference],
        tokenize="13a",
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