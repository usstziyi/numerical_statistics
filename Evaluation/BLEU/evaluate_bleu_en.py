import evaluate

bleu = evaluate.load("sacrebleu")

# 系统输出（模型生成的译文）
predictions = [
    "The dog bit the man.",
    "It wasn't surprising.",
    "The cat is on the mat.",
]

# 参考译文：内层列表 = 该句子的所有参考译文（与 predictions 一一对应）
references = [
    # 第 1 句的参考译文
    [
        "The dog bit the man.",
        "The dog had bit the man.",
    ],

    # 第 2 句的参考译文
    [
        "It was not unexpected.",
        "No one was surprised.",
    ],

    # 第 3 句的参考译文
    [
        "There is a cat on the mat.",
        "A cat is on the mat.",
    ],
]

results = bleu.compute(
    predictions=predictions,
    references=references,
    tokenize="13a"
)

for key, value in results.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")