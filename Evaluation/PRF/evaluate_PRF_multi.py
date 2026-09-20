import evaluate

metric = evaluate.combine([
    "precision",
    "recall",
    "f1"
])

# 真实标签
references = [
    0, 0, 0, 0,      # 4只猫
    1, 1, 1, 1,      # 4只狗
    2, 2, 2, 2       # 4只兔子
]

# 模型预测
predictions = [
    0, 0, 1, 0,      # 猫：3个预测正确，1个错成狗
    1, 1, 2, 1,      # 狗：3个预测正确，1个错成兔子
    2, 0, 2, 2       # 兔子：3个预测正确，1个错成猫
]

results = metric.compute(
    predictions=predictions,
    references=references,
    average="macro" # 每个类别先单独算指标，再平均
)

print(results)