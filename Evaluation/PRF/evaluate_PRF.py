import evaluate

metric = evaluate.combine([
    "accuracy",
    "precision",
    "recall",
    "f1"
])

"""
真实世界里有 100 封邮件，其中：
- 20 封是真正的垃圾邮件
- 80 封是正常邮件
模型预测了 15 封是垃圾邮件。
其中：
- 12 封确实是垃圾邮件
- 3 封其实是正常邮件
另外还有：
- 8 封垃圾邮件没有被模型发现
"""

references = [1] * 20 + [0] * 80

predictions = (
    [1] * 12 +
    [0] * 8 +
    [1] * 3 +
    [0] * 77
)


results = metric.compute(
    predictions=predictions,
    references=references
)

for key, value in results.items():
    print(f"{key}: {value:.4f}")

