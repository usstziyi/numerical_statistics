import evaluate

rouge = evaluate.load("rouge")

predictions = [
    "the cat is on the mat",
    "the dog is running"
]

references = [
    "the cat sat on the mat",
    "the dog is running fast"
]

results = rouge.compute(
    predictions=predictions,
    references=references
)


for key, value in results.items():
    print(f"{key}: {value:.4f}")

"""
rouge1: 0.8611    # 单词重叠程度
rouge2: 0.7286    # 连续两个词重叠程度
rougeL: 0.8611    # 最长公共子序列 LCS 重叠程度
rougeLsum: 0.8611 # 面向多句摘要的ROUGE-L 重叠程度
"""