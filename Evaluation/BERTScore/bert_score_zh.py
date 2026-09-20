from evaluate import load

"""
BERTScore

任务：
英文 -> 中文翻译

source:
    英文原文

predictions:
    模型生成的中文译文

references:
    人工参考中文译文
"""

bertscore = load("bertscore")


# 英文原文
source = [
    "The cat is sitting on the mat.",
    "The weather is very nice today.",
    "The child is running in the park.",
]


# 模型生成的中文译文
predictions = [
    "猫坐在垫子上。",
    "今天天气很好。",
    "孩子正在公园里跑步。",
]


# 人工参考中文译文
references = [
    "一只猫坐在垫子上。",
    "今天的天气非常好。",
    "那个小孩正在公园中奔跑。",
]


# 注意：
# predictions 和 references 都是中文
# 所以 lang="zh"

results = bertscore.compute(
    predictions=predictions,
    references=references,
    lang="zh"
)


print("Precision:", results["precision"]) # 精确率
print("Recall:", results["recall"]) # 召回率
print("F1:", results["f1"]) # F1 分数

# Hashcode: bert-base-chinese_L8_no-idf_version=0.3.12(hug_trans=4.57.6)
print("Hashcode:", results["hashcode"])


# 数据集平均 BERTScore
mean_precision = sum(results["precision"]) / len(results["precision"])
mean_recall = sum(results["recall"]) / len(results["recall"])
mean_f1 = sum(results["f1"]) / len(results["f1"])


print("\n平均 Precision:", mean_precision)
print("平均 Recall:", mean_recall)
print("平均 F1:", mean_f1)