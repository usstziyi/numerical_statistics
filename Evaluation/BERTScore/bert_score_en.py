from evaluate import load

"""
BERTScore

任务：
中文 -> 英文翻译

source:
    中文原文

predictions:
    模型生成的英文译文

references:
    人工参考英文译文

BERTScore实际上只使用：
    predictions + references

不会使用 source
"""

# 加载 BERTScore
bertscore = load("bertscore")


# 中文原文
source = [
    "猫坐在垫子上。",
    "今天天气非常好。",
    "那个孩子正在公园里跑步。",
]


# 模型生成的英文译文
predictions = [
    "The cat sits on the mat.",
    "The weather is very nice today.",
    "The child is running in the park.",
]


# 人工参考译文
references = [
    "The cat is sitting on the mat.",
    "The weather is very good today.",
    "The kid is running in the park.",
]


# 计算 BERTScore
results = bertscore.compute(
    predictions=predictions,
    references=references,
    # model_type="roberta-large",
    lang="en"
)

print("Hashcode:", results["hashcode"])
"""
Hashcode: roberta-large_L17_no-idf_version=0.3.12(hug_trans=4.57.6)
它是本次打分配置的 指纹 ——记录“用哪把尺子量的分”。
roberta-large 用的模型 L17 取第 17 层隐状态 
no-idf 没开 IDF 加权（idf=True 时会变成_idf） 
version=0.3.12 bert_score 库版本 
hug_trans=4.57.6 transformers 版本
"""


print("Precision:", results["precision"])
print("Recall:", results["recall"])
print("F1:", results["f1"])




# 计算整体平均分
mean_precision = sum(results["precision"]) / len(results["precision"])
mean_recall = sum(results["recall"]) / len(results["recall"])
mean_f1 = sum(results["f1"]) / len(results["f1"])


print("\n平均 Precision:", mean_precision)
print("平均 Recall:", mean_recall)
print("平均 F1:", mean_f1)