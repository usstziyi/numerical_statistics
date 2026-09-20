import os

# HF evaluate 的 meteor 内部调用 nltk（word_tokenize 需要 punkt_tab）；
# NLTK 新版的 SSRF 防护会拒绝通过代理下载数据，本机走代理，这里显式放行
os.environ["NLTK_ALLOW_PROXIED_URLOPEN"] = "1"

import evaluate
import nltk

# evaluate 的 meteor 脚本只自动下载 wordnet/omw，不下载分词器
nltk.download("punkt_tab", quiet=True)


# ==========================
# Load METEOR metric
# ==========================

meteor = evaluate.load("meteor")


# ==========================
# Example 1
# 完全一致
# ==========================

predictions = [
    "The cat is on the mat."
]

references = [
    "The cat is on the mat."
]


result = meteor.compute(
    predictions=predictions,
    references=references
)

print("Example 1")
print(result)



# ==========================
# Example 2
# 词形变化
# sitting vs sits
# ==========================

predictions = [
    "The cat sits on the mat."
]

references = [
    "The cat is sitting on the mat."
]


result = meteor.compute(
    predictions=predictions,
    references=references
)


print("\nExample 2")
print(result)



# ==========================
# Example 3
# 多句评价
# ==========================

predictions = [
    "A dog is running in the park.",
    "The cat is sleeping on the bed.",
    "A man is playing guitar."
]


references = [
    "A dog runs in the park.",
    "The cat sleeps on the bed.",
    "A person plays guitar."
]


result = meteor.compute(
    predictions=predictions,
    references=references
)


print("\nExample 3 Corpus METEOR")
print(result)