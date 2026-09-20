import os

# NLTK 新版的 SSRF 防护会拒绝通过代理下载数据；
# 本机需要走代理访问 GitHub，这里显式放行（官方提供的开关）
os.environ["NLTK_ALLOW_PROXIED_URLOPEN"] = "1"

from nltk.translate.meteor_score import meteor_score
import nltk


# ============================
# 下载资源（第一次运行需要）
# METEOR 的三级匹配机制里，前两级 精确匹配 和 词干匹配 （sitting vs sits ，用 Porter 词干器，不需要额外数据）都能开箱即用，
# 只有第三级 同义词匹配 需要这两个资源。没有它们的话，Example 3 那种同义词句子会得 0 分左右，其余例子不受影响。
# 自动级联， 优先级固定：精确 > 词干 > 同义词 ，不需要（也没法）手动指定等级。
# meteor_score 内部的同义词匹配是 写死的英语查询 （`wordnet.synsets(word)` ，不带`lang` 参数），不会走 omw 的中文数据。
# METEOR 就留给英语场景用。
# ============================

nltk.download("wordnet")
nltk.download("omw-2.0")



# ============================
# Example 1:
# 普通词匹配
# ============================

reference = [
    "the",
    "cat",
    "is",
    "on",
    "the",
    "mat"
]

candidate = [
    "the",
    "cat",
    "is",
    "on",
    "the",
    "mat"
]


score = meteor_score(
    [reference],
    candidate
)

print("Example 1 METEOR:")
print(score)




# ============================
# Example 2:
# 词形变化
# sitting vs sits
# ============================

reference = [
    "the",
    "cat",
    "is",
    "sitting",
    "on",
    "the",
    "mat"
]


candidate = [
    "the",
    "cat",
    "sits",
    "on",
    "the",
    "mat"
]

"""
meteor_score([ref1, ref2], candidate)
# 等价于
max(
    meteor_score([ref1], candidate),
    meteor_score([ref2], candidate)
)
"""


score = meteor_score(
    [reference], # 参考答案，注意外面套了一层列表
    candidate # 模型输出
)


print("\nExample 2 METEOR:")
print(score)



# ============================
# Example 3:
# 同义词
# car vs automobile
# ============================


reference = [
    "the",
    "car",
    "is",
    "fast"
]


candidate = [
    "the",
    "automobile",
    "is",
    "quick"
]


score = meteor_score(
    [reference],
    candidate
)


print("\nExample 3 METEOR:")
print(score)



# ============================
# Example 4:
# 多句评价（论文常用）
# ============================

references = [
    [
        "a",
        "dog",
        "runs",
        "in",
        "the",
        "park"
    ],

    [
        "the",
        "cat",
        "is",
        "sleeping",
        "on",
        "the",
        "bed"
    ],

    [
        "a",
        "man",
        "is",
        "playing",
        "guitar"
    ]
]


predictions = [
    [
        "a",
        "dog",
        "running",
        "in",
        "the",
        "park"
    ],

    [
        "the",
        "cat",
        "sleeps",
        "on",
        "the",
        "bed"
    ],

    [
        "a",
        "person",
        "plays",
        "guitar"
    ]
]


scores = []

for ref, pred in zip(references, predictions):

    score = meteor_score(
        [ref],
        pred
    )

    scores.append(score)



print("\nSentence METEOR:")
for s in scores:
    print(s)



# corpus METEOR

mean_score = sum(scores) / len(scores)

print("\nAverage METEOR:")
print(mean_score)