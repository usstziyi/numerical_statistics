import evaluate
from pycocoevalcap.cider.cider import Cider

# ========================
# 相同的数据
# ========================

predictions = [
    "a cat is sitting on the mat",
    "a dog is running on the grass"
]

references = [
    [
        "a cat is sitting on the mat",
        "a cat sits on the mat",
        "there is a cat on the mat"
    ],
    [
        "a dog is running in the grass",
        "a dog runs through the grass",
        "there is a dog running on the grass"
    ]
]

# ========================
# evaluate: sunhill/cider
# ========================

cider = evaluate.load("sunhill/cider")

results = cider.compute(
    predictions=predictions,
    references=references
)

print("sunhill:", results["cider_score"])


# ========================
# pycocoevalcap
# ========================

gts = {
    i: references[i]
    for i in range(len(references))
}

res = {
    i: [predictions[i]]
    for i in range(len(predictions))
}

scorer = Cider()

score, scores = scorer.compute_score(gts, res)

print("pycocoevalcap:", score)
print("每个样本:", scores)

# ========================
# 对比缩放
# ========================

print(
    "sunhill × 10:",
    results["cider_score"] * 10
)