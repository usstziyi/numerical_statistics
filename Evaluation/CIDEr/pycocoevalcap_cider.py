from pycocoevalcap.cider.cider import Cider

gts = {
    0: [
        "a cat is sitting on the mat",
        "a cat sits on the mat",
        "there is a cat on the mat"
    ],
    1: [
        "a dog is running in the grass",
        "a dog runs through the grass",
        "there is a dog running on the grass"
    ]
}

res = {
    0: ["a cat is sitting on the mat"],
    1: ["a dog is running on the grass"]
}

scorer = Cider()

score, scores = scorer.compute_score(gts, res)

print("CIDEr:", score)
print("每个样本的 CIDEr:", scores)