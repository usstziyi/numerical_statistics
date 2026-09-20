from sacrebleu.metrics import CHRF

chrf = CHRF()

score = chrf.corpus_score(
    ["猫在垫子上。"],
    [["猫坐在垫子上。"]]
)

print(score)
print(type(score))