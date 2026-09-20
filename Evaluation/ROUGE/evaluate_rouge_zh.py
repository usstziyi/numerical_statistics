import evaluate
import jieba

rouge = evaluate.load("rouge")

predictions = ["猫在垫子上"]
references = ["猫坐在垫子上"]

results = rouge.compute(
    predictions=predictions,
    references=references,
    tokenizer=jieba.lcut
)

for key, value in results.items():
    print(f"{key}: {value:.4f}")