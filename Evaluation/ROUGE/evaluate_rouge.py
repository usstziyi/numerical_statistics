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