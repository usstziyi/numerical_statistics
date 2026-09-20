from evaluate import load

comet_metric = load('comet')
# 也可以指定具体模型，例如：load('comet', 'Unbabel/wmt22-comet-da')

source = ["Dem Feuer konnte Einhalt geboten werden", "Schulen und Kindergärten wurden eröffnet."]
hypothesis = ["The fire could be stopped", "Schools and kindergartens were open"]
reference = ["They were able to control the fire.", "Schools and kindergartens opened"]

results = comet_metric.compute(predictions=hypothesis, references=reference, sources=source)
print(results['scores'])      # 句子级 COMET 分数
print(results['mean_score'])  # 系统级平均分
