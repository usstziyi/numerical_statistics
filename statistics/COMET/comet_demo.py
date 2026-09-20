"""
COMET Metric Demo

source:
    原始输入语言

mt:
    Machine Translation
    模型生成结果

ref:
    Human Reference
    人工参考答案
"""


from comet import download_model, load_from_checkpoint


# =====================================================
# 1. 准备测试数据
# =====================================================
sources = [
    "The cat is sitting on the mat.",
    "I love artificial intelligence.",
    "The weather is very good today."
]


# 模型生成结果
hypotheses = [
    "猫坐在垫子上。",
    "我喜欢人工智能。",
    "今天的天气很好。"
]


# 人工参考翻译
references = [
    "猫正在垫子上坐着。",
    "我热爱人工智能。",
    "今天天气非常好。"
]



# =====================================================
# 2. 下载 COMET 模型
# =====================================================
model_path = download_model("Unbabel/wmt22-comet-da")


# =====================================================
# 3. 加载模型
# =====================================================
model = load_from_checkpoint(model_path)


# =====================================================
# 4. 构造 COMET 输入格式
# =====================================================
data = []
for src, hyp, ref in zip(sources, hypotheses, references):
    sample = {
        "src": src, # 原始句子
        "mt": hyp, # 模型输出
        "ref": ref, # 人工参考
    }
    data.append(sample)

# 查看输入
print("\nInput data:")
for item in data:
    print(item)

# =====================================================
# 5. COMET 推理
# =====================================================
print("\nCalculating COMET...")
output = model.predict(
    data,
    batch_size=4,
    gpus=0
)

# =====================================================
# 6. 查看结果
# =====================================================
print("\n==============================")
print("Sentence level COMET score")
print("==============================")


for i, score in enumerate(output.scores):
    print()
    print("Source:")
    print(sources[i])
    print("Hypothesis:")
    print(hypotheses[i])
    print("Reference:")
    print(references[i])
    print(
        "COMET score:",
        score
    )

# =====================================================
# 7. 查看整体系统分数
# =====================================================
print("\n==============================")
print("System COMET score")
print("==============================")

print(
    output.system_score
)