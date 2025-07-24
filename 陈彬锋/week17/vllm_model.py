from vllm import LLM, SamplingParams

# 1. 初始化模型和采样参数
sampling_params = SamplingParams(
    temperature=0.8,  # 控制生成随机性 (0-1)
    top_p=0.95,       # 核采样概率阈值
    max_tokens=100    # 最大生成token数量
)

model_name = "google/gemma-2b-it"

# 2. 加载模型
# dtype="auto"会自动选择最佳数据类型，也可显式指定为"float16"或"bfloat16"
llm = LLM(model=model_name, dtype="auto")

# 3. 准备输入提示
prompts = [
    "Deep learning is",
    "The future of artificial intelligence",
    "Machine learning applications include",
    "The capital of France is"
]

# 4. 生成文本
outputs = llm.generate(prompts, sampling_params)

# 5. 输出结果
print("\n文本生成结果:")
for i, output in enumerate(outputs):
    generated_text = output.outputs[0].text
    print(f"提示 {i+1}: {output.prompt!r}")
    print(f"生成文本: {generated_text!r}\n")
    print("-" * 80)