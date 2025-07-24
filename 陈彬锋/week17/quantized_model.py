from accelerate import init_empty_weights
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch

# 配置量化参数
bnb_quantization_config = BnbQuantizationConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# 创建空模型框架
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(
        AutoModelForCausalLM.config_class.from_pretrained("google/gemma-2b-it")
    )

# 加载并量化模型
quantized_model = load_and_quantize_model(
    model,
    weights_location="google/gemma-2b-it",
    bnb_quantization_config=bnb_quantization_config,
    device_map="auto"
)

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")

# 生成文本
prompt = "China is famous for"
inputs = tokenizer(prompt, return_tensors="pt").to(quantized_model.device)
outputs = quantized_model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))