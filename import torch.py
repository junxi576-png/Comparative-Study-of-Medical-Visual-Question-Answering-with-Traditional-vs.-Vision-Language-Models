import torch
from transformers import BitsAndBytesConfig

# 检查显卡
print(f"GPU: {torch.cuda.get_device_name(0)}")

# 尝试定义量化配置
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
print("量化配置加载成功，你可以进行本地大模型微调了！")