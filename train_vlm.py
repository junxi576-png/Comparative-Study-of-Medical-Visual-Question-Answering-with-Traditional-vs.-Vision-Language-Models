import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig
# 如果报错请运行 | If error occurs, run: pip install datasets
from datasets import load_dataset 

# 1. 加载模型与处理器 (利用 RTX 显卡进行 4-bit 量化)
# Load model and processor (utilizing RTX GPU for 4-bit quantization)
model_id = "llava-hf/llava-1.5-7b-hf"

# 配置 4-bit 量化以显著降低显存占用 | Configure 4-bit quantization to reduce VRAM usage
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=torch.float16
)

print("正在从 Hugging Face 加载模型... | Loading model from Hugging Face...")
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    quantization_config=bnb_config, 
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)

# 2. 配置 LoRA (微调核心：低秩适配)
# Configure LoRA (Fine-tuning core: Low-Rank Adaptation)
# r=16: 秩，决定可训练参数量 | Rank, determines the number of trainable parameters
# target_modules: 指定微调注意力机制中的哪些权重 | Specify which weights in the attention mechanism to fine-tune
config = LoraConfig(
    r=16, 
    lora_alpha=32, 
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)

# 将 LoRA 适配器应用到基础模型 | Apply LoRA adapter to the base model
model = get_peft_model(model, config)

# 3. 准备数据格式提示 | Data Format Reminders
# 提示：VLM 需要 Prompt 引导 | Note: VLMs require specific prompt guidance
# 例如: "USER: <image>\nWhat is shown in this image?\nASSISTANT:"
# Example: "USER: <image>\nWhat is shown in this image?\nASSISTANT:"

print("\n--- 模型状态 Model Status ---")
print("VLM 模型已准备就绪。 | VLM model is ready.")
print("由于大模型微调代码较长，建议先确认模型能否成功加载。 | Suggest verifying model load before full training.")

# 打印可训练参数百分比 | Print the percentage of trainable parameters
model.print_trainable_parameters()