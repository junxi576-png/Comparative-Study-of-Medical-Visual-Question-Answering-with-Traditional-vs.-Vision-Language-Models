import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from PIL import Image
import pandas as pd
import os

# 1. 配置 | Configuration
IMG_DIR = "data/VQA_RAD Image Folder"
TRAIN_CSV = "data/train.csv"
MODEL_ID = "llava-hf/llava-1.5-7b-hf"

# 2. 加载模型：使用 4-bit 量化以节省显存 | Load Model: Using 4-bit quantization to save VRAM
print("正在加载模型... | Loading model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=torch.float16
)
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID, 
    quantization_config=bnb_config, 
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

# 3. LoRA 配置：只微调特定的投影层 | LoRA Configuration: Fine-tune specific projection layers
# r=16: 秩大小，平衡训练参数量与效果 | Rank size, balances params vs performance
lora_config = LoraConfig(
    r=16, 
    lora_alpha=32, 
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 4. 数据准备 | Data Preparation
# 确保所有内容为字符串并处理缺失值 | Ensure strings and handle missing values
df = pd.read_csv(TRAIN_CSV).astype(str).replace('nan', 'missing')
train_dataset = Dataset.from_list(df.to_dict(orient="records"))

# 5. 核心修复：修改数据整理函数 | Core Fix: Update Data Collator
def collate_fn(batch):
    """
    将图像和文本处理为模型可接受的 Tensor
    Process images and text into tensors acceptable by the model
    """
    # 遵循 LLaVA 的 Prompt 模板 | Follow LLaVA Prompt Template
    texts = [f"USER: <image>\n{item['question_clean']}\nASSISTANT: {item['answer_clean']}" for item in batch]
    images = [Image.open(os.path.join(IMG_DIR, item['image_name'])).convert("RGB") for item in batch]
    
    # 关键点：LLaVA 的图像会被编码为 576 个 Token
    # Key point: LLaVA images are encoded into 576 tokens
    inputs = processor(
        text=texts, 
        images=images, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=800  # 576 (image) + extra space for text
    )
    
    # labels 用于计算交叉熵损失 | Labels used for calculating Cross-Entropy loss
    inputs["labels"] = inputs["input_ids"].clone()
    return inputs

# 6. 训练参数设置 | Training Arguments
training_args = TrainingArguments(
    output_dir="./llava-vqa-results",
    per_device_train_batch_size=1,      # 减小 Batch Size 防止显存溢出 | Reduce BS to prevent OOM
    gradient_accumulation_steps=8,      # 梯度累积维持等效 BS=8 | Gradient accumulation for effective BS=8
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,                          # 开启半精度加速 | Enable mixed precision
    logging_steps=5,
    save_strategy="epoch",
    remove_unused_columns=False,        # 必须设为 False 以保留图像数据 | Must be False to keep image data
    report_to="none"
)

# 7. 初始化训练器 | Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn
)

print("🚀 重新启动微调 (已修复 Token 匹配问题)... | Restarting fine-tuning...")


try:
    trainer.train()
    # 保存微调后的 LoRA 适配器 | Save the fine-tuned LoRA adapter
    model.save_pretrained("./vqa_final_model")
    print("✅ 训练完成！ | Training Complete!")
except Exception as e:
    print(f"❌ 错误 Error: {e}")