from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch

def setup_advanced_vlm():
    model_id = "llava-hf/llava-1.5-7b-hf"
    
    # 4-bit quantization configuration (saves GPU memory and time)
    # 4位量化配置（节约显存和时间）
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        quantization_config=bnb_config, 
        device_map="auto"
    )

    # LoRA configuration: fine-tune only a small number of parameters [cite: 154]
    # LoRA配置：仅微调少量参数 [cite: 154]
    config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, config)
    return model, processor