import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
from PIL import Image
import matplotlib.pyplot as plt
import os

# 配置路径 | Path Configuration
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
LORA_PATH = "./vqa_final_model"
# 请确保路径指向实际存在的医学图像 | Ensure the path points to an actual medical image
TEST_IMAGE = "data/VQA_RAD Image Folder/synpic19118.jpg" 

# 1. 使用 4-bit 量化加载基础模型 | Load base model with 4-bit quantization
# 这有助于节省显存并避免某些 ValueError | Helps save VRAM and avoid potential ValueErrors
print("正在以 4-bit 模式加载模型... | Loading model in 4-bit mode...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

base_model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID, 
    quantization_config=bnb_config, 
    device_map="auto"
)

# 2. 加载 LoRA 适配层与处理器 | Load LoRA adapter and processor
print("正在加载 LoRA 权重... | Loading LoRA weights...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)
processor = AutoProcessor.from_pretrained(MODEL_ID)

# 3. 准备推理数据 | Prepare data for inference
if not os.path.exists(TEST_IMAGE):
    print(f"❌ 找不到图片: {TEST_IMAGE} | Image not found at the specified path.")
else:
    image = Image.open(TEST_IMAGE).convert("RGB")
    # LLaVA 需要特定的 Prompt 模板：USER: <image>\nQuestion\nASSISTANT:
    # LLaVA requires a specific prompt template
    prompt = "USER: <image>\nWhat abnormality is present in this image?\nASSISTANT:"
    
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")

    # 4. 模型推理生成回答 | Generate answer via model inference
    print("🚀 模型正在生成回答... | Generating response...")
    with torch.inference_mode():
        output = model.generate(
            **inputs, 
            max_new_tokens=100, 
            do_sample=False # 使用贪婪搜索以确保医疗问答的稳定性 | Use Greedy Search for deterministic medical answers
        )
    
    # 5. 解码并提取答案 | Decode and extract the answer
    full_response = processor.decode(output[0], skip_special_tokens=True)
    # 提取 ASSISTANT 之后的内容 | Extract content following "ASSISTANT:"
    answer = full_response.split("ASSISTANT:")[-1].strip()

    print(f"\n--- 推理结果 Inference Result ---\n问题 Question: What abnormality is present?\n模型回答 Answer: {answer}\n")

    # 6. 结果可视化与保存 | Visualization and saving results
    plt.figure(figsize=(10, 7))
    plt.imshow(image)
    plt.title(f"VQA Case Study\nPredict: {answer}", fontsize=12, pad=15)
    plt.axis('off')
    plt.savefig('inference_result.png', bbox_inches='tight')
    plt.show()
    print("✅ 结果图已保存为 inference_result.png | Result image saved.")