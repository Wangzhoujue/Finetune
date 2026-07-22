import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
from config import (
    MODEL_PATH,
    OUTPUT_MODEL_PATH,
    FINAL_MODEL_DIR
)

# 路径配置
base_model_path = MODEL_PATH
lora_path = FINAL_MODEL_DIR
output_path = OUTPUT_MODEL_PATH

print("正在加载基模型...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="cuda",  # 使用GPU加载
    trust_remote_code=True
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(
    base_model,
    lora_path,
    device_map="auto"
)

print("正在合并LoRA...")
merged_model = model.merge_and_unload()

print(f"Saving merged model to {output_path}...")
os.makedirs(output_path, exist_ok=True)
merged_model.save_pretrained(output_path, safe_serialization=True)

print("Saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained(output_path)

print("✅ 合并成功")