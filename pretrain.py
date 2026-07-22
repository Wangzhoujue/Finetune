import torch
import swanlab
from peft import PeftModel
from config import (
    MODEL_PATH,
    DATASET_PATH,
    OUTPUT_DIR,
    FINAL_MODEL_DIR,
    SWANLAB_PROJECT,
    SWANLAB_EXPERIMENT_NAME,
)

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
)

from trl import (
    SFTTrainer,
    SFTConfig,
)

#######################################################
# 初始化实验
#######################################################

run = swanlab.init(
    project=SWANLAB_PROJECT,
    experiment_name=SWANLAB_EXPERIMENT_NAME,
    config={
        "model": MODEL_PATH,
        "lr": 1e-4,
        "epoch": 3,
        "batch_size": 1,
        "lora_r": 16,
        "lora_alpha": 32,
    }
)
#######################################################
# 分词器（Tokeniser)
#######################################################

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    chat_template_kwargs={
        "enable_thinking": False
    }
)
tokenizer.pad_token = tokenizer.eos_token

#######################################################
# 数据集（Dataset）
#######################################################

dataset = load_dataset(
    "json",
    data_files=DATASET_PATH,
    split="train",
)

#######################################################
# QLoRA
#######################################################

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

#######################################################
# 基模型（Base Model）
#######################################################

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

model = PeftModel.from_pretrained(
    model,
    FINAL_MODEL_DIR,   # 已有 LoRA 路径
    is_trainable=True # 很重要，允许继续训练
)
model.config.use_cache = False
model.enable_input_require_grads()


#######################################################
# 训练配置（Train Config）
#######################################################

args = SFTConfig(

    output_dir=FINAL_MODEL_DIR,

    learning_rate=1e-4,

    num_train_epochs=3,

    per_device_train_batch_size=1,

    gradient_accumulation_steps=16,

    logging_steps=10,

    save_steps=200,

    save_total_limit=2,

    bf16=True,

    gradient_checkpointing=True,

    max_length=1024,

    packing=False,

    optim="paged_adamw_32bit",

    lr_scheduler_type="cosine",

    warmup_ratio=0.03,

    report_to="swanlab",
)

#######################################################
# Trainer
#######################################################


def formatting_func(example):

    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False
    )

    text = text.replace(
        "<think>\n\n</think>\n",
        ""
    )

    return text

print(formatting_func(dataset[0]))

trainer = SFTTrainer(

    model=model,

    args=args,

    processing_class=tokenizer,

    train_dataset=dataset,


)
trainer.train(resume_from_checkpoint=True)

trainer.save_model(FINAL_MODEL_DIR)