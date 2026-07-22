import torch
import swanlab
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
#Swanlab:
#实验跟踪与可视化（在 Web 仪表盘上实时生成图表）
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
        "enable_thinking": False   #关闭训练时的思考模式
    }
)
tokenizer.pad_token = tokenizer.eos_token

#######################################################
# 数据集（Dataset）
#######################################################

dataset = load_dataset(
    "json", #数据集类型
    data_files=DATASET_PATH, #数据集路径
    split="train",  #用于标识这部分数据的用途
)

#######################################################
# QLoRA
#######################################################

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # 启用 4-bit 量化加载
    bnb_4bit_quant_type="nf4",      # 量化类型：Normal Float 4
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算精度
    bnb_4bit_use_double_quant=True  # 使用双重量化
)

#######################################################
# 基模型（Base Model）
#######################################################

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto", #设备映射（GPU/CPU）
    torch_dtype=torch.bfloat16, #模型计算时的默认数据类型
)
model.config.use_cache = False  #训练时不使用 KV Cache
model.enable_input_require_grads() #强制模型的输入张量（input_ids）需要梯度

#######################################################
# LoRA
#######################################################

peft_config = LoraConfig(
    task_type="CAUSAL_LM",        # 任务类型
    r=16,                         # LoRA 秩（低秩矩阵的维度）
    lora_alpha=32,                # 缩放因子
    lora_dropout=0.05,            # Dropout 概率
    bias="none",                  # 偏置参数处理方式
    target_modules=[              # 应用 LoRA 的目标模块
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

#######################################################
# 训练配置（Train Config）
#######################################################

args = SFTConfig(

    output_dir=OUTPUT_DIR,

    learning_rate=1e-4,  #学习速率

    num_train_epochs=3,  #训练轮数

    per_device_train_batch_size=1,  #每个设备的训练批次大小

    gradient_accumulation_steps=16,  #梯度累积步数

    logging_steps=10,  #日志记录步数

    save_steps=200,  #保存步数

    save_total_limit=2,  #保存的总限制

    bf16=True,  #使用 BF16 精度

    gradient_checkpointing=True,  #使用梯度检查点

    max_length=1024,  #最大长度

    packing=False,  #是否打包

    optim="paged_adamw_32bit", ##优化器类型

    lr_scheduler_type="cosine",##学习率调度器类型

    warmup_ratio=0.03, #预热比例

    report_to="swanlab", #报告到 swanlab
)

#######################################################
# Trainer
#######################################################

trainer = SFTTrainer(
    model=model,                    # 量化后的基础模型
    args=args,                      # 训练参数配置
    processing_class=tokenizer,     # 分词器
    train_dataset=dataset,          # 训练数据集
    peft_config=peft_config,        # LoRA 配置
)

trainer.train()

trainer.save_model(FINAL_MODEL_DIR)