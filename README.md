# Qwen3-0.6B QLoRA 微调

基于 **Hugging Face** 生态的 **Qwen3-0.6B** QLoRA 微调示例项目。

采用目前主流的大模型微调方案：

- 🤗 Transformers
- 🚀 TRL（SFTTrainer）
- 🎯 PEFT（LoRA）
- ⚡ QLoRA（4-bit）
- 📦 Datasets
- 🛠️ Accelerate
- 📋 uv 包管理

---
## 环境准备
uv sync
### 注意torch需要下载GPU版本
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

## 项目结构

```text
qwen_lora/
├── data/
│   ├── train-00000-of-00001.parquet
│   └── test-00000-of-00001.parquet
├── output/
├── train.py
├── inference.py
├── pyproject.toml
├── uv.lock
├── README.md
└── .gitignore