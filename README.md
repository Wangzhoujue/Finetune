# Qwen3-4B QLoRA 微调

基于 **Hugging Face** 生态的 **Qwen3-4B** QLoRA 微调示例项目。

采用目前主流的大模型微调方案：

- 🤗 Transformers
- 🚀 TRL（SFTTrainer）
- 🎯 PEFT（LoRA）
- ⚡ QLoRA（4-bit）
- 📦 Datasets
- 🛠️ Accelerate
- 📋 uv 包管理
- Swanlab可视化参数


## 项目结构

```text
qwen_lora/
├── data/
│   ├── train-00000-of-00001.parquet
│   └── test-00000-of-00001.parquet
├── output/
├── train.py    #训练入口
├── pretrain.py    #再训练
├── merge_lora.py    #合并LoRA适配器 
├── pyproject.toml   
├── uv.lock
├── README.md
└── .gitignore
```

---
## 环境准备
```text
#powershell 运行下载依赖
uv sync
```
### 注意：torch需要下载GPU版本
```
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
### 安装完输入
```
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```
### 若出现类似输出即可开始微调：
```
PyTorch: 2.11.0+cu128
CUDA可用: True
GPU: Your GPU Name
```
---

## 下载模型
### 运行``load_model.py``即可从Modelscope下载Qwen3_4B
---

## 微调前准备

### 在powershell输入登录并粘贴你的 API Key
```
swanlab login
```
### 在训练中访问```https://swanlab.cn```即可看到可视化图表
训练过程中，所有指标都会实时同步到你的 SwanLab 项目（Oroject_v1）中。你可以登录 https://swanlab.cn 查看：
📈 训练曲线：Loss、学习率、梯度范数等随时间的变化

📋 参数记录：你配置的 lr、epoch、lora_r、batch_size 等

💻 系统资源：GPU 利用率、显存占用、训练速度

📝 训练日志：每一步的详细输出

📁 模型检查点：训练过程中保存的 checkpoints（在 ./output/ 目录下）

### 运行 ``train.py``即开始微调
等待模型训练完成得到LoRA 适配器文件
可选择在模型部署时加载LoRA适配器，部署时类似指令即可
```bash
vllm serve meta-llama/Qwen3-4B-Merged \
    --enable-lora \
    --lora-modules sql-lora=output/final
```
### 运行``merge_lora.py``可合并 LoRA 适配器到基础模型
这样过后部署新保存的模型即可使用微调后的模型

### 若运行效果没达到预期可运行``pretrain.py``进行再次训练


