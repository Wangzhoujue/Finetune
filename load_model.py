from modelscope import snapshot_download
from config import  MODEL_PATH

model_dir = snapshot_download('Qwen/Qwen3-4B', local_dir=MODEL_PATH)
print(f"模型下载至: {model_dir}")