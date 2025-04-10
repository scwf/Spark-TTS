# 在你的 Conda 环境 (sparktts) 中运行 Python
from huggingface_hub import snapshot_download
import os

local_dir = "pretrained_models/Spark-TTS-0.5B"
print(f"尝试下载模型到: {local_dir}")
os.makedirs(local_dir, exist_ok=True) # 确保目录存在

try:
    snapshot_download(
        "SparkAudio/Spark-TTS-0.5B",
        local_dir=local_dir,
        local_dir_use_symlinks=False # 在 Windows 上设为 False 通常更可靠
    )
    print("模型下载完成。")
except Exception as e:
    print(f"模型下载失败: {e}")
    # 这里可以考虑退出脚本或给出进一步指示