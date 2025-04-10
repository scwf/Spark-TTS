import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version used by PyTorch: {torch.version.cuda}")
    print(f"Number of GPUs detected by PyTorch: {torch.cuda.device_count()}")
    print(f"Current GPU detected by PyTorch: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}") # 尝试获取 GPU 0 的名字