import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU is not available to PyTorch in this virtual environment.")
    print("Install the CUDA build of PyTorch inside .venv if you want GPU support.")
