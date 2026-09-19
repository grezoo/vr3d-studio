import torch

print("=== HARDWARE CHECK ===")
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Count:", torch.cuda.device_count())
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("VRAM Total (GB):", round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2))
    # Quick tensor test on GPU
    x = torch.ones((1000, 1000), device="cuda")
    y = x @ x
    print("CUDA Matrix Multi Math Test:", "PASSED" if y[0, 0].item() == 1000 else "FAILED")
else:
    print("Warning: Running on CPU!")
print("=======================")
