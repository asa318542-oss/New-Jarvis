import torch

print("CUDA disponibile:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

x = torch.rand(1000, 1000).cuda()
y = torch.rand(1000, 1000).cuda()

z = torch.matmul(x, y)
print("Test completato su GPU")
