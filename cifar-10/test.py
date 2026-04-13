import torch

model = torch.jit.load("alexnet_mobile.torchscript.pt")
model.eval()

x = torch.randn(1, 3, 32, 32)
out = model(x)

print("输出 shape =", out.shape)
print("输出 =", out)
