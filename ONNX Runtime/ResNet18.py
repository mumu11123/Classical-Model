import torch
import torch.nn as nn
import torchvision.models as models

#1.加载官方 ResNet18
model = models.resnet18(weights=None)

#2.修改最后的全连接层 => CIFAR-10 只有 10 个类别
model.fc = nn.Linear(model.fc.in_features, 10)

#3.进入推理模式
model.eval()
print(model)
#4.构造一个 CIFAR-10 尺寸的虚拟输入
dummy = torch.randn(1, 3, 32, 32)

#5.导出为 ONNX
torch.onnx.export(
    model,#要导出的模型
    dummy,#示例输入
    "resnet18_cifar10.onnx",#输出的文件名
    input_names=["input"],#输入节点名称
    output_names=["output"],
    opset_version=13,
    dynamic_axes={"input": {0: "batch"}}#指定动态维度
)

print("ONNX 导出成功！")