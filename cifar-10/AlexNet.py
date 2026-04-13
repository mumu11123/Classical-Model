#原版第一层是
#nn.Conv2d(3,96,kernel=11,stride=4,padding=1)
#重点：Dropout 在训练时随机丢弃神经元（比例 p=0.5 常用），强制网络不要过度依赖单个神经元，从而提高泛化。
#在 PyTorch 中，model.train() 时 Dropout 有效，model.eval() 时自动禁用——你不用手动控制。
#AlexNet 原版在两个全连接层之间用了 Dropout(0.5)。
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
#可视化数据增强后的图像
#训练可视化，损失和精确度
#第一层卷积核可视化
#数据加载
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

trainset=torchvision.datasets.CIFAR10(root='./data', train=True,download=True,transform=train_transform)
trainloader=torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset=torchvision.datasets.CIFAR10(root='./data', train=False,download=True,transform=test_transform)
testloader=torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

import matplotlib.pyplot as plt
import numpy as np

def unnorm(t):
    return (t * 0.5 + 0.5).clamp(0,1)   # 反归一化

# 显示一批增强后的图像
images, labels = next(iter(trainloader))

plt.figure(figsize=(10,4))
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(unnorm(images[i]).permute(1,2,0).cpu().numpy())
    plt.axis("off")
plt.suptitle("Augmented CIFAR-10 Images")
plt.show()

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(#8层
            nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            #inplace指原地操作可以节省空间，直接修改x
            nn.Conv2d(64,192,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(192,384,3,1,1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384,256,3,1,1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256,256,3,1,1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256*4*4,4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,10),
        )
    def forward(self, x):
        x=self.features(x)
        x=self.flatten(x)
        x=self.classifier(x)
        return x

def visualize_conv1_filters(model):
    weights = model.features[0].weight.data.cpu()  # [64,3,3,3]

    plt.figure(figsize=(10, 8))
    for i in range(32):  # 只显示前 32 个
        w = weights[i]
        w = (w - w.min()) / (w.max() - w.min())  # 归一化
        w = w.permute(1,2,0)  # 变成 HWC
        plt.subplot(4, 8, i+1)
        plt.imshow(w)
        plt.axis("off")
    plt.suptitle("Conv1 filters")
    plt.show()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model=AlexNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
EPOCHS=20
train_loss_list = []
test_acc_list = []

for epoch in range(EPOCHS):
    running_loss = 0.0
    model.train()

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(trainloader)
    train_loss_list.append(avg_loss)

    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # 修正：使用更清晰的语法
    acc = 100 * correct / total
    test_acc_list.append(acc)

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}, Test Acc: {acc:.2f}%")

# 可视化训练结果
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(range(1, EPOCHS+1), train_loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("Training Loss")

plt.subplot(1,2,2)
plt.plot(range(1, EPOCHS+1), test_acc_list)
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title("Test Accuracy")

plt.tight_layout()
plt.show()

# 可视化第一层卷积核
#visualize_conv1_filters(model)

# ===== 保存模型权重 =====
#torch.save(model.state_dict(), "alexnet_cifar10.pth")
#print("模型已保存：alexnet_cifar10.pth")
#example = torch.randn(1, 3, 32, 32).to(device)

#traced = torch.jit.trace(model, example)
#torch.jit.save(traced, "alexnet_mobile.pt")
#print("TorchScript 导出成功 alexnet_mobile.pt")
