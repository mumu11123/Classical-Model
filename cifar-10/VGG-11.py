#小卷积核+深层堆叠==真正意义上的深层卷积神经网络模型
#容易过拟合，需要强数据增强和Dropout
#计算量大，尤其是512通道
#VGG-11
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
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


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VGG().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

EPOCHS = 10
train_loss_list = []
test_acc_list = []#用于记录每个epoch的训练损失和测试准确率，便于后面画曲线

for epoch in range(EPOCHS):
    running_loss = 0.0
    model.train()#启用dropout与batch统计
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)#前向传播
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(trainloader)
    train_loss_list.append(avg_loss)#记录平均损失
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct+=predicted.eq(labels).sum().item()
    acc = correct / total*100
    test_acc_list.append(acc)
    print(f"Epoch [{epoch + 1}/{EPOCHS}] Loss: {avg_loss:.4f}, Test Acc: {acc:.2f}%")

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(train_loss_list)
plt.title("Training Loss")

plt.subplot(1,2,2)
plt.plot(test_acc_list)
plt.title("Test Accuracy")

plt.show()
