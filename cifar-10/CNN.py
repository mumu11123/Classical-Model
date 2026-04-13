
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
#CIFAR-10每张图片3*32*32，RGB
#数据加载
transform = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomCrop(32, padding=4),transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset=torchvision.datasets.CIFAR10(root='./data', train=True,download=True,transform=transform)
trainloader=torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset=torchvision.datasets.CIFAR10(root='./data', train=False,download=True,transform=transform)
testloader=torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

dataiter=iter(trainloader)
images,labels=next(dataiter)
print(images.shape)
print(labels.shape)

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1,padding=1)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.flatten=nn.Flatten()
        self.fc1=nn.Linear(32*8*8,128)
        self.fc2=nn.Linear(128,10)
    def forward(self,x):
       x=self.pool(torch.relu(self.conv1(x)))
       x=self.pool(torch.relu(self.conv2(x)))
       x=x.view(-1,32*8*8)
       x=torch.relu(self.fc1(x))
       x=self.fc2(x)
       return x
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=CNN().to(device)
'''
常用于分类任务（multi-class）。在 PyTorch 中它期望模型输出 raw logits（未经过 softmax），
目标 labels 为整数类别（LongTensor）。内部实现是 log_softmax + nll_loss。
'''
criterion=nn.CrossEntropyLoss()
'''
params：model.parameters()，网络所有可学习参数（权重与偏置）。
lr：学习率，控制梯度更新步长。0.01 是常见起点。
momentum=0.9：动量项，用来加速 SGD、减小震荡。
weight_decay=5e-4：L2 正则化（等价于参数更新时对参数施加衰减），有助于泛化、减少过拟合。
'''
optimizer=optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
EPOCHS=20
for epoch in range(EPOCHS):
    running_loss=0.0
    model.train()
    for images, labels in trainloader:
        images=images.to(device)
        labels=labels.to(device)

        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels)#张量
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()#提取张量数值，计算平均损失
    avg=running_loss/len(trainloader)#batch数
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg:.4f}")
correct=0
total=0
model.eval()

with torch.no_grad():
    for images, labels in testloader:
        images=images.to(device)
        labels=labels.to(device)
        outputs=model(images)
        _, predicted = torch.max(outputs.data, 1)
        total+=labels.size(0)
        correct+=predicted.eq(labels).sum().item()
print(f"Accuracy: {100*correct/total:.2f}%")



