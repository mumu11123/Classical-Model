#cnn
import torch
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt
import matplotlib
import torchvision

LR=0.001
BATCH_SIZE=15
EPOCHS=5
DOWNLOAD_MNIST=True
#随机种子，让每次初始化都相同
torch.manual_seed(1)

class CNN(nn.Module):
     def __init__(self):
         super(CNN, self).__init__()
         #第一个卷积->激励函数->池化
         self.conv1 =nn.Sequential(
            nn.Conv2d(
                in_channels=1,#输入通道
                out_channels=16,#输出通道
                kernel_size=5,#卷积核大小
                stride=1,#步长
                padding=2,#扩充
            ),
             nn.ReLU(),
             nn.MaxPool2d(2,2),
             #此时输出大小为（16，14，14）
         )
         self.conv2 = nn.Sequential(
             nn.Conv2d(16,32,5,1,2),
             nn.ReLU(),
             nn.MaxPool2d(2, 2),
             # 此时输出大小为（16，14，14）
         )
         #全连接层
         self.out=nn.Linear(32*7*7,10)

     def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        #把每个批次每个输入都拉成一个维度
        x=x.view(x.size(0),-1)
        output=self.out(x)
        return output
def main():
    #准备数据
    train_data=torchvision.datasets.MNIST(
        root="./data",
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=DOWNLOAD_MNIST
    )
    test_data=torchvision.datasets.MNIST(
        root="./data",
        train=False
    )
    train_loader = data.DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)
    test_x=test_data.data.unsqueeze(1).float()[:2000]/255.0
    test_y=test_data.targets.numpy()[:2000]
    #模型初始化
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn=CNN().to(device)
    print(cnn)
    #优化器
    optimizer=torch.optim.Adam(cnn.parameters(),lr=LR)
    #损失函数
    loss_func=nn.CrossEntropyLoss()

    #训练
    for epoch in range(EPOCHS):
        for step, (b_x, b_y) in enumerate(train_loader):
            b_x,b_y=b_x.to(device),b_y.to(device)
            optimizer.zero_grad()
            output=cnn(b_x)
            loss=loss_func(output,b_y)
            loss.backward()
            optimizer.step()
            #每50个梯度输出一次
            if step%50==0:
                test_output=cnn(test_x.to(device))
                pred_y=torch.argmax(test_output,dim=1)
                acc=float((pred_y.cpu()==test_y).sum().item())/test_y.size(0)
                print(f'Epoch:{epoch} | train Loss:{loss.item():.4f} | accuracy:{acc:.2f}')
    #保存模型
    torch.save(cnn.state_dict(),'cnn2.pkl')
#测试
    cnn.eval()
    #把多个张量打包成一个数据集对象
    test_dataset=data.TensorDataset(test_x,test_y)
    test_loader=data.DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=True)
    correct=0
    total=0
    with torch.no_grad():
        for tx, ty in test_loader:
            tx, ty = tx.to(device), ty.to(device)
            output = cnn(tx)
            pred = torch.argmax(output, dim=1)
            correct += (pred == ty).sum().item()
            total += ty.size(0)

    accuracy = correct / total
    print(f"\n测试集整体准确率: {accuracy * 100:.2f}%")

    # 显示前 32 张图片的预测结果
    inputs = test_x[:32].to(device)
    labels = test_y[:32].to(device)

    with torch.no_grad():
        test_output = cnn(inputs)
        pred_y = torch.max(test_output, 1)[1].cpu().numpy()

    # 显示预测结果
    fig, axes = plt.subplots(4, 8, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(inputs[i][0].cpu().numpy(), cmap='gray')
        ax.set_title(f"Pred: {pred_y[i]}", fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    # 可视化第一层卷积核（16个 filters
    kernels = cnn.conv1[0].weight.data  # shape: [16,1,5,5]
    plt.figure(figsize=(10, 3))
    for i in range(16):
        plt.subplot(2, 8, i + 1)
        plt.imshow(kernels[i, 0].cpu().numpy(), cmap='gray')
        plt.axis('off')
    plt.suptitle('第一层卷积核（Conv1 filters）')
    plt.show()
if __name__ == "__main__":
    main()