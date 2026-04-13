#调用GPU
import torch
from torch.utils.data import DataLoader  # 用于加载数据
from torchvision import transforms
from torchvision.datasets import MNIST  # pytorch提供的标准数据集接口
import matplotlib.pyplot as plt  # 绘制图像，可视化结果


# 神经网络的主体
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 四个全连接层
        self.fc1 = torch.nn.Linear(28 * 28, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, 256)
        self.fc4 = torch.nn.Linear(256, 256)

    # 定义前向传播过程
    def forward(self, x):
        # 先做全连接线性计算，再套上一个激活函数
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)  # 对十个数字的预测概率
        return x


def get_data_loader(is_train):
    # transforms.Compose用于组合多个图像预处理操作的工具
    # transforms.ToTensor()意思是加载数据时将转换为张量
    to_tensor = transforms.Compose([transforms.ToTensor()])
    # 加载数据
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    # 每个批次加载十五个样本，每次加载数据时打乱顺序，避免让模型记住顺序影响训练效果
    return DataLoader(data_set, batch_size=10, shuffle=True)


# 计算测试集上的准确率
def evaluate(test_data, net, device):
    n_correct = 0  # 正确预测的数量
    n_total = 0  # 总样本数
    # 禁用梯度计算，避免计算过程中不必要的内存开销，因为我们只需要进行推理而非训练。
    with torch.no_grad():
        for (x, y) in test_data:
            # 将数据移动到设备
            x, y = x.to(device), y.to(device)
            # 展开是因为全连接层的输入需要一维向量
            outputs = net.forward(x.view(-1, 28 * 28))
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    return n_correct / n_total * 100


def main():
    # 选择设备：优先使用GPU，如果没有则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")

    train_data = get_data_loader(True)
    test_data = get_data_loader(False)
    net = Net()

    # 将模型移动到设备（GPU或CPU）
    net = net.to(device)
    print(f"模型当前设备: {next(net.parameters()).device}")

    # 初始准确率
    print("initial accuracy:", evaluate(test_data, net, device), "%")

    # Adam优化器，学习率为0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    # 训练过程
    for epoch in range(2):
        # 每批15，小批量训练循环
        for (x, y) in train_data:
            # 将数据移动到设备
            x, y = x.to(device), y.to(device)

            net.zero_grad()  # 把之前一次训练计算的梯度清零，因为pytorch会自动累积梯度
            output = net.forward(x.view(-1, 28 * 28))
            loss = torch.nn.functional.nll_loss(output, y)  # 负对数似然损失
            loss.backward()  # 反向传播
            optimizer.step()  # 用梯度更新参数

        accuracy = evaluate(test_data, net, device)
        print("epoch:", epoch, "accuracy:", accuracy, "%")

    # 可视化部分（需要将模型移回CPU）
    net_cpu = net.cpu()  # 将模型移回CPU进行可视化
    for (n, (x, _)) in enumerate(test_data):
        if n > 3:
            break
        predict = torch.argmax(net_cpu.forward(x[0].view(-1, 28 * 28)))  # 取出预测值最大的类别
        plt.figure(n)
        plt.imshow(x[0].view(28, 28), cmap='grey')  # 把图片恢复成二维并展示
        plt.title(f"prediction: {int(predict)}")
    plt.show()


if __name__ == "__main__":
    main()