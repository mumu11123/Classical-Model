#用MLP实现手写识别
import torch
from torch.utils.data import DataLoader#用于加载数据
from torchvision import transforms
from torchvision.datasets import MNIST#pytorch提供的标准数据集接口
import matplotlib.pyplot as plt#绘制图像，可视化结果
BATCH_SIZE = 15
LR=0.001
#神经网络的主体
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #四个全连接层
        self.fc1 = torch.nn.Linear(28 * 28, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 10)
    #定义前向传播过程
    def forward(self, x):
        #先做全连接线性计算，再套上一个激活函数
        x=torch.nn.functional.relu(self.fc1(x))
        x=torch.nn.functional.relu(self.fc2(x))
        x=torch.nn.functional.relu(self.fc3(x))
        x=torch.nn.functional.log_softmax(self.fc4(x),dim=1)#对十个数字的预测概率
        return x

def get_data_loader(is_train):
    #transforms.Compose用于组合多个图像预处理操作的工具
    #transforms.ToTensor()意思是加载数据时将转换为张量
    to_tensor=transforms.Compose([transforms.ToTensor()])
    #加载数据
    data_set=MNIST("",is_train,transform=to_tensor,download=True)
    #每个批次加载十五个样本，每次加载数据时打乱顺序，避免让模型记住顺序影响训练效果
    return DataLoader(data_set,batch_size=BATCH_SIZE,shuffle=True)
#计算测试集上的准确率
def evaluate(test_data,net):
    n_correct = 0#正确预测的数量
    n_total = 0#总样本数
    #禁用梯度计算，避免计算过程中不必要的内存开销，因为只需要进行推理而非训练。
    with torch.no_grad():
        for (x,y) in test_data:
            #展开是因为全连接层的输入需要一维向量
            outputs=net.forward(x.view(-1,28*28))
            for i,output in enumerate(outputs):#遍历时同时获得元素值和索引
                if torch.argmax(output)==y[i]:
                    n_correct += 1
                n_total += 1
    return n_correct/n_total*100

def main():

    train_data=get_data_loader(True)
    test_data=get_data_loader(False)
    net=Net()
    #初始准确率
    print("initial accuracy:",evaluate(test_data,net),"%")
    #Adam优化器，学习率为0.001
    #lr太大不稳定，太小收敛慢
    optimizer=torch.optim.Adam(net.parameters(),lr=LR)
    #训练过程
    for epoch in range(2):
        #每批15，小批量训练循环
        for (x,y) in train_data:
            net.zero_grad()#把之前一次训练计算的梯度清零，因为pytorch会自动累积梯度
            output=net.forward(x.view(-1,28*28))
            loss=torch.nn.functional.nll_loss(output,y)#负对数似然损失
            loss.backward()#反向传播
            optimizer.step()#用梯度更新参数
        print("epoch:",epoch,"accuracy:",evaluate(test_data,net),"%")
    for(n,(x,_)) in enumerate(test_data):
        if n>3:
            break
        predict=torch.argmax(net.forward(x[0].view(-1,28*28)))#取出预测值最大的类别
        plt.figure(n)
        plt.imshow(x[0].view(28,28),cmap='grey')#把图片恢复成二维并展示
        plt.title(f"prediction: {int(predict)}")
    plt.show()
if __name__=="__main__":
    main()