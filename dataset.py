import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T#常用数据预处理工具
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
matplotlib.rcParams['axes.unicode_minus'] = False
'''
导入库
创建numpydataset类，实现__init__,获取样本和标签，返回样本总数
噪声类noise=torch.randn_like(img)*0.1
主程序：
假数据，设置假数据的形式
设置完整的数据增强管道（transform
创建dataset和dataloader
测试batch_imgs,batch_labels=next(iter(dataloader))
可视化
'''
class NumpyDataset(Dataset):
    def __init__(self,images,labels,transform=None):
        assert len(images)==len(labels)
        self.images = images
        self.labels = labels
        self.transform = transform
    #返回样本总数
    def __len__(self):
        return len(self.images)
    #取出第idx个样本
    def __getitem__(self, idx):
        img = self.images[idx]
        label = int(self.labels[idx])
        if self.transform:
            img = self.transform(img)  # 这里 transform 会把 numpy 转 tensor
            return img, label

            # 手动把 numpy 转 tensor
        img = torch.from_numpy(img).float() / 255.0

        if img.ndim == 2:
            img = img.unsqueeze(0)  # H,W → 1,H,W
        elif img.ndim == 3:
            img = img.permute(2, 0, 1)  # H,W,C → C,H,W

        return img, label
class AddNoise():#加噪声
    def __call__(self,img):
        noise=torch.randn_like(img)*0.1
        return img+noise

def demo():#生成假数据并测试dataset
    N=100
    H,W=28,28
    images=(np.random.rand(N,H,W)*255).astype(np.uint8)
    labels=np.random.randint(0,10,size=(N,))
    #mnist
    transform=T.Compose([#数据增强
                         T.ToPILImage(),
                         T.RandomRotation(20),#随机旋转+-20
                         T.RandomAffine(0,translate=(0.1,0.1)),#随机平移
                         T.ToTensor(),
                         AddNoise(),#加噪声
                         T.Normalize((0.5,),(0.5,)),#灰度图常用的归一化
                         ])
    '''
    RGB图片
    transform=T.Compose([
        T.Resize(size=(224,224)),
        T.randomHorizontalFlip(),
        T.ColorJitter(0.1, 0.1, 0.1, 0.1),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    '''
    dataset = NumpyDataset(images, labels, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    batch_imgs, batch_labels = next(iter(loader))#取第一个batch
    print("Batch images shape:", batch_imgs.shape)  # -> [B, C, H, W]
    print("Batch labels shape:", batch_labels.shape)
    img, lbl = dataset[0]
    print(type(img))  # <class 'torch.Tensor'>
    print(img.shape)  # torch.Size([1, 28, 28])
    img0 = batch_imgs[0].squeeze(0).numpy()#把维度为1的删掉，因为pytorch的图片是CHW，灰度图只有一条通道
    # 查看增强前后
    img_raw = images[0]
    img_aug, _ = dataset[0]
#plt.subplot(1, 2, 1)：左边的子图（显示原图）
#plt.subplot(1, 2, 2)：右边的子图（显示增强后的图）
    plt.subplot(1, 2, 1)
    plt.imshow(img_raw, cmap='gray')
    plt.title("原图")

    plt.subplot(1, 2, 2)
    plt.imshow(img_aug.squeeze(0), cmap='gray')
    plt.title("增强后")
    plt.show()


if __name__ == "__main__":
    demo()