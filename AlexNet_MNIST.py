import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

LR = 0.001
BATCH_SIZE = 64
EPOCHS = 3
#数据增强
train_transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
])

test_transform = transforms.ToTensor()

train_data = torchvision.datasets.MNIST("./data", train=True,
                                        transform=train_transform,
                                        download=True)

test_data = torchvision.datasets.MNIST("./data", train=False,
                                       transform=test_transform)

train_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


class AlexNetMNIST(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 2 * 2, 256),
            nn.ReLU(True),

            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out


def train_and_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNetMNIST().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()

    train_losses = []
    test_accs = []

    for epoch in range(EPOCHS):
        model.train()
        for step, (bx, by) in enumerate(train_loader):
            bx, by = bx.to(device), by.to(device)

            optimizer.zero_grad()
            output = model(bx)
            loss = loss_func(output, by)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            if step % 100 == 0:
                print(f"Epoch[{epoch}] Step[{step}] Loss:{loss.item():.4f}")

        #每个 epoch 计算一次测试精度
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                pred = torch.argmax(out, dim=1)
                correct += (pred == by).sum().item()
                total += by.size(0)

        acc = correct / total
        test_accs.append(acc)
        print(f"Epoch[{epoch}] Test Accuracy: {acc:.4f}")

    # =============================
    #Loss 曲线
    plt.figure(figsize=(8,4))
    plt.plot(train_losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()

    # =============================
    # Accuracy 曲线
    plt.figure(figsize=(6,4))
    plt.plot(test_accs, marker='o')
    plt.title("Test Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()

    # =============================
    # 预测图像
    sample_x, sample_y = next(iter(test_loader))
    sample_x, sample_y = sample_x[:32].to(device), sample_y[:32]

    with torch.no_grad():
        pred = torch.argmax(model(sample_x), dim=1).cpu()

    sample_x = sample_x.cpu()

    plt.figure(figsize=(10,5))
    for i in range(32):
        plt.subplot(4, 8, i+1)
        plt.imshow(sample_x[i][0], cmap='gray')
        plt.title(f"{pred[i]}")
        plt.axis("off")
    plt.show()


if __name__ == "__main__":
    train_and_test()
