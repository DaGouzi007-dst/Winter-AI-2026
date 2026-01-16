import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 1. 准备“翻译器” (Transform)
# 图片是 0-255 的像素点，我们要把它变成 0-1 之间的张量(Tensor)，方便 AI 吃
transform = transforms.Compose([
    transforms.ToTensor(),
])

# 2. 下载并加载数据
print("正在下载 MNIST 数据集 (可能会有点慢，耐心等待)...")

# 训练集：用来学习 (60,000张)
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# 测试集：用来考试 (10,000张)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)


# print("数据加载完成！")
# print(f"训练集数量: {len(train_dataset)} 张")
# print(f"测试集数量: {len(test_dataset)} 张")



model =nn.Sequential(
    nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(64*7*7,128),
    nn.ReLU(),
    nn.Linear(128,10),
)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)
print("开始训练")
epochs=3
for epoch in range(epochs):
    for i ,(images,labels) in enumerate(train_loader):
        outputs=model(images)
        loss=criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%100==0:
            print(f"轮次[{epoch+1}/{epochs}],步数[{i+1}/{len(train_loader)}],Loss:{loss.item():.4f}")

print("训练完成")


print("\n正在进行期末考试")

model.eval()

correct=0
total=0
with torch.no_grad():
    for images,labels in  test_loader:
        outputs =model(images)
        _,predicted=torch.max(outputs.data,1)
        total+=labels.size(0)
        correct +=(predicted==labels).sum().item()
accuracy=100*correct/total
print(f"考试结束")
print(f'准确率：{accuracy:.2f}%')
