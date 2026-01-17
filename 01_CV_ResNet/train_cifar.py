import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

class ResidualBlock(nn.Module):
    def __init__(self, in_channels,out_channels,stride=1):
        super().__init__()
        self.main_path=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut=nn.Sequential()
        if stride!=1 or in_channels!=out_channels:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels)
                )
    
    def forward(self,x):
        out=self.main_path(x)
        out+=self.shortcut(x)
        out=nn.functional.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels=64
        self.conv1=nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1=self._make_layer(64,2,stride=1)
        self.layer2=self._make_layer(128,2,stride=2)
        self.layer3=self._make_layer(256,2,stride=2)
        self.layer4=self._make_layer(512,2,stride=2)
        self.avg_pool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(512,10)


    def _make_layer(self,out_channels,num_blocks,stride):
        strides=[stride]+[1]*(num_blocks-1)
        layers=[]
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels,out_channels,stride))
            self.in_channels=out_channels
        return nn.Sequential(*layers)

    def forward(self,x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        return out
    

if __name__ == "__main__":
    transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    print("\n准备下载。。。")

    train_dataset=torchvision.datasets.CIFAR10(root='./data', 
                                            train=True, 
                                            download=True, 
                                            transform=transform)
    train_loader=torch.utils.data.DataLoader(train_dataset,
                                        batch_size=128,
                                        shuffle=True)

    test_dataset=torchvision.datasets.CIFAR10(root='./data',
                                            train=False,
                                            download=True,
                                            transform=transform)
    test_loader=torch.utils.data.DataLoader(test_dataset,
                                            batch_size=100,
                                            shuffle=True)

    classes=('plane','car','bird','cat','deer',
            'dog','frog','horse','ship','truck')

    print('-'*30)
    print("CIFAR-10数据已就位")
    data_iter=iter(train_loader)
    images,lables=next(data_iter)
    print(f" 数据形状 (Batch Shape): {images.shape}")
    print(f" 单张图片尺寸: {images[0].shape}") 
    print("-" * 30)

    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training on {device}")

    model =ResNet().to(device)
    criterion=nn.CrossEntropyLoss()
    optimizer =optim.SGD(model.parameters(),lr=0.01,momentum=0.9,weight_decay=5e-4)


    print("开始训练。。。")
    total_epochs=10

    for epoch in range(total_epochs):
        model.train()
        running_loss=0.0
        for i ,(images,lables) in enumerate(train_loader):
            images,lables=images.to(device),lables.to(device)
            optimizer.zero_grad()
            outputs=model(images)
            loss=criterion(outputs,lables)
            loss.backward()
            optimizer.step()

            running_loss+=loss.item()

            if (i+1)%100==0:
                print(f"Epoch [{epoch+1}/{total_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    print("training finished")

    PATH = './resnet_cifar.pth'
    torch.save(model.state_dict(),PATH)
    print(f"模型已经保存到{PATH}")


    print("开始测试。。。")
    correct=0
    total=0

    with torch.no_grad():
        for images,lables in test_loader:
            images,lables =images.to(device),lables.to(device)
            outputs=model(images)
            _,predicted=torch.max(outputs,1)
            total+=lables.size(0)
            correct+=(predicted==lables).sum().item()

    print(f'🏆 最终成绩 (在 10,000 张测试图上的准确率): {100 * correct / total:.2f}%')

    class_correct=list(0. for i in range(10))
    class_total=list(0. for i in range(10))
    with torch.no_grad():
        for images,lables in test_loader:
            images,lables=images.to(device),lables.to(device)
            outputs=model(images)
            _,predicted=torch.max(outputs,1)
            c=(predicted==lables).squeeze()
            for i in range(len(lables)):
                lable =lables[i]
                class_correct[lable]+=c[i].item()
                class_total[lable]+=1

    print("-" * 20)
    for i in range(10):
        print(f'类别 {classes[i]:<5} 的准确率: {100 * class_correct[i] / class_total[i]:.2f}%')
    print("-" * 20)