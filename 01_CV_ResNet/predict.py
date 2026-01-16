import torch
import torchvision.transforms as transforms
from PIL import Image
import sys


try:
    from train_cifar import ResNet
except ImportError:
    print("导入错误，找不到ResNet类")
    sys.exit()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

transform=transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

def predict_imag(image_path,model_path):
    print(f"正在读取图片：{image_path}")

    try:
        img=Image.open(image_path)
        if img.mode !="RGB":
            img=img.convert('RGB')
    except:
        print("抄不到图片，请检查路径")
        return
    
    img_tensor=transform(img)
    img_tensor=img_tensor.unsqueeze(0).to(device)

    print(f"正在加载，模型权重：{model_path}")
    model=ResNet().to(device)

    try:
        model.load_state_dict(torch.load(model_path,map_location=device))
    except:
        print("权重加载错误，请检查路径")
        return
    
    model.eval()

    with torch.no_grad():
        outputs=model(img_tensor)
        probs=torch.nn.functional.softmax(outputs,dim=1)
        confidence,predicted=torch.max(probs,1)
        idx=predicted.item()
        score=confidence.item()

        print("\n"+"="*30)
        print(f"预测结果：{classes[idx]}(信心：{score:.2%})")


if __name__=="__main__":
    my_model_path="resnet_cifar.pth"
    my_image_path="test.jpg"
    predict_imag(my_image_path,my_model_path)

    