import numpy as np
import torchvision
from torch import nn
import torch
from  PIL import Image

# 把这个模型拿过来 防止模型加载的时候报错
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.model(x)
        return x




image = Image.open('cifar10test_data/plane.jpg')
print(image)  #<PIL.PngImagePlugin.PngImageFile image mode=RGBA size=719x719 at 0x1BB943224C0>
#  这里可以看到输出是ARGB类型，四通道，而我们的训练模式都是三通道的。
#  所以这里使转换成RGB三通道的格式
#
image = image.convert('RGB')

# 使用Compose组合改变数据类型,先变成32*32的 然后在变成tensor类型
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
model = torch.load('model_pytorch/model_10.pth')  # 这里面输出的话就是保存的各种参数。
image = torch.reshape(image,(1,3,32,32))
print(image.shape)

model.eval()
with torch.no_grad():
    image = image.cuda()
    output = model(image)
    print(output)
print(output.argmax(1))


