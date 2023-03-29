import torch
<<<<<<< HEAD

=======
import matplotlib.pyplot as plt
import numpy as np
import math

img = plt.imread("photo.jpg")
img = np.array(img)
img = torch.Tensor(img)
img = img.permute(2,0,1 ).contiguous()
print(img.shape)

#卷积
cov2 = torch.nn.Conv2d(3,1,(3,3),stride=1,padding=1,bias=False)
out = cov2(img)
# plt.imshow(img)
# plt.imshow(out)
print(out)
#激活
relu = torch.nn.ReLU()
out  = relu(out)
print(out.shape)
#最大池化
# max_pool = torch.nn.MaxPool2d(2)
# out = max_pool(out)
# print(out.shape)
#展成1维
lin = torch.nn.Flatten()
out = lin(out)
print(out.shape)
print(f"out[0]:{out[0][0]}")
#softmax
max = torch.max(out)
out = out/max
out = torch.exp(out)

sum = torch.sum(out)
print(f"sum:{sum}")
out = out/sum
# print(out)
print(f"out[0]:{out[0][0]}")
#crossentropy
i = 5
crossentropy = -1*torch.log(out[0][i])
print(crossentropy)
>>>>>>> f9c3b4d (trainmodel)
