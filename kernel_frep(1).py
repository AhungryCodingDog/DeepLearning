from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np

import matplotlib.pyplot as plt
import pylab
import cv2
import numpy as np

img = plt.imread("photo.jpg")                        #在这里读取图片
img1 = plt.imread("photo.jpg")                        #在这里读取图片
#img1 = torch.randn(1,5,5)
img1 = torch.Tensor(img1)
img1 = img1.permute(2,0,1 ).contiguous()
conv1 = nn.Conv2d(3,3,3,stride=1,padding=1)
img_conv1 = conv1(img1)
print(img_conv1.shape)



# print(img_conv1)
#
# # conv1.parameters()
# kernel_param = conv1.weight
# print(kernel_param[1][1])
# print(conv1.weight)
# # print(img1)
# # print(img1_conv1)




# plt.imshow(img)                                     #显示读取的图片
# pylab.show()

# fil_1= np.array([[ -1,-1, 0],                        #这个是设置的滤波，也就是卷积核
#                 [ -1, 0, 1],
#                 [  0, 1, 1]])
fil_1= np.array([[0, -4, 1],                        #这个是设置的滤波，也就是卷积核
                 [-4, 16, -4],
                 [0, -4, 0]])
fil_2= np.array([[1, 2, 1],                        #这个是设置的滤波，也就是卷积核
                 [0, 0, 0],
                 [-1, -2, -1]])
fil_3= np.array([[1, 0, -1],                        #这个是设置的滤波，也就是卷积核
                 [2, 0, -2],
                 [1, 0, -1]])
# fil_2= np.array([[ -1,-0.2, 0],                        #这个是设置的滤波，也就是卷积核
#                 [ -0.2, 0, 0.2],
#                 [  0, 0.2, 1]])
# fil_2 = fil_1*0.5
# print(fil_2)
res_1 = cv2.filter2D(img,-1,fil_1)                      #使用opencv的卷积函数
res_2 = cv2.filter2D(img,-1,fil_2)                      #使用opencv的卷积函数
res_3 = cv2.filter2D(img,-1,fil_3)                      #使用opencv的卷积函数

ax = plt.subplot(2, 2,1)  # 共25个子图，i+1是对应的位置，i=0,对应的位置是1
ax.imshow(img)  #
plt.axis('off')
ax = plt.subplot(2, 2,2)  # 共25个子图，i+1是对应的位置，i=0,对应的位置是1
ax.imshow(res_1)  #
plt.axis('off')
ax = plt.subplot(2, 2, 3)  # 共25个子图，i+1是对应的位置，i=0,对应的位置是1
ax.imshow(res_2)  #
plt.axis('off')
ax = plt.subplot(2, 2, 4)  # 共25个子图，i+1是对应的位置，i=0,对应的位置是1
ax.imshow(res_3)  #

# plt.title('r_%s' % (correct_target[i].item()))  #
plt.axis('off')

# plt.figure(1)
# plt.imshow(res_1)                                     #显示卷积后的图片
# plt.figure(2)                                         #显示卷积后的图片
# plt.imshow(res_2)                                     #显示卷积后的图片
# plt.figure(3)                                         #显示卷积后的图片
# plt.imshow(res_3)                                     #显示卷积后的图片

# plt.imsave("res_1.jpg",res)


#最大池化
pool = nn.MaxPool2d(3, stride=1, padding=1, dilation=1, return_indices=False, ceil_mode=False)
tensor_res_1 = torch.Tensor(res_1)
img_pool1 = pool(tensor_res_1)
print(img_pool1.shape)

#非线性激活



pylab.show()
