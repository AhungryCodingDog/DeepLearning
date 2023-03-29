import numpy as np
import cv2

def unpickle(file):
    import pickle
    with open(file,'rb') as fo:
        dict = pickle.load(fo , encoding= 'bytes')
        return dict

data_batch = unpickle("..cifar10_data/cifar-10-batches-py/data_batch_1")
cifar_data = data_batch[b'data']
cifar_data = np.array(cifar_data)
cifar_label = data_batch[b'labels']
cifar_label = np.array(cifar_label)

def imwrite_images(k):
    for i in range(k):
        image = cifar_data[i]
        image = image.reshape(-1 , 1024)
        r = image[0,:].reshape(32,32)
        g = image[1, :].reshape(32, 32)
        b = image[2, :].reshape(32, 32)
        img = np.zeros((32,32,3))

        img[:,:,0] = r
        img[:, :, 1] = g
        img[:, :, 2] = b
        cv2.imwrite(f"cifpicture/cifpiture_{i}.jpg",img)
    print("保存成功")

label_name = ['飞机' , '小车' , '鸟' , '猫' , '鹿' , '狗' , '青蛙' , '马' , '船' , '卡车']
imwrite_images(1)