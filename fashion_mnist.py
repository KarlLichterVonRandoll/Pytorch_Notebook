# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 15:36:56 2021

@author: 28663
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
from IPython import display
%matplotlib inline

mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST',train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST',train=False, download=True, transform=transforms.ToTensor())

print("以下将数值标签转成相应的文本标签")

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

# 定义一个可以在一行画出多张图像和对应标签的函数
def use_svg_display():
    display.set_matplotlib_formats('svg')

def show_fashion_mnist(images, labels):
    #use_svg_display()
    
    _, figs= plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28,28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

    
X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
show_fashion_mnist(X, get_fashion_mnist_labels(y))

