# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 09:57:59 2021

@author: 28663
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import sys
import random
from IPython import display
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import init


# 用矢量图显示
def use_svg_display():
    # 用矢量图显示
    display.set_matplotlib_formats('svg')
    
# 设置图的尺寸
def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


# 读取小批量数据样本
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本读取是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i+batch_size, num_examples)])  # 最后一次可能不足一个batch
        yield features.index_select(0, j), labels.index_select(0, j)
        
        
# 线性回归的矢量计算表达式
def linreg(X, w, b):
    return torch.mm(X, w) + b


# 定义均方差损失函数
def squared_loss(y_hat, y):
    # 这里返回的是向量，另外pytorch里的MSELoss没有除以二
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


# 获取fashion_mnist数据标签
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


# 定义一个可以在一行画出多张图像和对应标签的函数
def show_fashion_mnist(images, labels):
    use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()
    

# 加载fashion_mnist数据集
def load_data_fashion_mnist(batch_size, root='~/Datasets/FashionMNIST'):
    """Download the fashion mnist dataset and then load into memory."""
    transform = transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_iter, test_iter


# 类似的评价一下模型net在数据集data_iter上的准确率
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1)==y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


# 随机梯度下降函数更新梯度
def sgd(params, lr, batch_size):
    for param in params:
        # 这里更改param时用的是 param.data
        param.data -= lr * param.grad / batch_size


# 训练模型
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            
            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
                    
            l.backward()
            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()
                
            
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
        
# 定义一个FlattenLayer实现对 x 的形状转换功能
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    
    def forward(self, x): # x shape:(batch, *, *, ...)
        return x.view(x.shape[0], -1)
        
        
