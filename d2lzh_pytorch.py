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
        param.data -= lr * param.grad / batch_size

