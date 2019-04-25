#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: lemon

import numpy as np
from tqdm import tqdm

from sklearn.metrics import accuracy_score


class Variable(object):
    def __init__(self, w, b, dw, db):
        self.w = w
        self.dw = dw
        self.b = b
        self.db = db


class NN(object):
    def __init__(self):
        pass
    def forward(self, *args):
        pass
    def backward(self, grad):
        pass
    def params(self):
        pass
    def __call__(self, *args):
        return self.forward(*args)


class Linear(NN):
    """
    全连接层
    """
    def __init__(self, dim_in, dim_out):
        super(Linear, self).__init__()
        self.w = np.random.random((dim_in, dim_out)) * 0.005
        self.b = np.random.randn(dim_out) * 0.005
        self.dw = np.zeros(self.w.shape)
        self.db = np.zeros(self.b.shape)
        self.variable = Variable(self.w, self.b, self.dw, self.db)
        self.input = None
        self.output = None

    def params(self):
        return self.variable

    def forward(self, *args):
        self.input = args[0]
        self.output = np.dot(self.input, self.w) + self.b
        return self.output

    def backward(self, grad):
        self.db = grad
        self.dw += np.dot(self.input.T, grad)
        grad = np.dot(grad, self.w.T)
        return grad


class ReLU(NN):
    def __init__(self):
        super(ReLU, self).__init__()
        self.input = None
        self.output = None

    def forward(self, *args):
        x = args[0]
        self.input = x
        x[self.input<=0] *= 0
        self.output = x
        return self.output

    def backward(self, grad):
        grad[self.input>0] *= 1.0
        grad[self.input<=0] *= 0.0
        return grad


class Optim(object):
    """
    优化器

    """
    def __init__(self, pip):
        self.params = pip.params()
        self.lr = 1e-2

    def zero_grad(self):
        for param in self.params:
            param.dw *= 0
            param.db *= 0

    def setup(self):
        for param in self.params:
            param.w -= self.lr * param.dw
            param.b -= self.lr * param.db


class Loss(object):
    """
    损失函数
    """

    def __init__(self):
        self.label = None
        self.logit = None
        self.grad = None
        self.loss = None

    def forward(self, logit, label):
        self.logit, self.label = logit, label
        self.loss = np.sum(0.5 * np.square(self.logit - self.label))
        return self.loss

    def backward(self):
        self.grad = self.logit - self.label
        grad_ = np.sum(self.grad, axis=0)
        return np.expand_dims(grad_, axis=0)


class Pip(object):
    """
    神经网络管道
    """

    def __init__(self, layers):
        self.layers = []
        self.parameters = []
        for layer in layers:
            self.layers.append(layer)
            if isinstance(layer, Linear):
                self.parameters.append(layer.params())

    def forward(self, *args):
        x = args[0]
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, grad):
        for idx in range(len(self.layers)-1, -1, -1):
            grad = self.layers[idx].backward(grad)

    def params(self):
        return self.parameters


def get_data():
    train = np.load('./data/mnist/traindata.npy')
    trainlabel = np.load('./data/mnist/trainlabellogit.npy')
    val = np.load('./data/mnist/valdata.npy')
    vallabel = np.load('./data/mnist/vallabellogit.npy')
    return (train, trainlabel, val, vallabel)

def main():
    train_x, train_y, val_x, val_y = get_data()
    D_in = 28 * 28        # 图片像素 二维矩阵转换成一纬
    D_hidden = 28 * 28    # 全连接的神经元节点
    D_out = 10            # 输出的神经元节点

    layers = [Linear(D_in, D_hidden), ReLU(), Linear(D_hidden, D_out)]
    pip = Pip(layers)     # 生成训练管道
    optim = Optim(pip)    # 优化器，优化器需要用到 训练参数
    criterion = Loss()    # 损失函数

    val_batches = val_x.shape[0]
    for batch in tqdm(range(train_x.shape[0])):
        optim.zero_grad()                       # 反向传播的参数初始化
        input = train_x[batch:batch + 1] / 255  # 数据归一化
        label = train_y[batch:batch + 1]

        pred = pip.forward(input)               # 前向传播
        loss = criterion.forward(pred, label)   # 损失评估
        grad = criterion.backward()             # 后向传播
        pip.backward(grad)
        optim.setup()                           # 优化

        # 检查准确率
        if batch % 1000 == 0:
            pred_label = []
            label = []
            for val_idx in range(10):
                rnd_val_idx = np.random.randint(0, val_batches)
                valinput = val_x[rnd_val_idx:rnd_val_idx + 1] / 255
                valpred = pip.forward(valinput)
                pred_label.append(valpred.flatten().argmax())
                label.append(val_y[rnd_val_idx].flatten().argmax())
            print(pred_label, label)
            print("accuracy ", accuracy_score(pred_label, label))


if __name__ == "__main__":
    main()