import torch
import torchvision
from torchvision import transforms
from torch import nn, optim
import time

def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n


def train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):

    net = net.to(device)
    print("lpLocation-Net is training on", device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        # print(epoch)
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))


# 改进后LeNet网络结构
# Input: 3x144x144
# conv: kernel(5x5)   output: 6x140x140
# pool: kernel(2x2)   output: 6x70x70
# conv: kernel(7x7)   output: 16x64x64
# pool: kernel(2x2)   output: 16x32x32
# conv: kernel(5x5)   output: 40x28x28
# pool: kernel(2x2)   output: 40x14x14
# conv: kernel(5x5)   output: 64x10x10
# pool: kernel(2x2)   output: 64x5x5
# conv: kernel(5x5)   output: 120x1x1

# FC1: 120 -> 84
# FC2: 84 -> 10
# FC3: 10 -> 2

# 改进后的sNet网络
class sNet(nn.Module):
    def __init__(self):
        super(sNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 6, 5),  # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # kernel_size, stride

            nn.Conv2d(6, 16, 7),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 40, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(40, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 120, 5),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.ReLU(),
            nn.Linear(10, 2)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output