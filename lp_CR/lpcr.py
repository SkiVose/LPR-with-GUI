# 字符识别
import torch
import torchvision

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

from matplotlib import pyplot as plt
from torch import nn, optim

from PIL import Image
import numpy as np

import time
import os


import sys
sys.path.append("..")
from lp_CR import iLeNet as iln


def net_train(flag):
    # 运行在GPU上
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 改进网络输出个数，d0~d9:数字  a0~c3:字母  z0~z5:省份
    slps = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
            'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
            'W', 'X', 'Y', 'Z', "沪", "苏", "浙", "闽", "京", "粤"]   # 31个省份 24个字母 10个数字 （共65个字符）


    # for i in range(0, 65):
    #     print("%d: %s" % (i, lps[i]))

    # 设置训练或检测的图片规格为（宽32pix 高32pix）
    # 因为图片为灰度值，所以通道数为1  图片格式 -> torch.Size([1, 32, 32])

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # 对读入图像进行归一化
    transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize([32, 32]),
            transforms.ToTensor(),
            # normalize,
    ])

    project_path = os.path.abspath(os.getcwd())
    save_path = os.path.join(project_path, 'lp_CR/cr_net.pt')

    # 网络训练
    if flag == 1:
        data_path = "D:/Datasets/simple_lp_imgs"

        train_data = ImageFolder(os.path.join(data_path, 'train'), transform=transform)
        train_iter = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True, num_workers=0)
        test_data = ImageFolder(os.path.join(data_path, 'test'), transform=transform)
        test_iter = torch.utils.data.DataLoader(test_data, batch_size=256, shuffle=True, num_workers=0)

        # 超参数
        batch_size, lr, num_epochs = 256, 0.001, 50
        net = iln.LeNet()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        # sl.train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

        iln.train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

        torch.save(net, save_path)
    else:
        net = torch.load(save_path)

    # 字符识别
    lp_str = ''
    project_path = os.path.abspath(os.getcwd())
    check_path = os.path.join(project_path, 'chimg')
    check_data = ImageFolder(check_path, transform=transform)
    check_iter = torch.utils.data.DataLoader(check_data, batch_size=256, shuffle=False, num_workers=0)
    net.eval()  # 进入评估模式
    for X, y in check_iter:
        y = torch.tensor(y)
        # print(X.shape, y.shape)
        #
        # y = net(X.to(device))
        # values, indices = y.topk(2, dim=1)
        # print(values, indices)

        for i in net(X.to(device)).argmax(dim=1):
            print(slps[i.item()])
            lp_str = lp_str + slps[i.item()]
    net.train()   # 进行训练模式

    # 返回识别字符
    return lp_str
