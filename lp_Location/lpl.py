# 车牌定位
import cv2
import numpy as np
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import os
import sys
sys.path.append("..")
from lp_Location import lp_net as lpnet

def lp_loc(img_path, flag):

    if flag == 1:
        net_train()

    # file_path = 'C:/Users/Administrator/PycharmProjects/img/car63.jpg'
    img = cv2.imread(img_path)
    # cv2.imshow('img', img)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # cv2.imshow('hsv', img_hsv)
    # Opencv定义的结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    # 选出蓝色候选区
    blue_list = []
    # 每个蓝色候选区的概率
    blue2score = []
    count = 0
    for i in range(70, 221, 30):
        count += 1
        # print(i)
        lower_blue = np.array([100, 43, i])
        upper_blue = np.array([124, 255, i + 30])
        # print(lower_blue, upper_blue)
        mask = cv2.inRange(img_hsv, lower_blue, upper_blue)
        # cv2.imshow('mask', mask)
        dilated = cv2.dilate(mask, kernel)
        eroded = cv2.erode(dilated, kernel)
        dilated = cv2.dilate(eroded, kernel)
        eroded = cv2.erode(dilated, kernel)
        dilated = cv2.dilate(eroded, kernel)
        Glur = cv2.GaussianBlur(dilated, (9, 9), 0)
        canny = cv2.Canny(Glur, 80, 240, 3)
        # cv2.imshow('glur', Glur)
        # cv2.imshow('dilated', dilated)
        # cv2.imshow('canny', canny)
        # cv2.imshow('eroded', eroded)

        image, contours, hier = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if float(w)/h >= 1.5 and float(w)/h <= 4.0:
                if w >= 100 and h >= 30:
                    blue_list.append([x, y, w, h])

                # timg = img.copy()
                # cv2.rectangle(timg, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # cv2.imshow('timg', timg)
                # cv2.waitKey(0)

    project_path = os.path.abspath(os.getcwd())
    save_path = os.path.join(project_path, 'img/0')

    if len(blue_list) == 0:
        print("未找到任何候选区域!")
        return "#00000#"
        # exit()

    # tempImg = img.copy()
    # for i in range(len(blue_list)):
    #     x, y, w, h = blue_list[i]
    #     cv2.rectangle(tempImg, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # cv2.imshow('temp', tempImg)
    # cv2.waitKey(0)

    for i in range(len(blue_list)):
        x, y, w, h = blue_list[i]
        bimg = img[y:y+h, x:x+w]
        reImg = cv2.resize(bimg, (144, 144), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(save_path, 'lp_check.jpg'), reImg)  # 写入检测图片
        indice, value = net_getScore()   # 是否为车牌及其概率
        # cv2.waitKey(0)
        if indice == 0:  # 是车牌
            blue2score.append([i, value])
        else:
            blue2score.append([i, 0])
        os.remove(os.path.join(save_path, 'lp_check.jpg'))  # 删除检测图片

    # print(blue2score)

    # 按成绩降序排序
    blue2score.sort(key=takeSecond, reverse=True)
    # print(blue2score)

    fimg = img.copy()

    # 判断blue2score中有多少个车牌候选区域
    tot = 0
    for i in range(len(blue2score)):
        if blue2score[i][1] > 0:
            tot += 1
        else:
            break

    if tot == 0:
        print('未检测到车牌区域')
        return "#00000#"
        # exit()
    else:
        # 选取最大三个候选区
        for i in range(tot):
            if i >= 3:
                break
            fx, fy, fw, fh = blue_list[blue2score[i][0]]
            # cv2.rectangle(fimg, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 2)
            # cv2.imshow('fimg', fimg)
            # cv2.waitKey(0)

        # 仿NMS算法结合最多三个候选区，确认最终车牌区域
        if tot == 1:
            x, y, w, h = blue_list[blue2score[0][0]]
            cv2.rectangle(fimg, (fx, fy), (fx+fw, fy+fh), (0, 0, 255), 3)
        elif tot == 2:
            x, y, w, h = sel_rect(blue_list[blue2score[0][0]], blue_list[blue2score[1][0]])
        elif tot >= 3:
            x1, y1, w1, h1 = sel_rect(blue_list[blue2score[0][0]], blue_list[blue2score[1][0]])
            x2, y2, w2, h2 = sel_rect(blue_list[blue2score[1][0]], blue_list[blue2score[2][0]])
            x, y, w, h = sel_rect([x1, y1, w1, h1], [x2, y2, w2, h2])

        finalImg = img[y:y+h, x:x+w]
        cv2.rectangle(fimg, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.imwrite(os.path.join(project_path, 'final_data/final.jpg'), finalImg)
        cv2.imwrite(os.path.join(project_path, 'final_data/finalImg.jpg'), fimg)




    # cv2.imshow('fimg', fimg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def takeSecond(elem):
    return elem[1]


def sel_rect(rect1, rect2):
    x1, y1, w1, h1 = rect1
    area1 = w1 * h1
    x2, y2, w2, h2 = rect2
    area2 = w2 * h2

    # 如果两左上角坐标不超过(10, 10)的范围，则选取最大面积的区域
    if abs(x1 - x2) <= 10 and abs(y1 - y2) <= 10:
        (x, y, w, h) = (x1, y1, w1, h1) if area1 >= area2 else (x2, y2, w2, h2)
    # 如果两右上角坐标不超过(10, 10)的范围，则选取最大面积的区域
    elif abs((x1+w1) - (x2+w2)) <= 10 and abs(y1 - y2) <= 10:
        (x, y, w, h) = (x1, y1, w1, h1) if area1 >= area2 else (x2, y2, w2, h2)
    # 如果两左下角坐标不超过(10, 10)的范围，则选取最大面积的区域
    elif abs(x1-x2) <= 10 and abs((y1+h1) - (y2+h2)) <= 10:
        (x, y, w, h) = (x1, y1, w1, h1) if area1 >= area2 else (x2, y2, w2, h2)
    # 如果两右下角坐标不超过(10, 10)的范围，则选取最大面积的区域
    elif abs((x1+w1) - (x2+w2)) <= 10 and abs((y1+h1) - (y2+h2)) <= 10:
        (x, y, w, h) = (x1, y1, w1, h1) if area1 >= area2 else (x2, y2, w2, h2)
    else:
        x, y, w, h = x1, y1, w1, h1
    return x, y, w, h


# 网络训练
def net_train():
    # 选择CUDA计算
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 对图像进行归一化
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  #默认的值
    transform = transforms.Compose([
            transforms.RandomResizedCrop(size=(144, 144)),
            transforms.ToTensor(),
            normalize,
        ])

    # 加载训练集
    data_path = 'D:/Datasets/ccpd_simple/lp_simple_sorted'

    train_data = ImageFolder(os.path.join(data_path, 'train'), transform=transform)
    train_iter = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True, num_workers=0)
    test_data = ImageFolder(os.path.join(data_path, 'test'),  transform=transform)
    test_iter = torch.utils.data.DataLoader(test_data, batch_size=256, shuffle=True, num_workers=0)

    batch_size, lr, num_epochs = 256, 0.001, 50
    net = lpnet.sNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    lpnet.train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
    project_path = os.path.abspath(os.getcwd())
    save_path = os.path.join(project_path, 'lp_Location/lpl_net.pt')
    torch.save(net, save_path)


# 得到每个候选区的概率
def net_getScore():
    # 常规定义
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 默认的值
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(144, 144)),
        transforms.ToTensor(),
        normalize,
    ])

    # 加载训练好的模型
    project_path = os.path.abspath(os.getcwd())
    save_path = os.path.join(project_path, 'lp_Location/lpl_net.pt')
    net = torch.load(save_path)

    # 选择候选区图像路径
    project_path = os.path.abspath(os.getcwd())
    check_data = ImageFolder(os.path.join(project_path, 'img'), transform=transform)
    check_iter = torch.utils.data.DataLoader(check_data, batch_size=256, shuffle=False, num_workers=0)

    net.eval()  # 进入评估模式
    for X, y in check_iter:
        y = torch.tensor(y)
        # print(X.shape, y.shape)

        y = net(X.to(device))
        # 得到两个概率
        values, indices = y.topk(2, dim=1)
        print(values, indices)

    net.train()  # 进行训练模式

    # 返回判断和概率
    print(indices[0][0], values[0][0].item())
    return indices[0][0], values[0][0].item()
