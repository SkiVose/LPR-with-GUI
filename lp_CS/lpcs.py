# 字符分割

import os
import cv2
import math
import random
import numpy as np
from scipy import misc, ndimage
from matplotlib import pyplot as plt

# 水平投影
def getHProjection(image):
    # 将image转换为二值图，ret接受当前阈值，thresh1接受输出的二值图
    ret, thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    # cv2.imshow('Thresh1image', thresh1)
    h, w = thresh1.shape  # 返回高宽
    # print('h = %d, w = %d' % (h, w))
    a = [0 for z in range(0, h)]  # a = [0, 0, ... , 0] 记录每一行的白点数

    #记录每一行的波峰
    for y in range(0, h):  # 遍历一行
        for x in range(0, w):  # 遍历一列
            if thresh1[y, x] == 255:  # 如果该点为白点
                a[y] += 1     # 该列计数器加一
            thresh1[y, x] = 255  # 每个像素点涂白
    for y in range(0, h):
        for x in range(a[y]):  # 该行统计的白点数
            thresh1[y, x] = 0  # 涂黑

    # cv2.imshow('Himage', thresh1)

    # 把上下两边的框框去掉
    row1, row2 = 0, h
    count = 0
    for i in range(0, h):
        count += 1
        if a[i] <= 20:  # 阈值为10
            if row1 == 0:
                row1 = i
                count = 0
            else:
                if count < h * 0.3:
                    row1 = i
                    count = 0
                else:
                    row2 = i
                    break

    # print('row1 = %d, row2 = %d' % (row1, row2))
    Cimg = image[row1:row2, :]

    # cv2.imshow('Cutimage', Cimg)
    return Cimg

# 垂直投影
def getVProjection(image):
    # 将image转换为二值图，ret接受当前阈值，thresh1接受输出的二值图
    ret, thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    BinImg = thresh1.copy()
    # cv2.imshow('Sourceimage', thresh1)
    h, w = thresh1.shape  # 返回高宽
    # print('h = %d, w = %d' % (h, w))
    a = [0 for z in range(0, w)]  # a = [0, 0, ... , 0] 记录每一列的白点数

    # 记录每一列的波峰
    for x in range(0, w):  # 遍历一列
        for y in range(0, h):  # 遍历一行
            if thresh1[y, x] == 255:  # 如果该点为白点
                a[x] += 1  # 该列计数器加一
            thresh1[y, x] = 255  # 每个像素点涂白
    for x in range(0, w):
        for y in range(a[x]):  # 该列统计的白点数
            thresh1[y, x] = 0  # 涂黑

    # cv2.imshow('Vimage', thresh1)

    # 把左右两边的框框去掉，并把七个字符分割开来
    col1, col2 = 0, w
    ch = [[0 for i in range(2)] for i in range(10)]  # 每个字符的位置
    count, index = 0, 0    # 连续列值，字符下标
    area_part = 0.0
    test_count = 0
    for i in range(0, w):
        count += 1
        if a[i] <= 5:  # 阈值5
            if col1 == 0:
                col1 = i
            else:
                if area_part == 0:
                    continue
                test_count += 1
                # print('%d: %.6f' % (test_count, area_part / (w * h)))
                if area_part / (w * h) < 0.015:
                    col1 = 0
                    # count = 0
                else:  # 这是个有效字符
                    col2 = i
                    # print('%d: col1 = %d, col2 = %d' % (index + 1, col1, col2))
                    # 录入第index个字符数组
                    ch[index][0] = col1
                    ch[index][1] = col2
                    col1 = 0
                    col2 = 0
                    index += 1
                    # count = 0
            area_part = 0.0
        else:
            area_part += a[i]   # 该列有数值，统计该连续单位的数值

    project_path = os.path.abspath(os.getcwd())
    save_path = os.path.join(project_path, 'chimg/0')

    for i in range(0, index):
        Cimg = BinImg[:, ch[i][0]:ch[i][1]]
        # print(Cimg.shape)
        #Cimg = Cimg.resize(24, 24)
        # cv2.imshow('Ch%d' % (i+1), Cimg)
        C2img = cv2.resize(Cimg, (32, 32), interpolation=cv2.INTER_AREA)
        # BlurImg = cv2.GaussianBlur(C2img, (3, 3), 0)
        cv2.imwrite(os.path.join(save_path, '%d.bmp' % (i + 1)), C2img)

def lpcs():
    dir_path = os.path.abspath(os.getcwd())
    file_path = os.path.join(dir_path, 'final_data/final.jpg')

    img = cv2.imread(file_path)
    # cv2.imshow('origin', img)

    # 霍夫变换
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', gray)
    gaus = cv2.GaussianBlur(gray, (3, 3), 0)
    # cv2.imshow('gaus', gaus)
    edges = cv2.Canny(gaus, 80, 210, apertureSize=3)
    # cv2.imshow('edges', edges)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 0)
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
    t = float(y2 - y1) / (x2 - x1)
    rotate_angle = math.degrees(math.atan(t))
    if rotate_angle > 45:
        rotate_angle = -90 + rotate_angle
    elif rotate_angle < -45:
        rotate_angle = 90 + rotate_angle
    rotate_img = ndimage.rotate(img, rotate_angle)
    # cv2.imshow('rotate', rotate_img)
    # cv2.waitKey(0)

    # 转化为灰度图
    lpImage = cv2.cvtColor(rotate_img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('lpImage', lpImage)

    # 字符分割部分
    cimg = getHProjection(lpImage)
    getVProjection(cimg)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()




