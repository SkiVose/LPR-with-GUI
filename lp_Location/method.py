# 为车牌定位提供一些方法

def inter2exp(box1, box2):  # 两个相交区域融合成一个区域
    x1, y1, w1, h1 = box1    # 原始区域
    x2, y2, w2, h2 = box2    # 想要融合的区域
    new_box = []             # 新的候选区域
    # 判断是否存在重合部分
    if x1+w1 >= x2 and y1+h1 >= y2:
        # 区域2在区域1右下角重合
        if x1+w1 >= x2+w2:
            # 区域1的右上坐标x轴比区域2的要大
            if y1+h1 >= y2+h2:
                # 区域1的右下坐标y轴比区域2的要大
                new_box = [x1, y1, w1, h1]
            else:
                # 区域1的右下坐标y轴比区域2的要小
                new_box = [x1, y1, w1, h2]
        else:
            # 区域的右上坐标x轴比区域2的要小
            if y1 + h1 >= y2 + h2:
                # 区域1的右下坐标y轴比区域2的要大
                new_box = [x1, y1, w2, h1]
            else:
                # 区域1的右下坐标y轴比区域2的要小
                new_box = [x1, y1, w2, h2]

    elif x1+w1 >= x2 and y1+h1 >= y2+h2:
        # 区域2在区域1右上角重合
        if x1+w1 >= x2+w2:
            # 区域1的右上坐标x轴比区域2的要大
            if y1+h1 >= y2+h2:
                # 区域1的右下坐标y轴比区域2的要大
                new_box = [x1, y1, w1, h1]
            else:
                # 区域1的右下坐标y轴比区域2的要小
                new_box = [x1, y1, w1, h2]
        else:
            # 区域的右上坐标x轴比区域2的要小
            if y1 + h1 >= y2 + h2:
                # 区域1的右下坐标y轴比区域2的要大
                new_box = [x1, y1, w2, h1]
            else:
                # 区域1的右下坐标y轴比区域2的要小
                new_box = [x1, y1, w2, h2]