from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import os
import time

import sys
sys.path.append("..")
from lp_CR import lpcr as lr
from lp_CS import lpcs as ls
from lp_Location import lpl as ll

IMG_PATH = ''     # 原图路径
LP_STR = ''       # 识别车牌字符序列
FINAL_PATH = ''   # 最终车牌位置
train_flags = [0 for i in range(2)]   # 是否训练的标志

class LBL_Thread(QThread):   # 更新UI线程
    lbl_signal = pyqtSignal(str)
    def __int__(self):
        super(LBL_Thread, self).__int__()

    def run(self):     # 线程执行函数
        global FINAL_PATH
        if FINAL_PATH == '':
            self.lbl_signal.emit(IMG_PATH)
        else:
            self.lbl_signal.emit(FINAL_PATH)
            FINAL_PATH = ''
        self.sleep(1)  # 本函数休息1s


class LBL_Text_Thread(QThread):   # 更新文字线程
    lbl_Text_signal = pyqtSignal(str)
    def __int__(self):
        super(LBL_Text_Thread, self).__int__()

    def run(self):     # 线程执行函数
        self.lbl_Text_signal.emit(LP_STR)
        self.sleep(1)  # 本函数休息1s

class LPR_Thread(QThread):  # 识别线程
    lpr_signal = pyqtSignal()
    def __int__(self):
        super(LPR_Thread, self).__int__()

    def run(self):
        self.lpr_signal.emit()
        self.sleep(1)

class ui_MainWindow(QMainWindow):   # 主界面

    def __init__(self):
        super().__init__()

        self.initUi()  # 界面绘制交给InitUi方法

    def initUi(self):

        # 窗口800*600
        self.resize(800, 600)
        # 定位中心
        self.center()
        # 设置窗口的标题
        self.setWindowTitle('License Plate Recognition with UI')
        # 设置窗口的图标，引用当前目录下的web.png图片
        self.setWindowIcon(QIcon("./UI/umr.jpg"))


        # 读取原图按钮
        btn_ReadPic = QPushButton("加载图片", self)
        btn_ReadPic.resize(180, 50)
        btn_ReadPic.move(600, 180)
        btn_ReadPic.setToolTip('输入需要识别的图片')

        # 识别按钮
        btn_Recognition = QPushButton("车牌识别", self)
        btn_Recognition.resize(180, 50)
        btn_Recognition.move(600, 260)
        btn_Recognition.setToolTip('识别图中车牌')

        # 清空按钮
        btn_Clear = QPushButton("清空", self)
        btn_Clear.resize(180, 50)
        btn_Clear.move(600, 340)
        btn_Clear.setToolTip('清空图像区域')

        # 关于按钮
        btn_About = QPushButton("关于", self)
        btn_About.resize(180, 50)
        btn_About.move(600, 420)
        btn_About.setToolTip('关于小组项目')

        # 按钮信号与槽函数连接
        btn_ReadPic.clicked.connect(self.buttonClicked)
        btn_Recognition.clicked.connect(self.buttonClicked)
        btn_Clear.clicked.connect(self.buttonClicked)

        self.upd_pic_thread = LBL_Thread()  # 图像更新实例化线程对象
        self.upd_pic_thread.lbl_signal.connect(self.set_label_func)

        self.upd_text_thread = LBL_Text_Thread()  # 文字更新线程
        self.upd_text_thread.lbl_Text_signal.connect(self.set_label_text)

        self.lpr_thread = LPR_Thread()
        self.lpr_thread.lpr_signal.connect(self.LPR_func)

        # Qlabel显示图片
        self.lbl = QLabel(self)
        self.lbl.setFixedSize(550, 450)
        self.lbl.move(20, 20)
        self.lbl.setFrameShape(QFrame.Box)
        self.lbl.setFrameShadow(QFrame.Raised)
        self.lbl.setLineWidth(1)
        # self.lbl.setText('hello 1')

        self.lbl_lpStr = QLabel(self)
        self.lbl_lpStr.setFixedSize(180, 50)
        self.lbl_lpStr.move(600, 20)
        self.lbl_lpStr.setFrameShape(QFrame.Box)
        self.lbl_lpStr.setFrameShadow(QFrame.Raised)
        self.lbl_lpStr.setLineWidth(1)


        # 开启训练的复选框
        cb_lpl = QCheckBox("定位网络训练", self)
        cb_lpl.move(600, 90)
        cb_lpl.toggle()
        cb_lpl.setChecked(False)
        cb_lpl.stateChanged.connect(self.changeFlags_1)

        cb_lpcr = QCheckBox("识别网络训练", self)
        cb_lpcr.move(600, 130)
        cb_lpcr.toggle()
        cb_lpcr.setChecked(False)
        cb_lpcr.stateChanged.connect(self.changeFlags_2)


        # 状态栏
        self.statusBar()

    # 点击事件
    def buttonClicked(self):
        sender = self.sender()
        # self.statusBar().showMessage(sender.text() + ' was pressed', 500)
        if sender.text() == '加载图片':
            # dirFolder = QFileDialog.getExistingDirectory(self,
            #                 '选取文件', './')
            # print(dirFolder)
            Img_Path, Img_Type = QFileDialog.getOpenFileName(self,
                '选取文件', './', 'ALL Files (*)')
            print(Img_Path, Img_Type)
            global IMG_PATH
            IMG_PATH = Img_Path

            # 自定义信号连接的槽函数
            self.upd_pic_thread.start()



        if sender.text() == '车牌识别':
            if IMG_PATH == '':
                QMessageBox.warning(self, '提示消息',
                       "未能读取到图像信息", QMessageBox.Yes |
                        QMessageBox.Cancel, QMessageBox.Cancel)
                return
            self.statusBar().showMessage('正在识别图像...')
            self.lpr_thread.start()


        if sender.text() == '清空':
            global LP_STR
            global FINAL_PATH
            IMG_PATH = ''
            LP_STR = ''
            FINAL_PATH = ''
            self.upd_pic_thread.start()
            self.upd_text_thread.start()


    def changeFlags_1(self, state):
        if state == Qt.Checked:
            train_flags[0] = 1
        else:
            train_flags[0] = 0
        # print(train_flags[0])

    def changeFlags_2(self, state):
        if state == Qt.Checked:
            train_flags[1] = 1
        else:
            train_flags[1] = 0
        # print(train_flags[0])


    # 更新label图像
    def set_label_func(self, path):
        # pixmap = QPixmap(path)  # 按指定路径找到图片，注意路径必须用双引号包围，不能用单引号
        # self.lbl.setPixmap(pixmap)  # 在label上显示图片

        img = QImage(path)  # 创建图片实例
        width = img.width()
        height = img.height()

        print(width, height)

        if width <= 550 and height <= 450:
            size = QSize(width, height)
            pixImg = QPixmap.fromImage(img.scaled(size, Qt.KeepAspectRatio))
            self.lbl.setPixmap(pixImg)
            self.lbl.setAlignment(Qt.AlignCenter)
            return

        if width > 550 and height <= 450:
            # 图片的宽大于label的宽
            scale = round((550 / width), 5)

        if width <= 550 and height > 450:
            # 图片的高大于label的高
            scale = round((450 / height), 5)

        if width > 550 and height > 450:
            scale_w = round((550 / width), 5)
            scale_h = round((450 / height), 5)
            if scale_w >= scale_h:
                scale = scale_w
            else:
                scale = scale_h

        rewidth = int(width * scale)
        reheight = int(height * scale)  # 缩放宽高尺寸
        size = QSize(rewidth, reheight)

        pixImg = QPixmap.fromImage(img.scaled(size, Qt.IgnoreAspectRatio))  # 修改图片实例大小并从QImage实例中生成QPixmap实例以备放入QLabel控件中

        # self.lbl.resize(rewidth, reheight)
        self.lbl.setPixmap(pixImg)
        self.lbl.setAlignment(Qt.AlignCenter)


    # 更新label_text车牌字符
    def set_label_text(self, str):
        self.lbl_lpStr.setAlignment(Qt.AlignCenter)  # 居中显示
        self.lbl_lpStr.setFont(QFont("Roman times", 24, QFont.Bold))  # 字体风格, 大小
        self.lbl_lpStr.setText(str)

    # 车牌识别
    def LPR_func(self):
        start = time.time()
        global IMG_PATH
        ll.lp_loc(IMG_PATH, train_flags[0])
        ls.lpcs()
        lp_str = lr.net_train(train_flags[1])

        if lp_str != '#00000#':
            print('lrStr = ' + lp_str)
            global FINAL_PATH
            project_path = os.path.abspath(os.getcwd())
            FINAL_PATH = os.path.join(project_path, 'final_data/finalImg.jpg')
            global LP_STR
            LP_STR = lp_str
            # 自定义信号连接的槽函数
            self.upd_pic_thread.start()
            self.upd_text_thread.start()
        else:
            QMessageBox.warning(self, '提示消息',
                                "未能读取到车牌信息", QMessageBox.Yes |
                                QMessageBox.Cancel, QMessageBox.Cancel)
        spt = time.time() - start
        self.statusBar().showMessage('识别完成...用时%.6fs' % spt, 2000)


    # 控制窗口显示在屏幕中心的方法
    def center(self):
        # 获得窗口
        qr = self.frameGeometry()
        # 获得屏幕中心点
        cp = QDesktopWidget().availableGeometry().center()
        # 显示到屏幕中心
        qr.moveCenter(cp)
        self.move(qr.topLeft())


