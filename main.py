import sys
sys.path.append("..")
from UI import LPR_UI as lpr
from PyQt5.QtWidgets import *

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = lpr.ui_MainWindow()
    # 显示窗口
    ui.show()
    sys.exit(app.exec_())
