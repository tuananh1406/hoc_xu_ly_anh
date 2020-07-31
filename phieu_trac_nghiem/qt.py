import sys
import time
import threading
import cv2

import imutils

from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QAction
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QTextEdit
from PyQt5.QtGui import QPainter, QImage
from PyQt5.uic import loadUiType


# Image widget
class ImageWidget(QWidget):
    def __init__(self, parent=None):
        super(ImageWidget, self).__init__(parent)
        self.image = None

    def setImage(self, image):
        self.image = image
        self.setMinimumSize(image.size())
        self.update()

    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QPoint(0, 0), self.image)
        qp.end()


class CuaSoChinh(QMainWindow):
    '''
    Giao diện chính của phần mềm
    '''

    # Khởi tạo cửa sổ
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)

        self.central = QWidget(self)
        self.vlayout = QVBoxLayout()
        self.displays = QHBoxLayout()
        self.disp = ImageWidget(self)
        self.displays.addWidget(self.disp)
        self.vlayout.addLayout(self.displays)
        self.central.setLayout(self.vlayout)
        self.setCentralWidget(self.central)

    def xem_anh(self, hinh_anh, dinh_dang):
        if dinh_dang == 1:
            dinh_dang_anh = QImage.Format_Grayscale8
            du_lieu = hinh_anh.shape[1]
        else:
            dinh_dang_anh = QImage.Format_RGB888
            du_lieu = hinh_anh.shape[1] * 3

        q_hinh_anh = QImage(
                hinh_anh.data,
                hinh_anh.shape[1],
                hinh_anh.shape[0],
                du_lieu,
                dinh_dang_anh,
                )
        self.disp.setImage(q_hinh_anh)

    def doc_anh(self, duong_dan):
        hinh_anh = cv2.imread(duong_dan)

        # Chỉnh tỉ lệ hiển thị
        ti_le = hinh_anh.shape[0] / 1000.0
        anh_goc = hinh_anh.copy()
        hinh_anh = imutils.resize(hinh_anh, height=1000)

        # Chuyển sang ảnh đen trắng, làm mờ và tìm các cạnh
        den_trang = cv2.cvtColor(hinh_anh, cv2.COLOR_BGR2GRAY)
        den_trang = cv2.GaussianBlur(den_trang, (5, 5), 0)
        cac_canh = cv2.Canny(den_trang, 55, 200)
        self.xem_anh(den_trang, 1)

GIAO_DIEN = loadUiType('giao_dien.ui')
class ChuongTrinhChinh(QMainWindow, GIAO_DIEN):


if __name__ == '__main__':
    app = QApplication([])
    cua_so = CuaSoChinh()
    cua_so.show()
    cua_so.doc_anh(sys.argv[1])
    cua_so.setWindowTitle('Chương trình chính')
    app.exec_()
