import sys
import cv2

from PyQt5.QtCore import QPoint
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget
from PyQt5.QtGui import QPainter, QImage
from PyQt5.uic import loadUiType

import imutils


GIAO_DIEN, _ = loadUiType('giao_dien_xem_anh.ui')


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


class ChuongTrinhChinh(QMainWindow, GIAO_DIEN):
    '''
    Giao diện chương trình
    '''

    def __init__(self, parent=None):
        super(ChuongTrinhChinh, self).__init__(parent)
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.khoi_tao_su_kien()
        self.khung_hinh_anh = ImageWidget(self)
        self.khung_hien_thi_anh.addWidget(self.khung_hinh_anh)

    def khoi_tao_su_kien(self):
        '''
        Khởi tạo các sự kiện
        '''
        self.gia_tri_nho_nhat.valueChanged.connect(self.hien_gia_tri_nho_nhat)
        self.gia_tri_lon_nhat.valueChanged.connect(self.hien_gia_tri_lon_nhat)

    def hien_gia_tri_nho_nhat(self, gia_tri):
        '''
        Hiện giá trị nhỏ nhất
        '''
        self.hien_thi_nho_nhat.setText(str(gia_tri))

    def hien_gia_tri_lon_nhat(self, gia_tri):
        '''
        Hiện giá trị lớn nhất
        '''
        self.hien_thi_lon_nhat.setText(str(gia_tri))

    def hien_thi_anh(self, hinh_anh, dinh_dang):
        '''
        Hiển thị hình ảnh
        '''
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
        self.khung_hinh_anh.setImage(q_hinh_anh)

    def doc_anh(self, duong_dan):
        '''
        Đọc hình ảnh chỉ định
        '''
        hinh_anh = cv2.imread(duong_dan)

        # Chỉnh tỉ lệ hiển thị
        ti_le = hinh_anh.shape[0] / 1000.0
        anh_goc = hinh_anh.copy()
        hinh_anh = imutils.resize(hinh_anh, height=1000)

        # Chuyển sang ảnh đen trắng, làm mờ và tìm các cạnh
        den_trang = cv2.cvtColor(hinh_anh, cv2.COLOR_BGR2GRAY)
        den_trang = cv2.GaussianBlur(den_trang, (5, 5), 0)
        cac_canh = cv2.Canny(den_trang, 55, 200)
        self.hien_thi_anh(den_trang, 1)


if __name__ == '__main__':
    chuong_trinh = QApplication([])
    cua_so = ChuongTrinhChinh()
    cua_so.show()
    cua_so.doc_anh(sys.argv[1])
    chuong_trinh.exec_()
