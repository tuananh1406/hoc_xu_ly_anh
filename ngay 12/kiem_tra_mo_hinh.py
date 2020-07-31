# coding: utf-8
from __future__ import print_function
from keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2


def hinh_anh_sang_vector(hinh_anh, size=(32,32)):
    return cv2.resize(hinh_anh, size).flatten()

#Xây dựng các tham số tùy chọn và lấy các tham số
tuy_chon = argparse.ArgumentParser()
tuy_chon.add_argument(
        '-m',
        '--model',
        required=True,
        help='Đường dẫn tệp mô hình',
        )
tuy_chon.add_argument(
        '-t',
        '--test-images',
        required=True,
        help='Đường dẫn thư mục ảnh kiểm tra',
        )
tuy_chon.add_argument(
        '-b',
        '--batch-size',
        type=int,
        default=32,
        help='Thông số kích thước của các gói trong mạng',
        )
cac_tuy_chon = vars(tuy_chon.parse_args())

#Khởi tạo nhãn của các lớp cho tập dữ liệu
cac_lop = ['meo', 'cho']

#Tải mô hình mạng lưới
print('[INFO] Tải mô hình mạng lưới đã huấn luyện...')
mo_hinh = load_model(cac_tuy_chon['model'])
print('[INFO] Kiểm tra trên tập hình ảnh: {}'.format(
    cac_tuy_chon['test_images'],
    ))
#Lặp qua từng ảnh trong tập ảnh kiểm tra
for duong_dan_anh in paths.list_images(cac_tuy_chon['test_images']):
    #Tải hình ảnh, chỉnh cỡ về 32 x 32 pixel (bỏ qua tỉ lệ khung
    #hình), sau đó lấy các thuộc tính từ nó
    print('[INFO] Đang phân loại ảnh {}'.format(
        duong_dan_anh[duong_dan_anh.rfind('/') + 1:]
        ))
    hinh_anh = cv2.imread(duong_dan_anh)
    cac_thuoc_tinh = hinh_anh_sang_vector(hinh_anh) / 255.0
    cac_thuoc_tinh = np.array([cac_thuoc_tinh])

    #Phân loại hình ảnh sử dụng các thuộc tính đã tách và mô hình mạng
    #lưới đã huấn luyện
    xac_suat = mo_hinh.predict(cac_thuoc_tinh)[0]
    du_doan = xac_suat.argmax(axis=0)

    #Gắn lớp và xác suất cho ảnh kiểm tra và hiển thị lên màn hình
    nhan = '{}: {:.2f}%'.format(
            cac_lop[du_doan],
            xac_suat[du_doan] * 100,
            )
    cv2.putText(
            hinh_anh,
            nhan,
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            3,
            )
    cv2.imshow('Hinh anh', hinh_anh)
    cv2.waitKey(0)
