#coding: utf-8
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2


#Xây dựng các tham số tùy chọn và lấy tùy chọn
tuy_chon = argparse.ArgumentParser()
tuy_chon.add_argument(
        '-p',
        '--shape-predictor',
        required=True,
        help='Đường dẫn đến tệp nhận diện điểm mốc',
        )
tuy_chon.add_argument(
        '-i',
        '--image',
        required=True,
        help='Đường dẫn đến tệp ảnh đầu vào',
        )
cac_tuy_chon = vars(tuy_chon.parse_args())

#Khởi tạo trình nhận diện khuôn mặt của dlib (dựa trên HOG) và sau đó
#tạo trình nhận diện điểm mốc
nhan_dien_khuon_mat = dlib.get_frontal_face_detector()
nhan_dien_diem_moc = dlib.shape_predictor(
        cac_tuy_chon['shape_predictor']
        )

#Tải ảnh đầu vào, chỉnh cỡ và chuyển sang đen trắng
hinh_anh = cv2.imread(cac_tuy_chon['image'])
hinh_anh = imutils.resize(hinh_anh, width=500)
den_trang = cv2.cvtColor(hinh_anh, cv2.COLOR_BGR2GRAY)

#Nhận diện khuôn mặt trong ảnh đen trắng
khuon_mat = nhan_dien_khuon_mat(den_trang, 1)

#Lặp qua từng khuôn mặt
for (i, mat) in enumerate(khuon_mat):
    #Xác định các điểm mốc trên vùng khuôn mặt, sau đó chuyển các điểm
    #mốc từ hệ tọa độ x, y sang mảng Numpy
    vung_diem_moc = nhan_dien_diem_moc(den_trang, mat)
    vung_diem_moc = face_utils.shape_to_np(vung_diem_moc)

    #Chuyển hình đa giác của dlib sang hình bao ngoài theo dạng của
    #OpenCV (VD: (x, y, w, h)), sau đó vẽ lại hình bao khuôn mặt
    (x, y, w, h) = face_utils.rect_to_bb(mat)

    #Hiển thị số khuôn mặt
    cv2.putText(
            hinh_anh,
            'Khuon mat #{}'.format(i + 1),
            (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            )

    #Lặp qua các cặp tọa độ (x, y) của các điểm mốc và vẽ nó lên hình
    #ảnh
    for (x, y) in vung_diem_moc:
        cv2.circle(
                hinh_anh,
                (x, y),
                1,
                (0, 0, 255),
                -1,
                )

#Hiển thị ảnh kết quả với khuôn mặt được nhận diện và các điểm mốc
cv2.imshow('Ket qua', hinh_anh)
cv2.waitKey(0)
