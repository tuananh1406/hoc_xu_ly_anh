# coding: utf-8
from dulieumau.pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils


#Xây dựng các thông số tùy chọn và lấy tùy chọn
tuychon = argparse.ArgumentParser()
tuychon.add_argument('-i', '--image',
        required=True,
        help="Đường dẫn tệp ảnh đã quét",
        )
cactuychon = vars(tuychon.parse_args())

#Đọc hình ảnh, tính toán tỉ lệ, nhân bản hình ảnh và thay đổi kích thước ảnh
hinhanh = cv2.imread(cactuychon['image'])
tile = hinhanh.shape[0] / 500.00
hinhanh_goc = hinhanh.copy()
hinhanh = imutils.resize(hinhanh, height = 500)

#Chuyển hình ảnh sang đen trắng, làm mịn và tìm các cạnh khép kín
dentrang = cv2.cvtColor(
        hinhanh,
        cv2.COLOR_BGR2GRAY,
        )
dentrang = cv2.GaussianBlur(
        dentrang,
        (5, 5),
        0,
        )
timcanh = cv2.Canny(
        dentrang,
        75,
        200,
        )

#Hiển thị ảnh gốc và ảnh các cạnh khép kín được tìm thấy
print('Bước 1: Tìm cạnh khép kín')
#cv2.imshow('Hinh anh goc', hinhanh)
#cv2.imshow('Hinh tim thay cac canh', timcanh)

#Tìm các đường viền của các cạnh trong ảnh, chỉ giữ lại đường lớn nhất, sau đó
#phân tích đường viền của màn hình
cacduongvien = cv2.findContours(
        timcanh.copy(),
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
        )
cacduongvien = imutils.grab_contours(cacduongvien)
cacduongvien = sorted(cacduongvien, key = cv2.contourArea, reverse=True)[:5]

#Lặp qua các đường viền
for d in cacduongvien:
    #Ước lượng đường viền bao quanh các cạnh
    chuvi = cv2.arcLength( #Tính độ dài cung tròn
            d,
            True, #True là đường tròn khép kín, False là đường cong
            )
    #Vẽ một hình bao quanh đối tượng tìm được
    uocluong = cv2.approxPolyDP(
            d,
            0.02 * chuvi,
            True,
            )
    #Nếu hình ước lượng có 4 góc, sẽ giả sử là tìm được hình tờ giấy
    if len(uocluong) == 4:
        hinhgiay = uocluong
        break

#Hiển thị đường bao của tờ giấy
print('Buoc 2: Tim duong vien cua to giay')
cv2.drawContours(
        hinhanh,
        [hinhgiay],
        -1,
        (0, 255, 0),
        2,
        )
#cv2.imshow('Duong bao to giay', hinhanh)

#Áp dụng phương thức chuyển 4 điểm để xem ảnh gốc từ trên xuống
anhchuyendoi = four_point_transform(
        hinhanh_goc,
        hinhgiay.reshape(4, 2) * tile,
        )

#Chuyển ảnh chuyển đổi sang dạng đen trắng, sau đó sử dụng ngưỡng
anhchuyendoi = cv2.cvtColor(anhchuyendoi, cv2.COLOR_BGR2GRAY)
nguong = threshold_local(
        anhchuyendoi,
        11,
        offset = 10,
        method = 'gaussian',
        )
anhchuyendoi = (anhchuyendoi > nguong).astype('uint8') * 255

#Hiển thị ảnh gốc và ảnh đã quét
print('Buoc 3: Su dung phep chuyen khung hinh')
cv2.imshow('Anh goc', imutils.resize(hinhanh_goc, height = 650))
cv2.imshow('Anh da quet', imutils.resize(anhchuyendoi, height = 650))

cv2.waitKey(0)
cv2.destroyAllWindows()
