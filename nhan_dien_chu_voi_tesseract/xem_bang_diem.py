# coding: utf-8
import argparse
import cv2
import pytesseract

import imutils
import numpy as np
from imutils.perspective import four_point_transform
from ham_xu_ly import sap_xep


# Xây dựng các tham số tùy chọn và lấy tham số
tuy_chon = argparse.ArgumentParser()
tuy_chon.add_argument(
        '-i',
        '--image',
        type=str,
        help='Đường dẫn ảnh',
        )
cac_tuy_chon = vars(tuy_chon.parse_args())

# Đọc ảnh
hinh_anh = cv2.imread(cac_tuy_chon['image'], 0)
hinh_anh_goc = hinh_anh.copy()
lam_mo = cv2.GaussianBlur(hinh_anh, (5, 5), 0)
den_trang = cv2.adaptiveThreshold(
        lam_mo,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2,
        )

# Lọc các cấu trúc hình học theo chiều dọc và chiều ngang của hình ảnh
chieu_doc = den_trang
chieu_ngang = den_trang
ti_le_doc = int(hinh_anh.shape[1] / 20)
ti_le_ngang = int(hinh_anh.shape[0] / 15)

# Lấy cấu trúc hình học dọc
cau_truc_doc = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (ti_le_doc, 1),
        )
chieu_doc = cv2.erode(
        chieu_doc,
        cau_truc_doc,
        (-1, -1),
        )
chieu_doc = cv2.dilate(
        chieu_doc,
        cau_truc_doc,
        (-1, -1),
        )

# Lấy cấu trúc hình học ngang
cau_truc_ngang = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, ti_le_ngang),
        )
chieu_ngang = cv2.erode(
        chieu_ngang,
        cau_truc_ngang,
        (-1, -1),
        )
chieu_ngang = cv2.dilate(
        chieu_ngang,
        cau_truc_ngang,
        (-1, -1),
        )

ket_qua = chieu_doc + chieu_ngang

# Tìm các đường viền trong hình các cạnh và sắp xếp theo thứ tự giảm dần
cac_duong_vien = cv2.findContours(
        ket_qua,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
        )
cac_duong_vien = imutils.grab_contours(cac_duong_vien)
cac_duong_vien = sorted(cac_duong_vien, key=cv2.contourArea, reverse=True)

# Tính toán đường bao xấp xỉ với đường viền lớn nhất
peri = cv2.arcLength(cac_duong_vien[0], True)
duong_bao = cv2.approxPolyDP(
        cac_duong_vien[0],
        0.02 * peri,
        True,
        )

# Cắt ảnh theo đường bao xấp xỉ và loại bỏ phần thừa
anh_ket_qua_1 = four_point_transform(
        chieu_doc,
        duong_bao.reshape(4, 2),
        )

# Cắt ảnh gốc theo đường bao xấp xỉ
anh_goc_1 = four_point_transform(
        hinh_anh_goc,
        duong_bao.reshape(4, 2),
        )

# Cắt ảnh kết quả theo đường bao xấp xỉ
ket_qua_1 = four_point_transform(
        ket_qua,
        duong_bao.reshape(4, 2),
        )

# Tìm các đường viền trong ảnh kết quả
cac_duong_vien_1 = cv2.findContours(
        ket_qua_1,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
        )
cac_duong_vien_1 = imutils.grab_contours(cac_duong_vien_1)
cac_duong_vien_1 = sorted(cac_duong_vien_1, key=cv2.contourArea, reverse=True)

# Lặp qua các đường viền và tính đường bao xấp xỉ
cac_duong_bao = []
stt = 1
for duong_vien in cac_duong_vien_1[1:]:
    # Tính toán đường bao xấp xỉ quanh các đường viền
    peri = cv2.arcLength(duong_vien, True)
    duong_bao = cv2.approxPolyDP(
            duong_vien,
            0.02 * peri,
            True,
            )
    # Lưu các đường bao có 4 góc vào danh sách cac_duong_bao và hiển thị số
    # thứ tự của đường bao đó tại tâm của nó
    if len(duong_bao) == 4:
        cac_duong_bao.append(duong_bao)

# Sắp xếp lại các đường bao
# cac_duong_bao, boundingBoxes = sap_xep(cac_duong_bao, 'left-to-right')

# Để sử dụng tesseract v4 cần cung cấp các thiết lập về
# -l ngôn ngữ sử dụng
# --oem mô hình mạng thần kinh muốn sử dụng
# --psm giá trị oem, 7 tức là xem hình ảnh là 1 dòng văn bản đơn
cac_thiet_lap = ('-l vie --oem 1 --psm 7')

for duong_bao in cac_duong_bao:
    # Lấy tọa độ của từng ô
    (x_dau, y_dau) = duong_bao[0][0]
    (x_cuoi, y_cuoi) = duong_bao[2][0]
    # Tính tọa độ tâm của từng ô
    M = cv2.moments(duong_bao)
    toa_do_tam_x = int(M['m10'] / M['m00'])
    toa_do_tam_y = int(M['m01'] / M['m00'])
    # Sắp xếp tọa độ các điểm theo thứ tự tọa độ
    # Hiển thị số thứ tự tại tâm của từng đường bao
    cv2.putText(
            ket_qua_1,
            '%s - %s' % (stt, (y_cuoi)),
            (toa_do_tam_x - 10, toa_do_tam_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            )
    roi = anh_goc_1[y_dau:y_cuoi, x_dau:x_cuoi]
    stt += 1

# Hiển thị hình ảnh
cv2.imshow('Anh ket qua 1', ket_qua_1)
# cv2.imshow('Anh goc', anh_goc_1)
cv2.waitKey(0)
