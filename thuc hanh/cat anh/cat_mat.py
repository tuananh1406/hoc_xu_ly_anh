# coding: utf-8
import numpy as np
import argparse
import cv2
import imutils
import os
from imutils import paths


def cat_anh(duong_dan_anh, confi, mo_hinh, duong_dan_luu, so_thu_tu):
    #Tải ảnh đầu vào và đổi cỡ nếu ảnh quá lớn
    hinhanh_goc = cv2.imread(duong_dan_anh)
    hinhanh = hinhanh_goc.copy()
    (h, w) = hinhanh.shape[:2]
    thu_nho = False
    while h > 1000 or w > 1000:
        #print('Hình ảnh quá lớn (%s x %s)' % (w, h))
        print('[INFO] Tiến hành thu nhỏ ảnh')
        hinhanh = cv2.resize(hinhanh, ((int(w / 10)), (int(h / 10))))
        (h, w) = hinhanh.shape[:2]
        #print('Thu nhỏ về (%s x %s)' % (w, h))
        thu_nho = True

    #Xây dựng một blob đầu vào cho ảnh
    #bằng cách thay đổi cỡ sang 300x300 và sau đó chuẩn hóa
    blob = cv2.dnn.blobFromImage(
            cv2.resize(hinhanh, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0),
            )

    #Đưa blob vào mạng và tìm kiếm các sự phát hiện và phỏng đoán
    print("[INFO] Tìm kiếm các đối tượng....")
    mo_hinh.setInput(blob)
    cacphathien = mo_hinh.forward()
    so_khuon_mat = cacphathien.shape[2]

    #Lặp qua các phát hiện
    for i in range(0, cacphathien.shape[2]):
        #Lấy các thông tin liên quan đế phỏng đoán
        thongtin = cacphathien[0, 0, i, 2]
        #Lọc ra các phát hiện kém bằng các lấy các thông tin lớn hơn
        #thông tin tối thiểu cần thiết
        if thongtin > confi:
            so_thu_tu += 1
            #Tính toán tọa độ x, y của hộp bao quanh đối tượng
            hopbao = cacphathien[0, 0, i, 3:7] * np.array([w, h, w, h])
            (X_dau, Y_dau, X_cuoi, Y_cuoi) = hopbao.astype("int")

            #Vẽ hộp bao xung quanh khuôn mặt với các thông tin liên quan
            noidung = "{:.2f}%".format(thongtin * 100)
            y = Y_dau - 10 if Y_dau - 10 > 10 else Y_dau + 10
            #Cắt hình ảnh khuôn mặt
            if thu_nho:
                x_dau_cat = (X_dau - 10) * 10
                x_cuoi_cat = (X_cuoi + 10) * 10
                y_dau_cat = (Y_dau - 10) * 10
                y_cuoi_cat = (Y_cuoi + 10) * 10
            else:
                x_dau_cat = X_dau - 10
                x_cuoi_cat = X_cuoi + 10
                y_dau_cat = Y_dau - 10
                y_cuoi_cat = Y_cuoi + 10
            hinh_khuon_mat = hinhanh_goc[
                    y_dau_cat:y_cuoi_cat,
                    x_dau_cat:x_cuoi_cat,
                    ]
            '''
            cv2.rectangle(
                    hinhanh_goc,
                    (X_dau, Y_dau),
                    (X_cuoi, Y_cuoi),
                    (0, 0, 255),
                    2,
                    )
            cv2.putText(
                    hinhanh_goc,
                    noidung,
                    (X_dau, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    2,
                    )

            '''
            #Hiện ảnh kết quả
            #print('Hiện kết quả...')
            #cv2.imshow("Ket qua", hinh_khuon_mat)
            ten_anh = 'Khuon_mat_%s.jpg' % (so_thu_tu)
            duong_dan_luu_anh = os.path.join(duong_dan_luu, ten_anh)
            print('[INFO] Cắt được ảnh tại %s' % (duong_dan_luu_anh))
            try:
                cv2.imwrite(
                        duong_dan_luu_anh,
                        hinh_khuon_mat,
                        )
            except:
                continue

if __name__ == '__main__':
    #Bắt đầu chương trình
    print("Bắt đầu chạy chương trình")
    print("Lấy các tùy chọn...")

    #Tạo các tùy chọn và lấy tùy chọn
    tuychon = argparse.ArgumentParser()
    tuychon.add_argument(
            "-d",
            "--images",
            required=True,
            help="Đường dẫn thư mục hình ảnh",
            )
    tuychon.add_argument(
            "-p",
            "--prototxt",
            required=True,
            help="Đường dẫn đến tệp prototxt để triển khai Caffe",
            )
    tuychon.add_argument(
            "-m",
            "--model",
            required=True,
            help="Đường dẫn đến mô hình Caffe đã học",
            )
    tuychon.add_argument(
            "-c",
            "--confidence",
            type=float,
            default=0.5,
            help="Xác suất tối thiểu để lọc các phát hiện kém",
            )
    tuychon.add_argument(
            '-l',
            '--noi-luu',
            required=True,
            help='Nơi lưu hình ảnh',
            )
    cactuychon = vars(tuychon.parse_args())

    #Tải mô hình được tuần tự hóa từ đĩa
    print("[INFO] Đang tải mô hình...")
    mo_hinh = cv2.dnn.readNetFromCaffe(cactuychon["prototxt"], cactuychon["model"])
    #Lấy đường dẫn ảnh
    duong_dan_anh = sorted(list(paths.list_images(cactuychon['images'])))
    confi = cactuychon['confidence']
    so_thu_tu = 0
    duong_dan_luu = cactuychon['noi_luu']
    if not os.path.exists(duong_dan_luu):
        os.mkdir(duong_dan_luu)
    #Lặp qua từng đường dẫn ảnh
    for duong_dan in duong_dan_anh:
        cat_anh(duong_dan, confi, mo_hinh, duong_dan_luu, so_thu_tu)
        so_thu_tu += 1

    cv2.waitKey(0)
    print('Thoát chương trình')
