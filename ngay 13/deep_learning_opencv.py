# coding: utf-8
import numpy as np
import argparse
import time
import cv2


#Xây dựng tham số tùy chọn, lấy tham số
tuy_chon = argparse.ArgumentParser()
tuy_chon.add_argument(
        '-i',
        '--image',
        required=True,
        help='Đường dẫn đến ảnh đầu vào',
        )
tuy_chon.add_argument(
        '-p',
        '--prototxt',
        required=True,
        help='Đường dẫn đến tệp prototxt Caffe',
        )
tuy_chon.add_argument(
        '-m',
        '--model',
        required=True,
        help='Đường dẫn đến mô hình tiền huấn luyện Caffe',
        )
tuy_chon.add_argument(
        '-l',
        '--labels',
        required=True,
        help='Đường dẫn đến nhãn ImageNet (VD: syn-sets)',
        )
cac_tuy_chon = vars(tuy_chon.parse_args())

#Tải ảnh đầu vào
hinh_anh = cv2.imread(cac_tuy_chon['image'])

#Tải nhãn các lớp
du_lieu_hang = open(cac_tuy_chon['labels']).read().strip().split('\n')
cac_lop = [ hang[hang.find(' ') + 1:].split(',')[0]
        for hang in du_lieu_hang
        ]

#mô hình CNN yêu cầu chỉnh kích thước không gian của ảnh đầu vào nên
#cần chuyển ảnh về kích thước 224x224 pixel khi áp dụng phép trừ trung
#bình (mean subtraction) (104, 117, 123) để chuẩn hóa đầu vào, sau khi
#thực hiện phép chuyển này, đối tượng blob sẽ có dạng (1, 3, 224, 224)
blob = cv2.dnn.blobFromImage(
        hinh_anh,
        1,
        (224, 224),
        (104, 117, 123),
        )

#Tải mô hình
print('[INFO] Tải mô hình...')
mo_hinh = cv2.dnn.readNetFromCaffe(
        cac_tuy_chon['prototxt'],
        cac_tuy_chon['model'],
        )

#Thiết lập đầu vào cho mô hình mạng là đối tượng blob và thực hiện
#chuyển tiếp để lấy kết quả phân loại
mo_hinh.setInput(blob)
bat_dau = time.time()
du_doan = mo_hinh.forward()
ket_thuc = time.time()
print('[INFO] Quá trình phân loại hết: {:.5} giây'.format(
    ket_thuc - bat_dau))

#Sắp xếp chỉ số của các xác suất theo thứ tự giảm dần và lấy 5 xác
#suất đầu tiên (chính xác nhất)
cac_chi_so = np.argsort(du_doan[0])[::-1][:5]

#Lặp qua 5 xác suất và hiển thị chúng
for (i, chi_so) in enumerate(cac_chi_so):
    #Hiển thị xác suất trên ảnh đầu vào
    if i == 0:
        noi_dung = 'Nhan: {}, {:.2f}%'.format(
                cac_lop[chi_so],
                du_doan[0][chi_so] * 100,
                )
        cv2.putText(
                hinh_anh,
                noi_dung,
                (5, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                )

    #Hiển thị nhãn độ chính xác của dự doán và xác xuất ra màn hình
    #terminal
    print('[INFO] {}. nhan: {}, xac suat: {:.5}'.format(
        i + 1,
        cac_lop[chi_so],
        du_doan[0][chi_so],
        ))

#Hiển thị ảnh kết quả
cv2.imshow('Hinh anh', hinh_anh)
cv2.waitKey(0)
