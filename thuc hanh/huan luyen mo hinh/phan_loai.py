# coding: utf-8
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os


# Xây dựng tham số tùy chọn và lấy tham số tùy chọn
tuy_chon = argparse.ArgumentParser()
tuy_chon.add_argument(
        '-m',
        '--model',
        required=True,
        help='Đường dẫn đến mô hình đã huấn luyện',
        )
tuy_chon.add_argument(
        '-l',
        '--labelbin',
        required=True,
        help='Đường dẫn đến tệp nhãn nhị phân',
        )
tuy_chon.add_argument(
        '-i',
        '--image',
        required=True,
        help='Đường dẫn đến ảnh đầu vào',
        )
cac_tuy_chon = vars(tuy_chon.parse_args())

#Đọc ảnh
hinh_anh = cv2.imread(cac_tuy_chon['image'])
ket_qua = hinh_anh.copy()

#Tiền xử lý hình ảnh để chuẩn bị phân loại
hinh_anh = cv2.resize(hinh_anh, (96, 96))
hinh_anh = hinh_anh.astype('float') / 255.0
hinh_anh = img_to_array(hinh_anh)
hinh_anh = np.expand_dims(hinh_anh, axis=0)

#Tải mô hình mạng lưới đơn giản đã huấn luyện và các nhãn đã nhị phân
#hóa
print('[INFO] Tải mô hình...')
mo_hinh = load_model(cac_tuy_chon['model'])
nhan_nhi_phan = pickle.loads(
        open(cac_tuy_chon['labelbin'], 'rb').read(),
        )

#Phân loại ảnh đầu vào
print('[INFO] Tiến hành phân loại ảnh...')
thuoc_tinh = mo_hinh.predict(hinh_anh)[0]
chi_so = np.argmax(thuoc_tinh)
nhan = nhan_nhi_phan.classes_[chi_so]

#Đặt kết quả dự đoán là 'chính xác' cho các ảnh đầu vào có tên tệp
#chứa nội dung nhãn đã được dự đoán (rõ ràng đây là cách đã dùng để
#đặt tên ảnh
ten_tep = cac_tuy_chon ['image'][
        cac_tuy_chon['image'].rfind(os.path.sep) + 1:
        ]
chinh_xac = 'chinh xac' if ten_tep.rfind(nhan) != -1 else 'sai'

#Xây dựng một nhãn mới và hiển thịlên màn hình
nhan = '{}: {:.2f}% ({})'.format(
        nhan,
        thuoc_tinh[chi_so] * 100,
        chinh_xac,
        )
ket_qua = imutils.resize(ket_qua, width=400)
cv2.putText(
        ket_qua,
        nhan,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        )

#Hiển thị ảnh kết quả
print('[INFO] {}'.format(nhan))
cv2.imshow('Ket qua', ket_qua)
cv2.waitKey(0)
