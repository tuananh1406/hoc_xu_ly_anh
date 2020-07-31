# coding: utf-8
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense
from keras.utils import np_utils
from imutils import paths
import numpy as np
import argparse
import cv2
import os


def hinh_anh_sang_vector(hinh_anh, size=(32, 32)):
    #Chỉnh cỡ ảnh sang cỡ cố định, sau đó cắt ảnh sang một danh sách
    #cường độ của các điểm ảnh thô
    return cv2.resize(hinh_anh, size).flatten()

#Xây dựng các tham số tùy chọn và lấy các tham số
tuy_chon = argparse.ArgumentParser()
tuy_chon.add_argument(
        '-d',
        '--dataset',
        required=True,
        help='Đường dẫn đến tập dữ liệu',
        )
tuy_chon.add_argument(
        '-m',
        '--model',
        required=True,
        help='Đường dẫn của tệp model kết quả',
        )
cac_tuy_chon = vars(tuy_chon.parse_args())

#Lấy danh sách của các ảnh sẽ được dùng để miêu tả
print('[INFO] Tải dữ liệu ảnh...')
danh_sach_duong_dan_anh = list(paths.list_images(cac_tuy_chon['dataset']))

#Khởi tạo ma trận dữ liệu và danh sách nhãn
du_lieu = []
cac_nhan = []

#Lặp qua các ảnh đầu vào
for (i, duong_dan_anh) in enumerate(danh_sach_duong_dan_anh):
    #Tải ảnh và tách lớp nhãn (giả sử rằng đường dẫn có dạng:
    #/đường/dẫn/đến/ảnh/{lớp}.{số thứ tự ảnh}.jpg
    hinh_anh = cv2.imread(duong_dan_anh)
    nhan = duong_dan_anh.split(os.path.sep)[-1].split('.')[0]

    #Xây dựng thuộc tính vector các cường độ của điểm ảnh thô, sau đó
    #cập nhật vào ma trận dữ liệu và danh sách nhãn
    cac_thuoc_tinh = hinh_anh_sang_vector(hinh_anh)
    du_lieu.append(cac_thuoc_tinh)
    cac_nhan.append(nhan)

    #Hiển thị một thông báo cho mỗi 1000 ảnh
    if i > 0 and i % 1000 == 0:
        print('[INFO] xử lý xong {}/{} ảnh'.format(
            i,
            len(danh_sach_duong_dan_anh),
            ))

#Mã hóa danh sách nhãn, chuyển từ chuỗi sang số nguyên
ma_hoa = LabelEncoder()
cac_nhan = ma_hoa.fit_transform(cac_nhan)

#Mở rộng các điểm ảnh đầu vào sang dải [0, 1], sau đó chuyển danh sách
#nhãn sang vector theo dải [0, số lớp] --cái này sẽ tạo ra một vector
#cho mỗi một nhãn với chỉ số của nhãn được đặt là '1' và các giá trị
#còn lại đặt là '0'
du_lieu = np.array(du_lieu) / 255.0
cac_nhan = np_utils.to_categorical(cac_nhan, 2)

#Phân vùng dữ liệu vào dữ liệu học và dữ liệu kiểm tra, sử dụng 75% dữ
#liệu để học và 25% để kiểm tra mô hình
print('[INFO] Chia dữ liệu thành dữ liệu học và dữ liệu kiểm tra...')
(
    du_lieu_hoc,
    du_lieu_kiem_tra,
    nhan_du_lieu_hoc,
    nhan_du_lieu_kiem_tra,
    ) = train_test_split(
            du_lieu,
            cac_nhan,
            test_size=0.25,
            random_state=42,
            )

#Định nghĩa kiến trúc của mạng
mo_hinh = Sequential()
mo_hinh.add(Dense(
    768,#Lớp ẩn thứ nhất có 768 nút
    input_dim=3072, #Số nút của lớp đầu vào, 32x32x3 (sizexchannels)
    init='uniform',
    activation='relu',
    ))
mo_hinh.add(Dense(
    384,
    activation='relu',
    kernel_initializer='uniform',
    ))
mo_hinh.add(Dense(2))
mo_hinh.add(Activation('softmax'))
#Huấn luyện mô hình sử dụng SGD (Stochastic Gradient Descent)
print('[INFO] Huấn luyện mô hình...')
sgd = SGD(lr=0.01) #learning rate = 0.01
mo_hinh.compile(
        loss='binary_crossentropy',#Hàm sai số, nếu nhãn lớp >2 thì
        #nên dùng crossentropy thay vì binary_crossentropy
        optimizer=sgd,
        metrics=['accuracy'],
        )
mo_hinh.fit(
        du_lieu_hoc,
        nhan_du_lieu_hoc,
        epochs=50, #mô hình sẽ học mỗi ảnh 50 lần
        batch_size=128,
        verbose=1,
        )

#Hiển thị độ chính xác trên tập dữ liệu kiểm tra
print('[INFO] Đánh giá độ chính xác trên dữ liệu kiểm tra...')
(sai_so, do_chinh_xac) = mo_hinh.evaluate(
        du_lieu_kiem_tra,
        nhan_du_lieu_kiem_tra,
        batch_size=128,
        verbose=1,
        )
print('[INFO] Sai số: {:.4f}'.format(sai_so))
print('[INFO] Độ chính xác: {:.4f}%'.format(do_chinh_xac * 100))

#Sao lưu kiến trúc và quy mô của mạng vào tệp
print('[INFO] Sao lưu kiến trúc và quy mô của mạng vào tệp...')
mo_hinh.save(cac_tuy_chon['model'])
