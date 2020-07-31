# coding: utf-8
#Thiết lập matplotlib chạy nền để các số liệu được lưu trong nền
import matplotlib
matplotlib.use('Agg')

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from vgg_net import MangVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os


#Xây dựng tham số tùy chọn và lấy tham số tùy chọn
tuy_chon = argparse.ArgumentParser()
tuy_chon.add_argument(
        '-d',
        '--dataset',
        required=True,
        help='Đường dẫn dến tập dữ liệu (Thư mục ảnh)',
        )
tuy_chon.add_argument(
        '-m',
        '--model',
        required=True,
        help='Đường dẫn đến nhãn nhị phân kết quả',
        )
tuy_chon.add_argument(
        '-l',
        '--labelbin',
        required=True,
        help='Đường dẫn đến tệp nhãn nhị phân',
        )
tuy_chon.add_argument(
        '-p',
        '--plot',
        type=str,
        default='plot.png',
        help='Đường dẫn của biểu đồ chính xác/sai số',
        )
cac_tuy_chon = vars(tuy_chon.parse_args())

#Khởi tạo số lần lặp lại việc huấn luyện (epochs), tỉ lệ học được,
#kích thước batch, và kích thước ảnh
EPOCHS = 100
TI_LE_HOC = 1e-3
BATCH = 32
KICH_THUOC_ANH = (96, 96, 3)

#Khởi tạo tập dữ liệu và nhãn
du_lieu = []
nhan = []

#Lấy đường dẫn ảnh và xáo trộn ngẫu nhiên
print('[INFO] Đang xem ảnh...')
duong_dan_anh = sorted(list(paths.list_images(cac_tuy_chon['dataset'])))
random.seed(42)
random.shuffle(duong_dan_anh)

#Lặp qua từng ảnh đầu vào
for duong_dan in duong_dan_anh:
    #Xem ảnh, tiền xử lý và lưu vào danh sách dữ liệu
    hinh_anh = cv2.imread(duong_dan)
    hinh_anh = cv2.resize(
            hinh_anh,
            (KICH_THUOC_ANH[1], KICH_THUOC_ANH[0]),
            )
    hinh_anh = img_to_array(hinh_anh)
    du_lieu.append(hinh_anh)

    #Lấy tên lớp từ đường dẫn ảnh và đưa vào danh sách tên nhãn
    ten_nhan = duong_dan.split(os.path.sep)[-2]
    nhan.append(ten_nhan)

#Điều chỉnh cường độ điểm ảnh thô về dải [0, 1]
du_lieu = np.array(du_lieu, dtype='float') / 255.0
nhan = np.array(nhan)
print('[INFO] Ma trận dữ liệu: {:.2f} MB'.format(
    du_lieu.nbytes / (1024 * 1000.0),
    ))

#Nhị phân hóa nhãn
nhan_nhi_phan = LabelBinarizer()
nhan = nhan_nhi_phan.fit_transform(nhan)

#Phân vùng dữ liệu thành dữ liệu huấn luyện và dữ liệu kiểm tra, sử
#dụng 80% dữ liệu để huấn luyện, 20% còn lại để kiểm tra
(X_huan_luyen, X_kiem_tra, Y_huan_luyen, Y_kiem_tra) = train_test_split(
        du_lieu,
        nhan,
        test_size = 0.2,
        random_state = 42,
        )

#Xây dựng một trình tạo ảnh để tăng số lượng dữ liệu
tang_anh = ImageDataGenerator(
        rotation_range = 25,
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        fill_mode = 'nearest',
        )

#Khởi tạo mô hình
print('[INFO] Khởi tạo mô hình...')
mo_hinh = MangVGGNet.xay_dung(
        width = KICH_THUOC_ANH[1],
        height = KICH_THUOC_ANH[0],
        depth = KICH_THUOC_ANH[2],
        classes = len(nhan_nhi_phan.classes_),
        )
thiet_lap = Adam(
        lr = TI_LE_HOC,
        decay = TI_LE_HOC / EPOCHS,
        )
mo_hinh.compile(
        #loss='categorical_crossentropy',
        loss='sparse_categorical_crossentropy',
        optimizer = thiet_lap,
        metrics=['accuracy'],
        )

#Huấn luyện mạng lưới
print('[INFO] Đang huấn luyện mạng lưới...')
H = mo_hinh.fit_generator(
        tang_anh.flow(
            X_huan_luyen,
            Y_huan_luyen,
            batch_size = BATCH,
            ),
        validation_data = (X_kiem_tra, Y_kiem_tra),
        steps_per_epoch = len(X_huan_luyen) // BATCH,
        epochs = EPOCHS,
        verbose = 1,
        )

#Lưu mô hình ra tệp
print('[INFO] Lưu mô hình...')
mo_hinh.save(cac_tuy_chon['model'])

#Lưu nhãn nhị phân ra tệp
print('[INFO] Lưu nhãn nhị phân...')
tep = open(cac_tuy_chon['labelbin'], 'wb')
tep.write(pickle.dumps(nhan_nhi_phan))
tep.close()

#Vẽ biểu đồ hiển thị độ chính xác và sai số của quá trình huấn luyện
plt.style.use('ggplot')
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history['loss'], label='sai so huan luyen')
plt.plot(np.arange(0, N), H.history['val_loss'], label='sai so kiem tra')
plt.plot(np.arange(0, N), H.history['accuracy'], label='do chinh xac huan luyen')
plt.plot(np.arange(0, N), H.history['val_accuracy'], label='do chinh xac kiem tra')
plt.title('Sai so va do chinh xac')
plt.xlabel('Epoch #')
plt.ylabel('Sai so/Do chinh xac')
plt.legend(loc='upper left')
plt.savefig(cac_tuy_chon['plot'])
