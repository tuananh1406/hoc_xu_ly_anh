#coding: utf-8
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils.video import FileVideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2
import os


def chuong_canh_bao(duong_dan):
    #Phát nhạc cảnh báo
    playsound.playsound(duong_dan)

def ti_le_mat(mat):
    #Tính tỉ lệ trên hình của mắt
    #Bằng cách tính khoảng cách theo Euclidean giũa 2 tập tọa độ (x,
    #y) theo chiều ngang của các điểm mốc của mắt
    A = dist.euclidean(mat[1], mat[5])
    B = dist.euclidean(mat[2], mat[4])

    #Tương tự tính khoảng cách Euclidean giữa 2 tập tọa độ (x, y) theo
    #chiều dọc của các điểm mốc của mắt
    C = dist.euclidean(mat[0], mat[3])

    #Tính tỉ lệ khung hình của mắt
    ti_le = (A + B) / (2.0 * C)
    return ti_le

#Xây dựng tham số tùy chọn và lấy tham số tùy chọn
tuy_chon = argparse.ArgumentParser()
tuy_chon.add_argument(
        '-p',
        '--shape-predictor',
        required=True,
        help="Đường dẫn tệp nhận diện điểm mốc",
        )
tuy_chon.add_argument(
        '-a',
        '--alarm',
        type=str,
        default='',
        help='Đường dẫn âm thanh cảnh báo',
        )
tuy_chon.add_argument(
        '-w',
        '--webcam',
        type=int,
        default=0,
        help='Chỉ số của webcam trong hệ thống',
        )
tuy_chon.add_argument(
        '-v',
        '--video',
        type=str,
        help='Đường dẫn tệp video',
        )
cac_tuy_chon = vars(tuy_chon.parse_args())

#Khởi tạo 2 biến, biến tỉ lệ khung hình của mắt để tính số lần nháy
#mắt, 1 biến cho số khung hình liên tiếp mắt nhắm - biến này nếu vượt
#ngưỡng nhất định sẽ được xem là ngủ gật
NGUONG_TI_LE_MAT = 0.3
SO_KHUNG_HINH = 48

#Khởi tạo bộ đếm khung hình cùng với biến trạng thái của âm cảnh báo
BO_DEM = 0
CHUONG_BAO = False

#Khởi tạo bộ nhận diện khuôn mặt (HOG-based) và sau đó tạo ra bộ nhận
#diện các điểm mốc trên mặt
print('[INFO] Tải tệp nhận diện điểm mốc...')
nhan_dien_khuon_mat = dlib.get_frontal_face_detector()
nhan_dien_diem_moc = dlib.shape_predictor(cac_tuy_chon['shape_predictor'])

#lấy các chỉ số điểm mốc của mắt trái và mắt phải
(dau_trai, cuoi_trai) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(dau_phai, cuoi_phai) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

#Bắt đầu quay phim
print('[INFO] Bắt đầu nhận hình ảnh...')
if cac_tuy_chon['webcam']:
    print('[INFO] Bật camera...')
    video = VideoStream(src=cac_tuy_chon['webcam']).start()
    truc_tiep = False
if cac_tuy_chon['video']:
    print('[INFO] Mở tệp phim...')
    video = FileVideoStream(cac_tuy_chon['video']).start()
    truc_tiep = True
time.sleep(1.0)

#Lặp qua các khung hình của phim
while True:
    #Kiểm tra xem nếu đọc từ tệp thì kiểm tra có bị tràn bộ nhớ đệm
    #không
    if truc_tiep and not video.more():
        break

    #Lấy khung hình, thay đổi kích cỡ và chuyển sang đen trắng
    khung_hinh = video.read()
    if khung_hinh is None:
        break
    khung_hinh = imutils.resize(
            khung_hinh,
            width=450,
            )
    den_trang = cv2.cvtColor(
            khung_hinh,
            cv2.COLOR_BGR2GRAY,
            )

    #Tìm các khuôn mặt có trong ảnh đen trắng
    cac_khuon_mat = nhan_dien_khuon_mat(den_trang, 0)

    #Lặp qua từng khuôn mặt và tìm các điểm mốc
    for khuon_mat in cac_khuon_mat:
        #Chuyển điểm mốc từ tọa độ (x, y) sang mảng Numpy
        hinh_khuon_mat = nhan_dien_diem_moc(den_trang, khuon_mat)
        hinh_khuon_mat = face_utils.shape_to_np(hinh_khuon_mat)

        #Lấy tọa độ mắt trái, mắt phải, và sử dụng nó để tính tỉ lệ
        #mắt
        mat_trai = hinh_khuon_mat[dau_trai:cuoi_trai]
        mat_phai = hinh_khuon_mat[dau_phai:cuoi_phai]
        ti_le_mat_trai = ti_le_mat(mat_trai)
        ti_le_mat_phai = ti_le_mat(mat_phai)

        #Tỉ lệ trung bình của 2 mắt
        ti_le_trung_binh = (ti_le_mat_trai + ti_le_mat_phai) / 2.0

        #Tính đường bao bên người mắt trái và mắt phải, sau đó hiển
        #thị chúng trên hình ảnh của từng mắt
        duong_bao_mat_trai = cv2.convexHull(mat_trai)
        duong_bao_mat_phai = cv2.convexHull(mat_phai)
        cv2.drawContours(
                khung_hinh,
                [duong_bao_mat_trai],
                -1,
                (0, 255, 0),
                1,
                )
        cv2.drawContours(
                khung_hinh,
                [duong_bao_mat_phai],
                -1,
                (0, 255, 0),
                1,
                )
        #Kiểm tra xem nếu tỉ lệ mắt nhỏ hơn ngưỡng nháy mắt, tăng bộ
        #đếm khung hình lên
        if ti_le_trung_binh < NGUONG_TI_LE_MAT:
            BO_DEM += 1

            #Nếu các mắt nhắm lại trong thời gian lâu thì phát âm cảnh
            #báo
            if BO_DEM >= SO_KHUNG_HINH:
                #Nếu âm cảnh báo chưa bật, bật nó lên
                if not CHUONG_BAO:
                    CHUONG_BAO = True

                    #Kiểm tra xem có đường dẫn chứa tệp âm cảnh báo
                    #không, nếu có thì mở phát âm cảnh báo chạy nền
                    if cac_tuy_chon['alarm'] != '':
                        t = Thread(
                                target = chuong_canh_bao,
                                args = (cac_tuy_chon['alarm'],)
                                )
                        t.deamon = True
                        t.start()

                #Vẽ một cảnh báo lên khung hình
                cv2.putText(
                        khung_hinh,
                        'PHAT HIEN NGU GAT',
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                        )
        #Nếu tỉ lệ mắt trung bình không dưới ngưỡng, đặt lại bộ đếm và
        #âm cảnh báo
        else:
            BO_DEM = 0
            CHUONG_BAO = False

        #Vẽ tỉ lệ mắt lên khung hình để theo dõi và cài đặt ngưỡng
        #nháy mắt, bộ đếm khung hình
        cv2. putText(
                khung_hinh,
                'TI LE MAT: {:.2f}'.format(ti_le_trung_binh),
                (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                )
    #Hiển thị khung hình
    cv2.imshow('Khunh hinh', khung_hinh)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

#Dọn dẹp chương trình
cv2.destroyAllWindows()
video.stop()
