#coding: utf-8
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2


def ty_le_mat(mat):
    #Tính tỉ lệ khung hình của mắt
    #Tính khoảng cách euclidean giữa 2 tập hợp các điểm mốc dọc của mắt
    #theo hệ tọa độ (x, y)
    A = dist.euclidean(mat[1], mat[5])
    B = dist.euclidean(mat[2], mat[4])

    #Tính khoảng cách euclidean giữa 2 tập hợp các điểm mốc ngang của mắt
    #theo hệ tọa độ (x, y)
    C = dist.euclidean(mat[0], mat[3])

    #Tính tỉ lệ khung hình của mắt
    ti_le = (A + B) / (2.0 * C)
    return ti_le

#Xây dựng các tham số tùy chọn và lấy các tùy chọn
tuy_chon = argparse.ArgumentParser()
tuy_chon.add_argument(
        '-p',
        '--shape-predictor',
        required=True,
        help='Đường dẫn đến tệp nhận diện điểm mốc',
        )
tuy_chon.add_argument(
        '-v',
        '--video',
        type=str,
        default='',
        help='Đường dẫn đến tệp video',
        )
cac_tuy_chon = vars(tuy_chon.parse_args())

#Khởi tạo 2 hằng số, một cho tỉ lệ khung hình của mắt để nhận diện
#nháy mắt, một cho số các khung hình liên tiếp mà mắt nằm dưới ngưỡng
NGUONG_TI_LE_MAT = 0.3
SO_KHUNG_HINH = 3

#Khởi tạo bộ đếm khung hình và tổng số lần nháy mắt
BO_DEM = 0
TONG = 0

#Khởi tạo bộ nhận diện khuôn mặt của dlib (dựa vào HOG) và sau đó tạo
#ra bộ nhận diện các điểm mốc
print('[INFO] Đang tải tệp nhận diện điểm mốc...')
nhan_dien_khuon_mat = dlib.get_frontal_face_detector()
nhan_dien_diem_moc = dlib.shape_predictor(cac_tuy_chon['shape_predictor'])

#Lấy các chỉ số của các điểm mốc cho mắt trái và mắt phải tương ứng
(diem_dau_trai, diem_cuoi_trai) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(diem_dau_phai, diem_cuoi_phai) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

#Bắt đầu mở tệp video
print('[INFO] Mở tệp video...')
video = FileVideoStream(cac_tuy_chon['video']).start()
truc_tiep = True
#Code nếu dùng camera quay phim trực tiếp
#video = VideoStream(src=0).start()
#Nếu dùng camera của RaspberryPi
#video = VideoStream(usePiCamera=True).start()
#truc_tiep = False
time.sleep(1.0)

#Lặp qua từng khung hình của video
while True:
    #Nếu đọc video từ tệp, cần kiểm tra xem có khung hình nào tràn bộ
    #nhớ đệm hay không
    if truc_tiep and not video.more():
        break

    #Lấy khung hình từ tệp video, chỉnh cỡ, và chuyển sang đen trắng
    khung_hinh = video.read()
    if khung_hinh is None:
        break
    khung_hinh = imutils.resize(khung_hinh, width=450)
    den_trang = cv2.cvtColor(khung_hinh, cv2.COLOR_BGR2GRAY)

    #Nhận diện các khuôn mặt có trong khung hình đen trắng
    cac_khuon_mat = nhan_dien_khuon_mat(den_trang, 0)

    #Lặp qua từng khuôn mặt
    for khuon_mat in cac_khuon_mat:
        #Xác định vùng khuôn mặt, chuyển các tọa độ điểm mốc từ hệ tọa
        #độ (x, y) sang dạng mảng NumPy
        vung_khuon_mat = nhan_dien_diem_moc(den_trang, khuon_mat)
        vung_khuon_mat = face_utils.shape_to_np(vung_khuon_mat)

        #Tách tọa độ của mắt trái và mắt phải, sau đó sử dụng các tọa
        #độ để tính tỉ lệ hình mắt cho cả 2 mắt 
        mat_trai = vung_khuon_mat[diem_dau_trai:diem_cuoi_trai]
        mat_phai = vung_khuon_mat[diem_dau_phai:diem_cuoi_phai]
        ti_le_mat_trai = ty_le_mat(mat_trai)
        ti_le_mat_phai = ty_le_mat(mat_phai)

        #Tính giá trị trung bình cho cả 2 mắt
        ti_le_mat_trung_binh = (ti_le_mat_trai + ti_le_mat_phai) / 2.0

        #Tính hình bao bên ngoài cho mắt trái và mắt phải, sau đó hiển
        #thị cho từng mắt
        hinh_bao_mat_trai = cv2.convexHull(mat_trai)
        hinh_bao_mat_phai = cv2.convexHull(mat_phai)
        cv2.drawContours(
                khung_hinh,
                [hinh_bao_mat_trai],
                -1,
                (0, 255, 0),
                1,
                )
        cv2.drawContours(
                khung_hinh,
                [hinh_bao_mat_phai],
                -1,
                (0, 255, 0),
                1,
                )

        #Kiểm tra nếu tỉ lệ mắt nhỏ hơn ngưỡng nháy mắt, thì tăng bộ
        #đếm khung hình
        if ti_le_mat_trung_binh < NGUONG_TI_LE_MAT:
            BO_DEM += 1

        #Nếu tỉ lệ mắt không nhỏ hơn
        else:
            #Nếu các mắt nhắm trong số khung hình đủ lớn, tăng tổng số
            #lần nháy mắt lên
            if BO_DEM >= SO_KHUNG_HINH:
                TONG += 1

            #Đặt lại bộ đếm khung hình
            BO_DEM = 0

        #Hiển thị tổng số lần nháy mắt trên khung hình cùng với tính
        #toán tỉ lệ mắt cho khung hình
        cv2.putText(
                khung_hinh,
                'So lan nhay mat: {}'.format(TONG),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                )
        cv2.putText(
                khung_hinh,
                'Ti le: {}'.format(ti_le_mat_trung_binh),
                (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                )
    #Hiển thị khung hình
    cv2.imshow('Khung hinh', khung_hinh)
    key = cv2.waitKey(1) & 0xFF

    #Thoát chương trình nếu ấn 'q'
    if key == ord('q'):
        break

#Dọn dẹp chương trình
cv2.destroyAllWindows()
video.stop()
