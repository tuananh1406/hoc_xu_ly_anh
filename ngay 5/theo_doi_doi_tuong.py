# coding: utf-8
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time


#Xây dựng các tham số tùy chọn và lấy tham số tùy chọn
tuy_chon = argparse.ArgumentParser()
tuy_chon.add_argument(
        "-v",
        "--video",
        help="Đường dẫn tệp video (tùy chọn)",
        )
tuy_chon.add_argument(
        '-b',
        '--buffer',
        type=int,
        default=64,
        help='kích thước tối đa của vùng đệm',
        )
cac_tuy_chon = vars(tuy_chon.parse_args())

#Xác định ranh giới trên và dưới của màu xanh lục theo phổ màu HSV
#Sau đó khởi tạo danh sách các điểm đã theo dõi
ranh_gioi_duoi = (29, 86, 6)
ranh_gioi_tren = (64, 255, 255)
pts = deque(maxlen=cac_tuy_chon['buffer'])

#Nếu đường dẫn tệp video không được đưa ra, lấy hình ảnh từ camera
if not cac_tuy_chon.get('video', False):
    video = VideoStream(src=0).start()
#Nếu có đường dẫn, lấy tệp video
else:
    video = cv2.VideoCapture(cac_tuy_chon['video'])

#Đợi camera hoặc video được tải vào
time.sleep(2.0)

#Tạo vòng lặp vô hạn
while True:
    #Lấy khung hình hiện tại
    khung_hinh = video.read()

    #Tải khung hình từ camera hoặc tệp video
    khung_hinh = khung_hinh[1] if cac_tuy_chon.get('video', False) else khung_hinh

    #Nếu đang xem video mà không có khung hình nào tức là video đã hết
    if khung_hinh is None:
        break
    #Chỉnh kích thước khung hình, làm mịn, chuyển sang hệ màu HSV
    khung_hinh = imutils.resize(khung_hinh, width=600)
    lam_min = cv2.GaussianBlur(khung_hinh, (11, 11), 0)
    hsv = cv2.cvtColor(lam_min, cv2.COLOR_BGR2HSV)

    #Xây dựng một mặt nạ cho màu xanh lục, sau đó thực hiện giãn và làm mịn
    #viền để loại bỏ các hạt bụi nhỏ trong mặt nạ
    mat_na = cv2.inRange(hsv, ranh_gioi_duoi, ranh_gioi_tren)
    mat_na = cv2.erode(mat_na, None, iterations=2) #Làm mịn đường viền
    mat_na = cv2.dilate(mat_na, None, iterations=2) #Loại bỏ bụi ở trong

    #Tìm các đường viền trong mặt nạ và khởi tọa tọa độ tâm (x,y) của
    #quả bóng
    cac_duong_vien = cv2.findContours(
            mat_na.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
            )
    cac_duong_vien = imutils.grab_contours(cac_duong_vien)
    toa_do_tam = None

    #Chỉ thực hiện xử lý nếu có ít nhất một đường viền được tìm thấy
    if len(cac_duong_vien) > 0:
        #Tìm đường viền lớn nhất trong mặt nạ, sau đó sử dụng nó để
        #tính toán vòng tròn nhỏ nhất bao quanh nó và tâm 
        duong_tron = max(cac_duong_vien, key=cv2.contourArea)
        ((x, y), ban_kinh) = cv2.minEnclosingCircle(duong_tron)
        #Tính toán moment để tìm tâm của đối tượng
        moment = cv2.moments(duong_tron)
        toa_do_tam = (
                int( moment['m10'] / moment['m00']),
                int( moment['m01'] / moment['m00']),
                )
        #Chỉ xử lý nếu bán kính có kích thước nhỏ nhất
        if ban_kinh > 10:
            #Vẽ hình tròn và tâm lên khung hình, sau đó cập nhật danh
            #sách theo dõi các điểm
            cv2.circle(
                    khung_hinh,
                    (int(x), int(y)),
                    int(ban_kinh),
                    (0, 255, 255),
                    2,
                    )
            cv2.circle(
                    khung_hinh,
                    toa_do_tam,
                    5,
                    (0, 0, 255),
                    -1,
                    )
        #Cập nhật danh sách các điểm tọa độ tâm
        pts.appendleft(toa_do_tam)

    #Lặp qua danh sách các điểm được theo dõi
    for i in range(1, len(pts)):
        #Nếu điểm theo dõi là None thì bỏ qua nó
        if pts[i - 1] is None or pts[i] is None:
            continue
        #Nếu không, tính toán độ dày của đường kẻ và vẽ một đường kẻ
        do_day = int(np.sqrt( cac_tuy_chon['buffer'] / float(i + 1)) * 2.5)
        cv2.line(
                khung_hinh,
                pts[i - 1],
                pts[i],
                (0, 0, 255),
                do_day,
                )

    #Hiển thị kết quả
    cv2.imshow(
            'khung hinh',
            khung_hinh,
            #mat_na,
            )
    key = cv2.waitKey(1) & 0xFF
    #Ấn 'q' để  thoát chương trình
    if key == ord('q'):
        print('[INFO] Thoát chương trình')
        break

#Dọn dẹp chương trình
#Nếu không sử dụng tệp video, dừng trực tiếp từ camera
if not cac_tuy_chon.get('video', False):
    video.stop()
#Nếu không, giải phóng camera
else:
    video.release()
cv2.destroyAllWindows()
