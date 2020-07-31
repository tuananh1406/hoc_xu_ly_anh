# coding: utf-8
from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2


#Xây dựng các tham số cần thiết và lấy tham số
tuychon = argparse.ArgumentParser()
tuychon.add_argument(
        "-f",
        "--file",
        required=True,
        help="Đường dẫn đến tệp video",
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
print('[INFO] Lấy các tham số tùy chọn...')
cactuychon = vars(tuychon.parse_args())

#Tải mô hình từ đĩa
print('[INFO] Tải mô hình...')
net = cv2.dnn.readNetFromCaffe(cactuychon['prototxt'], args['model'])

#Mở video và bắt đầu bộ đếm thời gian FPS
print('[INFO] Tải tệp phim...')
video = FileVideoStream(cactuychon['file']).start()
time.sleep(1)

#Chạy bộ đếm thời gian
fps = FPS().start()

#Lặp qua các khung hình từ tệp phim
while video.more():
    #Lấy khung hình từ tệp phim và đổi cỡ về độ rộng lớn nhất là 400pixel
    khunghinh = video.read()
    khunghinh = imutils.resize(khunghinh, width=400)

    #Lấy kích thước khung hình và chuyển đổi nó sang blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
            cv2.resize(khunghinh, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0),
            )
    #Đưa blob vào mạng và tìm kiếm các sự phát hiện và phỏng đoán
    net.setInput(blob)
    cacphathien = net.forward()

    #Lặp qua các phát hiện 
    for i in range(0, cacphathien.shape[2]):
        #Tách các thông tin liên quan đến dự đoán
        thongtin = cacphathien[0, 0, i, 2]

        #Loại bỏ các phát hiện kém bằng thông tin confidence
        #nhập vào (mặc định là 0.5)
        if thongtin < cactuychon['confidence']:
            continue

        #Tính toán tọa độ x, y của hộp bao cho đối tượng
        hopbao = cacphathien[0, 0, i, 3:7] * np.array([w, h, w, h])
        (X_dau, Y_dau, X_cuoi, Y_cuoi) = hopbao.astype('int')

        #Vẽ hộp bao xung quanh khuôn mặt nhận diện được
        noidung = "{:.2f}%".format(thongtin * 100)
        y = Y_dau - 10 if Y_dau - 10 > 10 else Y_dau + 10
        cv2.rectangle(
                khunghinh,
                (X_dau, Y_dau),
                (X_cuoi, Y_cuoi),
                (0, 0, 255),
                2,
                )
        cv2.putText(
                khunghinh,
                noidung,
                (X_dau, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                2,
                )

    #Hiện kết quả
    cv2.imshow('Ket qua', khunghinh)
    key = cv2.waitKey(1) & 0xFF

    #Ấn 'q' để thoát chương trình
    if key == ord('q'):
        print('[INFO] Thoát chương trình...')
        break

#Dọn dẹp chương trình
cv2.destroyAllWindows()
vs.stop()
