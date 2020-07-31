# coding: utf-8
import numpy as np
import argparse
import cv2


#Bắt đầu chương trình
print("Bắt đầu chạy chương trình")
print("Lấy các tùy chọn...")

#Tạo các tùy chọn và lấy tùy chọn
tuychon = argparse.ArgumentParser()
tuychon.add_argument(
        "-i",
        "--image",
        required=True,
        help="Đường dẫn hình ảnh",
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
cactuychon = vars(tuychon.parse_args())

#Tải mô hình được tuần tự hóa từ đĩa
print("[INFO] Đang tải mô hình...")
net = cv2.dnn.readNetFromCaffe(cactuychon["prototxt"], cactuychon["model"])

#Tải ảnh đầu vào và đổi cỡ nếu ảnh quá lớn
hinhanh = cv2.imread(cactuychon["image"])
(h, w) = hinhanh.shape[:2]
while h > 1000 or w > 1000:
    print('Hình ảnh quá lớn (%s x %s)' % (w, h))
    print('Tiến hành thu nhỏ ảnh')
    hinhanh = cv2.resize(hinhanh, ((int(w / 10)), (int(h / 10))))
    (h, w) = hinhanh.shape[:2]
    print('Thu nhỏ về (%s x %s)' % (w, h))

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
net.setInput(blob)
cacphathien = net.forward()

#Lặp qua các phát hiện
for i in range(0, cacphathien.shape[2]):
    #Lấy các thông tin liên quan đế phỏng đoán
    thongtin = cacphathien[0, 0, i, 2]
    #Lọc ra các phát hiện kém bằng các lấy các thông tin lớn hơn
    #thông tin tối thiểu cần thiết
    if thongtin > cactuychon['confidence']:
        #Tính toán tọa độ x, y của hộp bao quanh đối tượng
        hopbao = cacphathien[0, 0, i, 3:7] * np.array([w, h, w, h])
        (X_dau, Y_dau, X_cuoi, Y_cuoi) = hopbao.astype("int")

        #Vẽ hộp bao xung quanh khuôn mặt với các thông tin liên quan
        noidung = "{:.2f}%".format(thongtin * 100)
        y = Y_dau - 10 if Y_dau - 10 > 10 else Y_dau + 10
        cv2.rectangle(
                hinhanh,
                (X_dau, Y_dau),
                (X_cuoi, Y_cuoi),
                (0, 0, 255),
                2,
                )
        cv2.putText(
                hinhanh,
                noidung,
                (X_dau, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                2,
                )

#Hiện ảnh kết quả
print('Hiện kết quả...')
cv2.imshow("Ket qua", hinhanh)
cv2.waitKey(0)
print('Thoát chương trình')
