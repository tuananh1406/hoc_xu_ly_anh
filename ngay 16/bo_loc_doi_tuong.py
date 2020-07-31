#coding: utf-8
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2


#Xây dựng tham số tùy chọn và lấy tham số tùy chọn
tuy_chon = argparse.ArgumentParser()
tuy_chon.add_argument(
        '-p',
        '--prototxt',
        required=True,
        help='Đường dẫn đến tệp Caffe prototxt',
        )
tuy_chon.add_argument(
        '-m',
        '--model',
        required=True,
        help='Đường dẫn đến mô hình Caffe đã huấn luyện',
        )
tuy_chon.add_argument(
        '-c',
        '--confidence',
        type=float,
        default=0.2,
        help='Ngưỡng nhỏ nhất để loại bỏ các đối tượng sai',
        )
cac_tuy_chon = vars(tuy_chon.parse_args())

#Khởi tạo danh sách các nhãn lớp MobileNet SSD đã được huấn luyện để
#phát hiện, sau đó tạo ra một tập hợp các màu cho từng lớp
CAC_LOP = [
        "background",
        "aeroplane",
        "xe dap",
        "chim",
        "thuyen",
	"chai su",
        "xe buyet",
        "xe hoi",
        "meo",
        "ghe",
        "bo",
        "ban an",
	"cho",
        "ngua",
        "xe may",
        "nguoi",
        "cay canh",
        "cuu",
	"ghe sofa",
        "tau hoa",
        "man hinh tivi",
        ]
BO_QUA = set(['nguoi'])
MAU_SAC = np.random.uniform(0, 255, size=(len(CAC_LOP), 3))

#Tải mô hình từ tệp
print('[INFO] Tải tệp mô hình...')
mo_hinh = cv2.dnn.readNetFromCaffe(
        cac_tuy_chon['prototxt'],
        cac_tuy_chon['model'],
        )

#Khởi tạo trình quay phim, bật máy quay và khởi tạo bộ đếm FPS
print('[INFO] Bắt đầu quay phim...')
video = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

#Lặp qua từng khung hình
while True:
    #Lấy khung hình từ phim và chuyển kích cỡ sang chiều rộng tối đa
    #là 400pixel
    khung_hinh = video.read()
    if khung_hinh is None:
        break
    khung_hinh = imutils.resize(khung_hinh, width=400)

    #Lấy kích thước khung hình và chuyển hình sang dạng blob
    (h, w) = khung_hinh.shape[:2]
    blob = cv2.dnn.blobFromImage(
            cv2.resize(khung_hinh, (300, 300)),
            0.07843,
            (300, 300),
            127.5,
            )
    #Đưa blob qua mạng mô hình, bắt đầu dự đoán và tìm kiếm đối tượng
    mo_hinh.setInput(blob)
    cac_doi_tuong = net.forward()

    #Lặp qua từng đối tượng được tìm thấy
    for i in np.arrange(0, cac_doi_tuong.shape[2]):
        #Tách thông số độ tin cậy và dự đoán
        tin_cay = cac_doi_tuong[0, 0, i, 2]

        #Lọc ra các đối tượng kém bằng cách so sánh với chỉ số tin cậy
        if tin_cay > cac_tuy_chon['confidence']:
            #Lấy chỉ số của nhãn lớp từ danh sách đối tượng
            chi_so = int(cac_doi_tuong[0, 0, i, 1])

            #Nếu nhãn của đối tượng được dự đoán nằm trong danh sách
            #bỏ qua thì bỏ qua đối tượng đó
            if CAC_LOP[chi_so] in BO_QUA:
                continue

            #Tính toán tọa độ (x, y) của đường bao ngoài của đối tượng
            duong_bao = cac_doi_tuong[0, 0, i, 3:7] * np.array([w, h, w, h])
            (X_dau, Y_dau, X_cuoi, Y_cuoi) = duong_bao.astype('int')

            #Vẽ đường bao quanh đối tượng phát hiện được
            nhan = '{}: {:.2f}%'.format( CAC_LOP[chi_so], tin_cay * 100)
            cv2.rectangle(
                    khung_hinh,
                    (X_dau, Y_dau),
                    (X_cuoi, Y_cuoi).
                    MAU_SAC[chi_so],
                    2,
                    )
            y = Y_dau - 15 if Y_dau - 15 > 15 else Y_dau + 15
            cv2.putText(
                    khung_hinh,
                    nhan,
                    (X_dau, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    MAU_SAC[chi_so],
                    2,
                    )
        #Hiển thị khung hình kết quả
        cv2.imshow('Khung hinh', khung_hinh)
        key = cv2.waitKey(1) & 0xFF

        #Ấn 'q' để thoát
        if key == ord('q'):
            break

        #Cập nhật FPS
        fps.update()

#Dừng bộ đếm thời gian và hiển thị thông tin FPS
fps.stop()
print('[INFO] Tổng thời gian chạy: {:.2f}'.format(fps.elapsed()))
print('[INFO] FPS ước tính: {:.2f}'.format(fps.fps()))

#Dọn dẹp
cv2.destroyAllWindows()
video.stop()
