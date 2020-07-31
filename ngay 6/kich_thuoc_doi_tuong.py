# coding: utf-8
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2


#Khai báo hàm tìm điểm nằm giữa 2 điểm (trung điểm)
def trung_diem(diem_A, diem_B):
    return (
            (diem_A[0] + diem_B[0]) * 0.5,
            (diem_A[1] + diem_B[1]) * 0.5,
            )

#Xây dựng các tham số tùy chọn và lấy tùy chọn
tuy_chon = argparse.ArgumentParser()
tuy_chon.add_argument(
        '-i',
        '--image',
        required=True,
        help='Đường dẫn đến ảnh đầu vào',
        )
tuy_chon.add_argument(
        '-w',
        '--width',
        type=float,
        required=True,
        help='''Chiều dài của đối tượng nằm ngoài cùng bên trái bức
        ảnh (inches)''',
        )
cac_tuy_chon = vars(tuy_chon.parse_args())

#Tải hình ảnh, chuyển sang đen trắng, làm mịn
hinh_anh = cv2.imread(cac_tuy_chon['image'])
den_trang = cv2.cvtColor(hinh_anh, cv2.COLOR_BGR2GRAY)
den_trang = cv2.GaussianBlur(den_trang, (7, 7), 0)

#Sử dụng bộ phát hiện cạnh, sau đó làm giãn và làm mượt đường viền
#giữa 2 cạnh của đối tượng
canh_phat_hien = cv2.Canny(den_trang, 50, 100)
canh_phat_hien = cv2.dilate(canh_phat_hien, None, iterations=1)
canh_phat_hien = cv2.erode(canh_phat_hien, None, iterations=1)

#Tìm các đường viền nằm trong vùng bao phủ bởi cạnh
cac_duong_vien = cv2.findContours(
        canh_phat_hien.copy(), #Nhân bản ảnh các cạnh
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
        )
cac_duong_vien = imutils.grab_contours(cac_duong_vien)

#Sắp xếp các đường viền từ trái qua phải và khởi tạo giá trị số điểm
#ảnh trên 1 đơn vị đo
(cac_duong_vien, _) = contours.sort_contours(cac_duong_vien)
don_vi = None

#Lặp trên từng vòng lặp riêng lẻ
for duong_vien in cac_duong_vien:
    #Nếu đường viền không đủ lớn, bỏ qua nó
    if cv2.contourArea(duong_vien) < 100:
        continue

    #Tính toán đường bao nghiêng của đối tượng
    anh_goc = hinh_anh.copy()
    duong_bao = cv2.minAreaRect(duong_vien)
    #Nếu OpenCV 2.4
    if imutils.is_cv2():
        duong_bao = cv2.cv.BoxPoints(duong_bao)
    #Nếu là OpenCV 3
    else:
        duong_bao = cv2.boxPoints(duong_bao)
    duong_bao = np.array(duong_bao, dtype='int')

    #Sắp xếp các điểm trong đường viền theo hiển thị từ trái qua phải,
    #trên xuống dưới, sau đó vẽ đường bao nghiêng
    duong_bao = perspective.order_points(duong_bao)
    cv2.drawContours(
            anh_goc,
            [duong_bao.astype('int')],
            -1,
            (0, 255, 0),
            2,
            )

    #Lặp qua từng điểm gốc và vẽ chúng
    for (x, y) in duong_bao:
        cv2.circle(
                anh_goc,
                (int(x), int(y)),
                5,
                (0, 0, 255),
                -1,
                )

    #Chia tách đường bao nghiêng, sau đó tính trung điểm giữa
    #trái-trên và phải-trên của lưới tọa độ, sau đó làm tương tự cho
    #đường dưới
    (trai_tren, phai_tren, phai_duoi, trai_duoi) = duong_bao
    (X_duong_tren, Y_duong_tren) = trung_diem(trai_tren, phai_tren)
    (X_duong_duoi, Y_duong_duoi) = trung_diem(trai_duoi, phai_duoi)

    #Tính trung điểm cho các cạnh trái, cạnh phải
    (X_canh_trai, Y_canh_trai) = trung_diem(trai_tren, trai_duoi)
    (X_canh_phai, Y_canh_phai) = trung_diem(phai_tren, phai_duoi)

    #Vẽ các trung điểm trên hình ảnh
    cv2.circle(
            anh_goc,
            (int(X_duong_tren), int(Y_duong_tren)),
            5,
            (255, 0, 0),
            -1,
            )
    cv2.circle(
            anh_goc,
            (int(X_duong_duoi), int(Y_duong_duoi)),
            5,
            (255, 0, 0),
            -1,
            )
    cv2.circle(
            anh_goc,
            (int(X_canh_trai), int(Y_canh_trai)),
            5,
            (255, 0, 0),
            -1,
            )
    cv2.circle(
            anh_goc,
            (int(X_canh_phai), int(Y_canh_phai)),
            5,
            (255, 0, 0),
            -1,
            )

    #Vẽ các đường thẳng giữa các trung điểm
    cv2.line(
            anh_goc,
            (int(X_duong_tren), int(Y_duong_tren)),
            (int(X_duong_duoi), int(Y_duong_duoi)),
            (255, 0, 255),
            2,
            )
    cv2.line(
            anh_goc,
            (int(X_canh_trai), int(Y_canh_trai)),
            (int(X_canh_phai), int(Y_canh_phai)),
            (255, 0, 255),
            2,
            )

    #Tính toán khoảng cách Euclidean giữa các trung điểm
    khoang_cach_A = dist.euclidean(
            (X_duong_tren, Y_duong_tren),
            (X_duong_duoi, Y_duong_duoi),
            )
    khoang_cach_B = dist.euclidean(
            (X_canh_trai, Y_canh_trai),
            (X_canh_phai, Y_canh_phai),
            )


    #Nếu chưa có thông tin số điểm ảnh trên đơn vị, tính toán tỉ lệ
    #của số điểm ảnh theo đơn vị đo (trong ví dụ sử dụng inches)
    if don_vi is None:
        don_vi = khoang_cach_B / cac_tuy_chon['width']

    #Tính toán kích thước của đối tượng
    kich_thuoc_A = khoang_cach_A / don_vi
    kich_thuoc_B = khoang_cach_B / don_vi

    #Hiển thị kích thước đối tượng lên hình ảnh
    cv2.putText(
            anh_goc,
            '{:.1f}in'.format(kich_thuoc_A),
            (int(X_duong_tren - 15), int(Y_duong_tren - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            )
    cv2.putText(
            anh_goc,
            '{:.1f}in'.format(kich_thuoc_B),
            (int(X_canh_phai + 10), int(Y_canh_phai)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            )

    #Hiển thị ảnh kết quả
    cv2.imshow('Hình ảnh', anh_goc)
    cv2.waitKey(0)
