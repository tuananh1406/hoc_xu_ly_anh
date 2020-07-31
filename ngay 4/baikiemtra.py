# coding: utf-8
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2


#Xây dựng các tham số và lấy tham số tùy chọn
tuychon = argparse.ArgumentParser()
tuychon.add_argument(
        '-i',
        '--image',
        required=True,
        help = 'Đường dẫn hỉnh ảnh đầu vào',
        )
cactuychon = vars(tuychon.parse_args())

#Tạo từ điển với khóa là câu hỏi, giá trị là đáp án đúng của nó
ds_dap_an_dung = {
        0: 1, #Câu 1: B
        1: 4, #Câu 2: E
        2: 0, #Câu 3: A
        3: 3,
        4: 1,
        }

#Tải ảnh đầu vào, chuyển sang đen trắng, làm mịn và tìm các đối tượng khép kín
hinh_anh = cv2.imread(cactuychon['image'])
den_trang = cv2.cvtColor(
        hinh_anh,
        cv2.COLOR_BGR2GRAY,
        )
lam_min_anh = cv2.GaussianBlur(
        den_trang,
        (5, 5),
        0,
        )
doi_tuong_kin = cv2.Canny(
        lam_min_anh,
        75,
        200,
        )

#Tìm đường bao lớn nhất, sau đó phóng to nó ra toàn màn hình
duong_vien = cv2.findContours(
        doi_tuong_kin.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
        )
duong_vien = imutils.grab_contours(duong_vien)

#Đảm bảo có ít nhất 1 đường viền được tìm thấy
if len(duong_vien) > 0:
    #Sắp xếp các đường viền theo thứ tự kích cỡ giảm dần
    duong_vien = sorted(
            duong_vien,
            key = cv2.contourArea,
            reverse = True,
            )
    #Lặp qua từng đường viền
    for d in duong_vien:
        #Vẽ đường viền gần giống đường gốc
        do_dai_cung = cv2.arcLength(
                d,
                True,
                )
        duong_vien_gan_dung = cv2.approxPolyDP(
                d,
                0.02 * do_dai_cung,
                True,
                )

        #Nếu đường viền gần đúng có 4 điểm, ta có thể giả định đó là trang giấy
        if len(duong_vien_gan_dung) == 4:
            trang_giay = duong_vien_gan_dung
            break

'''
#Hiển thị ảnh đường bao của trang giấy
cv2.drawContours(
        hinh_anh,
        [trang_giay],
        -1,
        (0, 255, 0),
        2,
        )
cv2.imshow('Trang giay', hinh_anh)
'''

#Áp dụng phép chuyển 4 điểm cho ảnh gốc và ảnh đen trắng để cắt phần trang giấy
bai_lam = four_point_transform(
        hinh_anh,
        trang_giay.reshape(4, 2),
        )
bai_lam_den_trang = four_point_transform(
        den_trang,
        trang_giay.reshape(4, 2),
        )

#Sử dụng phương thức tính ngưỡng của Otsu để chuyển sang ảnh nhị phân
nguong = cv2.threshold(
        bai_lam_den_trang,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
        )[1]

'''
#Hiển thị hình ảnh ngưỡng
cv2.imshow('Hinh anh da chia nguong', nguong)
'''

#Tìm các đường viền trong ảnh đã chia ngưỡng, sau đó tính toán danh sách các
#đường viền tương ứng với các câu hỏi
cac_duong_vien = cv2.findContours(
        nguong.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
        )
cac_duong_vien = imutils.grab_contours(cac_duong_vien)
duong_vien_dap_an = []

#Lặp qua từng đường viền
for d in cac_duong_vien:
    #Tính toán đường bao của đường viền, sau đó sử dụng đường bao để lấy tỉ lệ
    #khung hình
    (x, y, w, h) = cv2.boundingRect(d)
    ti_le = w / float(h)
    
    #Một vùng được xem là đáp án nếu nó đủ rộng, đủ cao và có tỉ lệ gần bằng 1
    if w >= 20 and h >= 20 and ti_le >= 0.9 and ti_le <= 1.1:
        duong_vien_dap_an.append(d)

'''
#Hiển thị ảnh đường bao của các đáp án
cv2.drawContours(
        bai_lam,
        duong_vien_dap_an,
        -1,
        (0, 255, 0),
        2,
        )
cv2.imshow('Trang giay', bai_lam)
'''

#Sắp xếp các đường viền đáp án từ trên xuống dưới, sau đó tính toán tổng số đáp
#án đúng
duong_vien_dap_an = contours.sort_contours(
        duong_vien_dap_an,
        method = 'top-to-bottom',
        )[0]
so_cau_dung = 0

#Mỗi cau hỏi có 5 phương án chọn, lặp một câu hỏi với vòng lặp 5
for (cau_hoi, dap_an) in enumerate(
        np.arange(0, len(duong_vien_dap_an), 5),
        ):
    #Sắp xếp các đường viền của câu hỏi hiện tại từ trái sang phải, sau đó phân
    #tích chỉ số của đáp án
    cac_duong_vien_cau_hoi = contours.sort_contours(
            duong_vien_dap_an[dap_an:dap_an + 5],
            )[0]
    vung_to = None

    #Lặp qua các đường viền đã sắp xếp
    for (j, c) in enumerate(cac_duong_vien_cau_hoi):
        #Tạo một mặt nạ bao phủ đáp án hiện tại
        mat_na = np.zeros(nguong.shape, dtype='uint8')
        cv2.drawContours(mat_na, [c], -1, 255, -1)

        #Áp dụng mặt nạ cho ảnh đã chia ngưỡng, sau đó đếm số điểm ảnh khác
        #không trong vùng đáp án
        mat_na = cv2.bitwise_and(nguong, nguong, mask=mat_na)
        tong = cv2.countNonZero(mat_na)

        #Nếu tổng hiện tại có số tổng điểm ảnh khác không khá lớn, ta có thể
        #giả định nó là vùng tô đáp án
        if vung_to is None or tong > vung_to[0]:
            vung_to = (tong, j)

        '''
        #Hiển thị ảnh kết quả của từng vùng tô đáp án
        cv2.imshow('hinhanh', mat_na)
        cv2.waitKey(0)
        '''

    #Khởi tạo màu đường viền và chỉ số của đáp án đúng
    mau_duong_vien = (0, 0, 255)
    dap_an_dung = ds_dap_an_dung[cau_hoi]
    #Kiểm tra xem vùng tô màu có phải đáp án không
    if dap_an_dung == vung_to[1]:
        mau_duong_vien = (0, 255, 0)
        so_cau_dung += 1

    #Vẽ đường viền cho đáp án đúng
    cv2.drawContours(
            bai_lam,
            #[cac_duong_vien_cau_hoi[dap_an_dung]],
            [cac_duong_vien_cau_hoi[vung_to[1]]],
            -1,
            mau_duong_vien,
            3,
            )

#Tính điểm
diem = (so_cau_dung / 5.0) * 100
print('[INFO] diem thi: {:.2f}%'.format(diem))
cv2.putText(
        bai_lam,
        '{:.2f}%'.format(diem),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2,
        )
cv2.imshow('Hinh goc', hinh_anh)
cv2.imshow('Ket qua', bai_lam)

cv2.waitKey(0)
