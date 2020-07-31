# coding: utf-8
import argparse

from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import cv2


def trich_xuat_thong_tin(cac_thong_tin, toa_do):
    '''
    Nhận vào các thông tin về số hàng và cột, sau đó khởi tạo một hình chữ
    nhật bao quanh và hiển thị độ chính xác tương ứng
    '''

    (so_hang, so_cot) = cac_thong_tin.shape[2:4]
    hinh_bao = []
    do_chinh_xac = []

    # Lặp qua các hàng
    for hang in range(0, so_hang):
        # Lấy thông tin (xác suất) cùng với thông tin tọa độ của hình bao xung
        # quanh văn bản
        xac_suat = cac_thong_tin[0, 0, hang]
        x0 = toa_do[0, 0, hang]
        x1 = toa_do[0, 1, hang]
        x2 = toa_do[0, 2, hang]
        x3 = toa_do[0, 3, hang]
        du_lieu_goc = toa_do[0, 4, hang]  # dữ liệu các góc

        # Lặp qua các cột
        for cot in range(0, so_cot):
            # Lọc bỏ các xác suất thấp hơn mức tối thiểu
            if xac_suat[cot] < cac_tuy_chon['xac_suat_toi_thieu']:
                continue

            # Tính toán phần bù vì tọa độ kết quả sẽ nhỏ hơn 4 lần so với ảnh
            (offset_x, offset_y) = (cot * 4.0, hang * 4.0)

            # Lấy góc nghiêng của kết quả và tính toán sin, cos
            goc_nghieng = du_lieu_goc[cot]
            cos = np.cos(goc_nghieng)
            sin = np.sin(goc_nghieng)

            # Tính toán chiều dài và chiều cao của hình bao
            chieu_cao = x0[cot] + x2[cot]
            chieu_dai = x1[cot] + x3[cot]

            # Tính toán tọa độ điểm đầu và điểm cuối của hình bao hiển thị văn
            # bản dự đoán
            diem_cuoi_x = int(offset_x + (cos * x1[cot]) + (sin * x2[cot]))
            diem_cuoi_y = int(offset_y - (sin * x1[cot]) + (cos * x2[cot]))
            diem_dau_x = diem_cuoi_x - chieu_dai
            diem_dau_y = diem_cuoi_y - chieu_cao

            # Thêm tọa độ hình bao và xác suất tương ứng vào danh sách
            hinh_bao.append((diem_dau_x, diem_dau_y, diem_cuoi_x, diem_cuoi_y))
            do_chinh_xac.append(xac_suat[cot])

    # Trả về 1 bộ chứa thông tin các hình bao và độ chính xác tương ứng
    return (hinh_bao, do_chinh_xac)


# Xây dựng các tùy chọn và lấy tham số tùy chọn
tuy_chon = argparse.ArgumentParser()
tuy_chon.add_argument(
        '-i',
        '--image',
        type=str,
        help='Đường dẫn tệp hình ảnh',
        )
tuy_chon.add_argument(
        '-east',
        '--east',
        type=str,
        help='Đường dẫn bộ tìm kiếm EAST',
        )
tuy_chon.add_argument(
        '-c',
        '--xac_suat_toi_thieu',
        type=float,
        default=0.5,
        help='Xác suất tối thiểu để hiển thị hình bao',
        )
tuy_chon.add_argument(
        '-w',
        '--width',
        type=int,
        default=320,
        help='Bội số gần nhất của 32 để thay đổi chiều dài',
        )
tuy_chon.add_argument(
        '-e',
        '--height',
        type=int,
        default=320,
        help='Bội số của 32 để thay đổi chiều cao',
        )
tuy_chon.add_argument(
        '-p',
        '--padding',
        type=float,
        default=0.0,
        help='Giá trị phần đệm để thêm vào các vùng ROI',
        )
cac_tuy_chon = vars(tuy_chon.parse_args())

# Tải ảnh đầu vào và lấy thông số của ảnh
hinh_anh = cv2.imread(cac_tuy_chon['image'])
anh_goc = hinh_anh.copy()
(chieu_cao_goc, chieu_dai_goc) = hinh_anh.shape[:2]

# Thiết lập chiều cao và chiều dài mới và tính tỉ lệ thay đổi tương ứng
chieu_dai_moi = cac_tuy_chon['width']
chieu_cao_moi = cac_tuy_chon['height']
ti_le_dai = chieu_dai_goc / float(chieu_dai_moi)
ti_le_cao = chieu_cao_goc / float(chieu_cao_moi)

# Chỉnh cỡ hình ảnh và lấy thông số hình ảnh mới
hinh_anh = cv2.resize(hinh_anh, (chieu_dai_moi, chieu_cao_moi))
(CHIEU_CAO, CHIEU_DAI) = hinh_anh.shape[:2]

# Định nghĩa tên 2 lớp kết quả cho mô hình tìm kiếm EAST đang sử dụng
# Lớp đầu tiên là các xác suất đầu ra và lớp thứ 2 là tọa độ hình bao
cac_ten_lop = [
        'feature_fusion/Conv_7/Sigmoid',
        'feature_fusion/concat_3',
        ]

# Tải mô hình đã huấn luyện của bộ tìm kiếm văn bản EAST
print('[INFO] Tải mô hình EAST...')
mo_hinh = cv2.dnn.readNet(cac_tuy_chon['east'])

# Xây dựng một blob từ hình ảnh và thực hiện một chuyển tiếp từ mô hình
# để có được 2 lớp đầu ra
blob = cv2.dnn.blobFromImage(
        hinh_anh,
        1.0,
        (CHIEU_DAI, CHIEU_CAO),
        (123.68, 116.78, 103.94),
        swapRB=True,
        crop=False,
        )
mo_hinh.setInput(blob)
(cac_thong_tin, toa_do) = mo_hinh.forward(cac_ten_lop)

# Lấy thông tin hình bao và độ chính xác, sau đó áp dụng non-maxima suppression
# để xóa các kết quả kém và các hình chồng nhau
(hinh_bao, do_chinh_xac) = trich_xuat_thong_tin(cac_thong_tin, toa_do)
ds_hinh_bao = non_max_suppression(np.array(hinh_bao), probs=do_chinh_xac)

# Hiển thị các kết quả tìm được
for (diem_dau_x, diem_dau_y, diem_cuoi_x, diem_cuoi_y) in ds_hinh_bao:
    # Điều chỉnh hệ tọa độ hình bao dựa trên tỉ lệ tương ứng
    diem_dau_x = int(diem_dau_x * ti_le_dai)
    diem_dau_y = int(diem_dau_y * ti_le_cao)
    diem_cuoi_x = int(diem_cuoi_x * ti_le_dai)
    diem_cuoi_y = int(diem_cuoi_y * ti_le_cao)
    ket_qua = anh_goc.copy()
    cv2.rectangle(
            ket_qua,
            (diem_dau_x, diem_dau_y),
            (diem_cuoi_x, diem_cuoi_y),
            (0, 0, 255),
            2,
            )

    # Hiển thị ảnh kết quả
    cv2.imshow('Anh ket qua', ket_qua)
    cv2.waitKey(0)

# Khởi tạo danh sách kết quả
ds_ket_qua = []

# Lặp qua các hình bao
for (diem_dau_x, diem_dau_y, diem_cuoi_x, diem_cuoi_y) in ds_hinh_bao:
    # Điều chỉnh hệ tọa độ hình bao dựa trên tỉ lệ tương ứng
    diem_dau_x = int(diem_dau_x * ti_le_dai)
    diem_dau_y = int(diem_dau_y * ti_le_cao)
    diem_cuoi_x = int(diem_cuoi_x * ti_le_dai)
    diem_cuoi_y = int(diem_cuoi_y * ti_le_cao)

    # Để có được kết quả tốt hơn, có thể áp dụng một vùng đệm xung quanh hình
    # bao, ở đây tính toán delta theo cả 2 hướng x và y
    dX = int((diem_cuoi_x - diem_dau_x) * cac_tuy_chon['padding'])
    dY = int((diem_cuoi_y - diem_dau_y) * cac_tuy_chon['padding'])
    # Áp dụng vùng đệm cho từng mặt của hình bao theo thứ tự
    diem_dau_x = max(0, diem_dau_x - dX)
    diem_dau_y = max(0, diem_dau_y - dY)
    diem_cuoi_x = min(chieu_dai_goc, diem_cuoi_x + (dX * 2))
    diem_cuoi_y = min(chieu_cao_goc, diem_cuoi_y + (dY * 2))

    # Lấy vùng đệm ROI thực tế
    print('[INFO] Lấy vùng ROI...')
    roi = anh_goc[diem_dau_y:diem_cuoi_y, diem_dau_x:diem_cuoi_x]

    # Để sử dụng tesseract v4 cần cung cấp các thiết lập về
    # -l ngôn ngữ sử dụng
    # --oem mô hình mạng thần kinh muốn sử dụng
    # --psm giá trị oem, 7 tức là xem hình ảnh là 1 dòng văn bản đơn
    cac_thiet_lap = ('-l eng --oem 1 --psm 1')
    van_ban = pytesseract.image_to_string(roi, config=cac_thiet_lap)

    # Thêm tọa độ hình bao và nội dung văn bản vào danh sách kết quả
    ds_ket_qua.append((
        (diem_dau_x, diem_dau_y, diem_cuoi_x, diem_cuoi_y),
        van_ban,
        ))

# Sắp xếp tọa độ các hình bao theo thứ tự từ trên xuống
ds_ket_qua = sorted(ds_ket_qua, key=lambda r: r[0][1])

# Lặp qua các kết quả
for (
        (diem_dau_x, diem_dau_y, diem_cuoi_x, diem_cuoi_y),
        van_ban,
        ) in ds_ket_qua:
    # Hiển thị văn bản tìm được
    print('Văn bản OCR')
    print('=' * 10)
    print('%s\n' % (van_ban))

    # Loại bỏ các ký tự không thuộc ASCII để có thể hiển thị bằng opencv
    # sau đó hiển thị văn bản và vẽ hình bao trên hình ảnh đầu vào
    van_ban = ''.join([c if ord(c) < 128 else '' for c in van_ban]).strip()
    ket_qua = anh_goc.copy()
    cv2.rectangle(
            ket_qua,
            (diem_dau_x, diem_dau_y),
            (diem_cuoi_x, diem_cuoi_y),
            (0, 0, 255),
            2,
            )
    cv2.putText(
            ket_qua,
            van_ban,
            (diem_dau_x, diem_dau_y - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
            )

    # Hiển thị ảnh kết quả
    cv2.imshow('Anh ket qua', ket_qua)
    cv2.waitKey(0)
