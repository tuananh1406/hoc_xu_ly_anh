'''
Chuyển đổi ảnh nghiêng sang ảnh thẳng
'''
import numpy as np
import cv2


def sap_xep_cac_diem(ds_diem):
    '''
    Sắp xếp các điểm mốc để đưa hình ảnh về phương thẳng đứng
    '''
    # Khởi tạo danh sách các điểm tọa độ theo thứ tự trên-trái, trên-phải,
    # dưới-phải, dưới-trái
    hinh_bao = np.zeros((4, 2), dtype="float32")

    # Tính tổng x+y của từng điểm, trong danh sách tổng khi đó:
    # Điểm trên-trái sẽ có tổng nhỏ nhất, điểm dưới phải sẽ có tổng lớn nhất
    ds_tong = ds_diem.sum(axis=1)
    hinh_bao[0] = ds_diem[np.argmin(ds_tong)]
    hinh_bao[2] = ds_diem[np.argmax(ds_tong)]

    # Tính toán sự khác biệt giữa các điểm,(x-y), khi đó:
    # Điểm trên-phải sẽ ít khác biệt nhất, điểm dưới trái sẽ khác biệt nhất
    so_sanh = np.diff(ds_diem, axis=1)
    hinh_bao[1] = ds_diem[np.argmin(so_sanh)]
    hinh_bao[3] = ds_diem[np.argmax(so_sanh)]

    return hinh_bao


def chuyen_doi_hinh(hinh_anh, ds_diem):
    '''
    Chuyển đổi hình ảnh sử dụng ds_diem làm điểm mốc
    '''
    # Sắp xếp lại ds_diem và tách thành các phần riêng biệt
    hinh_bao = sap_xep_cac_diem(ds_diem)
    (tt, tp, dp, dt) = hinh_bao

    # Tính toán chiều rộng của hình ảnh mới, đó sẽ là khoảng cách tối đa giữa
    # điểm dưới-phải và dưới-trái hoặc trên-phải và trên-trái theo trục x
    chieu_dai_A = np.sqrt(((dp[0] - dt[0]) ** 2) + ((dp[1] - dt[1]) ** 2))
    chieu_dai_B = np.sqrt(((tp[0] - tt[0]) ** 2) + ((tp[1] - tt[1]) ** 2))
    chieu_dai_lon_nhat = max(chieu_dai_A, chieu_dai_B)

    # Tính toán chiều cao của hình ảnh mới, đó sẽ là khoảng cách tối đa giữa
    # điểm trên-phải và dưới-phải hoặc trên-trái và dưới-trái theo trục y
    chieu_rong_A = np.sqrt(((tp[0] - dp[0]) ** 2) + ((tp[1] - dp[1]) ** 2))
    chieu_rong_B = np.sqrt(((tt[0] - dt[0]) ** 2) + ((tt[1] - dt[1]) ** 2))
    chieu_rong_lon_nhat = max(chieu_rong_A, chieu_rong_B)

    # Sau khi có được kích thước của ảnh mới, xây dựng danh sách các điểm đích
    # là điểm mốc mới của ảnh, chỉ định các điểm theo thứ tự trên-trái,
    # trên-phải, dưới-phải, dưới-trái
    ds_diem_dich = np.array([
        [0, 0],
        [chieu_dai_lon_nhat - 1, 0],
        [chieu_dai_lon_nhat - 1, chieu_rong_lon_nhat - 1],
        [0, chieu_rong_lon_nhat - 1]], dtype="float32")

    # Tính toán ma trận chuyển đổi và áp dụng nó
    M = cv2.getPerspectiveTransform(hinh_bao, ds_diem_dich)
    ket_qua = cv2.warpPerspective(
            hinh_anh,
            M,
            (chieu_dai_lon_nhat, chieu_rong_lon_nhat),
            )

    return ket_qua
