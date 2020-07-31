'''
Xử lý hình ảnh phiếu thi trắc nghiệm
'''
import argparse
import cv2
import imutils


# Lấy tham số
tuy_chon = argparse.ArgumentParser()
tuy_chon.add_argument(
        '-i',
        '--image',
        required=True,
        help='Đường dẫn hình ảnh',
        )
cac_tuy_chon = vars(tuy_chon.parse_args())

# Tải hình ảnh, tính toán tỉ lệ ảnh và chuyển về độ phân giải nhỏ hơn
hinh_anh = cv2.imread(cac_tuy_chon['image'])
ti_le = hinh_anh.shape[0] / 1000.0
anh_goc = hinh_anh.copy()
hinh_anh = imutils.resize(hinh_anh, height=1000)

# Chuyển sang ảnh đen trắng, làm mờ và tìm các cạnh
den_trang = cv2.cvtColor(hinh_anh, cv2.COLOR_BGR2GRAY)
# den_trang = cv2.GaussianBlur(den_trang, (5, 5), 0)
cac_canh = cv2.Canny(den_trang, 55, 200)

# Hiển thị hình ảnh và ảnh tìm cạnh
print('1: Tìm cạnh')
cv2.imshow('Hinh anh', hinh_anh)
cv2.imshow('Tim canh', den_trang)
cv2.waitKey(0)
cv2.destroyAllWindows()
