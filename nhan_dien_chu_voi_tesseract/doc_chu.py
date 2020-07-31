import pytesseract
import argparse
import cv2


# Xây dựng các tham số tùy chọn và lấy tham số
tuy_chon = argparse.ArgumentParser()
tuy_chon.add_argument(
        '-i',
        '--image',
        type=str,
        help='Đường dẫn ảnh',
        )
cac_tuy_chon = vars(tuy_chon.parse_args())

hinh_anh = cv2.imread(cac_tuy_chon['image'])
van_ban = pytesseract.image_to_string(
        cac_tuy_chon['image'],
        lang='vie',
        )
print(van_ban)
cv2.imshow('Hinh anh', hinh_anh)
cv2.waitKey(0)
