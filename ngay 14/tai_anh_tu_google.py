#coding: utf-8
from imutils import paths
import argparse
import requests
import cv2
import os


#Xây dựng tham số tùy chọn và lấy tham số tùy chọn
tuy_chon = argparse.ArgumentParser()
tuy_chon.add_argument(
        '-u',
        '--urls',
        required=True,
        help='Đường dẫn đến URLs ảnh',
        )
tuy_chon.add_argument(
        '-o',
        '--output',
        required=True,
        help='Đường dẫn thư mục kết quả',
        )
cac_tuy_chon = vars(tuy_chon.parse_args())

#Lấy danh sách url từ tệp đầu vào, sau đó khởi tạo tổng số ảnh tải
#được
danh_sach = open(cac_tuy_chon['urls']).read().strip().split('\n')
tong = 0

for duong_dan in danh_sach:
    try:
        #Tải ảnh
        anh_tho = requests.get(duong_dan, timeout=60)

        #Tạo thư mục nếu chưa có
        thu_muc = os.getcwd()
        thu_muc = os.path.join(thu_muc, cac_tuy_chon['output'])
        if not os.path.exists(thu_muc):
            os.mkdir(thu_muc)

        #Lưu ảnh vào máy
        luu_anh = os.path.sep.join(
                [
                    cac_tuy_chon['output'],
                    '{}.jpg'.format(str(tong).zfill(8)),
                    ]
                )
        tep = open(luu_anh, 'wb')
        tep.write(anh_tho.content)
        tep.close()
        print("[INFO] Tải thành công: {}".format(luu_anh))
        tong += 1
    except Exception as e:
        print('[ERROR] Gặp lỗi: {}'.format(e))

#Lặp qua danh sách đường dẫn ảnh vừa tải được
for duong_dan in paths.list_images(cac_tuy_chon['output']):
    #Khởi tạo trạng thái nên xóa ảnh hay không
    xoa_anh = False
    #Thử đọc ảnh
    try:
        hinh_anh = cv2.imread(duong_dan)

        #Nếu ảnh là None thì ảnh đã bị lỗi, nên xóa nó
        if hinh_anh is None:
            xoa_anh = True
    except:
        #Nếu có lỗi ở đây tức là OpenCV không đọc được ảnh, tức là tải
        #ảnh bị lỗi nên phải xóa nó đi
        print('[ERROR] OpenCV không đọc được ảnh: {}'.format(duong_dan))
        print('Ảnh sẽ bị xóa')
        xoa_anh = True

    if xoa_anh:
        print('[INFO] Tiến hành xóa ảnh: {}'.format(duong_dan))
        os.remove(duong_dan)
