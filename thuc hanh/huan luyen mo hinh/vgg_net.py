#coding: utf-8
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K


class MangVGGNet:
    #Lớp VGGNet đơn giản
    def xay_dung(
            width, #Chiều rộng ảnh
            height, #Chiều dài ảnh
            depth, #Số kênh ảnh
            classes, #Số lớp phân loại của mô hình
            ):
        #Định nghĩa phương thức khởi tạo mô hình cùng kích thước đầu
        #vào sẽ là 'kênh cuối' và kích thước các kênh ảnh 
        mo_hinh = Sequential()
        kich_thuoc_dau_vao = (height, width, depth)
        kich_thuoc_kenh = -1

        #Nếu đang sử dụng 'kênh đầu tiên', cập nhật kích thước đầu vào
        #và kích thước các kênh ảnh
        if K.image_data_format() == 'channels_first':
            kich_thuoc_dau_vao = (depth, height, width)
            kich_thuoc_kenh = 1

        #CONV => RELU => POOL
        mo_hinh.add(
                Conv2D(
                    32,
                    (3, 3),
                    padding='same',
                    input_shape=kich_thuoc_dau_vao,
                    )
                )
        mo_hinh.add(Activation('relu'))
        mo_hinh.add(BatchNormalization(axis=kich_thuoc_kenh))
        mo_hinh.add(MaxPooling2D(pool_size=(3, 3)))
        mo_hinh.add(Dropout(0.25))

        #(CONV => RELU) *2 => POOL
        mo_hinh.add(Conv2D(128, (3, 3), padding='same'))
        mo_hinh.add(Activation('relu'))
        mo_hinh.add(BatchNormalization(axis=kich_thuoc_kenh))
        mo_hinh.add(Conv2D(128, (3, 3), padding='same'))
        mo_hinh.add(Activation('relu'))
        mo_hinh.add(BatchNormalization(axis=kich_thuoc_kenh))
        mo_hinh.add(MaxPooling2D(pool_size=(2, 2)))
        mo_hinh.add(Dropout(0.25))

        #Tập hợp các lớp FC => RELU đầu tiên (và duy nhất)
        mo_hinh.add(Flatten())
        mo_hinh.add(Dense(1024))
        mo_hinh.add(Activation('relu'))
        mo_hinh.add(BatchNormalization())
        mo_hinh.add(Dropout(0.5))

        #Phân loại softmax
        mo_hinh.add(Dense(classes))
        mo_hinh.add(Activation('softmax'))

        #Trả về kiến trúc mạng đã được xây dựng
        return mo_hinh
