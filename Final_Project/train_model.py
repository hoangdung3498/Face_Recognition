
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np

# dimensions of our images
img_width, img_height = 150, 150
train_data_dir = 'dataset'
epochs = 15
nb_train_samples = 1140
batch_size = 20 #sủ dụng Mini-batch gradient descent, sử dụng batch_size dũ liệu trong tập N dữ liệu để cập nhật tham số 1 lần

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

#khởi tạo ConvNet layer
model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(150,150,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#Flat
model.add(Flatten())

#Fully Connect
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(3))
model.add(Activation('softmax'))

#chọn thật toán tối ưu, hàm lỗi
model.compile(loss='categorical_crossentropy', # or categorical_crossentropy
              optimizer='adam',# or adagrad
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255, rotation_range=20,
    horizontal_flip=True
    #,zoom_range=0.2
    )

#tạo ra một đối tượng bao gồm cả X: trainning và Y:label đã được mã hóa 
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')#->hỗ trợ chuyển tên(label) thành vector tương ứng số node của output layer: Cats=[0,1],Dogs=[1,0]

#train model
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    )
model.save('model.h5')
