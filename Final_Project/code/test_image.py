from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from os import listdir
from os.path import isfile, join


# dimensions of our images
img_width, img_height = 150,150

# load the model we saved
model = load_model('model.h5')
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

mypath = "predict/"
file = "rbcn.jpg"
# predicting images
img = image.load_img(file, target_size=(img_width, img_height))
x = image.img_to_array(img) #chuyển ảnh thành ma trận
x = np.expand_dims(x, axis=0) #mở rộng thêm 1 chiều của ma trận, tại sao cần mở rộng ?? có thể khi dự đoán nó nhận vào là 1 vector các ảnh
images = np.vstack([x]) #images là vector bao gồm các x. Trong đó x đã mở rộng thêm một chiều là vector các tensor

classes = model.predict((images).astype("int32"))
print(classes)