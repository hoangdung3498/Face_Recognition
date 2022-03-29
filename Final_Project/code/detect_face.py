import cv2, time
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from os import listdir
from os.path import isfile, join

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
# dimensions of our images
img_width, img_height = 150,150
cop_img = []
# load the model we saved
model = load_model('model.h5')
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#
image = cv2.imread('test/scarlet25.jpg')
image_cop = image.copy()
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
face = face_cascade.detectMultiScale(gray, 1.1, 5) #tìm các contour trong image
print(len(face))
for x,y,w,h in face:
    img = cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)
    crop_img = image_cop[y:y+h,x:x+w]
    cv2.imshow('crop', crop_img)
#resize_img = cv2.resize(image ,(150 , 150))
resize_img = cv2.resize(crop_img ,(150 , 150))
abc = np.expand_dims(resize_img, axis=0) #mở rộng thêm 1 chiều của ma trận, tại sao cần mở rộng ?? có thể khi dự đoán nó nhận vào là 1 vector các ảnh
images = np.vstack([abc]) #images là vector bao gồm các x. Trong đó x đã mở rộng thêm một chiều là vector các tensor

classes = model.predict((images).astype("int32"))
print(classes)
if(classes[0,0]>0.5):
  print('messi')
if(classes[0,1]>0.5):
  print('ronaldo')
if(classes[0,2]>0.5):
  print('scarlet')
cv2.imshow('loc',image) 
key = cv2.waitKey(0)


    

