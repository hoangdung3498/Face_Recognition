import cv2
import numpy as np
from keras.models import load_model

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
# dimensions 
img_width, img_height = 150,150

# load model
model = load_model('model.h5')
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#đọc ảnh input
image = cv2.imread('test/si6.jpg')
image_copy = image.copy()
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
face = face_cascade.detectMultiScale(gray, 1.1, 5) #Haar feature detect face

#xác định toạ độ khuôn mặt
for x,y,w,h in face:
    #img = cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)
    crop_img = image_copy[y:y+h,x:x+w]
    cv2.imshow('crop', crop_img)

#resize thành (150,150) theo input model
resize_img = cv2.resize(crop_img ,(150 , 150))

#mở rộng thêm 1 chiều của ma trận
img_expand = np.expand_dims(resize_img, axis=0) 

#images là vector bao gồm các x theo chiều dọc. Trong đó x đã mở rộng thêm một chiều là vector các tensor
images = np.vstack([img_expand]) 

#predict
classes = model.predict((images).astype("int32"))
print(classes)
if(classes[0,0]>0.7):
  image = cv2.putText(image,text='Messi',org=(face[0,0],face[0,1]+face[0,2]),fontScale = 2,thickness=2,fontFace=cv2.FONT_HERSHEY_SIMPLEX,color = (255,0,0))
  img = cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)
if(classes[0,1]>0.7):
  image = cv2.putText(image,text='Ronaldo',org=(face[0,0],face[0,1]+face[0,2]),fontScale = 2,thickness=2,fontFace=cv2.FONT_HERSHEY_SIMPLEX,color = (255,0,0))
  img = cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)
if(classes[0,2]>0.7):
  image = cv2.putText(image,text='Scarlet',org=(face[0,0],face[0,1]+face[0,2]),fontScale = 2,thickness=2,fontFace=cv2.FONT_HERSHEY_SIMPLEX,color = (255,0,0))
  img = cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)
cv2.imshow('loc',image) 
key = cv2.waitKey(0)


    

