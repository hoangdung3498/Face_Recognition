from keras.models import load_model
import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
vid = cv2.VideoCapture(0) #doc du lieu tu camera
#đọc dữ liệu bằng camera IP
#address = "http://192.168.0.103:8080/video"
#vid.open(address)

# dimensions of our images
img_width, img_height = 150,150

# load the model 
model = load_model('model.h5')
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# predicting images
while True:

    check, frame = vid.read() 
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, 1.3, 5)
    if(len(face)>0):
      for x,y,w,h in face:
          #img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3) 
          crop_img = frame[y:y+h,x:x+w]
      resize_img = cv2.resize(crop_img  , (150 , 150))
      resize_img = np.expand_dims(resize_img, axis=0)
      images = np.vstack([resize_img])
      #predict
      classes = model.predict((resize_img).astype("int32"))
      print(classes)
      #draw
      if(classes[0,0]>0.7):
        frame = cv2.putText(frame,text='Messi',org=(face[0,0],face[0,1]+face[0,2]),fontScale = 2,thickness=2,fontFace=cv2.FONT_HERSHEY_SIMPLEX,color = (255,0,0))
        img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
      if(classes[0,1]>0.7):
        frame = cv2.putText(frame,text='Ronaldo',org=(face[0,0],face[0,1]+face[0,2]),fontScale = 2,thickness=2,fontFace=cv2.FONT_HERSHEY_SIMPLEX,color = (255,0,0))
        img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
      if(classes[0,2]>0.7):
        frame = cv2.putText(frame,text='Scarlet',org=(face[0,0],face[0,1]+face[0,2]),fontScale = 2,thickness=2,fontFace=cv2.FONT_HERSHEY_SIMPLEX,color = (255,0,0))
        img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
      cv2.imshow('face_recognition',frame) 
      key = cv2.waitKey(1)
      if key ==ord('q'):
          break
vid.release()
cv2.destroyAllWindows()

