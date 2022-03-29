import os
import cv2, time
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
image_path = 'dataset/'
image_name = os.listdir(image_path)

#tiền xử lí ảnh: cân bằng histogram, lọc Gauss
def crop_face(list_image,image_path):
    for i in list_image:
        image = cv2.imread(image_path+'/'+i)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(gray, 1.3, 5)
    
        for x,y,w,h in face:
          #img = cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)
          crop_img = image[(y+10):(y+h-10),(x+10):(x+w-10)]
          crop_img = cv2.resize(crop_img,(150,150))
          img_yuv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2YUV)
          #equalize the histogram of the Y channel
          img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

          #convert the YUV image back to RGB format
          img_output = cv2.cvtColor(crop_img, cv2.COLOR_YUV2BGR)
          #lọc Gauss
          img = cv2.GaussianBlur(crop_img,(5,5),0)
          cv2.imwrite(i,img)

crop_face(image_name,image_path)
# Waits for a keystroke
cv2.waitKey(0)  
# Destroys all the windows created
cv2.destroyAllwindows() 
    