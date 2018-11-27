import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier("/home/suprit/anaconda3/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml")
i=0

isopen=cv2.VideoCapture(0)
while(isopen.isOpened()):
	ret, img=isopen.read()
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray,1.3,5)
  
	for(x,y,w,h) in faces:
		i+=1
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		roi = gray[y:y+h,x:x+w]
		cv2.imshow('img',roi)
		cv2.imwrite("images/im_%d"%i+".png", roi)
		
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
isopen.release()
cv2.destroyAllWindows()

