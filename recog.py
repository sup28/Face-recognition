import cv2
import numpy as np
#import serial
import time

face_cascade = cv2.CascadeClassifier('/home/suprit/anaconda3/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml')

#ser = serial.Serial('COM6', 9600, timeout=0)

isopen=cv2.VideoCapture(0)
rec=cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=16)
rec.read('abcd.yml')
time=0
count=0
id=0

while(isopen.isOpened()):
	#for time in range(0,30):
	ret, img=isopen.read()

	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray,1.3,5)
	for(x,y,w,h) in faces:
		id,conf=rec.predict(gray[y:y+h,x:x+w])
		print(id, conf)
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
		roi_gray = gray[y:y+h,x:x+w]

	cv2.imshow('img',img)
	if cv2.waitKey(10) & 0xFF == ord('q'):
		break

isopen.release()
cv2.destroyAllWindows()
