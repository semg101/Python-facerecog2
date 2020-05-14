import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('C:\opencv\data\haarcascades\haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

while (True):
	ret, img = cam.read();
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255, 0, 0), 2 )
	
	cv2.imshow('Faces', img);  
	if(cv2.waitKey(1) == ord('q')):
		break; 

cam.release()
cv2.destroyAllWindows()