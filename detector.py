import numpy as np
import cv2
import sqlite3

face_cascade = cv2.CascadeClassifier('C:\opencv\data\haarcascades\haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
rec = cv2.face.createLBPHFaceRecognizer();
rec.load("recognizer\\trainingData.yml")

def getProfile(id):
	conn = sqlite3.connect("FaceBase.db")
	cmd = "SELECT * FROM People WHERE ID =" + str(id)
	cursor = conn.execute(cmd)
	profile = None
	for row in cursor:
		profile = row
	conn.close()
	return profile

id = 0
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (255, 255, 255)

while (True):
	ret, img = cam.read();
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255, 0, 0), 2 )
		id, conf = rec.predict(gray[y:y+h, x:x+w])
		profile = getProfile(id)
		if(profile != None):	
		    cv2.putText(img, str(profile[1]), (x,y+h), fontface, fontscale, fontcolor) 
		    cv2.putText(img, str(profile[2]), (x,y+h+30), fontface, fontscale, fontcolor)
		    cv2.putText(img, str(profile[3]), (x,y+h+60), fontface, fontscale, fontcolor)
		    cv2.putText(img, str(profile[4]), (x,y+h+90), fontface, fontscale, fontcolor)
		
	cv2.imshow('Faces', img);  
	if(cv2.waitKey(1) == ord('q')):
		break; 

cam.release()
cv2.destroyAllWindows()