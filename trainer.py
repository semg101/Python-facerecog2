import os
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.face.createLBPHFaceRecognizer();
path = 'dataset'

def getImagesWithID (path):
	imagePaths = [os.path.join(path, f)  for f in os.listdir(path)]
	faces = []
	IDs = []
	for imagePath in imagePaths:
		#open the image in the directory and convert it into grayscale
	    faceImg = Image.open(imagePath).convert('L');
	    #convert the image into numpy array
	    faceNp = np.array(faceImg, 'uint8')
	    #to get the user ID which is in a string format, then convert it into integer
	    ID = int(os.path.split(imagePath)[-1].split('.')[1])
	    faces.append(faceNp)
	    print(ID)
	    IDs.append(ID)
	    cv2.imshow("training", faceNp)
	    cv2.waitKey(10)
	return IDs, faces
	
Ids, faces = getImagesWithID(path)	
recognizer.train(faces, np.array(Ids))
recognizer.save('recognizer/trainingData.yml')
cv2.destroyAllWindows()

