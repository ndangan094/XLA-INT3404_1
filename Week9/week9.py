import cv2
import numpy as np
import os, os.path

for i in range(1,4):
    path = "./input/" + str(i) + ".jpg"
    img = cv2.imread(path)

    #convert img to gray

    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #load cascade and detect faces

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(grayImg,scaleFactor=1.3,minNeighbors=3,minSize=(30,30))

    #faces area
    mask = np.zeros(img.shape, np.uint8)
    for (x,y,w,h) in faces:
        mask[y:y+h, x:x+w] = img[y:y+h, x:x+w]
    convertedImg = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)

    # skin areas

    min = np.array([0, 40, 40], dtype = "uint8")
    max = np.array([13, 150, 255], dtype = "uint8")
    skinMask = cv2.inRange(convertedImg, min, max)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,9))
    skinMask = cv2.erode(skinMask, kernel, iterations = 3)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 5)
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(img, img, mask = skinMask)
    
    cv2.imwrite("./output/"+ str(i) + ".jpg", skin)