import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector  = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5","Model/labels.txt")

offset = 20
imgsize = 300

folder = " D:\Cisco\Phyton\pythonProject\ u"
counter=0

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M","O", "P", "Q", "R", "S", "T", "V", "W", "X", "Y", "Z"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img =detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgsize, imgsize, 3), np.uint8)*255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

# Image collector Resolution Capture

        aspectRatio = h/w

        if aspectRatio > 1:
            k = imgsize/h                                        # Video capture Hight+Size Resolution
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgsize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgsize-wCal)/2)                   # Vide capture Centre postion
            imgWhite[:,wGap:wCal+wGap] = imgResize               # Declare Resolution
            prediction, index = classifier.getPrediction(imgWhite,draw=False)
            print(prediction, index)


        else:
            k = imgsize / w                     # Video capture data more than rectangle size that being set
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgsize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgsize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize  # Declare Resolution
            prediction, index = classifier.getPrediction(imgWhite,draw=False)
                                # HIGHT                              #LENGHT                          #cOLOUR CODE    #cOLOUR FILLED BOXES
        cv2.rectangle(imgOutput, (x - offset, y - offset-50), (x + w + offset-50, y - offset-50+50), (255, 64, 64), cv2.FILLED)
        cv2.putText(imgOutput,labels[index],(x,y-25),cv2.FONT_HERSHEY_COMPLEX, 2, (240,248,255), 2)
        cv2.rectangle(imgOutput,(x-offset,y-offset),(x+w+offset,y+h+offset),(255,64,64), 4)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

#for image collector jangan terlalu dekat nnti error
# jauh kit

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)


