import cv2
from cvzone.HandTrackingModule import  HandDetector
import numpy as np
import math
import time
cap = cv2.VideoCapture(0)
detector  = HandDetector(maxHands=1)

offset = 20
imgsize = 300

folder="D:\Cisco\Phyton\pythonProject\A"
counter=0

while True:
    success, img = cap.read()
    hands, img =detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgsize, imgsize, 3), np.uint8)*255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

# Image collectore Resolution Capture

        aspectRatio = h/w

        if aspectRatio > 1:
            k = imgsize/h                                        # Video capture Hight+Size Resolution
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgsize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgsize-wCal)/2)                   # Vide capture Centre postion
            imgWhite[:,wGap:wCal+wGap] = imgResize               # Declare Resolution

        else:
            k = imgsize / w                     # Video capture data more than rectangle size that being set
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgsize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgsize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize  # Declare Resolution

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

#for image collector jangan terlalu dekat nnti error
# jauh kit

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord ("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)

