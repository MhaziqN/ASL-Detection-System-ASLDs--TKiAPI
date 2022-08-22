from tkinter import *
from PIL import ImageTk,Image
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import pyttsx3


def scan_Gesture ():

    def talk(labels):
        engine.say(labels)
        engine.runAndWait()

    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
    engine = pyttsx3.init()
    voice = engine.getProperty('voice')
    engine.setProperty('voice', voice[0])


    offset = 20
    imgsize = 300

    folder = " D:\Cisco\Phyton\pythonProject\A"
    counter = 0

    labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N" ,"O", "P", "Q", "R", "S", "T","U", "V", "W",
              "X", "Y", "Z"]

    while True:
        success, img = cap.read()
        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            imgCropShape = imgCrop.shape

            # Image collector Resolution Capture

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgsize / h  # Video capture Hight+Size Resolution
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgsize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgsize - wCal) / 2)  # Vide capture Centre postion
                imgWhite[:, wGap:wCal + wGap] = imgResize  # Declare Resolution
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                print(prediction, index)


            else:
                k = imgsize / w  # Video capture data more than rectangle size that being set
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgsize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgsize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize  # Declare Resolution
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                # HIGHT                              #LENGHT                          #cOLOUR CODE    #cOLOUR FILLED BOXES
            cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x + w + offset - 50, y - offset - 50 + 50),
                          (255, 64, 64), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 25), cv2.FONT_HERSHEY_COMPLEX, 2, (240, 248, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 64, 64), 4)
            #talk(labels[index])

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

        # for image collector jangan terlalu dekat nnti error
        # jauh kit

        cv2.imshow("Image", imgOutput)
        Key=cv2.waitKey(1)
        if Key == ord('s'):
            talk('successfully captured !')
            with open('Word.txt', "a") as f:
                f.write(labels[index])
        elif Key == ord('a'):
            file = open("Word.txt", "r+")
            file.truncate(0)
            file.close()
        elif Key == ord('d'):

            with open ('Word.txt',"r") as f:
                for line in f.readlines():
                    print (line)

                    root = Tk()
                    root.title ('ASLDs')
                    root.iconbitmap('AS (1).ico')
                    root.configure(background="#C1CDCD")
                    text_Label = Label(root,text='Word Translated:'+ line, fg='black',bg='#FFFFFF')
                    text_Label.pack(pady=(30, 30))
                    text_Label.config(font=('Amasis MT Pro Medium', 10))
                    talk(line)

        if Key == ord('w'):
            break
    cap.close()

def data_collection():
    def ok_btn():

        user_C = option_list.get()

        if user_C.lower() == 'a':
            cap = cv2.VideoCapture(0)
            detector = HandDetector(maxHands=1)

            offset = 20
            imgsize = 300

            folder = "D:\Cisco\Phyton\pythonProject\A"
            counter = 0

            while True:
                success, img = cap.read()
                hands, img = detector.findHands(img)
                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']

                    imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
                    imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                    imgCropShape = imgCrop.shape

                    # Image collectore Resolution Capture

                    aspectRatio = h / w

                    if aspectRatio > 1:
                        k = imgsize / h  # Video capture Hight+Size Resolution
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgsize))
                        imgResizeShape = imgResize.shape
                        wGap = math.ceil((imgsize - wCal) / 2)  # Vide capture Centre postion
                        imgWhite[:, wGap:wCal + wGap] = imgResize  # Declare Resolution

                    else:
                        k = imgsize / w  # Video capture data more than rectangle size that being set
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgsize, hCal))
                        imgResizeShape = imgResize.shape
                        hGap = math.ceil((imgsize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize  # Declare Resolution

                    cv2.imshow("ImageCrop", imgCrop)
                    cv2.imshow("ImageWhite", imgWhite)

                # for image collector jangan terlalu dekat nnti error
                # jauh kit

                cv2.imshow("Image", img)
                key = cv2.waitKey(1)
                if key == ord("s"):
                    counter += 1
                    cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
                    print(counter)
                if key == ord('w'):

                    root = Tk()

                    root.title('Note')
                    root.iconbitmap('AS (1).ico')
                    root.geometry('300x150')
                    root.configure(background="#C1CDCD")

                    text_Label = Label(root, text='Successfully add ' + str(counter) + ' Data Capture !!', fg='black',
                                       bg='#FFFFFF')
                    text_Label.pack(pady=(30, 30))
                    text_Label.config(font=('Amasis MT Pro Medium', 10))

            cap.close()


        elif user_C.lower() == 'b':
            cap = cv2.VideoCapture(0)
            detector = HandDetector(maxHands=1)

            offset = 20
            imgsize = 300

            folder = "D:\Cisco\Phyton\pythonProject\B"
            counter = 0

            while True:
                success, img = cap.read()
                hands, img = detector.findHands(img)
                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']

                    imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
                    imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                    imgCropShape = imgCrop.shape

                    # Image collectore Resolution Capture

                    aspectRatio = h / w

                    if aspectRatio > 1:
                        k = imgsize / h  # Video capture Hight+Size Resolution
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgsize))
                        imgResizeShape = imgResize.shape
                        wGap = math.ceil((imgsize - wCal) / 2)  # Vide capture Centre postion
                        imgWhite[:, wGap:wCal + wGap] = imgResize  # Declare Resolution

                    else:
                        k = imgsize / w  # Video capture data more than rectangle size that being set
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgsize, hCal))
                        imgResizeShape = imgResize.shape
                        hGap = math.ceil((imgsize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize  # Declare Resolution

                    cv2.imshow("ImageCrop", imgCrop)
                    cv2.imshow("ImageWhite", imgWhite)

                # for image collector jangan terlalu dekat nnti error
                # jauh kit

                cv2.imshow("Image", img)
                key = cv2.waitKey(1)
                if key == ord("s"):
                    counter += 1
                    cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
                    print(counter)
                if key == ord('w'):
                    root = Tk()

                    root.title('Note')
                    root.iconbitmap('AS (1).ico')
                    root.geometry('300x150')
                    root.configure(background="#C1CDCD")

                    text_Label = Label(root, text='Successfully add ' + str(counter) + ' Data Capture !!', fg='black',
                                       bg='#FFFFFF')
                    text_Label.pack(pady=(30, 30))
                    text_Label.config(font=('Amasis MT Pro Medium', 10))

            cap.close()

        elif user_C.lower() == 'c':
            cap = cv2.VideoCapture(0)
            detector = HandDetector(maxHands=1)

            offset = 20
            imgsize = 300

            folder = "D:\Cisco\Phyton\pythonProject\C"
            counter = 0

            while True:
                success, img = cap.read()
                hands, img = detector.findHands(img)
                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']

                    imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
                    imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                    imgCropShape = imgCrop.shape

                    # Image collectore Resolution Capture

                    aspectRatio = h / w

                    if aspectRatio > 1:
                        k = imgsize / h  # Video capture Hight+Size Resolution
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgsize))
                        imgResizeShape = imgResize.shape
                        wGap = math.ceil((imgsize - wCal) / 2)  # Vide capture Centre postion
                        imgWhite[:, wGap:wCal + wGap] = imgResize  # Declare Resolution

                    else:
                        k = imgsize / w  # Video capture data more than rectangle size that being set
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgsize, hCal))
                        imgResizeShape = imgResize.shape
                        hGap = math.ceil((imgsize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize  # Declare Resolution

                    cv2.imshow("ImageCrop", imgCrop)
                    cv2.imshow("ImageWhite", imgWhite)

                # for image collector jangan terlalu dekat nnti error
                # jauh kit

                cv2.imshow("Image", img)
                key = cv2.waitKey(1)
                if key == ord("s"):
                    counter += 1
                    cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
                    print(counter)
                if key == ord('w'):
                    root = Tk()

                    root.title('Note')
                    root.iconbitmap('AS (1).ico')
                    root.geometry('300x150')
                    root.configure(background="#C1CDCD")

                    text_Label = Label(root, text='Successfully add ' + str(counter) + ' Data Capture !!', fg='black',
                                       bg='#FFFFFF')
                    text_Label.pack(pady=(30, 30))
                    text_Label.config(font=('Amasis MT Pro Medium', 10))

            cap.close()


        elif user_C.lower() == 'd':
            cap = cv2.VideoCapture(0)
            detector = HandDetector(maxHands=1)

            offset = 20
            imgsize = 300

            folder = "D:\Cisco\Phyton\pythonProject\D"
            counter = 0

            while True:
                success, img = cap.read()
                hands, img = detector.findHands(img)
                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']

                    imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
                    imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                    imgCropShape = imgCrop.shape

                    # Image collectore Resolution Capture

                    aspectRatio = h / w

                    if aspectRatio > 1:
                        k = imgsize / h  # Video capture Hight+Size Resolution
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgsize))
                        imgResizeShape = imgResize.shape
                        wGap = math.ceil((imgsize - wCal) / 2)  # Vide capture Centre postion
                        imgWhite[:, wGap:wCal + wGap] = imgResize  # Declare Resolution

                    else:
                        k = imgsize / w  # Video capture data more than rectangle size that being set
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgsize, hCal))
                        imgResizeShape = imgResize.shape
                        hGap = math.ceil((imgsize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize  # Declare Resolution

                    cv2.imshow("ImageCrop", imgCrop)
                    cv2.imshow("ImageWhite", imgWhite)

                # for image collector jangan terlalu dekat nnti error
                # jauh kit

                cv2.imshow("Image", img)
                key = cv2.waitKey(1)
                if key == ord("s"):
                    counter += 1
                    cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
                    print(counter)
                if key == ord('w'):
                    root = Tk()

                    root.title('Note')
                    root.iconbitmap('AS (1).ico')
                    root.geometry('300x150')
                    root.configure(background="#C1CDCD")

                    text_Label = Label(root, text='Successfully add ' + str(counter) + ' Data Capture !!', fg='black',
                                       bg='#FFFFFF')
                    text_Label.pack(pady=(30, 30))
                    text_Label.config(font=('Amasis MT Pro Medium', 10))

            cap.close()

        elif user_C.lower() == 'e':
            cap = cv2.VideoCapture(0)
            detector = HandDetector(maxHands=1)

            offset = 20
            imgsize = 300

            folder = "D:\Cisco\Phyton\pythonProject\E"
            counter = 0

            while True:
                success, img = cap.read()
                hands, img = detector.findHands(img)
                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']

                    imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
                    imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                    imgCropShape = imgCrop.shape

                    # Image collectore Resolution Capture

                    aspectRatio = h / w

                    if aspectRatio > 1:
                        k = imgsize / h  # Video capture Hight+Size Resolution
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgsize))
                        imgResizeShape = imgResize.shape
                        wGap = math.ceil((imgsize - wCal) / 2)  # Vide capture Centre postion
                        imgWhite[:, wGap:wCal + wGap] = imgResize  # Declare Resolution

                    else:
                        k = imgsize / w  # Video capture data more than rectangle size that being set
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgsize, hCal))
                        imgResizeShape = imgResize.shape
                        hGap = math.ceil((imgsize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize  # Declare Resolution

                    cv2.imshow("ImageCrop", imgCrop)
                    cv2.imshow("ImageWhite", imgWhite)

                # for image collector jangan terlalu dekat nnti error
                # jauh kit

                cv2.imshow("Image", img)
                key = cv2.waitKey(1)
                if key == ord("s"):
                    counter += 1
                    cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
                    print(counter)
                if key == ord('w'):
                    root = Tk()

                    root.title('Note')
                    root.iconbitmap('AS (1).ico')
                    root.geometry('300x150')
                    root.configure(background="#C1CDCD")

                    text_Label = Label(root, text='Successfully add ' + str(counter) + ' Data Capture !!', fg='black',
                                       bg='#FFFFFF')
                    text_Label.pack(pady=(30, 30))
                    text_Label.config(font=('Amasis MT Pro Medium', 10))

            cap.close()

        elif user_C.lower() == 'f':
            cap = cv2.VideoCapture(0)
            detector = HandDetector(maxHands=1)

            offset = 20
            imgsize = 300

            folder = "D:\Cisco\Phyton\pythonProject\F"
            counter = 0

            while True:
                success, img = cap.read()
                hands, img = detector.findHands(img)
                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']

                    imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
                    imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                    imgCropShape = imgCrop.shape

                    # Image collectore Resolution Capture

                    aspectRatio = h / w

                    if aspectRatio > 1:
                        k = imgsize / h  # Video capture Hight+Size Resolution
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgsize))
                        imgResizeShape = imgResize.shape
                        wGap = math.ceil((imgsize - wCal) / 2)  # Vide capture Centre postion
                        imgWhite[:, wGap:wCal + wGap] = imgResize  # Declare Resolution

                    else:
                        k = imgsize / w  # Video capture data more than rectangle size that being set
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgsize, hCal))
                        imgResizeShape = imgResize.shape
                        hGap = math.ceil((imgsize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize  # Declare Resolution

                    cv2.imshow("ImageCrop", imgCrop)
                    cv2.imshow("ImageWhite", imgWhite)

                # for image collector jangan terlalu dekat nnti error
                # jauh kit

                cv2.imshow("Image", img)
                key = cv2.waitKey(1)
                if key == ord("s"):
                    counter += 1
                    cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
                    print(counter)
                if key == ord('w'):
                    root = Tk()

                    root.title('Note')
                    root.iconbitmap('AS (1).ico')
                    root.geometry('300x150')
                    root.configure(background="#C1CDCD")

                    text_Label = Label(root, text='Successfully add ' + str(counter) + ' Data Capture !!', fg='black',
                                       bg='#FFFFFF')
                    text_Label.pack(pady=(30, 30))
                    text_Label.config(font=('Amasis MT Pro Medium', 10))

            cap.close()

        elif user_C.lower() == 'g':
            cap = cv2.VideoCapture(0)
            detector = HandDetector(maxHands=1)

            offset = 20
            imgsize = 300

            folder = "D:\Cisco\Phyton\pythonProject\G"
            counter = 0

            while True:
                success, img = cap.read()
                hands, img = detector.findHands(img)
                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']

                    imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
                    imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                    imgCropShape = imgCrop.shape

                    # Image collectore Resolution Capture

                    aspectRatio = h / w

                    if aspectRatio > 1:
                        k = imgsize / h  # Video capture Hight+Size Resolution
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgsize))
                        imgResizeShape = imgResize.shape
                        wGap = math.ceil((imgsize - wCal) / 2)  # Vide capture Centre postion
                        imgWhite[:, wGap:wCal + wGap] = imgResize  # Declare Resolution

                    else:
                        k = imgsize / w  # Video capture data more than rectangle size that being set
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgsize, hCal))
                        imgResizeShape = imgResize.shape
                        hGap = math.ceil((imgsize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize  # Declare Resolution

                    cv2.imshow("ImageCrop", imgCrop)
                    cv2.imshow("ImageWhite", imgWhite)

                # for image collector jangan terlalu dekat nnti error
                # jauh kit

                cv2.imshow("Image", img)
                key = cv2.waitKey(1)
                if key == ord("s"):
                    counter += 1
                    cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
                    print(counter)
                if key == ord('w'):
                    root = Tk()

                    root.title('Note')
                    root.iconbitmap('AS (1).ico')
                    root.geometry('300x150')
                    root.configure(background="#C1CDCD")

                    text_Label = Label(root, text='Successfully add ' + str(counter) + ' Data Capture !!', fg='black',
                                       bg='#FFFFFF')
                    text_Label.pack(pady=(30, 30))
                    text_Label.config(font=('Amasis MT Pro Medium', 10))

            cap.close()

        elif user_C.lower() == 'h':
            cap = cv2.VideoCapture(0)
            detector = HandDetector(maxHands=1)

            offset = 20
            imgsize = 300

            folder = "D:\Cisco\Phyton\pythonProject\H"
            counter = 0

            while True:
                success, img = cap.read()
                hands, img = detector.findHands(img)
                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']

                    imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
                    imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                    imgCropShape = imgCrop.shape

                    # Image collectore Resolution Capture

                    aspectRatio = h / w

                    if aspectRatio > 1:
                        k = imgsize / h  # Video capture Hight+Size Resolution
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgsize))
                        imgResizeShape = imgResize.shape
                        wGap = math.ceil((imgsize - wCal) / 2)  # Vide capture Centre postion
                        imgWhite[:, wGap:wCal + wGap] = imgResize  # Declare Resolution

                    else:
                        k = imgsize / w  # Video capture data more than rectangle size that being set
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgsize, hCal))
                        imgResizeShape = imgResize.shape
                        hGap = math.ceil((imgsize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize  # Declare Resolution

                    cv2.imshow("ImageCrop", imgCrop)
                    cv2.imshow("ImageWhite", imgWhite)

                # for image collector jangan terlalu dekat nnti error
                # jauh kit

                cv2.imshow("Image", img)
                key = cv2.waitKey(1)
                if key == ord("s"):
                    counter += 1
                    cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
                    print(counter)
                if key == ord('w'):
                    root = Tk()

                    root.title('Note')
                    root.iconbitmap('AS (1).ico')
                    root.geometry('300x150')
                    root.configure(background="#C1CDCD")

                    text_Label = Label(root, text='Successfully add ' + str(counter) + ' Data Capture !!', fg='black',
                                       bg='#FFFFFF')
                    text_Label.pack(pady=(30, 30))
                    text_Label.config(font=('Amasis MT Pro Medium', 10))

            cap.close()

        elif user_C.lower() == 'i':
            cap = cv2.VideoCapture(0)
            detector = HandDetector(maxHands=1)

            offset = 20
            imgsize = 300

            folder = "D:\Cisco\Phyton\pythonProject\I"
            counter = 0

            while True:
                success, img = cap.read()
                hands, img = detector.findHands(img)
                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']

                    imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
                    imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                    imgCropShape = imgCrop.shape

                    # Image collectore Resolution Capture

                    aspectRatio = h / w

                    if aspectRatio > 1:
                        k = imgsize / h  # Video capture Hight+Size Resolution
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgsize))
                        imgResizeShape = imgResize.shape
                        wGap = math.ceil((imgsize - wCal) / 2)  # Vide capture Centre postion
                        imgWhite[:, wGap:wCal + wGap] = imgResize  # Declare Resolution

                    else:
                        k = imgsize / w  # Video capture data more than rectangle size that being set
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgsize, hCal))
                        imgResizeShape = imgResize.shape
                        hGap = math.ceil((imgsize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize  # Declare Resolution

                    cv2.imshow("ImageCrop", imgCrop)
                    cv2.imshow("ImageWhite", imgWhite)

                # for image collector jangan terlalu dekat nnti error
                # jauh kit

                cv2.imshow("Image", img)
                key = cv2.waitKey(1)
                if key == ord("s"):
                    counter += 1
                    cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
                    print(counter)
                if key == ord('w'):
                    root = Tk()

                    root.title('Note')
                    root.iconbitmap('AS (1).ico')
                    root.geometry('300x150')
                    root.configure(background="#C1CDCD")

                    text_Label = Label(root, text='Successfully add ' + str(counter) + ' Data Capture !!', fg='black',
                                       bg='#FFFFFF')
                    text_Label.pack(pady=(30, 30))
                    text_Label.config(font=('Amasis MT Pro Medium', 10))

            cap.close()

        elif user_C.lower() == 'j':
            cap = cv2.VideoCapture(0)
            detector = HandDetector(maxHands=1)

            offset = 20
            imgsize = 300

            folder = "D:\Cisco\Phyton\pythonProject\J"
            counter = 0

            while True:
                success, img = cap.read()
                hands, img = detector.findHands(img)
                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']

                    imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
                    imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                    imgCropShape = imgCrop.shape

                    # Image collectore Resolution Capture

                    aspectRatio = h / w

                    if aspectRatio > 1:
                        k = imgsize / h  # Video capture Hight+Size Resolution
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgsize))
                        imgResizeShape = imgResize.shape
                        wGap = math.ceil((imgsize - wCal) / 2)  # Vide capture Centre postion
                        imgWhite[:, wGap:wCal + wGap] = imgResize  # Declare Resolution

                    else:
                        k = imgsize / w  # Video capture data more than rectangle size that being set
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgsize, hCal))
                        imgResizeShape = imgResize.shape
                        hGap = math.ceil((imgsize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize  # Declare Resolution

                    cv2.imshow("ImageCrop", imgCrop)
                    cv2.imshow("ImageWhite", imgWhite)

                # for image collector jangan terlalu dekat nnti error
                # jauh kit

                cv2.imshow("Image", img)
                key = cv2.waitKey(1)
                if key == ord("s"):
                    counter += 1
                    cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
                    print(counter)
                if key == ord('w'):
                    root = Tk()

                    root.title('Note')
                    root.iconbitmap('AS (1).ico')
                    root.geometry('300x150')
                    root.configure(background="#C1CDCD")

                    text_Label = Label(root, text='Successfully add ' + str(counter) + ' Data Capture !!', fg='black',
                                       bg='#FFFFFF')
                    text_Label.pack(pady=(30, 30))
                    text_Label.config(font=('Amasis MT Pro Medium', 10))

            cap.close()

        elif user_C.lower() == 'k':
            cap = cv2.VideoCapture(0)
            detector = HandDetector(maxHands=1)

            offset = 20
            imgsize = 300

            folder = "D:\Cisco\Phyton\pythonProject\K"
            counter = 0

            while True:
                success, img = cap.read()
                hands, img = detector.findHands(img)
                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']

                    imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
                    imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                    imgCropShape = imgCrop.shape

                    # Image collectore Resolution Capture

                    aspectRatio = h / w

                    if aspectRatio > 1:
                        k = imgsize / h  # Video capture Hight+Size Resolution
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgsize))
                        imgResizeShape = imgResize.shape
                        wGap = math.ceil((imgsize - wCal) / 2)  # Vide capture Centre postion
                        imgWhite[:, wGap:wCal + wGap] = imgResize  # Declare Resolution

                    else:
                        k = imgsize / w  # Video capture data more than rectangle size that being set
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgsize, hCal))
                        imgResizeShape = imgResize.shape
                        hGap = math.ceil((imgsize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize  # Declare Resolution

                    cv2.imshow("ImageCrop", imgCrop)
                    cv2.imshow("ImageWhite", imgWhite)

                # for image collector jangan terlalu dekat nnti error
                # jauh kit

                cv2.imshow("Image", img)
                key = cv2.waitKey(1)
                if key == ord("s"):
                    counter += 1
                    cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
                    print(counter)
                if key == ord('w'):
                    root = Tk()

                    root.title('Note')
                    root.iconbitmap('AS (1).ico')
                    root.geometry('300x150')
                    root.configure(background="#C1CDCD")

                    text_Label = Label(root, text='Successfully add ' + str(counter) + ' Data Capture !!', fg='black',
                                       bg='#FFFFFF')
                    text_Label.pack(pady=(30, 30))
                    text_Label.config(font=('Amasis MT Pro Medium', 10))

            cap.close()


        elif user_C.lower() == 'l':
            cap = cv2.VideoCapture(0)
            detector = HandDetector(maxHands=1)

            offset = 20
            imgsize = 300

            folder = "D:\Cisco\Phyton\pythonProject\L"
            counter = 0

            while True:
                success, img = cap.read()
                hands, img = detector.findHands(img)
                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']

                    imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
                    imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                    imgCropShape = imgCrop.shape

                    # Image collectore Resolution Capture

                    aspectRatio = h / w

                    if aspectRatio > 1:
                        k = imgsize / h  # Video capture Hight+Size Resolution
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgsize))
                        imgResizeShape = imgResize.shape
                        wGap = math.ceil((imgsize - wCal) / 2)  # Vide capture Centre postion
                        imgWhite[:, wGap:wCal + wGap] = imgResize  # Declare Resolution

                    else:
                        k = imgsize / w  # Video capture data more than rectangle size that being set
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgsize, hCal))
                        imgResizeShape = imgResize.shape
                        hGap = math.ceil((imgsize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize  # Declare Resolution

                    cv2.imshow("ImageCrop", imgCrop)
                    cv2.imshow("ImageWhite", imgWhite)

                # for image collector jangan terlalu dekat nnti error
                # jauh kit

                cv2.imshow("Image", img)
                key = cv2.waitKey(1)
                if key == ord("s"):
                    counter += 1
                    cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
                    print(counter)
                if key == ord('w'):
                    root = Tk()

                    root.title('Note')
                    root.iconbitmap('AS (1).ico')
                    root.geometry('300x150')
                    root.configure(background="#C1CDCD")

                    text_Label = Label(root, text='Successfully add ' + str(counter) + ' Data Capture !!', fg='black',
                                       bg='#FFFFFF')
                    text_Label.pack(pady=(30, 30))
                    text_Label.config(font=('Amasis MT Pro Medium', 10))

            cap.close()
        elif user_C.lower() == 'm':
            cap = cv2.VideoCapture(0)
            detector = HandDetector(maxHands=1)

            offset = 20
            imgsize = 300

            folder = "D:\Cisco\Phyton\pythonProject\M"
            counter = 0

            while True:
                success, img = cap.read()
                hands, img = detector.findHands(img)
                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']

                    imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
                    imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                    imgCropShape = imgCrop.shape

                    # Image collectore Resolution Capture

                    aspectRatio = h / w

                    if aspectRatio > 1:
                        k = imgsize / h  # Video capture Hight+Size Resolution
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgsize))
                        imgResizeShape = imgResize.shape
                        wGap = math.ceil((imgsize - wCal) / 2)  # Vide capture Centre postion
                        imgWhite[:, wGap:wCal + wGap] = imgResize  # Declare Resolution

                    else:
                        k = imgsize / w  # Video capture data more than rectangle size that being set
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgsize, hCal))
                        imgResizeShape = imgResize.shape
                        hGap = math.ceil((imgsize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize  # Declare Resolution

                    cv2.imshow("ImageCrop", imgCrop)
                    cv2.imshow("ImageWhite", imgWhite)

                # for image collector jangan terlalu dekat nnti error
                # jauh kit

                cv2.imshow("Image", img)
                key = cv2.waitKey(1)
                if key == ord("s"):
                    counter += 1
                    cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
                    print(counter)
                if key == ord('w'):
                    break
            cap.close()
        elif user_C.lower() == 'n':
            cap = cv2.VideoCapture(0)
            detector = HandDetector(maxHands=1)

            offset = 20
            imgsize = 300

            folder = "D:\Cisco\Phyton\pythonProject\AN"
            counter = 0

            while True:
                success, img = cap.read()
                hands, img = detector.findHands(img)
                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']

                    imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
                    imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                    imgCropShape = imgCrop.shape

                    # Image collectore Resolution Capture

                    aspectRatio = h / w

                    if aspectRatio > 1:
                        k = imgsize / h  # Video capture Hight+Size Resolution
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgsize))
                        imgResizeShape = imgResize.shape
                        wGap = math.ceil((imgsize - wCal) / 2)  # Vide capture Centre postion
                        imgWhite[:, wGap:wCal + wGap] = imgResize  # Declare Resolution

                    else:
                        k = imgsize / w  # Video capture data more than rectangle size that being set
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgsize, hCal))
                        imgResizeShape = imgResize.shape
                        hGap = math.ceil((imgsize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize  # Declare Resolution

                    cv2.imshow("ImageCrop", imgCrop)
                    cv2.imshow("ImageWhite", imgWhite)

                # for image collector jangan terlalu dekat nnti error
                # jauh kit

                cv2.imshow("Image", img)
                key = cv2.waitKey(1)
                if key == ord("s"):
                    counter += 1
                    cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
                    print(counter)
                if key == ord('w'):
                    root = Tk()

                    root.title('Note')
                    root.iconbitmap('AS (1).ico')
                    root.geometry('300x150')
                    root.configure(background="#C1CDCD")

                    text_Label = Label(root, text='Successfully add ' + str(counter) + ' Data Capture !!', fg='black',
                                       bg='#FFFFFF')
                    text_Label.pack(pady=(30, 30))
                    text_Label.config(font=('Amasis MT Pro Medium', 10))

            cap.close()
        elif user_C.lower() == 'o':
            cap = cv2.VideoCapture(0)
            detector = HandDetector(maxHands=1)

            offset = 20
            imgsize = 300

            folder = "D:\Cisco\Phyton\pythonProject\O"
            counter = 0

            while True:
                success, img = cap.read()
                hands, img = detector.findHands(img)
                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']

                    imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
                    imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                    imgCropShape = imgCrop.shape

                    # Image collectore Resolution Capture

                    aspectRatio = h / w

                    if aspectRatio > 1:
                        k = imgsize / h  # Video capture Hight+Size Resolution
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgsize))
                        imgResizeShape = imgResize.shape
                        wGap = math.ceil((imgsize - wCal) / 2)  # Vide capture Centre postion
                        imgWhite[:, wGap:wCal + wGap] = imgResize  # Declare Resolution

                    else:
                        k = imgsize / w  # Video capture data more than rectangle size that being set
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgsize, hCal))
                        imgResizeShape = imgResize.shape
                        hGap = math.ceil((imgsize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize  # Declare Resolution

                    cv2.imshow("ImageCrop", imgCrop)
                    cv2.imshow("ImageWhite", imgWhite)

                # for image collector jangan terlalu dekat nnti error
                # jauh kit

                cv2.imshow("Image", img)
                key = cv2.waitKey(1)
                if key == ord("s"):
                    counter += 1
                    cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
                    print(counter)
                if key == ord('w'):
                    root = Tk()

                    root.title('Note')
                    root.iconbitmap('AS (1).ico')
                    root.geometry('300x150')
                    root.configure(background="#C1CDCD")

                    text_Label = Label(root, text='Successfully add ' + str(counter) + ' Data Capture !!', fg='black',
                                       bg='#FFFFFF')
                    text_Label.pack(pady=(30, 30))
                    text_Label.config(font=('Amasis MT Pro Medium', 10))

            cap.close()
        elif user_C.lower() == 'p':
            cap = cv2.VideoCapture(0)
            detector = HandDetector(maxHands=1)

            offset = 20
            imgsize = 300

            folder = "D:\Cisco\Phyton\pythonProject\P"
            counter = 0

            while True:
                success, img = cap.read()
                hands, img = detector.findHands(img)
                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']

                    imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
                    imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                    imgCropShape = imgCrop.shape

                    # Image collectore Resolution Capture

                    aspectRatio = h / w

                    if aspectRatio > 1:
                        k = imgsize / h  # Video capture Hight+Size Resolution
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgsize))
                        imgResizeShape = imgResize.shape
                        wGap = math.ceil((imgsize - wCal) / 2)  # Vide capture Centre postion
                        imgWhite[:, wGap:wCal + wGap] = imgResize  # Declare Resolution

                    else:
                        k = imgsize / w  # Video capture data more than rectangle size that being set
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgsize, hCal))
                        imgResizeShape = imgResize.shape
                        hGap = math.ceil((imgsize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize  # Declare Resolution

                    cv2.imshow("ImageCrop", imgCrop)
                    cv2.imshow("ImageWhite", imgWhite)

                # for image collector jangan terlalu dekat nnti error
                # jauh kit

                cv2.imshow("Image", img)
                key = cv2.waitKey(1)
                if key == ord("s"):
                    counter += 1
                    cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
                    print(counter)
                if key == ord('w'):
                    root = Tk()

                    root.title('Note')
                    root.iconbitmap('AS (1).ico')
                    root.geometry('300x150')
                    root.configure(background="#C1CDCD")

                    text_Label = Label(root, text='Successfully add ' + str(counter) + ' Data Capture !!', fg='black',
                                       bg='#FFFFFF')
                    text_Label.pack(pady=(30, 30))
                    text_Label.config(font=('Amasis MT Pro Medium', 10))

            cap.close()
        elif user_C.lower() == 'q':
            cap = cv2.VideoCapture(0)
            detector = HandDetector(maxHands=1)

            offset = 20
            imgsize = 300

            folder = "D:\Cisco\Phyton\pythonProject\Q"
            counter = 0

            while True:
                success, img = cap.read()
                hands, img = detector.findHands(img)
                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']

                    imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
                    imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                    imgCropShape = imgCrop.shape

                    # Image collectore Resolution Capture

                    aspectRatio = h / w

                    if aspectRatio > 1:
                        k = imgsize / h  # Video capture Hight+Size Resolution
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgsize))
                        imgResizeShape = imgResize.shape
                        wGap = math.ceil((imgsize - wCal) / 2)  # Vide capture Centre postion
                        imgWhite[:, wGap:wCal + wGap] = imgResize  # Declare Resolution

                    else:
                        k = imgsize / w  # Video capture data more than rectangle size that being set
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgsize, hCal))
                        imgResizeShape = imgResize.shape
                        hGap = math.ceil((imgsize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize  # Declare Resolution

                    cv2.imshow("ImageCrop", imgCrop)
                    cv2.imshow("ImageWhite", imgWhite)

                # for image collector jangan terlalu dekat nnti error
                # jauh kit

                cv2.imshow("Image", img)
                key = cv2.waitKey(1)
                if key == ord("s"):
                    counter += 1
                    cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
                    print(counter)
                if key == ord('w'):
                    root = Tk()

                    root.title('Note')
                    root.iconbitmap('AS (1).ico')
                    root.geometry('300x150')
                    root.configure(background="#C1CDCD")

                    text_Label = Label(root, text='Successfully add ' + str(counter) + ' Data Capture !!', fg='black',
                                       bg='#FFFFFF')
                    text_Label.pack(pady=(30, 30))
                    text_Label.config(font=('Amasis MT Pro Medium', 10))

            cap.close()
        elif user_C.lower() == 'r':
            cap = cv2.VideoCapture(0)
            detector = HandDetector(maxHands=1)

            offset = 20
            imgsize = 300

            folder = "D:\Cisco\Phyton\pythonProject\R"
            counter = 0

            while True:
                success, img = cap.read()
                hands, img = detector.findHands(img)
                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']

                    imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
                    imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                    imgCropShape = imgCrop.shape

                    # Image collectore Resolution Capture

                    aspectRatio = h / w

                    if aspectRatio > 1:
                        k = imgsize / h  # Video capture Hight+Size Resolution
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgsize))
                        imgResizeShape = imgResize.shape
                        wGap = math.ceil((imgsize - wCal) / 2)  # Vide capture Centre postion
                        imgWhite[:, wGap:wCal + wGap] = imgResize  # Declare Resolution

                    else:
                        k = imgsize / w  # Video capture data more than rectangle size that being set
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgsize, hCal))
                        imgResizeShape = imgResize.shape
                        hGap = math.ceil((imgsize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize  # Declare Resolution

                    cv2.imshow("ImageCrop", imgCrop)
                    cv2.imshow("ImageWhite", imgWhite)

                # for image collector jangan terlalu dekat nnti error
                # jauh kit

                cv2.imshow("Image", img)
                key = cv2.waitKey(1)
                if key == ord("s"):
                    counter += 1
                    cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
                    print(counter)
                if key == ord('w'):
                    root = Tk()

                    root.title('Note')
                    root.iconbitmap('AS (1).ico')
                    root.geometry('300x150')
                    root.configure(background="#C1CDCD")

                    text_Label = Label(root, text='Successfully add ' + str(counter) + ' Data Capture !!', fg='black',
                                       bg='#FFFFFF')
                    text_Label.pack(pady=(30, 30))
                    text_Label.config(font=('Amasis MT Pro Medium', 10))

            cap.close()
        elif user_C.lower() == 's':
            cap = cv2.VideoCapture(0)
            detector = HandDetector(maxHands=1)

            offset = 20
            imgsize = 300

            folder = "D:\Cisco\Phyton\pythonProject\S"
            counter = 0

            while True:
                success, img = cap.read()
                hands, img = detector.findHands(img)
                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']

                    imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
                    imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                    imgCropShape = imgCrop.shape

                    # Image collectore Resolution Capture

                    aspectRatio = h / w

                    if aspectRatio > 1:
                        k = imgsize / h  # Video capture Hight+Size Resolution
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgsize))
                        imgResizeShape = imgResize.shape
                        wGap = math.ceil((imgsize - wCal) / 2)  # Vide capture Centre postion
                        imgWhite[:, wGap:wCal + wGap] = imgResize  # Declare Resolution

                    else:
                        k = imgsize / w  # Video capture data more than rectangle size that being set
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgsize, hCal))
                        imgResizeShape = imgResize.shape
                        hGap = math.ceil((imgsize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize  # Declare Resolution

                    cv2.imshow("ImageCrop", imgCrop)
                    cv2.imshow("ImageWhite", imgWhite)

                # for image collector jangan terlalu dekat nnti error
                # jauh kit

                cv2.imshow("Image", img)
                key = cv2.waitKey(1)
                if key == ord("s"):
                    counter += 1
                    cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
                    print(counter)
                if key == ord('w'):
                    root = Tk()

                    root.title('Note')
                    root.iconbitmap('AS (1).ico')
                    root.geometry('300x150')
                    root.configure(background="#C1CDCD")

                    text_Label = Label(root, text='Successfully add ' + str(counter) + ' Data Capture !!', fg='black',
                                       bg='#FFFFFF')
                    text_Label.pack(pady=(30, 30))
                    text_Label.config(font=('Amasis MT Pro Medium', 10))

            cap.close()
        elif user_C.lower() == 't':
            cap = cv2.VideoCapture(0)
            detector = HandDetector(maxHands=1)

            offset = 20
            imgsize = 300

            folder = "D:\Cisco\Phyton\pythonProject\T"
            counter = 0

            while True:
                success, img = cap.read()
                hands, img = detector.findHands(img)
                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']

                    imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
                    imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                    imgCropShape = imgCrop.shape

                    # Image collectore Resolution Capture

                    aspectRatio = h / w

                    if aspectRatio > 1:
                        k = imgsize / h  # Video capture Hight+Size Resolution
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgsize))
                        imgResizeShape = imgResize.shape
                        wGap = math.ceil((imgsize - wCal) / 2)  # Vide capture Centre postion
                        imgWhite[:, wGap:wCal + wGap] = imgResize  # Declare Resolution

                    else:
                        k = imgsize / w  # Video capture data more than rectangle size that being set
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgsize, hCal))
                        imgResizeShape = imgResize.shape
                        hGap = math.ceil((imgsize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize  # Declare Resolution

                    cv2.imshow("ImageCrop", imgCrop)
                    cv2.imshow("ImageWhite", imgWhite)

                # for image collector jangan terlalu dekat nnti error
                # jauh kit

                cv2.imshow("Image", img)
                key = cv2.waitKey(1)
                if key == ord("s"):
                    counter += 1
                    cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
                    print(counter)
                if key == ord('w'):
                    root = Tk()

                    root.title('Note')
                    root.iconbitmap('AS (1).ico')
                    root.geometry('300x150')
                    root.configure(background="#C1CDCD")

                    text_Label = Label(root, text='Successfully add ' + str(counter) + ' Data Capture !!', fg='black',
                                       bg='#FFFFFF')
                    text_Label.pack(pady=(30, 30))
                    text_Label.config(font=('Amasis MT Pro Medium', 10))

            cap.close()
        elif user_C.lower() == 'u':
            cap = cv2.VideoCapture(0)
            detector = HandDetector(maxHands=1)

            offset = 20
            imgsize = 300

            folder = "D:\Cisco\Phyton\pythonProject\OU"
            counter = 0

            while True:
                success, img = cap.read()
                hands, img = detector.findHands(img)
                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']

                    imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
                    imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                    imgCropShape = imgCrop.shape

                    # Image collectore Resolution Capture

                    aspectRatio = h / w

                    if aspectRatio > 1:
                        k = imgsize / h  # Video capture Hight+Size Resolution
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgsize))
                        imgResizeShape = imgResize.shape
                        wGap = math.ceil((imgsize - wCal) / 2)  # Vide capture Centre postion
                        imgWhite[:, wGap:wCal + wGap] = imgResize  # Declare Resolution

                    else:
                        k = imgsize / w  # Video capture data more than rectangle size that being set
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgsize, hCal))
                        imgResizeShape = imgResize.shape
                        hGap = math.ceil((imgsize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize  # Declare Resolution

                    cv2.imshow("ImageCrop", imgCrop)
                    cv2.imshow("ImageWhite", imgWhite)

                # for image collector jangan terlalu dekat nnti error
                # jauh kit

                cv2.imshow("Image", img)
                key = cv2.waitKey(1)
                if key == ord("s"):
                    counter += 1
                    cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
                    print(counter)
                if key == ord('w'):
                    root = Tk()

                    root.title('Note')
                    root.iconbitmap('AS (1).ico')
                    root.geometry('300x150')
                    root.configure(background="#C1CDCD")

                    text_Label = Label(root, text='Successfully add ' + str(counter) + ' Data Capture !!', fg='black',
                                       bg='#FFFFFF')
                    text_Label.pack(pady=(30, 30))
                    text_Label.config(font=('Amasis MT Pro Medium', 10))

            cap.close()
        elif user_C.lower() == 'v':
            cap = cv2.VideoCapture(0)
            detector = HandDetector(maxHands=1)

            offset = 20
            imgsize = 300

            folder = "D:\Cisco\Phyton\pythonProject\V"
            counter = 0

            while True:
                success, img = cap.read()
                hands, img = detector.findHands(img)
                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']

                    imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
                    imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                    imgCropShape = imgCrop.shape

                    # Image collectore Resolution Capture

                    aspectRatio = h / w

                    if aspectRatio > 1:
                        k = imgsize / h  # Video capture Hight+Size Resolution
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgsize))
                        imgResizeShape = imgResize.shape
                        wGap = math.ceil((imgsize - wCal) / 2)  # Vide capture Centre postion
                        imgWhite[:, wGap:wCal + wGap] = imgResize  # Declare Resolution

                    else:
                        k = imgsize / w  # Video capture data more than rectangle size that being set
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgsize, hCal))
                        imgResizeShape = imgResize.shape
                        hGap = math.ceil((imgsize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize  # Declare Resolution

                    cv2.imshow("ImageCrop", imgCrop)
                    cv2.imshow("ImageWhite", imgWhite)

                # for image collector jangan terlalu dekat nnti error
                # jauh kit

                cv2.imshow("Image", img)
                key = cv2.waitKey(1)
                if key == ord("s"):
                    counter += 1
                    cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
                    print(counter)
                if key == ord('w'):
                    root = Tk()

                    root.title('Note')
                    root.iconbitmap('AS (1).ico')
                    root.geometry('300x150')
                    root.configure(background="#C1CDCD")

                    text_Label = Label(root, text='Successfully add ' + str(counter) + ' Data Capture !!', fg='black',
                                       bg='#FFFFFF')
                    text_Label.pack(pady=(30, 30))
                    text_Label.config(font=('Amasis MT Pro Medium', 10))

            cap.close()
        elif user_C.lower() == 'w':
            cap = cv2.VideoCapture(0)
            detector = HandDetector(maxHands=1)

            offset = 20
            imgsize = 300

            folder = "D:\Cisco\Phyton\pythonProject\W"
            counter = 0

            while True:
                success, img = cap.read()
                hands, img = detector.findHands(img)
                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']

                    imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
                    imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                    imgCropShape = imgCrop.shape

                    # Image collectore Resolution Capture

                    aspectRatio = h / w

                    if aspectRatio > 1:
                        k = imgsize / h  # Video capture Hight+Size Resolution
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgsize))
                        imgResizeShape = imgResize.shape
                        wGap = math.ceil((imgsize - wCal) / 2)  # Vide capture Centre postion
                        imgWhite[:, wGap:wCal + wGap] = imgResize  # Declare Resolution

                    else:
                        k = imgsize / w  # Video capture data more than rectangle size that being set
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgsize, hCal))
                        imgResizeShape = imgResize.shape
                        hGap = math.ceil((imgsize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize  # Declare Resolution

                    cv2.imshow("ImageCrop", imgCrop)
                    cv2.imshow("ImageWhite", imgWhite)

                # for image collector jangan terlalu dekat nnti error
                # jauh kit

                cv2.imshow("Image", img)
                key = cv2.waitKey(1)
                if key == ord("s"):
                    counter += 1
                    cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
                    print(counter)
                if key == ord('w'):
                    root = Tk()

                    root.title('Note')
                    root.iconbitmap('AS (1).ico')
                    root.geometry('300x150')
                    root.configure(background="#C1CDCD")

                    text_Label = Label(root, text='Successfully add ' + str(counter) + ' Data Capture !!', fg='black',
                                       bg='#FFFFFF')
                    text_Label.pack(pady=(30, 30))
                    text_Label.config(font=('Amasis MT Pro Medium', 10))

            cap.close()
        elif user_C.lower() == 'x':
            cap = cv2.VideoCapture(0)
            detector = HandDetector(maxHands=1)

            offset = 20
            imgsize = 300

            folder = "D:\Cisco\Phyton\pythonProject\X"
            counter = 0

            while True:
                success, img = cap.read()
                hands, img = detector.findHands(img)
                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']

                    imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
                    imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                    imgCropShape = imgCrop.shape

                    # Image collectore Resolution Capture

                    aspectRatio = h / w

                    if aspectRatio > 1:
                        k = imgsize / h  # Video capture Hight+Size Resolution
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgsize))
                        imgResizeShape = imgResize.shape
                        wGap = math.ceil((imgsize - wCal) / 2)  # Vide capture Centre postion
                        imgWhite[:, wGap:wCal + wGap] = imgResize  # Declare Resolution

                    else:
                        k = imgsize / w  # Video capture data more than rectangle size that being set
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgsize, hCal))
                        imgResizeShape = imgResize.shape
                        hGap = math.ceil((imgsize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize  # Declare Resolution

                    cv2.imshow("ImageCrop", imgCrop)
                    cv2.imshow("ImageWhite", imgWhite)

                # for image collector jangan terlalu dekat nnti error
                # jauh kit

                cv2.imshow("Image", img)
                key = cv2.waitKey(1)
                if key == ord("s"):
                    counter += 1
                    cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
                    print(counter)
                if key == ord('w'):
                    root = Tk()

                    root.title('Note')
                    root.iconbitmap('AS (1).ico')
                    root.geometry('300x150')
                    root.configure(background="#C1CDCD")

                    text_Label = Label(root, text='Successfully add ' + str(counter) + ' Data Capture !!', fg='black',
                                       bg='#FFFFFF')
                    text_Label.pack(pady=(30, 30))
                    text_Label.config(font=('Amasis MT Pro Medium', 10))

            cap.close()
        elif user_C.lower() == 'y':
            cap = cv2.VideoCapture(0)
            detector = HandDetector(maxHands=1)

            offset = 20
            imgsize = 300

            folder = "D:\Cisco\Phyton\pythonProject\Y"
            counter = 0

            while True:
                success, img = cap.read()
                hands, img = detector.findHands(img)
                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']

                    imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
                    imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                    imgCropShape = imgCrop.shape

                    # Image collectore Resolution Capture

                    aspectRatio = h / w

                    if aspectRatio > 1:
                        k = imgsize / h  # Video capture Hight+Size Resolution
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgsize))
                        imgResizeShape = imgResize.shape
                        wGap = math.ceil((imgsize - wCal) / 2)  # Vide capture Centre postion
                        imgWhite[:, wGap:wCal + wGap] = imgResize  # Declare Resolution

                    else:
                        k = imgsize / w  # Video capture data more than rectangle size that being set
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgsize, hCal))
                        imgResizeShape = imgResize.shape
                        hGap = math.ceil((imgsize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize  # Declare Resolution

                    cv2.imshow("ImageCrop", imgCrop)
                    cv2.imshow("ImageWhite", imgWhite)

                # for image collector jangan terlalu dekat nnti error
                # jauh kit

                cv2.imshow("Image", img)
                key = cv2.waitKey(1)
                if key == ord("s"):
                    counter += 1
                    cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
                    print(counter)
                if key == ord('w'):
                    root = Tk()

                    root.title('Note')
                    root.iconbitmap('AS (1).ico')
                    root.geometry('300x150')
                    root.configure(background="#C1CDCD")

                    text_Label = Label(root, text='Successfully add ' + str(counter) + ' Data Capture !!', fg='black',
                                       bg='#FFFFFF')
                    text_Label.pack(pady=(30, 30))
                    text_Label.config(font=('Amasis MT Pro Medium', 10))

            cap.close()
        elif user_C.lower() == 'z':
            cap = cv2.VideoCapture(0)
            detector = HandDetector(maxHands=1)

            offset = 20
            imgsize = 300

            folder = "D:\Cisco\Phyton\pythonProject\Z"
            counter = 0

            while True:
                success, img = cap.read()
                hands, img = detector.findHands(img)
                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']

                    imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
                    imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                    imgCropShape = imgCrop.shape

                    # Image collectore Resolution Capture

                    aspectRatio = h / w

                    if aspectRatio > 1:
                        k = imgsize / h  # Video capture Hight+Size Resolution
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgsize))
                        imgResizeShape = imgResize.shape
                        wGap = math.ceil((imgsize - wCal) / 2)  # Vide capture Centre postion
                        imgWhite[:, wGap:wCal + wGap] = imgResize  # Declare Resolution

                    else:
                        k = imgsize / w  # Video capture data more than rectangle size that being set
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgsize, hCal))
                        imgResizeShape = imgResize.shape
                        hGap = math.ceil((imgsize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize  # Declare Resolution

                    cv2.imshow("ImageCrop", imgCrop)
                    cv2.imshow("ImageWhite", imgWhite)

                # for image collector jangan terlalu dekat nnti error
                # jauh kit

                cv2.imshow("Image", img)
                key = cv2.waitKey(1)
                if key == ord("s"):
                    counter += 1
                    cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
                    print(counter)
                if key == ord('w'):
                    root = Tk()

                    root.title('Note')
                    root.iconbitmap('AS (1).ico')
                    root.geometry('300x150')
                    root.configure(background="#C1CDCD")

                    text_Label = Label(root, text='Successfully add ' + str(counter) + ' Data Capture !!', fg='black',
                                       bg='#FFFFFF')
                    text_Label.pack(pady=(30, 30))
                    text_Label.config(font=('Amasis MT Pro Medium', 10))

            cap.close()
    root = Tk()

    root.title('ASLDs')
    root.iconbitmap('AS (1).ico')
    root.geometry('350x500')
    root.configure(background="#C1CDCD")

    text1_Label = Label(root, text='Data Option', fg='black', bg='#FFFFFF')
    text1_Label.pack(pady=(20,10))
    text1_Label.config(font=('Amasis MT Pro Medium', 15))

    text1_Label = Label(root, text='Please choose your Alphabet option for Data collection\n A-Z' , fg='black', bg='#FFFFFF')
    text1_Label.pack(pady=(10,5))
    text1_Label.config(font=('Amasis MT Pro Medium', 9))

    option_list = Entry(root, width=30)
    option_list.pack(ipady=6,pady=(1,15))

    ok1_btn = Button(root, text='OK', fg='black', bg='azure1', width=30, height=1, command=ok_btn)
    ok1_btn.pack(pady=(2, 40))
    ok1_btn.config(font=('Amasis MT Pro Medium', 10))

def exit_btn():
    root.destroy()

root = Tk()

root.title('ASLDs')
root.iconbitmap('AS (1).ico')
root.geometry('350x500')
root.configure(background="#C1CDCD")

img=Image.open('AS.png')
resize_img=img.resize((100,100))
img=ImageTk.PhotoImage(resize_img)

image_label=Label(root,image=img)
image_label.pack (pady=(11,11))

text_Label=Label(root,text='MENU',fg='black',bg='#FFFFFF')
text_Label.pack()
text_Label.config(font=('Amasis MT Pro Medium',15))

scan_btn=Button(root,text='Scan', fg='black',bg='azure1', width=30,height=1, command=scan_Gesture)
scan_btn.pack(pady=(50,40))
scan_btn.config(font=('Amasis MT Pro Medium',10))

datac_btn=Button(root,text='New Sign Gesture', fg='black',bg='azure1', width=30,height=1,command=data_collection)
datac_btn.pack(pady=(2,40))
datac_btn.config(font=('Amasis MT Pro Medium',10))

exit_btn=Button(root,text='Exit', fg='black',bg='azure1', width=30,height=1,command=exit_btn)
exit_btn.pack(pady=(2,40))
exit_btn.config(font=('Amasis MT Pro Medium',10))

root.mainloop()
