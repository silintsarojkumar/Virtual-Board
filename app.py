import cv2
import numpy as np
import time
import os
from cvzone.HandTrackingModule import HandDetector
import HandTracking as htm
import datetime

folderPath = 'Header'
myList = os.listdir(folderPath)

overlayList = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)



header = overlayList[0]
drawColor = (255,0,255)
brushThickness = 15
ereaserThickness = 80



os.makedirs("Screenshots", exist_ok=True)
cap = cv2.VideoCapture(1)
cap.set(3,1860)
cap.set(4,1080)
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

detector = htm.handDetector(detectionCon=0.85,maxHands=2)

xp,yp = 0,0

takeScreenshot = False
while True:
    # import Image
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img,1)

    if 'imgCanvas' not in globals():
        imgCanvas = np.zeros_like(img)


    # Find hand Landmark
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    if len(lmList) != 0:
        

        # tip of index and middle finger
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]
       

        # Check which finger are up
        fingers = detector.fingersUp()
        


        # if selection mode two fingers are up
        if fingers[1] and fingers[2]:
            xp,yp = x1,y1
            print(x1,y1)
            if y1 < 125:
                if 0<x1<450:
                    header = overlayList[0]
                    drawColor = (255, 0, 255)
                elif 450<x1<550:
                    header = overlayList[1]
                    drawColor = (0, 165, 255)
                elif 680<x1<730:
                    header = overlayList[2]
                    drawColor = (255, 255, 255)
                elif 815<x1<920:
                    header = overlayList[3]
                    drawColor = (0,0,0)
                elif 1020<x1<1120:
                    header = overlayList[0]
                    takeScreenshot = True

                    

            cv2.rectangle(img,(x1,y1-25),(x2,y2+25),drawColor,cv2.FILLED)

                


            
        

        # if drawing mode - Index finger is up
        if fingers[1] and not fingers[2]:
            cv2.circle(img,(x1,y1),15,drawColor,cv2.FILLED)
            if xp==0 and yp==0:
                xp,yp=x1,y1

            

                

            if drawColor == (0,0,0):
                cv2.line(imgCanvas, (xp, yp), (x1, y1), (0,0,0), ereaserThickness)
            else:
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
        xp,yp = x1,y1






    
    


    # Setting The Header image
    # Merge canvas with webcam image
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)

    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    if takeScreenshot:
        filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
        cv2.imwrite(f"Screenshots/{filename}", img)
        print(f"Screenshot saved: {filename}")
        takeScreenshot = False
        time.sleep(1)

    


    h, w = header.shape[:2]
    header = cv2.resize(header, (img.shape[1], header.shape[0]))
    img[0:header.shape[0], 0:img.shape[1]] = header
















    cv2.imshow("Image", img)
 
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
