import numpy as np
import cv2
#from CurvedLaneDetection import utlis
from utlis import *


import cv2
import numpy as np
from scipy.stats import itemfreq
import RPi.GPIO as GPIO
import datetime
import time
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

IN1=2
IN2=3
IN3=4
IN4=17

GPIO.setup(IN1,GPIO.OUT)
GPIO.setup(IN2,GPIO.OUT)
GPIO.setup(IN3,GPIO.OUT)
GPIO.setup(IN4,GPIO.OUT)

GPIO.output(IN1,False)
GPIO.output(IN2,False)
GPIO.output(IN3,False)
GPIO.output(IN4,False)

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
time.sleep(1)

def stop():
    print ("stop")
    GPIO.output(IN1,False)
    GPIO.output(IN2, False)
    GPIO.output(IN3,False)
    GPIO.output(IN4,False)
    #time.sleep(1)
    
def forward():
    GPIO.output(IN1,False)
    GPIO.output(IN2, True)
    GPIO.output(IN3,False)
    GPIO.output(IN4,True)
    
    print ("Forward")

def backward():
    GPIO.output(IN1,True)
    GPIO.output(IN2, False)
    GPIO.output(IN3,True)
    GPIO.output(IN4,False)
    time.sleep(1)
    print ("backword")

def left():
    GPIO.output(IN1,False)
    GPIO.output(IN2, True)
    GPIO.output(IN3,True)
    GPIO.output(IN4,False)
    time.sleep(1)
    print ("left")

def right():
    GPIO.output(IN1,True)
    GPIO.output(IN2, False)
    GPIO.output(IN3,False)
    GPIO.output(IN4,True)
    #time.sleep(1)
    print("right")
####################################################

cameraFeed= False
videoPath = 'project_video.mp4'
cameraNo= 1
frameWidth= 640
frameHeight = 480

if cameraFeed:intialTracbarVals = [24,55,12,100] #  #wT,hT,wB,hB
else:intialTracbarVals = [42,63,14,87]   #wT,hT,wB,hB

####################################################


if cameraFeed:
    cap = cv2.VideoCapture(cameraNo)
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)
else:
    cap = cv2.VideoCapture(0)
count=0
noOfArrayValues =10
global arrayCurve, arrayCounter
arrayCounter=0
arrayCurve = np.zeros([noOfArrayValues])
myVals=[]
initializeTrackbars(intialTracbarVals)


while True:
    forward()
    success, img = cap.read()
    #img = cv2.imread('test3.jpg')
    if cameraFeed== False:img = cv2.resize(img, (frameWidth, frameHeight), None)
    imgWarpPoints = img.copy()
    imgFinal = img.copy()
    imgCanny = img.copy()

    imgUndis = undistort(img)
    imgThres,imgCanny,imgColor = thresholding(imgUndis)
    src = valTrackbars()
    imgWarp = perspective_warp(imgThres, dst_size=(frameWidth, frameHeight), src=src)
    imgWarpPoints = drawPoints(imgWarpPoints, src)
    imgSliding, curves, lanes, ploty = sliding_window(imgWarp, draw_windows=True)

    try:
        curverad =get_curve(imgFinal, curves[0], curves[1])
        lane_curve = np.mean([curverad[0], curverad[1]])
        imgFinal = draw_lanes(img, curves[0], curves[1],frameWidth,frameHeight,src=src)

        # ## Average
        currentCurve = lane_curve // 50
        if  int(np.sum(arrayCurve)) == 0:averageCurve = currentCurve
        else:
            averageCurve = np.sum(arrayCurve) // arrayCurve.shape[0]
        if abs(averageCurve-currentCurve) >100: arrayCurve[arrayCounter] = averageCurve
        else :arrayCurve[arrayCounter] = currentCurve
        arrayCounter +=1
        if arrayCounter >=noOfArrayValues : arrayCounter=0
        cv2.putText(imgFinal, str(int(averageCurve)), (frameWidth//2-70, 70), cv2.FONT_HERSHEY_DUPLEX, 1.75, (0, 0, 255), 2, cv2.LINE_AA)
        print(int(averageCurve))
        if(int(averageCurve)<-70):
            print('right side lane')
            left()
##            time.sleep(5)
##            data.write(str.encode('r'))
            #time.sleep(3)
            
        elif(int(averageCurve)>60):
            print('left side lane')
            right()
##            time.sleep(5)
##            data.write(str.encode('l'))
           # time.sleep(3)
        else:
            print('Straight')
            forward()
##            time.sleep(5)
##            data.write(str.encode('f'))
            #time.sleep(3)

    except:
        lane_curve=00
        pass

    imgFinal= drawLines(imgFinal,lane_curve)


    imgThres = cv2.cvtColor(imgThres,cv2.COLOR_GRAY2BGR)
    imgBlank = np.zeros_like(img)
    imgStacked = stackImages(0.7, ([img,imgUndis,imgWarpPoints],
                                         [imgColor, imgCanny, imgThres],
                                         [imgWarp,imgSliding,imgFinal]
                                         ))

    cv2.imshow("PipeLine",imgStacked)
    cv2.imshow("Result", imgFinal)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
