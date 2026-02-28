import cv2
import numpy as np
import time
# from scipy.stats import itemfreq 
# itemfreq is deprecated, use np.unique(labels, return_counts=True)
try:
    import RPi.GPIO as GPIO
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
except ImportError:
    print("RPi.GPIO not found in obstacle_detector. Mocking for demo.")
    GPIO = None

if GPIO:
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

if GPIO:
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    time.sleep(1)

def stop():
    #print ("stop")
    if GPIO:
        GPIO.output(IN1,False)
        GPIO.output(IN2, False)
        GPIO.output(IN3,False)
        GPIO.output(IN4,False)
        time.sleep(1)
    
def forward():
    if GPIO:
        GPIO.output(IN1,False)
        GPIO.output(IN2, True)
        GPIO.output(IN3,False)
        GPIO.output(IN4,True)
        time.sleep(1)
    #print ("Forward")

def backward():
    if GPIO:
        GPIO.output(IN1,True)
        GPIO.output(IN2, False)
        GPIO.output(IN3,True)
        GPIO.output(IN4,False)
        time.sleep(1)
    #print ("backword")

def left():
    if GPIO:
        GPIO.output(IN1,False)
        GPIO.output(IN2, True)
        GPIO.output(IN3,True)
        GPIO.output(IN4,False)
        time.sleep(1)
   #print ("left")

def right():
    if GPIO:
        GPIO.output(IN1,True)
        GPIO.output(IN2, False)
        GPIO.output(IN3,False)
        GPIO.output(IN4,True)
        time.sleep(1)
    
def get_dominant_color(image, n_colors):
    pixels = np.float32(image).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    flags, labels, centroids = cv2.kmeans(
        pixels, n_colors, None, criteria, 10, flags)
    palette = np.uint8(centroids)
    
    # Replace itemfreq with np.unique
    unique, counts = np.unique(labels, return_counts=True)
    dominant_label = unique[np.argmax(counts)]
    return palette[dominant_label]


clicked = False
def onMouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True

def detect_obstacles(frame, cascade=None, scaleVal=1.3, neig=8, minArea=1000, color=(255,0,255), objectName='pothole'):
    # This function is meant to replace the global execution
    # It takes a frame and returns marked pothole detections
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    objects = []
    
    if cascade is not None:
        objects = cascade.detectMultiScale(gray, scaleVal, neig)
        for (x, y, w, h) in objects:
            area = w * h
            if area > minArea:
                stop()
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                cv2.putText(frame, objectName, (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
    
    return frame, objects

if __name__ == '__main__':
    # This allows the script to still be run standalone if desired
    cap = cv2.VideoCapture(0)
    path = '../models/cascade.xml'
    try:
        cascade = cv2.CascadeClassifier(path)
    except:
        cascade = None
        
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        frame, objects = detect_obstacles(frame, cascade)
        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

