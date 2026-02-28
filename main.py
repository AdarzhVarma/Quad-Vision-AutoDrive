import cv2
import numpy as np
import time
import argparse
from src.lane_tracker import pipeline, perspective_warp, sliding_window, get_curve, draw_lanes, textDisplay
from src.obstacle_detector import get_dominant_color

# ==========================================
# GPIO Setup (Wrapped in try-except for demo)
# ==========================================
USE_GPIO = True
try:
    import RPi.GPIO as GPIO
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
except ImportError:
    print("RPi.GPIO not found. Running in DEMO mode without physical motors.")
    USE_GPIO = False


# ==========================================
# Motor Control Functions
# ==========================================
def stop():
    print("ACTION: STOP")
    if USE_GPIO:
        GPIO.output(IN1,False)
        GPIO.output(IN2, False)
        GPIO.output(IN3,False)
        GPIO.output(IN4,False)
    
def forward():
    print("ACTION: FORWARD")
    if USE_GPIO:
        GPIO.output(IN1,False)
        GPIO.output(IN2, True)
        GPIO.output(IN3,False)
        GPIO.output(IN4,True)

def left():
    print("ACTION: LEFT")
    if USE_GPIO:
        GPIO.output(IN1,False)
        GPIO.output(IN2, True)
        GPIO.output(IN3,True)
        GPIO.output(IN4,False)

def right():
    print("ACTION: RIGHT")
    if USE_GPIO:
        GPIO.output(IN1,True)
        GPIO.output(IN2, False)
        GPIO.output(IN3,False)
        GPIO.output(IN4,True)

# ==========================================
# Main Processing Loop
# ==========================================
def main(demo_mode, video_path):
    print("Initializing Autonomous Vehicle System...")

    # Load Cascade Classifier for Potholes
    pothole_cascade_path = 'models/cascade.xml'
    try:
        cascade = cv2.CascadeClassifier(pothole_cascade_path)
    except Exception as e:
        print(f"Error loading cascade: {e}")
        return

    # Initialize Video Capture
    if demo_mode:
        cap = cv2.VideoCapture(video_path)
        print(f"Running on video file: {video_path}")
    else:
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)
        print("Running on Webcam.")

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Video Writer for Demo Output
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('data/demo_output.avi', fourcc, 20.0, (1280, 720))

    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Video stream ended or error reading frame.")
            break
        
        frame = cv2.resize(frame, (1280, 720))
        display_frame = frame.copy()
        
        # ---------------------------------------------------------
        # 1. Pothole / Obstacle Detection
        # ---------------------------------------------------------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        potholes = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=8)
        
        obstacle_detected = False
        for (x, y, w, h) in potholes:
            area = w * h
            if area > 1000:  # Min area threshold
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
                cv2.putText(display_frame, "POTHOLE DETECTED", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                obstacle_detected = True
                stop()
                break  # Prioritize stop if any pothole is found

        # ---------------------------------------------------------
        # 2. Traffic Sign Detection
        # ---------------------------------------------------------
        if not obstacle_detected:
            img_blur = cv2.medianBlur(gray, 37)
            circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, 50, param1=120, param2=40)
            
            sign_detected = False
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    x_c, y_c, r = i[0], i[1], i[2]
                    
                    if y_c > r and x_c > r:
                        square = frame[y_c-r:y_c+r, x_c-r:x_c+r]
                        if square.size > 0:
                            dominant_color = get_dominant_color(square, 2)
                            
                            cv2.circle(display_frame, (x_c, y_c), r, (0, 255, 0), 2)
                            
                            # Simple color heuristic from original script
                            if dominant_color[2] > 100: # Red-ish
                                cv2.putText(display_frame, "STOP SIGN", (x_c, y_c-r-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                                stop()
                                sign_detected = True
                            elif dominant_color[0] > 80: # Blue-ish (Turn signs)
                                # Simplified logic: arbitrarily call left/right based on zones if needed
                                cv2.putText(display_frame, "BLUE SIGN", (x_c, y_c-r-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                                sign_detected = True

            # ---------------------------------------------------------
            # 3. Lane Detection
            # ---------------------------------------------------------
            if not sign_detected:
                dst_size = (1280, 720)
                src_pts = np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)])
                
                # Apply Pipeline
                img_binary = pipeline(frame)
                
                # Perspective Warp
                img_warped = perspective_warp(img_binary, dst_size=dst_size, src=src_pts)
                
                # Sliding Window
                out_img, curves, lanes, ploty = sliding_window(img_warped, draw_windows=False)
                
                # Curvature
                try:
                    curverad = get_curve(display_frame, curves[0], curves[1])
                    lane_curve = curverad[2]
                    
                    # Draw Lanes
                    display_frame = draw_lanes(display_frame, curves[0], curves[1], 1280, 720, src=src_pts)
                    textDisplay(lane_curve, display_frame)

                    # Motor control based on lane curve
                    if lane_curve > 10:
                        right()
                    elif lane_curve < -10:
                        left()
                    elif -10 <= lane_curve <= 10:
                        forward()
                except Exception as e:
                    # In case curves fail to generate
                    pass

        # Write to demo output and display
        out.write(display_frame)
        cv2.imshow("Autonomous Vehicle Stream", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    if USE_GPIO:
        GPIO.cleanup()
    print("Processing Complete. Demo saved to data/demo_output.avi")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Vehicle Controller")
    parser.add_argument('--demo', action='store_true', help="Run in demo mode using a video file")
    parser.add_argument('--video', type=str, default='data/project_video.mp4', help="Path to video file for demo mode")
    args = parser.parse_args()
    
    main(args.demo, args.video)
