import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

def convertKeypointsfromCam():
    # For webcam input:
    cap = cv2.VideoCapture(0)
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    up = False
    down = False
    reset = False
    count = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = pose.process(image)

        if not results.pose_landmarks:
            continue

        shoulder = [results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x,
                    results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y,
                    results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].z]
        elbow = [results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].x, 
                 results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].y,
                 results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].z]
        wrist = [results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].x, 
                 results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].y,
                 results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].z]

        print(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].z)
        print()         

        angle = calculateAngle(shoulder, elbow, wrist)

        if(angle > 130 and down == False):
            down = True
        elif(angle < 50 and up == False):
            up = True
        elif(angle > 130 and down == True and up == True):
            count = count + 1
            reset = True
            up = False
        elif(angle > 70 and angle < 100 and reset):
            down = False
            up = False
            reset = False
        
        cv2.putText(image, 'Angle : ' + str(angle), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, ( 0+10*count if 0+10*count<255 else 255,0,0), 2,cv2.LINE_AA)
        cv2.putText(image, 'Count : ' + str(count), (130,200), cv2.FONT_HERSHEY_SIMPLEX, 1 + 0.2*count, ( 0+10*count if 0+10*count<255 else 255,0,0), 2,cv2.LINE_AA)
        # print('--------')
        # print(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER])
        # print(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW])
        # print(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST])
        # print('--------')
        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break   
    cap.release()

def calculateAngle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    # radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    # radians1 = np.arccos(np.einsum('nt,nt->n',a-b, c-b))
    # angle1 = np.degrees(radians1)

    v1 = a-b
    v2 = c-b
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    dot_product = np.dot(v1_norm, v2_norm)
    radian1 = np.arccos(dot_product)
    angle1 = np.degrees(radian1)

    # radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    # angle = np.abs(radians*180.0/np.pi)
    # print(angle1, 'vs', 360-angle)  
    # if angle >180.0:
    #     angle = 360-angle
        
    return angle1


if __name__ == "__main__":
    print('cam converter')

