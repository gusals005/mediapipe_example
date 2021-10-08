import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

'''
    argument : pose results.
    nose : 0
    left_eye : 2, right_eye : 5
    left_ear: 7, right_ear : 8
    left_shoulder : 11, right_shoulder: 12
    left_elbow : 13, right_elbow:14
    left_wrist: 15, right_wrist:16
    left_hip : 23, right_hip: 24
    left_knee : 25, right_knee: 26
    left_ankle: 27, left_ankle: 28
'''
def getAngleThreePoint(results, poseIdx1, poseIdx2, poseIdx3):
    keypoint1 = [results.pose_landmarks.landmark[poseIdx1].x,results.pose_landmarks.landmark[poseIdx1].y]
    keypoint2 = [results.pose_landmarks.landmark[poseIdx2].x,results.pose_landmarks.landmark[poseIdx2].y]
    keypoint3 = [results.pose_landmarks.landmark[poseIdx3].x,results.pose_landmarks.landmark[poseIdx3].y]
    return calculateAngle(keypoint1,keypoint2,keypoint3)

def getAngleThreePoint3D(results, poseIdx1, poseIdx2, poseIdx3):
    keypoint1 = [results.pose_landmarks.landmark[poseIdx1].x,results.pose_landmarks.landmark[poseIdx1].y, results.pose_landmarks.landmark[poseIdx1].z]
    keypoint2 = [results.pose_landmarks.landmark[poseIdx2].x,results.pose_landmarks.landmark[poseIdx2].y, results.pose_landmarks.landmark[poseIdx2].z]
    keypoint3 = [results.pose_landmarks.landmark[poseIdx3].x,results.pose_landmarks.landmark[poseIdx3].y, results.pose_landmarks.landmark[poseIdx3].z]
    return calculateAngle(keypoint1,keypoint2,keypoint3)

def calculateAngles(results):
    leftElbowAngle = getAngleThreePoint(results, 11,13,15)
    rightElbowAngle = getAngleThreePoint(results, 12,14,16)
    leftShoudlerAngle = getAngleThreePoint(results, 12,11,13)
    rightShoudlerAngle = getAngleThreePoint(results, 11,12,14)
    leftHip2ElbowAngle = getAngleThreePoint(results, 23,11,13)
    rightHip2ElbowAngle = getAngleThreePoint(results, 24,12,14)
    leftHipAngle = getAngleThreePoint(results, 11,23,25)
    rightHipAngle = getAngleThreePoint(results, 12,24,26)
    leftHip2KneeAngle = getAngleThreePoint(results, 24,23,25)
    rightHip2KneeAngle = getAngleThreePoint(results, 23,24,26)
    leftKneeAngle = getAngleThreePoint(results, 23,25,27)
    rightKneeAngle = getAngleThreePoint(results, 24,26,28)

    returnDict = {
        'leftElbowAngle':leftElbowAngle,
        'rightElbowAngle':rightElbowAngle,
        'leftShoulderAngle':leftShoudlerAngle,
        'rightShoulderAngle':rightShoudlerAngle,
        'leftHip2ElbowAngle':leftHip2ElbowAngle,
        'rightHip2ElbowAngle':rightHip2ElbowAngle,
        'leftHipAngle':leftHipAngle,
        'rightHipAngle':rightHipAngle,
        'leftHip2KneeAngle':leftHip2KneeAngle, 
        'rightHip2KneeAngle':rightHip2KneeAngle,
        'leftKneeAngle':leftKneeAngle,
        'rightKneeAngle':rightKneeAngle
    }

    return returnDict

def workoutSeleter(angles, filename, up, down, reset, count):
    if filename == 'W007': 
        angle = angles['leftHipAngle']
        up, down, reset, count = countWorkout(angle, 160,120, up,down,reset,count, 1)
    elif filename == 'W008':
        angle = angles['rightShoulderAngle']
        up, down, reset, count = countWorkout(angle, 160,90, up,down,reset,count, 1)
    elif filename == 'W009':
        angle = angles['rightHip2ElbowAngle']
        up, down, reset, count = countWorkout(angle, 160,40, up,down,reset,count, 1)
    elif filename == 'W011':
        angle = angles['leftHipAngle']
        up, down, reset, count = countWorkout(angle, 110,95, up,down,reset,count, 0)
    elif filename == 'W012':    
        angle = angles['leftHipAngle']
        up, down, reset, count = countWorkout(angle, 110,90, up,down,reset,count, 0)

    return angle, up, down, reset, count
                

def countWorkout(angle, upper, lower, up,down,reset,count, mode):
    if mode == 0:
        if(angle > upper and up == False):
            up = True
        elif(angle < lower and down == False):
            down = True
        elif(angle > upper and up == True and down == True):
            count = count + 1
            reset = True
            down = False
        elif(angle > lower+5 and angle < upper-5 and reset):
            down = False
            up = False
            reset = False
    else:
        if(angle < lower and down == False):
            down = True
        elif(angle > upper and up == False):
            up = True
        elif(angle < lower and down == True and up == True):
            count = count +1
            reset = True
            up = False
        elif(angle > lower+20 and angle < upper-20 and reset):
            down = False
            up = False
            reset = False

    return up, down, reset, count




def convertKeypointsfromCam(filename):
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

        # angle3d = getAngleThreePoint3D(results, 11,13,15)
        angles = calculateAngles(results)
        
        angle, up, down, reset, count = workoutSeleter(angles, filename, up,down,reset,count)     

        cv2.putText(image, 'Angle2D : ' + str(angle), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, ( 255,0,0), 2,cv2.LINE_AA)
        # cv2.putText(image, 'Angle3D : ' + str(angle3d), (10,100), cv2.FONT_HERSHEY_SIMPLEX, 2, ( 255,0,0), 2,cv2.LINE_AA)
        cv2.putText(image, 'Count : ' + str(count), (130,200), cv2.FONT_HERSHEY_SIMPLEX, 1 + 0.2*count, ( 0+10*count if 0+10*count<255 else 255,0,0), 2,cv2.LINE_AA)

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

