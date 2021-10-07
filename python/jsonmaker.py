import cv2
import mediapipe as mp
import json
from collections import OrderedDict
import time
from keypointinfo import POSELANDMARKS, TARGETLANDMARKS
from camconverter import calculateAngle 
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

result = OrderedDict()

def convertVideotoJson(filename):
    nowWorkoutmp4 = filename + '.mp4'
    nowWorkoutjson = filename + '.json'
    vidcap = cv2.VideoCapture('./workout/' + nowWorkoutmp4)
    start = time.time()
    idx = 0
    printCheck = True

    up = False
    down = False
    reset = False
    count = 0

    totalDict = OrderedDict()
    totalDict["workout"] = nowWorkoutmp4
    totalDict["posedata"] = []
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        while vidcap.isOpened():
            success, image = vidcap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                print('time : ' + str(time.time()-start))
                break
    
            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            #image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = pose.process(image)

            if not results.pose_landmarks:
                continue
            
            shoulder1 = [results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x,
                    results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y,
                    results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].z]
            elbow1 = [results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].x, 
                    results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].y,
                    results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].z]
            wrist1 = [results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].x, 
                    results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].y,
                    results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].z]
            
            shoulder = [results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x,
                    results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y]
            elbow = [results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].x, 
                    results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].y]
            wrist = [results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].x, 
                    results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].y]

            
            angle = calculateAngle(shoulder, elbow, wrist)
            angle3d = calculateAngle(shoulder1, elbow1, wrist1)

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
            
            cv2.putText(image, 'Angle2D : ' + str(angle), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, ( 255,0,0), 2,cv2.LINE_AA)
            cv2.putText(image, 'Angle3D : ' + str(180- angle3d), (10,100), cv2.FONT_HERSHEY_SIMPLEX, 2, ( 255,0,0), 2,cv2.LINE_AA)
            cv2.putText(image, 'Count : ' + str(count), (130,200), cv2.FONT_HERSHEY_SIMPLEX, 1 + 0.2*count, ( 0+10*count if 0+10*count<255 else 255,0,0), 2,cv2.LINE_AA)

            if printCheck :
                print(totalDict["workout"], image.shape)
                printCheck = False
            frame = OrderedDict()
            frame["frame"] = idx
            frame["landmarks"] = []
            lidx =0
            for i in results.pose_landmarks.landmark:
                landmark = OrderedDict()
                if not POSELANDMARKS[lidx] in TARGETLANDMARKS:
                    lidx = lidx +1
                    continue
                landmark["part"]=POSELANDMARKS[lidx]
                landmark["x"]=i.x
                landmark["y"]=i.y
                landmark["z"]=i.z
                landmark["visibility"] = i.visibility
                lidx = lidx +1
                frame["landmarks"].append(landmark)

            totalDict["posedata"].append(frame)
            #print(frame)
            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            #print(results.pose_landmarks.landmark)
            idx = idx+1
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imshow('MediaPipe Pose', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    vidcap.release()
    #cap.release()
    cv2.destroyAllWindows()

    file_path = './json/' + nowWorkoutjson
    with open(file_path, 'w', encoding='utf-8') as outfile:
        json.dump(totalDict,outfile,ensure_ascii=False,indent="\t")

    print("finish!")
    #print(json.dumps(totalDict, ensure_ascii=False, indent=4))

if __name__ == "__main__":
    print('video json converter main start')