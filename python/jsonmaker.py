import cv2
import mediapipe as mp
import json
from collections import OrderedDict
import time
from keypointinfo import POSELANDMARKS, TARGETLANDMARKS
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