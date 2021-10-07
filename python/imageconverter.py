import cv2
import math
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

DESIRED_HEIGHT = 2944
DESIRED_WIDTH = 1324

def resize_and_show(image):
  h, w = image.shape[:2]
  if h < w:
    img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
  else:
    img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
  cv2.imshow("image",img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

# For static images:
def imageConvert(imgFile):
    IMAGE_FILES = [imgFile +'.jpeg']

    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.5)
    
    for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(file)
        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
        if not results.pose_landmarks:
            continue

        print(
            f'Nose coordinates: ('
            f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
            f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height})'
        )
        # Draw pose landmarks on the image.
        annotated_image = image.copy()
        #print(results.pose_landmarks)
        idx = 0
        for i in results.pose_landmarks.landmark:
            #print('part : ' + POSELANDMARKS[idx])
            #print(i)
            idx = idx +1
        mp_drawing.draw_landmarks(
            annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imwrite('./annotated_image' + str(idx) + '.png', annotated_image)

        resize_and_show(annotated_image)

        # Plot pose world landmarks.
        #mp_drawing.plot_landmarks(
        #    results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

if __name__ == "__main__":
    print('image main')