import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import axes
import time
import mediapipe as mp
from PIL import Image
import math

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities


def mediapipe_detection(image, model):
    # print (image.shape)
    if (np.size(image) > 0):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False                  # Image is no longer writeable
        results = model.process(image)                 # Make prediction
        image.flags.writeable = True                   # Image is now writeable
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
        return image, results
    else:
        return image, model
    

def draw_landmarks(image, results):
    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections


def draw_styled_landmarks(image, results):
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=1, circle_radius=3),
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=1, circle_radius=3))
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=1, circle_radius=3),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=1, circle_radius=3))
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=3),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=1, circle_radius=3))
    

def resize_and_show(image):
    DESIRED_WIDTH = 300
    DESIRED_HEIGHT = 300
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
    # cv2.imshow("TEST", img)


def extract_each_frame(pathWorkspace="", filePath=""):
    cap = cv2.VideoCapture(os.path.join(pathWorkspace, filePath))
    index = 0
    with mp_holistic.Holistic(min_detection_confidence=0.5, 
                              min_tracking_confidence=0.5) as holistic:
        resultEachFrame = []
        while cap.isOpened():

            # try:
            ret, frame = cap.read()

            # print (type(frame))
            if not ret:
                print("break process_image_extract_keypoint")
                break

            imageZeros = np.zeros([frame.shape[0], 
                                   frame.shape[1], 3], 
                                   dtype=np.uint8)
            imageZeros[:, :] = 255
            # Make detections
            image, tempResults = mediapipe_detection(frame, holistic)
            allTempResult.append(tempResults)
            # print (np.asarray(frame.shape))
            # print (type(tempResults))

            # # Draw landmarks
            draw_styled_landmarks(imageZeros, tempResults)
            resultEachFrame.append([image, tempResults])

            # # Show to screen
            # # cv2.imshow('OpenCV Feed', image)
            print(index)
            # cv2.imshow("OK",frame)
            # cv2.imshow("OK1",imageZeros)

            newDir = os.path.join(pathWorkspace, "visualizeKeypoint")
            if not os.path.isdir(newDir):
                os.makedirs(newDir)

            cv2.imwrite(os.path.join(newDir, "frame{}.png".format(index)), frame)
            cv2.imwrite(os.path.join(newDir, "keypoint{}.png".format(index)), imageZeros)
            # resize_and_show(image)

            index += 1
        resultEachImage.append(resultEachFrame)
        cap.release()
        cv2.destroyAllWindows()


def drawKeypointFromExtractKeypoint(pathWorkspace, 
                                    folder_path_keypoint):
        xTemp = []
        newXTemp = []
        yTemp = []
        DATA_PATH_KEYPOINT = os.path.join(pathWorkspace, folder_path_keypoint)
        actionsVideoInput = np.array(os.listdir(DATA_PATH_KEYPOINT))

        for tempData in actionsVideoInput:
            keypointData = np.load(os.path.join(DATA_PATH_KEYPOINT, tempData))
            namefileSplit = tempData.split(".")
            # print(keypointData[0])
            for ii in range(0, keypointData.shape[0], 2):
                plt.scatter(keypointData[ii], keypointData[ii+1]*-1, color="blue")
            plt.axis('on')
            plt.xlabel("X axis")
            plt.ylabel("Y axis")

            newDir = os.path.join(pathWorkspace, 
                                  "visualizeNormalizeAndNot", 
                                  folder_path_keypoint)
            
            if not os.path.isdir(os.path.join(newDir)):
                os.makedirs(os.path.join(newDir))
    
            plt.savefig(os.path.join(newDir, namefileSplit[0]))
            plt.clf()

if __name__ == '__main__':
    resultEachImage = []
    frameFinal = []
    
    pathWorkspace = '/home/bra1n/Documents/signLanguage/paperNeuralComputing'
    filePath = "DataTraining100/hat/26726.mp4"
    allTempResult = []
    # drawing one videos
    # extract_each_frame(pathWorkspace=pathWorkspace, filePath=filePath)


    label="accident"
    idVideo= "00627"
    pathDataToDraw = "Keypoint{}WLASL100_normalization_option2/{}/{}".format("Training",
                                                                         label,
                                                                         idVideo)
    # drawing 1 folder
    drawKeypointFromExtractKeypoint(pathWorkspace, 
                                    pathDataToDraw)
