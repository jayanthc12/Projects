#The create daatest code aims to read images from subdirectories of the 'data' directory, use the Hands model from mediapipe to extract hand landmarks from those images,
# normalize the coordinates, and save the extracted data along with corresponding labels in a pickle file.

import os
import pickle # pickle- Used for serializing and deserializing Python objects, enabling the script to save and load data to/from a file.

import mediapipe as mp #mediapipe python library to detect face and hand landmarks (for landmark detection)
import cv2
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands #which provides hand tracking functionality
mp_drawing = mp.solutions.drawing_utils #provides drawing utilities to visualize landmarks on images.
mp_drawing_styles = mp.solutions.drawing_styles #contains predefined drawing styles used by mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3) #static_image_mode=True: Specifies that the model will be used for static images (single images) rather than video streams
#min_detection_confidence=0.3: Sets the minimum confidence value required for a hand detection to be considered valid. Detections with confidence lower than 0.3 will be discarded.

DATA_DIR = './data'

data = []   # these lists will be used to store the extracted hand landmarks data and their corresponding labels
labels = []
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
                                # data_aux, x_, and y_ lists are initialized to store temporary data during the processing of each image.
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))   #image is read using cv2.imread, and then it is converted from BGR to RGB using cv2.cvtColor.
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   #this conversion is necessary because the mediapipe library processes images in RGB format

        results = hands.process(img_rgb)
       #hands model processes the RGB image img_rgb to detect hand landmarks.
       # If hand landmarks are detected in the image, the code proceeds to extract the X and Y coordinates of each hand landmark.

        if results.multi_hand_landmarks: # x and y coordinates of each landmark are appended to the corresponding temporary x_ and y_ lists
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))  #x and y coordinates are normalized by subtracting the minimum x and y values found in the respective x_ and y_ lists.
                    data_aux.append(y - min(y_))

           #data_aux list containing the normalized x and y coordinates of hand landmarks is appended to the data list and the corresponding label is appended to the labels list.
            # This process repeats for all images in all subdirectories.

            data.append(data_aux)
            labels.append(dir_)

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
