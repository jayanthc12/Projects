import pickle

import cv2
import mediapipe as mp #mediapipe for hand landmark detection
import numpy as np #numpy for numerical operations

#loads a pre-trained machine learning model from a file named 'model.p' using pickle.load()
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0) #initializes the video capture from the default camera (index 0)


# hand landmark detection from mediapipe library
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'L'} # 'labels_dict' is created to map the model's output class indices (0, 1, 2) to their corresponding labels ('A', 'B', 'L')

while True:

    data_aux = [] #empty lists data_aux, x_, and y_ to store hand landmark data
    x_ = []
    y_ = []
    # then captures a frame from the camera using the cap.read() function
    ret, frame = cap.read()

    H, W, _ = frame.shape #retrieves the height (H) and width (W) of the frame using the shape attribute of the frame variable
                          # third value -underscore represents the number of channels (but not used here)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #converts the frame from BGR to RGB color space using cv2.cvtColor()
    # hands.process() method expects RGB images as input
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks: #checks if hand landmarks were detected in the frame
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,  #draws the detected hand landmarks on the original frame using mp_drawing.draw_landmarks()
                mp_drawing_styles.get_default_hand_landmarks_style(), #visualizes the hand landmarks and hand connections on the frame
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)
     #processes the detected hand landmarks to calculate the normalized (x, y) coordinates with respect to the minimum (x, y) values.
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
       #calculate the bounding box coordinates (x1, y1) and (x2, y2) for the detected hand landmarks
       #which will be used to draw a rectangle around the hand in the frame.
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])
        #pre-trained machine learning model to predict the class label based on the processed hand landmarks
        predicted_character = labels_dict[int(prediction[0])]

        # draw a rectangle around the hand and display the predicted character label on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()