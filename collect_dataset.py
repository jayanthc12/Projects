import os  #The os library allows the script to interact with the operating system, enabling it to create directories for storing the collected data.

import cv2  #The cv2 library, which is part of OpenCV, is used for computer vision tasks, such as reading frames from a video source, displaying them, and saving them as images.


DATA_DIR = './data'

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
#The DATA_DIR variable is assigned the value './data', which is a relative path to a directory named 'data' in the current working directory.
# it is created if it doesn't already exist. The os.path.exists(DATA_DIR) function checks if the 'data' directory already exists,
# and if not, os.makedirs(DATA_DIR) is used to create it

#two variables are defined: number_of_classes and dataset_size.
number_of_classes = 3 # number_of_classes is set to 3, indicating that there are three classes in the dataset.
dataset_size = 100 # dataset_size is set to 100, meaning that 100 images will be collected for each class.


cap = cv2.VideoCapture(0)  # VideoCapture object is responsible for capturing video frames from a video source. The 0 in cv2.VideoCapture(0) indicates the camera index (if only one camera is present)
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))): #checks if the corresponding directory exists
        os.makedirs(os.path.join(DATA_DIR, str(j)))  # If the directory does not exist, it creates it using os.makedirs()

    print('Collecting data for class {}'.format(j)) # prints a message indicating which class's data is being collected

    done = False
    while True: #captures the video frames in real-time and displays them in a window. The loop waits until the user presses the 'Q' key to start capturing images for the current class.
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()        # captures and saves dataset_size number of images for the current class.
        cv2.imshow('frame', frame)     ## It enters a loop that captures images until the counter variable reaches dataset_size.
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()
