# Projects
This project aims to build a sample sign language detection system using Python and computer vision techniques. The goal is to enable real-time sign language translation by capturing and interpreting hand gestures through a webcam.

### Sample Data 
This contains the pictures of sample data is is collected with the help of the collect_dataset program.

### collect_dataset.py
This code utilizes the OpenCV library to capture real-time video frames and save them as images, creating a labeled dataset for machine learning tasks.

### create_dataset.py
The create daatest code aims to read images from subdirectories of the 'data' directory, use the Hands model from mediapipe to extract hand landmarks from those images,
normalize the coordinates, and save the extracted data along with corresponding labels in a pickle file.

### test_classifier.py
This py file integrates hand landmark detection using the Mediapipe library and a pre-trained machine learning model to identify and classify hand gestures in real-time video frames, showcasing the labeled predictions within a bounding box on the frame.

### train_classifier.py
It loads data and labels, splits them into training and testing sets and trains a Random Forest classifier to assesses accuracy and finally saves the trained model using pickle for subsequent usage.




