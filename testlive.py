import torch
from cnn import CNN
import pandas as pd
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
import matplotlib.pyplot as plt

class_to_letter = ['B', 'C', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'W', 'nothing']

# Load the pre-trained model
loaded_model = torch.load('./model_KernelTestBetter')
loaded_model.to('cuda')

def transform(image):
    if image is None or image.shape[0] == 0:
        return None

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # RGB to Gray scale

    sharpening_kernel = np.array([
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2]
    ])

    sharpened_image = cv2.filter2D(image_gray, -1, sharpening_kernel) # applying sharpening kernal
    image_resized = cv2.resize(sharpened_image, (50, 50), interpolation=cv2.INTER_AREA) # resizing the images to 50x50

    return image_resized


transformed = transforms.Compose([
    # transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])


# Function to make a prediction on a single image
def predict_image(image):
    if image is None:
        return "No image provided"



    # Transform the image
    image = transform(image)
    if image is None:
        return "Failed to transform image"

    image = Image.fromarray(image)
    image = transformed(image)
    
    # print_image = image.squeeze(0)

    # plt.figure(figsize=(4, 4))
    # plt.imshow(print_image, cmap='gray')
    # plt.title('Captured Hand Image')
    # plt.show(block=False)
    # plt.pause(2)
    # plt.close()

    # Make a prediction on the single image
    output = loaded_model(image.unsqueeze(0).to('cuda'))
    _, predicted_class = torch.max(output, 1)

    # Get the predicted class
    predicted_class_index = predicted_class.item()
    predicted_letter = class_to_letter[predicted_class_index]

    return predicted_letter

# Initialize the MediaPipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Initialize the MediaPipe pose detector
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

# Initialize the MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # Use MediaPipe to detect the hand and pose in the frame
    results_hands = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Draw the hand and pose landmarks on the frame
    # if results_hands.multi_hand_landmarks:
        # for hand_landmarks in results_hands.multi_hand_landmarks:
        #     mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if results_hands.multi_hand_landmarks:
        # Get the hand landmark coordinates
        hand_landmarks = results_hands.multi_hand_landmarks[0]

        # Find the minimum and maximum x and y coordinates
        x_coords = [landmark.x for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y for landmark in hand_landmarks.landmark]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Calculate the region of the hand
        x_min = int(x_min * frame.shape[1] - 15)
        y_min = int(y_min * frame.shape[0] - 15)
        x_max = int(x_max * frame.shape[1] + 15)
        y_max = int(y_max * frame.shape[0] + 15)

        hand_region = frame[y_min:y_max, x_min:x_max]

        if cv2.waitKey(1) & 0xFF == ord('s'):
            predicted_letter = predict_image(hand_region)
            if(predicted_letter == 'L'):
                predicted_letter = 'ל'
                print(f"Predicted letter: {predicted_letter}")
            elif (predicted_letter == 'W'):
                predicted_letter = 'ש'
                print(f"Predicted letter: {predicted_letter}")
            elif(predicted_letter == 'F'):
                predicted_letter = 'ט'
                print(f"Predicted letter: {predicted_letter}")
               
            
        

    # Display the frame
    cv2.imshow('Real-time ASL Prediction', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()