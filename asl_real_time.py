import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
import time

model_path = "C:/Users/adith/OneDrive/Desktop/ASL/my_model.h5"
try:
    model = load_model(model_path, compile=False)
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading model:", e)
    exit()

class_labels = {i: chr(65 + i) for i in range(26)}  # A-Z mapping

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)  # Detect one hand at a time
imgSize = 400  # Match training image size

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam activated! Press 'q' to exit.")

# Sentence formation variables
sentence = ""
current_word = ""
last_predicted_letter = None
last_detection_time = time.time()
cooldown_period = 2.5  # Increased cooldown time
confirmation_threshold = 7  # Require 7 consistent detections
confirmation_counter = 0

frame_width = 1200
frame_height = 800
cv2.namedWindow("ASL Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ASL Detection", frame_width, frame_height)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    frame = cv2.flip(frame, 1)  
    hands, _ = detector.findHands(frame, draw=False) 
    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
    predicted_letter = ""

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]

        if len(lmList) == 0:
            continue

        x_min, y_min = np.min(lmList, axis=0)[:2]
        x_max, y_max = np.max(lmList, axis=0)[:2]

        w, h = x_max - x_min, y_max - y_min
        scale = (imgSize * 0.8) / max(w, h)
        lmList = np.array(lmList)[:, :2]
        lmList = (lmList - [x_min, y_min]) * scale
        lmList += (imgSize - np.array([w, h]) * scale) / 2

        # Draw landmarks on white background
        for x, y in lmList.astype(int):
            cv2.circle(imgWhite, (x, y), 8, (0, 0, 0), -1)

        # Connect landmarks
        connections = [[0, 1], [1, 2], [2, 3], [3, 4],
                       [0, 5], [5, 6], [6, 7], [7, 8],
                       [0, 9], [9, 10], [10, 11], [11, 12],
                       [0, 13], [13, 14], [14, 15], [15, 16],
                       [0, 17], [17, 18], [18, 19], [19, 20]]

        for p1, p2 in connections:
            cv2.line(imgWhite, tuple(lmList[p1].astype(int)), tuple(lmList[p2].astype(int)), (0, 0, 0), 4)

        imgWhite_gray = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2GRAY)
        img_input = cv2.resize(imgWhite_gray, (128, 128))
        img_input = img_input / 255.0
        img_input = np.expand_dims(img_input, axis=-1)
        img_input = np.expand_dims(img_input, axis=0)

        try:
            prediction = model.predict(img_input)
            predicted_label = np.argmax(prediction)
            predicted_letter = class_labels.get(predicted_label, "")

            if predicted_letter == last_predicted_letter:
                confirmation_counter += 1
            else:
                confirmation_counter = 0  # Reset if new letter detected

            if confirmation_counter >= confirmation_threshold:
                current_time = time.time()
                if current_time - last_detection_time > cooldown_period:
                    current_word += predicted_letter
                    last_detection_time = current_time
                    print(f"Confirmed Letter: {predicted_letter}")
                confirmation_counter = 0

            last_predicted_letter = predicted_letter

        except Exception as e:
            print(f"Error during prediction: {e}")
            break

        # Show the processed white image (top-right corner)
        imgWhite_resized = cv2.resize(imgWhite, (200, 200))
        cv2.imshow("Landmarks", imgWhite_resized)

    # Sentence formation logic
    key = cv2.waitKey(1) & 0xFF

    if key == 32:  # Spacebar → End word
        if current_word:
            sentence += current_word + " "
            current_word = ""

    elif key == ord('.'):  # Period → End sentence
        if current_word:
            sentence += current_word
        sentence = sentence.strip() + "."
        print(f"Final Sentence: {sentence}")
        current_word = ""
    
    elif key == 8:  # Backspace → Remove last letter
        if current_word:
            current_word = current_word[:-1]

    # Improved Sentence Display Overlay
    height, width, _ = frame.shape
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, height - 120), (width, height), (0, 0, 0), -1)  # Larger black background for text
    alpha = 0.6  # Transparency factor
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)  # Apply transparency

    cv2.putText(frame, f"Sentence: {sentence}", (50, height - 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(frame, f"Word: {current_word}", (50, height - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the main frame with text overlay
    cv2.imshow("ASL Detection", frame)

    # Exit when 'q' is pressed
    if key == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Webcam closed.")
