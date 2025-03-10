import json

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from tensorflow import keras
from tensorflow.keras.preprocessing import image

# Load trained model
model = keras.models.load_model("asl_gesture_model.keras")

# Load class indices from JSON file
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Reverse mapping
gesture_map = {v: k for k, v in class_indices.items()}

# Gesture-action mapping for Dino Game
gesture_mapping = {
    "A": lambda: pyautogui.press("space"),  # Jump
    "B": lambda: pyautogui.keyDown("down"),  # Duck (hold down)
    "C": lambda: pyautogui.keyUp("down")  # Stop Ducking
}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)


def predict_gesture(frame):
    """ Convert frame to model input and predict gesture """
    img = cv2.resize(frame, (64, 64))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)
    return predicted_label


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            x_min = min([lm.x for lm in hand_landmarks.landmark]
                        ) * frame.shape[1]
            x_max = max([lm.x for lm in hand_landmarks.landmark]
                        ) * frame.shape[1]
            y_min = min([lm.y for lm in hand_landmarks.landmark]
                        ) * frame.shape[0]
            y_max = max([lm.y for lm in hand_landmarks.landmark]
                        ) * frame.shape[0]

            if int(y_min) < 0 or int(y_max) > frame.shape[0] or int(x_min) < 0 or int(x_max) > frame.shape[1]:
                continue  # Skip invalid frame regions
            hand_crop = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
            if hand_crop.size == 0:
                continue
            predicted_label = predict_gesture(hand_crop)
            predicted_gesture = gesture_map.get(predicted_label, "Unknown")

            if predicted_gesture in gesture_mapping:
                gesture_mapping[predicted_gesture]()

            cv2.putText(frame, predicted_gesture, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("ASL Gesture Dino Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
