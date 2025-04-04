import collections
import time

import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

# Open Webcam
cap = cv2.VideoCapture(0)

# Gesture Control Mapping
gesture_mapping = {
    "Palm Open": lambda: pyautogui.press("space"),  # Jump
    "Fist": lambda: pyautogui.keyUp("down"),  # Step Ducking
    "Victory": lambda: pyautogui.keyDown("down")  # Duck (hold)
}


# Gesture History for Stability
gesture_history = collections.deque(maxlen=5)
stable_gesture = None
gesture_cooldown = 0.5  # 1 second cooldown
last_trigger_time = 0


def classify_hand_gesture(landmarks):
    """Classifies gesture based on hand landmarks using DIP joints."""
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    index_dip = landmarks[7]  # DIP joint of index finger
    middle_dip = landmarks[11]  # DIP joint of middle finger
    ring_dip = landmarks[15]  # DIP joint of ring finger
    pinky_dip = landmarks[19]  # DIP joint of pinky

    # Check if fingers are extended (tip above DIP joint)
    index_up = index_tip.y < index_dip.y
    middle_up = middle_tip.y < middle_dip.y
    ring_up = ring_tip.y < ring_dip.y
    pinky_up = pinky_tip.y < pinky_dip.y

    # Fist: All fingers curled (tip below DIP)
    if not index_up and not middle_up and not ring_up and not pinky_up:
        return "Fist"

    # Palm Open: All fingers extended
    if index_up and middle_up and ring_up and pinky_up:
        return "Palm Open"

    # Victory: Only index and middle fingers extended, ring and pinky curled
    if index_up and middle_up and not ring_up and not pinky_up:
        return "Victory"

    return "Unknown"


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    # Flip and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hands
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks,
                                   mp_hands.HAND_CONNECTIONS)

            # Classify gesture
            detected_gesture = classify_hand_gesture(hand_landmarks.landmark)
            gesture_history.append(detected_gesture)

            # Check if gesture is stable (last 5 frames are the same)
            if len(set(gesture_history)) == 1 and detected_gesture != stable_gesture:
                if time.time() - last_trigger_time > gesture_cooldown:
                    stable_gesture = detected_gesture
                    last_trigger_time = time.time()

                    # Execute action if stable
                    if stable_gesture in gesture_mapping:
                        gesture_mapping[stable_gesture]()

            # Display detected gesture
            cv2.putText(frame, f"{stable_gesture}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the webcam feed
    cv2.imshow("MediaPipe Dino Game Control", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
