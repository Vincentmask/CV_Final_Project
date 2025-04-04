import time

import cv2
import keyboard  # Import keyboard module for direct key events
import mediapipe as mp
import numpy as np
import pyautogui

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
cap = cv2.VideoCapture(0)

# Screen size for cursor mapping
screen_w, screen_h = pyautogui.size()

# State tracking
holding_unit = False
gesture_cooldown = 1
last_gesture_time = time.time()

# Cursor smoothing parameters
prev_x, prev_y = screen_w // 2, screen_h // 2
SMOOTHING = 0.2

# Process every N frames for gesture classification
gesture_frame_interval = 3
frame_count = 0


def classify_hand_gesture(landmarks):
    """Classifies gesture based on hand landmarks using DIP joints."""
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    index_dip = landmarks[7]
    middle_dip = landmarks[11]
    ring_dip = landmarks[15]
    pinky_dip = landmarks[19]

    # Measure distances for different pinch gestures
    pinch_distance_index = np.linalg.norm(
        np.array([thumb_tip.x, thumb_tip.y]) -
        np.array([index_tip.x, index_tip.y])
    )
    pinch_distance_middle = np.linalg.norm(
        np.array([thumb_tip.x, thumb_tip.y]) -
        np.array([middle_tip.x, middle_tip.y])
    )

    pinch_threshold = 0.04

    # Check if fingers are extended
    index_up = index_tip.y < index_dip.y
    middle_up = middle_tip.y < middle_dip.y
    ring_up = ring_tip.y < ring_dip.y
    pinky_up = pinky_tip.y < pinky_dip.y

    if index_up and middle_up and not ring_up and not pinky_up:
        return "Victory"  # Refresh Shop

    if pinky_up and not index_up and not middle_up and not ring_up:
        return "Pinky Up"  # Buy XP

    if pinch_distance_index < pinch_threshold:
        return "Left Click"  # Drag & Drop (Left Mouse Button)

    if pinch_distance_middle < pinch_threshold:
        return "Right Click"  # Right Mouse Button

    if pinch_distance_index >= pinch_threshold and pinch_distance_middle >= pinch_threshold:
        return "Release"  # Drop Champion (Release Click)

    return "Unknown"


def move_cursor(x, y, screen_width, screen_height):
    """Smooth cursor movement based on hand position."""
    global prev_x, prev_y
    screen_x = int(x * screen_width)
    screen_y = int(y * screen_height)

    # Apply smoothing
    SMOOTHING = 0.2
    new_x = prev_x + (screen_x - prev_x) * SMOOTHING
    new_y = prev_y + (screen_y - prev_y) * SMOOTHING

    pyautogui.moveTo(int(new_x), int(new_y),  _pause=False)

    # Update previous position
    prev_x, prev_y = new_x, new_y


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    frame_count += 1

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            move_cursor(
                hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y, screen_w, screen_h)

            if frame_count % gesture_frame_interval == 0:
                detected_gesture = classify_hand_gesture(
                    hand_landmarks.landmark)

                if detected_gesture == "Left Click" and not holding_unit:
                    pyautogui.mouseDown()
                    holding_unit = True

                if detected_gesture == "Right Click":
                    pyautogui.rightClick()  # Right-click action

                if detected_gesture == "Release" and holding_unit:
                    pyautogui.mouseUp()
                    holding_unit = False

                if time.time() - last_gesture_time > gesture_cooldown:
                    if detected_gesture == "Victory":
                        keyboard.press_and_release("d")  # Refresh Shop

                        last_gesture_time = time.time()
                    elif detected_gesture == "Pinky Up":
                        keyboard.press_and_release("f")  # Buy XP
                        last_gesture_time = time.time()

                cv2.putText(frame, f"Gesture: {detected_gesture}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("TFT Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
