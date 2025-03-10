import collections
import time

import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# ✅ Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

# ✅ Open Webcam
cap = cv2.VideoCapture(0)

# ✅ Gesture Control Mapping
gesture_mapping = {
    "Fist": lambda: pyautogui.press("space"),  # Jump
    "Palm Open": lambda: pyautogui.keyDown("down"),  # Duck (hold)
    "Neutral": lambda: pyautogui.keyUp("down")  # Stop Ducking
}

# ✅ Gesture History for Stability
gesture_history = collections.deque(maxlen=5)
stable_gesture = None
gesture_cooldown = 1.0  # 1 second cooldown
last_trigger_time = 0


def classify_hand_gesture(landmarks):
    """Classifies gesture based on hand landmarks using MediaPipe coordinates."""
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    palm_base = landmarks[0]

    # Measure distances between fingers and palm base
    thumb_index_dist = np.linalg.norm(
        np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y]))
    index_palm_dist = np.linalg.norm(
        np.array([index_tip.x, index_tip.y]) - np.array([palm_base.x, palm_base.y]))
    middle_palm_dist = np.linalg.norm(
        np.array([middle_tip.x, middle_tip.y]) - np.array([palm_base.x, palm_base.y]))
    ring_palm_dist = np.linalg.norm(
        np.array([ring_tip.x, ring_tip.y]) - np.array([palm_base.x, palm_base.y]))
    pinky_palm_dist = np.linalg.norm(
        np.array([pinky_tip.x, pinky_tip.y]) - np.array([palm_base.x, palm_base.y]))

    # Define palm size for reference
    palm_size = np.linalg.norm(
        np.array([palm_base.x, palm_base.y]) - np.array([middle_tip.x, middle_tip.y]))

    # **Fist**: All fingers curled near palm
    if (index_palm_dist < 0.2 * palm_size and
        middle_palm_dist < 0.2 * palm_size and
        ring_palm_dist < 0.2 * palm_size and
            pinky_palm_dist < 0.2 * palm_size):
        return "Fist"

    # **Palm Open**: All fingers extended significantly above palm
    elif (index_tip.y < palm_base.y and
          middle_tip.y < palm_base.y and
          ring_tip.y < palm_base.y and
          pinky_tip.y < palm_base.y and
          thumb_tip.y < palm_base.y):
        return "Palm Open"

    # **Thumbs Up**: Thumb extended, other fingers curled
    elif (thumb_tip.y < palm_base.y and
          index_palm_dist < 0.2 * palm_size and
          middle_palm_dist < 0.2 * palm_size and
          ring_palm_dist < 0.2 * palm_size and
          pinky_palm_dist < 0.2 * palm_size):
        return "Thumbs Up"

    return "Unknown"


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    # ✅ Flip and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ✅ Process hands
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks,
                                   mp_hands.HAND_CONNECTIONS)

            # ✅ Classify gesture
            detected_gesture = classify_hand_gesture(hand_landmarks.landmark)
            gesture_history.append(detected_gesture)

            # ✅ Check if gesture is stable (last 5 frames are the same)
            if len(set(gesture_history)) == 1 and detected_gesture != stable_gesture:
                if time.time() - last_trigger_time > gesture_cooldown:
                    stable_gesture = detected_gesture
                    last_trigger_time = time.time()

                    # ✅ Execute action if stable
                    if stable_gesture in gesture_mapping:
                        gesture_mapping[stable_gesture]()

            # ✅ Display detected gesture
            cv2.putText(frame, f"{stable_gesture}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # ✅ Show the webcam feed
    cv2.imshow("MediaPipe Dino Game Control", frame)

    # ✅ Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ Cleanup
cap.release()
cv2.destroyAllWindows()
