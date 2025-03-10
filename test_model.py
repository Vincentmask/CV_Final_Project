import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image

# Load trained model
model = keras.models.load_model("asl_gesture_model.keras")

# Path to test images
test_dataset_path = "asl_alphabet_test/asl_alphabet_test"  # Make sure this is correct

# Get all test images
test_images = [f for f in os.listdir(test_dataset_path) if f.endswith(".jpg")]

# Manually define labels based on filenames
gesture_map = {
    "A_test.jpg": "A", "B_test.jpg": "B", "C_test.jpg": "C",
    "D_test.jpg": "D", "E_test.jpg": "E", "F_test.jpg": "F",
    "G_test.jpg": "G", "H_test.jpg": "H", "I_test.jpg": "I",
    "J_test.jpg": "J", "K_test.jpg": "K", "L_test.jpg": "L",
    "M_test.jpg": "M", "N_test.jpg": "N", "O_test.jpg": "O",
    "P_test.jpg": "P", "Q_test.jpg": "Q", "R_test.jpg": "R",
    "S_test.jpg": "S", "T_test.jpg": "T", "U_test.jpg": "U",
    "V_test.jpg": "V", "W_test.jpg": "W", "X_test.jpg": "X",
    "Y_test.jpg": "Y", "Z_test.jpg": "Z", "space_test.jpg": "space",
    "nothing_test.jpg": "nothing"
}

correct_predictions = 0
total_predictions = len(test_images)

for img_name in test_images:
    img_path = os.path.join(test_dataset_path, img_name)
    img = image.load_img(img_path, target_size=(64, 64)
                         )  # Resize to model input
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict gesture
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)

    # Reverse map the prediction index to actual label
    label_map = {v: k for k, v in gesture_map.items()}  # Reverse mapping
    predicted_gesture = label_map.get(predicted_label, "Unknown")

    print(
        f"🖼️ Image: {img_name} | 🎯 True: {gesture_map[img_name]} | 🤖 Predicted: {predicted_gesture}")

    if predicted_gesture == gesture_map[img_name]:
        correct_predictions += 1

# Print final accuracy
accuracy = (correct_predictions / total_predictions) * 100
print(f"\n✅ Test Accuracy: {accuracy:.2f}%")
