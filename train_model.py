import json
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define dataset path
dataset_path = "./asl_alphabet_train/asl_alphabet_train"  # Update with correct path

# Image data generator for preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    validation_split=0.2  # 20% validation split
)

# Load training and validation data
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Build CNN Model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(64, 64, 3)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(len(train_data.class_indices), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_data, validation_data=val_data, epochs=10, batch_size=32)

# Save the trained model
model.save("asl_gesture_model.keras")
# Save correct class indices
class_indices = train_data.class_indices

# Print to verify correctness before saving
print("Class Indices:", class_indices)

with open("class_indices.json", "w") as f:
    json.dump(class_indices, f)

print("Model training complete. Saved as asl_gesture_model.keras")
