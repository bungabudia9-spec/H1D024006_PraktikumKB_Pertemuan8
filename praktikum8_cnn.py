import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Path ke dataset (sesuaikan dengan lokasi Anda)
dataset_path = "./rps-cv-images"
# Image Data Generator untuk training dan validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Generator untuk data training
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training',
)

# Generator untuk data validation
validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
)

# Membangun model CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(3, activation='softmax')
])

# Menampilkan summary model
print("\n=== Model Summary ===")
model.summary()

# Kompilasi model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Melatih model
print("\n=== Training Model ===")
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)

# Evaluasi model
print("\n=== Evaluasi Model ===")
val_loss, val_acc = model.evaluate(validation_generator)
print(f'Validation loss: {val_loss}, Validation accuracy: {val_acc}')

# Visualisasi training history
print("\n=== Visualisasi Training History ===")
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# Prediksi pada data validation
print("\n=== Prediksi ===")
predictions = model.predict(validation_generator)
print("Sample predictions (probabilitas):")
print(predictions[:5])

print("\n=== Selesai ===")