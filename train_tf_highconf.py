#!/usr/bin/env python3
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import shutil

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Dataset paths
bio_path = '/home/rtlguy/Downloads/TEST/B/'
non_bio_path = '/home/rtlguy/Downloads/TEST/N/'

# Verify datasets exist
if not os.path.exists(bio_path) or not os.path.exists(non_bio_path):
    print(" Dataset paths not found!")
    exit(1)

print(f" Biodegradable: {len(os.listdir(bio_path))} images")
print(f" Non-biodegradable: {len(os.listdir(non_bio_path))} images")

# Create structured dataset
temp_data_dir = '/tmp/waste_classification'
train_dir = os.path.join(temp_data_dir, 'train')
os.makedirs(os.path.join(train_dir, 'biodegradable'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'non_biodegradable'), exist_ok=True)

# Copy and organize images
print("Organizing dataset...")
for file in os.listdir(bio_path):
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        src = os.path.join(bio_path, file)
        dst = os.path.join(train_dir, 'biodegradable', file)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

for file in os.listdir(non_bio_path):
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        src = os.path.join(non_bio_path, file)
        dst = os.path.join(train_dir, 'non_biodegradable', file)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

# Enhanced parameters for better confidence
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 16  
EPOCHS = 50 

# Enhanced data augmentation for more diverse training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,        
    width_shift_range=0.3,    
    height_shift_range=0.3,
    shear_range=0.2,          
    zoom_range=0.2,          
    horizontal_flip=True,
    brightness_range=[0.8, 1.2], 
    fill_mode='nearest',
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    seed=42
)

validation_generator = val_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    seed=42
)

print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")


model = Sequential([
    # First block
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    BatchNormalization(),  # Added batch normalization
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.25),  # Added dropout early
    
    # Second block
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    # Third block
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    # Fourth block
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    # Dense layers
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),  # Added extra dense layer
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Enhanced compilation with learning rate scheduling
initial_learning_rate = 0.001
model.compile(
    optimizer=Adam(learning_rate=initial_learning_rate),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks for better training
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

# Train the model
print("Starting enhanced training...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# Save model
model.save('enhanced_waste_classifier.h5')
print("Enhanced model saved!")


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 3)
lr_history = []
for i in range(len(history.history['loss'])):
    lr_history.append(initial_learning_rate * (0.5 ** (i // 5)))
plt.plot(lr_history, label='Learning Rate')
plt.title('Learning Rate Schedule')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.yscale('log')
plt.legend()

plt.tight_layout()
plt.savefig('enhanced_training_history.png', dpi=300, bbox_inches='tight')
plt.show()

print("🎉 Enhanced training completed!")
print(f"Final training accuracy: {max(history.history['accuracy']):.4f}")
print(f"Final validation accuracy: {max(history.history['val_accuracy']):.4f}")

# Clean up
shutil.rmtree(temp_data_dir, ignore_errors=True)

