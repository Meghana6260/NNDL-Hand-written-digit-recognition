
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# Load MNIST
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Preprocess
train_images = train_images.reshape((-1,28,28,1)).astype('float32') / 255.0
test_images  = test_images.reshape((-1,28,28,1)).astype('float32') / 255.0

#  small data augmentation 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, height_shift_range=0.08)
datagen.fit(train_images)

# Build model
model = models.Sequential([
    layers.Input(shape=(28,28,1)),
    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)), #image size 
    layers.Dropout(0.25),  #of 

    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(10, activation='softmax')    #softmax for multi-class
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',   
              metrics=['accuracy'])

model.summary()

# Train
batch_size = 128
epochs = 12
history = model.fit(datagen.flow(train_images, train_labels, batch_size=batch_size),
                    steps_per_epoch = train_images.shape[0] // batch_size,
                    epochs = epochs,
                    validation_data = (test_images, test_labels),
                    verbose=2)


model.save("tf-cnn-model.h5")
