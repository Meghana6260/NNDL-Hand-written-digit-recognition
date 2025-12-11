import tensorflow as tf
from tensorflow.keras import models

# Load MNIST test dataset
(_, _), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Preprocess
test_images = test_images.reshape((-1,28,28,1)).astype('float32') / 255.0

# Load model
model = models.load_model("tf-cnn-model.h5", compile=False)

# IMPORTANT: compile before evaluating
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Evaluate accuracy
loss, accuracy = model.evaluate(test_images, test_labels, verbose=0)
print("Model Test Accuracy: {:.2f}%".format(accuracy * 100))
