import matplotlib.pyplot as plt
import tensorflow as tf

# Load MNIST
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Display first 10 images
plt.figure(figsize=(50,2))
for i in range(50):
    plt.subplot(1,50,i+1)
    plt.imshow(train_images[i], cmap='gray')
    plt.axis('off')
    plt.title(train_labels[i])
plt.show()
print("Train images:", train_images.shape)
print("Train labels:", train_labels.shape)
print("Test images:", test_images.shape)
