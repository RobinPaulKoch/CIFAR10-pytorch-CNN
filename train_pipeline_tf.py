from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
import math
from tensorflow import keras
import keras.layers as layers
from tensorflow.keras.models import Sequential


def display_images(images, cols=3, max_images=15):

    # if images not ndarray:
    #     print("This is no np.ndarray! Please submit a np.ndarray to this function")
    #     return

    if np.size(images, 0) > max_images:
        print(f"Showing {max_images} images of {np.size(images, 0)}:")
        images=images[0:max_images]

    fig = plt.figure()
    length = np.size(images, 0)
    rows = math.ceil(length / 3)
    for i, img in enumerate(images):
        ax = fig.add_subplot(rows, cols, i + 1)
        imgplot = plt.imshow(img, interpolation="bicubic")


#Download the dataset
X_train, X_test = tf.keras.datasets.cifar10.load_data()

#Show some of the images
display_images(X_train[0][0:20])


#Define model
model = Sequential(
    layers.Conv2d(16, 3)
)
