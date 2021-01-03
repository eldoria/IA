import matplotlib.pyplot as plt
import tensorflow.keras as keras
import numpy as np
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator


dataGen_training = ImageDataGenerator(
    rotation_range=5,
    rescale=1. / 255,
    shear_range=15,
    channel_shift_range=90.0,
)

dataGen_testing = ImageDataGenerator(
  rescale=1. / 255,
)


def print_image(x, y):
    figure = plt.figure()
    i = 0

    for x_batch, y_batch in dataGen_training.flow(x, y):
        a = figure.add_subplot(4, 4, i + 1)
        plt.imshow(np.squeeze(x_batch), cmap=plt.get_cmap('gray'))
        a.axis('off')
        if i == 15:
            break
        i += 1
    figure.set_size_inches(np.array(figure.get_size_inches()) * 3)
    plt.show()


if __name__ == "__main__":
    nb_iterations = 5
    
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    for i in range(nb_iterations):
        rd = random.randint(0, 60000 - 1)

        image = x_train[rd]

        x = np.expand_dims(image, (0, 3))
        y = np.asarray(['any-label'])

        plt.imshow(image, cmap=plt.get_cmap('gray'))

        print_image(x, y)
