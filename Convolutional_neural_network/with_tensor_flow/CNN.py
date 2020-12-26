import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, Reshape, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

dataGen_training = ImageDataGenerator(
  horizontal_flip=True,
  vertical_flip=False,
  rescale=1. / 255,
  brightness_range=(0.1, 0.9),
  channel_shift_range=15.0,
)

dataGen_testing = ImageDataGenerator(
  rescale=1. / 255,
)


def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * tf.math.exp(-0.07)


def plot_log(all_logs):
    for logs in all_logs:
        losses = logs.history['loss']
        name = logs.history['name']
        plt.plot(list(range(len(losses))), losses, label=name)
    plt.xlabel("number of epochs")
    plt.ylabel("error")
    plt.title("error on training data")
    plt.legend()
    plt.show()

    for logs in all_logs:
        losses = logs.history['val_loss']
        name = logs.history['name']
        plt.plot(list(range(len(losses))), losses, label=name)
    plt.xlabel("number of epochs")
    plt.ylabel("error")
    plt.title("error on testing data")
    plt.legend()
    plt.show()

    for logs in all_logs:
        metric = logs.history['categorical_accuracy']
        name = logs.history['name'] + " - training"
        plt.plot(list(range(len(metric))), metric, label=name)
        metric = logs.history['val_categorical_accuracy']
        name = logs.history['name'] + " - testing"
        plt.plot(list(range(len(metric))), metric, label=name)
    plt.xlabel("number of epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.title("prediction accuracy on training/testing")
    plt.show()


def linear_model(x, y, val_x, val_y, opt, loss_func, epochs, batch_size):
    model = keras.Sequential([
        # convert a two dimensional matrix into a vector
        Flatten(),
        Dense(10, activation=keras.activations.softmax),
    ])

    model.compile(optimizer=opt, loss=loss_func, metrics=keras.metrics.categorical_accuracy)

    logs = model.fit(x, y, validation_data=(val_x, val_y), epochs=epochs, batch_size=batch_size,
                     callbacks=[keras.callbacks.LearningRateScheduler(scheduler),
                                keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)])
    model.summary()

    return logs


def multi_layer_perceptron(x, y, val_x, val_y, opt, loss_func, epochs, batch_size, activation, dropout):

    model = keras.Sequential([
        # convert a two dimensional matrix into a vector
        Flatten(),
        Dense(120, activation=activation),
        Dropout(dropout),
        Dense(120, activation=activation),
        Dropout(dropout),
        Dense(10, activation=keras.activations.softmax),
    ])

    model.compile(optimizer=opt, loss=loss_func, metrics=keras.metrics.categorical_accuracy)

    logs = model.fit(x, y, validation_data=(val_x, val_y), epochs=epochs, batch_size=batch_size,
                     callbacks=[keras.callbacks.LearningRateScheduler(scheduler),
                                keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)])
    model.summary()

    return logs


def convolutional_neural_network(activation):
    model = keras.Sequential([
        Reshape((28, 28, 1)),
        BatchNormalization(),

        Conv2D(196, (3, 3), padding="same", activation=activation),
        BatchNormalization(),
        Conv2D(196, (3, 3), padding="same", activation=activation),
        BatchNormalization(),
        MaxPool2D(),

        Conv2D(92, (3, 3), padding="same", activation=activation),
        BatchNormalization(),
        Conv2D(92, (3, 3), padding="same", activation=activation),
        BatchNormalization(),
        MaxPool2D(),

        Conv2D(48, (3, 3), padding="same", activation=activation),
        BatchNormalization(),
        Conv2D(48, (3, 3), padding="same", activation=activation),
        BatchNormalization(),

        keras.layers.Flatten(),

        keras.layers.Dense(30, activation=activation),

        BatchNormalization(),

        keras.layers.Dense(10, activation=keras.activations.softmax)
    ])

    model.compile(optimizer=keras.optimizers.RMSprop(lr=0.01), loss=keras.losses.categorical_crossentropy,
                  metrics=keras.metrics.categorical_accuracy)

    logs = model.fit_generator(
        train_generator,
        steps_per_epoch=len(x_train) // batch_size,
        epochs=200,
        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
                   keras.callbacks.LearningRateScheduler(scheduler)],
        validation_data=validation_generator,
        validation_freq=1,
        validation_steps=valid_steps,
        verbose=2,
    )

    model.summary()

    return logs


if __name__ == "__main__":
    # how many time the model will review the training data
    epochs = 200
    # number of data images who spreed through the network (forward propagation), after that the network
    # mean the sum of errors and make only one backpropagation
    # batch size increase the available computational parallelism and make it converge faster to optimum local
    # but algorithm with large batch size will hardly find the minimum global compared to little bach size
    batch_size = 216

    # get data of training and testing from fashion mnist dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)

    # pixel have values from 0 to 255, normalize them
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # transform label (containing a value from O to 9) to matrix of 10 (one hot encoding)
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    train_generator = dataGen_training.flow(x_train, y_train, batch_size=batch_size)

    x_valid = x_train[:30 * batch_size]
    y_valid = y_train[:30 * batch_size]

    valid_steps = x_valid.shape[0] // batch_size
    validation_generator = dataGen_testing.flow(x_valid, y_valid, batch_size=batch_size)

    all_logs = []

    '''
    log = linear_model(x_train, y_train, x_test, y_test, keras.optimizers.SGD(lr=0.05, momentum=0.95),
                       keras.losses.categorical_crossentropy, epochs=epochs, batch_size=batch_size)
    log.history['name'] = "linear model"
    all_logs.append(log)
    '''

    '''
    log = multi_layer_perceptron(x_train, y_train, x_test, y_test, keras.optimizers.SGD(lr=0.05, momentum=0.95),
                                 keras.losses.categorical_crossentropy, epochs, batch_size, "relu", dropout)
    log.history['name'] = "MLP - relu"
    all_logs.append(log)
    '''

    log = convolutional_neural_network("relu")
    log.history['name'] = "relu"
    all_logs.append(log)

    plot_log(all_logs)
