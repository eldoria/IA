import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt


def scheduler(epoch, lr):
    if epoch < 40:
        return lr
    else:
        return lr * tf.math.exp(-0.5)


def plot_log(all_logs):
    for logs in all_logs:
        losses = logs.history['loss']
        plt.plot(list(range(len(losses))), losses)
    plt.xlabel("number of epochs")
    plt.ylabel("error")
    plt.title("error on training data")
    plt.show()

    for logs in all_logs:
        losses = logs.history['val_loss']
        plt.plot(list(range(len(losses))), losses)
    plt.xlabel("number of epochs")
    plt.ylabel("error")
    plt.title("error on testing data")
    plt.show()

    for logs in all_logs:
        metric = logs.history['categorical_accuracy']
        plt.plot(list(range(len(metric))), metric)
    plt.xlabel("number of epochs")
    plt.ylabel("accuracy")
    plt.title("prediction accuracy on training test")
    plt.show()

    for logs in all_logs:
        metric = logs.history['val_categorical_accuracy']
        plt.plot(list(range(len(metric))), metric)
    plt.xlabel("number of epochs")
    plt.ylabel("accuracy")
    plt.title("prediction accuracy on testing test")
    plt.show()


def linear_model(x, y, val_x, val_y, opt, loss_func, epochs, batch_size):
    model = keras.Sequential([
        # convert a two dimensional matrix into a vector
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation=keras.activations.softmax),
    ])

    model.compile(optimizer=opt, loss=loss_func, metrics=keras.metrics.categorical_accuracy)

    logs = model.fit(x, y, validation_data=(val_x, val_y), epochs=epochs, batch_size=batch_size,
                     callbacks=[keras.callbacks.LearningRateScheduler(scheduler)])
    model.summary()

    return logs


if __name__ == "__main__":
    # how many time the model will review the training data
    epochs = 50
    # number of data images who spreed through the network (forward propagation), after that the network
    # mean the sum of errors and make only one backpropagation
    # batch size increase the available computational parallelism and make it converge faster to optimum local
    # but algorithm with large batch size will hardly find the minimum global compared to little bach size
    batch_size = 1024

    # get data of training and testing from fashion mnist dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    # pixel have values from 0 to 255, normalize them
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # transform label (containing a value from O to 9) to matrix of 10 (one hot encoding)
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    all_logs = []
    log = linear_model(x_train, y_train, x_test, y_test, keras.optimizers.SGD(lr=0.05, momentum=0.95),
                       keras.losses.categorical_crossentropy, epochs=epochs, batch_size=batch_size)

    all_logs.append(log)

    plot_log(all_logs)


