import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt


def scheduler(epoch, lr):
    if epoch < 40:
        return lr
    else:
        return lr * tf.math.exp(-0.04)


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
        name = logs.history['name']
        plt.plot(list(range(len(metric))), metric, label=name)
    plt.xlabel("number of epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.title("prediction accuracy on training test")
    plt.show()

    for logs in all_logs:
        metric = logs.history['val_categorical_accuracy']
        name = logs.history['name']
        plt.plot(list(range(len(metric))), metric, label=name)
    plt.xlabel("number of epochs")
    plt.ylabel("accuracy")
    plt.title("prediction accuracy on testing test")
    plt.legend()
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


def multi_layer_perceptron(x, y, val_x, val_y, opt, loss_func, epochs, batch_size, activation, dropout):
    model = keras.Sequential([
        # convert a two dimensional matrix into a vector
        keras.layers.Flatten(),
        keras.layers.Dense(120, activation=activation),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(120, activation=activation),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(10, activation=keras.activations.softmax),
    ])

    model.compile(optimizer=opt, loss=loss_func, metrics=keras.metrics.categorical_accuracy)

    logs = model.fit(x, y, validation_data=(val_x, val_y), epochs=epochs, batch_size=batch_size,
                     callbacks=[keras.callbacks.LearningRateScheduler(scheduler),
                                keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)])
    model.summary()

    return logs


if __name__ == "__main__":
    # how many time the model will review the training data
    epochs = 200
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
    '''
    log = linear_model(x_train, y_train, x_test, y_test, keras.optimizers.SGD(lr=0.05, momentum=0.95),
                       keras.losses.categorical_crossentropy, epochs=epochs, batch_size=batch_size)
    log.history['name'] = "linear model"
    all_logs.append(log)
    '''

    data = [('elu', "0.2"), ('relu', "0.2"),  ('tanh', "0.2")]

    for activation, dropout in data:
        dropout = float(dropout)

        log = multi_layer_perceptron(x_train, y_train, x_test, y_test, keras.optimizers.SGD(lr=0.05, momentum=0.95),
                                     keras.losses.categorical_crossentropy, epochs, batch_size, activation, dropout)
        log.history['name'] = activation
        all_logs.append(log)

    plot_log(all_logs)
