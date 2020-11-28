import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt


def scheduler(epoch, lr):
    if epoch < 40:
        return lr
    else:
        return lr * tf.math.exp(-0.01)


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


def multi_layer_perceptron_relu(x, y, val_x, val_y, opt, loss_func, epochs, batch_size):
    model = keras.Sequential([
        # convert a two dimensional matrix into a vector
        keras.layers.Flatten(),
        keras.layers.Dense(120, activation=keras.activations.relu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(120, activation=keras.activations.relu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=keras.activations.softmax),
    ])

    model.compile(optimizer=opt, loss=loss_func, metrics=keras.metrics.categorical_accuracy)

    logs = model.fit(x, y, validation_data=(val_x, val_y), epochs=epochs, batch_size=batch_size,
                     callbacks=[keras.callbacks.LearningRateScheduler(scheduler)])
    model.summary()

    return logs

def multi_layer_perceptron_sigmoid(x, y, val_x, val_y, opt, loss_func, epochs, batch_size):
    model = keras.Sequential([
        # convert a two dimensional matrix into a vector
        keras.layers.Flatten(),
        keras.layers.Dense(120, activation=keras.activations.sigmoid),
        keras.layers.Dense(120, activation=keras.activations.sigmoid),
        keras.layers.Dense(10, activation=keras.activations.softmax),
    ])

    model.compile(optimizer=opt, loss=loss_func, metrics=keras.metrics.categorical_accuracy)

    logs = model.fit(x, y, validation_data=(val_x, val_y), epochs=epochs, batch_size=batch_size,
                     callbacks=[keras.callbacks.LearningRateScheduler(scheduler)])
    model.summary()

    return logs

def multi_layer_perceptron_tanh(x, y, val_x, val_y, opt, loss_func, epochs, batch_size):
    model = keras.Sequential([
        # convert a two dimensional matrix into a vector
        keras.layers.Flatten(),
        keras.layers.Dense(120, activation=keras.activations.tanh),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(120, activation=keras.activations.tanh),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=keras.activations.softmax),
    ])

    model.compile(optimizer=opt, loss=loss_func, metrics=keras.metrics.categorical_accuracy)

    logs = model.fit(x, y, validation_data=(val_x, val_y), epochs=epochs, batch_size=batch_size,
                     callbacks=[keras.callbacks.LearningRateScheduler(scheduler)])
    model.summary()

    return logs

def multi_layer_perceptron_selu(x, y, val_x, val_y, opt, loss_func, epochs, batch_size):
    model = keras.Sequential([
        # convert a two dimensional matrix into a vector
        keras.layers.Flatten(),
        keras.layers.Dense(120, activation=keras.activations.selu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(120, activation=keras.activations.selu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=keras.activations.softmax),
    ])

    model.compile(optimizer=opt, loss=loss_func, metrics=keras.metrics.categorical_accuracy)

    logs = model.fit(x, y, validation_data=(val_x, val_y), epochs=epochs, batch_size=batch_size,
                     callbacks=[keras.callbacks.LearningRateScheduler(scheduler)])
    model.summary()

    return logs

def multi_layer_perceptron_elu(x, y, val_x, val_y, opt, loss_func, epochs, batch_size):
    model = keras.Sequential([
        # convert a two dimensional matrix into a vector
        keras.layers.Flatten(),
        keras.layers.Dense(120, activation=keras.activations.elu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(120, activation=keras.activations.elu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=keras.activations.softmax),
    ])

    model.compile(optimizer=opt, loss=loss_func, metrics=keras.metrics.categorical_accuracy)

    logs = model.fit(x, y, validation_data=(val_x, val_y), epochs=epochs, batch_size=batch_size,
                     callbacks=[keras.callbacks.LearningRateScheduler(scheduler)])
    model.summary()

    return logs


if __name__ == "__main__":
    # how many time the model will review the training data
    epochs = 100
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

    log = multi_layer_perceptron_elu(x_train, y_train, x_test, y_test, keras.optimizers.SGD(lr=0.05, momentum=0.95),
                       keras.losses.categorical_crossentropy, epochs=epochs, batch_size=batch_size)
    log.history['name'] = "elu"
    all_logs.append(log)

    log = multi_layer_perceptron_relu(x_train, y_train, x_test, y_test, keras.optimizers.SGD(lr=0.05, momentum=0.95),
                                     keras.losses.categorical_crossentropy, epochs=epochs, batch_size=batch_size)
    log.history['name'] = "relu"
    all_logs.append(log)

    log = multi_layer_perceptron_selu(x_train, y_train, x_test, y_test, keras.optimizers.SGD(lr=0.05, momentum=0.95),
                                     keras.losses.categorical_crossentropy, epochs=epochs, batch_size=batch_size)
    log.history['name'] = "selu"
    all_logs.append(log)

    log = multi_layer_perceptron_sigmoid(x_train, y_train, x_test, y_test, keras.optimizers.SGD(lr=0.05, momentum=0.95),
                                     keras.losses.categorical_crossentropy, epochs=epochs, batch_size=batch_size)
    log.history['name'] = "sigmoid"
    all_logs.append(log)

    log = multi_layer_perceptron_tanh(x_train, y_train, x_test, y_test, keras.optimizers.SGD(lr=0.05, momentum=0.95),
                                     keras.losses.categorical_crossentropy, epochs=epochs, batch_size=batch_size)
    log.history['name'] = "tanh"
    all_logs.append(log)

    plot_log(all_logs)


