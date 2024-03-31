import numpy as np

from tensorflow.keras.datasets import mnist, fashion_mnist

IMG_DIM = 28


def get_train_test(training=True):
    (x_train_val, y_train_val), (x_test, y_test) = mnist.load_data()
    if training:
        x_train_val = x_train_val.astype('float32')
        x_train_val /= 255.
        x_train_val = x_train_val.reshape(-1, IMG_DIM, IMG_DIM, 1)

        x_train = x_train_val[:-10000]
        y_train = y_train_val[:-10000]

        x_val = x_train_val[-10000:]
        y_val = y_train_val[-10000:]
        return (x_train, y_train), (x_val, y_val)
    else:
        x_test = x_test.astype('float32')
        x_test /= 255.
        x_test = x_test.reshape(-1, IMG_DIM, IMG_DIM, 1)
        return (x_test, y_test)


def get_train_test_fashion(training=True):
    (x_train_val, y_train_val), (x_test, y_test) = fashion_mnist.load_data()
    if training:
        x_train_val = x_train_val.astype('float32')
        x_train_val /= 255.
        x_train_val = x_train_val.reshape(-1, IMG_DIM, IMG_DIM, 1)

        x_train = x_train_val[:-10000]
        y_train = y_train_val[:-10000]

        x_val = x_train_val[-10000:]
        y_val = y_train_val[-10000:]
        return (x_train, y_train), (x_val, y_val)
    else:
        x_test = x_test.astype('float32')
        x_test /= 255.
        x_test = x_test.reshape(-1, IMG_DIM, IMG_DIM, 1)
        return (x_test, y_test)


def get_eval_data():
    x_k_test, y_k_test = get_train_test(training=False)
    x_u_test, y_u_test = get_train_test_fashion(training=False)

    y_u_test = np.full(y_u_test.shape, 10)

    x_all = np.concatenate((x_k_test, x_u_test), axis=0)
    y_all = np.concatenate((y_k_test, y_u_test), axis=0)
    return (x_all, y_all)
