import math
import numpy as np
import tensorflow as tf

from tensorflow.keras.datasets import mnist, fashion_mnist

IMG_DIM = 28
BATCH_SIZE = 256


class DataLoader(tf.keras.utils.Sequence):
    def __init__(self, x, y, batch_size=BATCH_SIZE):
        self.x = x
        self.y = y
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        low = idx * self.batch_size
        high = min(low + self.batch_size, len(self.x))

        batch_x = self.x[low:high]
        batch_y = self.y[low:high]
        return np.asarray(batch_x), np.asarray(batch_y)

    def get_all(self):
        return np.asarray(self.x), np.asarray(self.y)

    def get_class_num(self):
        return len(np.unique(np.asarray(self.y), axis=0))


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

        train_ds = DataLoader(x_train, y_train)
        val_ds = DataLoader(x_val, y_val)
        return train_ds, val_ds
    else:
        x_test = x_test.astype('float32')
        x_test /= 255.
        x_test = x_test.reshape(-1, IMG_DIM, IMG_DIM, 1)

        test_ds = DataLoader(x_test, y_test)
        return test_ds


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

        train_ds = DataLoader(x_train, y_train)
        val_ds = DataLoader(x_val, y_val)
        return train_ds, val_ds
    else:
        x_test = x_test.astype('float32')
        x_test /= 255.
        x_test = x_test.reshape(-1, IMG_DIM, IMG_DIM, 1)

        test_ds = DataLoader(x_test, y_test)
        return test_ds


def get_eval_data():
    test_k_ds = get_train_test(training=False)
    test_u_ds = get_train_test_fashion(training=False)

    x_k_test, y_k_test = test_k_ds.get_all()
    x_u_test, y_u_test = test_u_ds.get_all()

    y_u_test = np.full(y_u_test.shape, 10)

    x_all = np.concatenate((x_k_test, x_u_test), axis=0)
    y_all = np.concatenate((y_k_test, y_u_test), axis=0)

    eval_ds = DataLoader(x_all, y_all)
    return eval_ds
