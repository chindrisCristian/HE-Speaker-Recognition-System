import os.path as path
import random
from glob import glob

import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from Utils import Utils


class Dataset:
    @staticmethod
    def load_data(path_to_npy):
        npy_files = glob("{}*.npy".format(path_to_npy))
        x = np.load(npy_files[0])
        y = np.zeros(x.shape[0])
        for i, npy_file in enumerate(npy_files[1:]):
            temp = np.load(npy_file)
            x = np.vstack((x, temp))
            y = np.append(y, np.full(temp.shape[0], fill_value=(i + 1)))
        x = Dataset.transform_values(x)
        Utils.print_info("Dataset size is {}".format(x.shape[0]))
        x_train, x_test, y_train, y_test = train_test_split(x, y)
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        return x_train, y_train, x_test, y_test

    @staticmethod
    def get_labels(directory_path):
        files = glob("{}*.*".format(directory_path))
        labels = []
        for file in files:
            filename_w_ext = path.basename(file)
            filename, ext = path.splitext(filename_w_ext)
            labels.append(filename)
        return labels

    @staticmethod
    def get_user(path_to_npy, index):
        labels = Dataset.get_labels(path_to_npy)
        x = np.load(path_to_npy + labels[index] + ".npy")
        x = Dataset.transform_values(x)
        start = random.randint(0, x.shape[0] - 10)
        return x[start : start + 10]

    @staticmethod
    def transform_values(input_val):
        input_val = input_val.astype("float32")
        input_val /= 255.0
        input_val = input_val.reshape(
            input_val.shape[0], input_val.shape[1], input_val.shape[2], 1
        )
        return input_val
