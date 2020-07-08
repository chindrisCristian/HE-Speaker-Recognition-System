from glob import glob

import numpy as np
import skimage.io as io

from keras.layers import Input

from Dataset import Dataset
from SpectogramGenerator import SpectogramGenerator
from Utils import Utils
from Network import Network

import math


def from_wav_to_png(path_to_wav, path_to_png):
    wav_files = glob("{}*.wav".format(path_to_wav), recursive=True)
    for wav_file in wav_files:
        Utils.print_info("Working on file {}".format(wav_file))
        sg = SpectogramGenerator(wav_file)
        sg.get_spectograms(path_to_png)
        Utils.print_info("All set for {}.".format(wav_file))


def from_png_to_npy(path_to_png, path_to_npy, labels):
    for label in labels:
        vectors = []
        files = glob("{}{}_*.png".format(path_to_png, label))
        for file in files:
            vectors.append(io.imread(file))
        np.save("{}{}.npy".format(path_to_npy, label), vectors)
        Utils.print_info("Saved the array for {}.".format(label))


from_wav_to_png("dataset/wav/", "dataset/png/")
labels = Dataset.get_labels("dataset/wav/")
from_png_to_npy("dataset/png/", "dataset/npy/", labels)

