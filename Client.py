import argparse
import os
import sys
import time

import numpy as np

import pyhe_client
from Dataset import Dataset
from Utils import Utils


def test_network(FLAGS):
    x_test = Dataset.get_user("dataset/npy/", FLAGS.user)
    data = x_test.flatten("C")

    client = pyhe_client.HESealClient(
        FLAGS.hostname,
        FLAGS.port,
        FLAGS.batch_size,
        {FLAGS.tensor_name: (FLAGS.encrypt_data_str, data)},
    )

    results = np.round(client.get_results(), 2)

    y_pred_reshape = np.array(results).reshape(FLAGS.batch_size, 9)
    with np.printoptions(precision=3, suppress=True):
        print(y_pred_reshape)

    y_pred = y_pred_reshape.argmax(axis=1)
    print("y_pred", y_pred)

    # correct = np.sum(np.equal(y_pred, y_test.argmax(axis=1)))
    # acc = correct / float(FLAGS.batch_size)
    # print("correct", correct)
    # print("Accuracy (batch size", FLAGS.batch_size, ") =", acc * 100.0, "%")


if __name__ == "__main__":
    FLAGS, unparsed = Utils.client_argument_parser().parse_known_args()
    if unparsed:
        print("Unparsed flags:", unparsed)
        exit(1)

    test_network(FLAGS)
