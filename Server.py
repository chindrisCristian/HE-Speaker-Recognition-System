import time
import numpy as np
import tensorflow as tf
import ngraph_bridge

from Utils import Utils
from Dataset import Dataset


def test_network(FLAGS):
    # Load user
    x_test = Dataset.get_user("dataset/", FLAGS.user)
    # Load saved model
    tf.import_graph_def(Utils.load_pb_file(FLAGS.model_file))

    Utils.print_info("Model loaded")
    Utils.print_nodes()

    # Get input / output tensors
    x_input = tf.compat.v1.get_default_graph().get_tensor_by_name(FLAGS.input_node)
    y_output = tf.compat.v1.get_default_graph().get_tensor_by_name(FLAGS.output_node)

    # Create configuration to encrypt input
    FLAGS, unparsed = Utils.server_argument_parser().parse_known_args()
    config = Utils.server_config_from_flags(FLAGS, x_input.name)
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        start_time = time.time()
        y_hat = y_output.eval(feed_dict={x_input: x_test})
        elasped_time = time.time() - start_time
        Utils.print_info("Total time(s): {}".format(np.round(elasped_time, 3)))

    if not FLAGS.enable_client:

        if x_test.shape[0] < 60:
            print("y_hat", np.round(y_hat, 2))

        y_pred = np.argmax(y_hat, 1)
        Utils.print_info(
            "Output array of users from the {} samples:".format(x_test.shape[0])
        )
        Utils.print_info("Results array: {}".format(y_pred))
        user = np.zeros(13)
        for value in y_pred:
            user[value] = user[value] + 1
        correct_user = np.argmax(user)
        test_accuracy = user[correct_user] / 10

        if test_accuracy > 0.8:
            labels = Dataset.get_labels("dataset/")
            Utils.print_info("User is {}".format(labels[correct_user]))
            Utils.print_info("Accuracy: {:5.2f}%".format(100 * test_accuracy))
            Utils.print_info(
                "Correct answers: {} of {}".format(user[correct_user], x_test.shape[0])
            )
            Utils.print_info("Users: {}".format(labels))
        else:
            Utils.print_info("The test has failed, please try again!")


if __name__ == "__main__":
    FLAGS, unparsed = Utils.server_argument_parser().parse_known_args()

    if unparsed:
        print("Unparsed flags:", unparsed)
        exit(1)
    if FLAGS.encrypt_server_data and FLAGS.enable_client:
        raise Exception(
            "encrypt_server_data flag only valid when client is not enabled. Note: the client can specify whether or not to encrypt the data using 'encrypt' or 'plain' in the configuration map"
        )
    if FLAGS.model_file == "":
        raise Exception("FLAGS.model_file must be set")

    test_network(FLAGS)
