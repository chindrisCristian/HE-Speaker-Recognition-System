from keras.layers import Input

from Network import Network
from Utils import Utils


def train_network(FLAGS):
    x = Input(shape=(128, 101, 1), name="input")
    network = Network(x, 9)

    network.train(FLAGS.npy_path, FLAGS.epochs, FLAGS.batch_size)

    network.save_squashed_model(FLAGS.directory, FLAGS.model)


if __name__ == "__main__":
    FLAGS, unparsed = Utils.train_argument_parser().parse_known_args()
    if unparsed:
        print("Unparsed flags:", unparsed)
        exit(1)

    train_network(FLAGS)
