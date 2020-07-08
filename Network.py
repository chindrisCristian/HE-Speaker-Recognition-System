import numpy as np
import tensorflow as tf
from keras.layers import (
    Activation,
    AveragePooling2D,
    Conv2D,
    Dense,
    Flatten,
    Input,
    Reshape,
)
from keras.losses import categorical_crossentropy
from keras.models import Model
from keras.optimizers import rmsprop

from Dataset import Dataset
from Utils import Utils


class Network:
    def __init__(self, input_values, num_classes):
        self.input_values = input_values
        self.num_classes = num_classes
        self.model = None

    def train(self, path_to_npy, epochs, batch_size=100):
        x_train, y_train, x_test, y_test = Dataset.load_data(path_to_npy)

        Utils.print_info("x_train size: {}".format(x_train.shape))
        Utils.print_info("y_train size: {}".format(y_train.shape))
        Utils.print_info("x_test size: {}".format(x_test.shape))
        Utils.print_info("y_test size: {}".format(y_test.shape))
        Utils.print_info("Number of classes: {}".format(self.num_classes))

        def loss(labels, logits):
            return categorical_crossentropy(labels, logits, from_logits=True)

        optimizer = rmsprop(lr=0.0001, decay=1e-6)

        speaker_model = self.__get_model()
        Utils.print_info("Speaker model summary:")
        print(speaker_model.summary())

        speaker_model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

        speaker_model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, y_test),
            verbose=2,
        )

        test_loss, test_acc = speaker_model.evaluate(x_test, y_test, verbose=2)
        Utils.print_info("Trained model, accuracy: {:5.2f}%".format(100 * test_acc))

        self.model = speaker_model

    def save_squashed_model(self, directory, filename):
        weights = self.__squash_layers()
        (conv1_weights, squashed_weights, fc2_weights) = weights[0:3]
        tf.reset_default_graph()
        sess = tf.compat.v1.Session()

        y = self.__get_squashed_model(conv1_weights, squashed_weights, fc2_weights)

        sess.run(tf.compat.v1.global_variables_initializer())
        self.__save_model(sess, ["output/BiasAdd"], directory, filename)

    @classmethod
    def __square_activation(cls, x):
        return x * x

    def __get_model(self):
        y = Conv2D(
            filters=8,
            kernel_size=3,
            strides=2,
            padding="same",
            input_shape=(
                self.input_values.shape[1],
                self.input_values.shape[2],
                self.input_values.shape[3],
            ),
            name="conv2d_1",
        )(self.input_values)
        y = Activation(Network.__square_activation)(y)

        y = Conv2D(filters=8, kernel_size=3, padding="same", name="conv2d_2")(y)
        y = AveragePooling2D(pool_size=3, strides=1, padding="same")(y)
        y = Conv2D(filters=16, kernel_size=3, padding="same", name="conv2d_3")(y)
        y = Conv2D(filters=16, kernel_size=3, padding="same", name="conv2d_4")(y)
        y = AveragePooling2D(pool_size=2, strides=1, padding="same")(y)
        y = Flatten()(y)

        y = Dense(units=15 * self.num_classes, name="fc_1")(y)
        y = Activation(Network.__square_activation)(y)
        y = Dense(units=self.num_classes, name="fc_2")(y)

        return Model(inputs=self.input_values, outputs=y)

    def __get_squashed_model(self, conv1_weights, squashed_weights, fc2_weights):
        x = Input(
            shape=(
                int(self.input_values.shape[1]),
                int(self.input_values.shape[2]),
                int(self.input_values.shape[3]),
            ),
            name="input",
        )
        y = Conv2D(
            filters=8,
            kernel_size=3,
            strides=2,
            padding="same",
            input_shape=(
                self.input_values.shape[1],
                self.input_values.shape[2],
                self.input_values.shape[3],
            ),
            trainable=False,
            kernel_initializer=tf.compat.v1.constant_initializer(conv1_weights[0]),
            bias_initializer=tf.compat.v1.constant_initializer(conv1_weights[1]),
            name="conv2d_1",
        )(x)
        y = Activation(Network.__square_activation)(y)
        y = tf.reshape(
            y,
            [
                -1,
                8
                * (int(self.input_values.shape[1] // 2))
                * (int(self.input_values.shape[2] // 2) + 1),
            ],
        )
        y = Dense(
            units=15 * self.num_classes,
            trainable=False,
            kernel_initializer=tf.compat.v1.constant_initializer(squashed_weights[0]),
            bias_initializer=tf.compat.v1.constant_initializer(squashed_weights[1]),
            name="squashed_fc_1",
        )(y)

        y = Activation(Network.__square_activation)(y)
        y = Dense(
            units=self.num_classes,
            trainable=False,
            kernel_initializer=tf.compat.v1.constant_initializer(fc2_weights[0]),
            bias_initializer=tf.compat.v1.constant_initializer(fc2_weights[1]),
            name="output",
        )(y)

        return y

    def __squash_layers(self):
        Utils.print_info("Starting the squashing layers operation...")

        if self.model is None:
            raise Exception("Model not trained!")

        sess = tf.compat.v1.keras.backend.get_session()
        layers = self.model.layers
        layer_names = [layer.name for layer in layers]
        conv1_weights = layers[layer_names.index("conv2d_1")].get_weights()
        conv2_weights = layers[layer_names.index("conv2d_2")].get_weights()
        conv3_weights = layers[layer_names.index("conv2d_3")].get_weights()
        conv4_weights = layers[layer_names.index("conv2d_4")].get_weights()
        fc1_weights = layers[layer_names.index("fc_1")].get_weights()
        fc2_weights = layers[layer_names.index("fc_2")].get_weights()

        y = self.__get_squashing_layers_model(
            fc1_weights, conv2_weights, conv3_weights, conv4_weights
        )

        sess.run(tf.compat.v1.global_variables_initializer())

        squashed_bias = y.eval(
            session=sess,
            feed_dict={
                "squashed_input:0": np.zeros(
                    (
                        1,
                        (int(self.input_values.shape[1] // 2))
                        * (int(self.input_values.shape[2] // 2) + 1)
                        * 8,
                    )
                )
            },
        )

        squashed_bias_plus_weights = y.eval(
            session=sess,
            feed_dict={
                "squashed_input:0": np.eye(
                    (int(self.input_values.shape[1] // 2))
                    * (int(self.input_values.shape[2] // 2) + 1)
                    * 8
                )
            },
        )

        squashed_weights = squashed_bias_plus_weights - squashed_bias

        return (conv1_weights, (squashed_weights, squashed_bias), fc2_weights)

    def __get_squashing_layers_model(
        self, fc1_weights, conv2_weights, conv3_weights, conv4_weights
    ):
        y = Input(
            shape=(
                (int(self.input_values.shape[1] // 2))
                * (int(self.input_values.shape[2] // 2) + 1)
                * 8,
            ),
            name="squashed_input",
        )
        y = Reshape(
            (
                int(self.input_values.shape[1] // 2),
                int(self.input_values.shape[2] // 2) + 1,
                8,
            )
        )(y)
        y = Conv2D(
            filters=8,
            kernel_size=3,
            padding="same",
            trainable=False,
            kernel_initializer=tf.compat.v1.constant_initializer(conv2_weights[0]),
            bias_initializer=tf.compat.v1.constant_initializer(conv2_weights[1]),
            name="conv2_test",
        )(y)
        y = AveragePooling2D(pool_size=3, strides=1, padding="same")(y)
        y = Conv2D(
            filters=16,
            kernel_size=3,
            padding="same",
            trainable=False,
            kernel_initializer=tf.compat.v1.constant_initializer(conv3_weights[0]),
            bias_initializer=tf.compat.v1.constant_initializer(conv3_weights[1]),
            name="conv3_test",
        )(y)
        y = Conv2D(
            filters=16,
            kernel_size=3,
            padding="same",
            trainable=False,
            kernel_initializer=tf.compat.v1.constant_initializer(conv4_weights[0]),
            bias_initializer=tf.compat.v1.constant_initializer(conv4_weights[1]),
            name="conv4_test",
        )(y)
        y = AveragePooling2D(pool_size=2, strides=1, padding="same")(y)
        y = Flatten()(y)
        y = Dense(
            units=15 * self.num_classes,
            trainable=False,
            kernel_initializer=tf.compat.v1.constant_initializer(fc1_weights[0]),
            bias_initializer=tf.compat.v1.constant_initializer(fc1_weights[1]),
            name="fc1_test",
        )(y)

        return y

    def __save_model(self, sess, output_names, directory, filename):
        frozen_graph = self.__freeze_session(sess, output_names=output_names)
        Utils.print_nodes(frozen_graph)
        tf.io.write_graph(frozen_graph, directory, filename + ".pb", as_text=False)
        Utils.print_info("Model saved to: {}.pb".format(filename))

    def __freeze_session(
        self, session, keep_var_names=None, output_names=None, clear_devices=True
    ):

        from tensorflow.python.framework.graph_util import (
            convert_variables_to_constants,
            remove_training_nodes,
        )

        graph = session.graph
        with graph.as_default():
            freeze_var_names = list(
                set(v.op.name for v in tf.global_variables()).difference(
                    keep_var_names or []
                )
            )
            output_names = output_names or []
            output_names += [v.op.name for v in tf.global_variables()]
            # Graph -> GraphDef ProtoBuf
            input_graph_def = graph.as_graph_def()
            Utils.print_nodes(input_graph_def)
            if clear_devices:
                for node in input_graph_def.node:
                    node.device = ""
            frozen_graph = convert_variables_to_constants(
                session, input_graph_def, output_names, freeze_var_names
            )
            frozen_graph = remove_training_nodes(frozen_graph)
            return frozen_graph
