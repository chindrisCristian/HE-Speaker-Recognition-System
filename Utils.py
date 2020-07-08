import argparse
import os

import numpy as np
import skimage.io as io
import tensorflow as tf

from tensorflow.core.protobuf import rewriter_config_pb2


class Utils:
    @staticmethod
    def append_id(filename, value):
        return "{0}_{2}.{1}".format(*filename.rsplit(".", 1) + [str(value)])

    @staticmethod
    def scale_minmax(X, min=0.0, max=1.0):
        X_std = (X - X.min()) / (X.max() - X.min())
        X_scaled = X_std * (max - min) + min
        return X_scaled

    @staticmethod
    def save_image(img, out):
        io.imsave(out, img)

    @staticmethod
    def print_info(msg):
        print("----->   ", msg)
        print()

    @staticmethod
    def load_pb_file(filename):
        if not os.path.isfile(filename):
            raise Exception("File, " + filename + " does not exist")

        with tf.io.gfile.GFile(filename, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        print("Model restored")
        return graph_def

    @staticmethod
    def print_nodes(graph_def=None):
        if graph_def is None:
            nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]
        else:
            nodes = [n.name for n in graph_def.node]
        Utils.print_info("Current nodes in graph: {}".format(nodes))

    @staticmethod
    def train_argument_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--epochs",
            type=int,
            default=1,
            help="Number of epochs to train the network",
        )
        parser.add_argument(
            "--directory",
            type=str,
            default="./models",
            help="Directory to save the model file",
        )
        parser.add_argument(
            "--npy_path", type=str, default="dataset/", help="Path to .npy files"
        )
        parser.add_argument(
            "--model",
            type=str,
            default="speaker_model",
            help="File name for the model file",
        )
        parser.add_argument(
            "--batch_size", type=int, default=100, help="The size of each batch of data"
        )

    @staticmethod
    def client_argument_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument("--user", type=int, default=0, help="User to identify")
        parser.add_argument("--batch_size", type=int, default=10, help="Batch size")
        parser.add_argument(
            "--hostname", type=str, default="localhost", help="Hostname of server"
        )
        parser.add_argument("--port", type=int, default=34000, help="Port of server")
        parser.add_argument(
            "--encrypt_data_str",
            type=str,
            default="encrypt",
            help='"encrypt" to encrypt client data, "plain" to not encrypt',
        )
        parser.add_argument(
            "--tensor_name", type=str, default="import/input", help="Input tensor name"
        )

        return parser

    @staticmethod
    def server_argument_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--user", type=int, default=0, help="Which user to identify"
        )
        parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
        parser.add_argument(
            "--enable_client", type=str2bool, default=False, help="Enable the client"
        )
        parser.add_argument(
            "--enable_gc", type=str2bool, default=False, help="Enable garbled circuits"
        )
        parser.add_argument(
            "--mask_gc_inputs",
            type=str2bool,
            default=False,
            help="Mask garbled circuits inputs",
        )
        parser.add_argument(
            "--mask_gc_outputs",
            type=str2bool,
            default=False,
            help="Mask garbled circuits outputs",
        )
        parser.add_argument(
            "--num_gc_threads",
            type=int,
            default=1,
            help="Number of threads to run garbled circuits with",
        )
        parser.add_argument(
            "--backend", type=str, default="HE_SEAL", help="Name of backend to use"
        )
        parser.add_argument(
            "--encryption_parameters",
            type=str,
            default="",
            help="Filename containing json description of encryption parameters, or json description itself",
        )
        parser.add_argument(
            "--encrypt_server_data",
            type=str2bool,
            default=False,
            help="Encrypt server data (should not be used when enable_client is used)",
        )
        parser.add_argument(
            "--pack_data",
            type=str2bool,
            default=True,
            help="Use plaintext packing on data",
        )
        parser.add_argument(
            "--start_batch", type=int, default=0, help="Test data start index"
        )
        parser.add_argument(
            "--model_file",
            type=str,
            default="",
            help="Filename of saved protobuf model",
        )
        parser.add_argument(
            "--input_node",
            type=str,
            default="import/input:0",
            help="Tensor name of data input",
        )
        parser.add_argument(
            "--output_node",
            type=str,
            default="import/output/BiasAdd:0",
            help="Tensor name of model output",
        )

        return parser

    @staticmethod
    def server_config_from_flags(FLAGS, tensor_param_name):
        rewriter_options = rewriter_config_pb2.RewriterConfig()
        rewriter_options.meta_optimizer_iterations = (
            rewriter_config_pb2.RewriterConfig.ONE
        )
        rewriter_options.min_graph_nodes = -1
        server_config = rewriter_options.custom_optimizers.add()
        server_config.name = "ngraph-optimizer"
        server_config.parameter_map["ngraph_backend"].s = FLAGS.backend.encode()
        server_config.parameter_map["device_id"].s = b""
        server_config.parameter_map[
            "encryption_parameters"
        ].s = FLAGS.encryption_parameters.encode()
        server_config.parameter_map["enable_client"].s = str(
            FLAGS.enable_client
        ).encode()
        server_config.parameter_map["enable_gc"].s = (str(FLAGS.enable_gc)).encode()
        server_config.parameter_map["mask_gc_inputs"].s = (
            str(FLAGS.mask_gc_inputs)
        ).encode()
        server_config.parameter_map["mask_gc_outputs"].s = (
            str(FLAGS.mask_gc_outputs)
        ).encode()
        server_config.parameter_map["num_gc_threads"].s = (
            str(FLAGS.num_gc_threads)
        ).encode()

        if FLAGS.enable_client:
            server_config.parameter_map[tensor_param_name].s = b"client_input"
        elif FLAGS.encrypt_server_data:
            server_config.parameter_map[tensor_param_name].s = b"encrypt"

        if FLAGS.pack_data:
            server_config.parameter_map[tensor_param_name].s += b",packed"

        config = tf.compat.v1.ConfigProto()
        config.MergeFrom(
            tf.compat.v1.ConfigProto(
                graph_options=tf.compat.v1.GraphOptions(
                    rewrite_options=rewriter_options
                )
            )
        )

        return config

    @staticmethod
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("on", "yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("off", "no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")
