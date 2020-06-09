from .mpn import MPN
from chemprop.args import TrainArgs
from chemprop.nn_utils import get_activation_function

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Softmax


class MoleculeModel(tf.keras.Model):
    """A MoleculeModel is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, args: TrainArgs):
        """
        Initializes the MoleculeModel.

        :param args: Arguments.
        """
        super(MoleculeModel, self).__init__()

        self.classification = args.dataset_type == 'classification'
        self.multiclass = args.dataset_type == 'multiclass'

        self.output_size = args.num_tasks
        if self.multiclass:
            self.output_size *= args.multiclass_num_classes

        if self.classification:
            self.sigmoid = tf.keras.activations.sigmoid

        if self.multiclass:
            self.multiclass_softmax = Softmax(axis=2)

        self.create_encoder(args)
        self.create_ffn(args)

    def create_encoder(self, args: TrainArgs):
        """
        Creates the message passing encoder for the model.

        :param args: Arguments.
        """
        self.encoder = MPN(args)

    def create_ffn(self, args: TrainArgs):
        """
        Creates the feed-forward network for the model.

        :param args: Arguments.
        """
        self.multiclass = args.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.hidden_size
            if args.use_input_features:
                first_linear_dim += args.features_size

        dropout = Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                Dense(self.output_size, input_shape=(first_linear_dim,))
            ]
        else:
            ffn = [
                dropout,
                Dense(args.ffn_hidden_size, input_shape=(first_linear_dim,))
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    Dense(args.ffn_hidden_size, input_shape=(args.ffn_hidden_size,)),
                ])
            ffn.extend([
                activation,
                dropout,
                Dense(self.output_size, input_shape=(args.ffn_hidden_size,)),
            ])

        self.ffn = ffn

    def call(self, *input):
        """
        Runs the MoleculeModel on input.

        :param input: Molecular input.
        :return: The output of the MoleculeModel.
        """

        output = self.encoder(*input)
        for layer in ffn:
            output = layer(output)

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = tf.reshape(output, (output.size(0), -1, self.num_classes))  # batch size x num targets x num classes per target
            if not self.training:
                output = self.multiclass_softmax(output)  # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss

        return output
