from typing import List, Union

import numpy as np
from rdkit import Chem

from chemprop.args import TrainArgs
from chemprop.features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from chemprop.nn_utils import index_select_ND, get_activation_function

import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Dense, Dropout

# TODO inheriting from Layer?
class MPNEncoder(layers.Layer):
    """A message passing neural network for encoding a molecule."""

    def __init__(self, args: TrainArgs, atom_fdim: int, bond_fdim: int):
        """Initializes the MPNEncoder.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        :param atom_messages: Whether to use atoms to pass messages instead of bonds.
        """
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.atom_messages = args.atom_messages
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
        self.layers_per_message = 1
        self.undirected = args.undirected
        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.device = args.device
        self.args = args

        if self.features_only:
            return

        # Dropout
        self.dropout_layer = Dropout(self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Cached zeros
        self.cached_zero_vector = tf.zeros(self.hidden_size)

        # Input
        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i = Dense(self.hidden_size, input_shape=(input_dim,), use_bias=self.bias)

        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size

        # Shared weight matrix across depths (default)
        self.W_h = Dense(self.hidden_size, input_shape=(w_h_input_size,), use_bias=self.bias)

        self.W_o = Dense(self.hidden_size, input_shape=(self.atom_fdim + self.hidden_size,))

    @tf.function(experimental_relax_shapes=True)
    def call(self, f_atoms, f_bonds, a2b, a2b2, b2a, b2revb, a_scope, b_scope) -> tf.Tensor:
    #mol_graph: BatchMolGraph) -> tf.Tensor:
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A BatchMolGraph representing a batch of molecular graphs.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """

        #f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components(atom_messages=self.atom_messages)

        if self.atom_messages:
            a2a = mol_graph.get_a2a()

        # Input
        if self.atom_messages:
            input = self.W_i(f_atoms)  # num_atoms x hidden_size
        else:
            input = self.W_i(f_bonds)  # num_bonds x hidden_size
        message = self.act_func(input)  # num_bonds x hidden_size

        # Message passing
        for depth in range(self.depth - 1):
            if self.undirected:
                message = (message + message[b2revb]) / 2

            if self.atom_messages:
                raise Exception("we are bailing out")
                nei_a_message = index_select_ND(message, a2a)  # num_atoms x max_num_bonds x hidden
                nei_f_bonds = index_select_ND(f_bonds, a2b)  # num_atoms x max_num_bonds x bond_fdim
                nei_message = tf.concat((nei_a_message, nei_f_bonds), 2)  # num_atoms x max_num_bonds x (hidden + bond_fdim)
                message = tf.math.reduce_sum(nei_message, axis=1)  # num_atoms x (hidden + bond_fdim)
            else:
                # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
                # message      a_message = sum(nei_a_message)      rev_message
                #nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
                #a_message = tf.math.reduce_sum(nei_a_message, axis=1)  # num_atoms x hidden
                a_message = tf.math.segment_sum(message, a2b2)
                rev_message = tf.gather(message, b2revb) # num_bonds x hidden
                message = tf.gather(a_message, b2a) - rev_message  # num_bonds x hidden

            message = self.W_h(message)
            message = self.act_func(input + message)  # num_bonds x hidden_size
            message = self.dropout_layer(message)  # num_bonds x hidden

        a2x = a2a if self.atom_messages else a2b
        nei_a_message = index_select_ND(message, a2x)  # num_atoms x max_num_bonds x hidden
        a_message = tf.math.reduce_sum(nei_a_message, axis=1)  # num_atoms x hidden
        a_input = tf.concat((f_atoms, a_message), axis=1)  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden

        # Readout
        mol_vecs = tf.math.segment_mean(atom_hiddens, a_scope)[1:]

        return mol_vecs  # num_molecules x hidden


class MPN(layers.Layer):
    """A message passing neural network for encoding a molecule."""

    def __init__(self,
                 args: TrainArgs,
                 atom_fdim: int = None,
                 bond_fdim: int = None):
        """
        Initializes the MPN.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        """
        super(MPN, self).__init__()
        self.args = args
        self.atom_fdim = atom_fdim or get_atom_fdim()
        self.bond_fdim = bond_fdim or get_bond_fdim(atom_messages=args.atom_messages)
        self.encoder = MPNEncoder(self.args, self.atom_fdim, self.bond_fdim)

    def call(self, batch: Union[List[str], List[Chem.Mol], BatchMolGraph]) -> tf.Tensor:
        """
        Encodes a batch of molecular SMILES strings.

        :param batch: A list of SMILES strings, a list of RDKit molecules, or a BatchMolGraph.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        if type(batch) != BatchMolGraph:
            batch = mol2graph(batch)

        f_atoms, f_bonds, a2b, a2b2, b2a, b2revb, a_scope, b_scope = batch.get_components()
        output = self.encoder(f_atoms, f_bonds, a2b, a2b2, b2a, b2revb, a_scope, b_scope)

        return output
