import logging
from typing import Callable

from tensorboardX import SummaryWriter
from tqdm import tqdm

from chemprop.args import TrainArgs
from chemprop.data import MoleculeDataLoader, MoleculeDataset
from chemprop.nn_utils import compute_gnorm, compute_pnorm, NoamLR

import tensorflow as tf

def train(model: tf.keras.Model,
          data_loader: MoleculeDataLoader,
          loss_func: Callable,
          optimizer,
          args: TrainArgs,
          n_iter: int = 0,
          logger: logging.Logger = None,
          writer: SummaryWriter = None) -> int:
    """
    Trains a model for an epoch.

    :param model: Model.
    :param data_loader: A MoleculeDataLoader.
    :param loss_func: Loss function.
    :param optimizer: An Optimizer.
    :param args: Arguments.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param logger: A logger for printing intermediate results.
    :param writer: A tensorboardX SummaryWriter.
    :return: The total number of iterations (training examples) trained on so far.
    """
    debug = logger.debug if logger is not None else print
    
    loss_sum, iter_count = 0, 0

    for batch in tqdm(data_loader, total=len(data_loader)):
        with tf.GradientTape() as tape:
            # Prepare batch
            mol_batch, features_batch, target_batch = batch.batch_graph(), batch.features(), batch.targets()
            mask = tf.convert_to_tensor([[x is not None for x in tb] for tb in target_batch], dtype=float)
            targets = tf.convert_to_tensor([[0 if x is None else x for x in tb] for tb in target_batch], dtype=float)
            class_weights = tf.ones(targets.shape)

            # Run model
            preds = model(mol_batch, training=True)

            if args.dataset_type == 'multiclass':
                targets = targets.long()
                loss = tf.concat([loss_func(preds[:, target_index, :], targets[:, target_index]).unsqueeze(1) for target_index in range(preds.size(1))], axis=1) * class_weights * mask
            else:
                loss = loss_func(preds, targets) * class_weights * mask
            loss = tf.math.reduce_sum(loss) / tf.math.reduce_sum(mask)

            loss_sum += loss
            iter_count += len(batch)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        n_iter += len(batch)

        # Log and/or add to tensorboard
        if (n_iter // args.batch_size) % args.log_frequency == 0:
            #pnorm = compute_pnorm(model)
            #gnorm = compute_gnorm(model)
            loss_avg = loss_sum / iter_count
            loss_sum, iter_count = 0, 0

            #debug(f'Loss = {loss_avg:.4e}, PNorm = {pnorm:.4f}, GNorm = {gnorm:.4f}')
            debug(f'Loss = {loss_avg:.4e}')

            if writer is not None:
                writer.add_scalar('train_loss', loss_avg.numpy(), n_iter)
                #writer.add_scalar('param_norm', pnorm, n_iter)
                #writer.add_scalar('gradient_norm', gnorm, n_iter)

    return n_iter
