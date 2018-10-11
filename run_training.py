import argparse
import logging
import os
import numpy as np
import tensorflow as tf

from containers import STL10Container, POSSIBLE_AUGMENTATIONS
from models import CNN
from tqdm import tqdm


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument('-augmentations', nargs='+', type=str, choices=POSSIBLE_AUGMENTATIONS, default=[])
parser.add_argument('-batch_size', type=int, default=50)
parser.add_argument('-evaluation_step', type=int, default=100)
parser.add_argument('-learning_rate', type=float, default=0.0001)
parser.add_argument('-iterations', type=int, default=4000)
parser.add_argument('-weight_decay', type=float, default=0.0005)

args = parser.parse_args()

if len(args.augmentations) == 0:
    logging.info('Using no augmentations.')
else:
    logging.info('Used augmentations: %s.' % ', '.join(args.augmentations))

logging.info('Loading data...')

train_set = STL10Container('train', args.batch_size, args.augmentations)
test_set = STL10Container('test', args.batch_size)

logging.info('Constructing model...')

network = CNN([96, 96, 3], 10)

ground_truth_placeholder = tf.placeholder(tf.int64, shape=[None])

base_loss = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ground_truth_placeholder, logits=network.outputs)
)
weights = sum([layer.weights for layer in network.layers], [])
weight_decay_loss = args.weight_decay * tf.add_n([tf.nn.l2_loss(w) for w in weights])
total_loss = base_loss + weight_decay_loss

optimizer = tf.train.AdamOptimizer(args.learning_rate)
train_step = optimizer.minimize(total_loss)

logging.info('Training model...')

with tf.Session() as session:
    def _get_ground_truth_and_predictions(dataset):
        ground_truth = np.empty(dataset.size, np.int64)
        predictions = np.empty(dataset.size, np.int64)

        current_index = 0

        for batch_inputs, batch_ground_truth in tqdm(dataset.epoch(), total=dataset.batches_per_epoch()):
            batch_predictions = np.argmax(
                network.outputs.eval(feed_dict={network.inputs: batch_inputs}, session=session),
                axis=-1
            )

            for gt, p in zip(batch_ground_truth, batch_predictions):
                ground_truth[current_index] = gt
                predictions[current_index] = p

                current_index += 1

        return ground_truth, predictions

    session.run(tf.global_variables_initializer())

    n_epochs = args.iterations // args.evaluation_step

    for i in range(n_epochs):
        logging.info('Processing epoch %d/%d...' % (i + 1, n_epochs))

        base_losses = []
        weight_decay_losses = []
        total_losses = []

        for _ in tqdm(range(args.evaluation_step)):
            batch_images, batch_labels = train_set.batch()

            feed_dict = {
                network.inputs: batch_images, ground_truth_placeholder: batch_labels, network.is_training: True
            }

            _, batch_base_loss, batch_weight_decay_loss, batch_total_loss = session.run(
                [train_step, base_loss, weight_decay_loss, total_loss], feed_dict=feed_dict
            )

            base_losses.append(batch_base_loss)
            weight_decay_losses.append(batch_weight_decay_loss)
            total_losses.append(batch_total_loss)

        logging.info('Observed base loss = %.4f, weight decay loss = %.4f and total loss = %.4f.' %
                     (np.mean(base_losses), np.mean(weight_decay_losses), np.mean(total_losses)))
        logging.info('Evaluating on test data...')

        ground_truth, predictions = _get_ground_truth_and_predictions(test_set)
        test_accuracy = np.mean(ground_truth == predictions)

        logging.info('Observed test accuracy = %.4f.' % test_accuracy)
