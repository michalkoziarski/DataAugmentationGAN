import argparse
import logging
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from containers import CIFAR10Container, MNISTContainer, STL10Container, POSSIBLE_AUGMENTATIONS
from models import Classifier
from pathlib import Path
from tqdm import tqdm


RESULTS_PATH = Path(__file__).parent / 'results'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument('-augmentations', nargs='+', type=str, choices=POSSIBLE_AUGMENTATIONS, default=[])
parser.add_argument('-batch_size', type=int, default=50)
parser.add_argument('-dataset', type=str, choices=['cifar10', 'mnist', 'stl10'], required=True)
parser.add_argument('-evaluation_step', type=int, default=100)
parser.add_argument('-gaussian_noise_std', type=int, default=2)
parser.add_argument('-generated_data_name_suffix', type=str)
parser.add_argument('-iterations', type=int, default=15000)
parser.add_argument('-learning_rate', type=float, default=0.0001)
parser.add_argument('-n_generated_images', type=int, default=0)
parser.add_argument('-name_suffix', type=str)
parser.add_argument('-rotation_range', type=int, default=30)
parser.add_argument('-scale_range', type=float, default=1.8)
parser.add_argument('-snp_noise_probability', type=float, default=0.001)
parser.add_argument('-translation_range', type=float, default=0.25)
parser.add_argument('-weight_decay', type=float, default=0.0005)

args = parser.parse_args()

if len(args.augmentations) == 0:
    logging.info('Using no augmentations.')

    experiment_name = 'experiment_augmentations=none'
else:
    logging.info('Used augmentations: %s.' % ', '.join(args.augmentations))

    experiment_name = 'experiment_augmentations=' + '+'.join(args.augmentations)

if args.n_generated_images > 0:
    experiment_name += '_n_generated_images=%d' % args.n_generated_images

if args.name_suffix is not None:
    experiment_name += '_%s' % args.name_suffix

logging.info('Experiment name: %s/%s.' % (args.dataset, experiment_name))
logging.info('Loading data...')

if args.dataset == 'cifar10':
    Container = CIFAR10Container
elif args.dataset == 'mnist':
    Container = MNISTContainer
elif args.dataset == 'stl10':
    Container = STL10Container
else:
    raise NotImplementedError

train_set = Container('train', args.batch_size, args.augmentations, args.rotation_range, args.scale_range,
                      args.translation_range, args.gaussian_noise_std, args.snp_noise_probability,
                      image_size=[64, 64], n_generated_images=args.n_generated_images,
                      generated_data_name_suffix=args.generated_data_name_suffix)
test_set = Container('test', args.batch_size, image_size=[64, 64])

logging.info('Constructing model...')

network = Classifier([64, 64, 3], [10])

ground_truth_placeholder = tf.placeholder(tf.int64, shape=[None])

base_loss = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ground_truth_placeholder, logits=network.outputs)
)
weights = sum([layer.weights for layer in network.layers], [])
weight_decay_loss = args.weight_decay * tf.add_n([tf.nn.l2_loss(w) for w in weights])
total_loss = base_loss + weight_decay_loss

optimizer = tf.train.AdamOptimizer(args.learning_rate)

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    train_step = optimizer.minimize(total_loss)

logging.info('Training model...')

results = {'iteration': [], 'accuracy': []}

with tf.Session() as session:
    def _get_ground_truth_and_predictions(dataset):
        ground_truth = np.empty(dataset.n_images, np.int64)
        predictions = np.empty(dataset.n_images, np.int64)

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
                     (float(np.mean(base_losses)), float(np.mean(weight_decay_losses)), float(np.mean(total_losses))))
        logging.info('Evaluating on test data...')

        ground_truth, predictions = _get_ground_truth_and_predictions(test_set)
        test_accuracy = np.mean(ground_truth == predictions)

        logging.info('Observed test accuracy = %.4f.' % test_accuracy)

        results['iteration'].append(i + 1)
        results['accuracy'].append('%.4f' % test_accuracy)

logging.info('Saving results...')

dataset_results_path = RESULTS_PATH / args.dataset
dataset_results_path.mkdir(parents=True, exist_ok=True)

pd.DataFrame(results).to_csv(str(dataset_results_path / ('%s.csv' % experiment_name)), index=False)
