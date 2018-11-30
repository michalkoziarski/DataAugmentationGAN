import argparse
import imageio
import logging
import numpy as np
import os
import tensorflow as tf

from containers import CIFAR10Container, MNISTContainer, STL10Container
from models import Discriminator, Generator
from pathlib import Path
from tqdm import tqdm


def one_hot(labels, n_classes):
    return np.squeeze(np.eye(n_classes)[labels.reshape(-1)])


SAMPLES_PATH = Path(__file__).parent / 'samples'
GENERATED_DATA_PATH = Path(__file__).parent / 'generated_data'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument('-batch_size', type=int, default=100)
parser.add_argument('-dataset', type=str, choices=['cifar10', 'mnist', 'stl10'], required=True)
parser.add_argument('-evaluation_step', type=int, default=50)
parser.add_argument('-generated_images_per_class', type=int, default=10000)
parser.add_argument('-iterations', type=int, default=7500)
parser.add_argument('-image_similarity_decay', type=float, default=0.001)
parser.add_argument('-learning_rate', type=float, default=0.0002)
parser.add_argument('-z_shape', type=int, default=100)

args = parser.parse_args()

logging.info('Loading data...')

if args.dataset == 'cifar10':
    Container = CIFAR10Container
elif args.dataset == 'mnist':
    Container = MNISTContainer
elif args.dataset == 'stl10':
    Container = STL10Container
else:
    raise NotImplementedError

dataset = Container('train', args.batch_size, image_size=[64, 64])
n_classes = len(np.unique(dataset.labels))

logging.info('Constructing model...')

is_training = tf.placeholder_with_default(False, [])
conditional_inputs = tf.placeholder(tf.float32, [None, n_classes])

generator = Generator([1, 1, args.z_shape], is_training=is_training, conditional_inputs=conditional_inputs)
real_discriminator = Discriminator([64, 64, 3], [n_classes + 1], is_training=is_training)
fake_discriminator = Discriminator(inputs=generator.outputs, output_shape=[n_classes + 1], is_training=is_training,
                                   reuse=True)


def image_similarity(x, y):
    return tf.nn.l2_loss(x - y)


def average_image_similarity(x, ys):
    return tf.reduce_mean(tf.map_fn(lambda y: image_similarity(x, y), ys, dtype=tf.float32))


image_similarity_loss = -args.image_similarity_decay * tf.reduce_mean([
    average_image_similarity(generator.outputs[i], generator.outputs) for i in range(args.batch_size)
])
generator_loss = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=fake_discriminator.logits, labels=tf.argmax(conditional_inputs, axis=1)
    )
) + image_similarity_loss
discriminator_loss_real = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=real_discriminator.logits, labels=tf.argmax(conditional_inputs, axis=1)
    )
)
discriminator_loss_fake = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=fake_discriminator.logits, labels=tf.fill([args.batch_size], n_classes)
    )
)
discriminator_loss = discriminator_loss_real + discriminator_loss_fake

generator_variables = [var for var in tf.trainable_variables() if 'Generator' in var.name]
discriminator_variables = [var for var in tf.trainable_variables() if 'Discriminator' in var.name]

generator_weights = [var for var in generator_variables if 'kernel' in var.name]
discriminator_weights = [var for var in discriminator_variables if 'kernel' in var.name]

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    generator_train_step = tf.train.AdamOptimizer(args.learning_rate, beta1=0.5).\
        minimize(generator_loss, var_list=generator_variables)
    discriminator_train_step = tf.train.AdamOptimizer(args.learning_rate, beta1=0.5).\
        minimize(discriminator_loss, var_list=discriminator_variables)

logging.info('Training model...')

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    n_epochs = args.iterations // args.evaluation_step

    for i in range(n_epochs):
        logging.info('Processing epoch %d/%d...' % (i + 1, n_epochs))

        generator_losses = []
        discriminator_losses = []
        image_similarity_losses = []

        for _ in tqdm(range(args.evaluation_step)):
            batch_images, batch_labels = dataset.batch()
            batch_noise = np.random.uniform(-1, 1, [args.batch_size, 1, 1, args.z_shape]).astype(np.float32)

            feed_dict = {
                generator.inputs: batch_noise,
                real_discriminator.inputs: batch_images,
                is_training: True,
                conditional_inputs: one_hot(batch_labels, n_classes)
            }

            _, batch_discriminator_loss = session.run(
                [discriminator_train_step, discriminator_loss], feed_dict=feed_dict
            )
            discriminator_losses.append(batch_discriminator_loss)

            for _ in range(2):
                _, batch_generator_loss, batch_image_similarity_loss = session.run(
                    [generator_train_step, generator_loss, image_similarity_loss], feed_dict=feed_dict
                )

                generator_losses.append(batch_generator_loss)
                image_similarity_losses.append(batch_image_similarity_loss)

        logging.info('Observed generator loss = %.4f, discriminator loss = %.4f, image similarity loss = %.4f..' %
                     (float(np.mean(generator_losses)), float(np.mean(discriminator_losses)), float(np.mean(image_similarity_losses))))
        logging.info('Generating samples...')

        for cls in tqdm(range(n_classes)):
            generated_images = session.run([generator.outputs], feed_dict={
                generator.inputs: np.random.uniform(-1, 1, [args.batch_size, 1, 1, args.z_shape]).astype(np.float32),
                conditional_inputs: one_hot(np.repeat(cls, args.batch_size), n_classes)
            })[0]

            generated_images /= 2.0
            generated_images += 0.5
            generated_images *= 255.0
            generated_images = np.clip(generated_images, 0, 255)
            generated_images = generated_images.astype(np.uint8)

            epoch_samples_path = SAMPLES_PATH / dataset.name / ('epoch_%.5d' % (i + 1)) / str(cls)
            epoch_samples_path.mkdir(parents=True, exist_ok=True)

            for j in range(len(generated_images)):
                imageio.imwrite(str(epoch_samples_path / ('%.5d.png' % (j + 1))), generated_images[j])

    logging.info('Generating final images...')

    for cls in range(n_classes):
        logging.info('Generating final images for class %d...' % cls)

        for i in tqdm(range(args.generated_images_per_class // args.batch_size)):
            generated_images = session.run([generator.outputs], feed_dict={
                generator.inputs: np.random.uniform(-1, 1, [args.batch_size, 1, 1, args.z_shape]).astype(np.float32),
                conditional_inputs: one_hot(np.repeat(cls, args.batch_size), n_classes)
            })[0]

            generated_images /= 2.0
            generated_images += 0.5
            generated_images *= 255.0
            generated_images = np.clip(generated_images, 0, 255)
            generated_images = generated_images.astype(np.uint8)

            epoch_samples_path = GENERATED_DATA_PATH / dataset.name / str(cls)
            epoch_samples_path.mkdir(parents=True, exist_ok=True)

            for j in range(len(generated_images)):
                imageio.imwrite(str(epoch_samples_path / ('%.5d.png' % (i * args.batch_size + j + 1))), generated_images[j])
