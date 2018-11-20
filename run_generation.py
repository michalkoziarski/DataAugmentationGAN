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


SAMPLES_PATH = Path(__file__).parent / 'samples'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument('-batch_size', type=int, default=100)
parser.add_argument('-dataset', type=str, choices=['cifar10', 'mnist', 'stl10'], required=True)
parser.add_argument('-evaluation_step', type=int, default=50)
parser.add_argument('-iterations', type=int, default=7500)
parser.add_argument('-learning_rate', type=float, default=0.0002)

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

dataset = Container('train', args.batch_size, image_size=[64, 64], greyscale=True)

logging.info('Constructing model...')

generator = Generator([1, 1, 100])
real_discriminator = Discriminator([64, 64, 1])
fake_discriminator = Discriminator(inputs=generator.outputs, reuse=True)

generator_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(
        logits=fake_discriminator.logits, labels=tf.ones([args.batch_size, 1, 1, 1])
    )
)
discriminator_loss_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(
        logits=real_discriminator.logits, labels=tf.ones([args.batch_size, 1, 1, 1])
    )
)
discriminator_loss_fake = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(
        logits=fake_discriminator.logits, labels=tf.zeros([args.batch_size, 1, 1, 1])
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

        for _ in tqdm(range(args.evaluation_step)):
            batch_images, batch_labels = dataset.batch()
            batch_noise = np.random.uniform(-1, 1, [args.batch_size, 1, 1, 100]).astype(np.float32)

            feed_dict = {
                generator.inputs: batch_noise,
                real_discriminator.inputs: batch_images,
                generator.is_training: True,
                real_discriminator.is_training: True,
                fake_discriminator.is_training: True
            }

            _, batch_discriminator_loss = session.run(
                [discriminator_train_step, discriminator_loss], feed_dict=feed_dict
            )
            discriminator_losses.append(batch_discriminator_loss)

            for _ in range(2):
                _, batch_generator_loss = session.run(
                    [generator_train_step, generator_loss], feed_dict=feed_dict
                )

                generator_losses.append(batch_generator_loss)

        logging.info('Observed generator loss = %.4f and discriminator loss = %.4f.' %
                     (np.mean(generator_losses), np.mean(discriminator_losses)))
        logging.info('Generating images...')

        generated_images = session.run([generator.outputs], feed_dict={
            generator.inputs: np.random.uniform(-1, 1, [args.batch_size, 1, 1, 100]).astype(np.float32)
        })[0]

        generated_images /= 2.0
        generated_images += 0.5
        generated_images *= 255.0
        generated_images = np.clip(generated_images, 0, 255)
        generated_images = generated_images.astype(np.uint8)

        epoch_samples_path = SAMPLES_PATH / ('epoch_%.4d' % (i + 1))
        epoch_samples_path.mkdir(parents=True, exist_ok=True)

        for j in range(len(generated_images)):
            imageio.imwrite(str(epoch_samples_path / ('%d.png' % j)), generated_images[j])
