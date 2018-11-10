import argparse
import imageio
import logging
import numpy as np
import os
import tensorflow as tf

from containers import STL10Container
from models import Discriminator, Generator
from pathlib import Path
from tqdm import tqdm


SAMPLES_PATH = Path(__file__).parent / 'samples'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument('-batch_size', type=int, default=50)
parser.add_argument('-evaluation_step', type=int, default=100)
parser.add_argument('-iterations', type=int, default=15000)
parser.add_argument('-learning_rate', type=float, default=0.0002)

args = parser.parse_args()

logging.info('Loading data...')

dataset = STL10Container('train', args.batch_size)

logging.info('Constructing model...')

generator = Generator([100])
discriminator = Discriminator([96, 96, 3])
combined_gen_dis = Discriminator(inputs=generator.outputs, reuse=True)

generator_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(
        logits=combined_gen_dis.outputs, labels=tf.ones_like(combined_gen_dis.outputs)
    )
)
discriminator_loss_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(
        logits=discriminator.outputs, labels=tf.ones_like(discriminator.outputs)
    )
)
discriminator_loss_fake = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(
        logits=combined_gen_dis.outputs, labels=tf.zeros_like(combined_gen_dis.outputs)
    )
)
discriminator_loss = discriminator_loss_real + discriminator_loss_fake

generator_variables = [var for var in tf.trainable_variables() if 'Generator' in var.name]
discriminator_variables = [var for var in tf.trainable_variables() if 'Discriminator' in var.name]

generator_train_step = tf.train.AdamOptimizer(args.learning_rate).\
    minimize(generator_loss, var_list=generator_variables)
discriminator_train_step = tf.train.AdamOptimizer(args.learning_rate).\
    minimize(discriminator_loss, var_list=discriminator_variables)

logging.info('Training model...')

SAMPLES_PATH.mkdir(parents=True, exist_ok=True)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    n_epochs = args.iterations // args.evaluation_step

    for i in range(n_epochs):
        logging.info('Processing epoch %d/%d...' % (i + 1, n_epochs))

        generator_losses = []
        discriminator_losses = []

        for _ in tqdm(range(args.evaluation_step)):
            batch_images, batch_labels = dataset.batch()
            batch_noise = np.random.uniform(-1, 1, [args.batch_size, 100]).astype(np.float32)

            feed_dict = {
                discriminator.inputs: batch_images,
                generator.inputs: batch_noise,
                discriminator.is_training: True,
                generator.is_training: True
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
            generator.inputs: np.random.uniform(-1, 1, [args.batch_size, 100]).astype(np.float32)
        })[0]

        generated_images /= 2.0
        generated_images += 0.5
        generated_images *= 255.0
        generated_images = np.clip(generated_images, 0, 255)
        generated_images = generated_images.astype(np.uint8)

        for j in range(len(generated_images)):
            imageio.imwrite(str(SAMPLES_PATH / ('%d.png' % j)), generated_images[j])
