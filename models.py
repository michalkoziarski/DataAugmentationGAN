import inspect
import tensorflow as tf

from abc import ABC, abstractmethod


class AbstractModel(ABC):
    def __init__(self, input_shape=None, output_shape=None, inputs=None, reuse=False):
        assert not (input_shape is None and inputs is None)

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.reuse = reuse

        if inputs is None:
            self.inputs = tf.placeholder(tf.float32, shape=[None] + list(input_shape))
        else:
            self.inputs = inputs

        self.outputs = self.inputs
        self.layers = []
        self.is_training = tf.placeholder_with_default(False, [])

        self.setup()

    def add(self, layer):
        self.layers.append(layer)

        if 'training' in inspect.getfullargspec(layer.call).args:
            self.outputs = layer(self.outputs, training=self.is_training)
        else:
            self.outputs = layer(self.outputs)

    @abstractmethod
    def setup(self):
        pass


class Classifier(AbstractModel):
    def setup(self):
        with tf.variable_scope('Classifier', reuse=self.reuse):
            self.add(tf.layers.Conv2D(64, 3, padding='same', activation=tf.nn.relu))
            self.add(tf.layers.Conv2D(64, 3, padding='same', activation=tf.nn.relu))
            self.add(tf.layers.MaxPooling2D(2, 2, padding='same'))

            self.add(tf.layers.Conv2D(128, 3, padding='same', activation=tf.nn.relu))
            self.add(tf.layers.Conv2D(128, 3, padding='same', activation=tf.nn.relu))
            self.add(tf.layers.MaxPooling2D(2, 2, padding='same'))

            self.add(tf.layers.Conv2D(256, 3, padding='same', activation=tf.nn.relu))
            self.add(tf.layers.Conv2D(256, 3, padding='same', activation=tf.nn.relu))
            self.add(tf.layers.MaxPooling2D(2, 2, padding='same'))

            self.add(tf.layers.Conv2D(512, 3, padding='same', activation=tf.nn.relu))
            self.add(tf.layers.Conv2D(512, 3, padding='same', activation=tf.nn.relu))
            self.add(tf.layers.MaxPooling2D(2, 2, padding='same'))

            self.add(tf.layers.Flatten())

            self.add(tf.layers.Dense(1024, activation=tf.nn.relu))
            self.add(tf.layers.Dropout(0.5))

            self.add(tf.layers.Dense(1024, activation=tf.nn.relu))
            self.add(tf.layers.Dropout(0.5))

            self.add(tf.layers.Dense(self.output_shape[0], activation=tf.nn.relu))


class Discriminator(AbstractModel):
    def setup(self):
        with tf.variable_scope('Discriminator', reuse=self.reuse):
            self.add(tf.layers.Conv2D(64, 5, 2, 'same', activation=tf.nn.leaky_relu, use_bias=False))
            self.add(tf.layers.BatchNormalization(momentum=0.9))
            self.outputs = tf.nn.relu(self.outputs)

            self.add(tf.layers.Conv2D(128, 5, 2, 'same', activation=tf.nn.leaky_relu, use_bias=False))
            self.add(tf.layers.BatchNormalization(momentum=0.9))
            self.outputs = tf.nn.relu(self.outputs)

            self.add(tf.layers.Conv2D(256, 5, 2, 'same', activation=tf.nn.leaky_relu, use_bias=False))
            self.add(tf.layers.BatchNormalization(momentum=0.9))
            self.outputs = tf.nn.relu(self.outputs)

            self.add(tf.layers.Conv2D(512, 5, 1, 'same', activation=tf.nn.leaky_relu, use_bias=False))
            self.add(tf.layers.BatchNormalization(momentum=0.9))
            self.outputs = tf.nn.relu(self.outputs)

            self.add(tf.layers.Flatten())

            self.add(tf.layers.Dense(1, activation=tf.nn.sigmoid))


class Generator(AbstractModel):
    def setup(self):
        with tf.variable_scope('Generator', reuse=self.reuse):
            self.add(tf.layers.Dense(1024 * 4 * 4, use_bias=False))
            self.add(tf.layers.BatchNormalization(momentum=0.9))
            self.outputs = tf.nn.relu(self.outputs)

            self.outputs = tf.reshape(self.outputs, (-1, 4, 4, 1024))

            self.add(tf.layers.Conv2DTranspose(512, 5, 2, 'same', use_bias=False))
            self.add(tf.layers.BatchNormalization(momentum=0.9))
            self.outputs = tf.nn.relu(self.outputs)

            self.add(tf.layers.Conv2DTranspose(256, 5, 2, 'same', use_bias=False))
            self.add(tf.layers.BatchNormalization(momentum=0.9))
            self.outputs = tf.nn.relu(self.outputs)

            self.add(tf.layers.Conv2DTranspose(128, 5, 2, 'same', use_bias=False))
            self.add(tf.layers.BatchNormalization(momentum=0.9))
            self.outputs = tf.nn.relu(self.outputs)

            self.add(tf.layers.Conv2DTranspose(3, 5, 3, 'same', activation=tf.nn.tanh))
