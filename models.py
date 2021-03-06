import inspect
import tensorflow as tf

from abc import ABC, abstractmethod


class AbstractModel(ABC):
    def __init__(self, input_shape=None, output_shape=None, inputs=None, reuse=False, is_training=None):
        assert not (input_shape is None and inputs is None)

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.reuse = reuse

        if inputs is None:
            self.inputs = tf.placeholder(tf.float32, shape=[None] + list(input_shape))
        else:
            self.inputs = inputs

        if is_training is None:
            self.is_training = tf.placeholder_with_default(False, [])
        else:
            self.is_training = is_training

        self.outputs = self.inputs
        self.logits = None
        self.layers = []

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
            self.add(tf.layers.Conv2D(128, 4, 2, 'same', use_bias=False))
            self.add(tf.layers.BatchNormalization())
            self.outputs = tf.nn.leaky_relu(self.outputs)

            self.add(tf.layers.Conv2D(256, 4, 2, 'same', use_bias=False))
            self.add(tf.layers.BatchNormalization())
            self.outputs = tf.nn.leaky_relu(self.outputs)

            self.add(tf.layers.Conv2D(512, 4, 2, 'same', use_bias=False))
            self.add(tf.layers.BatchNormalization())
            self.outputs = tf.nn.leaky_relu(self.outputs)

            self.add(tf.layers.Conv2D(1024, 4, 2, 'same', use_bias=False))
            self.add(tf.layers.BatchNormalization())
            self.outputs = tf.nn.leaky_relu(self.outputs)

            self.add(tf.layers.Conv2D(self.output_shape[0], 4, 1, 'valid'))
            self.outputs = tf.reshape(self.outputs, [-1, self.output_shape[0]])
            self.logits = self.outputs

            self.outputs = tf.nn.softmax(self.outputs)


class Generator(AbstractModel):
    def __init__(self, input_shape=None, output_shape=None, inputs=None, reuse=False, is_training=None,
                 conditional_inputs=None):
        self.conditional_inputs = conditional_inputs

        super().__init__(input_shape, output_shape, inputs, reuse, is_training)

    def setup(self):
        with tf.variable_scope('Generator', reuse=self.reuse):
            if self.conditional_inputs is not None:
                self.outputs = tf.concat([
                    self.outputs,
                    tf.reshape(self.conditional_inputs, [-1, 1, 1, int(self.conditional_inputs.shape[1])])
                ], -1)

            self.add(tf.layers.Conv2DTranspose(1024, 4, 1, 'valid', use_bias=False))
            self.add(tf.layers.BatchNormalization())
            self.outputs = tf.nn.leaky_relu(self.outputs)

            self.add(tf.layers.Conv2DTranspose(512, 4, 2, 'same', use_bias=False))
            self.add(tf.layers.BatchNormalization())
            self.outputs = tf.nn.leaky_relu(self.outputs)

            self.add(tf.layers.Conv2DTranspose(256, 4, 2, 'same', use_bias=False))
            self.add(tf.layers.BatchNormalization())
            self.outputs = tf.nn.leaky_relu(self.outputs)

            self.add(tf.layers.Conv2DTranspose(128, 4, 2, 'same', use_bias=False))
            self.add(tf.layers.BatchNormalization())
            self.outputs = tf.nn.leaky_relu(self.outputs)

            self.add(tf.layers.Conv2DTranspose(3, 4, 2, 'same', activation=tf.nn.tanh))
