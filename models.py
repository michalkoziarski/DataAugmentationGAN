import inspect
import tensorflow as tf

from abc import ABC, abstractmethod


class AbstractModel(ABC):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.inputs = tf.placeholder(tf.float32, shape=[None] + list(input_shape))
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
