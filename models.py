import inspect
import tensorflow as tf


class CNN:
    def __init__(self, input_shape, n_classes):
        self.input_shape = input_shape
        self.n_classes = n_classes

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

        self.add(tf.layers.Dense(self.n_classes, activation=tf.nn.relu))
