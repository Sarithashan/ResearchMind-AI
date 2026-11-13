"""Wrap a model in tf.function so generating summaries is fast."""
import tensorflow as tf


class FastModel:
    def __init__(self, model):
        self._f = tf.function(lambda e, d: model([e, d], training=False))

    def __call__(self, inputs, training=False):
        return self._f(tf.constant(inputs[0]), tf.constant(inputs[1]))