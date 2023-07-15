import tensorflow as tf


class Identity(tf.keras.layers.Layer):
    """Identity layer used solely for the purpose of naming model outputs and properly displaying outputs when plotting
    some multi-output models.

    Returns input without changing them.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.identity(inputs)