import tensorflow as tf

class StackedSCINetLoss(tf.keras.losses.Loss):
    """Compute loss for a Stacked SCINet via intermediate supervision.

    `loss = sum of mean normalised difference between each stack's output and ground truth`

    `y_pred` should be the output of a StackedSCINet layer.
    """

    def __init__(self, name='stacked_scienet_loss', **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        stacked_outputs = y_pred
        horizon = stacked_outputs.shape[2]

        errors = stacked_outputs - y_true
        loss = tf.linalg.normalize(errors, axis=3)[1]
        loss = tf.reduce_sum(loss, 2)
        loss /= horizon
        loss = tf.reduce_sum(loss)

        return loss