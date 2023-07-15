import tensorflow as tf

class SCINet(tf.keras.layers.Layer):
    def __init__(self, horizon: int, features: int, levels: int, h: int, kernel_size: int,
                 kernel_regularizer=None, activity_regularizer=None, name='scinet', **kwargs):
        """
        :param horizon: number of time stamps in output
        :param levels: height of the binary tree + 1
        :param h: scaling factor for convolutional module in each SCIBlock
        :param kernel_size: kernel size of convolutional module in each SCIBlock
        :param kernel_regularizer: kernel regularizer for the fully connected layer at the end
        :param activity_regularizer: activity regularizer for the fully connected layer at the end
        """
        if levels < 1:
            raise ValueError('Must have at least 1 level')

        super().__init__(name=name, **kwargs)
        self.horizon = horizon
        self.features = features
        self.levels = levels
        self.h = h
        self.kernel_size = kernel_size

        self.interleave = Interleave()
        self.flatten = tf.keras.layers.Flatten()

        # tree of sciblocks
        self.sciblocks = [SCIBlock(features=features, kernel_size=self.kernel_size, h=self.h)
                          for _ in range(2 ** self.levels - 1)]
        self.dense = tf.keras.layers.Dense(
            self.horizon * features,
            kernel_regularizer=kernel_regularizer,
            activity_regularizer=activity_regularizer
        )

    def build(self, input_shape):
        if input_shape[1] / 2 ** self.levels % 1 != 0:
            raise ValueError(f'timestamps {input_shape[1]} must be evenly divisible by a tree with '
                             f'{self.levels} levels')
        super().build(input_shape)

    def call(self, inputs):
        # cascade input down a binary tree of sci-blocks
        lvl_inputs = [inputs]  # inputs for current level of the tree
        for i in range(self.levels):
            i_end = 2 ** (i + 1) - 1
            i_start = i_end - 2 ** i
            lvl_outputs = [output for j, tensor in zip(range(i_start, i_end), lvl_inputs)
                           for output in self.sciblocks[j](tensor)]
            lvl_inputs = lvl_outputs

        x = self.interleave(lvl_outputs)
        x += inputs

        # not sure if this is the correct way of doing it. The paper merely said to use a fully connected layer to
        # produce an output. Can't use TimeDistributed wrapper. It would force the layer's timestamps to match that of
        # the input -- something SCINet is supposed to solve
        x = self.flatten(x)
        x = self.dense(x)
        x = tf.reshape(x, (-1, self.horizon, self.features))

        return x

    def get_config(self):
        config = super().get_config()
        config.update({'horizon': self.horizon, 'levels': self.levels})
        return config