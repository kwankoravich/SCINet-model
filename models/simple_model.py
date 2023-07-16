import tensorflow as tf
from SCINet import SCINet, StackedSCINet
from ..utils.tools import Identity
from ..utils.loss import StackedSCINetLoss

def make_simple_scinet(input_shape, horizon: int, L: int, h: int, kernel_size: int, learning_rate: float,
                       kernel_regularizer=None, activity_regularizer=None, diagram_path=None):
    """Compiles a simple SCINet and saves model diagram if given a path.

    Intended to be a demonstration of simple model construction. See paper for details on the hyperparameters.
    """
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(input_shape[1], input_shape[2]), name='inputs'),
        SCINet(horizon, features=input_shape[-1], levels=L, h=h, kernel_size=kernel_size,
               kernel_regularizer=kernel_regularizer, activity_regularizer=activity_regularizer)
    ])

    model.summary()
    if diagram_path:
        tf.keras.utils.plot_model(model, to_file=diagram_path, show_shapes=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse',
                  metrics=['mse', 'mae']
                  )

    return model

def make_simple_stacked_scinet(input_shape, horizon: int, K: int, L: int, h: int, kernel_size: int,
                               learning_rate: float, kernel_regularizer=None, activity_regularizer=None,
                               diagram_path=None):
    """Compiles a simple StackedSCINet and saves model diagram if given a path.

    Intended to be a demonstration of simple model construction. See paper for details on the hyperparameters.
    """
    inputs = tf.keras.Input(shape=(input_shape[1], input_shape[2]), name='lookback_window')
    x = StackedSCINet(horizon=horizon, features=input_shape[-1], stacks=K, levels=L, h=h,
                      kernel_size=kernel_size, kernel_regularizer=kernel_regularizer,
                      activity_regularizer=activity_regularizer)(inputs)
    outputs = Identity(name='outputs')(x[-1])
    intermediates = Identity(name='intermediates')(x)
    model = tf.keras.Model(inputs=inputs, outputs=[outputs, intermediates])

    model.summary()
    if diagram_path:
        tf.keras.utils.plot_model(model, to_file=diagram_path, show_shapes=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss={
                      # 'outputs': 'mse',
                      'intermediates': StackedSCINetLoss()
                  },
                  metrics={'outputs': ['mse', 'mae']}
                  )

    return model