from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

# Fallback implementation for GDN (Generalized Divisive Normalization)
class GDN(tf.keras.layers.Layer):
    def __init__(self, inverse=False, beta_min=1e-6, gamma_init=0.1, **kwargs):
        super(GDN, self).__init__(**kwargs)
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
    
    def build(self, input_shape):
        self.beta = self.add_weight(
            name='beta',
            shape=(input_shape[-1],),
            initializer=tf.constant_initializer(1.0),
            trainable=True
        )
        self.gamma = self.add_weight(
            name='gamma',
            shape=(input_shape[-1],),
            initializer=tf.constant_initializer(self.gamma_init),
            trainable=True
        )
        super(GDN, self).build(input_shape)
    
    def call(self, x):
        beta = tf.maximum(self.beta, self.beta_min)
        if self.inverse:
            return x * beta
        else:
            return x / (beta + self.gamma * tf.abs(x))

# Simplified SignalConv2D using standard Conv2D
def SimpleConv2D(filters, kernel_size, strides_down=1, strides_up=1, 
                 padding="same", use_bias=True, activation=None):
    """Simplified convolution layer"""
    if strides_down > 1:
        return tf.layers.Conv2D(filters, kernel_size, strides=(strides_down, strides_down),
                                padding=padding, use_bias=use_bias, activation=activation)
    elif strides_up > 1:
        # Transposed convolution for upsampling
        return tf.layers.Conv2DTranspose(filters, kernel_size, strides=(strides_up, strides_up),
                                         padding=padding, use_bias=use_bias, activation=activation)
    else:
        return tf.layers.Conv2D(filters, kernel_size, padding=padding,
                                use_bias=use_bias, activation=activation)

def MV_analysis(tensor, num_filters, M):
  """Builds the analysis transform."""

  with tf.variable_scope("MV_analysis"):
    with tf.variable_scope("layer_0"):
      tensor = tf.layers.conv2d(tensor, num_filters, 3, strides=2, padding="same",
                               activation=tf.nn.relu)

    with tf.variable_scope("layer_1"):
      tensor = tf.layers.conv2d(tensor, num_filters, 3, strides=2, padding="same",
                               activation=tf.nn.relu)

    with tf.variable_scope("layer_2"):
      tensor = tf.layers.conv2d(tensor, num_filters, 3, strides=2, padding="same",
                               activation=tf.nn.relu)

    with tf.variable_scope("layer_3"):
      tensor = tf.layers.conv2d(tensor, M, 3, strides=2, padding="same", activation=None)

    return tensor


def MV_synthesis(tensor, num_filters, out_filters=2):
  """Builds the synthesis transform."""

  with tf.variable_scope("MV_synthesis"):
    with tf.variable_scope("layer_0"):
      tensor = tf.layers.conv2d_transpose(tensor, num_filters, 3, strides=2, padding="same",
                                         activation=tf.nn.relu)

    with tf.variable_scope("layer_1"):
      tensor = tf.layers.conv2d_transpose(tensor, num_filters, 3, strides=2, padding="same",
                                         activation=tf.nn.relu)

    with tf.variable_scope("layer_2"):
      tensor = tf.layers.conv2d_transpose(tensor, num_filters, 3, strides=2, padding="same",
                                         activation=tf.nn.relu)

    with tf.variable_scope("layer_3"):
      tensor = tf.layers.conv2d_transpose(tensor, out_filters, 3, strides=2, padding="same",
                                         activation=None)

    return tensor

def Res_analysis(tensor, num_filters, M, reuse=False):
  """Builds the analysis transform."""

  with tf.variable_scope("analysis", reuse=reuse):
    with tf.variable_scope("layer_0"):
      tensor = tf.layers.conv2d(tensor, num_filters, 5, strides=2, padding="same",
                               activation=tf.nn.relu)

    with tf.variable_scope("layer_1"):
      tensor = tf.layers.conv2d(tensor, num_filters, 5, strides=2, padding="same",
                               activation=tf.nn.relu)

    with tf.variable_scope("layer_2"):
      tensor = tf.layers.conv2d(tensor, num_filters, 5, strides=2, padding="same",
                               activation=tf.nn.relu)

    with tf.variable_scope("layer_3"):
      tensor = tf.layers.conv2d(tensor, M, 5, strides=2, padding="same", activation=None)

    return tensor

def Res_synthesis(tensor, num_filters, reuse=False):
  """Builds the synthesis transform."""

  with tf.variable_scope("synthesis", reuse=reuse):
    with tf.variable_scope("layer_0"):
      tensor = tf.layers.conv2d_transpose(tensor, num_filters, 5, strides=2, padding="same",
                                         activation=tf.nn.relu)

    with tf.variable_scope("layer_1"):
      tensor = tf.layers.conv2d_transpose(tensor, num_filters, 5, strides=2, padding="same",
                                         activation=tf.nn.relu)

    with tf.variable_scope("layer_2"):
      tensor = tf.layers.conv2d_transpose(tensor, num_filters, 5, strides=2, padding="same",
                                         activation=tf.nn.relu)

    with tf.variable_scope("layer_3"):
      tensor = tf.layers.conv2d_transpose(tensor, 3, 5, strides=2, padding="same", activation=None)

    return tensor