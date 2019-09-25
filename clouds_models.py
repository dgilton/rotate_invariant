#
# + cleaning version: TO BE trian.py
#
# + History
# 09/10 Regulate rotation angle only {0, 120, 240}
#


import tensorflow as tf
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model, Sequential
import tensorflow.contrib.slim as slim
  

def fully_connected_model(shape=(128,128,6)) :
    """
      Reference: https://blog.keras.io/building-autoencoders-in-keras.html
    """
    ## start construction

    x = inp = Input(shape=shape, name='encoding_input')
    #x = Conv2D(filters=1, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)

    x = Flatten()(x)
    x = Dense(500)(x)
    x = ReLU()(x)
    # x = Dense(300)(x)
    # x = ReLU()(x)
    x = Dense(128)(x)
    x = ReLU()(x)
    #x = tf.reshape(x, shape=(-1,2,2,32))
    x = Lambda(lambda x: tf.reshape(x, shape=(-1,2,2,32)))(x)

    # build model for encoder + digit layer
    encoder = Model(inp, x, name='encoder')
             
    x = inp = Input(x.shape[1:], name="decoder_input")
    x = Flatten()(x)
    # x = Dense(300)(x)
    # x = ReLU()(x)
    x = Dense(300)(x)
    x = ReLU()(x)
    # x = Dense(500)(x)
    # x = ReLU()(x)
    x = Dense(128*128*6)(x)
    x = ReLU()(x)
    #x = tf.reshape(x, shape=(-1,28,28,1))
    x = Lambda(lambda x: tf.reshape(x, shape=(-1,128,128,6)))(x)
    decoder = Model(inp, x, name='decoder')
             
    return encoder, decoder


def channel_wise_fc_layer(input, name):  # bottom: (7x7x512)
    _, width, height, n_feat_map = input.get_shape().as_list()
    input_reshape = tf.reshape(input, [-1, width * height, n_feat_map])
    input_transpose = tf.transpose(input_reshape, [2, 0, 1])

    with tf.variable_scope(name):
        W = tf.get_variable(
            "W",
            shape=[n_feat_map, width * height, width * height],  # (512,49,49)
            initializer=tf.random_normal_initializer(0., 0.05))
        output = tf.matmul(input_transpose, W)

    output_transpose = tf.transpose(output, [1, 2, 0])
    output_reshape = tf.reshape(output_transpose, [-1, height, width, n_feat_map])

    return output_reshape

def modified_convolutional_architecture():

    def encoder(images, is_training=True):
        activation_fn = leaky_relu  # tf.nn.relu
        # activation_fn = tf.nn.elu
        weight_decay = 0.0
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.batch_norm],
                                is_training=is_training):
                with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                    weights_initializer=tf.truncated_normal_initializer(stddev=0.05),
                                    weights_regularizer=slim.l2_regularizer(weight_decay),
                                    normalizer_fn=slim.batch_norm):
                    net = slim.conv2d(images, 32, [4, 4], 1, activation_fn=activation_fn, scope='Conv2d_1')
                    net = slim.conv2d(net, 64, [4, 4], 1, activation_fn=activation_fn, scope='Conv2d_2')
                    net = slim.conv2d(net, 128, [4, 4], 2, activation_fn=activation_fn, scope='Conv2d_3')
                    net = slim.conv2d(net, 256, [4, 4], 2, activation_fn=activation_fn, scope='Conv2d_4')
                    net = slim.conv2d(net, 512, [4, 4], 2, activation_fn=activation_fn, scope='Conv2d_5')
                    net = channel_wise_fc_layer(net, name='channel_fc_1')
                    net = slim.conv2d(net, 512, [2, 2], 1, activation_fn=activation_fn, scope='Conv2d_6')

                    # net = slim.flatten(net)
                    # fc1 = slim.fully_connected(net, self.latent_variable_dim, activation_fn=None, normalizer_fn=None, scope='Fc_1')
                    # fc2 = slim.fully_connected(net, self.latent_variable_dim, activation_fn=None, normalizer_fn=None, scope='Fc_2')
        return net

    def decoder(latent_var, is_training=True):
        activation_fn = leaky_relu  # tf.nn.relu
        # activation_fn = tf.nn.elu
        weight_decay = 0.0
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.batch_norm],
                                is_training=is_training):
                with slim.arg_scope([slim.convolution2d_transpose, slim.fully_connected],
                                    weights_initializer=tf.truncated_normal_initializer(stddev=0.05),
                                    weights_regularizer=slim.l2_regularizer(weight_decay),
                                    normalizer_fn=slim.batch_norm):
                    # net = slim.fully_connected(latent_var, , activation_fn=None, normalizer_fn=None, scope='Fc_1')
                    # net = tf.reshape(net, [-1,16,16,4], name='Reshape')

                    # net = tf.image.resize_nearest_neighbor(net, size=(8,8), name='Upsample_1')
                    # net = slim.convolution2d_transpose(latent_var, 512, [4, 4], 2, activation_fn=activation_fn,
                    #                                    scope='Conv2d_1')
                    net = slim.convolution2d_transpose(latent_var, 512, [4, 4], 2, activation_fn=activation_fn,
                                                       scope='Conv2d_1')

                    # net = tf.image.resize_nearest_neighbor(net, size=(16,16), name='Upsample_2')
                    net = slim.convolution2d_transpose(net, 256, [4, 4], 2, activation_fn=activation_fn,
                                                       scope='Conv2d_2')
                    net = slim.convolution2d_transpose(net, 128, [4, 4], 2, activation_fn=activation_fn,
                                                       scope='Conv2d_3')
                    net = slim.convolution2d_transpose(net, 64, [4, 4], 1, activation_fn=activation_fn,
                                                       scope='Conv2d_4')
                    net = slim.convolution2d_transpose(net, 6, [4, 4], 1, activation_fn=activation_fn, scope='Conv2d_5')


        return net

    return encoder, decoder


def leaky_relu(x):
    return tf.maximum(0.1 * x, x)