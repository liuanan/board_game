#! /usr/bin/python 
#-*-encoding:cp936-*-
############################################
#
# Author: liuanan
# Create time: 2023-04-16 19:30:58
# E-Mail: liuanan123@qq.com
# version 1.1
#
############################################

import tensorflow as tf

class ConvBN(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides,
                 chan_dim,
                 use_bias=True,
                 padding='same',
                 activation='relu',
                 kernel_initializer='glorot_uniform',
                 regularizer=1e-4):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.chan_dim = chan_dim
        self.use_bias = use_bias
        self.padding = padding
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.regularizer = regularizer

    def build(self, inputs_shape):
        self.conv = tf.keras.layers.Conv2D(self.filters,
                                           self.kernel_size,
                                           self.strides,
                                           self.padding,
                                           use_bias=self.use_bias,
                                           kernel_initializer=self.kernel_initializer,
                                           kernel_regularizer=tf.keras.regularizers.L2(self.regularizer),
                                           bias_regularizer=tf.keras.regularizers.L2(self.regularizer),
                                           input_shape=inputs_shape[1:])
        self.bn = tf.keras.layers.BatchNormalization(axis=self.chan_dim,
                                                     momentum=0.9,
                                                     epsilon=2e-5)
        self.act = None
        if self.activation != None:
           self.act =  tf.keras.layers.Activation(self.activation)
        super().build(inputs_shape)

    def call(self, inputs):
        y = self.conv(inputs)
        y = self.bn(y)
        if self.act != None:
            y = self.act(y)
        return y

class ResBlock(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides,
                 chan_dim,
                 use_bias=True,
                 padding='same',
                 activation='relu',
                 kernel_initializer='glorot_uniform',
                 regularizer=1e-4):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.chan_dim = chan_dim
        self.use_bias = use_bias
        self.padding = padding
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.regularizer = regularizer

    def build(self, inputs_shape):
        self.conv_bn_1 = ConvBN(filters=self.filters,
                                kernel_size=self.kernel_size,
                                strides=self.strides,
                                chan_dim=self.chan_dim,
                                use_bias=self.use_bias,
                                padding=self.padding,
                                activation=self.activation,
                                kernel_initializer=self.kernel_initializer,
                                regularizer=self.regularizer)
        self.conv_bn_2 = ConvBN(filters=self.filters,
                                kernel_size=self.kernel_size,
                                strides=self.strides,
                                chan_dim=self.chan_dim,
                                use_bias=self.use_bias,
                                padding=self.padding,
                                activation=None,
                                kernel_initializer=self.kernel_initializer,
                                regularizer=self.regularizer)       
        self.act = tf.keras.layers.Activation(self.activation)
        super().build(inputs_shape)

    def call(self, inputs):
        y = self.conv_bn_1(inputs)
        y = self.conv_bn_2(y)
        y += inputs
        return self.act(y)

