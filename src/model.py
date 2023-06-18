#! /usr/bin/python 
#-*-encoding:cp936-*-
############################################
#
# Author: liuanan
# Create time: 2023-04-16 18:48:18
# E-Mail: liuanan123@qq.com
# version 1.1
#
############################################

import tensorflow as tf
import numpy as np
from layer import *

class ResNet(tf.keras.Model):
    def __init__(self, policy_units, num_res_blocks=19):
        super().__init__()
        #self.policy = tf.keras.mixed_precision.Policy('mixed_float16')
        #tf.keras.mixed_precision.set_global_policy(self.policy)
        self.l2 = 3e-5
        self.num_res_blocks = num_res_blocks
        self.activation = 'relu'
        self.input_conv_bn = ConvBN(filters=256,
                                    kernel_size=(3,3),
                                    strides=(1,1),
                                    chan_dim=-1,
                                    activation=self.activation,
                                    regularizer=self.l2)

        self.res_blocks = []
        for i in range(num_res_blocks):
            res_block = ResBlock(filters=256,
                                 kernel_size=(3,3),
                                 strides=(1, 1),
                                 chan_dim=-1,
                                 activation=self.activation,
                                 regularizer=self.l2)
            self.res_blocks.append(res_block)

        self.policy_head_conv = ConvBN(filters=32,
                                       kernel_size=(1,1),
                                       strides=(1,1),
                                       chan_dim=-1,
                                       activation=self.activation,
                                       regularizer=self.l2)
        self.value_head_conv  = ConvBN(filters=32,
                                       kernel_size=(1,1),
                                       strides=(1,1),
                                       chan_dim=-1,
                                       activation=self.activation,
                                       regularizer=self.l2)
        self.policy_flatten = tf.keras.layers.Flatten()
        self.value_flatten = tf.keras.layers.Flatten()
        self.policy_dense = tf.keras.layers.Dense(units=policy_units,
                                                  kernel_regularizer=tf.keras.regularizers.L2(self.l2),
                                                  bias_regularizer=tf.keras.regularizers.L2(self.l2),
                                                  dtype='float32')
        self.value_dense1 = tf.keras.layers.Dense(units=256,
                                                  activation=self.activation,
                                                  kernel_regularizer=tf.keras.regularizers.L2(self.l2),
                                                  bias_regularizer=tf.keras.regularizers.L2(self.l2),
                                                  dtype='float32')

        self.value_dense2 = tf.keras.layers.Dense(units=1,
                                                  activation='softsign',
                                                  use_bias=True,
                                                  kernel_regularizer=tf.keras.regularizers.L2(self.l2),
                                                  bias_regularizer=tf.keras.regularizers.L2(self.l2),
                                                  dtype='float32')

    def call(self, inputs):
        x = self.input_conv_bn(inputs)
        for i in range(self.num_res_blocks):
            x = self.res_blocks[i](x)
        policy_logic = self.policy_dense(self.policy_flatten(self.policy_head_conv(x)))
        value = self.value_dense1(self.value_flatten(self.value_head_conv(x)))
        value = self.value_dense2(value)
        return policy_logic, value

