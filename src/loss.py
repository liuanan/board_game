#! /usr/bin/python 
#-*-encoding:cp936-*-
############################################
#
# Author: liuanan
# Create time: 2023-04-16 21:00:29
# E-Mail: liuanan123@qq.com
# version 1.1
#
############################################

import tensorflow as tf

class RLLoss(tf.keras.losses.Loss):
    def __init__(self, policy_weight=1., value_weight=1., name='RLLoss'):
        super().__init__(name=name)
        self.policy_weight = policy_weight
        self.value_weight = value_weight

    def call(self, y_true, y_pred):
        policy_loss, value_loss = RLLoss.compare_fun(y_true, y_pred)
        return policy_loss * policy_weight + value_loss * value_weight

    @staticmethod
    def compare_fun(y_true, y_pred):
        policy_probs_true, value_true = y_true
        policy_logic_pre, value_pre = y_pre
        policy_loss = tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(policy_probs_true, policy_logic_pre))
        value_loss  = tf.math.reduce_mean(tf.math.square(value_pre - value_true))

        return policy_loss, value_loss

