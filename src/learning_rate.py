#! /usr/bin/python 
#-*-encoding:cp936-*-
############################################
#
# Author: liuanan
# Create time: 2023-04-29 11:54:21
# E-Mail: liuanan123@qq.com
# version 1.1
#
############################################

import tensorflow as tf

class RLLearningRate():
    @staticmethod
    def lr_callback(training_step, lr):
        if training_step < 400000:
            new_lr = 1e-2
        elif training_step < 600000:
            new_lr = 1e-3
        else:
            new_lr = 1e-4
        print(f"lr: {new_lr}")
        return new_lr
