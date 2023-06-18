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

import os
import time
import tensorflow as tf
import numpy as np
from layer import *
from model import *
from trainer import *

def test():
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    physical_devices = tf.config.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(physical_devices[0], True)
    model = ResNet(num_res_blocks=0, policy_units=9*9+1)
    loss_fn = [tf.keras.losses.BinaryCrossentropy(from_logits=True), tf.keras.losses.MeanSquaredError()]
    weights = [1., 1.]
    optimizer = tf.keras.optimizers.SGD(momentum=0.9)
    #optimizer = tf.keras.optimizers.Adam()
    #model.compile(optimizer=optimizer,
    #              loss=loss_fn,
    #              loss_weights=weights)

    batch_sizes = [1, 8, 24, 32, 48, 64, 96, 128]
    batch_sizes = [512]
    inputs = np.ones([33, 9, 9, 17], dtype=np.float32)
    logic, v = model(inputs)
    for batch_size in batch_sizes:
        begin = time.perf_counter()
        test_cnt = 1
        for _ in range(test_cnt):
            inputs = np.where(np.random.random([batch_size, 9, 9, 17]) >= 0.7, 1., 0.)
            labels = Trainer.fake_predictor(inputs)
            #model.fit(inputs, labels)
            print('=================')
            logic, v = model(inputs, training=False)
            logic = logic.numpy()
            v = v.numpy()
            logic_mean = np.mean(logic, axis=0)
            v_mean = np.mean(v, axis=0)
            print(f"value:\n{v}")
            print(f"value mean:\n{v_mean}")
            print(f"logic mean:\n{logic_mean}")
        batch_time =  (time.perf_counter() - begin) * 1000 / test_cnt
        per_instance_time = batch_time / batch_size

        print(f"batch_size={batch_size} batch_time={batch_time:.6f}ms per_instance_time={per_instance_time:.6f}ms")
    #print(model.trainable_variables)
    #print(model.layers)
    #print(model.policy.compute_dtype)
    #print(model.policy.variable_dtype)
    #model.save('model')
if __name__ == '__main__':
   test() 
