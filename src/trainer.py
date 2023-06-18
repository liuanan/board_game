#! /usr/bin/python 
#-*-encoding:cp936-*-
############################################
#
# Author: liuanan
# Create time: 2023-04-22 16:56:16
# E-Mail: liuanan123@qq.com
# version 1.1
#
############################################

import os
import sys
import collections
import random
import time
import numpy as np
import tensorflow as tf
from multiprocessing import Process, Queue, Pipe

from mcts_factory import MCTSFactory
from model import ResNet
from learning_rate import RLLearningRate

class Trainer():
    def __init__(self, total_game_num, batch_size, simulations_num, processor_num, mcts_tree_num_each_processor, from_base_model):
        #print(f"DEBUG: {__name__} line {sys._getframe().f_lineno}", flush=True)
        a = tf.config.experimental.tensor_float_32_execution_enabled()
        print(f"tf32 flag: {tf.config.experimental.tensor_float_32_execution_enabled()}")
        self.total_game_num = total_game_num    # num of total game num
        self.processor_num = processor_num
        self.mcts_tree_num_each_processor = mcts_tree_num_each_processor
        self.iter_game_num = processor_num * mcts_tree_num_each_processor       # num of game for each iterator
        self.epochs = 5
        self.batch_size = batch_size            # bs for each training iterator
        self.optimizer = tf.keras.optimizers.SGD(momentum=0.9)
        self.lr = tf.keras.callbacks.LearningRateScheduler(RLLearningRate.lr_callback)
        self.checkpoint_path = f'../model/gomoku_model/{total_game_num}/checkpoint/'
        self.save_path = f'../model/gomoku_model/{total_game_num}/saved_model/'
        self.cp = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                     save_freq=1000)
        self.simulations_num=simulations_num
        self.from_base_model = from_base_model  # from a pre train model?
        self.training_data_buffer = []
        self.validation_data_buffer = []
        if self.from_base_model:
            self.model = tf.keras.models.load_model(self.checkpoint_path)
        else:
            self.model = ResNet(policy_units=MCTSFactory.get_grid_num(), num_res_blocks=13)
        #print(f"DEBUG: {__name__} line {sys._getframe().f_lineno}", flush=True)
        #self.loss_fn = [tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM), tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)]
        self.loss_fn = [tf.keras.losses.CategoricalCrossentropy(from_logits=True), tf.keras.losses.MeanSquaredError()]
        self.metrics_fn = None  #[[tf.keras.metrics.CategoricalCrossentropy(from_logits=True)], [tf.keras.metrics.MeanSquaredError()]]
        self.weights = [1., 1.]

    def fake_predictor(features):
       return [(np.random.random((features.shape[0], MCTSFactory.get_grid_num())) - 0.5) * 10, np.random.random((features.shape[0], 1)) * 2 - 1]

    def predictor(self, features, training=False, using_fake_predictor=False):
        if using_fake_predictor:
            return Trainer.fake_predictor(features)
        else:
            #print(f"shape={features.shape}", flush=True)
            return [y.numpy() for y in self.model(features, training=training)]

    def multi_process_data_generator(self, is_generate_treaining_data, game_num, epochs, request_queue, predict_pipe_arr, processor_arr):
        game_num += self.iter_game_num - 1
        data_buffer = None
        if is_generate_treaining_data:
            data_buffer = self.training_data_buffer
        else:
            data_buffer = self.validation_data_buffer
        for i in range(0, game_num, self.iter_game_num):
            print(f"\n\n+++++++++++++++GAME:{i}-{i+self.iter_game_num-1}+++++++++++++++")
            print(f"iter {i} begin time: {time.asctime(time.localtime(time.time()))}", flush=True)
            begin = time.time()
            self.generator_data(data_buffer,
                                predict_pipe_arr,
                                request_queue,
                                processor_arr)
            print(f"\niter {i} end time: {time.asctime(time.localtime(time.time()))}")
            print(f"time: {time.time()-begin:.2f}s", flush=True)
            if len(data_buffer) >= self.batch_size * 2:
                batch_num = len(data_buffer) // self.batch_size
                print(f"generate_data: epochs={epochs}, batch_num={batch_num}, len={len(data_buffer)}", flush=True)
                data_idx = np.array(range(len(data_buffer)))
                for epoch_idx in range(epochs):
                    if is_generate_treaining_data:
                        np.random.shuffle(data_idx)
                    for batch_id in range(batch_num):
                        begin_idx = batch_id * self.batch_size
                        end_idx = begin_idx + self.batch_size
                        mini_batch_idx = data_idx[begin_idx:end_idx]
                        policy_batch = np.array([data_buffer[idx][0] for idx in mini_batch_idx])
                        value_batch = np.array([data_buffer[idx][1] for idx in mini_batch_idx])
                        features = np.array([data_buffer[idx][2] for idx in mini_batch_idx])
                        yield features, (policy_batch, value_batch)
                data_buffer.clear()
            else:
                print(f"skip iterator {i}, data_buffer={len(data_buffer)}")

    def get_single_batch(self, data_buffer):
        mini_batch = random.sample(data_buffer, self.batch_size)
        policy_batch = np.array([data[0] for data in mini_batch])
        value_batch = np.array([data[1] for data in mini_batch])
        features = np.array([data[2] for data in mini_batch])
        return features, (policy_batch, value_batch)

    def compile(self):
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss_fn,
                           loss_weights=self.weights,
                           metrics=self.metrics_fn)
    
    def fit(self, request_queue, predict_pipe_arr, processor_arr):
        self.model.fit(x=self.multi_process_data_generator(is_generate_treaining_data=True,
                                                           game_num=self.total_game_num,
                                                           epochs=self.epochs,
                                                           request_queue=request_queue,
                                                           predict_pipe_arr=predict_pipe_arr,
                                                           processor_arr=processor_arr),
                       validation_data=self.multi_process_data_generator(is_generate_treaining_data=False,
                                                                         game_num=10,
                                                                         epochs=1,
                                                                         request_queue=request_queue,
                                                                         predict_pipe_arr=predict_pipe_arr,
                                                                         processor_arr=processor_arr),
                       callbacks=[self.lr, self.cp])


    def save(self):
        self.model.save(self.save_path)

    def predict_server(self, data_buffer, predict_pipe_arr, request_queue, processor_arr, using_fake_predictor=False):
        self.generator_data(data_buffer, predict_pipe_arr, request_queue, processor_arr, using_fake_predictor)

    def generator_data(self, data_buffer, predict_pipe_arr, request_queue, processor_arr, using_fake_predictor=False):
        finish_num = 0
        unstart_num = self.iter_game_num
        work_num = len(predict_pipe_arr)
        running_process = 0

        for p in predict_pipe_arr:
            if unstart_num <= 0:
                break
            p[1].send("start")
            running_process += 1
            unstart_num -= self.mcts_tree_num_each_processor
        feature_batch = []
        work_id_arr = []
        while True:
            if len(feature_batch) > 0:
                #begin = time.perf_counter()
                np_feature_batch = np.concatenate(feature_batch)
                #batch_size = np_feature_batch.shape[0]
                #print(f"using_fake_predictor={using_fake_predictor}", flush=True)
                #print(f"np_feature_batch={np_feature_batch}", flush=True)
                y_batch = self.predictor(np_feature_batch, using_fake_predictor=using_fake_predictor)
                #batch_time =  (time.perf_counter() - begin) * 1000
                #per_instance_time = batch_time / batch_size
                #print(f"batch_size={batch_size} batch_time={batch_time:.6f}ms per_instance_time={per_instance_time:.6f}ms", flush=True)
                #print(y_batch, flush=True)
                offset = 0
                for i in range(len(work_id_arr)):
                    worker_batch_size = work_id_arr[i][1]
                    y = y_batch[0][offset:offset+worker_batch_size], y_batch[1][offset:offset+worker_batch_size]
                    offset += worker_batch_size
                    predict_pipe_arr[work_id_arr[i][0]][1].send((work_id_arr[i][0], 'predict', y))
                feature_batch.clear()
                work_id_arr.clear()
            else:
                #print("waiting for request", flush=True)
                recv_data = request_queue.get()
                #print(recv_data, flush=True)
                if recv_data[1] == 'done':
                    finish_num += self.mcts_tree_num_each_processor
                    data_buffer.extend(recv_data[2])
                    running_process -= 1
                    if unstart_num > 0:
                        predict_pipe_arr[recv_data[0]][1].send("start")
                        unstart_num -= self.mcts_tree_num_each_processor
                        running_process += 1
                elif recv_data[1] == 'predict':
                    #print(recv_data, flush=True)
                    if type(recv_data[2]) is np.ndarray:
                        #print("recv data from worker %d, shape=%s" % (recv_data[0], ','.join([str(i) for i in recv_data[2].shape])), flush=True)
                        feature_batch.append(recv_data[2])
                        work_id_arr.append((recv_data[0], recv_data[2].shape[0]))
                    else:
                        feature_batch.extend(recv_data[2])
                        work_id_arr.append((recv_data[0], len(recv_data[2])))
                    #print(feature_batch, flush=True)
                else:
                    print("unknow data type: %s" % (recv_data[1]), flush=True)
                if finish_num == self.iter_game_num:
                    assert unstart_num == 0
                    break

    @staticmethod
    def start_mcts_worker(work_id, mcts_tree_num, is_self_play, simulations_num, request_queue, client, human_player, fn):
        np.random.seed(work_id+int(time.time()))
        predictor_fun = lambda features: Trainer.remote_predictor(work_id, request_queue, client, features)
        while True:
            recv_data = client.recv()
            if recv_data == "start":
                factory = MCTSFactory(mcts_tree_num=mcts_tree_num,
                                      is_self_play=is_self_play,
                                      predictor=predictor_fun,
                                      simulations_num=simulations_num,
                                      log_file="../log/log_%d" % (work_id),
                                      human_player=human_player,
                                      fn=fn)
                data_buffer = []
                factory.run(data_buffer)
                request_queue.put((work_id, 'done', data_buffer))
            elif recv_data == "end":
                client.close()
                break
            else:
                print("worker %d recv a unknown signal" % (work_id))
                print(recv_data)

    @staticmethod
    def remote_predictor(work_id, request_queue, predict_pipe, features):
        #print(f"worker {work_id}, feature_shape={str(features.shape)}", flush=True)
        request_queue.put((work_id, 'predict', features))

        recv_data = predict_pipe.recv()

        if len(recv_data) != 3 or recv_data[0] != work_id or recv_data[1] != 'predict':
            print("len_recv_data=%d, work_id_%d!=%d, recv_data_type=%s" % (len(recv_data), work_id, recv_data[0], recv_data[1]))
        assert len(recv_data) == 3 and recv_data[0] == work_id and recv_data[1] == 'predict'
        #print(np.array(recv_data[2][0].shape), flush=True)
        #return recv_data[2][0][0:1], recv_data[2][1][0:1]
        return recv_data[2]

    @staticmethod
    def __run_multi_process(processor_num, mcts_tree_num_each_processor, total_game_num, batch_size, simulations_num, is_self_play, human_player=None):
        request_queue = Queue()
        predict_pipe_arr = []
        processor_arr = []
        fn = sys.stdin.fileno()
        for i in range(processor_num):
            p = Pipe([False])
            predict_pipe_arr.append(p)
            processor_args = (i, mcts_tree_num_each_processor, is_self_play, simulations_num, request_queue, p[0], human_player, fn)
            processor_arr.append(Process(target=Trainer.start_mcts_worker, args=processor_args))
            processor_arr[i].start()

        trainer = Trainer(total_game_num=total_game_num,
                          batch_size=batch_size,
                          simulations_num=simulations_num,
                          processor_num=processor_num,
                          mcts_tree_num_each_processor=mcts_tree_num_each_processor,
                          from_base_model=not is_self_play)
                          #from_base_model=True)

        if is_self_play:
            trainer.compile()
            trainer.fit(request_queue, predict_pipe_arr, processor_arr)
            trainer.save()
        else:
            data_buffer = []
            trainer.predict_server(data_buffer=data_buffer,
                                    predict_pipe_arr=predict_pipe_arr,
                                    request_queue=request_queue,
                                    processor_arr=processor_arr)
        for i in range(processor_num):
            predict_pipe_arr[i][1].send("end")
            predict_pipe_arr[i][1].close()

    @staticmethod
    def run_trainer(processor_num, mcts_tree_num_each_processor, total_game_num, batch_size, simulations_num):
        #os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        physical_devices = tf.config.list_physical_devices('GPU')
        print(physical_devices)
        if len(physical_devices) > 0:
            tf.config.experimental.enable_tensor_float_32_execution(False)
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        Trainer.__run_multi_process(processor_num, mcts_tree_num_each_processor, total_game_num, batch_size, simulations_num, is_self_play=True)

    @staticmethod
    def run_game(simulations_num, human_player=1):
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        physical_devices = tf.config.list_physical_devices('GPU')
        print(physical_devices)
        if len(physical_devices) > 0:
            tf.config.experimental.enable_tensor_float_32_execution(False)
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        Trainer.__run_multi_process(processor_num=1,
                                    mcts_tree_num_each_processor=1,
                                    total_game_num=1,
                                    batch_size=0,
                                    simulations_num=simulations_num,
                                    is_self_play=False,
                                    human_player=human_player)
