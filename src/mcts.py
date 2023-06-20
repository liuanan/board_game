#! /usr/bin/python 
#-*-encoding:cp936-*-
############################################
#
# Author: liuanan
# Create time: 2023-04-15 18:50:33
# E-Mail: liuanan123@qq.com
# version 1.1
#
############################################

import numpy as np
import copy
import time
import sys
import os

from base_board import BaseBoard as Board
from node import Node

class MCTS:
    def __init__(self, is_self_play, simulations_num=1600, curr_step=1, log_file=None, human_player=None, fn=None):
        #print("MCTS.__init__", flush=True)
        self.root = Node(board=None, parents=None, player=1, deep=curr_step)
        self.curr_node = self.root
        self.step = curr_step
        self.is_self_play = is_self_play
        self.winner = -100
        self.diff = 0
        self.simulations_num = simulations_num
        self.c_puct = self.simulations_num ** 0.5 / 10
        self.log_file = log_file
        self.simulation_batch_size = 8
        self.begin_time = 0.
        self.end_time = time.time()
        self.leaf_nodes = []
        self.fout = None
        if self.is_self_play and log_file is not None:
            self.fout = open(self.log_file, 'a')
        else:
            self.fout = sys.stdout
        self.x = None
        self.y = None
        self.predict_x = None
        self.predict_y = None
        self.judge_time = -1.
        self.generate_training_data_time = -1.
        self.human_player = human_player
        if self.is_self_play is False:
            sys.stdin = os.fdopen(fn)

    def uninit(self):
        if self.fout != None:
            self.fout.close()
            self.fout = None

    @staticmethod
    def get_grid_num():
        return Board.GRID_NUM

    def debug_info(self):
        self.end_time = time.time()
        logger = self.fout.write
        idx = Board.xy2idx(self.x, self.y)
        predict_idx = Board.xy2idx(self.predict_x, self.predict_y)
        logger("==\n")
        logger(f"Step {self.step-1}, time {self.end_time - self.begin_time:.2f}\n")
        logger(f"x,y: {self.x},{self.y}  idx: {idx}\n")
        logger(f"move by model, x,y: {self.predict_x},{self.predict_y}\n")
        self.curr_node.parents.debug_info(logger, idx, predict_idx)
        if self.winner != -100 or self.is_self_play is False:
            self.curr_node.board.print(logger)
            logger("\n")
            logger(f"Diff:{self.diff}\n")
            if self.winner != -100:
                logger(f"winner: {Board.player_id2str(self.winner)}\n")
                logger(f"judge:{self.judge_time:.2f}, generate:{self.generate_training_data_time:.2f}\n")
            logger(f"end time: {time.asctime(time.localtime(time.time()))}\n")
            logger("=========================\n")
        self.fout.flush()
        self.begin_time = time.time()

    def simulation(self, simulation_batch_size):
        self.leaf_nodes = self.curr_node.simulation(simulation_batch_size=simulation_batch_size, c_puct=self.c_puct, is_self_play=self.is_self_play)

    @staticmethod
    def expand(predictor, leaf_nodes):
        Node.expand(predictor, leaf_nodes)

    def backup(self):
        for node in self.leaf_nodes:
            assert node != None
            if node.parents != None:
                #node.parents.backup(-node.V, node.idx, self.step, self.fout.write)
                node.parents.backup(-node.V, node.idx, self.step)
        self.leaf_nodes.clear()
    
    def play(self):
        if self.is_self_play and self.step <= Board.TEMP_DECAY:  # t = 1
            temp = 1.
        else:   # t -> 0
            temp = 1e-3
        x = None
        y = None
        if self.is_self_play == False and self.curr_node.player == self.human_player:
            if self.step == 1:
                self.curr_node.board.print(self.fout.write)
            arr = []
            while True:
                self.fout.write("input x,y: ")
                self.fout.flush()
                line = sys.stdin.readline().strip()
                #print(f"DEBUG: '{line}'", flush=True)
                arr = line.strip().split(',')
                if len(arr) == 2:
                    break
                else:
                    self.fout.write("input error, try again.")
            x, y = int(arr[0]), int(arr[1])
        self.curr_node, self.x, self.y, self.predict_x, self.predict_y = self.curr_node.play(temp, x, y)
        self.step += 1

    def judge(self):
        begin_time = time.time()
        assert self.winner == -100
        self.winner, self.diff = self.curr_node.board.judge()
        self.judge_time = time.time() - begin_time

    def generate_training_data(self, data_buffer):
        if self.is_self_play is False:
            self.generate_training_data_time = 0
            return
        begin_time = time.time()
        assert self.winner != -100
        node = self.curr_node.parents
        features = None
        while node != None:
            features = node.generate_features(features)
            data_buffer.append((node.mcts_p, self.winner*node.player, features))
            node = node.parents
        self.generate_training_data_time = time.time() - begin_time
