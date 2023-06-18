#! /usr/bin/python 
#-*-encoding:cp936-*-
############################################
#
# Author: liuanan
# Create time: 2023-04-30 11:46:42
# E-Mail: liuanan123@qq.com
# version 1.1
#
############################################

import time
import numpy as np
import sys
from mcts import MCTS

class MCTSFactory:
    def __init__(self, mcts_tree_num, is_self_play, predictor, simulations_num, log_file, curr_step=1, human_player=None, fn=None):
        #print("MCTSFactory.__init__", flush=True)
        self.trees = []
        self.mcts_tree_num = mcts_tree_num
        self.predictor = predictor
        self.is_self_play = is_self_play
        if self.is_self_play is False:
            assert human_player == -1 or human_player == 1
            self.human_player = human_player
        for i in range(mcts_tree_num):
            self.trees.append(MCTS(is_self_play, simulations_num, curr_step, f'{log_file}_{i}.log', human_player, fn))

    def run(self, data_buffer):
        finish_tree_cnt = 0
        simulation_nodes_num_arr = np.zeros(self.mcts_tree_num, dtype=np.int)
        begin = time.time()
        for tree in self.trees:
            tree.begin_time = begin
        while finish_tree_cnt < self.mcts_tree_num:
            all_leaf_nodes = []
            for tree_id in range(self.mcts_tree_num):
                if simulation_nodes_num_arr[tree_id] == -1: # tree.curr_node.is_end() is True
                    continue
                tree = self.trees[tree_id]
                simulation_batch_size = tree.simulation_batch_size
                if simulation_nodes_num_arr[tree_id] + simulation_batch_size > tree.simulations_num:
                    simulation_batch_size = tree.simulations_num - simulation_nodes_num_arr[tree_id]
                tree.simulation(simulation_batch_size=simulation_batch_size)    # first step: simulation
                simulation_nodes_num_arr[tree_id] += len(tree.leaf_nodes)
                all_leaf_nodes.extend(tree.leaf_nodes)
            assert len(all_leaf_nodes) != 0
            MCTS.expand(self.predictor, all_leaf_nodes)                         # second step: expand
            for tree_id in range(self.mcts_tree_num):
                tree = self.trees[tree_id]
                if simulation_nodes_num_arr[tree_id] >= 0:
                    assert len(tree.leaf_nodes) != 0
                    tree.backup()                                               # third step: backup and clear leaf_nodes
                if simulation_nodes_num_arr[tree_id] == tree.simulations_num: 
                    tree.play()
                    if tree.curr_node.is_end():
                        tree.judge()
                        tree.generate_training_data(data_buffer)
                        finish_tree_cnt += 1
                        tree.debug_info()
                        simulation_nodes_num_arr[tree_id] = -1
                        tree.uninit()
                    else:
                        simulation_nodes_num_arr[tree_id] = 0       # next simulation
                        if tree_id == 0:
                            tree.debug_info()
    @staticmethod
    def get_grid_num():
        return MCTS.get_grid_num()
