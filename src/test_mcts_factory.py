#! /usr/bin/python 
#-*-encoding:cp936-*-
############################################
#
# Author: liuanan
# Create time: 2023-04-30 17:58:07
# E-Mail: liuanan123@qq.com
# version 1.1
#
############################################

import numpy as np
from trainer import Trainer
from mcts_factory import MCTSFactory
from board import Board

def test():
    simulations_num = 400
    mcts_tree_num_each_processor = 1
    #Board.ROWS = 3
    #Board.COLS = 3
    #Board.FILTERS_NUM = 3
    #Board.GRID_NUM = Board.ROWS * Board.COLS + 1
    factory = MCTSFactory(mcts_tree_num=mcts_tree_num_each_processor,
                          is_self_play=True,
                          predictor=Trainer.fake_predictor,
                          simulations_num=simulations_num,
                          log_file="../log/xxx")
    buf = []
    factory.run(buf)
    print(buf)

if __name__ == '__main__':
    test()
