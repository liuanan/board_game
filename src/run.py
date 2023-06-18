#! /usr/bin/python 
#-*-encoding:cp936-*-
############################################
#
# Author: liuanan
# Create time: 2023-04-29 14:53:18
# E-Mail: liuanan123@qq.com
# version 1.1
#
############################################

from trainer import Trainer
import time
import sys

def run(training=True):
    begin = time.time()
    if training:
        Trainer.run_trainer(processor_num=4,    # 4
                            mcts_tree_num_each_processor=64, # 24
                            total_game_num=50000,
                            batch_size=128,
                            simulations_num=1600)
    else:
        Trainer.run_game(simulations_num=1600,
                         human_player=-1)
    end = time.time()
    sec = end - begin
    minutes = sec // 60
    sec %= 60
    if minutes >= 60:
        hours = minutes // 60
        minutes %= 60
        print("%d:" % (hours), end='')
    if minutes > 0:
        print("%d:" % (minutes), end='')
    print("%d" % (sec))

if __name__ == '__main__':
    print(f"mode: {sys.argv[1]}")
    if sys.argv[1] == "train":
        training = True
    else:
        training = False
    run(training)
