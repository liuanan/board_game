#! /usr/bin/python 
#-*-encoding:cp936-*-
############################################
#
# Author: liuanan
# Create time: 2023-05-10 21:03:42
# E-Mail: liuanan123@qq.com
# version 1.1
#
############################################

import numpy as np

class BaseBoard(object):
    ROWS = 9
    COLS = 9
    #TEMP_DECAY = 30
    TEMP_DECAY = 8
    FILTERS_NUM = 17

    GRID_NUM = ROWS * COLS + 1
    MAX_STEP = ROWS * COLS * 2

    place_holder = ['x ', '. ', 'o ']
    
    # board status
    PASS = -1
    SUCCEED = 0
    ILLEGAL = 1

    def __init__(self):
        self.board = np.zeros([BaseBoard.ROWS, BaseBoard.COLS], dtype=np.int8)
        self.status = BaseBoard.SUCCEED
        self.death_stones = None
        self.winner = 0
        self.diff = 0
    
    def move(self, player, position, params):
        pass

    def judge(self):
        pass

    def print(self, logger):
        logger('    '+' '.join([' ' if i < 10 else '1' for i in range(BaseBoard.COLS)]))
        logger("\n")
        logger('    '+' '.join([str(i%10) for i in range(BaseBoard.COLS)]))
        logger("\n")
        logger('   '+'--'*BaseBoard.COLS+'-')
        logger("\n")
        for i in range(BaseBoard.ROWS):
            logger('%2d| ' % (i))
            for j in range(BaseBoard.COLS):
                logger(BaseBoard.player_id2str(self.board[i, j]))
            logger("|\n")
        logger('   '+'--'*BaseBoard.COLS+"-\n")
        logger("status: %d\n" % (self.status))

    @staticmethod
    def idx2xy(idx):
        return idx // BaseBoard.COLS, idx % BaseBoard.COLS

    @staticmethod
    def xy2idx(x, y):
        return x * BaseBoard.COLS + y

    @staticmethod
    def player_id2str(player):
        return BaseBoard.place_holder[player+1]

    def get_dir_alpha(self):
        dir_alpha = 10. / (1+np.sum(np.where(self.board, 0, 1)))
        return dir_alpha

