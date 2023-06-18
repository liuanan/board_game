#! /usr/bin/python 
#-*-encoding:cp936-*-
############################################
#
# Author: liuanan
# Create time: 2023-05-10 21:54:57
# E-Mail: liuanan123@qq.com
# version 1.1
#
############################################

import numpy as np
from base_board import BaseBoard

class GomokuBoard(BaseBoard):
    delta_arr = [[(-1 ,0), (1, 0)], [(0, -1), (0, 1)], [(-1, -1), (1, 1)], [(-1, 1), (1, -1)]]
    MAX_STEP = BaseBoard.ROWS * BaseBoard.COLS
    def __init__(self):
        super().__init__()

    def move(self, player, position, last_death_stones=None):
        assert player == -1 or player == 1
        assert last_death_stones is None
        if position == BaseBoard.GRID_NUM-1:
            self.status = BaseBoard.PASS
            return self.status
        x, y = BaseBoard.idx2xy(position)
        if self.board[x, y] != 0:
            self.status = BaseBoard.ILLEGAL
            return self.status
        self.board[x, y] = player
        self.status = BaseBoard.SUCCEED
        self.winner, self.diff = self.__bingo(x, y)
        return self.status

    def judge(self):
        return self.winner, self.diff

    def __bingo(self, x, y):
        player = self.board[x, y]
        assert player != 0
        flag = False
        for delta in GomokuBoard.delta_arr:
            num = 1
            for direction in delta:
                next_x = x + direction[0]
                next_y = y + direction[1]
                while next_x >= 0 and next_x < BaseBoard.ROWS and next_y >= 0 and next_y < BaseBoard.COLS \
                        and self.board[next_x, next_y] == player:
                    num += 1
                    next_x = next_x + direction[0]
                    next_y = next_y + direction[1]
            if num >= 5:
                return player, num
        return 0, 0
