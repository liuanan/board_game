#! /usr/bin/python 
#-*-encoding:cp936-*-
############################################
#
# Author: liuanan
# Create time: 2023-04-15 19:18:30
# E-Mail: liuanan123@qq.com
# version 1.1
#
############################################

import numpy as np
from base_board import BaseBoard

class GoBoard(BaseBoard):
    delta_arr = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    KOMI = 1
    
    def __init__(self):
        super().__init__()
        self.death_stones = []

    def move(self, player, position, last_death_stones):
        assert player == -1 or player == 1
        self.death_stones.clear()
        if position == GoBoard.GRID_NUM-1:
            self.status = GoBoard.PASS
            return self.status
        x, y = GoBoard.idx2xy(position)
        self.status = self.__check_and_get_death_stones(player, x, y, last_death_stones)
        if self.status == GoBoard.SUCCEED:
            self.board[x, y] = player
            self.__remove_stones()
        else:
            self.death_stones.clear()
        return self.status

    def judge(self):
        komi = GoBoard.KOMI
        black_area = self.__count_player(1)
        white_area = self.__count_player(-1)
        diff = black_area - white_area - komi
        if diff > 0.1:
            return 1, diff
        elif diff < -0.1:
            return -1, diff
        return 0, diff

    def print(self, logger):
        super().print(logger)
        logger("death stones: ")
        logger(' '.join(["[%d,%d]" % (x[0], x[1]) for x in self.death_stones]))
        logger("\n")


    def __check_and_get_death_stones(self, player, x, y, last_remove_lsit):
        if self.board[x, y] != 0:
            return GoBoard.ILLEGAL
        self.board[x, y] = player
        search_flag = np.zeros([GoBoard.ROWS, GoBoard.COLS], dtype=bool)
        fake_death_stones = {}
        for delta in GoBoard.delta_arr + [(0,0)]:
            is_alive, connect_stones = self.__get_connect_stons(x+delta[0], y+delta[1], search_flag)
            if is_alive is False:
                color = self.board[connect_stones[0][0], connect_stones[0][1]]
                if color in fake_death_stones:
                    fake_death_stones[color].extend(connect_stones)
                else:
                    fake_death_stones[color] = connect_stones
        if -player in fake_death_stones:
            self.death_stones = fake_death_stones[-player]
        elif player in fake_death_stones:
            self.death_stones = fake_death_stones[player]
        self.board[x, y] = 0
        if len(self.death_stones) == 1 \
            and len(last_remove_lsit) == 1 \
            and last_remove_lsit[0][0] == x \
            and last_remove_lsit[0][1] == y:
                return GoBoard.ILLEGAL
        return GoBoard.SUCCEED

    def __get_connect_stons(self, x, y, search_flag):
        connect_stones = []
        alive = True
        if x >= 0 and x < GoBoard.ROWS and y >= 0 and y < GoBoard.COLS:
            alive = self.__get_connect_stons_color(self.board[x,y],
                                                    x,
                                                    y,
                                                    search_flag,
                                                    connect_stones)
        if len(connect_stones) == 0:
            alive = True
        return alive, connect_stones
    def __get_connect_stons_color(self, color, x, y, search_flag, connect_stones):
        if x >= 0 and x < GoBoard.ROWS and y >= 0 and y < GoBoard.COLS:
            if self.board[x, y] == 0:
                return True
            elif search_flag[x, y]:
                return False
            elif self.board[x, y] == color:
                connect_stones.append((x, y))
                search_flag[x, y] = True
                flag = False
                for delta in GoBoard.delta_arr:
                    flag = self.__get_connect_stons_color(color, x+delta[0], y+delta[1], search_flag, connect_stones) or flag
                return flag 
        return False
    def __remove_stones(self):
        if len(self.death_stones) != 0:
            idx = np.array(self.death_stones)
            self.board[idx[:,0], idx[:,1]] = 0

    def __count_player(self, player):
        search_flag = np.zeros([GoBoard.ROWS, GoBoard.COLS], dtype=bool)
        area = 0
        for i in range(GoBoard.ROWS):
            for j in range(GoBoard.COLS):
                if self.board[i, j] == player and not search_flag[i, j]:
                    area += self.__get_connect_area(player, i, j, search_flag)
        return area

    def __get_connect_area(self, color, x, y, search_flag):
        area = 0
        if x >= 0 and x < GoBoard.ROWS and y >= 0 and y < GoBoard.COLS:
            if search_flag[x, y] or self.board[x, y] * color < 0:
                search_flag[x, y] = True
            else:
                area += 1
                search_flag[x, y] = True
                for delta in GoBoard.delta_arr:
                    area += self.__get_connect_area(color, x+delta[0], y+delta[1], search_flag)
        return area


