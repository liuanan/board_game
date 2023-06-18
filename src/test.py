#! /usr/bin/python 
#-*-encoding:cp936-*-
############################################
#
# Author: liuanan
# Create time: 2023-04-15 20:24:55
# E-Mail: liuanan123@qq.com
# version 1.1
#
############################################

import sys
import copy
from board import Board
from mcts import MCTS

def test():
    board = Board()
    player = -1
    fout = open('move.out', 'w')
    for line in sys.stdin:
        fout.write('='*26)
        arr = line.strip().split(',')
        if len(arr) == 3:
            x, y, player = arr
        elif len(arr) == 2:
            x, y = arr
            player *= -1
        x = int(x)
        y = int(y)
        player = int(player)
        board.move(player, x * Board.COLS + y, copy.deepcopy(board.death_stones))
        board.print(fout.write)
        winner, diff = board.judge()
        fout.write("player: %d\n" % (player))
        fout.write("x, y: %d, %d\n" % (x, y))
        fout.write("judge: winner=%d, diff=%.1f\n" % (winner, diff))
    fout.close()

def test_self_play():
    mcts = MCTS(is_self_play=True, model=None)
    mcts.run()

if __name__ == '__main__':
    test()
