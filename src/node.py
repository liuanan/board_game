#! /usr/bin/python 
#-*-encoding:cp936-*-
############################################
#
# Author: liuanan
# Create time: 2023-04-16 10:51:14
# E-Mail: liuanan123@qq.com
# version 1.1
#
############################################

import numpy as np
import copy
#from go_board import GoBoard as Board
from gomoku_board import GomokuBoard as Board

class Node:
    def __init__(self, board, parents, player, deep):
        # node info
        self.player = player    # 1 or -1
        self.position = None
        self.parents = parents
        self.children = None
        self.board = None
        if board is None:
            self.board = Board()
        else:
            self.board = copy.deepcopy(board)
        self.V = None   # win_loss_draw if the game is over else predict by model
        self.Q = None   # mcts value
        self.deep = deep
        self.game_over = None
        self.dir_noise = None

        # info of children
        self.N = None
        #self.Z = None
        self.P = None       # predict by model
        self.mcts_p = None
        self.q_plus_u = None

        # parents.move(self.idx) -> self
        self.idx = None
        self.alpha = None

    def clear(self):
        #if self.children is not None:
        #    self.children.clear()
        self.q_plus_u = None
        self.N = np.zeros([Board.GRID_NUM], dtype=np.uint32)
        self.Q = np.zeros([Board.GRID_NUM], dtype=np.float32)

    def next_player(self):
        return -self.player

    def move(self, idx):    # board init by parents board, parents board -(play)->  this.node.board
        self.idx = idx
        ret = self.board.move(self.parents.player, idx, self.parents.board.death_stones)
        return ret

    def simulation(self, simulation_batch_size, c_puct, is_self_play=True):
        leaf_nodes = []
        if self.V is None:
            leaf_nodes.append(self)     # root node need expand
        else:
            simulation_idx_set = set()
            for i in range(simulation_batch_size):
                if is_self_play:
                    if self.dir_noise is None:
                        assert self.alpha is None
                        self.alpha = self.board.get_dir_alpha()
                        self.dir_noise = np.random.dirichlet(self.alpha*np.ones(self.board.GRID_NUM, dtype=np.float32))
                        self.dir_noise[:-1] = np.where(np.reshape(self.board.board, (-1)), 0., self.dir_noise[:-1])
                        self.dir_noise /= np.sum(self.dir_noise)
                leaf_node = self.__select(c_puct, self.dir_noise, simulation_idx_set)
                if leaf_node is None:
                    break
                leaf_nodes.append(leaf_node)
        return leaf_nodes

    def __select(self, c_puct, dir_noise=None, simulation_idx_set=None):
        if self.V is None:
            pass
        elif not self.is_end():
            tmp = self.N.astype('float32')
            sum_n = np.sum(tmp)
            max_idx = None
            #tmp = np.square(tmp)
            if sum_n < 0.5:
                sum_n += 1.
            if dir_noise is None:
                self.q_plus_u = self.Q + c_puct * self.P * np.sqrt(sum_n) / (1. + tmp)
            else:       # root node
                self.q_plus_u = self.Q + c_puct * (0.75 * self.P + 0.25 * dir_noise) * np.sqrt(sum_n) / (1. + tmp)
            self.q_plus_u[:Board.GRID_NUM-1] = np.where(np.reshape(self.board.board, (-1)) == 0, self.q_plus_u[:Board.GRID_NUM-1], np.finfo(np.float32).min)
            max_idx = np.argmax(self.q_plus_u)
            if simulation_idx_set is not None:
                if max_idx in simulation_idx_set:
                    return None
                simulation_idx_set.add(max_idx)
            if self.children[max_idx] is None:
                self.children[max_idx] = Node(self.board, self, self.next_player(), self.deep+1)
                self.children[max_idx].move(max_idx)
            self.N[max_idx] += 1    # backup N
            return self.children[max_idx].__select(c_puct)
        return self

    @staticmethod
    def expand(predictor, leaf_nodes):
        feature_batch = []
        need_predict_nodes = []
        for node in leaf_nodes:
            if node.is_end():
                if node.V is None:
                    reward, _ = node.board.judge()
                    node.V = reward * node.player
            else:
                feature_batch.append(node.generate_features())
                need_predict_nodes.append(node)
        if len(feature_batch) != 0:
            p_batch, v_batch = predictor(np.array(feature_batch))
            for i in range(len(need_predict_nodes)):
                need_predict_nodes[i].PTMP = p_batch[i]
                need_predict_nodes[i].P = Node.softmax(p_batch[i])
                need_predict_nodes[i].V = v_batch[i][0]
                need_predict_nodes[i].N = np.zeros([Board.GRID_NUM], dtype=np.uint32)
    
                need_predict_nodes[i].Q = np.zeros([Board.GRID_NUM], dtype=np.float32)
                need_predict_nodes[i].children = [None] * (Board.GRID_NUM)
        return len(need_predict_nodes)
    
    def backup(self, v, idx, step, logger=None):
        if self.deep >= step:
            # update self.N in select step
            # Q = (Q * (N - 1) + v)/ N = Q - (Q - v) / N 
            self.Q[idx] += (v - self.Q[idx]) / self.N[idx]
            if logger is not None:
                logger(f"step:{step} deep:{self.deep} self.q_plus_u[{idx}]:{self.q_plus_u[idx]} q[{idx}]:{self.Q[idx]} v:{v} n[{idx}]:{self.N[idx]}\n")
            if self.parents is not None:
                self.parents.backup(-v, self.idx, step, logger)

    def is_end(self):
        if self.game_over is None:
            if self.board.winner != 0 \
                    or self.deep > Board.MAX_STEP \
                    or (self.deep > 2 \
                        and self.board.status != Board.SUCCEED \
                        and self.parents.board.status != Board.SUCCEED):
                self.game_over = True
            else:
                self.game_over = False
        return self.game_over

    def play(self, temp, x=None, y=None):
        max_idx = -1
        self.mcts_p = Node.softmax(1/temp * np.log(self.N.astype('float32') + 1e-10))
        self.mcts_p = np.where(self.N == 0, 0., self.mcts_p)
        if x is None:
            max_idx = np.random.choice(range(Board.GRID_NUM),
                                   p=self.mcts_p)
        else:
            max_idx = Board.xy2idx(x, y)
        if self.children[max_idx] == None:
            assert x is not None
            #print(f"len={len(self.children)} max_idx={max_idx}", flush=True)
            self.children[max_idx] = Node(self.board, self, self.next_player(), self.deep+1)
            self.children[max_idx].move(max_idx)
        for i in range(len(self.children)):
            if i != max_idx:
                self.children[i] = None
        x, y = Board.idx2xy(max_idx)
        #self.children[max_idx].clear()
        return self.children[max_idx], x, y


    def generate_features(self, children_feature=None):
        features = np.zeros([Board.ROWS, Board.COLS, Board.FILTERS_NUM], dtype=np.float32)
        features[:,:,Board.FILTERS_NUM-1] = (self.player + 1) // 2
        node = self
        half_num = Board.FILTERS_NUM // 2
        if children_feature is None:
            for i in range(half_num):
                features[:,:,i*2] = (node.board.board + 1) // 2
                features[:,:,i*2+1] = (node.board.board - 1) // -2
                node = node.parents
                if node is None:
                    break
        else:
            features[:,:,0:Board.FILTERS_NUM-3] = children_feature[:,:,2:Board.FILTERS_NUM-1]
            for i in range(half_num-1):
                node = node.parents
                if node is None:
                    break
            if node is not None:
                features[:,:,Board.FILTERS_NUM-3] = (node.board.board + 1) // 2
                features[:,:,Board.FILTERS_NUM-2] = (node.board.board - 1) // -2
        return features

    @staticmethod
    def softmax(x):
        probs = np.exp(x-np.max(x))
        return probs/np.sum(probs)

    def debug_info(self, logger, idx):
        logger("player: %s\n" % (Board.player_id2str(self.player)))
        logger("mcts_p:%.4f  p:%.4f  logic:%.4f\n" % (self.mcts_p[idx], self.P[idx], self.PTMP[idx]))
        logger("q_u:%.4f  Q:%.4f  V:%.6f\n" % (self.q_plus_u[idx], self.Q[idx], self.V))
        logger("N:%d  SUM:%d\n" % (self.N[idx], np.sum(self.N)))
        #logger("self.P:\n%s\nself.QU:\n%s\nslef.N:\n%s\nself.Q:\n%s\n" % (str(self.P), str(self.q_plus_u), str(self.N), str(self.Q)))
        logger("\n")

