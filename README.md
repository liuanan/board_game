# board_game

tensorflow版本>=2.7

train：
1、进入src文件夹
2、游戏（棋盘）类型
   1）gomoku_board.GomokuBoard 是五子棋
   2）go_board.GoBoard 是围棋
   在node.py中，通过import不同的棋盘，来决定玩什么游戏。可以继承 base_board.BaseBoard 来构建自己的棋盘类型。
   棋盘的参数在BaseBoard中定义：
     ROWS、COLS：棋盘大小
     TEMP_DECAY：每盘棋前TEMP_DECAY步会根据概率进行探索
     FILTERS_NUM：特征数量
     
2、训练：参数在文件run.py
   processor_num：进程数量
   mcts_tree_num_each_processor：每个进程并行进行的游戏数量
   total_game_num：总共运行的游戏场数
   batch_size：训练时每个batch的样本数量
   simulations_num：MCTS搜索模拟次数

   运行 sh run.sh --mode train，开始训练，每模拟processor_num * mcts_tree_num_each_processor场游戏会进行一轮训练，每1000个step保存一次模型。

3. 人机对弈
   在run.py中修改human_player参数来控制先手和后手，1表示先手，-1表示后手。
   运行 sh run.sh --mode game
   
在model文件夹中已经有一个左右互搏3万场比赛训练出来的五子棋模型。
