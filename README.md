# 使用
## 运行环境
      请先安装git-lfs并且初始化lfs环境之后，在clone代码！！！
      tensorflow version >= 2.7
      python version >= 3.8
      git-lfs >= 3.3.0，模型文件大小超过100M，需要使用lfs管理。安装lfs后一定要记得运行初始化命令 ‘git lfs install’ ，否则模型文件不能正常clone。
      centos/linux/macos(M1)上测试可以正常运行

## 配置参数
      代码在src文件夹下，训练相关的参数都在代码里，暂未把参数聚合成配置文件的方式。
      
      1、游戏（棋盘）类型
         * 五子棋：src/go_board.py，gomoku_board.GomokuBoard
         * 围棋：src/gomoku_board.py，go_board.GoBoard
         在node.py中，通过import不同的棋盘，来决定玩什么游戏。也可以继承 base_board.BaseBoard 来构建自己的棋盘类型。
         
      2、棋盘参数，在BaseBoard中定义：
         * ROWS、COLS：棋盘大小
         * TEMP_DECAY：每盘棋前TEMP_DECAY步会根据概率进行探索
         * FILTERS_NUM：特征数量
         
      3、训练超参数，src/run.py
         * processor_num：进程数量，控制MCTS逻辑使用的算力上限
         * mcts_tree_num_each_processor：每个进程并行进行的游戏数量
         * total_game_num：总共运行的游戏场数
         * batch_size：训练时每个batch的样本数量
         * simulations_num：MCTS搜索模拟次数
         p.s. processor_num和mcts_tree_num_each_processor的数量可以根据机器硬件情况来确定，训练开始后在log文件夹会有日志，可以在日志文件里面看每一步棋的运行时间。
      
      4、模型超参数，src/trainer.py
         * optimize: self.optimizer
         * learning rate: self.lr，在learning_rate.py中控制
         * loss function: self.loss_fn = [policy_loss_function，value_loss_function]
         * num_res_blocks: 模型中ResNet堆叠数量
         * tf.config.experimental.enable_tensor_float_32_execution(False): 不开启GPU的TF32功能，使用TF32会导致效果下降。
         p.s. num_res_blocks数量越多，效果越好，但是训练速度也越慢；使用GPU训练的话，最好关闭TF32。

      5、人机对弈参数，src/run.py
         * human_player：1 表示人类选手先手，-1 表示人类选手后手
         
## 运行命令
### 训练
      sh run.sh --mode train
      
      开始训练，每模拟 processor_num * mcts_tree_num_each_processor 场游戏会生成一批训练数据进行训练，训练过程中每1000个step保存一次模型。训练过程中会有日志产出，日志文件在log文件夹下，有3类日志文件：
      * log_{process_id}_{tree_id}.log：每一棵树的日志信息，每个process的第0棵树会保留详细的日志信息，包括每一步的耗时、下棋位置、概率、价值、选择的次数等信息，可以在这里根据每一步的耗时调整processor_num和mcts_tree_num_each_processor参数，达到最优的硬件使用效率。
      * run.log: 训练效率&效果相关的信息，每一批训练数据的大小、loss、耗时。可以把这个文件cat到一个新的文件中，比如‘cat run.log > log‘，然后在log中替换一下特殊字符"%s/^H\+/\r/g"，可以更清晰看到详细的训练log。
      * run.err: 训练过程中显示的错误日志，调试使用。

      在GPU/CPU机器上都验证过，包括linux和macos，有条件的话，最好使用GPU训练，训练速度有非常大的提升。最后是在autodl上租用了3080卡进行训练的，训练其实也挺慢的，代码的运行效率还是有挺大的运行空间的。因为五子棋的终止步数比围棋少一半，所以最终可用的模型只训练了五子棋的，太花钱了，暂时没有训练围棋的模型，希望以后算力能便宜下来，作为个人使用者也能很实惠的使用到大规模算力资源。感觉研究和工业生产方向对个人研究越来越不友好了，动不动就得要多几多卡的资源，特别是预训练方向，几千上万张A100的卡，个人研究不太可能有这么多的资源。
      
### 人机对弈
      sh run.sh --mode game
      
      在model文件夹中已经有一个左右互搏3万场比赛训练出来的五子棋模型，可以使用这个模型来进行人机对战。因为人机对弈的时候，模型预估所需的算力要求不高，所以在CPU机器上也能比较快的完成计算，每步棋在10s这个量级。

# 感悟
   最开始是在工作中想看看能不能使用到强化学习的的技术，顺着把alphago系列、chatgpt相关的技术了解了一下，在学习过程中觉得强化学习还挺有意思的，在alphago的应用过程中有点左右互搏的意思，想着自己能不能把代码撸出来。说干就干，23年五一之前的两个周末的下午和晚上、五一假期的第一天，基本上就把代码写完并完成了调试，调试过程中也做了挺多的策略调整，包括：
   * 根节点的理解：当前这一步作为根节点的局面信息。
   * 并行化：训练效率太低了，发现GPU使用率特别低，得想办法把GPU利用率提上来。整个训练任务的耗时主要是 expand 阶段，并行化的策略也是做了挺多调整的，最开始是每个进程运行一棵MCTS树，在主进程里面把各个进程所需要inference的数据合并起来交给GPU进行预估（合并策略也尝试了很多，比如每次等到超过一半的MCTS节点有inference请求的时候合并一次交给GPU、预估数据的共享队列没有数据的时候马上就合并数据交给GPU进行预估 etc），结果发现每个进程CPU利用率很低，并且GPU的利用率波动很大，所以改成每个进程内部维护多棵MCTS树，提高单个进程的计算能力，同时主进程也不进行预估数据的合并，完全由processor_num和mcts_tree_num_each_processor控制GPU的效率；每一步棋的simulation并行策略和论文、网上找到的资料不太一样，每一步棋从根节点进行多次 select 操作，当select遇到相同的子节点(根节点的子节点)或者超过simulation_batch_size次select，则停止select，进行expand和backup操作，这样的话可以保证在同一轮expand的叶子节点不会重复，并且保证并行 select 和串行 select 的逻辑是等价的。
   * 狄利克雷噪声添加的位置：网上资料有很多不同的解读，在实现过程中也尝试了不同的方法，最后是在根节点棋面通过模型预估的策略分布上添加狄利克雷噪声。
   * 狄利克雷噪声alpha参数：论文中根据棋的类型有一些推荐值，但是无法根据棋盘大小自动调整。这里也摸索了一段时间，感觉是应该通过当前这步棋合法落子的位置数量来动态修改要好一些，可落子的范围越大添加的噪声方差越大，可落子的范围越小，添加的噪声方差越小，再根据论文提到的19*19围棋的aplha参数是0.03，设置 alpha = 10/可落子范围，实现在base_board.get_dir_alpha方法中。
   * c_puct参数：c_puct是一个探索参数，基于simulations次数越多、根据先验概率探索次数越多的想法，设置 c_puct = simulations_num^0.5 / 10。这里再看下U的计算公式 u = c_puct * P * np.sqrt(sum_n) / (1. + n)，u的量纲应该是概率分布，公式中已经有一个P了，所以c_puct的量纲应该和np.sqrt(sum_n) / (1. + n)相反，simulations_num^0.5 / 10的量纲刚好也符合要求。

   代码撸完了得找资源跑训练任务，之前调试是在m1 mac上做的，resblock数量调整到13的时候，并行化的单步耗时已经在分钟级别了，太慢了！在云平台上租用了3080卡进行训练，速度有非常大的提升，5-6倍吧，本来还想租用A100显卡，奈何高端显卡租用费太贵了。后面有时间的话，还是需要把GPU并行版本也搞一下。

   五子棋训练9*9的棋盘，有很多的平局，导致训练一定的步数后，value网络容易拟合平局的结果，后面可以尝试下先手未胜判负的策略，降低先手优势，避免value值趋同的问题。
     
   在找资料的过程中，遇到了很多写的特别好的文章，非常感谢大佬在网上免费传播知识，在这里推荐3篇我觉得写的很通透的文章，对我理解强化学习、棋类游戏有非常大的帮助：
   * [强化学习极简入门：通俗理解MDP、DP MC TC和Q学习、策略梯度、PPO](https://blog.csdn.net/v_JULY_v/article/details/128965854)
   * [浅述：从 Minimax 到 AlphaZero，完全信息博弈之路（1）](https://zhuanlan.zhihu.com/p/31809930)
   * [浅述：从 Minimax 到 AlphaZero，完全信息博弈之路（2](https://zhuanlan.zhihu.com/p/32073374)
