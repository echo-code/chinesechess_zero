"""自我对弈收集数据"""

from collections import deque
import copy
import os
import pickle
import time
from game_logic import Board, Game, move_action_2_move_id, move_id_2_move_action, flip_map
from mcts import MCTSPlayer
from config import CONFIG
import zip_array
from net import PolicyValueNet


# 定义整个对弈收集数据流程
class CollectPipeline:

    def __init__(self):
        # 象棋逻辑和棋盘
        self.board = Board()
        self.game = Game(self.board)
        # 对弈参数
        self.temp = 1  # 温度
        self.n_playout = CONFIG['play_out']  # 每次移动的模拟次数
        self.c_puct = CONFIG['c_puct']  # u的权重
        self.buffer_size = CONFIG['buffer_size']  # 经验池大小
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.iters = 0

    # 从主体加载模型
    def load_model(self):
        model_path = CONFIG['pytorch_model_path']
        try:
            self.policy_value_net = PolicyValueNet(model_file=model_path)
            print('已加载最新模型')
        except:
            self.policy_value_net = PolicyValueNet()
            print('已加载初始模型')
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

    def get_double_data(self, play_data):
        extend_data = []
        for state, mcts_prob, winner in play_data:
            extend_data.append(zip_array.zip_state_mcts_prob((state, mcts_prob, winner)))
            state_flip = state.transpose([1, 2, 0])
            state = state.transpose([1, 2, 0])
            for i in range(10):
                for j in range(9):
                    state_flip[i][j] = state[i][8 - j]
            state_flip = state_flip.transpose([2, 0, 1])
            mcts_prob_flip = copy.deepcopy(mcts_prob)
            for i in range(len(mcts_prob_flip)):
                mcts_prob_flip[i] = mcts_prob[move_action_2_move_id[flip_map(move_id_2_move_action[i])]]
            extend_data.append(zip_array.zip_state_mcts_prob((state_flip, mcts_prob_flip, winner)))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        def add_data():
            if os.path.exists(CONFIG['train_data_buffer_path']):
                while True:
                    try:
                        with open(CONFIG['train_data_buffer_path'], 'rb') as data_dict:
                            data_file = pickle.load(data_dict)
                            self.data_buffer = deque(maxlen=self.buffer_size)
                            self.data_buffer.extend(data_file['data_buffer'])
                            self.iters = data_file['iters']
                            del data_file
                            self.data_buffer.extend(play_data)
                            self.iters += 1
                        print('成功载入数据')
                        break
                    except:
                        time.sleep(30)
            else:
                self.data_buffer.extend(play_data)
                self.iters += 1

        # 收集自我对弈的数据
        for i in range(n_games):
            self.load_model()  # 从本体处加载最新模型
            winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp, is_shown=False)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # 增加数据
            play_data = self.get_double_data(play_data)
            add_data()
            data_dict = {'data_buffer': self.data_buffer, 'iters': self.iters}
            with open(CONFIG['train_data_buffer_path'], 'wb') as data_file:
                pickle.dump(data_dict, data_file)
        return self.iters

    def run(self):
        try:
            while True:
                iters = self.collect_selfplay_data()
                print('batch: {}, episode_len: {}'.format(
                    iters, self.episode_len))
        except KeyboardInterrupt:
            print('\n\rquit')


collecting_pipeline = CollectPipeline()
collecting_pipeline.run()