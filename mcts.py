"""蒙特卡洛树搜索"""

import numpy as np
import copy
from config import CONFIG


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


# 定义节点
class TreeNode(object):
    """
    mcts节点位状态，边为行动
    每个节点的信息：自身的Q，先验概率P，其访问次数调整的u
    """

    def __init__(self, parent, prior_p):
        """
        parent: 当前节点的父节点
        prior_p: 当前节点被选择的先验概率
        """
        self._parent = parent
        self._children = {}  # 从动作到TreeNode的映射
        self._n_visits = 0  # 当前当前节点的访问次数
        self._Q = 0  # 当前节点对应动作的平均动作价值
        self._u = 0  # 当前节点的置信上限 puct
        self._P = prior_p

    def select(self, c_puct):
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def expand(self, action_priors):  # 这里把不合法的动作概率为0
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def get_value(self, c_puct):
        self._u = c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits)
        return self._Q + self._u

    def update(self, leaf_value):
        self._n_visits += 1
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    # 使用递归的方法对所有节点（当前节点对应的支线）进行一次更新
    def update_recursive(self, leaf_value):
        # 如果它不是根节点，则应首先更新此节点的父节点
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None


# 蒙特卡洛搜索树
class MCTS(object):

    def __init__(self, policy_value_fn, c_puct=CONFIG['c_puct'], n_playout=2000):
        """policy_value_fn: 接收board的盘面状态，返回落子概率和盘面评估得分"""
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        node = self._root
        while True:
            if node.is_leaf():
                break
            # 贪心算法选择下一步行动
            action, node = node.select(self._c_puct)
            state.make_move(action)

        # 使用网络评估叶子节点，网络输出（动作，概率）元组p的列表以及当前玩家视角的得分[-1, 1]
        action_probs, leaf_value = self._policy(state)
        # 查看游戏是否结束
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        else:
            # 对于结束状态，将叶子节点的值换成1或-1
            if winner == -1:  # Tie
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == state.get_current_player_id() else -1.0
                )
        # 在本次遍历中更新节点的值和访问次数
        # 必须添加符号，因为两个玩家共用一个搜索树
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # 跟据根节点处的访问计数来计算移动概率
        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
        return acts, act_probs

    def update_with_move(self, last_move):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return 'MCTS'


# 基于MCTS的AI玩家
class MCTSPlayer(object):

    def __init__(self, policy_value_function, c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay
        self.agent = "AI"

    def set_player_ind(self, p):
        self.player = p

    # 重置搜索树
    def reset_player(self):
        self.mcts.update_with_move(-1)

    def __str__(self):
        return 'MCTS {}'.format(self.player)

    # 得到行动
    def get_action(self, board, temp=1e-3, return_prob=0):
        # 像alphaGo_Zero论文一样使用MCTS算法返回的pi向量
        move_probs = np.zeros(2086)

        acts, probs = self.mcts.get_move_probs(board, temp)
        move_probs[list(acts)] = probs
        if self._is_selfplay:
            # 添加Dirichlet Noise进行探索（自我对弈需要）
            move = np.random.choice(
                acts,
                p=0.75 * probs + 0.25 * np.random.dirichlet(CONFIG['dirichlet'] * np.ones(len(probs)))
            )
            # 更新根节点并重用搜索树
            self.mcts.update_with_move(move)
        else:
            # 使用默认的temp=1e-3，它几乎相当于选择具有最高概率的移动
            move = np.random.choice(acts, p=probs)
            # 重置根节点
            self.mcts.update_with_move(-1)
        if return_prob:
            return move, move_probs
        else:
            return move
