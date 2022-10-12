from game_logic import move_action_2_move_id, Game, Board
from mcts import MCTSPlayer

from net import PolicyValueNet


class Human:
    def get_action(self, board):
        move = move_action_2_move_id[input('请输入，行先列后')]
        return move

    def set_player_ind(self, p):
        self.player = p


policy_value_net = PolicyValueNet(model_file='current_policy.pkl')

mcts_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=5, n_playout=100, is_selfplay=0)

human = Human()

game = Game(board=Board())
game.start_play(mcts_player, human, start_player=1, is_shown=1)
