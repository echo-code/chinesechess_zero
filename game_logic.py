"""游戏基本逻辑"""

import numpy as np
import copy
import time
from config import CONFIG
from collections import deque

# 列表来表示棋盘，红方在上，黑方在下。使用时需要使用深拷贝
init_state_list = [['红车', '红马', '红象', '红士', '红帅', '红士', '红象', '红马', '红车'],
                   ['　　', '　　', '　　', '　　', '　　', '　　', '　　', '　　', '　　'],
                   ['　　', '红炮', '　　', '　　', '　　', '　　', '　　', '红炮', '　　'],
                   ['红兵', '　　', '红兵', '　　', '红兵', '　　', '红兵', '　　', '红兵'],
                   ['　　', '　　', '　　', '　　', '　　', '　　', '　　', '　　', '　　'],
                   ['　　', '　　', '　　', '　　', '　　', '　　', '　　', '　　', '　　'],
                   ['黑兵', '　　', '黑兵', '　　', '黑兵', '　　', '黑兵', '　　', '黑兵'],
                   ['　　', '黑炮', '　　', '　　', '　　', '　　', '　　', '黑炮', '　　'],
                   ['　　', '　　', '　　', '　　', '　　', '　　', '　　', '　　', '　　'],
                   ['黑车', '黑马', '黑象', '黑士', '黑帅', '黑士', '黑象', '黑马', '黑车']]

# deque来存储棋盘状态，长度为4
state_deque = deque(maxlen=4)
for _ in range(4):
    state_deque.append(copy.deepcopy(init_state_list))

# 字典：字符串到数组的映射
string_2_array = {'红车': np.array([1, 0, 0, 0, 0, 0, 0]), '红马': np.array([0, 1, 0, 0, 0, 0, 0]),
                  '红象': np.array([0, 0, 1, 0, 0, 0, 0]), '红士': np.array([0, 0, 0, 1, 0, 0, 0]),
                  '红帅': np.array([0, 0, 0, 0, 1, 0, 0]), '红炮': np.array([0, 0, 0, 0, 0, 1, 0]),
                  '红兵': np.array([0, 0, 0, 0, 0, 0, 1]), '黑车': np.array([-1, 0, 0, 0, 0, 0, 0]),
                  '黑马': np.array([0, -1, 0, 0, 0, 0, 0]), '黑象': np.array([0, 0, -1, 0, 0, 0, 0]),
                  '黑士': np.array([0, 0, 0, -1, 0, 0, 0]), '黑帅': np.array([0, 0, 0, 0, -1, 0, 0]),
                  '黑炮': np.array([0, 0, 0, 0, 0, -1, 0]), '黑兵': np.array([0, 0, 0, 0, 0, 0, -1]),
                  '　　': np.array([0, 0, 0, 0, 0, 0, 0])}


# 函数：数组到字符串的映射
def array_2_string(array):
    return list(filter(lambda string: (string_2_array[string] == array).all(), string_2_array))[0]


# 移动棋子
def change_state(state_list, move):
    # move 从前到后
    copy_list = copy.deepcopy(state_list)
    y, x, toy, tox = int(move[0]), int(move[1]), int(move[2]), int(move[3])
    copy_list[toy][tox] = copy_list[y][x]
    copy_list[y][x] = '　　'
    return copy_list


# 输出棋盘
def print_board(state_array):
    print('    ',end='')
    for i in range(9):
        print(i, end=' , ')
    print()
    for i in range(10):
        for j in range(9):
            if j == 0:
                print(i, end=',')
            print(array_2_string(state_array[i][j]) + ',', end='')
        print()


# 将字符串盘面转化为数组盘面
def state_list_2_state_array(state_list):
    state_array = np.zeros([10, 9, 7])
    for i in range(10):
        for j in range(9):
            state_array[i][j] = string_2_array[state_list[i][j]]
    return state_array


# 拿到所有合法走子的集合，2086长度，神经网络预测的走子概率向量的长度
# 给每个移动方案move_action一个编号move_id
# 第一个字典：move_id到move_action
# 第二个字典：move_action到move_id
def get_all_legal_moves():
    move_id_2_move_action = {}
    move_action_2_move_id = {}
    row = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
    column = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    # 叉士
    advisor_labels = ['0314', '1403', '0514', '1405', '2314', '1423', '2514', '1425',
                      '9384', '8493', '9584', '8495', '7384', '8473', '7584', '8475']
    # 象走田
    bishop_labels = ['2002', '0220', '2042', '4220', '0224', '2402', '4224', '2442',
                     '2406', '0624', '2446', '4624', '0628', '2806', '4628', '2846',
                     '7052', '5270', '7092', '9270', '5274', '7452', '9274', '7492',
                     '7456', '5674', '7496', '9674', '5678', '7856', '9678', '7896']

    # 马走日
    idx = 0
    for l1 in range(10):
        for n1 in range(9):
            destinations = [(t, n1) for t in range(10)] + \
                           [(l1, t) for t in range(9)] + \
                           [(l1 + a, n1 + b) for (a, b) in
                            [(-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (-1, 2), (2, 1), (1, 2)]]
            for (l2, n2) in destinations:
                if (l1, n1) != (l2, n2) and l2 in range(10) and n2 in range(9):
                    action = column[l1] + row[n1] + column[l2] + row[n2]
                    move_id_2_move_action[idx] = action
                    move_action_2_move_id[action] = idx
                    idx += 1

    for action in advisor_labels:
        move_id_2_move_action[idx] = action
        move_action_2_move_id[action] = idx
        idx += 1

    for action in bishop_labels:
        move_id_2_move_action[idx] = action
        move_action_2_move_id[action] = idx
        idx += 1

    return move_id_2_move_action, move_action_2_move_id


move_id_2_move_action, move_action_2_move_id = get_all_legal_moves()


# 走子翻转 节省代码
def flip_map(string):
    new_str = ''
    for index in range(4):
        if index == 0 or index == 2:
            new_str += (str(string[index]))
        else:
            new_str += (str(8 - int(string[index])))
    return new_str


# 防止越界
def check_r(to_y, to_x):
    if 0 <= to_y <= 9 and 0 <= to_x <= 8:
        return True
    return False


# 不能吃自己字
def check_self(piece, current_player_color):
    # 当走到的位置存在棋子的时候，进行一次判断
    if piece != '　　':
        if current_player_color == '红':
            if '黑' in piece:
                return True
            else:
                return False
        elif current_player_color == '黑':
            if '红' in piece:
                return True
            else:
                return False
    else:
        return True


# 得到当前盘面合法走子集合
def get_legal_moves(state_deque, current_player_color):
    state_list = state_deque[-1]  # 现在棋盘
    old_state_list = state_deque[-4]  # 老棋盘

    moves = []  # 所有走法
    face_to_face = False  # 将对帅

    # 记录将军的位置信息
    k_x = None
    k_y = None
    K_x = None
    K_y = None

    # state_list : 10*9
    # 遍历移动初始位置
    for y in range(10):
        for x in range(9):
            if state_list[y][x] == '　　':
                pass
            else:
                # 棋子可以移动
                if state_list[y][x] == '黑车' and current_player_color == '黑':  # 黑车的合法走子
                    to_y = y
                    for to_x in range(x - 1, -1, -1):
                        m = str(y) + str(x) + str(to_y) + str(to_x)
                        if state_list[to_y][to_x] != '　　':
                            if '红' in state_list[to_y][to_x]:
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                            break
                        if change_state(state_list, m) != old_state_list:
                            moves.append(m)
                    for to_x in range(x + 1, 9):
                        m = str(y) + str(x) + str(to_y) + str(to_x)
                        if state_list[to_y][to_x] != '　　':
                            if '红' in state_list[to_y][to_x]:
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                            break
                        if change_state(state_list, m) != old_state_list:
                            moves.append(m)

                    to_x = x
                    for to_y in range(y - 1, -1, -1):
                        m = str(y) + str(x) + str(to_y) + str(to_x)
                        if state_list[to_y][to_x] != '　　':
                            if '红' in state_list[to_y][to_x]:
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                            break
                        if change_state(state_list, m) != old_state_list:
                            moves.append(m)
                    for to_y in range(y + 1, 10):
                        m = str(y) + str(x) + str(to_y) + str(to_x)
                        if state_list[to_y][to_x] != '　　':
                            if '红' in state_list[to_y][to_x]:
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                            break
                        if change_state(state_list, m) != old_state_list:
                            moves.append(m)

                elif state_list[y][x] == '红车' and current_player_color == '红':  # 红车的合法走子(复制粘贴黑车)
                    to_y = y
                    for to_x in range(x - 1, -1, -1):
                        m = str(y) + str(x) + str(to_y) + str(to_x)
                        if state_list[to_y][to_x] != '　　':
                            if '黑' in state_list[to_y][to_x]:
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                            break
                        if change_state(state_list, m) != old_state_list:
                            moves.append(m)
                    for to_x in range(x + 1, 9):
                        m = str(y) + str(x) + str(to_y) + str(to_x)
                        if state_list[to_y][to_x] != '　　':
                            if '黑' in state_list[to_y][to_x]:
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                            break
                        if change_state(state_list, m) != old_state_list:
                            moves.append(m)

                    to_x = x
                    for to_y in range(y - 1, -1, -1):
                        m = str(y) + str(x) + str(to_y) + str(to_x)
                        if state_list[to_y][to_x] != '　　':
                            if '黑' in state_list[to_y][to_x]:
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                            break
                        if change_state(state_list, m) != old_state_list:
                            moves.append(m)
                    for to_y in range(y + 1, 10):
                        m = str(y) + str(x) + str(to_y) + str(to_x)
                        if state_list[to_y][to_x] != '　　':
                            if '黑' in state_list[to_y][to_x]:
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                            break
                        if change_state(state_list, m) != old_state_list:
                            moves.append(m)

                # 黑马
                elif state_list[y][x] == '黑马' and current_player_color == '黑':
                    for i in range(-1, 3, 2):
                        for j in range(-1, 3, 2):
                            to_y = y + 2 * i
                            to_x = x + 1 * j
                            if check_r(to_y, to_x) \
                                    and check_self(state_list[to_y][to_x], current_player_color='黑') \
                                    and state_list[to_y - i][x] == '　　':
                                m = str(y) + str(x) + str(to_y) + str(to_x)
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                            to_y = y + 1 * i
                            to_x = x + 2 * j
                            if check_r(to_y, to_x) \
                                    and check_self(state_list[to_y][to_x], current_player_color='黑') \
                                    and state_list[y][to_x - j] == '　　':
                                m = str(y) + str(x) + str(to_y) + str(to_x)
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)

                # 红马
                elif state_list[y][x] == '红马' and current_player_color == '红':
                    for i in range(-1, 3, 2):
                        for j in range(-1, 3, 2):
                            to_y = y + 2 * i
                            to_x = x + 1 * j
                            if check_r(to_y, to_x) \
                                    and check_self(state_list[to_y][to_x], current_player_color='红') \
                                    and state_list[to_y - i][x] == '　　':
                                m = str(y) + str(x) + str(to_y) + str(to_x)
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                            to_y = y + 1 * i
                            to_x = x + 2 * j
                            if check_r(to_y, to_x) \
                                    and check_self(state_list[to_y][to_x], current_player_color='红') \
                                    and state_list[y][to_x - j] == '　　':
                                m = str(y) + str(x) + str(to_y) + str(to_x)
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)

                # 黑象
                elif state_list[y][x] == '黑象' and current_player_color == '黑':
                    for i in range(-2, 3, 4):
                        to_y = y + i
                        to_x = x + i
                        if check_r(to_y, to_x) \
                                and check_self(state_list[to_y][to_x], current_player_color='黑') \
                                and to_y >= 5 and state_list[y + i // 2][x + i // 2] == '　　':
                            m = str(y) + str(x) + str(to_y) + str(to_x)
                            if change_state(state_list, m) != old_state_list:
                                moves.append(m)
                        to_y = y + i
                        to_x = x - i
                        if check_r(to_y, to_x) \
                                and check_self(state_list[to_y][to_x], current_player_color='黑') \
                                and to_y >= 5 and state_list[y + i // 2][x - i // 2] == '　　':
                            m = str(y) + str(x) + str(to_y) + str(to_x)
                            if change_state(state_list, m) != old_state_list:
                                moves.append(m)

                # 红象
                elif state_list[y][x] == '红象' and current_player_color == '红':
                    for i in range(-2, 3, 4):
                        to_y = y + i
                        to_x = x + i
                        if check_r(to_y, to_x) \
                                and check_self(state_list[to_y][to_x], current_player_color='红') \
                                and to_y <= 4 and state_list[y + i // 2][x + i // 2] == '　　':
                            m = str(y) + str(x) + str(to_y) + str(to_x)
                            if change_state(state_list, m) != old_state_list:
                                moves.append(m)
                        to_y = y + i
                        to_x = x - i
                        if check_r(to_y, to_x) \
                                and check_self(state_list[to_y][to_x], current_player_color='红') \
                                and to_y <= 4 and state_list[y + i // 2][x - i // 2] == '　　':
                            m = str(y) + str(x) + str(to_y) + str(to_x)
                            if change_state(state_list, m) != old_state_list:
                                moves.append(m)

                # 黑士
                elif state_list[y][x] == '黑士' and current_player_color == '黑':
                    for i in range(-1, 3, 2):
                        to_y = y + i
                        to_x = x + i
                        if check_r(to_y, to_x) and check_self(state_list[to_y][to_x], current_player_color='黑') \
                                and to_y >= 7 and 3 <= to_x <= 5:
                            m = str(y) + str(x) + str(to_y) + str(to_x)
                            if change_state(state_list, m) != old_state_list:
                                moves.append(m)
                        to_y = y + i
                        to_x = x - i
                        if check_r(to_y, to_x) and check_self(state_list[to_y][to_x], current_player_color='黑') \
                                and to_y >= 7 and 3 <= to_x <= 5:
                            m = str(y) + str(x) + str(to_y) + str(to_x)
                            if change_state(state_list, m) != old_state_list:
                                moves.append(m)

                # 红士
                elif state_list[y][x] == '红士' and current_player_color == '红':
                    for i in range(-1, 3, 2):
                        to_y = y + i
                        to_x = x + i
                        if check_r(to_y, to_x) and check_self(state_list[to_y][to_x], current_player_color='红') \
                                and to_y <= 2 and 3 <= to_x <= 5:
                            m = str(y) + str(x) + str(to_y) + str(to_x)
                            if change_state(state_list, m) != old_state_list:
                                moves.append(m)
                        to_y = y + i
                        to_x = x - i
                        if check_r(to_y, to_x) and check_self(state_list[to_y][to_x], current_player_color='红') \
                                and to_y <= 2 and 3 <= to_x <= 5:
                            m = str(y) + str(x) + str(to_y) + str(to_x)
                            if change_state(state_list, m) != old_state_list:
                                moves.append(m)

                # 黑帅
                elif state_list[y][x] == '黑帅':
                    k_x = x
                    k_y = y
                    if current_player_color == '黑':
                        for i in range(2):
                            for sign in range(-1, 2, 2):
                                j = 1 - i
                                to_y = y + i * sign
                                to_x = x + j * sign

                                if check_r(to_y, to_x) and check_self(
                                        state_list[to_y][to_x],
                                        current_player_color='黑') and to_y >= 7 and 3 <= to_x <= 5:
                                    m = str(y) + str(x) + str(to_y) + str(to_x)
                                    if change_state(state_list, m) != old_state_list:
                                        moves.append(m)

                # 红帅
                elif state_list[y][x] == '红帅':
                    K_x = x
                    K_y = y
                    if current_player_color == '红':
                        for i in range(2):
                            for sign in range(-1, 2, 2):
                                j = 1 - i
                                to_y = y + i * sign
                                to_x = x + j * sign

                                if check_r(to_y, to_x) and check_self(
                                        state_list[to_y][to_x],
                                        current_player_color='红') and to_y <= 2 and 3 <= to_x <= 5:
                                    m = str(y) + str(x) + str(to_y) + str(to_x)
                                    if change_state(state_list, m) != old_state_list:
                                        moves.append(m)

                # 黑炮
                elif state_list[y][x] == '黑炮' and current_player_color == '黑':
                    to_y = y
                    hits = False
                    for to_x in range(x - 1, -1, -1):
                        m = str(y) + str(x) + str(to_y) + str(to_x)
                        if hits is False:
                            if state_list[to_y][to_x] != '　　':
                                hits = True
                            else:
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                        else:
                            if state_list[to_y][to_x] != '　　':
                                if '红' in state_list[to_y][to_x]:
                                    if change_state(state_list, m) != old_state_list:
                                        moves.append(m)
                                break
                    hits = False
                    for to_x in range(x + 1, 9):
                        m = str(y) + str(x) + str(to_y) + str(to_x)
                        if hits is False:
                            if state_list[to_y][to_x] != '　　':
                                hits = True
                            else:
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                        else:
                            if state_list[to_y][to_x] != '　　':
                                if '红' in state_list[to_y][to_x]:
                                    if change_state(state_list, m) != old_state_list:
                                        moves.append(m)
                                break

                    to_x = x
                    hits = False
                    for to_y in range(y - 1, -1, -1):
                        m = str(y) + str(x) + str(to_y) + str(to_x)
                        if hits is False:
                            if state_list[to_y][to_x] != '　　':
                                hits = True
                            else:
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                        else:
                            if state_list[to_y][to_x] != '　　':
                                if '红' in state_list[to_y][to_x]:
                                    if change_state(state_list, m) != old_state_list:
                                        moves.append(m)
                                break
                    hits = False
                    for to_y in range(y + 1, 10):
                        m = str(y) + str(x) + str(to_y) + str(to_x)
                        if hits is False:
                            if state_list[to_y][to_x] != '　　':
                                hits = True
                            else:
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                        else:
                            if state_list[to_y][to_x] != '　　':
                                if '红' in state_list[to_y][to_x]:
                                    if change_state(state_list, m) != old_state_list:
                                        moves.append(m)
                                break

                # 红炮
                elif state_list[y][x] == '红炮' and current_player_color == '红':
                    to_y = y
                    hits = False
                    for to_x in range(x - 1, -1, -1):
                        m = str(y) + str(x) + str(to_y) + str(to_x)
                        if hits is False:
                            if state_list[to_y][to_x] != '　　':
                                hits = True
                            else:
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                        else:
                            if state_list[to_y][to_x] != '　　':
                                if '黑' in state_list[to_y][to_x]:
                                    if change_state(state_list, m) != old_state_list:
                                        moves.append(m)
                                break
                    hits = False
                    for to_x in range(x + 1, 9):
                        m = str(y) + str(x) + str(to_y) + str(to_x)
                        if hits is False:
                            if state_list[to_y][to_x] != '　　':
                                hits = True
                            else:
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                        else:
                            if state_list[to_y][to_x] != '　　':
                                if '黑' in state_list[to_y][to_x]:
                                    if change_state(state_list, m) != old_state_list:
                                        moves.append(m)
                                break

                    to_x = x
                    hits = False
                    for to_y in range(y - 1, -1, -1):
                        m = str(y) + str(x) + str(to_y) + str(to_x)
                        if hits is False:
                            if state_list[to_y][to_x] != '　　':
                                hits = True
                            else:
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                        else:
                            if state_list[to_y][to_x] != '　　':
                                if '黑' in state_list[to_y][to_x]:
                                    if change_state(state_list, m) != old_state_list:
                                        moves.append(m)
                                break
                    hits = False
                    for to_y in range(y + 1, 10):
                        m = str(y) + str(x) + str(to_y) + str(to_x)
                        if hits is False:
                            if state_list[to_y][to_x] != '　　':
                                hits = True
                            else:
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                        else:
                            if state_list[to_y][to_x] != '　　':
                                if '黑' in state_list[to_y][to_x]:
                                    if change_state(state_list, m) != old_state_list:
                                        moves.append(m)
                                break

                # 黑兵
                elif state_list[y][x] == '黑兵' and current_player_color == '黑':
                    to_y = y - 1
                    to_x = x
                    if check_r(to_y, to_x) and check_self(state_list[to_y][to_x], current_player_color='黑'):
                        m = str(y) + str(x) + str(to_y) + str(to_x)
                        if change_state(state_list, m) != old_state_list:
                            moves.append(m)
                    # 小兵过河
                    if y < 5:
                        to_y = y
                        to_x = x + 1
                        if check_r(to_y, to_x) and check_self(state_list[to_y][to_x], current_player_color='黑'):
                            m = str(y) + str(x) + str(to_y) + str(to_x)
                            if change_state(state_list, m) != old_state_list:
                                moves.append(m)
                        to_x = x - 1
                        if check_r(to_y, to_x) and check_self(state_list[to_y][to_x], current_player_color='黑'):
                            m = str(y) + str(x) + str(to_y) + str(to_x)
                            if change_state(state_list, m) != old_state_list:
                                moves.append(m)

                # 红兵
                elif state_list[y][x] == '红兵' and current_player_color == '红':
                    to_y = y + 1
                    to_x = x
                    if check_r(to_y, to_x) and check_self(state_list[to_y][to_x], current_player_color='红'):
                        m = str(y) + str(x) + str(to_y) + str(to_x)
                        if change_state(state_list, m) != old_state_list:
                            moves.append(m)
                    if y > 4:
                        to_y = y
                        to_x = x + 1
                        if check_r(to_y, to_x) and check_self(state_list[to_y][to_x], current_player_color='红'):
                            m = str(y) + str(x) + str(to_y) + str(to_x)
                            if change_state(state_list, m) != old_state_list:
                                moves.append(m)
                        to_x = x - 1
                        if check_r(to_y, to_x) and check_self(state_list[to_y][to_x], current_player_color='红'):
                            m = str(y) + str(x) + str(to_y) + str(to_x)
                            if change_state(state_list, m) != old_state_list:
                                moves.append(m)

    # 判定将帅面对面
    if K_x is not None and k_x is not None and K_x == k_x:
        face_to_face = True
        for i in range(K_y + 1, k_y, 1):
            if state_list[i][K_x] != '　　':
                face_to_face = False

    # 直接吃掉对方帅或将
    if face_to_face is True:
        if current_player_color == '黑':
            m = str(k_y) + str(k_x) + str(K_y) + str(K_x)
            if change_state(state_list, m) != old_state_list:
                moves.append(m)
        else:
            m = str(K_y) + str(K_x) + str(k_y) + str(k_x)
            if change_state(state_list, m) != old_state_list:
                moves.append(m)

    moves_id = []
    for move in moves:
        moves_id.append(move_action_2_move_id[move])
    return moves_id


# 棋盘逻辑控制
class Board(object):

    def __init__(self):
        self.state_list = copy.deepcopy(init_state_list)
        self.game_start = False
        self.winner = None
        self.state_deque = copy.deepcopy(state_deque)

    # 初始化棋盘的方法
    def init_board(self, start_player=1):  # 传入先手玩家的id
        # 红方先手
        self.start_player = start_player

        if start_player == 1:
            self.id_2_color = {1: '红', 2: '黑'}
            self.color_2_id = {'红': 1, '黑': 2}
            self.backhand_player = 2
        elif start_player == 2:
            self.id_2_color = {2: '红', 1: '黑'}
            self.color_2_id = {'红': 2, '黑': 1}
            self.backhand_player = 1
        # 当前手玩家
        self.current_player_color = self.id_2_color[start_player]  # 红
        self.current_player_id = self.color_2_id['红']
        # 初始化棋盘状态
        self.state_list = copy.deepcopy(init_state_list)
        self.state_deque = copy.deepcopy(state_deque)
        # 初始化最后落子位置
        self.last_move = -1
        # 记录游戏中吃子的回合数
        self.kill_action = 0
        self.game_start = False
        self.action_count = 0  # 游戏动作计数器
        self.winner = None

    @property
    # 获的当前盘面的所有合法走子集合
    def availables(self):
        return get_legal_moves(self.state_deque, self.current_player_color)

    # 从当前玩家的视角返回棋盘状态，current_state_array: [9, 10, 9]
    def current_state(self):
        _current_state = np.zeros([9, 10, 9])
        # 使用9个平面来表示棋盘状态
        # 0-6个平面表示棋子位置，1代表红方棋子，-1代表黑方棋子, 队列最后一个盘面
        # 第7个平面表示对手player最近一步的落子位置，走子之前的位置为-1，走子之后的位置为1，其余全部是0
        # 第8个平面表示的是当前player是不是先手player，如果是先手player则整个平面全部为1，否则全部为0
        _current_state[:7] = state_list_2_state_array(self.state_deque[-1]).transpose([2, 0, 1])  # [7, 10, 9]

        if self.game_start:
            move = move_id_2_move_action[self.last_move]
            start_position = int(move[0]), int(move[1])
            end_position = int(move[2]), int(move[3])
            _current_state[7][start_position[0]][start_position[1]] = -1
            _current_state[7][end_position[0]][end_position[1]] = 1
        if self.action_count % 2 == 0:
            _current_state[8][:, :] = 1.0

        return _current_state

    # 根据move对棋盘状态做出改变
    def make_move(self, move):
        self.game_start = True  # 游戏开始
        self.action_count += 1  # 移动次数加1
        move_action = move_id_2_move_action[move]
        start_y, start_x = int(move_action[0]), int(move_action[1])
        end_y, end_x = int(move_action[2]), int(move_action[3])
        state_list = copy.deepcopy(self.state_deque[-1])
        # 判断是否吃子
        if state_list[end_y][end_x] != '　　':
            # 如果吃掉对方的帅，则返回当前的current_player胜利
            self.kill_action = 0
            if self.current_player_color == '黑' and state_list[end_y][end_x] == '红帅':
                self.winner = self.color_2_id['黑']
            elif self.current_player_color == '红' and state_list[end_y][end_x] == '黑帅':
                self.winner = self.color_2_id['红']
        else:
            self.kill_action += 1
        # 更改棋盘状态
        state_list[end_y][end_x] = state_list[start_y][start_x]
        state_list[start_y][start_x] = '　　'
        self.current_player_color = '黑' if self.current_player_color == '红' else '红'  # 改变当前玩家
        self.current_player_id = 1 if self.current_player_id == 2 else 2
        # 记录最后一次移动的位置
        self.last_move = move
        self.state_deque.append(state_list)

    # 是否产生赢家
    def has_a_winner(self):
        # 红胜 黑方胜 平局
        if self.winner is not None:
            return True, self.winner
        elif self.kill_action >= CONFIG['kill_action']:  # 平局先手判负
            return True, self.backhand_player
        return False, -1

    # 检查当前棋局是否结束
    def game_end(self):
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif self.kill_action >= CONFIG['kill_action']:  # 平局，没有赢家
            return True, -1
        return False, -1

    def get_current_player_color(self):
        return self.current_player_color

    def get_current_player_id(self):
        return self.current_player_id


# 在Board类基础上定义Game类，该类用于启动并控制一整局对局的完整流程，收集对局数据，进行棋盘展示
class Game(object):

    def __init__(self, board):
        self.board = board

    def graphic(self, board, player1_color, player2_color):
        print('player1 take: ', player1_color)
        print('player2 take: ', player2_color)
        print_board(state_list_2_state_array(board.state_deque[-1]))

    # 人机对战，人人对战
    def start_play(self, player1, player2, start_player=1, is_shown=1):
        if start_player not in (1, 2):
            raise Exception('start_player should be either 1 (player1 first) '
                            'or 2 (player2 first)')
        self.board.init_board(start_player)  # 初始化棋盘
        p1, p2 = 1, 2
        player1.set_player_ind(1)
        player2.set_player_ind(2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)

        while True:
            current_player = self.board.get_current_player_id()  # 红子对应的玩家id
            player_in_turn = players[current_player]  # 决定当前玩家的代理
            move = player_in_turn.get_action(self.board)  # 当前玩家代理拿到动作
            self.board.make_move(move)  # 棋盘做出改变
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if winner != -1:
                    print("游戏结束，胜者是：", players[winner])
                else:
                    print("游戏结束，平局。")
                return winner

    # 使用蒙特卡洛树搜索开始自我对弈，存储游戏状态（状态，蒙特卡洛落子概率，胜负手）三元组用于神经网络训练
    def start_self_play(self, player, is_shown=False, temp=1e-3):
        self.board.init_board()  # 初始化棋盘
        p1, p2 = 1, 2
        states, mcts_probs, current_players = [], [], []
        # 开始自我对弈
        _count = 0
        while True:
            _count += 1
            if _count % 20 == 0:
                start_time = time.time()
                move, move_probs = player.get_action(self.board,
                                                   temp=temp,
                                                   return_prob=1)
                print('走一步要花: ', time.time() - start_time)
            else:
                move, move_probs = player.get_action(self.board,
                                                   temp=temp,
                                                   return_prob=1)
            # 保存自我对弈的数据
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player_id)
            # 执行一步落子
            self.board.make_move(move)
            end, winner = self.board.game_end()
            if end:
                # 从每一个状态state对应的玩家的视角保存胜负信息
                winner_z = np.zeros(len(current_players))
                if winner != -1:
                    winner_z[np.array(current_players) == winner] = 1.0
                    winner_z[np.array(current_players) != winner] = -1.0
                # 重置蒙特卡洛根节点
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("游戏结束，胜者是：", winner)
                    else:
                        print('游戏结束，平局。')

                return winner, zip(states, mcts_probs, winner_z)
