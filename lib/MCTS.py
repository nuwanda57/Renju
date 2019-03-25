import math
import numpy as np
from enum import Enum

from config import GameState, BOARD_SIZE


class CautionType(Enum):
    MY_WIN = 1
    HIS_WIN = 2
    THE_ONLY_ONE_RIGHT = 3
    BAD_THREE_SITUATION = 4
    HIS_THREE_SITUATION = 5


EPS = 1e-8


# terminology: state(s) = hash(board(b))
class MCTS(object):
    def __init__(self, nnet, simulations_cnt):
        self._nnet = nnet
        self._simulations_cnt = simulations_cnt
        self._action_rewards = {}  # (s, a) -> reward
        self._take_action_from_state_cnt = {}  # (s, a) -> number of times action a was taken from s
        self._visit_state_cnt = {}  # s -> number of times s was visited
        self._nn_policies = {}  # (s, a) -> nn.predict(b)[0][a]

        self._cheat_stage = 0
        self._game_results = {}  # s -> game result

    def get_new_actions_probs(self, board, choose_one_action=False, his_move=None, cheat=False, caution=True):
        HIS_THREES = None
        if caution:
            move_maybe = self._caution(board)
            if move_maybe is not None:
                caution_type, info = move_maybe
                if caution_type == CautionType.THE_ONLY_ONE_RIGHT:
                    move = info
                    best_action = move[0] * 15 + move[1]
                    new_probs = [0] * (BOARD_SIZE * BOARD_SIZE)
                    new_probs[best_action] = 1
                    return new_probs
                if caution_type == CautionType.HIS_THREE_SITUATION:
                    HIS_THREES = info

        if cheat:
            moves_maybe = self._cheat(board)
            if moves_maybe is not None:
                return moves_maybe

        for i in range(self._simulations_cnt):
            self.search(board)

        s = board.hash()
        counts = [self._take_action_from_state_cnt[(s, a)]
                  if (s, a) in self._take_action_from_state_cnt else 0 for a in range(BOARD_SIZE * BOARD_SIZE)]

        if choose_one_action:
            best_action = np.argmax(counts)
            best_move = best_action // BOARD_SIZE, best_action % BOARD_SIZE
            if HIS_THREES and not best_move in HIS_THREES:
                for m in HIS_THREES:
                    best_move = m
                    best_action = best_move[0] * BOARD_SIZE + best_move[1]
                    break
            new_probs = [0] * len(counts)
            new_probs[best_action] = 1
            return new_probs

        best_action = np.argmax(counts)
        best_move = best_action // BOARD_SIZE, best_action % BOARD_SIZE
        if HIS_THREES and not best_move in HIS_THREES:
            counts = [0] * (BOARD_SIZE * BOARD_SIZE)
            for m in HIS_THREES:
                best_move = m
                best_action = best_move[0] * BOARD_SIZE + best_move[1]
                counts[best_action] = 1
        if sum(counts) != 0:
            new_probs = [float(x) / sum(counts) for x in counts]
        else:
            print('WARNING:::sum of counts = 0')
            new_probs = [float(0) for x in counts]
        return new_probs

    def search(self, board):
        s = board.hash()

        if s not in self._game_results:
            self._game_results[s] = board.get_state()
        if self._game_results[s] != GameState.InProgress:
            # game ended, so GameState is Draw or White if current player is BLACK
            # or Black if current player is WHITE
            if self._game_results[s] == GameState.Draw:
                return 0
            # If it's not draw then current player lost (state is loosing state), so the player
            # that called search won
            return 1

        if s not in self._nn_policies:
            # new node:
            #   - add to the tree
            #   - set values
            prediction = self._nnet.predict(board.get_board().reshape(1, BOARD_SIZE, BOARD_SIZE))
            self._nn_policies[s], v = prediction[0][0], prediction[1][0]
            legal_moves_set = board.get_legal_moves_set()
            for i in range(len(self._nn_policies[s])):
                if (i // BOARD_SIZE, i % BOARD_SIZE) not in legal_moves_set:
                    self._nn_policies[s][i] = np.float64(0)
            sum_nn_policies_s = np.sum(self._nn_policies[s])
            if sum_nn_policies_s > 0:
                self._nn_policies[s] /= sum_nn_policies_s
            else:
                for i in range(len(self._nn_policies[s])):
                    if (i // BOARD_SIZE, i % BOARD_SIZE) in legal_moves_set:
                        self._nn_policies[s][i] = np.float64(1)
                self._nn_policies[s] /= np.sum(self._nn_policies[s])
            self._visit_state_cnt[s] = 0
            return -v

        # node has been already visited
        legal_moves_set = board.get_legal_moves_set()
        cur_best_ucb = -float('inf')
        best_act = None

        # pick the action with the highest upper confidence bound
        for move in legal_moves_set:
            a = move[0] * BOARD_SIZE + move[1]
            if (s, a) in self._action_rewards:
                cur_ucb =\
                    self._action_rewards[(s, a)] + (
                            self._nn_policies[s][a] *
                            math.sqrt(self._visit_state_cnt[s]) /
                            (1 + self._take_action_from_state_cnt[(s, a)])
                    )
            else:
                cur_ucb = self._nn_policies[s][a] * math.sqrt(self._visit_state_cnt[s] + EPS)  # not visited
            if cur_ucb > cur_best_ucb:
                cur_best_ucb = cur_ucb
                best_act = a

        a = best_act
        if len(legal_moves_set) == 0:
            return 0
        assert a is not None
        assert (a // BOARD_SIZE, a % BOARD_SIZE) in legal_moves_set
        next_board, next_player = board.get_next_board(a)

        v = self.search(next_board)

        if (s, a) in self._action_rewards:
            self._action_rewards[(s, a)] =\
                (self._take_action_from_state_cnt[(s, a)] * self._action_rewards[(s, a)] + v) /\
                (self._take_action_from_state_cnt[(s, a)] + 1)
            self._take_action_from_state_cnt[(s, a)] += 1

        else:
            self._action_rewards[(s, a)] = v
            self._take_action_from_state_cnt[(s, a)] = 1

        self._visit_state_cnt[s] += 1
        # note that here is -v and not self._action_rewards((s, a))!
        # to understand look at formula for self._action_rewards((s, a))
        # it always uses new v as a reward of a new visited node
        # (with equal absolute value - backpropagate same absolute value)
        return -v

    def _cheat(self, board):
        if len(board.get_legal_moves_set()) == 225:
            move = (7, 14)
        elif len(board.get_legal_moves_set()) == 223:
            for i in range(15):
                if i != 7:
                    if not (i, 14) in board.get_legal_moves_set():
                        return None
            move = (8, 14)
        elif len(board.get_legal_moves_set()) == 221:
            for i in range(15):
                if i != 7 and i != 8:
                    if not (i, 14) in board.get_legal_moves_set():
                        return None
            move = (6, 14)
        else:
            return None
        new_probs = [float(0) for i in range(225)]
        new_probs[move[0] * 15 + move[1]] = 1
        return new_probs

    def _caution(self, board):
        color = board.get_player()
        move = self._maybe_get_my_win(board, color)
        if move is not None:
            return CautionType.THE_ONLY_ONE_RIGHT, move

        his_moves = self._maybe_get_his_win(board, -color)
        if his_moves and len(his_moves) == 1:
            for i in his_moves:
                return CautionType.THE_ONLY_ONE_RIGHT, i
        if his_moves and len(his_moves) > 1:
            return CautionType.HIS_WIN, his_moves

        my_three_situation = self._my_three_situation(board, color)
        if my_three_situation:
            return CautionType.THE_ONLY_ONE_RIGHT, my_three_situation

        necessary_save_moves = self._am_i_in_danger(board, -color)
        if necessary_save_moves and necessary_save_moves == CautionType.BAD_THREE_SITUATION:
            return CautionType.BAD_THREE_SITUATION, None
        if necessary_save_moves and len(necessary_save_moves) > 0:
            return CautionType.HIS_THREE_SITUATION, necessary_save_moves
        return None

    def _maybe_get_my_win(self, board, color):
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if board[row][col] != 0:
                    continue
                maybe = board.check_sequence((row, col), color, 5)
                if maybe:
                    return row, col
        return None

    def _maybe_get_his_win(self, board, his_color):
        his_win_pos = set()
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if board[row][col] != 0:
                    continue
                maybe = board.check_sequence((row, col), his_color, 5)
                if maybe:
                    his_win_pos.add((row, col))
        return his_win_pos

    def _my_three_situation(self, board, color):
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if board[row][col] != 0:
                    continue
                maybe = board._check_row_sequence((row, col), color, 4)
                if maybe:
                    m_from, m_to = maybe
                    if board.is_legal((m_from[0], m_from[1] - 1)):
                        if board.is_legal((m_to[0], m_to[1] + 1)):
                            win_move = row, col
                            return win_move
                maybe = board._check_column_sequence((row, col), color, 4)
                if maybe:
                    m_from, m_to = maybe
                    if board.is_legal((m_from[0] - 1, m_from[1])):
                        if board.is_legal((m_to[0] + 1, m_to[1])):
                            win_move = row, col
                            return win_move
                maybe = board._check_diagonals_direct_sequence((row, col), color, 4)
                if maybe:
                    m_from, m_to = maybe
                    if board.is_legal((m_from[0] - 1, m_from[1] - 1)):
                        if board.is_legal((m_to[0] + 1, m_to[1] + 1)):
                            win_move = row, col
                            return win_move
                maybe = board._check_diagonals_indirect_sequence((row, col), color, 4)
                if maybe:
                    m_from, m_to = maybe
                    if board.is_legal((m_from[0] + 1, m_from[1] - 1)):
                        if board.is_legal((m_to[0] - 1, m_to[1] + 1)):
                            win_move = row, col
                            return win_move
        return None

    def _am_i_in_danger(self, board, his_color):
        save_moves = set()
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if board[row][col] != 0:
                    continue
                maybe = board._check_row_sequence((row, col), his_color, 4)
                if maybe:
                    m_from, m_to = maybe
                    if board.is_legal((m_from[0], m_from[1] - 1)):
                        if board.is_legal((m_to[0], m_to[1] + 1)):
                            a1, a2, a3 = (m_from[0], m_from[1] - 1), (m_to[0], m_to[1] + 1), (row, col)
                            if len(save_moves) == 0:
                                save_moves.add(a1)
                                save_moves.add(a2)
                                save_moves.add(a3)
                            else:
                                if a1 in save_moves:
                                    save_moves.clear()
                                    save_moves.add(a1)
                                elif a2 in save_moves:
                                    save_moves.clear()
                                    save_moves.add(a2)
                                elif a3 in save_moves:
                                    save_moves.clear()
                                    save_moves.add(a3)
                                else:
                                    return CautionType.BAD_THREE_SITUATION
                maybe = board._check_column_sequence((row, col), his_color, 4)
                if maybe:
                    m_from, m_to = maybe
                    if board.is_legal((m_from[0] - 1, m_from[1])):
                        if board.is_legal((m_to[0] + 1, m_to[1])):
                            a1, a2, a3 = (m_from[0] - 1, m_from[1]), (m_to[0] + 1, m_to[1]), (row, col)
                            if len(save_moves) == 0:
                                save_moves.add(a1)
                                save_moves.add(a2)
                                save_moves.add(a3)
                            else:
                                if a1 in save_moves:
                                    save_moves.clear()
                                    save_moves.add(a1)
                                elif a2 in save_moves:
                                    save_moves.clear()
                                    save_moves.add(a2)
                                elif a3 in save_moves:
                                    save_moves.clear()
                                    save_moves.add(a3)
                                else:
                                    return CautionType.BAD_THREE_SITUATION
                maybe = board._check_diagonals_direct_sequence((row, col), his_color, 4)
                if maybe:
                    m_from, m_to = maybe
                    if board.is_legal((m_from[0] - 1, m_from[1] - 1)):
                        if board.is_legal((m_to[0] + 1, m_to[1] + 1)):
                            a1, a2, a3 = (m_from[0] - 1, m_from[1] - 1), (m_to[0] + 1, m_to[1] + 1), (row, col)
                            if len(save_moves) == 0:
                                save_moves.add(a1)
                                save_moves.add(a2)
                                save_moves.add(a3)
                            else:
                                if a1 in save_moves:
                                    save_moves.clear()
                                    save_moves.add(a1)
                                elif a2 in save_moves:
                                    save_moves.clear()
                                    save_moves.add(a2)
                                elif a3 in save_moves:
                                    save_moves.clear()
                                    save_moves.add(a3)
                                else:
                                    return CautionType.BAD_THREE_SITUATION
                maybe = board._check_diagonals_indirect_sequence((row, col), his_color, 4)
                if maybe:
                    m_from, m_to = maybe
                    if board.is_legal((m_from[0] + 1, m_from[1] - 1)):
                        if board.is_legal((m_to[0] - 1, m_to[1] + 1)):
                            a1, a2, a3 = (m_from[0] + 1, m_from[1] - 1), (m_to[0] - 1, m_to[1] + 1), (row, col)
                            if len(save_moves) == 0:
                                save_moves.add(a1)
                                save_moves.add(a2)
                                save_moves.add(a3)
                            else:
                                if a1 in save_moves:
                                    save_moves.clear()
                                    save_moves.add(a1)
                                elif a2 in save_moves:
                                    save_moves.clear()
                                    save_moves.add(a2)
                                elif a3 in save_moves:
                                    save_moves.clear()
                                    save_moves.add(a3)
                                else:
                                    return CautionType.BAD_THREE_SITUATION
        if len(save_moves) == 0:
            return None
        return save_moves
