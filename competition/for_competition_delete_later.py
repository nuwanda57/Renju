import os
import random
import concurrent.futures
import enum
import itertools
import logging
import numpy
import sys
import time
import traceback


POS_TO_LETTER = 'abcdefghjklmnop'
LETTER_TO_POS = {letter: pos for pos, letter in enumerate(POS_TO_LETTER)}

def to_move(pos):
    return POS_TO_LETTER[pos[1]] + str(pos[0] + 1)

def to_pos(move):
    return int(move[1:]) - 1, LETTER_TO_POS[move[0]]

def list_positions(board, player):
    return numpy.vstack(numpy.nonzero(board == player)).T

def sequence_length(board, I, J, value):
    length = 0

    for i, j in zip(I, J):
        if board[i, j] != value:
            break
        length += 1

    return length


def check_horizontal(board, pos):
    player = board[pos]
    if not player:
        return False

    i, j = pos
    length = 1

    length += sequence_length(
        board,
        itertools.repeat(i),
        range(j + 1, min(j + Game.line_length, Game.width)),
        player
    )

    length += sequence_length(
        board,
        itertools.repeat(i),
        range(j - 1, max(j - Game.line_length, -1), -1),
        player
    )

    return length >= Game.line_length

def check_vertical(board, pos):
    player = board[pos]
    if not player:
        return False

    i, j = pos
    length = 1

    length += sequence_length(
        board,
        range(i + 1, min(i + Game.line_length, Game.height)),
        itertools.repeat(j),
        player
    )

    length += sequence_length(
        board,
        range(i - 1, max(i - Game.line_length, -1), -1),
        itertools.repeat(j),
        player
    )

    return length >= Game.line_length

def check_main_diagonal(board, pos):
    player = board[pos]
    if not player:
        return False

    i, j = pos
    length = 1

    length += sequence_length(
        board,
        range(i + 1, min(i + Game.line_length, Game.height)),
        range(j + 1, min(j + Game.line_length, Game.width)),
        player
    )

    length += sequence_length(
        board,
        range(i - 1, max(i - Game.line_length, -1), -1),
        range(j - 1, max(j - Game.line_length, -1), -1),
        player
    )

    return length >= Game.line_length

def check_side_diagonal(board, pos):
    player = board[pos]
    if not player:
        return False

    i, j = pos
    length = 1

    length += sequence_length(
        board,
        range(i - 1, max(i - Game.line_length, -1), -1),
        range(j + 1, min(j + Game.line_length, Game.width)),
        player
    )

    length += sequence_length(
        board,
        range(i + 1, min(i + Game.line_length, Game.height)),
        range(j - 1, max(j - Game.line_length, -1), -1),
        player
    )

    return length >= Game.line_length

def check(board, pos):
    if not board[pos]:
        return False

    return check_vertical(board, pos) \
        or check_horizontal(board, pos) \
        or check_main_diagonal(board, pos) \
        or check_side_diagonal(board, pos)


class Player(enum.IntEnum):
    NONE = 0
    BLACK = -1
    WHITE = 1

    def another(self):
        return Player(-self)

    def __repr__(self):
        if self == Player.BLACK:
            return 'black'
        elif self == Player.WHITE:
            return 'white'
        else:
            return 'none'

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def from_str(s):
        if s == 'black':
            return Player.BLACK
        elif s == 'white':
            return Player.WHITE
        else:
            return Player.NONE


class Game:
    width, height = 15, 15
    shape = (width, height)
    line_length = 5

    def __init__(self):
        self._result = Player.NONE
        self._player = Player.BLACK
        self._board = numpy.full(self.shape, Player.NONE, dtype=numpy.int8)
        self._positions = list()

    def __bool__(self):
        return self.result() == Player.NONE and \
            len(self._positions) < self.width * self.height

    def move_n(self):
        return len(self._positions)

    def player(self):
        return self._player

    def result(self):
        return self._result

    def board(self):
        return self._board

    def positions(self, player=Player.NONE):
        if not player:
            return self._positions

        begin = 0 if player == Player.BLACK else 1
        return self._positions[begin::2]

    def dumps(self):
        return ' '.join(map(to_move, self._positions))

    @staticmethod
    def loads(dump):
        game = Game()
        for pos in map(to_pos, dump.split()):
            game.move(pos)
        return game


    def is_posible_move(self, pos):
        return 0 <= pos[0] < self.height \
            and 0 <= pos[1] < self.width \
            and not self._board[pos]

    def move(self, pos):
        assert self.is_posible_move(pos), 'impossible pos: {pos}'.format(pos=pos)

        self._positions.append(pos)
        self._board[pos] = self._player

        if not self._result and check(self._board, pos):
            self._result = self._player
            return

        self._player = self._player.another()

MAX_MOVE_N = Game.width * Game.height

def loop(game, black, white, max_move_n=MAX_MOVE_N , timeout=None):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        for agent in itertools.cycle([black, white]):
            if not game or game.move_n() >= max_move_n:
                break

            future = executor.submit(lambda game: agent.move(game), game)
            pos = to_pos(future.result(timeout=timeout))
            game.move(pos)

            yield game


def run(black, white, max_move_n=60, timeout=3):
    game = Game()

    try:
        for game in loop(game, black, white, max_move_n=max_move_n, timeout=timeout):
            # logging.debug(game.dumps())
            pass

    except:
        logging.error('Error!', exc_info=True, stack_info=True)
        return game.player().another(), game.dumps()

    return game.result(), game.dumps()


from enum import Enum

BOARD_SIZE = 15


# constants used both for players and moves
WHITE = -1
BLACK = 1


import sys

def wait_for_game_update():
    if not sys.stdin.closed:
        game_dumps = sys.stdin.readline()

        if game_dumps:
            return Game.loads(game_dumps)

    return None

def set_move(move):
    if sys.stdout.closed:
        return False

    sys.stdout.write(move + '\n')
    sys.stdout.flush()

    return True


def change_color(color):
    if color == WHITE:
        return BLACK
    if color == BLACK:
        return WHITE
    raise ValueError()


class GameState(Enum):
    InProgress = 1
    Draw = 2
    White = 3
    Black = 4



'''
Class representing game board.
'''
# TODO remove colors from arguments of some methods (use self._player)


import numpy as np
from copy import deepcopy, copy
from random import choice


class Board(object):
    @staticmethod
    def build_initial_board():
        return Board()

    @staticmethod
    def build_board_from_np(np_board):
        board = Board()
        board._size = 15
        ones = 0
        minus_ones = 0
        for row in range(15):
            for col in range(15):
                if np_board[row][col] == 1:
                    board._cells[row][col] = 1
                    board._legal_moves.remove((row, col))
                    ones += 2
                elif np_board[row][col] == -1:
                    board._cells[row][col] = -1
                    board._legal_moves.remove((row, col))
                    minus_ones += 1
        board._state = GameState.InProgress
        board._hash = None
        if ones == minus_ones:
            board._player = BLACK
        else:
            board._player = WHITE
        return board

    def __init__(self, other=None):
        if other:
            self._size = other._size
            self._cells = deepcopy(other._cells)
            self._legal_moves = deepcopy(other._legal_moves)
            self._state = other._state
            self._hash = other._hash
            self._player = other._player
        else:
            self._size = BOARD_SIZE
            self._cells = [list()] * self._size
            for i in range(self._size):
                self._cells[i] = [0] * self._size
            self._cells = np.array(self._cells)
            self._legal_moves = set(
                (row, col) for col in range(self._size) for row in range(self._size)
            )
            self._state = GameState.InProgress
            self._hash = None
            self._player = BLACK

    def __getitem__(self, index):
        return self._cells[index]

    def size(self):
        return self._size

    def get_player(self):
        return self._player

    def get_legal_moves(self):
        return list(self._legal_moves)

    def get_legal_moves_set(self):
        return copy(self._legal_moves)

    def get_next_board(self, action):
        next_board = Board(self)
        next_board.execute_move((action // BOARD_SIZE, action % BOARD_SIZE), self._player)
        return next_board, next_board._player

    def has_legal_moves(self):
        return len(self._legal_moves) != 0

    def get_legal_actions(self):
        return np.array(list(map(lambda x: x[0] * self.size() + x[1], self._legal_moves)))

    def get_board(self):
        return self._cells

    def get_state(self):
        return self._state

    def is_legal(self, move):
        return move in self._legal_moves

    def get_random_move(self):
        return choice(tuple(self._legal_moves))

    def rehash(self):
        self._hash = ''
        for row in range(BOARD_SIZE):
            for column in range(BOARD_SIZE):
                if self._cells[row][column] != 0:
                    self._hash += '{}-{}-{}-'.format(str(row), str(column), str(self._cells[row][column]))
        return copy(self._hash)

    def hash(self):
        if self._hash is None:
            self._hash = ''
            for row in range(BOARD_SIZE):
                for column in range(BOARD_SIZE):
                    if self._cells[row][column] != 0:
                        self._hash += '{}-{}-{}-'.format(str(row), str(column), str(self._cells[row][column]))
        return copy(self._hash)

    def execute_move(self, move, color, no_hash=False):
        # if no_hash == True -> no hash recalculation will be performed
        assert self._cells[move[0]][move[1]] == 0 # empty cell
        assert color == 1 or color == -1
        assert color == self._player
        self._cells[move[0]][move[1]] = color
        if self._player == WHITE:
            self._player = BLACK
        else:
            self._player = WHITE
        self._legal_moves.remove(move)
        self.check_state(move, color)
        if no_hash:
            return
        self.rehash()

    def check_state(self, move, color):
        assert color == -self._player
        if self._state != GameState.InProgress:
            return self._state
        if not self.has_legal_moves():
            self._state = GameState.Draw
            return GameState.Draw
        if (
                self._check_row(move, color) or
                self._check_column(move, color) or
                self._check_diagonals(move, color)
        ):
            if color == WHITE:
                self._state = GameState.White
                return GameState.White
            if color == BLACK:
                self._state = GameState.Black
                return GameState.Black
            else:
                raise ValueError('unsupported GameState')
        return GameState.InProgress

    def check_sequence(self, move, color, check_cnt=5):
        if (
                self._check_row_sequence(move, color, check_cnt) is not None or
                self._check_column_sequence(move, color, check_cnt) is  not None or
                self._check_diagonals_direct_sequence(move, color, check_cnt) is not None or
                self._check_diagonals_indirect_sequence(move, color, check_cnt)
        ):
            return True
        return False

    def _check_row_sequence(self, move, color, check_cnt=5):
        cnt = 1
        row, col = move
        left = move
        right = move
        for i in range(col + 1, BOARD_SIZE):
            if self._cells[row][i] == color:
                right = (row, i)
                #print(right)
                cnt += 1
            else:
                break
        for i in range(col - 1, -1, -1):
            if self._cells[row][i] == color:
                left = (row, i)
                #print(left)
                cnt += 1
            else:
                break
        if cnt >= check_cnt:
            #print(move)
            #print(self._cells)
            #print(left, right)
            return left, right
        return None

    def _check_column_sequence(self, move, color, check_cnt=5):
        cnt = 1
        row, col = move
        up, down = move, move
        for i in range(row + 1, BOARD_SIZE):
            if self._cells[i][col] == color:
                up = i, col
                cnt += 1
            else:
                break
        for i in range(row - 1, -1, -1):
            if self._cells[i][col] == color:
                down = i, col
                cnt += 1
            else:
                break
        if cnt >= check_cnt:
            return down, up
        return None

    def _check_diagonals_direct_sequence(self, move, color, check_cnt=5):
        cnt1 = 1
        row, col = move
        down_left, up_right = move, move
        for i in range(1, 15):
            if row + i < BOARD_SIZE and col + i < BOARD_SIZE and self._cells[row + i][col + i] == color:
                up_right = (row + i, col + i)
                cnt1 += 1
            else:
                break
        for i in range(1, 15):
            if row - i > -1 and col - i > -1 and self._cells[row - i][col - i] == color:
                down_left = (row - i, col - i)
                cnt1 += 1
            else:
                break
        if cnt1 >= check_cnt:
            return down_left, up_right
        return None

    def _check_diagonals_indirect_sequence(self, move, color, check_cnt=5):
        cnt2 = 1
        row, col = move
        up_left, down_right = move, move
        for i in range(1, 15):
            if row + i < BOARD_SIZE and col - i > -1 and self._cells[row + i][col - i] == color:
                up_left = (row + i, col - i)
                cnt2 += 1
                continue
            break
        for i in range(1, 15):
            if row - i > -1 and col + i < BOARD_SIZE and self._cells[row - i][col + i] == color:
                down_right = (row - i, col + i)
                cnt2 += 1
                continue
            break
        if cnt2 >= check_cnt:
            return up_left, down_right
        return None

    def _check_row(self, move, color, check_cnt=5):
        assert color == -self._player
        cnt = 1
        row, col = move
        for i in range(col + 1, BOARD_SIZE):
            if self._cells[row][i] == color:
                cnt += 1
                continue
            break
        for i in range(col - 1, -1, -1):
            if self._cells[row][i] == color:
                cnt += 1
                continue
            break
        return cnt >= check_cnt

    def _check_column(self, move, color, check_cnt=5):
        assert color == -self._player
        cnt = 1
        row, col = move
        for i in range(row + 1, BOARD_SIZE):
            if self._cells[i][col] == color:
                cnt += 1
                continue
            break
        for i in range(row - 1, -1, -1):
            if self._cells[i][col] == color:
                cnt += 1
                continue
            break
        return cnt >= check_cnt

    def _check_diagonals(self, move, color, check_cnt=5):
        assert color == -self._player
        cnt1, cnt2 = 1, 1
        row, col = move
        for i in range(1, 15):
            if row + i < BOARD_SIZE and col + i < BOARD_SIZE and self._cells[row + i][col + i] == color:
                cnt1 += 1
                continue
            break
        for i in range(1, 15):
            if row - i > -1 and col - i > -1 and self._cells[row - i][col - i] == color:
                cnt1 += 1
                continue
            break

        for i in range(1, 15):
            if row + i < BOARD_SIZE and col - i > -1 and self._cells[row + i][col - i] == color:
                cnt2 += 1
                continue
            break
        for i in range(1, 15):
            if row - i > -1 and col + i < BOARD_SIZE and self._cells[row - i][col + i] == color:
                cnt2 += 1
                continue
            break

        return cnt1 >= check_cnt or cnt2 >= check_cnt


import math
import numpy as np
from enum import Enum



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

        # if cheat:
        #     moves_maybe = self._cheat(board)
        #     if moves_maybe is not None:
        #         return moves_maybe

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


import keras
PATH_TO_THE_MODEL = './current_best/the_main_player.h5'
MCTS_DEPTH = 100


def my_move_to_pos(move):
    #logging.info('[nuwanda]::move {}'.format(move))
    pos = POS_TO_LETTER[move[1]] + str(move[0] + 1)
    #logging.info('[nuwanda]::move {}, pos: {}'.format(move, pos))
    return pos


def my_choose_random_move(their_board, mcts):
    board = Board.build_board_from_np(their_board)
    probs = mcts.get_new_actions_probs(board=board, his_move=None, caution=True)
    probs = list(enumerate(probs))
    probs.sort(key=lambda p: p[1], reverse=True)
    probs = [(p[0] // 15, p[0] % 15) for p in probs]
    #logging.info('[nuwanda]::probs {}'.format(probs))
    move = None
    for nn_move in probs:
        if board.is_legal(nn_move):
            move = nn_move
            break
    pos = my_move_to_pos(move)
    #logging.info('[nuwanda]::my {}'.format(pos))
    return pos


def main():
    pid = os.getpid()
    LOG_FORMAT = str(pid) + ':%(levelname)s:%(asctime)s: %(message)s'

    logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG)
    logging.debug("Start dummy backend...")

    my_agent = keras.models.load_model(PATH_TO_THE_MODEL)
    my_mcts = MCTS(my_agent, MCTS_DEPTH)

    try:
        while True:
            logging.debug("Wait for game update...")
            game = wait_for_game_update()

            if not game:
                logging.debug("Game is over!")
                return

            logging.debug('Game: [%s]', game.dumps())
            # my_board = Board.build_board_from_np(game.board())
            # logging.info(my_board.get_board())
            move = my_choose_random_move(game.board(), my_mcts)

            if not set_move(move):
                logging.error("Impossible set move!")
                return

            logging.debug('Random move: %s', move)

    except:
        logging.error('Error!', exc_info=True, stack_info=True)


if __name__ == "__main__":
    main()

