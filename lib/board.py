'''
Class representing game board.
'''
# TODO remove colors from arguments of some methods (use self._player)


from config import BOARD_SIZE, GameState, WHITE, BLACK

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
                cnt += 1
            else:
                break
        for i in range(col - 1, -1, -1):
            if self._cells[row][i] == color:
                left = (row, i)
                cnt += 1
            else:
                break
        if cnt >= check_cnt:
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
