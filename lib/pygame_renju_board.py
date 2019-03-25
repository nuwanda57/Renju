import pygame
import keras

from board import Board
from config import BLACK, WHITE, change_color, GameState, BOARD_SIZE
from MCTS import MCTS


PATH_TO_THE_MODEL = './../current_best/the_main_player.h5'
GRID_SIZE = 40
INIT_COORDINATES = 30, 50
BLACK_COLOR = 0, 0, 0
WHITE_COLOR = 255, 255, 255
MCTS_DEPTH = 100


class RenjuBoard:
    def __init__(self):
        self.game_over = False
        self.winner = None
        self._renju_agent = keras.models.load_model(PATH_TO_THE_MODEL)
        self._mcts = MCTS(self._renju_agent, MCTS_DEPTH)
        self._board = Board()
        self._color = BLACK

    def make_first_move(self, draw_callback):
        probs = self._mcts.get_new_actions_probs(board=Board(self._board), caution=True)
        probs = list(enumerate(probs))
        probs.sort(key=lambda p: p[1], reverse=True)
        probs = [(p[0] // 15, p[0] % 15) for p in probs]

        for nn_move in probs:
            if self._board.is_legal(nn_move):
                self._board.execute_move(nn_move, self._color)
                self._color = change_color(self._color)
                draw_callback()
                break

    def handle_key_event(self, event, draw_callback):
        origin_x, origin_y = INIT_COORDINATES[0] - GRID_SIZE // 2, INIT_COORDINATES[1] - GRID_SIZE // 2
        size = BOARD_SIZE * GRID_SIZE
        pos = event.pos
        if origin_x <= pos[0] <= origin_x + size and origin_y <= pos[1] <= origin_y + size:
            if not self.game_over:
                x, y = pos[0] - origin_x, pos[1] - origin_y
                row, col = int(y // GRID_SIZE), int(x // GRID_SIZE)
                move = (row, col)
                if self._board.is_legal(move):
                    self._board.execute_move(move, self._color)
                    if self._board.check_state(move, self._color) != GameState.InProgress:
                        self.winner = self._color
                        self.game_over = True
                    self._color = change_color(self._color)
                    if not self.game_over:
                        probs = self._mcts.get_new_actions_probs(board=Board(self._board), his_move=move, caution=True)
                        probs = list(enumerate(probs))
                        probs.sort(key=lambda p: p[1], reverse=True)
                        probs = [(p[0] // 15, p[0] % 15) for p in probs]
                        for nn_move in probs:
                            if self._board.is_legal(nn_move):
                                self._board.execute_move(nn_move, self._color)
                                if self._board.check_state(nn_move, self._color) != GameState.InProgress:
                                    self.winner = self._color
                                    self.game_over = True
                                self._color = change_color(self._color)
                                draw_callback()
                                break
                    return 1
            else:
                return 0
        return 1

    def draw(self, screen):
        pygame.draw.rect(screen, (185, 122, 87),
                         [INIT_COORDINATES[0] - GRID_SIZE // 2, INIT_COORDINATES[1] - GRID_SIZE // 2,
                          BOARD_SIZE * GRID_SIZE, BOARD_SIZE * GRID_SIZE], 0)

        for row in range(BOARD_SIZE):
            y = INIT_COORDINATES[1] + row * GRID_SIZE
            pygame.draw.line(
                screen, BLACK_COLOR, [INIT_COORDINATES[0], y],
                [INIT_COORDINATES[0] + GRID_SIZE * (BOARD_SIZE - 1), y], 2
            )

        for col in range(BOARD_SIZE):
            x = INIT_COORDINATES[0] + col * GRID_SIZE
            pygame.draw.line(
                screen, BLACK_COLOR, [x, INIT_COORDINATES[1]],
                [x, INIT_COORDINATES[1] + GRID_SIZE * (BOARD_SIZE - 1)], 2
            )

        cells = self._board.get_board()
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = cells[row][col]
                if piece != 0:
                    if piece == BLACK:
                        color = BLACK_COLOR
                    else:
                        color = WHITE_COLOR

                    x = INIT_COORDINATES[0] + col * GRID_SIZE
                    y = INIT_COORDINATES[1] + row * GRID_SIZE
                    pygame.draw.circle(screen, color, [x, y], GRID_SIZE // 2)
