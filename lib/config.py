from enum import Enum

BOARD_SIZE = 15


# constants used both for players and moves
WHITE = -1
BLACK = 1


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

