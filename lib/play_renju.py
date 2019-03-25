#!/usr/bin/python

import pygame

from pygame_renju_board import RenjuBoard
from config import BLACK, WHITE


END_GAME = -1
CONTINUE_GAME = 1
START_NEW = 2
BLACK_COLOR = 0, 0, 0
WHITE_COLOR = 255, 255, 255
INIT_IMAGE = 'thumbs-up2.png'
INIT_GREETING = "Hi There!\n    Let's play Renju!"
CHOOSING_PLAYER_TEXT = 'Please, choose color:\n             BLACK\n             WHITE'
CAPTION = '-----------------------R-E-N-J-U-----------------------'


def _render_multi_line(text, x, y, fsize, background, font, text_color, im_x=None, im_y=None, image=None):
    lines = text.splitlines()
    for i, l in enumerate(lines):
        background.blit(font.render(l, 0, text_color), (x, y + fsize * i))
    if image:
        background.blit(image, (im_x, im_y))


def _handle_initial_click(event):
    pos = event.pos
    return 600 <= pos[0] <= 1000 and 300 <= pos[1] <= 750


def _handle_choose_player_click(event):
    pos = event.pos
    if 400 <= pos[0] <= 800 and 250 <= pos[1] <= 350:
        return BLACK
    if 400 <= pos[0] <= 800 and 450 <= pos[1] <= 550:
        return WHITE
    return None


def _play_game(color=BLACK):
    screen = pygame.display.set_mode((1200, 800))

    renju_board = RenjuBoard()

    def update():
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return END_GAME
            elif e.type == pygame.MOUSEBUTTONDOWN:
                if not renju_board.handle_key_event(e, draw):
                    return START_NEW
        return CONTINUE_GAME

    def draw():
        screen.fill(WHITE_COLOR)

        renju_board.draw(screen)
        if renju_board.game_over:
            screen.blit(font.render("{0} Win".format(
                "Black" if renju_board.winner == BLACK else "White"),
                True, BLACK_COLOR), (700, 400)
            )

        pygame.display.update()

    font = pygame.font.Font(None, 36)

    if color == WHITE:
        renju_board.make_first_move(draw)
    while 1:
        status = update()
        draw()
        if status == END_GAME:
            return END_GAME
        if status == START_NEW:
            return START_NEW

    return END_GAME


def _choose_player():
    screen = pygame.display.set_mode((1200, 800))

    background = pygame.Surface(screen.get_size())
    background = background.convert()
    background.fill((81, 204, 201))

    font = pygame.font.Font(None, 100)
    _render_multi_line(CHOOSING_PLAYER_TEXT, 200, 50, 200, background, font, (100, 100, 100))

    screen.blit(background, (0, 0))
    pygame.display.flip()

    while 1:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return END_GAME
            if event.type == pygame.MOUSEBUTTONDOWN:
                if _handle_choose_player_click(event) == BLACK:
                    if _play_game(BLACK) == START_NEW:
                        return START_NEW
                    else:
                        return END_GAME
                if _handle_choose_player_click(event) == WHITE:
                    if _play_game(WHITE) == START_NEW:
                        return START_NEW
                    else:
                        return END_GAME


def _play():
    screen = pygame.display.set_mode((1200, 800))
    pygame.display.set_caption(CAPTION)

    background = pygame.Surface(screen.get_size())
    background = background.convert()
    background.fill((81, 204, 201))

    font = pygame.font.Font(None, 100)
    image = pygame.image.load(INIT_IMAGE)
    _render_multi_line(INIT_GREETING, 20, 200, 100, background, font, (100, 100, 100), 600, 300, image)

    screen.blit(background, (0, 0))
    pygame.display.flip()
    while 1:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return END_GAME
            if event.type == pygame.MOUSEBUTTONDOWN:
                if _handle_initial_click(event):
                    if _choose_player() == START_NEW:
                        return START_NEW
                    else:
                        return END_GAME
    return END_GAME


def main():
    pygame.init()
    while(_play() == START_NEW):
        continue


if __name__ == '__main__':
    main()
