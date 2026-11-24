"""
heuristics.py - Evaluation and move-ordering helpers for CP468 Project 1.

Used when a search is depth-limited (e.g. 4x4 / 5x5 boards), where the
algorithms cannot search to the end of the game and must estimate who is
winning instead.
"""

from functools import lru_cache
from game import EMPTY, DIRECTIONS, other


@lru_cache(maxsize=None)
def winning_lines(size, win_length):
    """
    Every group of `win_length` squares in a row that could win the game.
    Computed once per board shape and cached.
    """
    lines = []
    for r in range(size):
        for c in range(size):
            for dr, dc in DIRECTIONS:
                end_r = r + dr * (win_length - 1)
                end_c = c + dc * (win_length - 1)
                if 0 <= end_r < size and 0 <= end_c < size:
                    lines.append(tuple((r + dr * i, c + dc * i) for i in range(win_length)))
    return tuple(lines)


def evaluate(game, player):
    """
    Heuristic score of a non-finished position from `player`'s point of view.

    For every possible winning line:
      * only `player`'s pieces in it   -> +10^(number of pieces)
      * only the opponent's pieces     -> -10^(number of pieces)
      * both, or empty                 ->  0  (nobody can win with that line)

    So "two in an open line" is worth 10x more than "one in an open line",
    and blocked lines are worthless. The value always stays far below the
    score of an actual win (see WIN_SCORE in algorithms.py).
    """
    opponent = other(player)
    board = game.board
    score = 0
    for line in winning_lines(game.size, game.win_length):
        mine = theirs = 0
        for r, c in line:
            cell = board[r][c]
            if cell == player:
                mine += 1
            elif cell == opponent:
                theirs += 1
        if mine and not theirs:
            score += 10 ** mine
        elif theirs and not mine:
            score -= 10 ** theirs
    return score


def order_moves(game):
    """
    Return valid moves sorted centre-first.

    Centre squares belong to the most winning lines, so they are usually the
    best moves. Searching good moves first makes Alpha-Beta prune much more.
    """
    centre = (game.size - 1) / 2
    return sorted(game.get_valid_moves(),
                  key=lambda m: abs(m[0] - centre) + abs(m[1] - centre))