"""
game.py - Game environment for CP468 Project 1 (Multi-Agent Decision Making in Games).

Two games are provided:

  * TicTacToe           - deterministic N x N Tic-Tac-Toe, k-in-a-row to win.
  * StochasticTicTacToe - same game with a chance element: a piece only lands on
                          the chosen square with probability `success_prob`;
                          otherwise it "slips" to a random empty neighbouring
                          square. Used to test Expectiminimax (chance nodes).

Search algorithms treat every state as immutable: methods that change the board
return a NEW game object, so the AI can explore future positions without
touching the real game.

Three ways to apply a move:
  result(move)    -> the piece lands exactly on `move` (deterministic; used by
                     Minimax / Alpha-Beta search)
  outcomes(move)  -> list of (probability, new_state) for every way the move
                     can turn out (used by Expectiminimax chance nodes)
  make_move(move) -> what actually happens when the move is played in a real
                     game (random in the stochastic version)
"""

import random

EMPTY = " "
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))  # right, down, diagonal, anti-diagonal
NEIGHBOURS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def other(player):
    """Return the opponent of `player`."""
    return "O" if player == "X" else "X"


class TicTacToe:
    """Deterministic N x N Tic-Tac-Toe where `win_length` in a row wins."""

    def __init__(self, size=3, win_length=None):
        if size < 3:
            raise ValueError("Board size must be at least 3")
        self.size = size
        self.win_length = win_length if win_length else size
        if not 3 <= self.win_length <= size:
            raise ValueError("win_length must be between 3 and the board size")

        self.board = [[EMPTY] * size for _ in range(size)]
        self.current_player = "X"          # X always starts
        self.empty_count = size * size     # tracked so draw detection is O(1)
        self.last_move = None              # square where the last piece landed
        self._winner = None                # 'X', 'O', 'draw' or None (cached)

    # ------------------------------------------------------------------ queries
    def get_valid_moves(self):
        """Return a list of (row, col) for every empty square."""
        if self._winner is not None:
            return []
        return [(r, c)
                for r in range(self.size)
                for c in range(self.size)
                if self.board[r][c] == EMPTY]

    def winner(self):
        """Return 'X' or 'O' if someone won, 'draw' if the board is full, else None."""
        return self._winner

    def is_over(self):
        return self._winner is not None

    # -------------------------------------------------------------- transitions
    def result(self, move):
        """Deterministic: new state with the current player's piece exactly on `move`."""
        self._check_legal(move)
        return self._place(move)

    def outcomes(self, move):
        """All possible results of `move` as (probability, state). Deterministic -> one."""
        return [(1.0, self.result(move))]

    def make_move(self, move):
        """Play `move` for real and return the new game state."""
        return self.result(move)

    # ---------------------------------------------------------------- internals
    def _check_legal(self, move):
        r, c = move
        if self._winner is not None:
            raise ValueError("The game is already over")
        if not (0 <= r < self.size and 0 <= c < self.size):
            raise ValueError(f"Square {move} is off the board")
        if self.board[r][c] != EMPTY:
            raise ValueError(f"Square {move} is already taken")

    def _place(self, square):
        """Return a copy with the current player's piece on `square` and turns switched."""
        new = self.copy()
        r, c = square
        new.board[r][c] = self.current_player
        new.empty_count -= 1
        new.last_move = square
        # Optimisation: a new line can only pass through the square just played,
        # so we only check lines through (r, c) instead of scanning the whole board.
        if new._completes_line(r, c):
            new._winner = self.current_player
        elif new.empty_count == 0:
            new._winner = "draw"
        new.current_player = other(self.current_player)
        return new

    def _completes_line(self, r, c):
        """True if the piece at (r, c) is part of `win_length` in a row."""
        player = self.board[r][c]
        for dr, dc in DIRECTIONS:
            count = 1
            for sign in (1, -1):                      # look both ways along the line
                rr, cc = r + sign * dr, c + sign * dc
                while (0 <= rr < self.size and 0 <= cc < self.size
                       and self.board[rr][cc] == player):
                    count += 1
                    rr += sign * dr
                    cc += sign * dc
            if count >= self.win_length:
                return True
        return False

    def copy(self):
        """Fast copy: share every attribute except the board, which is copied row by row."""
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        new.board = [row[:] for row in self.board]
        return new

    def __str__(self):
        header = "   " + "   ".join(str(c) for c in range(self.size))
        separator = "  " + "+".join(["---"] * self.size)
        lines = [header]
        for r in range(self.size):
            lines.append(f"{r}  " + " | ".join(self.board[r]))
            if r < self.size - 1:
                lines.append(separator)
        return "\n".join(lines)


class StochasticTicTacToe(TicTacToe):
    """
    Tic-Tac-Toe with a chance element.

    When a player chooses a square, the piece lands there with probability
    `success_prob`. Otherwise it slips to one of the empty neighbouring squares
    (8 directions), each equally likely. If no neighbour is empty, the piece
    always lands on the chosen square.
    """

    def __init__(self, size=3, win_length=None, success_prob=0.8, seed=None):
        super().__init__(size, win_length)
        if not 0.0 < success_prob <= 1.0:
            raise ValueError("success_prob must be in (0, 1]")
        self.success_prob = success_prob
        self.rng = random.Random(seed)    # shared by all copies of this game

    def _empty_neighbours(self, move):
        r, c = move
        result = []
        for dr, dc in NEIGHBOURS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < self.size and 0 <= cc < self.size and self.board[rr][cc] == EMPTY:
                result.append((rr, cc))
        return result

    def outcomes(self, move):
        self._check_legal(move)
        slips = self._empty_neighbours(move)
        if not slips or self.success_prob == 1.0:
            return [(1.0, self._place(move))]
        p_slip = (1.0 - self.success_prob) / len(slips)
        return ([(self.success_prob, self._place(move))]
                + [(p_slip, self._place(square)) for square in slips])

    def make_move(self, move):
        """Roll the dice: the piece may land somewhere other than `move`."""
        possible = self.outcomes(move)
        weights = [p for p, _ in possible]
        states = [s for _, s in possible]
        return self.rng.choices(states, weights=weights)[0]


# Quick test: play Human vs Human in the terminal
if __name__ == "__main__":
    game = TicTacToe(size=3)
    while not game.is_over():
        print(game)
        print(f"\nPlayer {game.current_player}'s turn")
        try:
            r, c = map(int, input("Enter row and column (e.g. 1 2): ").replace(",", " ").split())
            game = game.make_move((r, c))
        except ValueError as e:
            print(f"Invalid move: {e}\n")
    print(game)
    result = game.winner()
    print("It's a draw!" if result == "draw" else f"Player {result} wins!")