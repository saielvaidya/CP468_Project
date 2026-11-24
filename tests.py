"""
tests.py - Unit tests for the game and search algorithms (CP468 Project 1).

Run with:   python3 tests.py
"""

import unittest

from game import TicTacToe, StochasticTicTacToe
from algorithms import (RandomAgent, MinimaxAgent, AlphaBetaAgent,
                        ExpectiminimaxAgent, WIN_SCORE)


def play_moves(game, moves):
    for m in moves:
        game = game.result(m)
    return game


def play_game(game, agent_x, agent_o):
    """Play a full game between two agents and return the winner."""
    while not game.is_over():
        agent = agent_x if game.current_player == "X" else agent_o
        game = game.make_move(agent.choose_move(game))
    return game.winner()


class TestGame(unittest.TestCase):
    def test_row_win(self):
        g = play_moves(TicTacToe(), [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)])
        self.assertEqual(g.winner(), "X")

    def test_column_win(self):
        g = play_moves(TicTacToe(), [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2), (2, 1)])
        self.assertEqual(g.winner(), "O")

    def test_diagonal_wins(self):
        g = play_moves(TicTacToe(), [(0, 0), (0, 1), (1, 1), (0, 2), (2, 2)])
        self.assertEqual(g.winner(), "X")
        g = play_moves(TicTacToe(), [(0, 2), (0, 1), (1, 1), (0, 0), (2, 0)])
        self.assertEqual(g.winner(), "X")

    def test_draw(self):
        g = play_moves(TicTacToe(), [(0, 0), (0, 1), (0, 2), (1, 1), (1, 0),
                                     (1, 2), (2, 1), (2, 0), (2, 2)])
        self.assertEqual(g.winner(), "draw")

    def test_illegal_moves(self):
        g = TicTacToe().result((1, 1))
        with self.assertRaises(ValueError):
            g.result((1, 1))       # taken
        with self.assertRaises(ValueError):
            g.result((3, 3))       # off board
        with self.assertRaises(ValueError):
            g.result((-1, 0))      # negative index

    def test_original_unchanged(self):
        g = TicTacToe()
        g.result((0, 0))
        self.assertEqual(len(g.get_valid_moves()), 9)

    def test_big_board_win_length(self):
        g = play_moves(TicTacToe(size=5, win_length=4),
                       [(0, 0), (4, 4), (1, 1), (4, 3), (2, 2), (4, 0), (3, 3)])
        self.assertEqual(g.winner(), "X")

    def test_stochastic_probabilities_sum_to_one(self):
        g = StochasticTicTacToe(success_prob=0.8)
        for move in g.get_valid_moves():
            total = sum(p for p, _ in g.outcomes(move))
            self.assertAlmostEqual(total, 1.0)

    def test_stochastic_slips(self):
        g = StochasticTicTacToe(success_prob=0.5, seed=1)
        landed = {g.make_move((1, 1)).last_move for _ in range(200)}
        self.assertIn((1, 1), landed)
        self.assertGreater(len(landed), 1)   # sometimes slips elsewhere


class TestAlgorithms(unittest.TestCase):
    def test_takes_immediate_win(self):
        g = play_moves(TicTacToe(), [(0, 0), (1, 0), (0, 1), (1, 1)])  # X to play, (0,2) wins
        for agent in (MinimaxAgent(), AlphaBetaAgent(), ExpectiminimaxAgent()):
            self.assertEqual(agent.choose_move(g), (0, 2), agent.name)

    def test_blocks_opponent(self):
        g = play_moves(TicTacToe(), [(0, 0), (1, 1), (2, 2), (0, 2)])  # X must block (2,0)
        for agent in (MinimaxAgent(), AlphaBetaAgent(), ExpectiminimaxAgent()):
            self.assertEqual(agent.choose_move(g), (2, 0), agent.name)

    def test_empty_board_is_draw(self):
        agent = AlphaBetaAgent()
        agent.choose_move(TicTacToe())
        self.assertEqual(agent.last_score, 0)   # perfect play -> draw

    def test_alphabeta_matches_minimax(self):
        positions = [[], [(0, 0)], [(1, 1)], [(0, 1), (1, 1)], [(0, 0), (2, 2), (0, 2)]]
        for moves in positions:
            g = play_moves(TicTacToe(), moves)
            mm, ab = MinimaxAgent(), AlphaBetaAgent()
            self.assertEqual(mm.choose_move(g), ab.choose_move(g))
            self.assertEqual(mm.last_score, ab.last_score)
            self.assertLess(ab.nodes_evaluated, mm.nodes_evaluated)

    def test_expectiminimax_equals_minimax_when_deterministic(self):
        g = play_moves(TicTacToe(), [(1, 1)])
        mm, em = MinimaxAgent(), ExpectiminimaxAgent()
        self.assertEqual(mm.choose_move(g), em.choose_move(g))
        self.assertEqual(mm.last_score, em.last_score)

    def test_minimax_never_loses_to_random(self):
        for seed in range(5):
            self.assertNotEqual(play_game(TicTacToe(), RandomAgent(seed), MinimaxAgent()), "X")

    def test_alphabeta_never_loses_to_random(self):
        for seed in range(20):
            self.assertNotEqual(play_game(TicTacToe(), AlphaBetaAgent(), RandomAgent(seed)), "O")
            self.assertNotEqual(play_game(TicTacToe(), RandomAgent(seed), AlphaBetaAgent()), "X")

    def test_perfect_play_draws(self):
        self.assertEqual(play_game(TicTacToe(), AlphaBetaAgent(), AlphaBetaAgent()), "draw")

    def test_depth_limited_big_board(self):
        # 5x5, 4 in a row: O threatens to complete row 4, X must block (or win first)
        g = play_moves(TicTacToe(size=5, win_length=4),
                       [(0, 0), (4, 0), (0, 4), (4, 1), (2, 0), (4, 2)])
        agent = AlphaBetaAgent(max_depth=3, move_ordering=True)
        self.assertEqual(agent.choose_move(g), (4, 3))
        self.assertLess(agent.last_time, 10)

    def test_expectiminimax_stochastic_runs(self):
        g = StochasticTicTacToe(success_prob=0.8, seed=0)
        g = play_moves(g, [(0, 0), (1, 1), (2, 2)])
        agent = ExpectiminimaxAgent(max_depth=3)
        move = agent.choose_move(g)
        self.assertIn(move, g.get_valid_moves())
        self.assertLess(abs(agent.last_score), WIN_SCORE)

    def test_tree_records_pruning(self):
        g = play_moves(TicTacToe(), [(0, 0), (1, 1), (2, 2)])
        agent = AlphaBetaAgent(record_tree=True)
        agent.choose_move(g)
        self.assertIsNotNone(agent.tree)
        self.assertGreater(agent.pruned_count, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)