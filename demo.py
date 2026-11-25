"""
demo.py - Try out the game and algorithms from the terminal (CP468 Project 1).

Run with:   python3 demo.py

Options:
  1. Human vs AI
  2. AI vs AI
  3. Compare algorithms on one position (nodes, time, pruning)
  4. Show a search tree (with pruned branches for Alpha-Beta)
"""

from game import TicTacToe, StochasticTicTacToe
from algorithms import (RandomAgent, MinimaxAgent, AlphaBetaAgent,
                        ExpectiminimaxAgent, format_tree)
from gemini_agent import GeminiAgent


def ask_int(prompt, default, allowed=None):
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            value = int(raw)
            if allowed is None or value in allowed:
                return value
        except ValueError:
            pass
        print("  Please enter a valid number.")


def setup_game():
    size = ask_int("Board size (3, 4 or 5)", 3, {3, 4, 5})
    win = ask_int("How many in a row to win", min(size, 4) if size > 3 else 3,
                  set(range(3, size + 1)))
    stochastic = ask_int("Game type: 1 = normal, 2 = stochastic (pieces can slip)", 1, {1, 2})
    if stochastic == 2:
        return StochasticTicTacToe(size, win, success_prob=0.8)
    return TicTacToe(size, win)


def pick_agent(label, game):
    default_depth = None if game.size == 3 else 4
    print(f"\nChoose {label}:")
    print("  1 = Random   2 = Minimax   3 = Alpha-Beta   4 = Expectiminimax   5 = Gemini")
    choice = ask_int("  Agent", 3, {1, 2, 3, 4, 5})
    if choice == 1:
        return RandomAgent()
    if choice == 5:
        level = ask_int("  Gemini difficulty: 1 = easy, 2 = medium, 3 = hard", 3, {1, 2, 3})
        return GeminiAgent(difficulty=["easy", "medium", "hard"][level - 1])
    shown = "full" if default_depth is None else default_depth
    depth = ask_int(f"  Search depth (0 = full search) [{shown}]", default_depth or 0)
    depth = None if depth == 0 else depth
    cls = {2: MinimaxAgent, 3: AlphaBetaAgent, 4: ExpectiminimaxAgent}[choice]
    return cls(max_depth=depth, move_ordering=(cls is AlphaBetaAgent))


def ai_turn(agent, game):
    move = agent.choose_move(game)
    new = game.make_move(move)
    slip = "" if new.last_move == move else f" -> slipped to {new.last_move}!"
    if isinstance(agent, GeminiAgent) and agent.last_reason:
        print(f"  Gemini says: {agent.last_reason}")
    print(f"\n{agent.name} ({game.current_player}) plays {move}{slip}   "
          f"[{agent.nodes_evaluated:,} nodes, {agent.last_time:.3f}s"
          + (f", {agent.pruned_count:,} pruned" if agent.pruned_count else "") + "]")
    return new


def human_turn(game):
    while True:
        try:
            r, c = map(int, input("\nYour move (row col): ").replace(",", " ").split())
            new = game.make_move((r, c))
            if new.last_move != (r, c):
                print(f"Unlucky! Your piece slipped to {new.last_move}")
            return new
        except ValueError as e:
            print(f"Invalid move: {e}")


def show_result(game):
    print("\n" + str(game))
    w = game.winner()
    print("\nResult: draw" if w == "draw" else f"\nResult: {w} wins!")


def human_vs_ai():
    game = setup_game()
    human = ask_int("Play as 1 = X (first) or 2 = O (second)", 1, {1, 2})
    human_symbol = "X" if human == 1 else "O"
    agent = pick_agent("your opponent", game)
    while not game.is_over():
        print("\n" + str(game))
        game = human_turn(game) if game.current_player == human_symbol else ai_turn(agent, game)
    show_result(game)


def ai_vs_ai():
    game = setup_game()
    agent_x = pick_agent("player X", game)
    agent_o = pick_agent("player O", game)
    while not game.is_over():
        print("\n" + str(game))
        agent = agent_x if game.current_player == "X" else agent_o
        game = ai_turn(agent, game)
    show_result(game)


def compare():
    print("\nComparing on an empty 3x3 board and a 4x4 board (depth 4)...\n")
    cases = [
        ("3x3 empty, full search", TicTacToe(3), None),
        ("4x4 (3 in a row), depth 4", TicTacToe(4, 3), 4),
    ]
    print(f"{'Position':30} {'Algorithm':28} {'Nodes':>12} {'Time (s)':>10} {'Pruned':>10}")
    print("-" * 94)
    for label, game, depth in cases:
        for agent in (MinimaxAgent(max_depth=depth),
                      AlphaBetaAgent(max_depth=depth),
                      AlphaBetaAgent(max_depth=depth, move_ordering=True),
                      ExpectiminimaxAgent(max_depth=depth)):
            agent.choose_move(game)
            name = agent.name + (" +ordering" if getattr(agent, "move_ordering", False) else "")
            print(f"{label:30} {name:28} {agent.nodes_evaluated:>12,} "
                  f"{agent.last_time:>10.3f} {agent.pruned_count:>10,}")


def show_tree():
    print("\nSearching from this position (X to move):")
    game = TicTacToe()
    for m in [(0, 0), (1, 1), (2, 2), (0, 2)]:
        game = game.result(m)
    print(game)
    for agent in (MinimaxAgent(record_tree=True), AlphaBetaAgent(record_tree=True)):
        agent.choose_move(game)
        print(f"\n=== {agent.name}: {agent.nodes_evaluated} nodes, "
              f"{agent.pruned_count} pruned ===")
        print(format_tree(agent.tree, max_depth=2))

    print("\n=== Expectiminimax on the stochastic game (depth 2) ===")
    sgame = StochasticTicTacToe()
    for m in [(0, 0), (1, 1), (2, 2), (0, 2)]:
        sgame = sgame.result(m)
    agent = ExpectiminimaxAgent(max_depth=2, record_tree=True)
    agent.choose_move(sgame)
    print(format_tree(agent.tree, max_depth=2))


if __name__ == "__main__":
    print("CP468 Project 1 - Game Tree Search Demo")
    print("  1 = Human vs AI\n  2 = AI vs AI\n  3 = Compare algorithms\n  4 = Show search trees")
    choice = ask_int("Choose", 1, {1, 2, 3, 4})
    {1: human_vs_ai, 2: ai_vs_ai, 3: compare, 4: show_tree}[choice]()