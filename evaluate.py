"""
evaluate.py - Testing and Evaluation for CP468 Project 1.

Run with:   python3 evaluate.py              (everything; Gemini only if a key is set)
            python3 evaluate.py --no-gemini  (skip the API games)
            python3 evaluate.py --gemini-games 6 --difficulty medium

It measures the three things the project asks for:

  1. SCALABILITY - nodes evaluated and execution time for Minimax vs
     Alpha-Beta (with and without move ordering) on 3x3, 4x4 and 5x5 boards,
     plus Expectiminimax on the stochastic version of each board.
  2. DEPTH GROWTH - how the number of nodes grows with search depth on 4x4
     (Minimax grows exponentially; Alpha-Beta grows much more slowly).
  3. SUCCESS RATE - win / draw / loss of each algorithm against a Random
     baseline and against the Gemini API agent.

Results are printed as tables and saved to the results/ folder as CSV files
and PNG charts (charts need matplotlib: pip install matplotlib).
"""

import argparse
import csv
import os

from game import TicTacToe, StochasticTicTacToe
from algorithms import RandomAgent, MinimaxAgent, AlphaBetaAgent, ExpectiminimaxAgent
from gemini_agent import GeminiAgent, gemini_available

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# The boards we test, from easiest to hardest: (label, size, win_length, depth limit)
BOARDS = [("3x3", 3, 3, None),     # None = full search to the end of the game
          ("4x4", 4, 4, 4),
          ("5x5", 5, 4, 4)]

# Same colours as the GUI (a colour-blind-safe categorical order)
COLOURS = ["#257de0", "#f47230", "#00b894", "#f2a600"]


def save_csv(name, header, rows):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  saved {path}")


def measure(agent, game):
    """Let `agent` choose the first move on `game`; return (nodes, seconds, pruned)."""
    agent.choose_move(game)
    return agent.nodes_evaluated, agent.last_time, agent.pruned_count


# =========================================================== 1. scalability
def scalability():
    print("\n=== 1. Scalability: first move on an empty board ===")
    print(f"{'Board':6} {'Depth':6} {'Algorithm':26} {'Nodes':>11} {'Time (s)':>9} {'Pruned':>9}")
    rows = []
    for label, size, win, depth in BOARDS:
        agents = [("Minimax", MinimaxAgent(max_depth=depth), TicTacToe(size, win)),
                  ("Alpha-Beta", AlphaBetaAgent(max_depth=depth), TicTacToe(size, win)),
                  ("Alpha-Beta + ordering",
                   AlphaBetaAgent(max_depth=depth, move_ordering=True), TicTacToe(size, win)),
                  # Expectiminimax has to explore every random outcome, so it is run on
                  # the stochastic game at a smaller depth to keep the time reasonable.
                  ("Expectiminimax (stoch.)", ExpectiminimaxAgent(max_depth=2),
                   StochasticTicTacToe(size, win, success_prob=0.8))]
        for name, agent, game in agents:
            nodes, secs, pruned = measure(agent, game)
            d = agent.max_depth if agent.max_depth else "full"
            print(f"{label:6} {str(d):6} {name:26} {nodes:>11,} {secs:>9.3f} {pruned:>9,}")
            rows.append([label, d, name, nodes, round(secs, 4), pruned])
    save_csv("scalability.csv", ["board", "depth", "algorithm", "nodes", "seconds", "pruned"], rows)
    return rows


# ========================================================== 2. depth growth
def depth_growth(max_depth=5):
    print("\n=== 2. Nodes vs search depth (4x4 board, 4 in a row) ===")
    print(f"{'Depth':>5} {'Minimax':>12} {'Alpha-Beta':>12} {'Saved':>8}")
    rows = []
    for depth in range(1, max_depth + 1):
        mm, _, _ = measure(MinimaxAgent(max_depth=depth), TicTacToe(4, 4))
        ab, _, _ = measure(AlphaBetaAgent(max_depth=depth, move_ordering=True), TicTacToe(4, 4))
        saved = 1 - ab / mm
        print(f"{depth:>5} {mm:>12,} {ab:>12,} {saved:>8.1%}")
        rows.append([depth, mm, ab, round(saved, 4)])
    save_csv("depth_growth.csv", ["depth", "minimax_nodes", "alphabeta_nodes", "saved"], rows)
    return rows


# ========================================================== 3. win rates
def play_game(game, agent_x, agent_o):
    """Play one full game and return 'X', 'O' or 'draw'."""
    while not game.is_over():
        agent = agent_x if game.current_player == "X" else agent_o
        game = game.make_move(agent.choose_move(game))
    return game.winner()


def match(make_ai, make_opponent, games, make_game):
    """
    Play `games` games, swapping who goes first each game, and count
    results from the AI's point of view. Returns (wins, draws, losses).
    """
    wins = draws = losses = 0
    for i in range(games):
        ai, opponent = make_ai(), make_opponent(i)
        ai_is_x = (i % 2 == 0)
        winner = play_game(make_game(i), *((ai, opponent) if ai_is_x else (opponent, ai)))
        if winner == "draw":
            draws += 1
        elif (winner == "X") == ai_is_x:
            wins += 1
        else:
            losses += 1
    return wins, draws, losses


def win_rates(games_vs_random, games_vs_gemini, difficulty, use_gemini):
    print("\n=== 3. Success rate (3x3 board, AI's point of view) ===")
    normal = lambda i: TicTacToe(3)       # a fresh board for game number i
    algorithms = [("Minimax", lambda: MinimaxAgent()),
                  ("Alpha-Beta", lambda: AlphaBetaAgent(move_ordering=True)),
                  ("Expectiminimax", lambda: ExpectiminimaxAgent())]

    # (AI name, make AI, opponent name, make opponent, number of games, make game)
    matchups = [(name, make, "Random", lambda i: RandomAgent(seed=i), games_vs_random, normal)
                for name, make in algorithms]
    if use_gemini:
        gemini = f"Gemini ({difficulty})"
        make_gemini = lambda i: GeminiAgent(difficulty=difficulty, seed=i)
        matchups += [(name, make, gemini, make_gemini, games_vs_gemini, normal)
                     for name, make in algorithms]
        matchups.append((gemini, lambda: GeminiAgent(difficulty=difficulty),
                         "Random", lambda i: RandomAgent(seed=i), games_vs_gemini, normal))
    else:
        print("  (Gemini games skipped - no API key or --no-gemini)")

    print(f"{'AI':26} {'vs':22} {'Games':>5} {'W':>3} {'D':>3} {'L':>3} {'Win%':>6} {'Not lost':>9}")
    rows = []
    for ai_name, make_ai, opp_name, make_opp, games, make_game in matchups:
        w, d, l = match(make_ai, make_opp, games, make_game)
        print(f"{ai_name:26} {opp_name:22} {games:>5} {w:>3} {d:>3} {l:>3} "
              f"{w / games:>6.0%} {(w + d) / games:>9.0%}")
        rows.append([ai_name, opp_name, games, w, d, l, round(w / games, 3)])
    save_csv("win_rates.csv", ["ai", "opponent", "games", "wins", "draws", "losses",
                               "win_rate"], rows)
    return rows


# ================================================================ charts
def make_charts(scal_rows, depth_rows, win_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")                       # draw to files, no window needed
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed - skipping charts: pip install matplotlib)")
        return
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e6e6e6", "axes.axisbelow": True})

    # --- Chart 1: nodes and time per board size (two panels, never two y-axes)
    algos = ["Minimax", "Alpha-Beta", "Alpha-Beta + ordering"]
    boards = [b[0] for b in BOARDS]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, column, title in ((axes[0], 3, "Nodes evaluated (log scale)"),
                              (axes[1], 4, "Execution time in seconds (log scale)")):
        width = 0.26
        for i, algo in enumerate(algos):
            values = [next(r[column] for r in scal_rows if r[0] == b and r[2] == algo)
                      for b in boards]
            xs = [x + (i - 1) * width for x in range(len(boards))]
            ax.bar(xs, values, width * 0.92, label=algo, color=COLOURS[i])
        ax.set_yscale("log")
        ax.set_xticks(range(len(boards)), [f"{b}\n(depth {r[1]})" for b, r in
                                           zip(boards, scal_rows[::4])])
        ax.set_title(title, loc="left")
    axes[0].legend(frameon=False)
    fig.suptitle("Minimax vs Alpha-Beta as the board grows (first move, empty board)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    _save(fig, "scalability.png")

    # --- Chart 2: nodes vs depth
    fig, ax = plt.subplots(figsize=(7, 4.2))
    depths = [r[0] for r in depth_rows]
    for i, (col, name) in enumerate(((1, "Minimax"), (2, "Alpha-Beta + ordering"))):
        values = [r[col] for r in depth_rows]
        ax.plot(depths, values, marker="o", markersize=7, linewidth=2,
                color=COLOURS[i], label=name)
        ax.annotate(name, (depths[-1], values[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center")
    ax.set_yscale("log")
    ax.set_xticks(depths)
    ax.set_xlabel("Search depth")
    ax.set_title("Nodes evaluated vs search depth (4x4 board, log scale)",
                 loc="left", fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, "depth_growth.png")

    # --- Chart 3: win / draw / loss stacked bars
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(win_rows) + 1.5))
    labels = [f"{r[0]}  vs  {r[1]}" for r in win_rows][::-1]
    data = [[r[3] / r[2], r[4] / r[2], r[5] / r[2]] for r in win_rows][::-1]
    left = [0.0] * len(data)
    for k, (name, colour) in enumerate((("Win", "#008a17"), ("Draw", "#b0b0b0"),
                                        ("Loss", "#ec4647"))):
        values = [d[k] for d in data]
        ax.barh(labels, values, left=left, color=colour, label=name,
                edgecolor="white", linewidth=2)
        for y, (v, l) in enumerate(zip(values, left)):
            if v >= 0.08:
                ax.text(l + v / 2, y, f"{v:.0%}", ha="center", va="center",
                        color="white" if name != "Draw" else "#191919", fontsize=9)
        left = [l + v for l, v in zip(left, values)]
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_title("Success rate (AI's point of view, 3x3)", loc="left",
                 fontweight="bold", pad=28)
    fig.tight_layout()
    _save(fig, "win_rates.png")


def _save(fig, name):
    import matplotlib.pyplot as plt
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CP468 testing and evaluation")
    parser.add_argument("--games", type=int, default=20, help="games per matchup vs Random")
    parser.add_argument("--gemini-games", type=int, default=4,
                        help="games per matchup vs Gemini (each move is an API call)")
    parser.add_argument("--difficulty", default="hard", choices=["easy", "medium", "hard"])
    parser.add_argument("--no-gemini", action="store_true", help="skip the Gemini games")
    args = parser.parse_args()

    scal = scalability()
    depth = depth_growth()
    wins = win_rates(args.games, args.gemini_games, args.difficulty,
                     use_gemini=gemini_available() and not args.no_gemini)
    make_charts(scal, depth, wins)
