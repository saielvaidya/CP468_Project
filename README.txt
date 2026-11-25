CP468 Project 1 - Multi-Agent Decision Making in Games
======================================================

Tic-Tac-Toe on 3x3, 4x4 and 5x5 boards (plus a stochastic version where pieces
can slip) played by Minimax, Alpha-Beta Pruning, Expectiminimax, a Random
baseline and a Gemini LLM agent. Supports Human vs AI and AI vs AI.


FILES
-----
game.py              Game environment (TicTacToe and StochasticTicTacToe)
algorithms.py        Minimax, Alpha-Beta, Expectiminimax and Random agents
heuristics.py        Board evaluation + move ordering for depth-limited search
gemini_agent.py      Gemini API agent (difficulty levels, retries, fallback)
config.example.json  Gemini configuration template (model, difficulty, rate limit) (RENAME THIS TO config.json AND ADD IN YOUR OWN GEMINI API KEY)
gui.py               Graphical interface: play games, see stats, view search trees
demo.py              Text-based interface (same features in the terminal)
evaluate.py          Testing & evaluation: runtime, node counts, win rates, charts
tests.py             Unit tests (Gemini tests use a fake client - no internet needed)
results/             CSV tables and PNG charts written by evaluate.py


SETUP
-----
1. Python 3.9 or newer.

2. Install the two extra packages:
       pip install -r requirements.txt
   (gui.py uses tkinter, which comes with Python on Windows/macOS.
    On Linux: sudo apt install python3-tk)

3. Gemini API key (free): create one at https://aistudio.google.com/apikey
   then rename config.example.json to config.json and paste the key into "api_key".
   config.json is in .gitignore so the key is never committed.
   Without a key everything still runs; the Gemini agent just plays random moves instead of using Gemini.


RUN
---
    python3 gui.py                 Graphical interface (recommended)
    python3 demo.py                Text interface
    python3 tests.py               Unit tests
    python3 evaluate.py            Full evaluation (saves results/ CSVs + charts)
    python3 evaluate.py --no-gemini            ...without the API games
    python3 evaluate.py --gemini-games 6 --difficulty medium
    python3 gemini_agent.py        One quick Gemini move, to check your key works


USING THE GUI
-------------
- Pick board size, how many in a row wins, Normal or Stochastic.
- Choose Player X and Player O: Human, Random, Minimax, Alpha-Beta,
  Expectiminimax or Gemini (easy / medium / hard). Two AIs = AI vs AI.
- Search depth 0 = auto (full search on normal 3x3, depth-limited otherwise).
- Press New Game. Humans click a square to move.
- The right panel shows the last AI decision: nodes evaluated, branches
  pruned, time, score, and Gemini's own explanation in the move log.
- Show Search Tree opens the tree of the last AI search. Click the arrows to
  expand. Green = chosen move, red = pruned by Alpha-Beta, blue = chance node.


GEMINI CONFIGURATION (config.example.json / config.json)
---------------------------------------------------------
model                      Gemini model name (default "gemini-flash-3.5")
difficulty                 default level: easy, medium or hard
temperatures               randomness used for each difficulty level
max_attempts               how many times to re-ask after an illegal answer
min_seconds_between_calls  pause between requests to respect free-tier limits
