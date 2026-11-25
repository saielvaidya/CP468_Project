"""
gemini_agent.py - An agent that asks Google's Gemini LLM to choose its moves.

How it works (one turn):
  1. Setup     - the game rules (board size, how many in a row wins, whether
                 pieces can slip) are sent to Gemini as a "system instruction".
  2. Prompt    - we describe the current board and list every legal move.
  3. Decision  - Gemini replies in JSON: {"row": r, "col": c, "reason": "..."}.
  4. Validate  - we check the move is actually legal. If Gemini answers with
                 something illegal or unreadable we tell it what went wrong and
                 ask again (up to `max_attempts` times).
  5. Fallback  - if it still fails (bad replies, no internet, rate limit...),
                 we play a random legal move so the game never crashes.

Difficulty levels:
  easy   - told to play like a beginner, high temperature (more random)
  medium - told to play reasonably, medium temperature
  hard   - told to play perfectly, low temperature (most consistent)


The API key is read from the
GEMINI_API_KEY environment variable, or from "api_key" in config.json.

"""

import json
import os
import random
import re
import time
import warnings

from algorithms import Agent

# My mac keeps warning me about this in terminal so I ignored it since it doesn't affect gemini
warnings.filterwarnings("ignore", category=FutureWarning, module="google")
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CONFIG = {
    "api_key": "",
    "model": "gemini-flash-latest",
    "difficulty": "hard",
    "temperatures": {"easy": 1.0, "medium": 0.6, "hard": 0.1},
    "backup_models": ["gemini-flash-lite-latest", "gemini-2.5-flash"], # backup if main model is overloaded
    "max_attempts": 4,
    "min_seconds_between_calls": 4.0,   # it was making too many calls and running through rate limits
}

# Prompts for gemini for each difficulty level
DIFFICULTY_STYLE = {
    "easy":   "You are a beginner. Play casually and do not think too far ahead.",
    "medium": "You are a decent player. Take wins and block obvious threats.",
    "hard":   ("You are an expert. Play perfectly: 1) win immediately if you can, "
               "2) otherwise block the opponent's winning move, 3) otherwise create "
               "double threats and prefer central squares."),
}

_last_call_time = 0.0   # shared by every GeminiAgent, used for rate limiting


def load_config(path=None):
    """Read config.json (or config.example.json) on top of the defaults."""
    config = dict(DEFAULT_CONFIG)
    candidates = [path] if path else [os.path.join(HERE, "config.json"),
                                      os.path.join(HERE, "config.example.json")]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            with open(candidate) as f:
                config.update(json.load(f))
            break

    # Checks first for environtment variable, then config.json, for the api key
    config["api_key"] = os.environ.get("GEMINI_API_KEY") or config.get("api_key", "")
    return config


def make_client(config):
    """Creating the Gemini client here so the project will still work if the google-genai package isnt installed."""
    if not config["api_key"]:
        raise RuntimeError("no API key (set GEMINI_API_KEY or config.json)")
    from google import genai
    return genai.Client(api_key=config["api_key"], http_options={"api_version": "v1beta"})


def list_models(config=None):
    """Print the models this API key can use to generate text (python3 gemini_agent.py --models)."""
    client = make_client(config or load_config())
    for model in client.models.list():
        actions = getattr(model, "supported_actions", None) or []
        if not actions or "generateContent" in actions:
            print(model.name.replace("models/", ""))


def gemini_available(config=None):
    """True if an API key is configured (used to hide Gemini when there is no key)."""
    return bool((config or load_config())["api_key"])


class GeminiAgent(Agent):
    """Asks the Gemini API for each move. Falls back to a random move on failure."""

    def __init__(self, difficulty=None, model=None, client=None, config=None, seed=None):
        self.config = config or load_config()
        self.difficulty = difficulty or self.config["difficulty"]
        if self.difficulty not in DIFFICULTY_STYLE:
            raise ValueError("difficulty must be easy, medium or hard")
        self.model = model or self.config["model"]
        self.client = client            # created lazily so tests can pass a fake one
        self.rng = random.Random(seed)
        super().__init__()

    @property
    def name(self):
        return f"Gemini ({self.difficulty})"

    def reset_stats(self):
        super().reset_stats()
        self.api_calls = 0         # requests sent this turn
        self.invalid_replies = 0   # replies that were not a legal move
        self.used_fallback = False # True if we had to play a random move
        self.last_reason = ""      # Gemini's explanation (shown in the GUI log)

    # ------------------------------------------------------------- main method
    def choose_move(self, game):
        if game.is_over():
            raise ValueError("Cannot choose a move: the game is over")
        self.reset_stats()
        start = time.perf_counter()

        legal = game.get_valid_moves()
        prompt = self.build_prompt(game)
        move = None

        for attempt in range(self.config["max_attempts"]):
            try:
                reply = self._ask_gemini(game, prompt)
            except Exception as error:          # network error, bad key, busy server...
                self.last_reason = f"API error: {error}"
                # If this model is overloaded (503), out of free quota for today (429)
                # or doesn't exist (404), switch to the next backup model from the
                # config. Each model has its own separate free-tier quota.
                if any(code in str(error) for code in ("503", "429", "404", "UNAVAILABLE",
                                                       "RESOURCE_EXHAUSTED", "NOT_FOUND")):
                    self._switch_to_backup_model()
                # "Exponential backoff": wait 2s, then 4s, then 8s... before retrying,
                # because errors like 503 (server busy) or 429 (too many requests)
                # usually clear up after a short wait.
                if self.config["min_seconds_between_calls"] > 0:
                    time.sleep(2 ** (attempt + 1))
                continue

            candidate, reason = self.parse_reply(reply)
            if candidate in legal:
                move, self.last_reason = candidate, reason
                break
            # Illegal or unreadable: explain the mistake and ask again.
            self.invalid_replies += 1
            prompt = (self.build_prompt(game)
                      + f"\n\nYour previous answer {reply!r} was NOT a legal move. "
                        "Pick one of the legal moves listed above.")

        if move is None:                         # give up -> random legal move
            self.used_fallback = True
            move = self.rng.choice(legal)
            self.last_reason = (self.last_reason or "No valid reply") + " -> random move"

        self.last_time = time.perf_counter() - start
        return move

    # ------------------------------------------------------- prompt building
    def rules_text(self, game):
        """GAME SETUP: the rules of this particular game, sent as the system instruction."""
        text = (f"You are playing Tic-Tac-Toe on a {game.size}x{game.size} board. "
                f"Players take turns placing their symbol (X or O) on an empty square. "
                f"The first player to get {game.win_length} in a row (horizontally, "
                f"vertically or diagonally) wins. If the board fills up it is a draw. "
                f"Rows and columns are numbered from 0.")
        if hasattr(game, "success_prob"):        # the stochastic version
            text += (f" Chance rule: a piece only lands on the chosen square with "
                     f"probability {game.success_prob:.0%}; otherwise it slips to a "
                     f"random empty neighbouring square.")
        text += (f"\n{DIFFICULTY_STYLE[self.difficulty]}\n"
                 'Reply ONLY with JSON like {"row": 1, "col": 2, "reason": "short explanation"}.')
        return text

    def build_prompt(self, game):
        """The current board plus the list of legal moves."""
        board = "\n".join(
            " ".join(cell if cell != " " else "." for cell in row) for row in game.board)
        legal = ", ".join(f"({r},{c})" for r, c in game.get_valid_moves())
        return (f"Current board ('.' = empty):\n{board}\n\n"
                f"You are {game.current_player}. Legal moves (row,col): {legal}\n"
                f"Choose your move.")

    # --------------------------------------------------------- API + parsing
    def _ask_gemini(self, game, prompt):
        """Send one request to Gemini and return the reply text."""
        global _last_call_time
        if self.client is None:
            if not self.config["api_key"]:
                raise RuntimeError("no API key (set GEMINI_API_KEY or config.json)")
            self.client = make_client(self.config)

        # Simple rate limiting so the free tier does not reject us.
        wait = self.config["min_seconds_between_calls"] - (time.time() - _last_call_time)
        if wait > 0:
            time.sleep(wait)
        _last_call_time = time.time()

        self.api_calls += 1
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "system_instruction": self.rules_text(game),
                "temperature": self.config["temperatures"][self.difficulty],
                "response_mime_type": "application/json",   # ask for pure JSON
            },
        )
        return response.text or ""

    def _switch_to_backup_model(self):
        """Move on to the next model in config["backup_models"] (if any are left)."""
        backups = [m for m in self.config.get("backup_models", []) if m != self.model]
        tried = getattr(self, "_tried_models", set())
        tried.add(self.model)
        self._tried_models = tried
        remaining = [m for m in backups if m not in tried]
        if remaining:
            self.model = remaining[0]

    @staticmethod
    def parse_reply(text):
        """
        Turn Gemini's reply into ((row, col), reason).
        Accepts proper JSON, and as a backup any text containing two numbers
        like "(1, 2)". Returns (None, "") if nothing usable is found.
        """
        try:
            data = json.loads(text)
            return (int(data["row"]), int(data["col"])), str(data.get("reason", ""))
        except (ValueError, KeyError, TypeError):
            pass
        match = re.search(r"(\d+)\s*[, ]\s*(\d+)", text or "")
        if match:
            return (int(match.group(1)), int(match.group(2))), ""
        return None, ""


# Quick manual check:  python3 gemini_agent.py
# List usable models:  python3 gemini_agent.py --models
if __name__ == "__main__":
    import sys
    if "--models" in sys.argv:              # python3 gemini_agent.py --models
        list_models()
        sys.exit()

    from game import TicTacToe
    g = TicTacToe(3)
    for m in [(0, 0), (1, 1), (0, 1)]:     # X threatens (0, 2): O must block
        g = g.result(m)
    print(g)
    agent = GeminiAgent(difficulty="hard")
    move = agent.choose_move(g)
    print(f"\nGemini plays {move}  ({agent.last_time:.1f}s, {agent.api_calls} call(s))")
    print("Reason:", agent.last_reason)
