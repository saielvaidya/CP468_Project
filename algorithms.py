"""
algorithms.py - Search agents for CP468 Project 1.

Agents:
  RandomAgent          - picks a random legal move (baseline for testing)
  MinimaxAgent         - full (or depth-limited) Minimax
  AlphaBetaAgent       - Minimax with alpha-beta pruning
  ExpectiminimaxAgent  - Minimax with chance nodes for the stochastic game

Every agent has the same interface:
  move = agent.choose_move(game)

and after each move exposes statistics for benchmarking:
  agent.nodes_evaluated  - number of game-tree nodes visited
  agent.last_time        - seconds taken to choose the move
  agent.last_score       - value of the chosen move
  agent.pruned_count     - branches skipped (Alpha-Beta only)
  agent.tree             - the search tree (only if record_tree=True)

Search options (all optional):
  max_depth=None      search to the end of the game; set e.g. 4 for big boards
  move_ordering=False try centre moves first (helps Alpha-Beta prune more)
  record_tree=False   save the search tree for visualisation (slow; use on
                      small positions only)
"""

import math
import random
import time

from heuristics import evaluate, order_moves

WIN_SCORE = 10 ** 9   # far larger than any heuristic value


def utility(result, me, depth):
    """Score of a finished game from `me`'s point of view."""
    if result == "draw":
        return 0
    if result == me:
        return WIN_SCORE - depth   # prefer faster wins
    return depth - WIN_SCORE       # prefer slower losses


# --------------------------------------------------------------------------- base
class Agent:
    name = "Agent"

    def __init__(self):
        self.reset_stats()

    def reset_stats(self):
        self.nodes_evaluated = 0
        self.last_time = 0.0
        self.last_score = None
        self.pruned_count = 0
        self.tree = None

    def choose_move(self, game):
        raise NotImplementedError

    def __str__(self):
        return self.name


class RandomAgent(Agent):
    name = "Random"

    def __init__(self, seed=None):
        super().__init__()
        self.rng = random.Random(seed)

    def choose_move(self, game):
        self.reset_stats()
        return self.rng.choice(game.get_valid_moves())


class SearchAgent(Agent):
    """Shared code for the three search algorithms."""

    def __init__(self, max_depth=None, move_ordering=False, record_tree=False,
                 eval_fn=evaluate):
        self.max_depth = max_depth
        self.move_ordering = move_ordering
        self.record_tree = record_tree
        self.eval_fn = eval_fn
        super().__init__()

    def choose_move(self, game):
        if game.is_over():
            raise ValueError("Cannot choose a move: the game is over")
        self.reset_stats()
        start = time.perf_counter()

        me = game.current_player            # the AI plays for whoever's turn it is
        root = self._new_node(None) if self.record_tree else None
        score, move = self._search(game, me, root)

        self.last_time = time.perf_counter() - start
        self.last_score = score
        self.tree = root
        return move

    def _search(self, game, me, root):
        raise NotImplementedError

    # -- helpers shared by all algorithms --
    def _moves(self, game):
        return order_moves(game) if self.move_ordering else game.get_valid_moves()

    def _leaf_value(self, game, me, depth):
        """Return a score if this node is a leaf (game over or depth limit), else None."""
        result = game.winner()
        if result is not None:
            return utility(result, me, depth)
        if self.max_depth is not None and depth >= self.max_depth:
            return self.eval_fn(game, me)
        return None

    @staticmethod
    def _new_node(move, **extra):
        node = {"move": move, "type": None, "score": None, "pruned": False, "children": []}
        node.update(extra)
        return node

    def _add_child(self, node, move, **extra):
        if node is None:
            return None
        child = self._new_node(move, **extra)
        node["children"].append(child)
        return child

    @property
    def name(self):
        depth = "full" if self.max_depth is None else f"d={self.max_depth}"
        return f"{self.base_name} ({depth})"


# ------------------------------------------------------------------------ minimax
class MinimaxAgent(SearchAgent):
    """Searches every possible future and assumes both players play perfectly."""
    base_name = "Minimax"

    def _search(self, game, me, root):
        return self._minimax(game, me, 0, root)

    def _minimax(self, game, me, depth, node):
        """Return (best score, best move) for this position."""
        self.nodes_evaluated += 1

        leaf = self._leaf_value(game, me, depth)
        if leaf is not None:
            if node is not None:
                node["type"], node["score"] = "LEAF", leaf
            return leaf, None

        maximizing = game.current_player == me      # MAX = my turn, MIN = opponent's
        best_score = -math.inf if maximizing else math.inf
        best_move = None

        for move in self._moves(game):
            child = self._add_child(node, move)
            score, _ = self._minimax(game.result(move), me, depth + 1, child)
            if (maximizing and score > best_score) or (not maximizing and score < best_score):
                best_score, best_move = score, move

        if node is not None:
            node["type"], node["score"] = ("MAX" if maximizing else "MIN"), best_score
        return best_score, best_move


# --------------------------------------------------------------------- alpha-beta
class AlphaBetaAgent(SearchAgent):
    """
    Minimax with alpha-beta pruning.

    alpha = best score MAX is already guaranteed on this path
    beta  = best score MIN is already guaranteed on this path
    When alpha >= beta, the remaining moves cannot change the decision, so
    they are skipped (pruned). Gives the same answer as Minimax, faster.
    """
    base_name = "Alpha-Beta"

    def _search(self, game, me, root):
        return self._alphabeta(game, me, 0, -math.inf, math.inf, root)

    def _alphabeta(self, game, me, depth, alpha, beta, node):
        self.nodes_evaluated += 1

        leaf = self._leaf_value(game, me, depth)
        if leaf is not None:
            if node is not None:
                node["type"], node["score"] = "LEAF", leaf
            return leaf, None

        maximizing = game.current_player == me
        best_score = -math.inf if maximizing else math.inf
        best_move = None
        moves = self._moves(game)

        for i, move in enumerate(moves):
            child = self._add_child(node, move)
            score, _ = self._alphabeta(game.result(move), me, depth + 1, alpha, beta, child)

            if maximizing:
                if score > best_score:
                    best_score, best_move = score, move
                alpha = max(alpha, best_score)
            else:
                if score < best_score:
                    best_score, best_move = score, move
                beta = min(beta, best_score)

            if alpha >= beta:                       # cut-off: skip the rest
                skipped = moves[i + 1:]
                self.pruned_count += len(skipped)
                for m in skipped:
                    self._add_child(node, m, pruned=True)
                break

        if node is not None:
            node["type"], node["score"] = ("MAX" if maximizing else "MIN"), best_score
        return best_score, best_move


# ----------------------------------------------------------------- expectiminimax
class ExpectiminimaxAgent(SearchAgent):
    """
    Minimax extended with CHANCE nodes for games with randomness.

    MAX node    -> pick the move with the highest expected score
    MIN node    -> pick the move with the lowest expected score
    CHANCE node -> weighted average of all outcomes:  sum(p * value)

    On a deterministic game every move has one outcome, so it behaves
    exactly like Minimax.
    """
    base_name = "Expectiminimax"

    def _search(self, game, me, root):
        return self._expectiminimax(game, me, 0, root)

    def _expectiminimax(self, game, me, depth, node):
        self.nodes_evaluated += 1

        leaf = self._leaf_value(game, me, depth)
        if leaf is not None:
            if node is not None:
                node["type"], node["score"] = "LEAF", leaf
            return leaf, None

        maximizing = game.current_player == me
        best_score = -math.inf if maximizing else math.inf
        best_move = None

        for move in self._moves(game):
            score = self._chance_value(game, move, me, depth, node)
            if (maximizing and score > best_score) or (not maximizing and score < best_score):
                best_score, best_move = score, move

        if node is not None:
            node["type"], node["score"] = ("MAX" if maximizing else "MIN"), best_score
        return best_score, best_move

    def _chance_value(self, game, move, me, depth, node):
        """Expected score of playing `move` (a CHANCE node)."""
        outcomes = game.outcomes(move)

        if len(outcomes) == 1:                       # no randomness: plain child
            child = self._add_child(node, move)
            score, _ = self._expectiminimax(outcomes[0][1], me, depth + 1, child)
            return score

        self.nodes_evaluated += 1                    # count the chance node itself
        chance = self._add_child(node, move, type="CHANCE")
        expected = 0.0
        for prob, state in outcomes:
            child = self._add_child(chance, state.last_move, prob=prob)
            score, _ = self._expectiminimax(state, me, depth + 1, child)
            expected += prob * score
        if chance is not None:
            chance["score"] = expected
        return expected


# ---------------------------------------------------------------- visualisation
def score_label(score):
    """Readable score: 'WIN', 'LOSS', 0, or a heuristic / expected value."""
    if score is None:
        return "?"
    if score >= WIN_SCORE - 1000:
        return "WIN"
    if score <= -WIN_SCORE + 1000:
        return "LOSS"
    if abs(score) >= 10 ** 6:                  # expected value mixing wins/losses
        return f"{score / WIN_SCORE:+.2f}×WIN"
    if isinstance(score, float) and not score.is_integer():
        return f"{score:.2f}"
    return str(int(score))


def format_tree(node, max_depth=2, _prefix="", _depth=0, _is_last=True, _is_root=True):
    """
    Return the recorded search tree as text, e.g.

        ROOT [MAX] = 0
        ├── (1, 1) [MIN] = 0
        │   ├── (0, 0) [MAX] = 0
        │   └── (0, 1)  ✂ PRUNED
        ...

    Use with an agent created with record_tree=True.
    """
    if node is None:
        return "(no tree recorded - create the agent with record_tree=True)"

    label = "ROOT" if _is_root else str(node["move"])
    if node.get("prob") is not None:
        label += f" p={node['prob']:.2f}"
    if node["pruned"]:
        text = f"{label}  ✂ PRUNED"
    else:
        text = f"{label} [{node['type']}] = {score_label(node['score'])}"

    if _is_root:
        lines = [text]
        child_prefix = ""
    else:
        lines = [_prefix + ("└── " if _is_last else "├── ") + text]
        child_prefix = _prefix + ("    " if _is_last else "│   ")

    children = node["children"]
    if children and _depth < max_depth:
        for i, child in enumerate(children):
            lines.append(format_tree(child, max_depth, child_prefix, _depth + 1,
                                     i == len(children) - 1, False))
    elif children:
        lines.append(child_prefix + f"└── ... ({len(children)} children not shown)")
    return "\n".join(lines)