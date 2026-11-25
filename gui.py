"""
gui.py - Graphical interface for CP468 Project 1 (Visualization component).

Run with:   python3 gui.py

What it shows:
  - The board, updated after every move (the last move is highlighted).
  - Human vs AI and AI vs AI (any player can be Human, Random, Minimax, Alpha-Beta, Expectiminimax or Gemini).
  - Decision-making stats for every AI move: nodes evaluated, time, branches
    pruned, the score of the chosen move, and Gemini's own explanation.
  - "Show Search Tree": expandable tree of the last AI search where
    PRUNED branches (Alpha-Beta) are highlighted in red, CHANCE nodes
    (Expectiminimax) in blue and the chosen move in green.

Since Tkinter comes with Python, there is nothing to install.
The AI moves run in the background thread so the window never freezes while an
agent (especially Gemini, which waits for the internet) is thinking.
"""

import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from game import TicTacToe, StochasticTicTacToe
from algorithms import (RandomAgent, MinimaxAgent, AlphaBetaAgent,
                        ExpectiminimaxAgent, SearchAgent, score_label)
from gemini_agent import GeminiAgent, gemini_available

PLAYER_CHOICES = ["Human", "Random", "Minimax", "Alpha-Beta", "Expectiminimax",
                  "Gemini (easy)", "Gemini (medium)", "Gemini (hard)"]
SEARCH_CLASSES = {"Minimax": MinimaxAgent, "Alpha-Beta": AlphaBetaAgent,
                  "Expectiminimax": ExpectiminimaxAgent}

# Colours
LAST_MOVE_BG = "#fff3b0"
PIECE_COLOUR = {"X": "#257de0", "O": "#f47230"}
TREE_TAGS = {"pruned": ("#fde2e2", "#b00020"),   # (background, text)
             "chance": ("#dceaff", "#0e3d7e"),
             "chosen": ("#d9f5dd", "#006414")}


def auto_depth(game):
    """
    Search depth used when the user leaves depth at 0 ("auto").
    Full search is only practical on a normal 3x3 board; bigger or random
    games use a depth limit + the heuristic from heuristics.py.
    """
    stochastic = isinstance(game, StochasticTicTacToe)
    if game.size == 3:
        return 3 if stochastic else None          # None = search to the end
    return 2 if stochastic else (4 if game.size == 4 else 3)


def make_agent(choice, game, depth=0, move_ordering=True):
    """Turn a dropdown choice into an agent object (None means Human)."""
    if choice == "Human":
        return None
    if choice == "Random":
        return RandomAgent()
    if choice.startswith("Gemini"):
        difficulty = choice.split("(")[1].rstrip(")")      # "Gemini (hard)" -> "hard"
        return GeminiAgent(difficulty=difficulty)
    cls = SEARCH_CLASSES[choice]
    max_depth = auto_depth(game) if depth == 0 else depth
    return cls(max_depth=max_depth, move_ordering=move_ordering)


class GameApp:
    def __init__(self, root):
        self.root = root
        root.title("CP468 - Multi-Agent Game Tree Search")
        self.results = queue.Queue()   # background AI thread -> GUI thread
        self.game_id = 0               # changes on New Game so old AI results are ignored
        self.thinking = False
        self.last_search = None        # (position before move, agent) for the tree window

        self._build_controls()
        self._build_info_panel()
        self.board_frame = ttk.Frame(root, padding=10)
        self.board_frame.grid(row=0, column=1, sticky="n")

        self.new_game()
        self._poll_results()

    # ================================================================ layout
    def _build_controls(self):
        panel = ttk.LabelFrame(self.root, text="Settings", padding=10)
        panel.grid(row=0, column=0, sticky="ns", padx=10, pady=10)

        def row(label, widget, r):
            ttk.Label(panel, text=label).grid(row=r, column=0, sticky="w", pady=3)
            widget.grid(row=r, column=1, sticky="ew", pady=3)

        self.size_var = tk.StringVar(value="3")
        size_box = ttk.Combobox(panel, textvariable=self.size_var, values=["3", "4", "5"],
                                width=14, state="readonly")
        size_box.bind("<<ComboboxSelected>>", lambda e: self._update_win_choices())
        row("Board size", size_box, 0)

        self.win_var = tk.StringVar(value="3")
        self.win_box = ttk.Combobox(panel, textvariable=self.win_var, width=14, state="readonly")
        row("In a row to win", self.win_box, 1)
        self._update_win_choices()

        self.type_var = tk.StringVar(value="normal")
        types = ttk.Frame(panel)
        ttk.Radiobutton(types, text="Normal", value="normal",
                        variable=self.type_var).pack(side="left")
        ttk.Radiobutton(types, text="Stochastic", value="stochastic",
                        variable=self.type_var).pack(side="left")
        row("Game type", types, 2)

        self.x_var = tk.StringVar(value="Human")
        self.o_var = tk.StringVar(value="Alpha-Beta")
        row("Player X", ttk.Combobox(panel, textvariable=self.x_var, values=PLAYER_CHOICES,
                                     width=14, state="readonly"), 3)
        row("Player O", ttk.Combobox(panel, textvariable=self.o_var, values=PLAYER_CHOICES,
                                     width=14, state="readonly"), 4)

        self.depth_var = tk.IntVar(value=0)
        row("Search depth\n(0 = auto)", ttk.Spinbox(panel, from_=0, to=9,
                                                    textvariable=self.depth_var, width=5), 5)
        self.ordering_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(panel, text="Move ordering (centre first)",
                        variable=self.ordering_var).grid(row=6, column=0, columnspan=2,
                                                         sticky="w", pady=3)

        ttk.Button(panel, text="New Game", command=self.new_game).grid(
            row=7, column=0, columnspan=2, sticky="ew", pady=(12, 3))
        ttk.Button(panel, text="Show Search Tree", command=self.show_tree).grid(
            row=8, column=0, columnspan=2, sticky="ew", pady=3)

        self.status = ttk.Label(panel, text="", font=("TkDefaultFont", 11, "bold"),
                                wraplength=200)
        self.status.grid(row=9, column=0, columnspan=2, sticky="w", pady=(12, 0))

    def _update_win_choices(self):
        size = int(self.size_var.get())
        self.win_box["values"] = [str(k) for k in range(3, size + 1)]
        self.win_var.set(str(3 if size == 3 else 4))

    def _build_info_panel(self):
        panel = ttk.LabelFrame(self.root, text="Last AI decision", padding=10)
        panel.grid(row=0, column=2, sticky="ns", padx=10, pady=10)
        self.stat_vars = {}
        for i, key in enumerate(["Agent", "Move", "Score", "Nodes evaluated",
                                 "Branches pruned", "Time"]):
            ttk.Label(panel, text=key + ":").grid(row=i, column=0, sticky="w")
            self.stat_vars[key] = tk.StringVar(value="-")
            ttk.Label(panel, textvariable=self.stat_vars[key], width=22).grid(
                row=i, column=1, sticky="w")

        ttk.Label(panel, text="Move log:").grid(row=6, column=0, sticky="w", pady=(10, 0))
        self.log = tk.Text(panel, width=42, height=18, wrap="word", state="disabled")
        self.log.grid(row=7, column=0, columnspan=2)

    def _build_board(self):
        for widget in self.board_frame.winfo_children():
            widget.destroy()
        size = self.game.size
        font_size = {3: 32, 4: 26, 5: 20}[size]
        self.cells = {}
        for r in range(size):
            for c in range(size):
                b = tk.Button(self.board_frame, text="", width=3, height=1,
                              font=("Helvetica", font_size, "bold"),
                              command=lambda m=(r, c): self.human_click(m))
                b.grid(row=r, column=c, padx=2, pady=2)
                self.cells[(r, c)] = b

    # ============================================================ game flow
    def new_game(self):
        size, win = int(self.size_var.get()), int(self.win_var.get())
        if self.type_var.get() == "stochastic":
            self.game = StochasticTicTacToe(size, win, success_prob=0.8)
        else:
            self.game = TicTacToe(size, win)

        depth, ordering = self.depth_var.get(), self.ordering_var.get()
        self.agents = {"X": make_agent(self.x_var.get(), self.game, depth, ordering),
                       "O": make_agent(self.o_var.get(), self.game, depth, ordering)}

        if any(isinstance(a, GeminiAgent) for a in self.agents.values()) \
                and not gemini_available():
            messagebox.showwarning("Gemini", "No Gemini API key found (set GEMINI_API_KEY "
                                   "or config.json). Gemini will play random moves.")

        self.game_id += 1
        self.thinking = False
        self.last_search = None
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        for var in self.stat_vars.values():
            var.set("-")
        self._build_board()
        self.write_log(f"New game: {size}x{size}, {win} in a row"
                       + (" (stochastic, 80% land)" if self.type_var.get() == "stochastic" else ""))
        self.write_log(f"X = {self._name('X')},  O = {self._name('O')}\n")
        self.refresh_board()
        self.next_turn()

    def _name(self, player):
        return self.agents[player].name if self.agents[player] else "Human"

    def next_turn(self):
        """Decide what happens next: game over, wait for a human, or start the AI."""
        if self.game.is_over():
            w = self.game.winner()
            text = "Draw!" if w == "draw" else f"{w} ({self._name(w)}) wins!"
            self.status.config(text=text)
            self.write_log("Result: " + text)
            return

        player = self.game.current_player
        agent = self.agents[player]
        if agent is None:
            self.status.config(text=f"Your turn ({player})")
            return

        # Run the AI in a background thread; the answer comes back through self.results.
        self.thinking = True
        self.status.config(text=f"{agent.name} ({player}) is thinking...")
        position, game_id = self.game, self.game_id

        def work():
            try:
                self.results.put((game_id, position, agent, agent.choose_move(position), None))
            except Exception as error:
                self.results.put((game_id, position, agent, None, error))

        threading.Thread(target=work, daemon=True).start()

    def _poll_results(self):
        """Check every 50 ms whether the AI thread has finished (Tkinter is not thread-safe)."""
        try:
            while True:
                game_id, position, agent, move, error = self.results.get_nowait()
                if game_id == self.game_id:          # ignore results from an old game
                    self.thinking = False
                    if error:
                        messagebox.showerror("AI error", str(error))
                    else:
                        self.apply_ai_move(position, agent, move)
        except queue.Empty:
            pass
        self.root.after(50, self._poll_results)

    def apply_ai_move(self, position, agent, move):
        player = position.current_player
        self.game = position.make_move(move)
        self.last_search = (position, agent)
        self.show_stats(agent, move)

        line = f"{player} {agent.name}: {move}"
        if self.game.last_move != move:
            line += f"  -> slipped to {self.game.last_move}!"
        if isinstance(agent, GeminiAgent) and agent.last_reason:
            line += f"\n    \"{agent.last_reason}\""
        self.write_log(line)

        self.refresh_board()
        self.root.after(400, self.next_turn)        # short pause so AI vs AI is watchable

    def human_click(self, move):
        if self.thinking or self.game.is_over() or self.agents[self.game.current_player]:
            return                                  # not the human's turn
        if move not in self.game.get_valid_moves():
            return
        player = self.game.current_player
        self.game = self.game.make_move(move)
        line = f"{player} Human: {move}"
        if self.game.last_move != move:
            line += f"  -> slipped to {self.game.last_move}!"
        self.write_log(line)
        self.refresh_board()
        self.next_turn()

    # ============================================================== display
    def refresh_board(self):
        for (r, c), button in self.cells.items():
            piece = self.game.board[r][c]
            button.config(text=piece.strip(), fg=PIECE_COLOUR.get(piece, "black"),
                          bg=LAST_MOVE_BG if (r, c) == self.game.last_move else "white",
                          activebackground="#eeeeee")

    def show_stats(self, agent, move):
        s = self.stat_vars
        s["Agent"].set(agent.name)
        s["Move"].set(str(move))
        s["Time"].set(f"{agent.last_time:.3f} s")
        if isinstance(agent, SearchAgent):
            s["Score"].set(score_label(agent.last_score))
            s["Nodes evaluated"].set(f"{agent.nodes_evaluated:,}")
            s["Branches pruned"].set(f"{agent.pruned_count:,}")
        elif isinstance(agent, GeminiAgent):
            s["Score"].set("(LLM - no score)")
            s["Nodes evaluated"].set(f"0  ({agent.api_calls} API call(s))")
            s["Branches pruned"].set("fallback used" if agent.used_fallback else "-")
        else:
            for key in ("Score", "Nodes evaluated", "Branches pruned"):
                s[key].set("-")

    def write_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    # ========================================================= search tree
    def show_tree(self):
        """Re-run the last AI search with tree recording on and show it."""
        if not self.last_search or not isinstance(self.last_search[1], SearchAgent):
            messagebox.showinfo("Search tree", "Let a Minimax, Alpha-Beta or "
                                "Expectiminimax agent make a move first.")
            return
        position, agent = self.last_search

        # Recording every node of a huge search would use too much memory,
        # so very large searches are re-run with a smaller depth for display.
        depth, note = agent.max_depth, ""
        if agent.nodes_evaluated > 150_000:
            depth, note = 3, "  (search was large - tree shown to depth 3)"
        recorder = type(agent)(max_depth=depth, move_ordering=agent.move_ordering,
                               record_tree=True)
        chosen = recorder.choose_move(position)

        win = tk.Toplevel(self.root)
        win.title(f"Search tree - {recorder.name}")
        ttk.Label(win, text=str(position), font=("Courier", 11)).pack(padx=10, pady=(10, 0))
        ttk.Label(win, text=f"{recorder.name} as {position.current_player}: "
                            f"{recorder.nodes_evaluated:,} nodes, "
                            f"{recorder.pruned_count:,} branches pruned, "
                            f"chose {chosen}{note}").pack(padx=10, pady=5)

        legend = ttk.Frame(win)
        legend.pack()
        for tag, text in (("chosen", "chosen move"), ("pruned", "✂ pruned (skipped)"),
                          ("chance", "chance node (random outcome)")):
            bg, fg = TREE_TAGS[tag]
            tk.Label(legend, text=text, bg=bg, fg=fg, padx=6).pack(side="left", padx=4)

        tree = ttk.Treeview(win, columns=("type", "score", "prob"), height=22)
        tree.heading("#0", text="Move")
        tree.heading("type", text="Node type")
        tree.heading("score", text="Score")
        tree.heading("prob", text="Probability")
        tree.column("#0", width=260)
        for col in ("type", "score", "prob"):
            tree.column(col, width=110, anchor="center")
        for tag, (bg, fg) in TREE_TAGS.items():
            tree.tag_configure(tag, background=bg, foreground=fg)
        tree.pack(fill="both", expand=True, padx=10, pady=10)

        # The tree can have thousands of nodes, so children are only added to the
        # widget when their parent is expanded ("lazy loading").
        nodes = {}

        def insert(parent_id, node, is_chosen=False):
            tags = ("pruned",) if node["pruned"] else \
                   ("chosen",) if is_chosen else \
                   ("chance",) if node["type"] == "CHANCE" else ()
            label = "ROOT" if node["move"] is None else str(node["move"])
            values = ("✂ PRUNED", "", "") if node["pruned"] else (
                node["type"], score_label(node["score"]),
                f"{node['prob']:.2f}" if node.get("prob") is not None else "")
            item = tree.insert(parent_id, "end", text=label, values=values, tags=tags)
            nodes[item] = node
            if node["children"]:
                tree.insert(item, "end", text="loading...")   # placeholder -> shows a ▸ arrow
            return item

        def on_open(_event):
            item = tree.focus()
            children = tree.get_children(item)
            if len(children) == 1 and tree.item(children[0], "text") == "loading...":
                tree.delete(children[0])
                for child in nodes[item]["children"]:
                    insert(item, child, is_chosen=(item == root_item and
                                                   child["move"] == chosen))

        tree.bind("<<TreeviewOpen>>", on_open)
        root_item = insert("", recorder.tree)
        tree.focus(root_item)
        tree.item(root_item, open=True)
        on_open(None)


if __name__ == "__main__":
    root = tk.Tk()
    GameApp(root)
    root.mainloop()
