# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Search, Planning and Discovery".

Plain Python 3, standard library only, written from the chapter's own description. The shop of the first chapter (`mdps`,
imported, never copied) is searched by a lookahead tree and compared with the sweeps of value iteration; a small grid
with mud is searched by breadth-first, uniform-cost, greedy best-first and A* search with every expansion counted; a
small network is trained to predict the cost to go on that grid and used as a heuristic; a take-away game is solved by
minimax and played by Monte Carlo tree search with the UCT rule and by a policy-guided variant (PUCT) inside a loop that
trains the policy on the search's own visit counts; a deceptive two-peak landscape is searched by a hill climber, by a
best-N population and by a MAP-Elites archive; a sparse-reward goal-reaching task is learned with and without hindsight
relabelling; and brain's two crates are reproduced from their rules: the archive's selection weight and admission rule,
the bandit that allocates search budget, and the retraced random walks of `crates/solve` on a permutation puzzle. Each
public function without a leading underscore that takes no arguments returns a markdown string that the chapter embeds
between `<!-- worked: search_planning.<name> -->` markers. Underscore functions and the constants are shared with the
chapter's figures, so a picture and the table beside it cannot disagree.

Anything sampled goes through the nudge rule of `stable`: the computation is repeated with a constant it really uses
(the exploration constant, the mutation width, an initial logit, the first weight of a network) changed by each of the
nudges of `stable.py`, and a number is printed only with the digits all five runs share. A count that is an integer is
printed when all five agree and as "unstable" when they do not.

Swedish Embedded AB implements solutions for planning and discovery systems that search first and learn from what the
search finds. If your team needs expertise in heuristic search, Monte Carlo tree search and quality-diversity methods,
you can procure our services by sending an email to info@swedishembedded.com.
"""
import heapq
import math
from functools import lru_cache

import bookrng
import mdps
import stable

VARIANTS = (0.0, *stable.NUDGES)       # the computation itself, then four copies with one used constant nudged

# ------------------------------------------------------------------ shared helpers


def _table(headers, rows, align=None):
    return mdps._table(headers, rows, align)


def _fmt(x, nd=3):
    return mdps._fmt(x, nd)


def _int_stable(values):
    """The integer all runs agree on, else 'unstable'."""
    return str(values[0]) if len(set(values)) == 1 else "unstable"


def _grid_text(rows):
    return "\n".join(rows)


# ------------------------------------------------------------------ the shop looked at from the root

LOOK_DEPTHS = tuple(range(1, 9))
LOOK_GAMMA = mdps.GAMMA


def _lookahead_values(depth, leaf):
    """Q_d(s, a) for every legal pair when the leaves of a tree of `depth` days are valued by `leaf` (a list over states):
    Q_1(s, a) = q(s, a; leaf), Q_d(s, a) = q(s, a; V_{d-1}) with V_{d-1}(s) = max_a Q_{d-1}(s, a), V_0 = leaf. The tree
    is evaluated with sharing here; the number of nodes of the unshared tree is counted by `_tree_nodes`."""
    V = list(leaf)
    qs = None
    for _ in range(depth):
        qs = [[mdps._q(s, a, V, LOOK_GAMMA) for a in range(mdps._n_actions(s))] for s in range(mdps.N_STATES)]
        V = [max(row) for row in qs]
    return qs, V


@lru_cache(maxsize=None)
def _tree_nodes(depth, s):
    """State nodes of the unshared lookahead tree below a node of stock s with `depth` days left, itself included: a node
    with days left has one child for every pair (a, j) of an order a and a next stock j with P(j | s, a) > 0."""
    if depth == 0:
        return 1
    total = 1
    for a in range(mdps._n_actions(s)):
        for j in range(mdps.N_STATES):
            if mdps.PROB[s][a][j] > 0:
                total += _tree_nodes(depth - 1, j)
    return total


def _first_best(qs_row):
    best = max(qs_row)
    return next(a for a, q in enumerate(qs_row) if q >= best - mdps.TIE)


def shop_lookahead():
    v_star = mdps._value_iteration(LOOK_GAMMA)[0]
    rows = []
    for d in LOOK_DEPTHS:
        q0, v0 = _lookahead_values(d, [0.0] * mdps.N_STATES)
        q1, _ = _lookahead_values(d, v_star)
        rows.append([d, _tree_nodes(d, 0), mdps.N_STATES * d, ", ".join(_fmt(x, 3) for x in q0[0]),
                     _first_best(q0[0]), _first_best(q1[0])])
    sweeps = _lookahead_values(LOOK_DEPTHS[-1], [0.0] * mdps.N_STATES)[1]
    hist = mdps._value_iteration(LOOK_GAMMA)[3]
    gap = max(abs(a - b) for a, b in zip(sweeps, hist[LOOK_DEPTHS[-1]]))
    return "\n".join([
        f"The shop of Chapter 1 (tables as printed there) at gamma = {LOOK_GAMMA}, searched from the empty shelf (stock 0). "
        "A lookahead of d days is the tree whose root is the state, whose children are the pairs (order a, next stock j) with "
        "P(j | s, a) > 0 and whose leaves lie d days down; a leaf is valued by a given vector (all zeros, or the exact V* of "
        "Chapter 1) and an inner node by the best order of R(s, a) + gamma times the probability-weighted mean of its "
        "children. 'Nodes' counts every state node of the tree without sharing, root and leaves included; 'values with "
        "sharing' is the number of (stock, days-left) pairs that need a value when equal subtrees are computed once, "
        "4 stocks times d. The columns Q_d(0, a) are the lookahead values of the orders a = 0, 1, 2, 3 at the root with "
        "zero leaves; 'best a' is the smallest order within 1e-9 of the largest value (the tie rule of Chapter 1).",
        "",
        _table(["d", "nodes", "values with sharing", "Q_d(0, a) with zero leaves", "best a, zero leaves", "best a, leaves V*"],
               rows, "lrrrrr"),
        "",
        f"The root values with sharing are the iterates of value iteration from V_0 = 0: the largest difference between the "
        f"{LOOK_DEPTHS[-1]}-day lookahead values of the four stocks and the vector after {LOOK_DEPTHS[-1]} sweeps of Chapter 1 is "
        f"{gap:.1e} (below 1e-12).",
    ])


# ------------------------------------------------------------------ the grid with mud

GRID = (
    "....S.......",
    "............",
    "~~~.~~~~~~~#",
    "~~~.~~~~~~~#",
    "~~~.~~~~~~~.",
    "......###...",
    "..........G.",
)
ROWS, COLS = len(GRID), len(GRID[0])
MOVES4 = ((-1, 0), (0, 1), (1, 0), (0, -1))      # up, right, down, left: the order in which neighbours are generated


def _find(ch):
    for r in range(ROWS):
        for c in range(COLS):
            if GRID[r][c] == ch:
                return (r, c)


START, GOAL_CELL = _find("S"), _find("G")


def _enter_cost(cell):
    """What it costs to move onto a cell: 3 on mud, 1 anywhere else."""
    return 3 if GRID[cell[0]][cell[1]] == "~" else 1


def _unit_cost(cell):
    return 1


def _neighbours(cell):
    r, c = cell
    for dr, dc in MOVES4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < ROWS and 0 <= cc < COLS and GRID[rr][cc] != "#":
            yield (rr, cc)


def _manhattan(cell):
    return abs(cell[0] - GOAL_CELL[0]) + abs(cell[1] - GOAL_CELL[1])


def _best_first(step_cost, priority, h=_manhattan, start=START, goal=GOAL_CELL):
    """The one routine behind breadth-first, uniform-cost, greedy and A* search.

    Queue entries are (priority(g, h(cell)), counter, cell, g): the counter is the number of pushes so far, so equal
    priorities leave the queue in the order they entered. The goal is tested when an entry is popped. A popped cell that
    is already closed is skipped and is not an expansion. An expansion closes the cell and generates its neighbours in the
    order up, right, down, left; a neighbour that is open or unseen is pushed whenever this route to it has a lower g
    than any route found so far. Returns (expansion order, path, pushes)."""
    counter = 0
    best_g = {start: 0.0}
    parent = {}
    queue = [(priority(0.0, h(start)), counter, start, 0.0)]
    counter += 1
    closed, order = set(), []
    while queue:
        _, _, cell, g = heapq.heappop(queue)
        if cell in closed:
            continue
        if cell == goal:
            path = [cell]
            while path[-1] != start:
                path.append(parent[path[-1]])
            path.reverse()
            return order, path, counter
        closed.add(cell)
        order.append(cell)
        for nxt in _neighbours(cell):
            if nxt in closed:
                continue
            g2 = g + step_cost(nxt)
            if g2 < best_g.get(nxt, math.inf):
                best_g[nxt] = g2
                parent[nxt] = cell
                heapq.heappush(queue, (priority(g2, h(nxt)), counter, nxt, g2))
                counter += 1
    return order, None, counter


def _path_cost(path):
    return sum(_enter_cost(c) for c in path[1:])


METHODS = (
    ("breadth-first", _unit_cost, lambda g, h: g),
    ("uniform-cost", _enter_cost, lambda g, h: g),
    ("A*, f = g + 0.5 h", _enter_cost, lambda g, h: g + 0.5 * h),
    ("A*, f = g + h", _enter_cost, lambda g, h: g + h),
    ("A*, f = g + 2 h", _enter_cost, lambda g, h: g + 2 * h),
    ("A*, f = g + 5 h", _enter_cost, lambda g, h: g + 5 * h),
    ("greedy, f = h", _enter_cost, lambda g, h: h),
)


def _run_method(name):
    for n, step, prio in METHODS:
        if n == name:
            return _best_first(step, prio)
    raise KeyError(name)


OPTIMAL_COST = _path_cost(_run_method("uniform-cost")[1])


def grid_map():
    cells = sum(1 for r in range(ROWS) for c in range(COLS) if GRID[r][c] != "#")
    mud = sum(1 for r in range(ROWS) for c in range(COLS) if GRID[r][c] == "~")
    walls = ROWS * COLS - cells
    return "\n".join([
        f"The map, {ROWS} rows by {COLS} columns, rows numbered from 0 at the top and columns from 0 at the left, a cell "
        f"written (row, column). S is the start, (0, 4); G is the goal, (6, 10); '.' is open ground and S and G are open "
        f"ground too; '~' is mud; '#' is a wall. There are {cells} cells that can be entered ({mud} of them mud) and "
        f"{walls} walls. Moving onto open ground costs 1 and onto mud costs 3; a move goes one cell up, right, down or left "
        "and never onto a wall or off the map.",
        "",
        "```text",
        *GRID,
        "```",
    ])


def search_methods():
    rows = []
    for name, step, prio in METHODS:
        order, path, pushes = _best_first(step, prio)
        cost = _path_cost(path)
        rows.append([name, len(order), pushes, len(path) - 1, cost, "yes" if cost == OPTIMAL_COST else f"no (+{cost - OPTIMAL_COST})"])
    return "\n".join([
        "Seven searches of the map above, all from the routine of the chapter. The step cost is 1 for breadth-first search "
        "(it counts moves, so its g is the number of moves) and the cost of the cell entered for the others; the heuristic "
        "is the Manhattan distance h(x) = |row - 6| + |column - 10| to the goal. Entries leave the queue in the order of "
        "their priority, ties in the order they entered; neighbours are generated up, right, down, left; the goal is "
        "tested when its entry is popped; a popped cell that is already closed is skipped and does not count. 'expansions' "
        "counts the cells closed, 'pushes' the entries ever put into the queue (the start included); 'moves' and 'cost' are "
        "those of the path found, the cost with mud counted at 3.",
        "",
        _table(["method", "expansions", "pushes", "moves", "cost", "cost-optimal"], rows, "lrrrrl"),
    ])


def _expansion_numbers(name):
    """(cell -> number of its expansion counted from 1, set of cells of the returned path) for a method of `METHODS`."""
    order, path, _ = _run_method(name)
    return {cell: i + 1 for i, cell in enumerate(order)}, set(path)


def _expansion_grid(name):
    order, path, _ = _run_method(name)
    number = {cell: i + 1 for i, cell in enumerate(order)}
    on_path = set(path)
    lines = []
    for r in range(ROWS):
        cells = []
        for c in range(COLS):
            cell = (r, c)
            if cell in number:
                cells.append(f"{number[cell]:>3}" + ("*" if cell in on_path else " "))
            elif cell == GOAL_CELL:
                cells.append("  G*")
            else:
                cells.append(f"  {GRID[r][c]} ")
        lines.append("".join(cells))
    return lines


def expansion_orders():
    out = [
        "Each cell shows the number of the expansion that closed it, counted from 1 (so S is 1); a star marks a cell of the "
        "path that was returned; G was never expanded because the search stopped when its entry was popped. "
        "Cells never expanded show their map symbol.",
        "",
        "Uniform-cost search:",
        "",
        "```text",
        *_expansion_grid("uniform-cost"),
        "```",
        "",
        "A* with f = g + h:",
        "",
        "```text",
        *_expansion_grid("A*, f = g + h"),
        "```",
    ]
    return "\n".join(out)


# ------------------------------------------------------------------ a learned heuristic

NET_HIDDEN, NET_LR = 12, 0.1
NET_CHECKPOINTS = (100, 300, 1000, 3000, 10000)
NET_NAME = "search_planning.heuristic_net"


def _cost_to_go():
    """The exact cost to go of every enterable cell: the cost of the cheapest path to G (a move onto a cell pays that
    cell's cost), by Dijkstra from the goal over reversed moves."""
    ctg = {GOAL_CELL: 0}
    queue = [(0, GOAL_CELL)]
    while queue:
        d, x = heapq.heappop(queue)
        if d > ctg[x]:
            continue
        for y in _neighbours(x):
            nd = d + _enter_cost(x)
            if nd < ctg.get(y, math.inf):
                ctg[y] = nd
                heapq.heappush(queue, (nd, y))
    return ctg


CTG = _cost_to_go()
CELLS = sorted(CTG)
SCALE = float(CTG[START])


def _features(cell):
    return (cell[0] / (ROWS - 1), cell[1] / (COLS - 1))


def _train_net(nudge):
    """Full-batch gradient descent on the mean squared error between the network's output and CTG / SCALE over all 79
    enterable cells. Two inputs (row / 6, column / 11), 12 tanh hidden units, one linear output. Initial weights from the
    book's generator in this order: W1[i][k] for i = 0, 1 and k = 0 .. 11 (row-major) each normal() / sqrt(2), then W2[k]
    for k = 0 .. 11 each normal() / sqrt(12); all biases 0. `nudge` is added to W1[0][0]. Returns the predictor after each
    checkpoint of NET_CHECKPOINTS and the training loss there."""
    rng = bookrng.generator(NET_NAME)
    H = NET_HIDDEN
    W1 = [[rng.normal() / math.sqrt(2.0) for _ in range(H)] for _ in range(2)]
    W2 = [rng.normal() / math.sqrt(H) for _ in range(H)]
    W1[0][0] += nudge
    b1, b2 = [0.0] * H, 0.0
    X = [_features(c) for c in CELLS]
    Y = [CTG[c] / SCALE for c in CELLS]
    n = len(X)
    done = {}
    for step in range(1, NET_CHECKPOINTS[-1] + 1):
        gW1 = [[0.0] * H for _ in range(2)]
        gb1, gW2, gb2 = [0.0] * H, [0.0] * H, 0.0
        loss = 0.0
        for x, y in zip(X, Y):
            hid = [math.tanh(x[0] * W1[0][k] + x[1] * W1[1][k] + b1[k]) for k in range(H)]
            p = sum(W2[k] * hid[k] for k in range(H)) + b2
            e = p - y
            loss += e * e
            d = 2.0 * e / n
            gb2 += d
            for k in range(H):
                gW2[k] += d * hid[k]
                dh = d * W2[k] * (1.0 - hid[k] * hid[k])
                gW1[0][k] += dh * x[0]
                gW1[1][k] += dh * x[1]
                gb1[k] += dh
        for k in range(H):
            W2[k] -= NET_LR * gW2[k]
            b1[k] -= NET_LR * gb1[k]
            W1[0][k] -= NET_LR * gW1[0][k]
            W1[1][k] -= NET_LR * gW1[1][k]
        b2 -= NET_LR * gb2
        if step in NET_CHECKPOINTS:
            snapshot = ([row[:] for row in W1], b1[:], W2[:], b2)

            def predict(cell, snap=snapshot):
                w1, bb1, w2, bb2 = snap
                x = _features(cell)
                hid = [math.tanh(x[0] * w1[0][k] + x[1] * w1[1][k] + bb1[k]) for k in range(H)]
                return (sum(w2[k] * hid[k] for k in range(H)) + bb2) * SCALE

            done[step] = (predict, loss / n)
    return done


@lru_cache(maxsize=None)
def _nets():
    return [_train_net(nu) for nu in VARIANTS]


def _astar_with(h):
    order, path, _ = _best_first(_enter_cost, lambda g, hv: g + hv, h=h)
    return len(order), _path_cost(path)


def learned_heuristic():
    runs = _nets()
    rows = []
    for step in NET_CHECKPOINTS:
        stats = []
        for done in runs:
            predict, loss = done[step]
            errs = [predict(c) - CTG[c] for c in CELLS]
            over = max(errs)
            raw = _astar_with(lambda c, p=predict: max(0.0, p(c)))
            safe = _astar_with(lambda c, p=predict, o=over: max(0.0, p(c) - o))
            stats.append((loss, over, sum(1 for e in errs if e > 1e-9), raw, safe))
        loss_t = stable.text(stats[0][0], [s[0] for s in stats[1:]], lambda x: f"{x:.5f}")
        over_t = stable.text(stats[0][1], [s[1] for s in stats[1:]], lambda x: f"{x:.2f}")
        cnt_t = _int_stable([s[2] for s in stats])
        raw_e = _int_stable([s[3][0] for s in stats])
        raw_c = _int_stable([s[3][1] for s in stats])
        safe_e = _int_stable([s[4][0] for s in stats])
        safe_c = _int_stable([s[4][1] for s in stats])
        rows.append([step, loss_t, cnt_t, over_t, raw_e, raw_c, safe_e, safe_c])
    exact = _astar_with(lambda c: float(CTG[c]))
    manh = _astar_with(_manhattan)
    zero = _astar_with(lambda c: 0.0)
    return "\n".join([
        f"A network trained to predict the exact cost to go of the map above. Training set: all {len(CELLS)} enterable cells, "
        f"target CTG(x) / {int(SCALE)} where CTG is the cost of the cheapest path from the cell to G (CTG(S) = {int(SCALE)}). "
        "Network: inputs (row / 6, column / 11), 12 tanh units, one linear output; all biases start at 0; "
        f"initial weights from the book's generator `{NET_NAME}`, {mdps._seed_text(NET_NAME)}: W1 (2 x 12, row-major, "
        "each normal() divided by sqrt(2)), then W2 (12 values, each normal() divided by sqrt(12)). Loss: mean squared "
        "error over the 79 cells; full-batch gradient descent, learning rate 0.1, tanh the correctly rounded library tanh. "
        "The prediction of a cell is the output times 14. Each line is the network after the stated number of steps. "
        "'too high' counts the cells whose prediction exceeds CTG; 'largest excess' is the largest prediction minus CTG. "
        "A* then uses h = max(0, prediction), 'raw', or h = max(0, prediction - largest excess), 'shifted', which never "
        "exceeds CTG on these cells. Priority g + h, the conventions of the previous table. Digits follow the nudge rule "
        "of `stable.py` (the first weight W1[0][0] changed by each nudge); the counts are integers.",
        "",
        _table(["steps", "training loss", "cells too high", "largest excess", "raw: expansions", "raw: cost",
                "shifted: expansions", "shifted: cost"], rows),
        "",
        f"For reference, the same search with h = CTG (exact) expands {exact[0]} cells and finds cost {exact[1]}; with the "
        f"Manhattan distance {manh[0]} cells, cost {manh[1]}; with h = 0 (uniform-cost) {zero[0]} cells, cost {zero[1]}.",
    ])


# ------------------------------------------------------------------ the take-away game

PILE = 10
GAME_MAX = 3


def _legal(pile):
    return range(1, min(GAME_MAX, pile) + 1)


@lru_cache(maxsize=None)
def _wins(pile):
    """True when the player to move with `pile` stones left wins with best play: she takes 1, 2 or 3 stones and whoever takes
    the last stone wins; with no stones left the player to move has already lost."""
    return any(not _wins(pile - m) for m in _legal(pile))


def _winning_moves(pile):
    return [m for m in _legal(pile) if not _wins(pile - m)]


@lru_cache(maxsize=None)
def _game_tree_nodes(pile):
    """Nodes of the unshared game tree from a pile: the position itself and the trees of the positions after each legal move."""
    return 1 + sum(_game_tree_nodes(pile - m) for m in _legal(pile))


def take_away_game():
    rows = []
    for n in range(0, 14):
        rows.append([n, "-" if n == 0 else ("wins" if _wins(n) else "loses"),
                     "-" if n == 0 else (", ".join(map(str, _winning_moves(n))) or "none")])
    return "\n".join([
        "The take-away game: a pile of n stones, two players alternate, each takes 1, 2 or 3 stones (never more than are "
        "left), and whoever takes the last stone wins. Solved exactly by induction on n: the player to move wins when some "
        "move leaves a position that loses for the other player, and with 0 stones the player to move has lost.",
        "",
        _table(["n", "player to move", "winning moves"], rows, "lrl"),
        "",
        "The unshared game tree from a pile of n stones has T(n) = 1 + T(n - 1) + T(n - 2) + T(n - 3) nodes (terms with "
        "a negative argument left out) with T(0) = 1; counting the position itself: "
        + ", ".join(f"T({n}) = {_game_tree_nodes(n):,}".replace(",", " ") for n in (10, 20, 30, 40)) + ".",
    ])


UCT_C = math.sqrt(2.0)
UCT_NAME = "search_planning.uct"


class _Tree:
    """Nodes in arrays: node 0 is the root. pile[v] stones left, depth[v] moves played to get there, parent[v], move[v] the
    move into v, N[v] visits and W[v] the number of visits whose playout was won by the player who made the move into v
    (the root has no such player, its W stays 0), child[v] the map from move to node."""

    def __init__(self, pile):
        self.pile, self.depth, self.parent, self.move = [pile], [0], [-1], [0]
        self.N, self.W, self.child = [0], [0], [{}]
        self.prior = [None]

    def new(self, v, m):
        self.pile.append(self.pile[v] - m)
        self.depth.append(self.depth[v] + 1)
        self.parent.append(v)
        self.move.append(m)
        self.N.append(0)
        self.W.append(0)
        self.child.append({})
        self.prior.append(None)
        self.child[v][m] = len(self.pile) - 1
        return len(self.pile) - 1


def _rollout_uniform(pile, depth, rng):
    """Uniform random moves to the end: one rng.integer(min(3, pile)) per move, move = 1 + draw. Returns the winner as
    the parity of the number of the move that took the last stone (player 0 moves at even depths): (depth - 1) mod 2,
    and the moves played."""
    played = []
    while pile > 0:
        m = 1 + rng.integer(min(GAME_MAX, pile))
        played.append(m)
        pile -= m
        depth += 1
    return (depth - 1) % 2, played


def _uct(n0, iters, c, rng, trace=None):
    t = _Tree(n0)
    for it in range(iters):
        v = 0
        chosen = []
        while t.pile[v] > 0 and len(t.child[v]) == min(GAME_MAX, t.pile[v]):
            best, best_score = -1, -math.inf
            for m in _legal(t.pile[v]):
                u = t.child[v][m]
                score = t.W[u] / t.N[u] + c * math.sqrt(math.log(t.N[v]) / t.N[u])
                if score > best_score:
                    best, best_score = u, score
            v = best
            chosen.append(t.move[v])
        expanded = None
        if t.pile[v] > 0:
            m = next(m for m in _legal(t.pile[v]) if m not in t.child[v])
            v = t.new(v, m)
            expanded = m
        winner, played = _rollout_uniform(t.pile[v], t.depth[v], rng)
        w = v
        while True:
            t.N[w] += 1
            if t.depth[w] >= 1 and (t.depth[w] - 1) % 2 == winner:
                t.W[w] += 1
            if t.parent[w] < 0:
                break
            w = t.parent[w]
        if trace is not None:
            trace.append((it + 1, chosen, expanded, played, winner))
    return t


def _best_root_move(t):
    """The root move with the most visits, a tie to the smaller move."""
    return max(sorted(t.child[0]), key=lambda m: (t.N[t.child[0][m]], -m))


def uct_trace():
    t0 = []
    rng = bookrng.generator(UCT_NAME)
    t = _uct(PILE, 12, UCT_C, rng, t0)
    rows = []
    for it, chosen, expanded, played, winner in t0:
        rows.append([it, " ".join(map(str, chosen)) or "-", expanded if expanded else "-", " ".join(map(str, played)) or "-",
                     "player 0" if winner == 0 else "player 1"])
    stats = []
    for m in sorted(t.child[0]):
        u = t.child[0][m]
        stats.append([m, PILE - m, t.N[u], t.W[u], _fmt(t.W[u] / t.N[u], 3)])
    return "\n".join([
        f"The first 12 iterations of Monte Carlo tree search on the take-away game from n = {PILE}, exploration constant "
        f"c = sqrt(2) = {UCT_C:.6f}. Player 0 moves at the root. Generator: the book's SplitMix64, "
        f"`{UCT_NAME}` {mdps._seed_text(UCT_NAME)}; the only draws are the playouts, one rng.integer(k) for each move, "
        "k = min(3, stones left), the move being 1 + the draw, consumed in iteration order. Selection: from the root, "
        "while every move of the node has a child, go to the child that maximises W/N + c sqrt(ln N_parent / N) (a tie "
        "to the smaller move), where N is a node's visit count and W its number of visits whose playout was won by the "
        "player who moved into it. Expansion: the smallest move without a child gets one. Playout: random moves to the "
        "end from the new node. Backup: N + 1 on every node of the path, W + 1 on those whose mover won. "
        "'selected' lists the moves of the selection part of the path, 'new' the move expanded and 'playout' the moves "
        "of the playout.",
        "",
        _table(["iteration", "selected", "new", "playout", "winner"], rows, "rlrll"),
        "",
        "After these 12 iterations the root's children are:",
        "",
        _table(["move", "stones left", "N", "W", "W / N"], stats, "rrrrr"),
    ])


def _uct_variants(n0, iters, c, rng_state):
    """The run for each variant of c (c itself and the four nudged copies), each started from the same generator state."""
    out = []
    for nu in VARIANTS:
        rng = bookrng.SplitMix64(rng_state)
        out.append(_uct(n0, iters, c + nu, rng))
    return out


def uct_visits():
    base = bookrng.seed_of(UCT_NAME + ".long")
    trees = _uct_variants(PILE, 1000, UCT_C, base)
    t = trees[0]
    rows = []
    for m in sorted(t.child[0]):
        u = t.child[0][m]
        counts = [tr.N[tr.child[0][m]] for tr in trees]
        wins = [tr.W[tr.child[0][m]] for tr in trees]
        rows.append([m, PILE - m, "wins" if not _wins(PILE - m) else "loses", _int_stable(counts), _int_stable(wins),
                     "unstable" if len(set(counts)) > 1 or len(set(wins)) > 1 else _fmt(wins[0] / counts[0], 3)])
    chosen = [_best_root_move(tr) for tr in trees]
    return "\n".join([
        f"The root of the same search after 1000 iterations from n = {PILE}, c = sqrt(2), a generator started from "
        f"{mdps._seed_text(UCT_NAME + '.long')} (the name `{UCT_NAME}.long`; not the continuation of the run above). "
        "'for the mover' says whether the move wins with best play (the table of the game); W / N is the share of visits "
        "whose playout the mover won. Counts follow the nudge rule (c changed by each nudge); a count is printed only if all "
        "five runs agree.",
        "",
        _table(["move", "stones left", "for the mover", "N", "W", "W / N"], rows, "rrlrrr"),
        "",
        f"The move chosen (most visits, a tie to the smaller move) is {_int_stable(chosen)}; the winning move is {_winning_moves(PILE)[0]}.",
    ])


BUDGETS = (10, 25, 50, 100, 200, 400, 800)
UCT_RUNS = 100
UCT_CONSTANTS = (0.1, 0.5, UCT_C, 4.0)
UCT_BUDGET_NAME = "search_planning.uct_budget"


def _uct_budget_counts(shift):
    table = []
    for i, c in enumerate(UCT_CONSTANTS):
        row = []
        for j, b in enumerate(BUDGETS):
            ok = 0
            for r in range(UCT_RUNS):
                rng = bookrng.generator(f"{UCT_BUDGET_NAME}.{i}.{j}.{r}")
                t = _uct(PILE, b, c + shift, rng)
                ok += _best_root_move(t) == _winning_moves(PILE)[0]
            row.append(ok)
        table.append(row)
    return table


@lru_cache(maxsize=None)
def _budget_tables():
    return [_uct_budget_counts(nu) for nu in VARIANTS]


def uct_budget():
    tables = _budget_tables()
    rows = []
    for i, c in enumerate(UCT_CONSTANTS):
        rows.append([f"{c:g}" if c != UCT_C else "sqrt(2)"] + [_int_stable([t[i][j] for t in tables]) for j in range(len(BUDGETS))])
    return "\n".join([
        f"How often search from n = {PILE} picks the winning move 2: {UCT_RUNS} independent searches for every exploration "
        "constant and budget, the root move chosen by most visits (a tie to the smaller move). The row of a constant is "
        "its position i = 0 .. 3 in the table, the column of a budget its position j = 0 .. 6, and search number "
        f"r = 0 .. {UCT_RUNS - 1} of that cell draws its playouts, as in the trace, from the book's generator whose seed "
        f"is seed_of of the text `{UCT_BUDGET_NAME}.i.j.r` with the three numbers written in decimal (for instance "
        f"`{UCT_BUDGET_NAME}.0.0.0` has {mdps._seed_text(UCT_BUDGET_NAME + '.0.0.0')}); every search restarts its own "
        "generator. Each entry is a count out of 100; counts follow the nudge rule (c changed by each nudge).",
        "",
        _table(["c \\ iterations"] + [str(b) for b in BUDGETS], rows),
    ])


# ------------------------------------------------------------------ search guided by a policy, and the loop that trains it

NP = 13                                  # piles 1 .. 13
EI_BUDGET, EI_CPUCT, EI_STEPS, EI_LR, EI_ROUNDS, EI_EVALS = 60, 1.5, 50, 1.0, 5, 20
EI_NAME = "search_planning.expert_iteration"


def _softmax(z):
    m = max(z)
    e = [math.exp(x - m) for x in z]
    s = sum(e)
    return [x / s for x in e]


def _priors(theta, pile):
    return _softmax(theta[pile][:min(GAME_MAX, pile)])


def _sample_index(p, rng):
    u = rng.uniform()
    acc = 0.0
    for i, x in enumerate(p):
        acc += x
        if u < acc:
            return i
    return len(p) - 1


def _rollout_policy(theta, pile, depth, rng):
    while pile > 0:
        m = 1 + _sample_index(_priors(theta, pile), rng)
        pile -= m
        depth += 1
    return (depth - 1) % 2


def _puct(theta, n0, iters, cpuct, rng):
    """Search that is guided by a policy: every node's children are created the first time the node is selected from, with
    the policy's probabilities as priors; the child maximising q + cpuct * P * sqrt(max(N_parent, 1)) / (1 + N_child),
    q = W / N (0 when N = 0), is followed (a tie to the smaller move), and a child never visited before is the leaf: it is
    evaluated by a playout in which both players draw moves from the policy."""
    t = _Tree(n0)
    for _ in range(iters):
        v = 0
        while True:
            if t.pile[v] == 0:
                break
            if t.N[v] == 0 and v != 0:
                break
            if not t.child[v]:
                for m in _legal(t.pile[v]):
                    t.new(v, m)
                t.prior[v] = _priors(theta, t.pile[v])
            sq = math.sqrt(max(t.N[v], 1))
            best, best_score = -1, -math.inf
            for i, m in enumerate(_legal(t.pile[v])):
                u = t.child[v][m]
                q = t.W[u] / t.N[u] if t.N[u] > 0 else 0.0
                score = q + cpuct * t.prior[v][i] * sq / (1 + t.N[u])
                if score > best_score:
                    best, best_score = u, score
            v = best
        winner = _rollout_policy(theta, t.pile[v], t.depth[v], rng)
        w = v
        while True:
            t.N[w] += 1
            if t.depth[w] >= 1 and (t.depth[w] - 1) % 2 == winner:
                t.W[w] += 1
            if t.parent[w] < 0:
                break
            w = t.parent[w]
    return t


def _expert_iteration(nudge):
    cpuct = EI_CPUCT + nudge
    theta = [[0.0] * GAME_MAX for _ in range(NP + 1)]
    winning = [n for n in range(1, NP + 1) if _winning_moves(n)]
    history = []
    for rnd in range(EI_ROUNDS + 1):
        policy_ok = 0
        for n in winning:
            p = _priors(theta, n)
            arg = max(range(len(p)), key=lambda i: (p[i], -i)) + 1
            policy_ok += arg in _winning_moves(n)
        search_ok = 0
        for n in winning:
            for r in range(EI_EVALS):
                rng = bookrng.generator(f"{EI_NAME}.{rnd}.eval.{n}.{r}")
                t = _puct(theta, n, EI_BUDGET, cpuct, rng)
                best = max(sorted(t.child[0]), key=lambda m: (t.N[t.child[0][m]], -m))
                search_ok += best in _winning_moves(n)
        history.append((rnd, policy_ok, search_ok))
        if rnd == EI_ROUNDS:
            break
        target = {}
        for n in range(1, NP + 1):
            rng = bookrng.generator(f"{EI_NAME}.{rnd}.target.{n}")
            t = _puct(theta, n, EI_BUDGET, cpuct, rng)
            visits = [t.N[t.child[0][m]] for m in _legal(n)]
            total = sum(visits)
            target[n] = [x / total for x in visits]
        for _ in range(EI_STEPS):
            grad = {n: [p - q for p, q in zip(_priors(theta, n), target[n])] for n in range(1, NP + 1)}
            for n in range(1, NP + 1):
                for i in range(len(grad[n])):
                    theta[n][i] -= EI_LR * grad[n][i]
    return history, len(winning)


@lru_cache(maxsize=None)
def _ei_runs():
    return [_expert_iteration(nu) for nu in VARIANTS]


def expert_iteration():
    runs = _ei_runs()
    nwin = runs[0][1]
    rows = []
    for rnd in range(EI_ROUNDS + 1):
        rows.append([rnd,
                     f"{_int_stable([r[0][rnd][1] for r in runs])} of {nwin}",
                     f"{_int_stable([r[0][rnd][2] for r in runs])} of {nwin * EI_EVALS}"])
    sims = NP * EI_BUDGET
    return "\n".join([
        f"Expert iteration on the take-away game, piles 1 to {NP}. The policy is a table of preferences theta[n][i] for the "
        f"legal moves i + 1 of every pile n (all 0 at the start, so the policy is uniform), with softmax probabilities. "
        f"One round: (1) for every pile n = 1 .. {NP} in order, one PUCT search of {EI_BUDGET} iterations with c_puct = "
        f"{EI_CPUCT} whose priors and playout moves come from the current policy, and the target for the pile is the "
        f"root's visit counts divided by their sum; (2) {EI_STEPS} full-batch gradient steps of rate {EI_LR} on the "
        "cross-entropy to those targets, theta[n][i] <- theta[n][i] - rate (policy[n][i] - target[n][i]), all piles "
        "updated together from the same current policy. Before every round the policy is evaluated on the "
        f"{nwin} piles that are won for the player to move (n not a multiple of 4): 'policy alone' counts the piles where "
        f"the policy's largest probability (a tie to the smaller move) is a winning move; 'search with the policy' counts, "
        f"over {EI_EVALS} searches of {EI_BUDGET} iterations per pile in order of n, how many choose a winning move by "
        f"most visits (a tie to the smaller move). Every search has a generator of its own, "
        f"the book's generator whose seed is seed_of of the text `{EI_NAME}.R.eval.n.r` for search r = 0 .. {EI_EVALS - 1} "
        f"of the evaluation of pile n in round R, and of `{EI_NAME}.R.target.n` for the target search of pile n in round R, "
        f"all numbers in decimal (for instance `{EI_NAME}.0.target.1` has {mdps._seed_text(EI_NAME + '.0.target.1')}). "
        "Counts follow the nudge rule (c_puct changed by each nudge).",
        "",
        _table(["rounds of training", "policy alone", "search with the policy"], rows, "rrr"),
        "",
        f"A round costs {NP} searches of {EI_BUDGET} iterations, {sims} iterations in all (each one playout); "
        "the policy alone costs no iteration at decision time.",
    ])


# ------------------------------------------------------------------ quality-diversity search

QD_PEAK_A, QD_PEAK_B = (-0.55, -0.55), (0.8, 0.8)
QD_HEIGHT_A, QD_SD_A, QD_HEIGHT_B, QD_SD_B, QD_EFFORT = 0.7, 0.45, 1.0, 0.12, 0.05
QD_BUDGET, QD_K, QD_INIT, QD_RUNS, QD_SIGMA, QD_FOUND = 2000, 10, 100, 20, 0.1, 0.9
QD_NAME = "search_planning.quality_diversity"


def _behaviour(g):
    """Two steps of a walker: step 1 = (g0 - 0.5, g1 - 0.5), step 2 = (g2 - 0.5, g3 - 0.5); p is their sum."""
    s1 = (g[0] - 0.5, g[1] - 0.5)
    s2 = (g[2] - 0.5, g[3] - 0.5)
    return (s1[0] + s2[0], s1[1] + s2[1]), s1, s2


def _landscape(p):
    a = QD_HEIGHT_A * math.exp(-((p[0] - QD_PEAK_A[0]) ** 2 + (p[1] - QD_PEAK_A[1]) ** 2) / (2 * QD_SD_A ** 2))
    b = QD_HEIGHT_B * math.exp(-((p[0] - QD_PEAK_B[0]) ** 2 + (p[1] - QD_PEAK_B[1]) ** 2) / (2 * QD_SD_B ** 2))
    return max(a, b)


def _fitness(g):
    p, s1, s2 = _behaviour(g)
    return _landscape(p) - QD_EFFORT * (math.hypot(*s1) + math.hypot(*s2))


def _cell(g, k=QD_K):
    p, _, _ = _behaviour(g)
    d = ((p[0] + 1.0) / 2.0, (p[1] + 1.0) / 2.0)
    return tuple(min(int(math.floor(x * k)), k - 1) for x in d)


def _mutate(g, sigma, rng):
    return [min(1.0, max(0.0, x + sigma * rng.normal())) for x in g]


def _random_genome(rng):
    return [rng.uniform() for _ in range(4)]


def _hill_climb(rng, sigma):
    g = _random_genome(rng)
    f = _fitness(g)
    for _ in range(QD_BUDGET - 1):
        h = _mutate(g, sigma, rng)
        fh = _fitness(h)
        if fh > f:
            g, f = h, fh
    return f, 1


def _best_n(rng, sigma):
    pop = []
    for _ in range(QD_INIT):
        g = _random_genome(rng)
        pop.append((_fitness(g), g))
    for _ in range(QD_BUDGET - QD_INIT):
        parent = pop[rng.integer(len(pop))][1]
        child = _mutate(parent, sigma, rng)
        fc = _fitness(child)
        worst = min(range(len(pop)), key=lambda i: (pop[i][0], i))
        if fc > pop[worst][0]:
            pop[worst] = (fc, child)
    return max(f for f, _ in pop), len({_cell(g) for _, g in pop})


def _map_elites(rng, sigma):
    arch = {}

    def offer(g):
        f = _fitness(g)
        c = _cell(g)
        if c not in arch or f > arch[c][0]:
            arch[c] = (f, g)

    for _ in range(QD_INIT):
        offer(_random_genome(rng))
    for _ in range(QD_BUDGET - QD_INIT):
        keys = sorted(arch)
        parent = arch[keys[rng.integer(len(keys))]][1]
        offer(_mutate(parent, sigma, rng))
    return arch


def _qd_experiment(shift):
    rng = bookrng.generator(QD_NAME)
    sigma = QD_SIGMA + shift
    hill = [_hill_climb(rng, sigma) for _ in range(QD_RUNS)]
    best = [_best_n(rng, sigma) for _ in range(QD_RUNS)]
    maps = [_map_elites(rng, sigma) for _ in range(QD_RUNS)]
    return hill, best, maps


@lru_cache(maxsize=None)
def _qd_runs():
    return [_qd_experiment(nu) for nu in VARIANTS]


def _stable_floats(vals, nd=3):
    return stable.text(vals[0], vals[1:], lambda x: f"{x:.{nd}f}")


def quality_diversity_results():
    runs = _qd_runs()

    def summarize(select):
        found = [sum(1 for f, _ in select(r) if f > QD_FOUND) for r in runs]
        best = [max(f for f, _ in select(r)) for r in runs]
        worst = [min(f for f, _ in select(r)) for r in runs]
        cover = [sum(c for _, c in select(r)) / QD_RUNS for r in runs]
        return found, best, worst, cover

    hill = summarize(lambda r: r[0])
    bestn = summarize(lambda r: r[1])
    me_sel = lambda r: [(max(f for f, _ in a.values()), len(a)) for a in r[2]]
    me = summarize(me_sel)
    rows = []
    for name, (found, best, worst, cover), cover_text in (
            ("hill climber (1 + 1)", hill, None), ("best-100 population", bestn, None), ("MAP-Elites, 10 x 10 cells", me, None)):
        rows.append([name, f"{_int_stable(found)} of {QD_RUNS}", _stable_floats(best), _stable_floats(worst),
                     _stable_floats(cover, 1)])
    return "\n".join([
        "A deceptive landscape and three searches with the same budget. A genome is four numbers g0 .. g3 in [0, 1]. A "
        "walker takes two steps, s1 = (g0 - 0.5, g1 - 0.5) and s2 = (g2 - 0.5, g3 - 0.5), and ends at p = s1 + s2, a "
        "point of the square [-1, 1]^2. Fitness is F(p) minus 0.05 (|s1| + |s2|) with Euclidean lengths, where F(p) is "
        "the larger of a broad low peak A(p) = 0.7 exp(-|p - (-0.55, -0.55)|^2 / (2 x 0.45^2)) and a narrow high peak "
        "B(p) = 1.0 exp(-|p - (0.8, 0.8)|^2 / (2 x 0.12^2)). The behaviour descriptor of a genome is its end point mapped "
        "to the unit square, d = ((p_x + 1) / 2, (p_y + 1) / 2); a 10 x 10 grid cuts it into cells, a coordinate d in "
        "[0, 1] falling in cell min(floor(10 d), 9) (lower edge inclusive, upper edge exclusive, d = 1 in the last "
        "cell). Every search makes 2000 fitness evaluations, one for every genome ever produced. Mutation: every gene "
        "g <- clip(g + 0.1 normal(), 0, 1), genes in order 0 .. 3, one normal() each, from the book's generator. "
        "Random genomes: four uniform() values in order. Hill climber: one random genome, then 1999 mutants of the "
        "current genome, each kept only if its fitness is strictly higher. Best-100: 100 random genomes, then 1900 "
        "times: a parent drawn by integer(len) from the list in position order, a mutant, which replaces the member of "
        "lowest fitness (a tie to the lowest position) if its fitness is strictly higher. MAP-Elites: 100 random genomes "
        "are offered to the archive, then 1900 times: a cell drawn by integer(number of occupied cells) from the "
        "occupied cells in ascending (column-of-p_x, column-of-p_y) order, its elite mutated and offered; an offer "
        "fills an empty cell and replaces an elite only if its fitness is strictly higher. "
        f"One generator, `{QD_NAME}` {mdps._seed_text(QD_NAME)}, serves all 60 runs: 20 hill climbs, then 20 "
        "best-100 runs, then 20 MAP-Elites runs, each run continuing the stream. 'found B' counts runs whose best "
        "fitness exceeds 0.9 (the narrow peak). 'cells' is the mean number of distinct cells the final population "
        "occupies (1 for the single genome of the hill climber). Digits follow the nudge rule (mutation width changed "
        "by each nudge).",
        "",
        _table(["search", "found B", "best fitness, highest run", "best fitness, lowest run", "cells"], rows, "lrrrr"),
    ])


def archive_of_first_run():
    """The archive of the first MAP-Elites run of `quality_diversity_results`: dict (column, row) -> fitness."""
    return {c: f for c, (f, _) in _qd_runs()[0][2][0].items()}


def quality_diversity_archive():
    arch = archive_of_first_run()
    empty = ", ".join(f"({cx}, {cy})" for cy in range(QD_K) for cx in range(QD_K) if (cx, cy) not in arch) or "none"
    header = ["y cell \\ x cell"] + [str(i) for i in range(QD_K)]
    rows = []
    for cy in range(QD_K - 1, -1, -1):
        rows.append([str(cy)] + [(f"{arch[(cx, cy)]:.2f}" if (cx, cy) in arch else "") for cx in range(QD_K)])
    return "\n".join([
        f"The archive of the first MAP-Elites run (the first of the 20 that follow the 40 runs before it in the stream): the "
        f"fitness of the elite of each of the 10 x 10 cells, two decimals, with the horizontal axis the cell of the "
        f"descriptor's first coordinate and the vertical axis that of its second (cell 9 on the top row). An empty cell "
        f"is blank. {len(arch)} of 100 cells are occupied; empty cells: {empty}. The narrow peak B = (0.8, 0.8) has "
        f"the descriptor (0.9, 0.9), the corner where the cells (8, 8), (9, 8), (8, 9) and (9, 9) meet.",
        "",
        _table(header, rows),
    ])


# ------------------------------------------------------------------ goals and hindsight

HER_N, HER_H, HER_EPISODES, HER_SWEEPS, HER_K = 8, 12, 300, 30, 4
HER_ALPHA, HER_GAMMA = 0.5, 0.9
HER_NAME = "search_planning.hindsight"
HER_MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _her_step(s, a):
    r, c = divmod(s, HER_N)
    dr, dc = HER_MOVES[a]
    r = min(HER_N - 1, max(0, r + dr))
    c = min(HER_N - 1, max(0, c + dc))
    return r * HER_N + c


def _her_episodes(rng):
    eps = []
    cells = HER_N * HER_N
    for _ in range(HER_EPISODES):
        s0 = rng.integer(cells)
        g = rng.integer(cells - 1)
        if g >= s0:
            g += 1
        traj, s = [s0], s0
        acts = []
        for _ in range(HER_H):
            a = rng.integer(4)
            s = _her_step(s, a)
            acts.append(a)
            traj.append(s)
        eps.append((g, traj, acts))
    return eps


def _her_transitions(eps, strategy, rng):
    out = []
    for g, traj, acts in eps:
        for t in range(HER_H):
            out.append((traj[t], acts[t], traj[t + 1], g))
            if traj[t + 1] == g:
                break
        if strategy == "final":
            for t in range(HER_H):
                out.append((traj[t], acts[t], traj[t + 1], traj[HER_H]))
        elif strategy == "future":
            for t in range(HER_H):
                for _ in range(HER_K):
                    k = t + 1 + rng.integer(HER_H - t)
                    out.append((traj[t], acts[t], traj[t + 1], traj[k]))
    return out


def _her_train(transitions):
    q = {}
    for _ in range(HER_SWEEPS):
        for s, a, s2, g in transitions:
            y = 1.0 if s2 == g else HER_GAMMA * max(q.get((s2, b, g), 0.0) for b in range(4))
            old = q.get((s, a, g), 0.0)
            q[(s, a, g)] = old + HER_ALPHA * (y - old)
    return q


def _her_success(q):
    cells = HER_N * HER_N
    ok = total = 0
    for s0 in range(cells):
        for g in range(cells):
            if s0 == g:
                continue
            total += 1
            s = s0
            for _ in range(2 * HER_H):
                vals = [q.get((s, b, g), 0.0) for b in range(4)]
                a = max(range(4), key=lambda b: (vals[b], -b))
                s = _her_step(s, a)
                if s == g:
                    ok += 1
                    break
    return ok, total


def hindsight_relabelling():
    rng = bookrng.generator(HER_NAME)
    eps = _her_episodes(rng)
    rows = []
    for name, strategy in (("original goals only", "none"), ("final state", "final"), ("future states, k = 4", "future")):
        tr = _her_transitions(eps, strategy, rng)
        pos = sum(1 for t in tr if t[2] == t[3])
        ok, total = _her_success(_her_train(tr))
        rows.append([name, len(tr), pos, f"{100.0 * pos / len(tr):.1f}%", f"{ok} of {total}", f"{100.0 * ok / total:.1f}%"])
    empty_ok, total = _her_success({})
    return "\n".join([
        f"A goal-reaching task on an {HER_N} x {HER_N} grid without walls, cells numbered 8 r + c, four moves (up, right, "
        "down, left; a move into the border leaves the agent where it is) and a reward of 1 for the move after which the "
        "agent stands on the goal, 0 otherwise. Data: "
        f"{HER_EPISODES} episodes by a random behaviour, from the book's generator `{HER_NAME}` {mdps._seed_text(HER_NAME)}. "
        "Per episode, in order: the start by integer(64), the goal by integer(63) with 1 added if it is at least the start "
        f"(so uniform over the other cells), then {HER_H} moves by integer(4) each. The episode is stored for its "
        "original goal up to and including the first move that reaches it (the rest of the walk is still drawn). "
        "'final state' adds, for every one of the 12 moves of the walk, the transition relabelled with the goal that the "
        "walk ended on; 'future states' adds, for every move t = 0 .. 11 in order, 4 transitions relabelled with the state "
        "after move k, where k = t + 1 + integer(12 - t) is drawn four times in a row for each t. The draws of the episodes "
        "come first (all 300 episodes), the draws of the relabelling after them, from the same generator and in the order of "
        "the three rows of the table (only the third row draws). A transition is (state, move, next state, goal) and its "
        f"reward is 1 when the next state is the goal. Learning: tabular Q(state, move, goal), start 0, {HER_SWEEPS} sweeps "
        f"over the stored transitions in stored order, target 1 when the next state is the goal and otherwise "
        f"{HER_GAMMA} times the largest Q of the next state for the same goal, step {HER_ALPHA}. Evaluation: for every "
        f"start-goal pair of different cells ({total}), follow the move of largest Q (a tie to the smaller move) for up "
        f"to {2 * HER_H} moves; success if the goal is reached. A table that is all zero (always the first move, up) "
        f"succeeds on {empty_ok} of {total} pairs.",
        "",
        _table(["relabelling", "transitions", "with reward 1", "share", "pairs solved", "share"], rows, "lrrrrr"),
    ])


# ------------------------------------------------------------------ brain: the archive and the bandit of crates/search

FOCUS, EDGE_WEIGHT = 4.0, 0.6
ARCHIVE_TOL = 0.01


def _selection_weight(reached, top, visits, focus):
    short = 1.0 - min(1.0, max(0.0, reached / top))
    return math.exp(-focus * short) / math.sqrt(visits + 1.0)


def archive_selection_weight():
    fractions = (1.0, 0.75, 0.5, 0.25, 0.0)
    rows = []
    for x in fractions:
        old = (1.0 + x) / 2.0
        new = _selection_weight(x, 1.0, 0, FOCUS)
        flat = _selection_weight(x, 1.0, 0, 1.0)
        rows.append([f"{x:.2f}", _fmt(old, 3), _fmt(flat, 3), _fmt(new, 4), f"1 in {1.0 / new:.1f}"])
    visits = [[v, _fmt(1.0 / math.sqrt(v + 1.0), 3)] for v in (0, 3, 8, 15, 99)]
    edge = [[f"{lo:.2f}", _fmt(1.0 + EDGE_WEIGHT * lo, 3)] for lo in (0.0, 0.25, 0.5, 1.0)]
    return "\n".join([
        "The weight with which `Archive::pick` draws a cell: exp(-focus (1 - reached / top)) / sqrt(visits + 1), the "
        "shortfall clamped into [0, 1], times (1 + 0.6 loneliness) when edge axes are named. First, the achievement factor "
        "for a cell with no visits, as a fraction of the weight of the best cell (reached / top = 1): the rule before "
        "commit b02e79a35, (1 + reached / top) / 2; the new rule at focus 1; the new rule at the default focus 4.0; and "
        "the last as the chance, among equals, of being picked rather than the best cell.",
        "",
        _table(["reached / top", "old rule", "focus 1", "focus 4.0", "against the best cell, focus 4.0"], rows),
        "",
        "Second, the visit factor 1 / sqrt(visits + 1) (it multiplies the first):",
        "",
        _table(["visits", "factor"], visits),
        "",
        "Third, the edge factor 1 + 0.6 x loneliness, where loneliness is the share of the cell's neighbours along the "
        "named axes (one step each way) that the archive does not hold:",
        "",
        _table(["loneliness", "factor"], edge),
    ])


def _better_than(a, b, tol):
    """Worth.better_than: achievements within tol are equal and the cost decides; the cost must be strictly lower."""
    if a[0] > b[0] + tol:
        return True
    if b[0] > a[0] + tol:
        return False
    return a[1] < b[1]


def archive_offers():
    held, gen, rows = (10.0, 500), 0, []
    rows.append(["(held)", "-", f"reached {held[0]}, cost {held[1]}", gen])
    for offer in ((9.9, 100), (10.0, 500), (10.005, 400), (10.02, 800), (10.0, 300), (11.0, 900)):
        if _better_than(offer, held, ARCHIVE_TOL):
            held, gen, verdict = offer, gen + 1, "Improved"
        else:
            verdict = "Rejected"
        rows.append([f"reached {offer[0]}, cost {offer[1]}", verdict, f"reached {held[0]}, cost {held[1]}", gen])
    return "\n".join([
        "Offers to one niche whose elite has Worth(reached 10.0, cost 500), tolerance 0.01, in the order shown, each judged "
        "against what is held after the one before. `better_than(a, b, tol)` is true when a.reached > b.reached + tol; "
        "false when b.reached > a.reached + tol; and otherwise when a.cost < b.cost. The generation of the cell counts "
        "the replacements.",
        "",
        _table(["offer", "verdict", "held afterwards", "generation"], rows, "lllr"),
    ])


ALLOC_ARMS = (("perturb", 2.0, 1.0), ("hunt", 4.0, 4.0), ("refine", 0.5, 2.0))   # (name, value per call, cost per call)
ALLOC_ROUNDS = (3, 6, 12, 30, 100, 300)


def _allocate(rounds):
    draws = [0] * len(ALLOC_ARMS)
    secs = [0.0] * len(ALLOC_ARMS)
    value = [0.0] * len(ALLOC_ARMS)
    seq = []
    snaps = {}
    for r in range(1, rounds + 1):
        untried = [i for i, d in enumerate(draws) if d == 0]
        if untried:
            arm = untried[0]
        else:
            rates = [value[i] / max(secs[i], 1e-3) for i in range(len(draws))]
            top = max(max(rates), 5e-324)
            total = math.log(max(sum(draws), 1))
            best, best_score = 0, -math.inf
            for i in range(len(draws)):
                score = rates[i] / top + math.sqrt(2.0) * math.sqrt(total / draws[i])
                if score > best_score:
                    best, best_score = i, score
            arm = best
        draws[arm] += 1
        secs[arm] += max(ALLOC_ARMS[arm][2], 1e-3)
        value[arm] += ALLOC_ARMS[arm][1]
        seq.append(arm)
        if r in ALLOC_ROUNDS:
            snaps[r] = draws[:]
    return seq, snaps


def allocator_rounds():
    seq, snaps = _allocate(ALLOC_ROUNDS[-1])
    rows = [[r] + snaps[r] for r in ALLOC_ROUNDS]
    first = " ".join(ALLOC_ARMS[i][0] for i in seq[:12])
    spec = "; ".join(f"{n}: value {v:g} per call, cost {c:g} per call, rate {v / c:g}" for n, v, c in ALLOC_ARMS)
    return "\n".join([
        f"UCB1 over three search operators, as in `Allocator::choose`. Every call of an operator earns the same fixed value and "
        f"costs the same fixed amount (the crate measures its cost in seconds of the clock; here the units are abstract): {spec}. "
        "Rule: every arm is tried once, in order; after that the arm with the largest rate / top + sqrt(2) sqrt(ln(rounds) / "
        "draws) is chosen, where rate is the arm's total value over its total cost, top the largest rate among the arms and "
        "rounds the number of calls made so far (a tie to the earlier arm). No randomness.",
        "",
        f"The first 12 choices: {first}.",
        "",
        _table(["calls made"] + [n for n, _, _ in ALLOC_ARMS], rows),
    ])


# ------------------------------------------------------------------ brain: retracing walks (crates/solve)

PUZ_N = 5                                 # tokens in a row; the moves swap neighbours 0-1, 1-2, 2-3, 3-4
PUZ_GOAL = tuple(range(PUZ_N))
PUZ_ROWS = 2000
PUZ_DEPTHS = (2, 4, 6, 8, 10, 12, 16, 24)
PUZ_STEPS = 24
PUZ_NAME = "search_planning.retrace"


def _puz_apply(s, m):
    t = list(s)
    t[m], t[m + 1] = t[m + 1], t[m]
    return tuple(t)


def _puz_distances():
    dist = {PUZ_GOAL: 0}
    frontier = [PUZ_GOAL]
    for x in frontier:
        for m in range(PUZ_N - 1):
            y = _puz_apply(x, m)
            if y not in dist:
                dist[y] = dist[x] + 1
                frontier.append(y)
    return dist


PUZ_DIST = _puz_distances()


def _retrace_batch(depth, rows, rng):
    """The generator of `crates/solve`, on this puzzle: walks of `depth` moves from the goal, a move that undoes the one
    before it never drawn, every state of a walk labelled with the inverse (here the same swap) of the move that reached
    it, an equal quota per walk depth with the spare rows going to the deepest depths, the overflow dropped. Returns the
    examples and the number of moves applied."""
    quota = [rows // depth] * depth
    for i in range(rows % depth):
        quota[depth - 1 - i] += 1
    filled = [0] * depth
    out, applied = [], 0
    while len(out) < rows:
        s, last = PUZ_GOAL, None
        for step in range(depth):
            while True:
                m = rng.integer(PUZ_N - 1)
                if m != last:
                    break
            s = _puz_apply(s, m)
            applied += 1
            last = m
            if filled[step] == quota[step]:
                continue
            out.append((s, m, step + 1))
            filled[step] += 1
    return out, applied


def retraced_labels():
    rows = []
    for depth in PUZ_DEPTHS:
        rng = bookrng.generator(PUZ_NAME)
        batch, applied = _retrace_batch(depth, PUZ_ROWS, rng)
        votes = {}
        for s, label, _ in batch:
            votes.setdefault(s, [0] * (PUZ_N - 1))[label] += 1
        policy = {s: max(range(PUZ_N - 1), key=lambda m: (v[m], -m)) for s, v in votes.items()}
        labelled = [s for s in policy if s != PUZ_GOAL]
        shortest = sum(1 for s in labelled if PUZ_DIST[_puz_apply(s, policy[s])] == PUZ_DIST[s] - 1)
        solved, length, optimal = 0, 0, 0
        for s0 in PUZ_DIST:
            if s0 == PUZ_GOAL:
                continue
            s = s0
            for step in range(1, PUZ_STEPS + 1):
                if s not in policy:
                    break
                s = _puz_apply(s, policy[s])
                if s == PUZ_GOAL:
                    solved += 1
                    length += step
                    optimal += PUZ_DIST[s0]
                    break
        rows.append([depth, f"{applied / PUZ_ROWS:.2f}", f"{(depth + 1) / 2:.1f}", len(votes),
                     f"{shortest} of {len(labelled)}", solved, f"{length / solved:.2f}" if solved else "-",
                     f"{optimal / solved:.2f}" if solved else "-"])
    diam = max(PUZ_DIST.values())
    return "\n".join([
        f"Retraced labels on a permutation puzzle: {PUZ_N} tokens in a row, goal (0, 1, 2, 3, 4), four moves, move m "
        f"swaps the tokens at positions m and m + 1 (every move is its own inverse, so the label of a state reached by "
        "move m is m). The puzzle has "
        f"{len(PUZ_DIST)} states and diameter {diam} (breadth-first search from the goal). The generator, as in `crates/solve`: "
        "a walk of D moves from the goal in which a move equal to the one just played is redrawn (`rng.integer(4)` until "
        f"it differs), every state of the walk labelled with the move that undoes the last move, each walk depth 1 .. D "
        f"receiving its quota of {PUZ_ROWS} // D rows with the remainder added to the deepest depths, overflow dropped. "
        f"{PUZ_ROWS} rows per depth, every depth restarting the book's generator `{PUZ_NAME}` {mdps._seed_text(PUZ_NAME)}. "
        "The policy is the table: for every labelled state the label that most rows give it (a tie to the smaller move). "
        "'moves per example' is the number of moves applied to generate the rows divided by the rows, against the (D + 1) / 2 "
        "of a generator that keeps one endpoint per walk of a uniformly drawn length 1 .. D; 'states labelled' is the number "
        "of distinct states in the rows; 'shortest labels' counts the labelled non-goal states whose chosen move reaches a "
        f"state one move nearer the goal; 'solved' counts the {len(PUZ_DIST) - 1} non-goal states from which following the "
        f"table reaches the goal in at most {PUZ_STEPS} moves (a state with no label stops the run), with the mean "
        "number of moves of those runs and the mean shortest distance of the same states.",
        "",
        _table(["D", "moves per example", "endpoint-only generator", "states labelled", "shortest labels", "solved",
                "moves of a run", "shortest"], rows),
    ])
