<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Search, Planning and Discovery

*When should an agent search instead of learn, and how do the two help each other?*

Take away the table of values that the first chapter spent so long computing and leave the keeper with nothing but the rules of the shop: the demand, the prices, the shelf that holds three units. She can still decide what to order on an empty morning. She imagines each order, imagines each demand that could follow, imagines the order she would place the next day, and so on for as many days as she has patience, scoring the end of every imagined future with a guess of what a shelf is worth. With a guess of zero for every shelf, a one-day look at the empty shelf scores the orders $0, 1, 2, 3$ as $0.000$, $0.000$, $-1.500$ and $-4.500$, a tie between ordering nothing and ordering one unit that the smallest-index rule resolves to nothing. Look two days ahead and the scores become $0.000$, $0.675$, $0.210$ and $-2.205$: one unit wins, and it is the order the exact solution of Chapter 1 makes. Two days of lookahead repaired what a table of values would have said at once.

The price of that repair is the part worth counting. Every imagined day branches over the orders that fit on the shelf and the demands that could follow, and the tree of two days from the empty shelf has $76$ nodes, of three days $542$, of eight days $9\,132\,652$. This is **search**, and the number of nodes is its cost. The table of values that Chapter 1 built is what search would leave behind if it kept every answer, and the tree is what a decision costs when it does not. Learning, the subject of the three chapters before this one, pays once and answers cheaply afterwards, but it has to be fed experience and it can be wrong in states it has not seen. Search pays at every decision, needs a model of what the world does (the rules of the shop, a simulator, a program that can check an answer), and is as exact as the model and the patience allow. Which to use, and how each can make the other cheaper, is the subject of the chapter.

Four ideas from earlier are the tools. Answer them from memory; the answers are at the end of the chapter.

1. (Volume XII, Chapter 1, "Markov Decision Processes", the lookahead $q(s, a)$.) With every value taken as $0$, what are $q(0, a)$ for $a = 0, 1, 2, 3$ on the empty shelf of the shop? Which action does the tie rule of that chapter choose?
2. (Volume XII, Chapter 1, the policy.) What is a policy, and how many deterministic policies does the shop have?
3. (Volume I, Chapter 10, "Probability: Reasoning Under Uncertainty", random variable and variance.) A random playout is won with probability $p$. Call $X$ the random variable that is $1$ when it is won and $0$ otherwise. What are the mean and the variance of $X$, and how many playouts does it take to halve the standard error of the average of the $X$ values?
4. (Volume III, Chapter 2, "Multilayer Networks", the multilayer perceptron.) How many parameters has a network with $2$ inputs, one hidden layer of $12$ tanh units and one linear output?

## What the lookahead of the shop costs

Here is the lookahead as a table. A lookahead of $d$ days is a tree whose root is the state, whose children are the pairs of an order and a next stock that can follow, and whose leaves lie $d$ days down. An inner node takes the best order of $R(s, a)$ plus $\gamma$ times the probability-weighted mean of its children, which is the one-step lookahead $q$ of Chapter 1 applied to the values of the children, and a leaf takes a value from a given vector.

<!-- worked: search_planning.shop_lookahead -->
The shop of Chapter 1 (tables as printed there) at gamma = 0.9, searched from the empty shelf (stock 0). A lookahead of d days is the tree whose root is the state, whose children are the pairs (order a, next stock j) with P(j | s, a) > 0 and whose leaves lie d days down; a leaf is valued by a given vector (all zeros, or the exact V* of Chapter 1) and an inner node by the best order of R(s, a) + gamma times the probability-weighted mean of its children. 'Nodes' counts every state node of the tree without sharing, root and leaves included; 'values with sharing' is the number of (stock, days-left) pairs that need a value when equal subtrees are computed once, 4 stocks times d. The columns Q_d(0, a) are the lookahead values of the orders a = 0, 1, 2, 3 at the root with zero leaves; 'best a' is the smallest order within 1e-9 of the largest value (the tie rule of Chapter 1).

| d | nodes | values with sharing | Q_d(0, a) with zero leaves | best a, zero leaves | best a, leaves V* |
|---|---:|---:|---:|---:|---:|
| 1 | 10 | 4 | 0.000, 0.000, -1.500, -4.500 | 0 | 1 |
| 2 | 76 | 8 | 0.000, 0.675, 0.210, -2.205 | 1 | 1 |
| 3 | 542 | 12 | 0.608, 1.282, 1.097, -0.788 | 1 | 1 |
| 4 | 3808 | 16 | 1.154, 1.829, 1.719, 0.079 | 1 | 1 |
| 5 | 26665 | 20 | 1.646, 2.321, 2.232, 0.684 | 1 | 1 |
| 6 | 186584 | 24 | 2.089, 2.764, 2.680, 1.165 | 1 | 1 |
| 7 | 1305396 | 28 | 2.488, 3.163, 3.080, 1.576 | 1 | 1 |
| 8 | 9132652 | 32 | 2.846, 3.521, 3.439, 1.938 | 1 | 1 |

The root values with sharing are the iterates of value iteration from V_0 = 0: the largest difference between the 8-day lookahead values of the four stocks and the vector after 8 sweeps of Chapter 1 is 0.0e+00 (below 1e-12).
<!-- /worked -->

Two things in the table matter for what follows. The first is the last paragraph under it: a lookahead whose equal subtrees are computed once is value iteration. The $d$-day values of the four stocks, started from zero leaves, are the vector after $d$ sweeps of Chapter 1, to the last bit. A tree search that remembers what it has already computed is dynamic programming, and the tree is only the unshared way of writing the same sum. What the shop has is four stocks, so sharing is nearly free: $4d$ values against $9\,132\,652$ nodes at $d = 8$. A problem whose states are not four but astronomically many offers no such saving, because most states never recur inside the tree and cannot be tabulated beforehand either, and that is where search earns its name.

The second is the last column. With the exact values $V^\ast$ at the leaves, one day of lookahead already picks the best order. The more the leaves know, the less tree is needed, and the less the leaves know, the more of the tree has to do the work. Everything in this chapter is a way of trading between the two: a better guess at the leaves (a heuristic, a learned value, a playout), a smaller tree (a priority, a prior), or a way to keep what a search found so that nobody has to look again (a policy fitted to it, an archive that stores it).

## Four ways to order a frontier on a map with mud

The shop has a handful of states. Search is at home in a space where the states are the cells of a map or the arrangements of a puzzle, each reached by a move from another, and the question is a path from a start to a goal. A **state-space search** keeps a **frontier** of states that have been reached but not examined and repeatedly takes one out, examines it, which is called **expanding** it, and puts the states one move from it into the frontier. All of the classical methods are this loop. They differ only in which state leaves the frontier first.

The map of this section has mud, so that the shortest path in moves and the cheapest path in cost are different paths.

<!-- worked: search_planning.grid_map -->
The map, 7 rows by 12 columns, rows numbered from 0 at the top and columns from 0 at the left, a cell written (row, column). S is the start, (0, 4); G is the goal, (6, 10); '.' is open ground and S and G are open ground too; '~' is mud; '#' is a wall. There are 79 cells that can be entered (30 of them mud) and 5 walls. Moving onto open ground costs 1 and onto mud costs 3; a move goes one cell up, right, down or left and never onto a wall or off the map.

```text
....S.......
............
~~~.~~~~~~~#
~~~.~~~~~~~#
~~~.~~~~~~~.
......###...
..........G.
```
<!-- /worked -->

Moves cost $1$ on open ground and $3$ onto mud. The cheapest path from $S$ to $G$ is the one a keeper would find by looking at the picture: round the mud by the open column at column $3$ and along the bottom, $14$ moves for a cost of $14$. The straight way across the mud band is $12$ moves but costs $18$, since three of its cells are mud. The four methods below see this map only through their priority rule.

Write $g(x)$ for the cost of the cheapest path to a state $x$ that the search has found so far (the "g" of the usual notation of the field; not the activation function of Volume III), and $h(x)$ for a **heuristic**, a guess at the cost of the rest of the way. Here $h$ is the Manhattan distance to the goal, $|{\rm row} - 6| + |{\rm column} - 10|$. The frontier is a priority queue, and the four methods are four priorities:

- **breadth-first search** takes the entry with the fewest moves, which is the priority $g$ when every move is counted as $1$;
- **uniform-cost search**, Dijkstra's method, takes the entry of lowest cost so far, the priority $g$ with the real costs;
- **greedy best-first search** takes the entry that looks nearest the goal, the priority $h$;
- **A\* search** takes the entry that looks cheapest in total, the priority $f(x) = g(x) + h(x)$.

Every convention that could change a printed number is fixed in the routine, because two readers who differ on one of them get different expansion counts: entries leave the queue in the order of their priority and, among equal priorities, in the order they entered (a counter makes the queue key a pair); the goal is tested when its entry is popped, not when it is generated; a popped cell that is already closed is skipped and does not count as an expansion; the neighbours of an expanded cell are generated in the order up, right, down, left; and a neighbour is pushed whenever this route to it is cheaper than any found so far.

```text
function best_first(start: (int, int), goal: (int, int),
        step_cost: function((int, int)) -> real,
        priority: function(real, real) -> real,
        h: function((int, int)) -> real,
        open_cell: function((int, int)) -> bool)
        -> (order: list<(int, int)>, path: list<(int, int)>, pushes: int):
    counter = 0
    best_g = map of (int, int) to real, empty
    parent = map of (int, int) to (int, int), empty
    closed = map of (int, int) to bool, empty
    queue = heap of (real, int, (int, int), real), empty
    order = empty list of (int, int)
    best_g[start] = 0.0
    push(queue, (priority(0.0, h(start)), counter, start, 0.0))
    counter = counter + 1
    while len(queue) > 0:
        (key, stamp, cell, g) = pop_min(queue)
        if cell in closed:
            continue
        if cell == goal:
            path = the cells from start to goal, found by following parent from goal
            return (order, path, counter)
        closed[cell] = true
        append cell to order
        for (dr, dc) in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            nxt = (cell[0] + dr, cell[1] + dc)
            if open_cell(nxt) and not (nxt in closed):
                g2 = g + step_cost(nxt)
                if g2 < get(best_g, nxt, infinity):
                    best_g[nxt] = g2
                    parent[nxt] = cell
                    push(queue, (priority(g2, h(nxt)), counter, nxt, g2))
                    counter = counter + 1
    return (order, empty list, counter)
```

`open_cell` is true for a cell inside the map that is not a wall. Breadth-first search is the routine with `step_cost` always $1$ and `priority(g, h) = g`; uniform-cost with `step_cost` the cost of the cell entered and the same priority; greedy with `priority(g, h) = h`; and A\* with `priority(g, h) = g + h`. The weighted form $g + w\,h$ with a number $w$ is the same routine with `priority(g, h) = g + w * h`.

<!-- worked: search_planning.search_methods -->
Seven searches of the map above, all from the routine of the chapter. The step cost is 1 for breadth-first search (it counts moves, so its g is the number of moves) and the cost of the cell entered for the others; the heuristic is the Manhattan distance h(x) = |row - 6| + |column - 10| to the goal. Entries leave the queue in the order of their priority, ties in the order they entered; neighbours are generated up, right, down, left; the goal is tested when its entry is popped; a popped cell that is already closed is skipped and does not count. 'expansions' counts the cells closed, 'pushes' the entries ever put into the queue (the start included); 'moves' and 'cost' are those of the path found, the cost with mud counted at 3.

| method | expansions | pushes | moves | cost | cost-optimal |
|---|---:|---:|---:|---:|---|
| breadth-first | 77 | 79 | 12 | 18 | no (+4) |
| uniform-cost | 72 | 76 | 14 | 14 | yes |
| A*, f = g + 0.5 h | 59 | 72 | 14 | 14 | yes |
| A*, f = g + h | 38 | 57 | 14 | 14 | yes |
| A*, f = g + 2 h | 19 | 31 | 12 | 18 | no (+4) |
| A*, f = g + 5 h | 12 | 28 | 12 | 18 | no (+4) |
| greedy, f = h | 12 | 28 | 12 | 18 | no (+4) |
<!-- /worked -->

Read the table from the top. Breadth-first search finds the path with the fewest moves, $12$, and pays $18$ for it, because it cannot see that three of those moves are expensive; it also closes $77$ of the $79$ cells, nearly the whole map. Uniform-cost search finds the cheapest path, $14$, and expands $72$ cells: it is exact and blind, and it spends its expansions in every direction, since a cell far to the left costs no more than one near the goal until the cheapest cost reaches it. A\* with the Manhattan distance returns the same cheapest path after $38$ expansions, because the heuristic pulls the frontier toward the goal. Greedy search expands $12$ cells, the length of its path, and returns the path through the mud at cost $18$: it never asked what the way had cost, only what the way still looked like.

The figure shows where the expansions went. Uniform-cost search floods the whole upper-left, including every cell of the open rows to the left of the start; A\* expands a narrow band toward the goal and leaves most of the left of the map untouched.

![Two copies of the seven by twelve map side by side. In uniform-cost search, 72 cells are numbered with their expansion order, including the whole top-left and the cells on both sides of the open column. In A* with f equal to g plus h, 38 cells are numbered, concentrated along the top rows and the open column and along the bottom row. Both outline the same path of cost 14.](img/search-expansion-order.svg)

The grids behind the figure, in text, so that a reader can compare an own implementation cell by cell:

<!-- worked: search_planning.expansion_orders -->
Each cell shows the number of the expansion that closed it, counted from 1 (so S is 1); a star marks a cell of the path that was returned; G was never expanded because the search stopped when its entry was popped. Cells never expanded show their map symbol.

Uniform-cost search:

```text
 19  13   8   4   1*  2   5   9  15  21  27  34 
 24  18  12   7*  3*  6  10  16  22  28  35  44 
 43  33  26  11* 14  20  25  31  40  50  56   # 
 62  55  32  17* 30  39  49  54  60  65  69   # 
 66  61  42  23* 41  59  64  68  71   ~   ~   . 
 53  48  38  29* 36* 45*  #   #   #  72   .   . 
 58  52  47  37  46  51* 57* 63* 67* 70*  G*  . 
```

A* with f = g + h:

```text
  .   .   .  15   1*  2   4   6   8  10  12  22 
  .   .   .  17*  3*  5   7   9  11  13  14  24 
  ~   ~   ~  26* 16  18  19  20  21  23  25   # 
  ~   ~   ~  27*  ~   ~   ~   ~   ~   ~   ~   # 
  ~   ~   ~  28*  ~   ~   ~   ~   ~   ~   ~   . 
  .   .   .  29* 30* 32*  #   #   #   .   .   . 
  .   .   .  31  33  34* 35* 36* 37* 38*  G*  . 
```
<!-- /worked -->

## What a heuristic has to promise

The A\* column is good, but the weighted rows of the table say that the promise is conditional. With $f = g + 2h$ the search expands $19$ cells and returns cost $18$; with $f = g + 5h$ it expands $12$ cells and returns the greedy path. The weight is a dial between uniform-cost search at $w = 0$ and greedy search at large $w$, and cost-optimality is lost somewhere along it.

Where it is lost can be derived. Write $h^\ast(x)$ for the true cost of the cheapest path from $x$ to the goal, the cost to go; it is what value iteration computes for a problem with deterministic moves and a cost per move, the negative of the value of Chapter 1. A heuristic is **admissible** if it never overestimates, $h(x) \le h^\ast(x)$ for every $x$ (with $h = 0$ at the goal). It is **consistent** if it never drops by more than the cost of a move, $h(x) \le c(x, y) + h(y)$ for every move from $x$ to $y$ of cost $c(x, y)$, and consistency implies admissibility by induction from the goal. The Manhattan distance is consistent on this map, since a move changes it by exactly $1$ and costs at least $1$.

The consequence that matters is for the priority $f = g + h$. Along a move from $x$ to $y$, with $g(y) = g(x) + c(x, y)$,

$$f(y) = g(x) + c(x, y) + h(y) \ge g(x) + h(x) = f(x),$$

so with a consistent heuristic the priority never decreases along a path. A cell leaves the queue, then, with the cheapest cost that any path to it has; if a cheaper path existed, the frontier would hold a cell on it with a priority no larger than the cell's, and that cell would have left first. The goal is no exception, so the first time its entry is popped the path is cheapest, and an admissible but inconsistent heuristic that is combined with a closed set can fail on this very step, because a cell closed too early is never reopened. With $w > 1$ the heuristic of this map is no longer admissible at all: along the cheapest route the true cost to go from $S$ is $14$ and $2h(S) = 2 \times 12 = 24$. The inflated heuristic hurries the search toward the straight way across the mud, whose true cost it never counts.

The other end of the dial is as instructive. If $h = h^\ast$ exactly, the priority of every cell on a cheapest path equals the optimal cost, and every cell off it is worse; the search expands little more than the path, which is what the cost-to-go table of Chapter 1 would have given. It does not expand exactly the path, as the reference line under the next section's table shows (a cheapest path of $14$ moves, $17$ expansions), because cells off the returned path that lie on other cheapest paths have the same priority and may leave the queue first.

## A heuristic that was learned from the exact costs

Chapter 1's value function is a table, and a table of $h^\ast$ over every cell is what a search is trying not to need. What a search can use is something cheaper than the table that is nearly right, and a network trained to predict the cost to go is the usual choice. Take the multilayer perceptron of Volume III with two inputs, the row and the column of the cell scaled into $[0, 1]$, twelve tanh units and one linear output: $2 \times 12 + 12 + 12 + 1 = 49$ parameters for a map of $79$ cells. It is trained by gradient descent to match the exact cost to go, and its prediction is then used as $h$.

<!-- worked: search_planning.learned_heuristic -->
A network trained to predict the exact cost to go of the map above. Training set: all 79 enterable cells, target CTG(x) / 14 where CTG is the cost of the cheapest path from the cell to G (CTG(S) = 14). Network: inputs (row / 6, column / 11), 12 tanh units, one linear output; all biases start at 0; initial weights from the book's generator `search_planning.heuristic_net`, seed 0x0f8da0f99485c251 (1120728876084216401): W1 (2 x 12, row-major, each normal() divided by sqrt(2)), then W2 (12 values, each normal() divided by sqrt(12)). Loss: mean squared error over the 79 cells; full-batch gradient descent, learning rate 0.1, tanh the correctly rounded library tanh. The prediction of a cell is the output times 14. Each line is the network after the stated number of steps. 'too high' counts the cells whose prediction exceeds CTG; 'largest excess' is the largest prediction minus CTG. A* then uses h = max(0, prediction), 'raw', or h = max(0, prediction - largest excess), 'shifted', which never exceeds CTG on these cells. Priority g + h, the conventions of the previous table. Digits follow the nudge rule of `stable.py` (the first weight W1[0][0] changed by each nudge); the counts are integers.

| steps | training loss | cells too high | largest excess | raw: expansions | raw: cost | shifted: expansions | shifted: cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.01979 | 40 | 3.76 | 35 | 14 | 40 | 14 |
| 300 | 0.01512 | 41 | 3.32 | 25 | 14 | 33 | 14 |
| 1000 | 0.01032 | 44 | 2.58 | 20 | 14 | 22 | 14 |
| 3000 | 0.00928 | 38 | 2.37 | 17 | 14 | 21 | 14 |
| 10000 | 0.00830 | 46 | 2.38 | 19 | 14 | 23 | 14 |

For reference, the same search with h = CTG (exact) expands 17 cells and finds cost 14; with the Manhattan distance 38 cells, cost 14; with h = 0 (uniform-cost) 72 cells, cost 14.
<!-- /worked -->

After $100$ steps of training the network has a squared error of about $0.02$ on the costs scaled by $14$, and A\* with its output as $h$ expands $35$ cells, already fewer than the $38$ of the Manhattan distance. With more training the count falls, and at $3000$ steps it reaches $17$, the same as the exact $h^\ast$ in the line below the table, against $72$ for no heuristic at all. The network is trained on all $79$ cells, so this is not generalisation: it is the effect that a learned approximation of the value function has when the search uses it as a guide.

Learned values are not promises, though. The column "cells too high" counts the cells where the network predicts more than the true cost to go, between $38$ and $46$ of the $79$ at every stage. A heuristic that overestimates in half the map is not admissible, and the printed cost, $14$ in every row, is luck that the search could not have promised. What can be done about it is the last pair of columns. Subtract the largest excess found on the cells from every prediction, and the result is admissible on these cells by construction. It costs expansions, $21$ against $17$ at $3000$ steps, which is the price of a guarantee that the raw network does not have; and it is only a guarantee on cells checked, so a system that relies on it checks the excess where it will search.

## Trees grown by playouts, on a game small enough to solve

The maps of the last three sections list the whole space and ask for one path. The second kind of problem is a game, where the path depends on what an opponent does, and where the tree is so wide that nobody looks at all of it. The game here is small enough to be solved exactly and so to check every claim: a pile of stones, two players taking $1$, $2$ or $3$ at a turn, and whoever takes the last stone wins.

<!-- worked: search_planning.take_away_game -->
The take-away game: a pile of n stones, two players alternate, each takes 1, 2 or 3 stones (never more than are left), and whoever takes the last stone wins. Solved exactly by induction on n: the player to move wins when some move leaves a position that loses for the other player, and with 0 stones the player to move has lost.

| n | player to move | winning moves |
|---|---:|---|
| 0 | - | - |
| 1 | wins | 1 |
| 2 | wins | 2 |
| 3 | wins | 3 |
| 4 | loses | none |
| 5 | wins | 1 |
| 6 | wins | 2 |
| 7 | wins | 3 |
| 8 | loses | none |
| 9 | wins | 1 |
| 10 | wins | 2 |
| 11 | wins | 3 |
| 12 | loses | none |
| 13 | wins | 1 |

The unshared game tree from a pile of n stones has T(n) = 1 + T(n - 1) + T(n - 2) + T(n - 3) nodes (terms with a negative argument left out) with T(0) = 1; counting the position itself: T(10) = 600, T(20) = 266 079, T(30) = 117 897 840, T(40) = 52 239 653 201.
<!-- /worked -->

The losing positions are the multiples of $4$, since whatever the player to move takes, her opponent can take the complement to $4$. From a pile of $10$ the only winning move is to take $2$ and leave $8$. This is the solved game, and the last line of the table says why one would not want to solve a larger one by exhausting the tree: from a pile of $40$ the unshared tree has $52$ billion nodes.

**Monte Carlo tree search** grows a tree of the positions that matter, one iteration at a time, and decides from what it has grown. It needs no heuristic: where the tree stops, the value of a position is estimated by a **playout**, a game played to the end by random moves, whose winner is a sample of the position's value. Each iteration has four parts.

1. **Selection.** From the root, while every legal move of the current node already has a child, go to the child with the highest score $W/N + c\sqrt{\ln N_{\rm parent}/N}$ and repeat. Here $N$ counts the iterations that passed through a node and $W$ counts those whose playout was won by the player who moved into it, so $W/N$ is the share of wins for the player who chose that move, and the second term is a bonus that grows for a child that has been visited little relative to its parent. The rule is called **UCT**, upper confidence bounds applied to trees. The constant $c$ sets how much exploration the bonus buys; the table below uses $c = \sqrt{2}$.
2. **Expansion.** At the node reached, if a move has no child yet, add a child for the smallest such move.
3. **Playout.** From the new node, both players take uniformly random moves until the pile is empty.
4. **Backup.** On every node of the path from the new node to the root, add $1$ to $N$, and add $1$ to $W$ if the winner of the playout is the player who moved into that node.

After all iterations, the move played is the root's child with the most visits (a tie to the smaller move). The most visited and not the one with the best $W/N$ is the choice because visits are where the search put its confidence: a child with one lucky playout has the best ratio and the least evidence.

```text
function uct_search(n0: int, iters: int, c: real, rng: Generator)
        -> (N: list<int>, W: list<int>, child: list<map<int, int>>):
    require n0 >= 1 and iters >= 1 and c >= 0
    pile = [n0]
    depth = [0]
    parent = [-1]
    N = [0]
    W = [0]
    child = [map of int to int, empty]
    for it in 0 .. iters:
        v = 0
        while pile[v] > 0 and len(child[v]) == min(3, pile[v]):
            best_score = -infinity
            for m in 1 ..= min(3, pile[v]):
                u = child[v][m]
                score = W[u] / N[u] + c * sqrt(log(N[v]) / N[u])
                if score > best_score:
                    best_score = score
                    best = u
            v = best
        if pile[v] > 0:
            m = 1
            while m in child[v]:
                m = m + 1
            append pile[v] - m to pile
            append depth[v] + 1 to depth
            append v to parent
            append 0 to N
            append 0 to W
            append (map of int to int, empty) to child
            child[v][m] = len(pile) - 1
            v = len(pile) - 1
        p = pile[v]
        d = depth[v]
        while p > 0:
            m = 1 + rng.integer(min(3, p))
            p = p - m
            d = d + 1
        winner = (d - 1) mod 2
        u = v
        loop:
            N[u] = N[u] + 1
            if depth[u] >= 1 and (depth[u] - 1) mod 2 == winner:
                W[u] = W[u] + 1
            if parent[u] < 0:
                break
            u = parent[u]
    return (N, W, child)
```

Node $0$ is the root and `depth[v]` counts the moves played to reach node $v$, so the player to move at the root is player $0$, who moves at every even depth, and the winner of a playout is the parity of the number of the move that took the last stone, `(d - 1) mod 2`. The score of a child is evaluated in the order of the moves $1, 2, 3$ and a tie goes to the first. The root's choice is the move $m$ that maximises `N[child[0][m]]`, a tie to the smaller $m$. `log` is the natural logarithm and the only randomness is `rng.integer`, one draw for each move of each playout. Here is the first dozen iterations, from a pile of $10$.

<!-- worked: search_planning.uct_trace -->
The first 12 iterations of Monte Carlo tree search on the take-away game from n = 10, exploration constant c = sqrt(2) = 1.414214. Player 0 moves at the root. Generator: the book's SplitMix64, `search_planning.uct` seed 0x52726b6560254b6f (5940928941604031343); the only draws are the playouts, one rng.integer(k) for each move, k = min(3, stones left), the move being 1 + the draw, consumed in iteration order. Selection: from the root, while every move of the node has a child, go to the child that maximises W/N + c sqrt(ln N_parent / N) (a tie to the smaller move), where N is a node's visit count and W its number of visits whose playout was won by the player who moved into it. Expansion: the smallest move without a child gets one. Playout: random moves to the end from the new node. Backup: N + 1 on every node of the path, W + 1 on those whose mover won. 'selected' lists the moves of the selection part of the path, 'new' the move expanded and 'playout' the moves of the playout.

| iteration | selected | new | playout | winner |
|---:|---|---:|---|---|
| 1 | - | 1 | 1 3 2 1 2 | player 1 |
| 2 | - | 2 | 2 1 2 3 | player 0 |
| 3 | - | 3 | 2 2 2 1 | player 0 |
| 4 | 2 | 1 | 1 3 1 1 1 | player 0 |
| 5 | 3 | 1 | 1 2 2 1 | player 1 |
| 6 | 2 | 2 | 1 3 2 | player 0 |
| 7 | 2 | 3 | 2 2 1 | player 0 |
| 8 | 2 1 | 1 | 1 3 1 1 | player 0 |
| 9 | 1 | 1 | 1 3 1 3 | player 1 |
| 10 | 3 | 2 | 3 1 1 | player 0 |
| 11 | 2 2 | 1 | 1 3 1 | player 1 |
| 12 | 3 | 3 | 2 2 | player 1 |

After these 12 iterations the root's children are:

| move | stones left | N | W | W / N |
|---:|---:|---:|---:|---:|
| 1 | 9 | 2 | 0 | 0.000 |
| 2 | 8 | 6 | 5 | 0.833 |
| 3 | 7 | 4 | 2 | 0.500 |
<!-- /worked -->

Follow a few rows. Iterations $1$ to $3$ expand the three moves of the root in order, with a playout each. In iteration $4$ every move of the root has a child, so selection is real: the children of moves $2$ and $3$ both have $W/N = 1$ and the same visit count, so they tie, and the smaller move wins. The row shows "selected $2$, new $1$": the search went to the child of move $2$ (a pile of $8$ and the opponent to move) and gave it a child for taking $1$. In iteration $9$ the selection goes to move $1$, whose single playout was lost, because after iterations $1$ to $8$ its bonus has grown past its poor ratio: that is exploration doing what $c$ asks of it. After the twelve iterations, the root's children have $N = 2, 6, 4$ with $W = 0, 5, 2$, and the winning move $2$ already has the largest count.

A longer run makes the concentration plain.

<!-- worked: search_planning.uct_visits -->
The root of the same search after 1000 iterations from n = 10, c = sqrt(2), a generator started from seed 0xa4f5e03c7c903eea (11886653341906779882) (the name `search_planning.uct.long`; not the continuation of the run above). 'for the mover' says whether the move wins with best play (the table of the game); W / N is the share of visits whose playout the mover won. Counts follow the nudge rule (c changed by each nudge); a count is printed only if all five runs agree.

| move | stones left | for the mover | N | W | W / N |
|---:|---:|---|---:|---:|---:|
| 1 | 9 | loses | 133 | 71 | 0.534 |
| 2 | 8 | wins | 808 | 673 | 0.833 |
| 3 | 7 | loses | 59 | 23 | 0.390 |

The move chosen (most visits, a tie to the smaller move) is 2; the winning move is 2.
<!-- /worked -->

After $1000$ iterations the winning move has $808$ visits, about four in five, and $W/N = 0.833$. The ratio is not $1$ because the playouts that follow a good move are played by random players on both sides, so even a won position loses sometimes; what the search has found is that move $2$ is much better than the others, not what it is worth. The mean of $W/N$ is an average of Bernoulli outcomes $X$ with variance $p(1 - p)$, and its standard error falls as $1/\sqrt{N}$, the fact of Volume I that recall question $3$ asks for. Four times the playouts halves it.

How much search is enough depends on how much exploration. The next table counts how often a search of a given budget picks the winning move.

<!-- worked: search_planning.uct_budget -->
How often search from n = 10 picks the winning move 2: 100 independent searches for every exploration constant and budget, the root move chosen by most visits (a tie to the smaller move). The row of a constant is its position i = 0 .. 3 in the table, the column of a budget its position j = 0 .. 6, and search number r = 0 .. 99 of that cell draws its playouts, as in the trace, from the book's generator whose seed is seed_of of the text `search_planning.uct_budget.i.j.r` with the three numbers written in decimal (for instance `search_planning.uct_budget.0.0.0` has seed 0xb4cb8354adb89ea9 (13027650746814340777)); every search restarts its own generator. Each entry is a count out of 100; counts follow the nudge rule (c changed by each nudge).

| c \ iterations | 10 | 25 | 50 | 100 | 200 | 400 | 800 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 30 | 37 | 43 | 43 | 51 | 61 | 56 |
| 0.5 | 46 | 48 | 42 | 50 | 67 | 89 | 100 |
| sqrt(2) | 29 | 41 | 37 | 48 | 55 | 80 | 100 |
| 4 | 37 | 36 | 37 | 60 | 41 | 46 | 85 |
<!-- /worked -->

Read each row as a trade. At $c = \sqrt{2}$ the search picks the right move $29$ times in $100$ with $10$ iterations, close to the one in three of guessing, and $100$ times in $100$ with $800$. A small constant, $c = 0.1$, stays between $30$ and $61$ even at $800$ iterations: the search follows the child that looks best, and a child that wins its first playouts by chance can keep the visits that would have shown it to be wrong. A large one, $c = 4$, spends the visits on the children it has already seen to be poor and reaches $85$. The best constants are the middle ones, $0.5$ and $\sqrt{2}$, both at $100$ with $800$ iterations. Each entry is a count out of $100$, so differences of five or so are noise: $c = 4$ at $100$ iterations is $60$ and at $200$ is $41$.

A word on the cost. This game is small: its whole tree from a pile of $10$ has $600$ nodes, fewer than the $800$ iterations of the last column, so on this toy the exact search is the cheaper one. The method pays when the tree is astronomically larger than the iterations (the $52$ billion nodes from a pile of $40$) and when no heuristic is at hand.

## A policy as a prior, and the loop that trains it

The playouts of the last section were uniform. A policy that already prefers good moves would make them better, and a policy could also say which children to visit first. **PUCT**, the rule of the systems that play board games, does the second: every node's children carry the policy's probabilities as **priors** $P$, and selection takes the child with the highest

$$q + c_{\rm puct}\,P\,\frac{\sqrt{N_{\rm parent}}}{1 + N},$$

with $q = W/N$ ($0$ for a child not visited yet). The bonus is proportional to the prior, so a move the policy likes is tried first and a move it dislikes has to earn its visits. Here all children of a node are created the first time the node is selected from, a child never visited before is the leaf that gets a playout, and the playout draws the moves of both players from the policy. The policy is a table of preferences $\theta[n][i]$ for the legal moves $i + 1$ of every pile $n$, with softmax probabilities as in Chapter 3.

```text
function puct_search(theta: list<list<real>>, n0: int, iters: int,
        c_puct: real, rng: Generator)
        -> (N: list<int>, W: list<int>, child: list<map<int, int>>):
    require n0 >= 1 and iters >= 1
    pile = [n0]
    depth = [0]
    parent = [-1]
    N = [0]
    W = [0]
    child = [map of int to int, empty]
    P = [empty list of real]
    for it in 0 .. iters:
        v = 0
        loop:
            if pile[v] == 0:
                break
            if N[v] == 0 and v != 0:
                break
            if len(child[v]) == 0:
                for m in 1 ..= min(3, pile[v]):
                    append pile[v] - m to pile
                    append depth[v] + 1 to depth
                    append v to parent
                    append 0 to N
                    append 0 to W
                    append (map of int to int, empty) to child
                    append (empty list of real) to P
                    child[v][m] = len(pile) - 1
                P[v] = softmax(theta[pile[v]][0 : min(3, pile[v])])
            s = sqrt(max(N[v], 1))
            best_score = -infinity
            for m in 1 ..= min(3, pile[v]):
                u = child[v][m]
                q = W[u] / N[u] if N[u] > 0 else 0.0
                score = q + c_puct * P[v][m - 1] * s / (1 + N[u])
                if score > best_score:
                    best_score = score
                    best = u
            v = best
        p = pile[v]
        d = depth[v]
        while p > 0:
            m = 1 + sample_index(softmax(theta[p][0 : min(3, p)]), rng)
            p = p - m
            d = d + 1
        winner = (d - 1) mod 2
        u = v
        loop:
            N[u] = N[u] + 1
            if depth[u] >= 1 and (depth[u] - 1) mod 2 == winner:
                W[u] = W[u] + 1
            if parent[u] < 0:
                break
            u = parent[u]
    return (N, W, child)
```

The other half of the loop is what the search leaves behind. The visit counts of the root's children, divided by their sum, are a distribution over moves that is the search's answer and is better than the policy that guided it (when it is), and fitting the policy to it is ordinary supervised learning with the cross-entropy of Volume III: the policy is the student and the visit distribution the target. A better policy then guides the next search better, and the loop is called **expert iteration**: the search is the expert, and the policy is trained to imitate it. One round for every pile $n = 1, \dots, 13$:

```text
function expert_round(theta: list<list<real>>, n_max: int, budget: int,
        c_puct: real, steps: int, rate: real, round: int)
        -> list<list<real>>:
    target = map of int to list<real>, empty
    for n in 1 ..= n_max:
        rng = Generator(seed_of("search_planning.expert_iteration." + round + ".target." + n))
        (N, W, child) = puct_search(theta, n, budget, c_puct, rng)
        visits = [N[child[0][m]] for m in 1 ..= min(3, n)]
        target[n] = visits / sum(visits)
    for step in 0 .. steps:
        grad = map of int to list<real>, empty
        for n in 1 ..= n_max:
            grad[n] = softmax(theta[n][0 : min(3, n)]) - target[n]
        for n in 1 ..= n_max:
            for i in 0 .. min(3, n):
                theta[n][i] = theta[n][i] - rate * grad[n][i]
    return theta
```

The first loop is in order of $n$. The second makes `steps` full-batch steps, each computing every gradient from the current $\theta$ before any entry is changed; $p - t$ is the gradient of the cross-entropy to the target $t$ with respect to the logits, the identity of Volume III. Text joined with `+` is the concatenation of the pieces, numbers written in decimal. Here is the loop on the take-away game, with all numbers printed:

<!-- worked: search_planning.expert_iteration -->
Expert iteration on the take-away game, piles 1 to 13. The policy is a table of preferences theta[n][i] for the legal moves i + 1 of every pile n (all 0 at the start, so the policy is uniform), with softmax probabilities. One round: (1) for every pile n = 1 .. 13 in order, one PUCT search of 60 iterations with c_puct = 1.5 whose priors and playout moves come from the current policy, and the target for the pile is the root's visit counts divided by their sum; (2) 50 full-batch gradient steps of rate 1.0 on the cross-entropy to those targets, theta[n][i] <- theta[n][i] - rate (policy[n][i] - target[n][i]), all piles updated together from the same current policy. Before every round the policy is evaluated on the 10 piles that are won for the player to move (n not a multiple of 4): 'policy alone' counts the piles where the policy's largest probability (a tie to the smaller move) is a winning move; 'search with the policy' counts, over 20 searches of 60 iterations per pile in order of n, how many choose a winning move by most visits (a tie to the smaller move). Every search has a generator of its own, the book's generator whose seed is seed_of of the text `search_planning.expert_iteration.R.eval.n.r` for search r = 0 .. 19 of the evaluation of pile n in round R, and of `search_planning.expert_iteration.R.target.n` for the target search of pile n in round R, all numbers in decimal (for instance `search_planning.expert_iteration.0.target.1` has seed 0x1ee51fc11488b1ca (2226220504938230218)). Counts follow the nudge rule (c_puct changed by each nudge).

| rounds of training | policy alone | search with the policy |
|---:|---:|---:|
| 0 | 4 of 10 | 146 of 200 |
| 1 | 8 of 10 | 200 of 200 |
| 2 | 10 of 10 | 200 of 200 |
| 3 | 10 of 10 | 200 of 200 |
| 4 | 10 of 10 | 200 of 200 |
| 5 | 10 of 10 | 200 of 200 |

A round costs 13 searches of 60 iterations, 780 iterations in all (each one playout); the policy alone costs no iteration at decision time.
<!-- /worked -->

The first line is the starting point: a uniform policy that always plays the first move, because every probability is equal and ties go to the smaller move, is right on the $4$ piles where taking $1$ wins and wrong on the other six; search from that policy, at $60$ iterations, picks a winning move $146$ times out of $200$. One round of training later the policy alone is right on $8$ piles and the search with it on all $200$ of $200$ evaluation searches. After the second round the policy alone is right on all ten won piles. The tables are small, but the structure is the one of much larger systems: the policy improves because search found better moves than it would have played, and search improves because the policy tells it where to look.

Note what the policy buys at decision time: after two rounds it alone is right on every won pile and needs no playout, because the search has been compressed into the table. The compression is imperfect in general (a policy cannot hold what its capacity does not allow), which is why the best systems keep searching at decision time too, and it is the reason the crates at the end of this chapter exist.

A value function can play the same part as a policy, replacing the playout instead of guiding the selection: where the leaf of a tree has a network's estimate of the position instead of a random game's winner, one forward pass replaces a whole playout. This is what the heuristic of the last-but-one section was, in the single-agent setting: a value network deciding the order of a frontier.

## Keeping one good solution per kind: an archive over behaviours

The searches so far look for a single best answer. There are problems in which the best answer cannot be reached by always improving the best answer so far, because the way there passes through answers that look worse. A solution to a puzzle that wins by a trick may begin by being worse than the plain solutions that score a little less; a robot gait that crosses a gap may begin with stumbling. A search that keeps only the highest scorers deletes the stepping stones, the first thing its comparison does.

The remedy is to stop asking for the best and ask for the best of each kind. Describe every candidate by a **behaviour descriptor**, a few numbers computed from what the candidate does (not from its parameters) that say what kind of solution it is: where a walker ends up, which keys a run holds. Cut the space of descriptors into cells, called **niches**. A **quality-diversity archive** keeps, for each niche, the best candidate found so far, its **elite**, and it generates new candidates by taking an elite from a niche and changing it. It is therefore a map from behaviours to the best known way of showing each, which is a set of many different good solutions. In the form of **MAP-Elites** (multi-dimensional archive of phenotypic elites), the archive is a grid:

```text
function map_elites(budget: int, n_init: int, k: int, sigma: real, rng: Generator)
        -> map<(int, int), (real, array<real>[4])>:
    require budget >= n_init and n_init >= 1 and k >= 1 and sigma > 0
    archive = map of (int, int) to (real, array<real>[4]), empty
    for i in 0 .. budget:
        genome = zeros(real, [4])
        if i < n_init:
            for j in 0 .. 4:
                genome[j] = rng.uniform()
        else:
            cells = keys(archive)
            parent = archive[cells[rng.integer(len(cells))]][1]
            for j in 0 .. 4:
                genome[j] = min(1.0, max(0.0, parent[j] + sigma * rng.normal()))
        f = fitness(genome)
        c = cell_of(genome, k)
        if not (c in archive) or f > archive[c][0]:
            archive[c] = (f, genome)
    return archive
```

A genome $\boldsymbol\theta = (\theta_0, \dots, \theta_3)$ is $4$ numbers in $[0, 1]$, its genes. The toy of this section is a walker that takes two steps, $\mathbf{s}_1 = (\theta_0 - 0.5,\, \theta_1 - 0.5)$ and $\mathbf{s}_2 = (\theta_2 - 0.5,\, \theta_3 - 0.5)$, and ends at $\mathbf{p} = \mathbf{s}_1 + \mathbf{s}_2$, a point of the square $[-1, 1]^2$. The **descriptor** is the end point mapped to the unit square, $\mathbf{d} = ((p_x + 1)/2,\ (p_y + 1)/2)$, and `cell_of(genome, k)` cuts it into $k \times k$ cells: a coordinate $d$ falls into the cell $\min(\lfloor k d \rfloor,\, k - 1)$, so a cell includes its lower edge and excludes its upper edge, and the coordinate $d = 1$, which clipped genes can produce, belongs to the last cell. The fitness is a landscape with two peaks, minus a small penalty for effort,

$$\text{fit}(\boldsymbol\theta) = \max\bigl(A(\mathbf{p}), B(\mathbf{p})\bigr) - 0.05\,(\lVert\mathbf{s}_1\rVert + \lVert\mathbf{s}_2\rVert),$$

with $A(\mathbf{p}) = 0.7\exp(-\lVert\mathbf{p} - (-0.55, -0.55)\rVert^2 / (2 \times 0.45^2))$, a broad low peak, and $B(\mathbf{p}) = 1.0\exp(-\lVert\mathbf{p} - (0.8, 0.8)\rVert^2 / (2 \times 0.12^2))$, a narrow high one, far apart. Between them the landscape is almost flat and almost zero. The toy is built to be deceptive, which is the point: a search that climbs from a random start will reach the broad peak with high probability and has no gradient to the narrow one.

Three searches with the same budget, $2000$ fitness evaluations each, are compared. The **hill climber** keeps one genome and replaces it by a mutant only if the mutant is strictly better. The **best-100 population** keeps the hundred best genomes found so far, picks a parent among them uniformly and lets a mutant replace the worst member if strictly better; this is the leaderboard. **MAP-Elites** keeps one elite per cell of a $10 \times 10$ grid. The mutation of all three adds $0.1$ times a standard normal to every gene and clips into $[0, 1]$.

<!-- worked: search_planning.quality_diversity_results -->
A deceptive landscape and three searches with the same budget. A genome is four numbers g0 .. g3 in [0, 1]. A walker takes two steps, s1 = (g0 - 0.5, g1 - 0.5) and s2 = (g2 - 0.5, g3 - 0.5), and ends at p = s1 + s2, a point of the square [-1, 1]^2. Fitness is F(p) minus 0.05 (|s1| + |s2|) with Euclidean lengths, where F(p) is the larger of a broad low peak A(p) = 0.7 exp(-|p - (-0.55, -0.55)|^2 / (2 x 0.45^2)) and a narrow high peak B(p) = 1.0 exp(-|p - (0.8, 0.8)|^2 / (2 x 0.12^2)). The behaviour descriptor of a genome is its end point mapped to the unit square, d = ((p_x + 1) / 2, (p_y + 1) / 2); a 10 x 10 grid cuts it into cells, a coordinate d in [0, 1] falling in cell min(floor(10 d), 9) (lower edge inclusive, upper edge exclusive, d = 1 in the last cell). Every search makes 2000 fitness evaluations, one for every genome ever produced. Mutation: every gene g <- clip(g + 0.1 normal(), 0, 1), genes in order 0 .. 3, one normal() each, from the book's generator. Random genomes: four uniform() values in order. Hill climber: one random genome, then 1999 mutants of the current genome, each kept only if its fitness is strictly higher. Best-100: 100 random genomes, then 1900 times: a parent drawn by integer(len) from the list in position order, a mutant, which replaces the member of lowest fitness (a tie to the lowest position) if its fitness is strictly higher. MAP-Elites: 100 random genomes are offered to the archive, then 1900 times: a cell drawn by integer(number of occupied cells) from the occupied cells in ascending (column-of-p_x, column-of-p_y) order, its elite mutated and offered; an offer fills an empty cell and replaces an elite only if its fitness is strictly higher. One generator, `search_planning.quality_diversity` seed 0xcd618910bd30e056 (14799260555502739542), serves all 60 runs: 20 hill climbs, then 20 best-100 runs, then 20 MAP-Elites runs, each run continuing the stream. 'found B' counts runs whose best fitness exceeds 0.9 (the narrow peak). 'cells' is the mean number of distinct cells the final population occupies (1 for the single genome of the hill climber). Digits follow the nudge rule (mutation width changed by each nudge).

| search | found B | best fitness, highest run | best fitness, lowest run | cells |
|---|---:|---:|---:|---:|
| hill climber (1 + 1) | 0 of 20 | 0.661 | 0.661 | 1.0 |
| best-100 population | 5 of 20 | 0.943 | 0.661 | 1.8 |
| MAP-Elites, 10 x 10 cells | 20 of 20 | 0.941 | 0.901 | 98.8 |
<!-- /worked -->

The hill climber finds the narrow peak in none of its $20$ runs; every one ends at the top of the broad peak, fitness $0.661$. The best-100 population has the same pressure with a hundred members and finds it in $5$ runs out of $20$, when a random start happened to land near it or a mutant jumped there; its final population occupies $1.8$ cells on average, so what it holds is one idea a hundred times. MAP-Elites finds it in every run, with best fitness between $0.901$ and $0.941$, and it ends with $98.8$ of the $100$ cells filled on average.

![Two ten by ten heat maps with the same colour scale. The left map, the landscape, is dark in the lower left around cells (1 to 3, 1 to 3) and dark in the top right cells (8 and 9, 8 and 9), pale elsewhere. The right map, the archive of the first MAP-Elites run, shows the same two dark regions, with the elites of the corner cells at 0.94 and 0.89, and shades of pale between the peaks.](img/archive-heat-map.svg)

The archive of the first MAP-Elites run, cell by cell:

<!-- worked: search_planning.quality_diversity_archive -->
The archive of the first MAP-Elites run (the first of the 20 that follow the 40 runs before it in the stream): the fitness of the elite of each of the 10 x 10 cells, two decimals, with the horizontal axis the cell of the descriptor's first coordinate and the vertical axis that of its second (cell 9 on the top row). An empty cell is blank. 99 of 100 cells are occupied; empty cells: (9, 9). The narrow peak B = (0.8, 0.8) has the descriptor (0.9, 0.9), the corner where the cells (8, 8), (9, 8), (8, 9) and (9, 9) meet.

| y cell \ x cell | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 9 | -0.06 | -0.05 | -0.04 | -0.04 | -0.04 | -0.04 | -0.04 | 0.09 | 0.79 |  |
| 8 | -0.04 | -0.02 | -0.01 | -0.01 | -0.01 | -0.02 | -0.03 | 0.15 | 0.94 | 0.89 |
| 7 | 0.00 | 0.03 | 0.03 | 0.03 | 0.02 | 0.00 | -0.01 | -0.02 | 0.13 | 0.06 |
| 6 | 0.08 | 0.13 | 0.13 | 0.10 | 0.10 | 0.05 | 0.02 | -0.02 | -0.03 | -0.05 |
| 5 | 0.17 | 0.27 | 0.29 | 0.25 | 0.19 | 0.10 | 0.05 | 0.00 | -0.02 | -0.04 |
| 4 | 0.36 | 0.46 | 0.46 | 0.45 | 0.35 | 0.19 | 0.08 | 0.02 | -0.02 | -0.04 |
| 3 | 0.51 | 0.59 | 0.62 | 0.58 | 0.45 | 0.27 | 0.14 | 0.04 | -0.01 | -0.04 |
| 2 | 0.53 | 0.65 | 0.66 | 0.62 | 0.48 | 0.30 | 0.15 | 0.04 | -0.02 | -0.05 |
| 1 | 0.52 | 0.62 | 0.65 | 0.61 | 0.43 | 0.28 | 0.12 | 0.02 | -0.03 | -0.05 |
| 0 | 0.41 | 0.52 | 0.54 | 0.51 | 0.37 | 0.21 | 0.08 | 0.00 | -0.03 | -0.06 |
<!-- /worked -->

Read the table along the diagonal from the broad peak at the lower left to the corner at the top right. Cells such as $(5, 5)$ and $(7, 7)$ have elites with fitness about $0.1$ and $0$, and cell $(7, 8)$ holds $0.15$. None of these would survive a ranking by fitness for long. They are stepping stones, and every one of them is also a starting point that the next mutation can set off from; the elites of cells $(8, 8)$ and $(9, 8)$ reached $0.94$ and $0.89$, and the cell $(9, 9)$ is the only one left empty. An archive that kept the best candidates would have deleted the whole chain. What MAP-Elites does that the climber does not is to change the question: it never asks for a candidate that scores higher than the best, only for one that does something not yet done, or does it better, and the comparison is local to the cell.

Two limits are worth stating. A grid of $k$ cells per coordinate has $k^d$ cells in $d$ descriptor dimensions, so the method is for a handful of dimensions; and the quality of the result depends on a descriptor that really separates kinds of solutions, which is a design decision, not something the algorithm finds. The next sections show how brain's crate makes both decisions.

## Asking for a goal that was not the one set

A last way learning and search meet is in the goals. An agent that has to reach many different goals is described by a **goal-conditioned policy**, a policy that takes the goal as an input beside the state, $\pi(a \mid s, s_{\mathrm{goal}})$; the value functions of Chapter 2 are conditioned in the same way, $Q(s, a, s_{\mathrm{goal}})$. One such function replaces a separate policy for every goal. The difficulty is the reward. A goal that is reached only by a long string of lucky moves pays $1$ once in a thousand attempts, and the other $999$ attempts teach the learner nothing at all.

Here is the situation in numbers. The environment is an $8 \times 8$ grid with four moves, and the reward is $1$ for the move after which the agent stands on the goal. A random behaviour generates $300$ episodes of $12$ moves each, every one with its own goal; a transition is the tuple (state, move, next state, goal).

**Hindsight relabelling** is the observation that an episode that missed its goal did reach something, namely the states it visited, and that it is a perfect example of how to reach those. Replace the goal of a transition by a state that the episode did reach, recompute the reward by the same rule, and the transition is a legitimate lesson for another goal. It needs no new interaction, since the dynamics do not depend on the goal, and it needs a learner that can learn from transitions generated by a behaviour that was aiming elsewhere, which is what the Q-learning of Chapter 2 is. Two strategies for choosing the new goal: the **final** strategy relabels the whole episode with the state it ended on, and the **future** strategy relabels each transition at time $t$ with $k$ states drawn uniformly from the later positions of the same episode.

```text
function relabel_future(traj: list<int>, acts: list<int>, H: int, k_per: int,
        rng: Generator) -> list<(int, int, int, int)>:
    require len(traj) == H + 1 and len(acts) == H
    out = empty list of (int, int, int, int)
    for t in 0 .. H:
        for j in 0 .. k_per:
            kk = t + 1 + rng.integer(H - t)
            append (traj[t], acts[t], traj[t + 1], traj[kk]) to out
    return out

function goal_q_sweeps(transitions: list<(int, int, int, int)>, sweeps: int,
        alpha: real, gamma: real) -> map<(int, int, int), real>:
    Q = map of (int, int, int) to real, empty
    for sweep in 0 .. sweeps:
        for (s, a, s2, goal) in transitions:
            if s2 == goal:
                y = 1.0
            else:
                y = gamma * max([get(Q, (s2, b, goal), 0.0) for b in 0 .. 4])
            old = get(Q, (s, a, goal), 0.0)
            Q[(s, a, goal)] = old + alpha * (y - old)
    return Q
```

The reward of a transition is $1$ when its next state is its goal and $0$ otherwise, and `goal_q_sweeps` is the update of Chapter 2 with the goal as part of the key. The table counts what each strategy gives the learner and what the learner does with it.

<!-- worked: search_planning.hindsight_relabelling -->
A goal-reaching task on an 8 x 8 grid without walls, cells numbered 8 r + c, four moves (up, right, down, left; a move into the border leaves the agent where it is) and a reward of 1 for the move after which the agent stands on the goal, 0 otherwise. Data: 300 episodes by a random behaviour, from the book's generator `search_planning.hindsight` seed 0xcae1f11c509ec582 (14619230969335301506). Per episode, in order: the start by integer(64), the goal by integer(63) with 1 added if it is at least the start (so uniform over the other cells), then 12 moves by integer(4) each. The episode is stored for its original goal up to and including the first move that reaches it (the rest of the walk is still drawn). 'final state' adds, for every one of the 12 moves of the walk, the transition relabelled with the goal that the walk ended on; 'future states' adds, for every move t = 0 .. 11 in order, 4 transitions relabelled with the state after move k, where k = t + 1 + integer(12 - t) is drawn four times in a row for each t. The draws of the episodes come first (all 300 episodes), the draws of the relabelling after them, from the same generator and in the order of the three rows of the table (only the third row draws). A transition is (state, move, next state, goal) and its reward is 1 when the next state is the goal. Learning: tabular Q(state, move, goal), start 0, 30 sweeps over the stored transitions in stored order, target 1 when the next state is the goal and otherwise 0.9 times the largest Q of the next state for the same goal, step 0.5. Evaluation: for every start-goal pair of different cells (4032), follow the move of largest Q (a tie to the smaller move) for up to 24 moves; success if the goal is reached. A table that is all zero (always the first move, up) succeeds on 224 of 4032 pairs.

| relabelling | transitions | with reward 1 | share | pairs solved | share |
|---|---:|---:|---:|---:|---:|
| original goals only | 3456 | 26 | 0.8% | 612 of 4032 | 15.2% |
| final state | 7056 | 672 | 9.5% | 2534 of 4032 | 62.8% |
| future states, k = 4 | 17856 | 5119 | 28.7% | 2781 of 4032 | 69.0% |
<!-- /worked -->

With the original goals only, of $3456$ stored transitions $26$ have a reward of $1$: under one percent, and a learner that sees these and nothing else solves $612$ of the $4032$ pairs, which is better than the $224$ of a table of zeros but still only $15.2\%$. Relabelling with the final state brings $672$ rewarded transitions (a share of $9.5\%$ of $7056$) and $2534$ pairs solved, $62.8\%$. Relabelling with future states, $4$ per transition, brings $5119$ rewarded transitions, $28.7\%$ of $17856$, and $2781$ pairs, $69.0\%$. The number of rewarded transitions grew by a factor of about $200$, and the dynamics did not change. The goal-conditioned table is only possible because the grid is small; for a larger space $(s, g)$ is the input of a network, the multilayer perceptron of Volume III, in which state and goal are two parts of one input vector.

The final and the future strategies differ in what they say about the data. The final goal is one per episode and always a state the episode reached, hence reachable; the future goals are many and come with their own short paths in the data, and so supply goals at every distance along the episode. The price of both is a shift of the data away from the goals that were set, which is why a system that relabels keeps some original goals too.

## How the archive of `crates/search` chooses and admits

`crates/search` is the quality-diversity half of this chapter as a library that a campaign can run. It is written for sequences of decisions that have to be discovered, such as the route through a level of a game or a sequence of moves of a puzzle, and the design is stated in its header (`crates/search/src/lib.rs`): policy improvement by PPO, GRPO, DPO or cloning "is a compression mechanism", which makes a better copy of behaviour that exists in its data, and reaching a strategy nobody has demonstrated is a search problem. The loop it serves is search, verify, select, compress, and the crate holds the first three: an archive, a bandit that allocates budget among search operators, and a ladder of evaluators that verifies. It knows nothing of models and depends on no model crate, so that a sample or a model crate can host a search (commit `96192c4f3`, 22 September 2026, introduced it as a leaf crate for this reason).

The archive in `crates/search/src/archive.rs` is the grid of the last section with four changes, each of them a response to what a real campaign needed.

First, a niche is not a computed cell of a fixed grid but a tuple of integers (`Niche`) that the caller chooses and the archive uses as a key: the caller knows what a coordinate means (which room, which keys held, how much of the level is cleared) and decides its resolution. The candidate itself is opaque to the archive, a generic payload `what` that the archive never looks inside.

Second, the quality of an elite is two numbers, not one. A `Worth` is a pair `reached` (how much was achieved, higher is better) and `cost` (what it took, lower is better, for a speedrun the elapsed game time). The comparison `better_than` makes achievement dominate and cost break ties, with a tolerance `tol`: two achievements within `tol` are equal. The table plays the rule on offers to one niche.

<!-- worked: search_planning.archive_offers -->
Offers to one niche whose elite has Worth(reached 10.0, cost 500), tolerance 0.01, in the order shown, each judged against what is held after the one before. `better_than(a, b, tol)` is true when a.reached > b.reached + tol; false when b.reached > a.reached + tol; and otherwise when a.cost < b.cost. The generation of the cell counts the replacements.

| offer | verdict | held afterwards | generation |
|---|---|---|---:|
| (held) | - | reached 10.0, cost 500 | 0 |
| reached 9.9, cost 100 | Rejected | reached 10.0, cost 500 | 0 |
| reached 10.0, cost 500 | Rejected | reached 10.0, cost 500 | 0 |
| reached 10.005, cost 400 | Improved | reached 10.005, cost 400 | 1 |
| reached 10.02, cost 800 | Improved | reached 10.02, cost 800 | 2 |
| reached 10.0, cost 300 | Rejected | reached 10.02, cost 800 | 2 |
| reached 11.0, cost 900 | Improved | reached 11.0, cost 900 | 3 |
<!-- /worked -->

The row that is rejected at equal achievement and equal cost shows that ties keep the incumbent, as in the toy, and the row with achievement $10.005$ and cost $400$ shows why the tolerance exists. Without it the comparison would be a plain `>` on a score that wobbles by a thousandth between two runs, a faster run would never be admitted and the cost would never come down, a point made in the module's header. It is also why a campaign drives time down everywhere at once: most niches are tied on achievement, so the cost decides. The last column is the `generation` of the cell. It has a history of its own that is told below.

Third, the draw of the cell to start from is a weighted draw. The weight is the product of three factors. The first is $1/\sqrt{\text{visits} + 1}$, the weight of the archive-based exploration method of Ecoffet and colleagues cited in the references, which favours cells that have rarely been set off from. The second is $\exp(-\text{focus} \times (1 - \text{reached}/\text{top}))$, which favours cells that achieved more, relative to the best cell. The third, only when the caller has named the axes along which a step can actually move, is $1 + 0.6 \times \text{loneliness}$ (the crate's constant `EDGE`), where loneliness is the share of a cell's neighbours along those axes that the archive does not hold, so that the edge of the ground reached is preferred to its middle. The draw walks the cells in niche order and takes its one uniform number from the crate's seeded generator, so a seed and an archive determine the cell whatever order the cells were inserted in (`the_same_seed_draws_the_same_cells_whatever_the_insertion_order` in `crates/search/tests/it/archive.rs`).

<!-- worked: search_planning.archive_selection_weight -->
The weight with which `Archive::pick` draws a cell: exp(-focus (1 - reached / top)) / sqrt(visits + 1), the shortfall clamped into [0, 1], times (1 + 0.6 loneliness) when edge axes are named. First, the achievement factor for a cell with no visits, as a fraction of the weight of the best cell (reached / top = 1): the rule before commit b02e79a35, (1 + reached / top) / 2; the new rule at focus 1; the new rule at the default focus 4.0; and the last as the chance, among equals, of being picked rather than the best cell.

| reached / top | old rule | focus 1 | focus 4.0 | against the best cell, focus 4.0 |
|---|---:|---:|---:|---:|
| 1.00 | 1.000 | 1.000 | 1.0000 | 1 in 1.0 |
| 0.75 | 0.875 | 0.779 | 0.3679 | 1 in 2.7 |
| 0.50 | 0.750 | 0.607 | 0.1353 | 1 in 7.4 |
| 0.25 | 0.625 | 0.472 | 0.0498 | 1 in 20.1 |
| 0.00 | 0.500 | 0.368 | 0.0183 | 1 in 54.6 |

Second, the visit factor 1 / sqrt(visits + 1) (it multiplies the first):

| visits | factor |
|---|---:|
| 0 | 1.000 |
| 3 | 0.500 |
| 8 | 0.333 |
| 15 | 0.250 |
| 99 | 0.100 |

Third, the edge factor 1 + 0.6 x loneliness, where loneliness is the share of the cell's neighbours along the named axes (one step each way) that the archive does not hold:

| loneliness | factor |
|---|---:|
| 0.00 | 1.000 |
| 0.25 | 1.150 |
| 0.50 | 1.300 |
| 1.00 | 1.600 |
<!-- /worked -->

The second factor has a history. The first version of the weight was $1 + \text{reached}/\text{top}$, a range of exactly two, which in the first column of the table is the "old rule" (scaled by the best cell): a cell that had achieved nothing was drawn half as often as the best one. Commit `b02e79a35` (22 September 2026) replaced it. Its message reports, from one long campaign of a game-level sample (a measurement this book does not reproduce), that two runs held their best score while their archives grew from $649$ cells to $1028$: new places were found all the time and the frontier did not advance, which the message puts down to a twofold preference spreading almost the whole budget over ground already covered. At the default focus of $4.0$ a cell at half the best's achievement is drawn about a seventh as often and one at a quarter about a twentieth, the "1 in 7.4" and "1 in 20.1" of the table; the weight never reaches zero, because a stepping stone that looks worthless is what an archive like this exists to keep. The test of that commit, `selection_strongly_prefers_what_has_achieved_most` in `crates/search/tests/it/archive.rs`, runs the flat weighting (`focused(0.0)`) beside the sharp one on the same archive as a control, so that it demonstrates a difference instead of asserting a number that some other weighting might also clear. The edge factor came a day later (commit `fd8ac3114`) and gives a cell with no neighbours $1.6$ times the weight of one that is hemmed in on every side.

When the archive is full, it evicts the cell of the lowest weight (the first two factors; the edge factor plays no part) and reuses its host slot, a handle on the engine's saved state, for the new cell. The whole archive serialises as a sorted list of elites, not as a map, because JSON has no key type but the string and a niche is a tuple of integers (commented at the field `cells`).

The history of the `generation` counter is the lesson of this crate, and it is a lesson about the structure of a quality-diversity archive. A cell's contents are replaced whenever a better way of reaching it turns up, and that is the mechanism by which the archive improves. A campaign had stored each cell's trajectory as a chain, "resume at cell $P$, then do these steps", because whole trajectories duplicated hundreds of megabytes of prefix. A chain is valid only against the version of $P$ that was in the cell when it was recorded: improve $P$ and every trajectory that hangs off it reads as the new prefix followed by steps that continued a different state, a sequence that describes nothing that ever happened and looks exactly like one that did. Commit `f080ffef2` (22 September 2026) added the counter that is bumped on a replacement and checked at use (`Elite::generation`, `knowledge #147`). The next day commit `7895e71cc` (23 September 2026) found that a counter on improvement was not enough: the archive also evicts cells, and an eviction invalidates the chains below it as completely as an improvement does, so the design "cannot work, structurally", in the commit's words, and its sample moved to trajectories that hold the whole way from the start of the level. The counter stays in the crate and its doc says it is for callers that chain; the sample that motivated it no longer does. The general rule of `knowledge #147`, that anything that stores "do $X$ relative to $Y$" while $Y$ can change needs a version on $Y$, checked at use, is as true as it was, and the second commit adds its own corollary: where the structure cannot be versioned, store the absolute thing.

## Three bandits, one ladder: `crates/search` allocates and verifies

`crates/search/src/allocate.rs` answers a different question: a campaign has several operators that can produce a candidate (return to an archived cell and perturb it, hunt a specific unmet goal, refine an elite locally, recombine two), and which of them is worth the next unit of budget changes as the archive fills. The crate's answer is UCB1, the bandit rule that gives the UCT of this chapter its name: choose the arm with the largest $\text{rate}/\text{top} + \sqrt{2}\,\sqrt{\ln(\text{rounds})/\text{draws}}$, where the rate is gain per unit of cost, normalised by the best arm's rate so that the exploration term is on a comparable scale. Every arm is tried once, in order, before any is repeated. The table runs the rule on three operators with fixed value and cost per call.

<!-- worked: search_planning.allocator_rounds -->
UCB1 over three search operators, as in `Allocator::choose`. Every call of an operator earns the same fixed value and costs the same fixed amount (the crate measures its cost in seconds of the clock; here the units are abstract): perturb: value 2 per call, cost 1 per call, rate 2; hunt: value 4 per call, cost 4 per call, rate 1; refine: value 0.5 per call, cost 2 per call, rate 0.25. Rule: every arm is tried once, in order; after that the arm with the largest rate / top + sqrt(2) sqrt(ln(rounds) / draws) is chosen, where rate is the arm's total value over its total cost, top the largest rate among the arms and rounds the number of calls made so far (a tie to the earlier arm). No randomness.

The first 12 choices: perturb hunt refine perturb perturb hunt perturb refine perturb hunt perturb perturb.

| calls made | perturb | hunt | refine |
|---|---:|---:|---:|
| 3 | 1 | 1 | 1 |
| 6 | 3 | 2 | 1 |
| 12 | 7 | 3 | 2 |
| 30 | 20 | 6 | 4 |
| 100 | 80 | 13 | 7 |
| 300 | 267 | 23 | 10 |
<!-- /worked -->

The operator with the best rate takes $267$ of $300$ calls and the worst is still called $10$ times, which is the property wanted: an operator that looks poor is not written off, because the rule's exploration term keeps growing with the number of rounds until it earns a call. The crate measures the cost in seconds of the clock, a choice it explains (two operators that find equally much are not equally good if one takes twenty times as long); the book's table uses abstract units so that its numbers do not depend on a machine.

The gain deserves a note, because a first version had a defect that the commit message describes clearly. The allocator counted every fresh niche as one, so the operator worth the budget was whichever could manufacture the most cells, which is not the same operator as the one advancing the search. Commit `74b03c457` (22 September 2026) reports, from the same campaign, that an operator that pressed against walls produced about twice the volume of cheap cells that the operator that played the level did (in the commit's count, $44\,513$ steps against $23\,272$), scored a comparable rate on a per-count metric and took a third of the budget, while the playing operator was the one reaching new parts of the level. `Gain::admitted` now credits an admission by its depth, the ratio of `reached` to the best the archive held before it, clamped into $[0.05, 1]$ (the floor is not zero, so that an operator that only produces shallow cells does not rate exactly zero), and a fresh niche is worth four times an improved one, a constant that the code calls a judgement and not a measurement. Credit by depth is the archive's own lesson applied to the allocator: a count of cells rewards diversity of the wrong kind.

The third module, `crates/search/src/cascade.rs`, is the verifier as a ladder of rungs. A `Rung` checks a candidate and returns `Pass` or `Reject(reason)`; a `Cascade` runs the rungs in the order they were added, cheapest first, and a candidate that a cheap rung refuses is never shown to an expensive one. The whole point is that the expensive rung is a replay of an entire episode against the real environment, and running it on candidates that a free check could have refused would waste the budget. The first test of the module, `a_candidate_refused_cheaply_never_reaches_the_expensive_rung`, asserts exactly this negative property, and the cascade keeps a tally of how many candidates each rung saw and refused, because a rung that refuses everything for one reason is reporting a defect, not filtering. An empty cascade admits, so that a caller who has not configured a verifier gets an unverified search instead of one that silently finds nothing.

## Walking away from the goal: `crates/solve`

`crates/solve` is the opposite choice, and it was introduced the same week by commit `199078b34` (23 September 2026) as a second, unrelated way to drive a puzzle. Where a search spends expansions at every decision, `crates/solve` spends none at decision time; its work is done in advance, in generating training data. The method needs one property of the problem: the moves are invertible. Suppose a space has a goal and a move $m$ with an inverse $m^{-1}$. Walk away from the goal by $m_1, m_2, \dots, m_d$; the state reached after $d$ moves is one at which the correct move, the one that undoes the last step, is $m_d^{-1}$. The label is exact, it is free, it needs no planner and no reward, and it is available at any depth: this is the same trick as hindsight, with the roles reversed. The walk that was generated without regard to success becomes a set of examples of how to succeed, by reading it backwards from the goal it started at.

The trait `StateSpace` (`crates/solve/src/lib.rs`) is what a problem supplies: the number of moves, the goal, `apply`, `inverse`, a feature vector for a state (a one-hot) and `redundant(a, b)`, which says whether playing $b$ straight after $a$ wastes walk length (undoing the last move is always redundant and is the default). The generator (`crates/solve/src/data.rs`, function `batch`) harvests every state of a walk, not only the endpoint, and keeps an equal quota of examples for each walk depth $1 \dots D$. The arithmetic of the quota is exact: for `rows` examples, each depth gets `rows // depth` of them and the remainder is added to the deepest depths, one each, from the deepest down, so that for $10$ rows at depth $4$ the quotas are $2, 2, 3, 3$. What the quota prevents is a skew that harvesting whole walks would introduce, since every walk passes through depth $1$ and only the end of a walk reaches depth $D$. The table runs the generator on a puzzle small enough to solve exactly, five tokens in a row whose moves swap neighbours, with a table in place of the network.

<!-- worked: search_planning.retraced_labels -->
Retraced labels on a permutation puzzle: 5 tokens in a row, goal (0, 1, 2, 3, 4), four moves, move m swaps the tokens at positions m and m + 1 (every move is its own inverse, so the label of a state reached by move m is m). The puzzle has 120 states and diameter 10 (breadth-first search from the goal). The generator, as in `crates/solve`: a walk of D moves from the goal in which a move equal to the one just played is redrawn (`rng.integer(4)` until it differs), every state of the walk labelled with the move that undoes the last move, each walk depth 1 .. D receiving its quota of 2000 // D rows with the remainder added to the deepest depths, overflow dropped. 2000 rows per depth, every depth restarting the book's generator `search_planning.retrace` seed 0x529b5eb604ff3547 (5952455468361135431). The policy is the table: for every labelled state the label that most rows give it (a tie to the smaller move). 'moves per example' is the number of moves applied to generate the rows divided by the rows, against the (D + 1) / 2 of a generator that keeps one endpoint per walk of a uniformly drawn length 1 .. D; 'states labelled' is the number of distinct states in the rows; 'shortest labels' counts the labelled non-goal states whose chosen move reaches a state one move nearer the goal; 'solved' counts the 119 non-goal states from which following the table reaches the goal in at most 24 moves (a state with no label stops the run), with the mean number of moves of those runs and the mean shortest distance of the same states.

| D | moves per example | endpoint-only generator | states labelled | shortest labels | solved | moves of a run | shortest |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1.00 | 1.5 | 13 | 13 of 13 | 13 | 1.69 | 1.69 |
| 4 | 1.00 | 2.5 | 49 | 48 of 48 | 48 | 3.06 | 3.06 |
| 6 | 1.00 | 3.5 | 91 | 90 of 90 | 90 | 4.19 | 4.19 |
| 8 | 1.00 | 4.5 | 112 | 109 of 111 | 111 | 4.84 | 4.80 |
| 10 | 1.00 | 5.5 | 120 | 112 of 119 | 119 | 5.28 | 5.04 |
| 12 | 1.00 | 6.5 | 119 | 107 of 118 | 108 | 4.96 | 4.70 |
| 16 | 1.00 | 8.5 | 120 | 105 of 119 | 113 | 5.58 | 4.91 |
| 24 | 1.01 | 12.5 | 120 | 91 of 119 | 112 | 6.91 | 5.02 |
<!-- /worked -->

```text
function retrace_batch(depth: int, rows: int, rng: Generator)
        -> (examples: list<(array<int>[5], int, int)>, applied: int):
    require depth >= 1 and rows >= 1
    quota = [rows // depth for d in 0 .. depth]
    for i in 0 .. rows mod depth:
        quota[depth - 1 - i] = quota[depth - 1 - i] + 1
    filled = [0 for d in 0 .. depth]
    examples = empty list of (array<int>[5], int, int)
    applied = 0
    while len(examples) < rows:
        s = [0, 1, 2, 3, 4]
        last = -1
        for step in 0 .. depth:
            loop:
                m = rng.integer(4)
                if m != last:
                    break
            swap s[m], s[m + 1]
            applied = applied + 1
            last = m
            if filled[step] < quota[step]:
                append (copy(s), m, step + 1) to examples
                filled[step] = filled[step] + 1
    return (examples, applied)
```

Each swap is its own inverse, so the label of a state reached by move $m$ is $m$, and each example keeps a copy of the row, since the walk goes on swapping it. Two claims of the crate can be read off the table. The first column of numbers is $1.00$ moves applied per example for depths up to $16$, which is what `crates/solve/tests/it/generator.rs` asserts of the crate's generator (`a_harvested_walk_costs_one_move_application_per_example`, under $1.2$), against the $(D + 1)/2$ of the previous generator that kept one endpoint per walk of a uniformly drawn length: $13.5$ at the depth $26$ of that test, and $12.5$ at depth $24$ here. Commit `f9a49d5ed` (23 September 2026) made the change, and the test asserts the mechanism, moves per example, because a clock is a poor witness on a loaded machine.

The second claim is the cost the crate's header states plainly: retraced labels are not optimal labels. A walk of length $L$ that revisits ground gives a label that leads home the long way, so the policy learns to solve, not to solve shortest, and past the diameter of the space (it is $10$ for this puzzle) the labels become noisy. In the table the number of labelled states whose label is a shortest move falls from $112$ of $119$ at $D = 10$ to $91$ of $119$ at $D = 24$, while the average length of a solved run rises from $5.28$ against a shortest $5.04$ to $6.91$ against $5.02$. At $D = 10$, the diameter, the table solves every state; for $D$ below the diameter it solves only the states it has seen, and for $D$ above it solves fewer states in the table's run ($108$ at $D = 12$ and $112$ at $D = 24$), by runs that lie further above the shortest ones. Hence the crate's advice that `Walk::depth` is a real hyperparameter, to be set near the diameter and not above it.

What replaces the table in the crate is a residual multilayer perceptron (`crates/solve/src/config.rs`): a projection from the one-hot to the trunk width, `blocks` residual blocks that each go up to `d_ff`, through a GELU, and down to `d_model`, a head with one logit per move, and a value head. The initialisation (`crates/solve/src/net.rs`, `init_weights`) draws normal weights scaled by $\sqrt{2/\text{fan-in}}$, with the last projection of each block scaled by $1/\sqrt{\text{blocks}}$ so that a deep residual stream does not grow before training starts, and biases zero. The input is a one-hot, so the first matrix product is a table lookup, and the network is the table of the toy with the capacity to share across states. Gradients are checked against finite differences at two sizes (`crates/solve/tests/it/gradcheck.rs`), because the register-tiled matrix kernels are selected only above a size that the small check never reaches.

At decision time `crates/solve/src/rollout.rs` plays the policy by taking its largest probability, one forward pass per move for a whole batch of instances in lockstep, with an option that masks the moves the generator would have called redundant. An instance that has not reached the goal within the step cap is reported stuck and counted against the policy. Sampling at a temperature, with several independent attempts from the original state, turns one sample from the policy into its distribution: attempts at rate $p$ succeed with probability $1 - (1 - p)^k$ in $k$ tries, so a policy that solves a tenth of the time solves about $65\%$ in ten, which is the rule the crate states.

Three stories came out of this crate, each with a lesson that applies far beyond it.

The first is an output that no check ever read. Commit `841ffea28` (23 September 2026) fixed a dispatch of the policy softmax that asked for `rows` threads when the kernel has $64$ threads cooperate on a row, so that only the first few rows of a batch were ever written and the rest held whatever was in the buffer. Training reads the logits (the cross-entropy kernels never touch the softmax), so the loss descended correctly the whole time; the finite-difference checks read the loss and passed for the same reason. Only inference read the probabilities, and a rolled-out policy looked as though it had learned almost nothing: the commit message reports one measurement, not reproduced here: the same checkpoint scored $16\%$ at a scramble of $4$ before the fix and $98\%$ once every row was written. The code now asks for `Dispatch::Workgroups(n)`, with a comment explaining why, and a test (`every_row_of_a_batch_gets_its_own_softmax`) feeds identical rows and requires identical distributions out, which fails on any row the kernel never reached. The lesson is that a check passes for exactly the paths it executes; the path that the user will take has to be one of them.

The second is a measured negative result recorded where the next reader will find it. Commit `3951f4063` (23 September 2026) added a value head beside the policy, $V(s)$, the estimated cost to go, with a rollout that chooses the successor with the lowest predicted cost to go (`solve_batch_by_value`) instead of naming a move blind. It trains the head on bootstrapped targets, $V(s) = \min_a (1 + V_{\rm frozen}(s'))$ over the successors with a frozen copy of the network (the sample that trains it, `samples/decision/rubiks/src/learned.rs`, computes exactly this), because regressing on the walk length would teach how far the generator wandered, an upper bound that says almost nothing about which of two neighbouring states is nearer. The commit message reports, as one measurement that the book does not reproduce, that after $3000$ training steps the greedy policy solved $25\%$ of eight-move cube scrambles and the value lookahead $12\%$: "it does not pay yet", and the message gives the reason that value iteration spreads about one move of distance per refresh, so fifteen refreshes cannot give a correct value at distance eighteen. It is recorded as evidence that the method needs a compute budget that had not been spent, not as evidence against the method. The first attempt at checking the value gradient also passed while the value path was disconnected, because the targets made the squared error fifty times the cross-entropy and the finite difference could not see the policy term; the fixture's targets are now small enough that both terms show in the same number.

The third is about what a rollout may be called. `crates/solve/src/rollout.rs` plays the policy in two ways, both greedy and neither reconsidering a move: `solve_batch` plays the policy's largest probability, and `solve_batch_by_value` evaluates the value head on every legal successor and plays the one with the lowest predicted cost to go. The second is one step of the lookahead of the first section of this chapter with a learned value at the leaves. It has no frontier and no backtracking, so it is the smallest search there is, depth one with the learned value as the leaf, and the distinction between it and the policy rollout is the one this chapter has been drawing, between learning a decision and searching for it.

## Check your understanding

1. In the weighted rows of the search table, A\* with $f = g + 2h$ expands only $19$ cells and returns cost $18$, and with $f = g + h$ it expands $38$ and returns cost $14$. Explain both facts, using admissibility, and say what the cheaper search gave up.
2. A colleague proposes $c = 0$ for UCT because "the best-looking child is the one to follow". Using the trace after iteration $3$ (children with $W/N = 0, 1, 1$), say what the search does from then on and why that can lock onto a wrong move. What does the budget table show for a constant of $0.1$?
3. Find the error: "MAP-Elites found the narrow peak because it uses more evaluations than the hill climber."
4. A learned heuristic overestimates the cost to go in $46$ of the $79$ cells and A\* still returned a path of cost $14$ in every row. A reviewer says this shows the learned heuristic is admissible in practice. What is wrong with the argument, and what would give the guarantee?
5. After two rounds of expert iteration the policy alone is right on all ten won piles, and no search is run at decision time. Which of the playouts that trained it are still paid for at decision time, and what would make a policy of this kind insufficient on a larger game?
6. Hindsight relabelling turned $26$ rewarded transitions into $672$ or $5119$. Why is a relabelled transition a correct example for the learner, and what would go wrong if the learner were an on-policy method such as the one of Chapter 3?
7. In `crates/solve` the label of a state reached by a walk of $L$ moves is the inverse of its last move. When is that label not a move toward the goal along a shortest path, and why does the crate's advice to set the walk depth near the diameter follow?
8. The `generation` of an archive cell is bumped on replacement and not on a refused offer. Why not on a refusal, and why did commit `7895e71cc` conclude that the counter was still not enough for chained trajectories?

## Summary

Search replaces what a table of values or a policy would have stored by a computation done at the moment of the decision, from a model of the world, a simulator or a checker. A lookahead tree is value iteration without sharing; a frontier ordered by cost so far, by a heuristic or by their sum is the family of breadth-first, uniform-cost, greedy and A\* search, whose work is counted in expansions, and a heuristic that never overestimates, and for the closed-set form one that is also consistent, makes the answer cost-optimal. A learned value can be that heuristic and cut the expansions of the map from $72$ to $17$, but it keeps its guarantee only where it keeps its promise. When the tree is too wide, Monte Carlo tree search grows only the part that playouts say matters, with the exploration constant deciding how the visits are spread; a policy can be the prior and the playout, and fitting the policy to the search's visit counts gives a loop in which each makes the other better. Quality-diversity search keeps the best of every kind of candidate, which is what lets it reach a narrow peak that a search for the best candidate cannot, and hindsight relabelling turns episodes that missed their goal into lessons about the goals they reached. In brain, `crates/search` is the archive, the bandit over operators and the ladder of verifiers that discovery needs, and `crates/solve` is the opposite choice, labels from walks away from the goal and a policy that needs no search at decision time. The next chapter leaves the problem of finding solutions and asks the question of the small learned models inside an application, which make fast decisions that have to be auditable and calibrated.

## References

- E. W. Dijkstra, "A Note on Two Problems in Connexion with Graphs", 1959.
- P. E. Hart, N. J. Nilsson, B. Raphael, "A Formal Basis for the Heuristic Determination of Minimum Cost Paths", 1968.
- P. Auer, N. Cesa-Bianchi, P. Fischer, "Finite-time Analysis of the Multiarmed Bandit Problem", 2002.
- L. Kocsis, C. Szepesvari, "Bandit Based Monte-Carlo Planning", 2006.
- D. Silver et al., "Mastering the game of Go with deep neural networks and tree search", 2016.
- T. Anthony, Z. Tian, D. Barber, "Thinking Fast and Slow with Deep Learning and Tree Search", 2017.
- J.-B. Mouret, J. Clune, "Illuminating search spaces by mapping elites", 2015.
- A. Ecoffet, J. Huizinga, J. Lehman, K. O. Stanley, J. Clune, "First return, then explore", 2021.
- T. Schaul, D. Horgan, K. Gregor, D. Silver, "Universal Value Function Approximators", 2015.
- M. Andrychowicz et al., "Hindsight Experience Replay", 2017.
- F. Agostinelli, S. McAleer, A. Shmakov, P. Baldi, "Solving the Rubik's cube with deep reinforcement learning and search", 2019.

## Exercises

1. **[Calculate]** From the tables of Chapter 1, count the nodes of the unshared two-day lookahead tree from stock $0$: the root, its children (an order and a next stock with positive probability), and their children, and check the total against the first table of the chapter.
2. **[Calculate]** A node of the take-away tree has been visited $10$ times; its children have $(W, N) = (6, 7)$ and $(1, 3)$. Compute the UCT scores for $c = \sqrt{2}$ and say which child is selected. Then give the smallest $c$ for which the other child would be.
3. **[Calculate]** For the walker of the MAP-Elites toy, give the cell of the genome $(0.9, 0.65, 0.85, 0.9)$ on the $10 \times 10$ grid, and the cell of the genome $(1, 1, 1, 1)$. Say which rule of the grid puts the second where it is.
4. **[Calculate]** An episode of $4$ moves visits the states $3, 4, 5, 5, 6$ and had the goal $9$. Count its transitions with reward $1$ under the original goal, under the final strategy, and under the future strategy with $k = 1$ when the draws of $kk$ for $t = 0, 1, 2, 3$ are $2, 4, 3, 4$.
5. **[Calculate]** For $10$ rows at walk depth $4$, give the per-depth quotas of the generator of `crates/solve`, the smallest number of walks that fills them, and the moves applied per example. Repeat for $26$ rows at depth $26$.
6. **[Investigate]** On the map of this chapter let the mud cost $m$ instead of $3$. For which $m$ does breadth-first search return a cheapest path, and what does the search table become at $m = 1$? Check your answer by running the routine.
7. **[Implement]** Reproduce `search_planning.grid_map`, `search_planning.search_methods` and `search_planning.expansion_orders`, including the order in which neighbours are generated, the counter that breaks ties in the queue, the test of the goal on a pop and the skipping of a closed cell.
8. **[Implement]** Reproduce `search_planning.take_away_game`, `search_planning.uct_trace` and `search_planning.uct_visits` with the book's generator, one draw per playout move in the order the trace states.
9. **[Implement]** Reproduce `search_planning.quality_diversity_results` and `search_planning.quality_diversity_archive`, including the cell rule, the replacement rule and the order of the draws of the three searches from the one generator.
10. **[Implement]** Reproduce `search_planning.hindsight_relabelling` and `search_planning.retraced_labels`, and then `search_planning.archive_offers` and `search_planning.allocator_rounds` from the rules of `crates/search`.
11. **[Investigate]** In `search_planning.expert_iteration` replace the playouts by uniform random ones but keep the priors. Does the policy still reach ten of ten after two rounds? What does that say about which of the two uses of the policy, prior or playout, matters on this game?

## Key terms

- **state-space search**: finding a path from a start to a goal by keeping a frontier of states that have been reached and repeatedly expanding one of them, with the order of the frontier (fewest moves, lowest cost so far, nearest by a heuristic, or the sum of cost so far and heuristic) deciding which method it is and how many expansions it spends. <!-- concept: state-space-search -->
- **quality-diversity archive**: a store that keeps, for every cell of a space of behaviour descriptors, the best candidate found so far, so that many different good solutions are kept and a candidate that scores low but does something new survives as a starting point for further search. <!-- concept: quality-diversity-archive -->
- **goal-conditioned policy**: a policy that takes the goal as an input beside the state, so that one function serves every goal. <!-- concept: goal-conditioned-policy -->
- **hindsight relabelling**: replacing the goal of a stored transition by a state that its episode actually reached and recomputing the reward, so that an episode that missed its goal becomes a rewarded example for the goal it did reach. <!-- concept: hindsight-relabeling -->

## Answers

**Recall questions.**

1. With every value $0$, $q(0, a) = R(0, a)$, which is $0$, $0$, $-1.5$ and $-4.5$ for $a = 0, 1, 2, 3$. The two largest are equal, so the tie rule (the smallest action within $10^{-9}$ of the best) chooses $a = 0$.
2. A policy is a rule that gives an action in each state; deterministic, it is one action per state. The shop has $4 \times 3 \times 2 \times 1 = 24$ of them, since the stocks $0, 1, 2, 3$ have $4, 3, 2, 1$ legal orders.
3. $X$ has mean $p$ and variance $p(1 - p)$. The average of $N$ independent copies has variance $p(1 - p)/N$ and standard error $\sqrt{p(1 - p)/N}$, which halves when $N$ is multiplied by $4$.
4. The hidden layer has $2 \times 12$ weights and $12$ biases and the output has $12$ weights and one bias: $24 + 12 + 12 + 1 = 49$.

**Check your understanding.**

1. The Manhattan distance is admissible and consistent, so $f = g + h$ expands the cells in an order that makes the first pop of the goal a cheapest path: cost $14$ at $38$ expansions. With $2h$ the heuristic overestimates the cost to go (along the cheapest route $2h(S) = 24$ against a true $14$), so the search is pulled toward the straight way across the mud, whose true cost it undercounts; it expands fewer cells and returns the path of $12$ moves and cost $18$. It gave up the guarantee of the cheapest path, for fewer expansions.
2. With $c = 0$ the score is $W/N$ alone. After iteration $3$ the children of moves $2$ and $3$ have $W/N = 1$ and move $1$ has $0$, so selection follows move $2$ (the smaller on the tie) as long as its ratio stays at $1$; a child that wins its first playouts by chance keeps the visits and the others are never revisited to show that the child is wrong, so the search can lock on a wrong move. The budget table shows the effect at $c = 0.1$: $56$ of $100$ correct at $800$ iterations, against $100$ for $c = 0.5$ and $c = \sqrt{2}$.
3. Every search of the table used the same number of evaluations, $2000$. What differs is what each search keeps: the hill climber and the best-100 population keep the best candidates only, which loses the stepping stones across the flat valley, and MAP-Elites keeps the best of every cell, which keeps them.
4. The cost of the path found is an outcome of one search on one map, and an overestimating heuristic can return a cheapest path by luck, as here; admissibility is a property of $h$ at every cell, and $46$ cells violate it. The guarantee comes from a heuristic that is admissible (and, for the closed-set routine, consistent): subtracting the largest excess on the cells, as the table does, gives admissibility on the cells that were checked, at the cost of more expansions ($21$ against $17$ at $3000$ steps).
5. None: after training, the policy is a table that is read, and the playouts were paid during training. On a larger game the table of preferences is replaced by a network with limited capacity, and a policy that imitates a search can be right on positions the search fixed and wrong on positions it did not visit, so the stronger systems keep searching at decision time and use the policy as the prior.
6. The dynamics of the environment do not depend on the goal, so a transition $(s, a, s')$ observed with one goal is a sample of what $a$ does in $s$ whatever the goal; only the reward changes, and it is recomputed by the same rule. Q-learning learns the value of the greedy policy from transitions generated by any behaviour, so it can use the relabelled ones. An on-policy method assumes the actions came from the policy being improved with the goal it is conditioned on, which a relabelled transition violates, and it would need an importance correction.
7. When the walk revisits states: a state reached by $L$ moves may be nearer than $L$, and the inverse of the last move leads back along the walk, which can be the long way home. Beyond the diameter every extra move is redundant, so the label names a move that is arbitrary among many. Setting the depth near the diameter gives labels that are short enough to be nearly shortest and long enough to cover the whole space.
8. A refused offer changes nothing, so invalidating the chains below the cell would be a self-inflicted wound. The counter was not enough because the archive also evicts cells to free a slot, and an eviction invalidates the chains below it as completely as a replacement does; the design had to store absolute trajectories instead.

**Exercises** (where a number is asked).

1. The root has $9$ children: the orders $a = 0, 1, 2, 3$ give the next stocks $\{0\}$, $\{0, 1\}$, $\{0, 1, 2\}$ and $\{1, 2, 3\}$, which are $1 + 2 + 3 + 3 = 9$ pairs. A node of stock $s$ has as many children as the pairs of an order and a next stock with positive probability: $9, 8, 6, 3$ for $s = 0, 1, 2, 3$. The two-day tree has $1 + 9 + (9 + (9 + 8) + (9 + 8 + 6) + (8 + 6 + 3)) = 1 + 9 + 66 = 76$ nodes.
2. The scores are $6/7 + \sqrt{2}\sqrt{\ln 10 / 7} = 1.6682$ and $1/3 + \sqrt{2}\sqrt{\ln 10 / 3} = 1.5723$, so the first child is selected. The second would be selected when $1/3 + c \times 0.8761 > 6/7 + c \times 0.5735$, that is $c > (6/7 - 1/3)/(0.8761 - 0.5735) = 0.5238/0.3026 = 1.731$.
3. The first genome: $\mathbf{s}_1 = (0.4, 0.15)$, $\mathbf{s}_2 = (0.35, 0.4)$, $\mathbf{p} = (0.75, 0.55)$, $\mathbf{d} = (0.875, 0.775)$, cell $(8, 7)$. The second: $\mathbf{p} = (1, 1)$, $\mathbf{d} = (1, 1)$ and $\lfloor 10 \times 1 \rfloor = 10$ is clamped to $9$, cell $(9, 9)$, by the rule that the last cell includes the coordinate $1$.
4. Under the original goal $9$: $0$ rewards. Under the final strategy the goal is $6$, reached by the move of $t = 3$ only: $1$. Under the future strategy with draws $2, 4, 3, 4$ the goals are $\text{traj}[2] = 5$, $\text{traj}[4] = 6$, $\text{traj}[3] = 5$, $\text{traj}[4] = 6$; the next states are $4, 5, 5, 6$, so the moves of $t = 2$ (next state $5$, goal $5$) and $t = 3$ (next state $6$, goal $6$) have reward $1$: $2$.
5. Ten rows at depth $4$: `rows // depth` is $2$ and the remainder $2$ goes to the two deepest depths, so the quotas for depths $1$ to $4$ are $2, 2, 3, 3$. Three walks fill depth $3$ and $4$ (one example per depth per walk), the walks of $4$ moves apply $12$ moves, and $12 / 10 = 1.2$ per example. Twenty-six rows at depth $26$: a quota of $1$ per depth, one walk, $26$ moves, $1.0$ per example.
6. A cheapest path through the mud costs $9 + 3m$ against $14$ around it, equal at $m = 5/3$. Breadth-first search returns a cheapest path for $m \le 5/3$ (at $m = 1$ the map has no mud and all the searches agree on cost $12$).

Exercises 7 to 11 are checked against the printed tables and the stated generators.
