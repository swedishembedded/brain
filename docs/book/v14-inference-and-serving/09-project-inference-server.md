<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Project: A Multi-Client Inference Server

*Can you build a server that answers many clients at once within fixed memory and stays safe under hostile input?*

Ten requests reach a small language-model server within its first thirteen steps, and the server answers every one of them. Its replies arrive, they have the right lengths, nothing raises an error, and when each request is sent again on its own, with the server otherwise empty, the reply comes back exactly as the request should be answered. Run the ten together again, and two of the ten replies are not the replies those requests get alone. The server in the second row below has one changed line: when it stops a sequence to make room and later brings it back, it forgets to empty the list of memory blocks the sequence used to hold, so the sequence comes back reading blocks that now belong to its neighbours. No test of one request at a time can see it, because a request alone is never stopped. The third row is the server with a different change, an admission rule that counts only the blocks of the prompt and no way of stopping a sequence once it runs, and it stops in step 7, its eighth, with nine requests unanswered.

<!-- worked: project_inference_server.opening -->
The ten requests of the workload below, served three ways. The server has a pool of 6 blocks of 4 positions, carries at most 4 sequences in a step, ends a reply after the token 0 or after max_new tokens, and admits by the rule of the text. 'Equal to alone' counts the replies that are token for token the reply of the same request served by itself on the dense cache of Volume V; 'ledger' is whether, after every step, the free blocks and the blocks in the tables of the running sequences are together each of the pool's blocks exactly once.

| build | replies returned | equal to alone | ledger holds | how the run ends |
|---|---:|---:|---|---|
| the specified server | 10 of 10 | 10 of 10 | yes | every request answered or refused after 21 steps |
| a preempted sequence keeps its old block table | 10 of 10 | 8 of 10 | no, from step 10 | every request answered or refused after 18 steps |
| admission counts only the prompt's blocks and nothing is preempted | 1 of 10 | 1 of 10 | yes | stopped in step 7: no free block for a decode row |
<!-- /worked -->

A server for many clients is held to two promises at once. It must give every client the reply that client would get with the server to itself, whatever else is running, and it must do so inside a memory it fixed before the first request arrived, whatever the clients send. Neither promise is visible from one request. The first breaks only when sequences share memory and are moved around in it, and the second only when somebody sends what no honest client would. So the build below carries its own referees: a reply computed for every request alone, on a plain cache, which every reply of the server must equal token for token; a ledger of the memory that must balance after every step; and, for the boundary, the answer each message must get. You assemble the server in seven steps, in any language: the model and its reference replies, the paged cache, the step of continuous batching, preemption and the rebuild of a sequence's cache, the admission rule and what it costs, the checks at the boundary, and the bounds on the waiting queue; a last run puts them all under load at once. Every step ends in numbers that your run must reproduce, and each comes with a planted mistake whose numbers you can produce on purpose to see the referee catch it.

![A left-to-right diagram of the server. A message from a client goes to the checks at the boundary; the first failing check sends back a refusal with its reason. An accepted request joins a waiting queue bounded per client and in total, then admission checks that the prompt and the next positions of the running sequences fit in the free blocks, and the step prefills what was admitted and runs one row for every running sequence, reading and writing the pool of blocks through the block tables. A finished sequence returns its reply and gives its blocks back. A preempted sequence gives its blocks back and goes to the head of the waiting queue.](img/server-system.svg)

The pieces come from the volume. The cache is kept in blocks of a shared pool and found through a table per sequence ([Managing the KV Cache](03-paged-kv.md)), the batch is rebuilt at every step so that finished sequences leave and waiting ones join ([Batching and Scheduling Requests](04-batching.md)), a request is admitted only when the memory it needs is there ([Batching and Scheduling Requests](04-batching.md)), and every quantity a client controls is bounded before anything is spent on it ([Security of Machine Learning Services](07-security.md)). The model is the smallest one the book has, so that every reply can be computed twice and compared. The module that produced every number below, `scripts/book/worked/project_inference_server.py`, imports the model and the routine `extend` of Volume V, Chapter 12 (`kv_cache.model`, `kv_cache.extend`) and writes the server around them. You need 64-bit floating point, the book's generator of [Volume I, Chapter 17](../v01-mathematics/17-randomness.md) with its `normal` and `integer`, and your own implementation of `extend` from [Volume V, Chapter 12](../v05-sequences/12-kv-cache.md). Five questions bring back what the steps lean on.

1. (Volume V, Chapter 12, the KV cache.) A cache holds the keys and values of $c$ positions, and one new token is run through the model. At which position does its key go, how many keys does its query read, and why may the cache grow by one row while every earlier row stays as it is?
2. ([Chapter 3](03-paged-kv.md), the paged KV cache.) A sequence holds 13 positions in blocks of 4. How many blocks does its table list, which entry of the table and which slot of that block hold position 9, and why can a block that one sequence gives back serve any other sequence next?
3. ([Chapter 4](04-batching.md), continuous batching.) Between two steps of a continuously batched server, what may change in the batch, and why does adding one more decoding sequence to a step add almost nothing to the bytes of weights the step reads?
4. ([Chapter 4](04-batching.md), admission control.) A waiting request's prompt fits in the free blocks. Why is that not enough to admit it, and what else must an admission rule count?
5. ([Chapter 7](07-security.md), input bounds.) A message declares that a million tokens follow and carries three. Why must the bound on the declared length be checked before anything is allocated from it, and what is the ratio of what the server would set aside to what the sender spent?

## The model, the requests and the replies they must get

Every reply of the project is a greedy continuation computed by one model, the tiny GPT that [Volume V, Chapter 12](../v05-sequences/12-kv-cache.md) used to show what a cache saves: $V = 12$ token ids, width $D = 16$, two heads of width 8, $L = 2$ pre-norm transformer blocks (called layers below, so that the word block is kept for the cache) with a SwiGLU network of hidden width 24, at most $T = 24$ positions, learned absolute positions, RMS norms and a head tied to the embedding table. Its 5,008 numbers are drawn from the book's SplitMix64 at the seed `seed_of('kv_cache.model')` = 0xe354ebf228c667c0, every one `1.0 * normal()`, row by row in the order $E$ $[V, D]$, $P$ $[T, D]$, then for each layer $W_Q$, $W_K$, $W_V$, $W_O$ ($D$ by $D$), $W_{\mathrm{gate}}$, $W_{\mathrm{up}}$ ($D$ by 24) and $W_{\mathrm{down}}$ (24 by $D$), with $W_O$ and $W_{\mathrm{down}}$ multiplied by $1/\sqrt{2L}$; all gains are 1. That chapter gives the forward pass and the routine `extend`, which runs $n$ new tokens through the model on top of a cache that holds $c$ earlier ones and returns the logits of the last new token. Nothing is trained. Greedy decoding takes the largest logit, the lowest id on a tie.

A request is a prompt and a budget. Its reply is the tokens the model chooses one after another, ending after the first token 0 (the stop id, which is part of the reply) or after $m$ tokens, $m$ being the budget, whichever comes first. Each request also carries the client that sent it and the step at which it arrives. The ten requests of the workload are drawn from one generator, and the table prints both the rule and the values, together with the reply each must receive.

<!-- worked: project_inference_server.requests -->
Generator: the book's SplitMix64, seed 2029128649351549169 (`seed_of('project_inference_server.workload')`), one generator for the whole workload, never restarted. Draws, request by request for requests 0 to 9: the gap to the arrival of the previous request, integer(3) (not drawn for request 0, which arrives at step 0); the client, integer(3); the prompt length n = 3 + integer(6); the n tokens, integer(12) each, in order; the budget max_new = 6 + integer(11). 'Reply alone' is the greedy reply of the request served by itself with the dense cache of Volume V (`generate` of its Chapter 12 on the model of seed `seed_of('kv_cache.model')`), cut after the first token 0 or after max_new tokens. 'Gap' is the smallest difference, over the tokens of that reply, between the largest and the second-largest logit, printed to 3 decimals; the smallest over the workload is 0.064.

| request | arrives at step | client | prompt | max_new | reply alone | gap |
|---:|---:|---:|---|---:|---|---:|
| 0 | 0 | 2 | 4 0 0 3 5 8 0 10 | 8 | 4 2 6 0 | 1.754 |
| 1 | 1 | 2 | 1 4 8 8 9 6 | 10 | 8 7 9 1 9 4 4 4 4 4 | 0.464 |
| 2 | 3 | 0 | 1 4 4 11 8 | 12 | 7 9 10 2 9 8 8 4 4 2 4 4 | 0.082 |
| 3 | 5 | 2 | 5 10 7 3 3 1 | 16 | 6 2 4 0 | 0.064 |
| 4 | 6 | 0 | 9 8 7 11 5 4 | 10 | 0 | 2.579 |
| 5 | 7 | 1 | 6 0 6 1 | 6 | 8 0 | 0.106 |
| 6 | 7 | 0 | 5 7 6 2 6 3 6 | 6 | 9 6 10 6 9 6 | 0.564 |
| 7 | 8 | 1 | 9 4 0 4 5 | 15 | 0 | 1.894 |
| 8 | 10 | 2 | 5 7 1 4 6 2 8 0 | 9 | 7 2 6 0 | 1.108 |
| 9 | 12 | 0 | 0 4 7 8 10 3 11 8 | 10 | 7 0 | 4.058 |
<!-- /worked -->

The reference reply of a request is the reply it gets alone, with the dense cache of Volume V: one array of keys and one of values per layer, row $j$ holding position $j$. Every later step compares the server's reply with it, token for token, so the reference has to be something another implementation reproduces exactly. A token is an argmax, and an argmax changes only when the best and the second-best logit trade places. The smallest such gap over all the tokens of all ten replies is 0.064 (the gap column; request 3's reply sets it), while two implementations that add the same numbers in different orders differ only in the last few bits of a 64-bit logit, far below $10^{-6}$ for numbers of this size. No reordering of sums that an honest implementation might choose can move a token of the table, and a reply that differs from it is a different computation, not a rounding.

Your server passes step 1 when your `extend` on the model of seed 0xe354ebf228c667c0 gives the ten replies of the table. A wrong reply of request 4 or 7 (each a single 0) points at the prompt's prefill; a reply that is right for a few tokens and then wrong points at the position of the decoded token, the error that Volume V traced to the length of the cache read after it has grown.

## Blocks, a free stack and a table for each sequence

The server's memory for caches is a pool of $N_{\mathrm{pool}} = 6$ blocks, each with room for $B_{\mathrm{blk}} = 4$ positions: for each of the two layers, four rows of keys and four rows of values of 16 numbers. A sequence does not own a region of the pool; it owns a block table, here a plain list of block ids in which entry $i$ names the block that holds the sequence's positions $iB_{\mathrm{blk}}$ to $(i + 1)B_{\mathrm{blk}} - 1$. Position $j$ of a sequence is slot $j \bmod B_{\mathrm{blk}}$ of block $\mathrm{table}[\lfloor j/B_{\mathrm{blk}} \rfloor]$, and a sequence of $c$ positions holds $\lceil c/B_{\mathrm{blk}} \rceil$ blocks. When it has filled the last slot of its last block, the next position needs a new block.

The free blocks are kept on a stack. It starts as the ids 5, 4, 3, 2, 1, 0 with 0 on top, so the first block handed out is block 0; a block given back is pushed on top, and the block given back last is the first handed out again. A sequence that finishes, or is stopped to make room, gives back the blocks of its table in table order, so that its last block ends up on top. None of these orders changes a reply, since a block holds the same numbers wherever it is, but they decide which block ids appear in the tables below, and your run must use the same orders to reproduce them.

```text
record Pool:
    keys: array<real>[L, NBLK, BLK, D]      # NBLK blocks of BLK positions; zero at the start
    values: array<real>[L, NBLK, BLK, D]
    free: list<int>                      # the top of the stack is the last element

function new_pool() -> Pool
    return Pool(zeros(real, [L, NBLK, BLK, D]), zeros(real, [L, NBLK, BLK, D]), [NBLK - 1 - i for i in 0 .. NBLK])

function blocks_for(c: int) -> int       # blocks that hold c positions
    return (c + BLK - 1) // BLK

function take_block(pool: inout Pool) -> int
    require len(pool.free) > 0, "no free block"
    return take_last(pool.free)

function give_back(pool: inout Pool, table: list<int>) -> none
    for b in table:
        append b to pool.free
```

The cache that `extend` of Volume V writes and reads becomes, for a sequence, its table and a count of the positions it holds. Nothing else in `extend` changes: the rows are computed exactly as before, in the same order, and only the addresses where they live are different.

```text
record Seq:
    id: int
    client: int
    arrival: int
    prompt: list<int>
    max_new: int
    table: list<int>          # block ids; logical block i is table[i]
    length: int               # positions stored, 0 .. length - 1
    generated: list<int>      # the reply so far
    resumed: bool             # true while it waits, or is prefilled, after a preemption

function paged_extend(model, pool: inout Pool, s: inout Seq, ids: list<int>) -> array<real>[V]
    # `extend` of Volume V, Chapter 12, with its cache replaced by the blocks of s:
    #   the write  cache.keys[l, c + i, :] = K[i, :]  becomes
    #              pool.keys[l, s.table[(c + i) // BLK], (c + i) mod BLK, :] = K[i, :]
    #   the read   cache.keys[l, j, lo:lo + d_k]  becomes
    #              pool.keys[l, s.table[j // BLK], j mod BLK, lo:lo + d_k]
    #   the values alike, and cache.length is s.length (c on entry, c + len(ids) on return)
    require len(ids) >= 1, "a prefill of no rows"
    require s.length + len(ids) <= T, "a position past the model's T"
    require len(s.table) >= blocks_for(s.length + len(ids)), "the blocks must be in the table before rows are written"
```

The slot of the row written for position $c + i$ is computed from $c + i$, the count before the new row is counted. That is the rule of the dense cache said once more, and it is where the planted mistake of this step goes: a server that advances its count first and then takes the slot from it writes position 0 into slot 1, position 1 into slot 2, position 3 into slot 0 of the same block, and reads them back from the slots where they belong. The table below serves each request alone on a fresh server through the paged cache, and compares the reply and every logit with the dense cache.

<!-- worked: project_inference_server.paged_alone -->
Each request of the workload served by itself on the paged cache (a fresh server, nothing else in it), against its reply on the dense cache. The largest difference is over every logit of every token up to the first token on which the two replies differ (all of them when the replies are equal).

| block table | replies equal to the dense cache | requests that differ | largest difference of a logit |
|---|---:|---|---|
| the specified block table | 10 of 10 | none | 0 (the same bits) |
| a new key and value written at the slot of the count after it advances | 2 of 10 | 1, 2, 3, 4, 5, 6, 7, 8 | 9.7 |
<!-- /worked -->

With the specified table every one of the 10 replies is equal, and every logit of every token is the same 64-bit number as on the dense cache: the rows are the same arithmetic, stored elsewhere. The mistake leaves only 2 replies right, those of requests 0 and 9, and moves a logit by 9.7 before the first wrong token, the size of a change in what attention reads, not of a rounding. It shows up with one request alone, so this is a mistake the first comparison of the project already catches; the mistakes of the next steps are not.

The memory of the server is fixed by now, and it is worth writing down before any request arrives, because every later step must stay inside it.

<!-- worked: project_inference_server.memory_bill -->
What the server holds, counted in numbers and in bytes. Cache and weights are 64-bit floats (8 bytes); block ids, tokens and the fields of a message are 32-bit integers (4 bytes). Replies being built and the bookkeeping of each sequence (its counters, its budget) are left out: they are bounded by the same limits, T tokens per sequence and 4 running sequences.

| what | numbers | bytes | rule |
|---|---:|---:|---|
| weights | 5,008 | 40,064 | every number of the model, 8 bytes each |
| one position of the cache | 64 | 512 | a key and a value of D numbers in each of L layers |
| one block | 256 | 2,048 | 4 positions |
| the pool | 1,536 | 12,288 | 6 blocks |
| block tables | 24 | 96 | at most 6 block ids for each of 4 running sequences, 4 bytes each |
| waiting messages | 416 | 1,664 | at most 16 messages of at most 104 bytes |
<!-- /worked -->

One position of cache costs 512 bytes and the pool 12,288; the weights, at 40,064 bytes, are the largest item. The waiting queue enters the bill because its size is bounded too, by two limits that step 7 justifies. Nothing a client sends can change any row of this table: the pool is the pool, a sequence holds at most 6 block ids since no sequence may hold more than $T = 24$ positions, and a message that would make the queue longer than its bound is refused, not stored.

Your server passes step 2 when the ten replies through the paged cache equal the dense ones, with every logit the same bits, and when your copy of the planted mistake gets exactly requests 0 and 9 right.

## One step of a server that serves many

The server runs in steps. A step is one pass of the model over a batch of rows, and the batch is built anew at every step, which is what makes the batching continuous: a sequence that finishes leaves at the end of a step, and a waiting request joins at the start of the next. Within a step the work happens in a fixed order, which your server must follow to reproduce the trace.

First the requests of the step arrive, in order of their ids, and are checked (the checks are step 6; the honest workload passes them all). Then requests are admitted from the head of the waiting queue for as long as the admission rule allows and fewer than 4 sequences are running. An admitted request is prefilled at once, its whole prompt in one call of `paged_extend`, and the logits of the prompt's last position give its first token. Then, if the pool cannot back one more position for every running sequence, the newest running sequences are stopped until it can (the next two sections are about this line). Then every running sequence, the ones admitted in this step included, decodes one token: its last token is fed through `paged_extend` at the next position, after a block is taken from the stack if its table is full. Last, the sequences whose replies are complete give their blocks back, in the order in which they run.

The admission rule is the heart of it. A waiting request with a context of $n$ tokens needs $\lceil (n + 1)/B_{\mathrm{blk}} \rceil$ blocks: its prompt, and the slot of its first new position, into which its first token is written when it decodes in the same step. The rule also counts what the sequences already running have been promised, the blocks each of them must take to store its next position, and admits the newcomer only if the free blocks cover both:

$$\text{admit} \iff \text{free} \ \ge\ \Bigl\lceil \frac{n + 1}{B_{\mathrm{blk}}} \Bigr\rceil + \sum_{s\ \text{running}} \Bigl( \Bigl\lceil \frac{c_s + 1}{B_{\mathrm{blk}}} \Bigr\rceil - |\mathrm{table}_s| \Bigr),$$

where $c_s$ is the number of positions sequence $s$ stores and $|\mathrm{table}_s|$ the number of blocks it holds. The sum is the number of running sequences whose last block is full; each of them takes one block in this step's decode.

```text
function promised(running: list<Seq>) -> int    # blocks the running sequences take to store one more position each
    return sum([blocks_for(s.length + 1) - len(s.table) for s in running])

function context(s: Seq) -> list<int>           # what a prefill puts into the cache
    k = len(s.generated)
    if k == 0:
        return s.prompt
    return concat(s.prompt, s.generated[0 : k - 1])   # the prompt and all of the reply but its last token

function complete(s: Seq) -> bool
    k = len(s.generated)
    return k > 0 and (s.generated[k - 1] == STOP or k == s.max_new)

function retire(pool: inout Pool, s: inout Seq) -> none       # give every block back and empty the table
    give_back(pool, s.table)
    s.table = []
    s.length = 0

function serve_step(model, pool: inout Pool, waiting: inout list<Seq>, running: inout list<Seq>,
                    replies: inout list<(int, list<int>)>) -> none
    # 1. admission and prefill, from the head of the queue
    while len(waiting) > 0 and len(running) < ROWS:
        ctx = context(waiting[0])
        if len(pool.free) < blocks_for(len(ctx) + 1) + promised(running):
            break
        s = take_first(waiting)
        for i in 0 .. blocks_for(len(ctx)):
            append take_block(pool) to s.table
        z = paged_extend(model, pool, s, ctx)
        if not s.resumed:
            append argmax(z) to s.generated            # the first token of the reply
        s.resumed = false
        if complete(s):
            retire(pool, s)
            append (s.id, s.generated) to replies
        else:
            append s to running
    # 2. room for this step's decode: stop the newest, never the last one running
    while len(pool.free) < promised(running) and len(running) > 1:
        v = take_last(running)
        retire(pool, v)
        v.resumed = true
        push_front v to waiting
    # 3. one decode row for every running sequence, in order
    for i in 0 .. len(running):
        if blocks_for(running[i].length + 1) > len(running[i].table):
            append take_block(pool) to running[i].table
        k = len(running[i].generated)
        z = paged_extend(model, pool, running[i], [running[i].generated[k - 1]])
        append argmax(z) to running[i].generated
    # 4. finished sequences give their blocks back, in running order
    still = []
    for i in 0 .. len(running):
        if complete(running[i]):
            retire(pool, running[i])
            append (running[i].id, running[i].generated) to replies
        else:
            append running[i] to still
    running = still
```

A request that arrives at step $t$ is checked and appended to the tail of `waiting` before `serve_step` runs for step $t$, and the steps run until no request is waiting or running and none is still to arrive. The trace of the specified server on the workload is the third table your run must reproduce.

<!-- worked: project_inference_server.trace -->
The specified server on the workload, step by step. In a step, in this order: the requests of the step arrive and are validated; waiting requests are admitted from the head of the queue while the rule allows, each prefilled at once ('first token' for a new request, 'rebuilt' for a preempted one, whose prefill only refills its cache); the newest running sequences are preempted until the pool can back one more position for every running sequence; every running sequence decodes one token; finished sequences give back their blocks. 'Rows' is the number of token rows the step runs through the model (prompt and rebuilt rows plus one per decoding sequence); 'running' and 'free blocks' are after the step.

| step | arrive | first token | rebuilt | preempted | finished | rows | running | free blocks |
|---:|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | - | - | - | 9 | 0 | 3 |
| 1 | 1 | 1 | - | - | - | 8 | 0 1 | 1 |
| 2 | - | - | - | - | 0 | 2 | 1 | 4 |
| 3 | 2 | 2 | - | - | - | 7 | 1 2 | 1 |
| 4 | - | - | - | - | - | 2 | 1 2 | 1 |
| 5 | 3 | - | - | - | - | 2 | 1 2 | 1 |
| 6 | 4 | - | - | - | - | 2 | 1 2 | 0 |
| 7 | 5 6 | - | - | 2 | - | 1 | 1 | 2 |
| 8 | 7 | - | - | - | - | 1 | 1 | 2 |
| 9 | - | - | - | - | 1 | 1 | - | 6 |
| 10 | 8 | 3 | 2 | - | - | 17 | 2 3 | 1 |
| 11 | - | - | - | - | - | 2 | 2 3 | 1 |
| 12 | 9 | - | - | - | 3 | 2 | 2 | 3 |
| 13 | - | 4 5 | - | - | 4 5 | 12 | 2 | 2 |
| 14 | - | 6 | - | - | - | 9 | 2 6 | 0 |
| 15 | - | - | - | 6 | - | 1 | 2 | 2 |
| 16 | - | - | - | - | 2 | 1 | - | 6 |
| 17 | - | 7 8 | 6 | - | 7 | 23 | 6 8 | 0 |
| 18 | - | - | - | - | - | 2 | 6 8 | 0 |
| 19 | - | - | - | - | 8 | 2 | 6 | 3 |
| 20 | - | 9 | - | - | 6 9 | 10 | - | 6 |
<!-- /worked -->

Three moments of the trace carry the ideas of the step. At step 1 request 1 is admitted beside request 0, which holds 3 blocks: request 1's prompt of 6 tokens and its first new position need 2 blocks, request 0 has filled 9 positions of its 12 and is promised nothing, so 3 free blocks cover 2 and both decode in that step, 8 rows in all (6 of prompt, 2 of decode). At step 5 request 3 arrives and is not admitted: its 6-token prompt needs 2 blocks and one is free. At step 7 the pool is empty and request 1 has filled all 12 slots of its 3 blocks, so its decode needs a block that does not exist; request 2, the newer of the two running sequences, is stopped and its 3 blocks go back to the stack, block 5 last and so on top; request 1 takes block 5 and goes on, and requests 5 and 6, which arrive in that step, wait. The figure shows the same run as the blocks see it.

![A grid with the six blocks of the pool as rows and the 21 steps as columns; each cell holds the id of the request whose table lists that block after the step, or is blank when the block is free. Request 0 holds blocks 0 to 2 for steps 0 and 1; request 1 holds blocks 3 and 4 from step 1, then 0 from step 3 and 5 from step 7, until it finishes at step 9, when every block is free. Request 2 holds blocks 2 and 1 from step 3 and 5 at step 6, is preempted at step 7, and holds blocks 5, 0 and 4 from step 10 and 2 from step 13 until step 15. Request 3 holds blocks at steps 10 and 11, request 6 at step 14 and at steps 17 to 19, request 8 at steps 17 and 18.](img/block-occupancy.svg)

Every column of the figure is a balance: the blocks in the tables of the running sequences and the blocks on the free stack are together each of the 6 ids exactly once. The project calls this the ledger and checks it after every step. It is cheap (a sort of 6 numbers) and it is the second referee: a server that lets one block into two tables, or into a table and the stack at once, breaks it at the step where it happens, before any reply is wrong.

Your server passes step 3 when its trace equals the table row for row: the same arrivals, first tokens, rebuilds, preemptions and completions in the same steps, the same rows and the same free blocks after every step, and the ledger balanced throughout. The first row where your trace differs names the part of the step to look at: a different 'first token' column points at admission, a different 'free blocks' at the order of the free stack or of giving back, a different 'rows' at the context of a rebuild.

## Why a decode never finds the pool empty

A decode row needs a free slot, and when its sequence's last block is full it needs a free block. If the stack is empty at that moment the server has nowhere to put the row. It cannot skip the sequence, since its reply would then depend on what else is running, and it cannot write the row anywhere else. The specified server never reaches that moment, and the reason is short enough to state as an argument rather than a hope.

The guarantee comes from the loop that runs just before every decode. If the free blocks do not cover what the running sequences have been promised, the blocks they must take to store their next positions, the loop stops the newest sequence and pushes its blocks onto the stack, which raises the free count by what that sequence held and lowers the promises by what it was owed. The loop ends in one of two states. Either the free blocks cover the promises, and every decode of the step finds its block; or one sequence is left. A sequence holds at most $\lceil (n + m - 1)/B_{\mathrm{blk}} \rceil$ blocks over its whole life, $n$ being its prompt and $m$ its budget (it stores every token but its last), and step 6 refuses any request for which this is more than the pool. So a sequence alone, with every other block free, always has the block it needs. In both states the decode finds room.

The promise in the admission rule does a different job. It keeps a newcomer out of the blocks that this very step's decode will need, which would only get the newcomer stopped in the same step, after its prefill has been paid for. The cost table of a later section has a row for a rule that counts the newcomer's prompt and first position but not the promises of the others: it gives the same ten replies, but at step 13 it admits request 6 and stops it again before its first decode, and it rebuilds more.

Without the loop, nothing guarantees anything. The third row of the opening table is a server whose admission counts only the blocks of the prompt, $\lceil n/B_{\mathrm{blk}} \rceil$, and which never stops a sequence. On this workload it admits the same requests at the same steps as the specified server until step 7, where the specified server stops request 2. The planted server has no such move: request 1's decode needs a block, the stack is empty, and the server stops with every sequence in flight lost, one reply returned of ten. It is the failure that brain's scheduler had before commit `a7d14c830` (19 October 2026), whose message says that admission "only proved a prompt fit, so admitted sequences could outgrow the KV pool together and crash the decode step when host offload was off or its budget spent".

The second row of the opening table breaks the ledger instead. A stopped sequence must give back its blocks and empty its table; the planted server gives back the blocks and keeps the list. When request 2 is rebuilt at step 10, its table lists its old ids 2, 1 and 5 and then the three blocks it takes for the rebuild, 5, 0 and 4: block 5 twice, and block 1, which request 3 was handed in the same step, in two tables at once. The ledger breaks at step 10, the step of the rebuild, and request 2 reads and writes its positions through blocks it does not own. When it finishes it gives its table back, duplicates included, and the stack holds 9 entries for a pool of 6 blocks; in step 13 request 8 is handed block 5 while request 5 still holds it. The replies of requests 2 and 8 are the two that are wrong, and every request still gets an answer. A server checked only by the shape of its outputs never finds this; the replies alone find the two wrong answers, and the ledger finds the step.

## Stopping a sequence and giving it back its cache

Stopping a running sequence to free its blocks is preemption, and it raises the question that the rest of the step depends on: what the sequence must carry with it so that, when it comes back, it continues exactly where it stopped. A stopped sequence keeps its prompt and its reply so far and loses its cache. When it reaches the head of the queue again and is admitted, its cache is rebuilt by a prefill of its context: the prompt followed by every generated token but the last. The last generated token is not in the cache, because it has not been fed yet; it is the token the next decode feeds, exactly as it would have been without the interruption. This is preemption by recompute, and it costs a prefill of the context, which the cost table below counts.

The rebuilt cache holds the same numbers as the lost one, not merely similar ones. Row $j$ of the cache is a function of the weights and of the tokens at positions 0 to $j$ (the statement that Volume V, Chapter 12 proved for the causal mask), and the arithmetic that computes a row in a prefill of many rows is, row for row, the arithmetic that computed it in a decode of one: every sum adds the same terms in the same order. The next token is then the token of the uninterrupted run, and so is every token after it. The comparison against the replies alone checks this end to end.

<!-- worked: project_inference_server.preemption -->
The preemptions of the specified server ('tokens' counts the tokens the sequence had generated when it was preempted; the rebuild prefills the prompt and all of them but the last, which is fed by the next decode step), and three ways of rebuilding a preempted sequence, each on the whole workload. 'Replies that differ' prints the reply the build returned; the replies alone are in the workload table.

| request | preempted in step | tokens | rebuilt in step | rows rebuilt |
|---:|---:|---:|---:|---|
| 2 | 7 | 5 | 10 | 5 + 4 = 9 |
| 6 | 15 | 2 | 17 | 7 + 1 = 8 |

| rebuild | preemptions | equal to alone | replies that differ |
|---|---:|---:|---|
| the specified rebuild | 2 | 10 of 10 | none |
| a preempted sequence is rebuilt from its prompt alone | 2 | 8 of 10 | 2: 7 9 10 2 9 7 7 9 3 7 9 10; 6: 9 6 0 |
| a preempted sequence is rebuilt with its pending token | 4 | 7 of 10 | 2: 7 9 10 2 9 9 6 4 2 4 4 9; 3: 6 2 4 10 4 6 6 5 6 1 2 8 4 4 2 4; 6: 9 6 10 11 4 2 |
<!-- /worked -->

Request 2 had produced 5 tokens when it was stopped at step 7; at step 10 its context of 5 prompt tokens and 4 generated ones, 9 rows, is prefilled, and its sixth token is decoded from its fifth in the same step, as if nothing had happened. Request 6 is stopped at step 15 with 2 tokens and rebuilt at step 17 from 8 rows. Both replies are equal to their replies alone.

The two planted rebuilds fail in opposite directions. Rebuilding from the prompt alone drops the generated tokens from the cache, so the decode feeds the pending token at the position right after the prompt and the reply continues from the wrong point: 8 replies of 10 are right, and request 2's reply is the right one for 5 tokens and then turns into 7 7 9 3 7 9 10. Rebuilding with the pending token included writes that token into the cache, and then the decode feeds it a second time, so the sequence reads one token twice: 7 replies of 10 are right. That build also stops sequences more often (4 preemptions instead of 2), since its rebuilt sequences hold one position more than they should, and request 3, which the specified server never stops, comes back wrong. Both keep the ledger balanced: every block is where it should be, and what is written into the blocks is wrong.

Your server passes step 4 when its preemption table equals the first table above and its three rebuilds give the second: all ten replies equal to their replies alone for the specified rebuild, and the printed replies, token for token, for the two mistakes. Your copies of the two planted servers of the opening table must end as their rows say: the kept table with 8 replies right and the ledger broken from step 10, the admission without preemption stopped in step 7.

The equality holds here because the reference arithmetic computes each row on its own. A device kernel that tiles its matrix products may add a row's terms in an order that depends on how many rows the call carries, and then a rebuilt cache can differ from the original in the last bits. brain met exactly this: the message of commit `354ba0f1a` (28 July 2026) records that the CPU backend's blocked matrix products are not invariant in the last bit to the number of rows, so that token equality across different ways of cutting a prompt "is not a meaningful contract" for that backend, and the prefix cache it introduced is tested against a tolerance on the hidden states instead. The gap column of step 1 is what makes the project's comparison exact: no token here is near a tie, so a difference in the last bits cannot reach a token.

## What each way of admitting costs in work

The promise rule admits a request into the blocks it needs now, and pays for the bet with an occasional rebuild when sequences grow together. The obvious alternative bets nothing: it reserves, at admission, every block the sequence could ever need, $\lceil (n + m - 1)/B_{\mathrm{blk}} \rceil$, and then no sequence can ever be short of a block and nothing is preempted. A third server runs one sequence at a time and needs no rule at all. Which is better is a question of work, and the work of a step is countable. A step that runs at least one row is one forward pass, and a forward pass reads the 4,544 numbers of the model's matrix products once, 36,352 bytes, however many rows it carries: the rows of a step are one batch, the assumption under which batching pays at all (Volume V, Chapter 12 counted that a decode row reads the weights once per step for the whole batch and its own cache once). The FLOPs follow the counting rule of that chapter.

<!-- worked: project_inference_server.admission_rules -->
Five admission rules on the workload, each with the same pool of 6 blocks. A forward pass is a step that runs at least one row; it reads the 4,544 numbers of the matrix products once, 36,352 bytes, whatever its number of rows (the rows of a step are one batch). FLOPs are counted as in Volume V, Chapter 12: two per multiply-add of every matrix product, the attention's included, and the head for the last row of each prefill and for each decode row; norms, softmax and the gate are not counted. 'Rebuilt' is the FLOPs of the prefills of preempted sequences. 'Wait' is the mean over the ten requests of the steps from arrival to the step of the last token.

| admission | forward passes | weight bytes | FLOPs, first pass | FLOPs rebuilt | preemptions | peak blocks | wait (steps) |
|---|---:|---:|---:|---:|---:|---:|---:|
| one sequence at a time | 36 | 1,308,672 | 957,056 | 0 | 0 | 4 | 18.5 |
| reserve the whole budget at admission | 35 | 1,272,320 | 957,056 | 0 | 0 | 6 | 18.1 |
| the prompt and its first position, preempt the newest | 21 | 763,392 | 957,056 | 224,000 | 3 | 6 | 8.2 |
| promise the next position, preempt the oldest | 23 | 836,096 | 957,056 | 250,496 | 2 | 6 | 10.4 |
| promise the next position, preempt the newest | 21 | 763,392 | 957,056 | 159,104 | 2 | 6 | 8.2 |
<!-- /worked -->

All five servers do the same first-pass arithmetic, 957,056 FLOPs: the same prompts and the same replies, since every one of them answers every request correctly. They differ in how many forward passes it takes and in the arithmetic they do twice. One sequence at a time takes 36 forward passes and reads 1,308,672 bytes of weights. The reservation rule is hardly better, 35 passes: the requests of this workload have budgets of 6 to 16 tokens, a reservation of $\lceil (n + m - 1)/B_{\mathrm{blk}} \rceil$ blocks is 3 to 6 blocks of a pool of 6, and a second sequence rarely finds room beside the first. Most of the reserved blocks are never used, because most replies end at a 0 long before their budget (request 3 reserves 6 blocks for a budget of 16 and its reply is 4 tokens); the reservation is sized for a ceiling and the ceiling is rarely reached. The promise rule takes 21 passes and 763,392 bytes, 58 percent of the bytes of serving one at a time, at the price of 159,104 FLOPs of rebuilt prefills, 17 percent of the first-pass arithmetic. The mean request finishes 8.2 steps after it arrives instead of 18.5. Counting only the newcomer's prompt and first position, without the promises of the running sequences, gives the same 21 passes and costs 224,000 FLOPs of rebuilds in 3 preemptions, one of them the stop of request 6 in the step that admitted it.

The victim matters too. Stopping the oldest sequence instead of the newest gives the same replies and costs more: 250,496 FLOPs of rebuilds and 23 passes. The oldest running sequence holds the longest context, so its rebuild is the longest prefill, and it is also the one nearest the end of its reply, the one that would have given its blocks back soonest if it had been let run. The newest sequence holds the least and has the most still to do.

Your server passes step 5 when its five rows equal the table's.

The comparison is a count of passes, bytes and FLOPs on a model of 5,008 numbers, where the bytes of a forward pass are tiny; it says which rule does less work on this workload and by how much, and it says nothing about the time a device takes. The count's premise carries to a large model only while the bytes of the weights a pass reads exceed the bytes of cache it reads, which fails for long contexts and large batches; the count of passes and of rebuilt FLOPs is the part that does not depend on the model's size.

## The checks a message meets before anything is spent on it

Everything so far assumed honest requests. A server open to many clients receives messages, and a message is a claim: it says how many tokens follow, what they are and how many tokens to generate, and any of these can be false, or true and ruinous. The defence is a list of checks run on every message, in a fixed order, before the message is queued or any memory is set aside for it; the first check that fails is the answer, sent back with its reason.

The message format of the project is a list of signed 32-bit fields: the budget $m$, the declared prompt length $n$, and then the tokens. Its size is bounded before anything else, at $4(2 + T) = 104$ bytes, the largest message an honest client can need, so that no field is read from a message the server has not yet decided to read. The client is not a field: it is the identity of the connection the message came on, so that a client cannot claim to be another.

```text
function check_message(msg: list<int>, NBLK: int, waiting_of_client: int, waiting_total: int) -> text
    if 4 * len(msg) > 4 * (2 + T):
        return "too large"
    if len(msg) < 2:
        return "malformed"
    m = msg[0]
    n = msg[1]
    toks = msg[2 : len(msg)]
    if n != len(toks):
        return "length mismatch"
    if n < 1:
        return "empty prompt"
    for t in toks:
        if t < 0 or t >= V:
            return "token out of range"
    if m < 1:
        return "max_new below 1"
    if n + m - 1 > min(T, NBLK * BLK):
        return "exceeds capacity"
    if waiting_of_client >= Q_CLIENT:
        return "client queue full"
    if waiting_total >= Q_TOTAL:
        return "queue full"
    return "accepted"
```

Client 4 sends ten messages, one of them honest. The table gives each the specified server's answer.

<!-- worked: project_inference_server.messages -->
Client 4's messages, each checked by the specified server with an empty queue. A message is a list of 32-bit fields: max_new, the declared prompt length n, then the tokens (printed as 'max_new; n; tokens'). The checks run in this order and the first that fails is the answer: the message is at most 104 bytes (2 + T fields); it has at least 2 fields; n equals the number of tokens; n is at least 1; every token is in 0 .. 11; max_new is at least 1; n + max_new - 1 is at most the capacity, min(T, blocks x 4) = 24 positions; the client has fewer than 4 requests waiting; fewer than 16 requests are waiting in all.

| message | fields | bytes | answer |
|---|---|---:|---|
| a token id past the vocabulary | 8; 3; 7 7 12 | 20 | token out of range |
| a negative token id | 8; 3; 7 -1 7 | 20 | token out of range |
| a budget of no tokens | 0; 3; 7 7 7 | 20 | max_new below 1 |
| a budget of a billion tokens | 1,000,000,000; 3; 7 7 7 | 20 | exceeds capacity |
| a declared length of a million | 8; 1,000,000; 7 7 7 | 20 | length mismatch |
| a prompt of no tokens | 8; 0; none | 8 | empty prompt |
| one field | 8 | 4 | malformed |
| 25 tokens | 8; 25; 25 times 2 | 108 | too large |
| 20 tokens and a budget of 8 | 8; 20; 20 times 2 | 88 | exceeds capacity |
| a well-formed request | 8; 3; 7 7 7 | 20 | accepted |
<!-- /worked -->

The capacity check is the one that ties the boundary to the argument that a decode never finds the pool empty: $n + m - 1$ positions is the most a sequence will ever store, and it must fit both the model's $T$ and the whole pool, $\min(24, 6 \times 4) = 24$ positions here. The message of 20 tokens and a budget of 8 is well-formed, its tokens are valid and its prompt fits, and it is still refused, because its life does not. The order of the checks matters for the reason reported and, more, for what is read: the length is compared with the bytes before a token is looked at, and the size before the length.

Each check stands between a message and a specific failure, and the failures can be produced, one check at a time, by removing it. The next table adds one of client 4's messages to the honest workload, at its step, on a server that lacks the check that refuses it.

<!-- worked: project_inference_server.unguarded -->
The honest workload with one of client 4's messages added at its step, on the specified server with one check removed. 'Honest replies' counts the replies of requests 0 to 9 that are equal to their replies alone. A server that stops takes every reply in flight with it. The reply to the negative id is the reply to the prompt 7 11 7: this module is written in Python, whose list read at index -1 is the last entry, so the row read is row 11 of the embedding table; an implementation in a language without that rule reads memory before the table.

| check removed | message | honest replies | what happens |
|---|---|---:|---|
| token out of range | a token id past the vocabulary | 0 of 10 | stopped in step 1: a token id outside the embedding table |
| token out of range | a negative token id | 10 of 10 | answered: 3 9 7 2 7 0 |
| max_new below 1 | a budget of no tokens | 2 of 10 | stopped in step 26: no free block for a decode row |
| exceeds capacity | a budget of a billion tokens | 2 of 10 | stopped in step 26: no free block for a decode row |
| exceeds capacity | 20 tokens and a budget of 8 | 4 of 10 | stopped in step 21: no free block for a decode row |
| empty prompt | a prompt of no tokens | 1 of 10 | stopped in step 3: a prefill of no rows |
<!-- /worked -->

A token id of 12 sends the embedding lookup past the last row of a 12-row table; the module's language raises an error there, the server stops in the step of the prefill, and the replies of all ten honest requests are lost with it. The negative id is worse because it does not stop anything: the reply comes back, 3 9 7 2 7 0, and it is the reply to another prompt. The budget of no tokens and the budget of a billion end alike, since a reply is complete when it holds $m$ tokens, and 0 is never reached by counting up while a billion is never reached at all; the prompt 7 7 7 is followed by 3 3 3 and never by a 0, so the sequence grows until it is alone in the pool and asks for a seventh block. The 20-token prompt with a budget of 8 is admitted when the pool is empty and stops the server when its sequence needs a seventh block in step 21. Every one of these is a whole server lost to one message of a few bytes.

The check of the declared length guards against another kind of loss, one that the server of the project cannot suffer because it never reads a length it has not compared with the bytes. A server that set aside memory from the fields of a message before checking them would be spending on the sender's word, and the ratio is the measure of the danger.

<!-- worked: project_inference_server.amplification -->
What a server would set aside if it sized a sequence's cache from the fields of the message instead of checking them: the declared positions (n, or n + max_new - 1 for a budget), the blocks of 4 positions they need and their bytes at 512 bytes per position, against the bytes of the message.

| message | message bytes | positions | blocks | cache bytes |
|---|---:|---:|---:|---:|
| a budget of a billion tokens | 20 | 1,000,000,002 | 250,000,001 | 512,000,001,024 |
| a declared length of a million | 20 | 1,000,000 | 250,000 | 512,000,000 |
<!-- /worked -->

A message of 20 bytes that declares a million tokens would have such a server set aside 512,000,000 bytes of cache, 25.6 million times what the sender spent; the budget of a billion would ask for 512 gigabytes. Both are refused by checks that cost a comparison each.

Your server passes step 6 when its answers to the ten messages are those of the table, in the same order of checks, and when, with one check removed at a time, its runs end as the third table says.

## A flood from one client and the bounds on the queue

The checks so far judge each message on its own. A client can also send many messages that are each perfectly honest, and the damage is then done by the number. Client 3 sends 40 well-formed requests at step 0, just after request 0. A server whose queue has no bound accepts them all, and since the queue is first in, first out, every honest request that arrives afterwards waits behind client 3's; the queue itself grows to 44 entries, each a message the server must hold. A bound on the total number of waiting requests fixes the memory but not the unfairness: once client 3's requests fill it, the honest ones are refused. The bound that protects the other clients is one per client.

<!-- worked: project_inference_server.flood_table -->
Client 3 sends 40 well-formed requests at step 0, just after request 0 (generator: the book's SplitMix64, seed 11948608382804451597, `seed_of('project_inference_server.flood')`, one generator, never restarted; each request drawn as an honest one is, the prompt length 3 + integer(6), its tokens integer(12), max_new 6 + integer(11); the first is 5 2 1 7 6 5 8 with max_new 13). 'Honest replies' counts the replies of requests 0 to 9 equal to their replies alone, 'honest refused' those refused at arrival, 'peak waiting' the most requests waiting at once and 'wait' the mean over the answered honest requests of the steps from arrival to the last token.

| queue | honest replies | honest refused | flood accepted | peak waiting | wait (steps) | steps |
|---|---:|---:|---:|---:|---:|---:|
| no flood | 10 of 10 | 0 | - | 7 | 8.2 | 21 |
| flood, both queue bounds | 10 of 10 | 0 | 4 of 40 | 8 | 19.0 | 33 |
| flood, the total bound alone | 7 of 10 | 3 | 15 of 40 | 16 | 47.6 | 67 |
| flood, no queue bound | 10 of 10 | 0 | 40 of 40 | 44 | 100.2 | 123 |
<!-- /worked -->

With both bounds, client 3 gets 4 of its 40 requests into the queue and 36 are refused with "client queue full", a refusal it may retry later; the honest replies are all right, none is refused, and the mean honest request waits 19.0 steps from arrival to its last token, against 8.2 with no flood. The difference is the work of client 3's four accepted requests, which are served like anyone else's. With the total bound alone the queue never holds more than 16, but 3 honest requests are refused and the others wait 47.6 steps on average. With no bound the server holds 44 waiting messages, answers everything correctly and makes the mean honest request wait 100.2 steps, with 123 steps for the run. The queue bound is admission control at the door, before a request is even waiting: the block rule of the step decides when a request runs, and the queue bounds decide whether it may wait at all.

Your server passes step 7 when the four rows come out as printed: the honest replies, the refusals, the peak of waiting requests and the mean waits.

## The whole server under honest and hostile load

The last run puts everything together: the honest workload, client 3's flood at step 0 and client 4's ten messages at their steps, against the specified server and against each of the planted mistakes of the earlier steps. The refusals of the specified server are the reference for the boundary, the replies alone for the replies, and the ledger for the memory.

<!-- worked: project_inference_server.whole_run -->
Everything at once: the ten honest requests, client 3's flood of 40 at step 0 and client 4's ten messages at their steps (the message table gives the step of each), on the specified server and with each planted mistake. The specified server refuses 45 messages (too large 1; malformed 1; length mismatch 1; empty prompt 1; token out of range 2; max_new below 1 1; exceeds capacity 2; client queue full 36), answers 4 requests of the flood and 1 of client 4's, and its peak of waiting requests is 9. 'Refusals as specified' counts the refusals that a build gives with the same reason as the specified server.

| build | honest replies | refusals as specified | ledger holds | how the run ends |
|---|---:|---:|---|---|
| the specified server | 10 of 10 | 45 of 45 | yes | every request answered or refused after 35 steps |
| a new key and value written at the slot of the count after it advances | 3 of 10 | 45 of 45 | yes | every request answered or refused after 22 steps |
| a preempted sequence keeps its old block table | 6 of 10 | 45 of 45 | no, from step 15 | every request answered or refused after 34 steps |
| admission counts only the prompt's blocks and nothing is preempted | 0 of 10 | 36 of 45 | yes | stopped in step 0: no free block for a decode row |
| a preempted sequence is rebuilt from its prompt alone | 7 of 10 | 45 of 45 | yes | every request answered or refused after 33 steps |
| a preempted sequence is rebuilt with its pending token | 8 of 10 | 45 of 45 | yes | every request answered or refused after 33 steps |
<!-- /worked -->

The specified server answers the ten honest requests with their replies alone, refuses the 45 messages it should with the reasons it should, and keeps its ledger for all 35 steps; client 3's 4 accepted requests and client 4's one well-formed request are answered too, and the queue never holds more than 9. Every planted mistake fails a check: the misplaced slot and the two wrong rebuilds give wrong honest replies, the kept table breaks the ledger at step 15, and the admission that counts only the prompt stops the server in its first step, since the flood fills the pool with prompts at once. No mistake changes the boundary's answers except by stopping the server before the later messages arrive (36 of 45 for the server that stops in step 0). The three referees together see every mistake; none of them alone does.

Your server is finished when its run reproduces the first row, and when each of your planted mistakes fails where its row says.

## Where brain keeps the parts of this server

Each part of the project has a counterpart in brain, and the evidence for each is the code.

The pool and the tables are `BlockAllocator` and `BlockTable` in `crates/model/src/paged.rs`. The allocator keeps its free blocks as a stack, filled from the highest id down so that the lowest ids are handed out first, and `decref` pushes a block whose last reference has gone back on top; a block can have several references because brain's tables may share blocks (a common prompt prefix, or the branches of speculative decoding), which the project leaves out. `BlockTable::append` computes the offset of the new position as the length modulo the block size before it adds one to the length, the order whose reversal is the planted mistake of step 2, and `release` gives back every block and then clears the list and the length, the line whose omission is the planted mistake of the opening table's second row.

The step is `Scheduler` in `crates/model/src/serve.rs`, generic over any decoder that implements `PagedDecoder`. Its admission (`admit_one`) asks for the blocks of the prompt and its first token (less the blocks the prompt adopts from the prefix cache, and plus the blocks of recurrent state that a hybrid decoder keeps in the same pool) plus `committed_blocks`, which is one more position for every running sequence and the rest of every prefill still in progress, the promise rule of step 3 with a prefill that can be cut into pieces. When the pool cannot back a decode, `ensure_decode_room` stops sequences newest first, a prompt still being prefilled before any decoding sequence, moving a sequence's cache to host memory when the decoder can and has room there, and otherwise preempting it by recompute: `requeue_for_recompute` gives back its blocks and puts it at the head of the queue with its prompt and its generated tokens, the last one held apart as the next input, exactly the context of step 4. The last active sequence is never stopped. Commit `a7d14c830` (19 October 2026) added the preemption by recompute and the counting of the running sequences' next positions at admission; the integration test that came with it, `sequences_outgrowing_the_pool_without_offload_finish_with_their_solo_tokens`, now in `crates/model/tests/it/serve_kv_pressure.rs`, runs six requests whose prompts fit together in a pool of 24 blocks and whose lives do not, and asserts that each finishes with the tokens it produces alone and that all 24 blocks are free at the end. The same file's `a_request_that_cannot_fit_the_whole_pool_is_rejected_not_queued_forever` checks the capacity bound of step 6 (a request of a 10-token prompt and a budget of 100 against 96 positions of pool is refused at once). The same commit changed that bound from the model's sequence limit alone to the smaller of it and the pool's positions, for the reason of the argument above: the last sequence is never stopped, so a sequence must be able to finish alone in the pool.

The comparison with replies alone is brain's conformance suite for serving, `crates/model/src/serve/conformance.rs`. Its `batched_output_equals_single_request_output` submits four prompts of 5, 9, 17 and 25 tokens to one scheduler and asserts that each decodes, greedily, to exactly the tokens it gets on a fresh scheduler alone. Its module documentation, on the property that speculative decoding produces plain greedy's tokens, says why the comparison is exact and not tolerant: a tolerant comparison would hide a verification step that really diverges, and a decoder whose verification sums in another order than its plain decode builds its test weights with a clear margin between the first and the second choice, so that no comparison rests on a near tie. The gap column plays that role in the project.

The token check has its own story. Commit `354ba0f1a` (28 July 2026) introduced the prefix cache, and its message records that writing the property test for it exposed a hole: nothing validated prompt token ids, and an id outside the vocabulary sent the embedding gather out of bounds, which on the trusted kernels gave "silent garbage" (NaN on the CPU, wrong finite values on the GPU) rather than an error. The scheduler has since refused such a prompt with `RejectReason::InvalidToken`; today the bad token is found once, when the request is submitted, and the refusal is reported when the scheduler reaches the request in its queue, so the queue keeps moving. The project's negative id is the same lesson in another language: where the language does not raise, the read succeeds and the answer is wrong.

At the edge, `crates/apiserve/src/lib.rs` bounds the body of an HTTP request to the dialect routes at `MAX_BODY_BYTES`, 8 MiB (the routes that carry media inline have a ceiling of their own, 64 MiB), with a layer that answers 413 before any handler reads the body (commit `ea82ba1b6`, 31 July 2026, which added the explicit bound after a security audit; the test `oversized_request_body_is_413` in `crates/apiserve/tests/it/api.rs` sends a body 1,024 bytes over it). `max_tokens` in `crates/apiserve/src/sampling.rs` refuses a token budget that is not a positive integer, the project's "max_new below 1". The routes of the provider dialects (chat, embeddings, images; not the listing of models) sit behind a global limit of `EDGE_CONCURRENCY` (256, defined in `crates/residency/src/admission.rs`) requests in flight, beyond which a request is shed at once with a 503 rather than queued. And the scheduler takes an `AdmissionPolicy` when a request is submitted: `MaxQueueDepth(max)` refuses a request when `max` or more requests are already waiting, and `DeadlineAware` when the requests ahead of it, at the mean service time measured so far, would take longer than its deadline. The default, `UnboundedQueue`, admits everything to the queue.

## Directions for the finished server

The server is small enough that each of these is a change of one part, checked by the same three referees. Prompts that begin alike can share blocks: a table that points at another sequence's full blocks, with a count of references per block so that a block is given back only when its last user is done, and a ledger that counts references instead of ids ([Managing the KV Cache](03-paged-kv.md)). A long prompt can be prefilled in slices of a fixed number of rows per step, so that one prompt does not make every running sequence wait for it; the promise of step 3 then covers the rest of the prompt still to come ([Batching and Scheduling Requests](04-batching.md)). A stopped sequence can be moved to a slower memory instead of being rebuilt, which turns the rebuild's FLOPs into bytes moved twice, and the cost table can be extended to say when that pays. Sampling can replace greedy decoding, with a generator per request whose state travels with a stopped sequence, so that the rebuilt sequence draws what it would have drawn. And the boundary can be given a bound on the rate of a client's messages, which no bound on the queue provides.

## Check your understanding

1. A colleague tests a server by sending each of a hundred requests alone and checking its reply, and all pass. Which mistakes of the project can such a test find, and which can it not? Use the rows of the opening and whole-run tables.
2. A server counts, at admission, the newcomer's prompt and its first position but not the blocks the running sequences are promised, and keeps the loop that stops the newest sequence before a decode. Predict whether any reply changes, and what happens to request 6 at step 13. The cost table has the row to check against.
3. The reservation rule never preempts and never rebuilds anything. Why does it need 35 forward passes where the promise rule needs 21, and what about the replies of this workload leaves most of its reserved blocks idle?
4. Find the error: "A preempted sequence must be rebuilt from its whole reply, since the cache must hold every token the sequence has produced."
5. The total bound on the waiting queue keeps the server's memory fixed under the flood. Why is it not enough, and what does the per-client bound rely on that a field of the message could not give?
6. A server checks that the declared length equals the number of tokens only after it has reserved blocks for the declared length. Which row of which table measures what the order costs, and how large is it?

## Summary

If every step passes, you have a server that holds the cache of every running sequence in a pool of 6 blocks of 4 positions, 12,288 bytes, rebuilds the batch at every step, admits a request only when its prompt, its next position and the next positions of everything running fit, stops the newest sequence and rebuilds its cache from its prompt and reply when the running sequences outgrow the pool, and refuses, with a reason and before anything is spent, every message that is too large, inconsistent, out of range, beyond its capacity or one too many from its client. On the workload it gives every request the reply that request gets alone, in 21 forward passes where one sequence at a time takes 36, and under a flood and ten hostile messages it still does, with its memory bill unchanged.

The three referees are what make the claim checkable: the replies alone see wrong arithmetic and wrong rebuilds, the ledger sees memory owned twice, and the answers of the boundary see a check that is missing or in the wrong order. Each catches mistakes the others miss. What the project leaves out is as definite: one model, greedy decoding, no shared blocks, prefills that are never cut into slices, and a cost stated in passes, bytes and FLOPs on a model too small for any of them to be a measure of speed. The same server with a real model on a device is the next thing to build, and the methods of Volume XV, porting a model with proof that it matches its reference and measuring what a change does, are how it is built.

## Answers

1. The new token's key goes at position $c$, the count before the token is added, and its query reads $c + 1$ keys: the $c$ cached ones and its own. Each earlier row depends only on the weights and on the tokens at positions up to its own (the causal mask), so a later token cannot change it, and the cache only grows.
2. $\lceil 13/4 \rceil = 4$ blocks. Position 9 is in logical block $\lfloor 9/4 \rfloor = 2$, so in the block named by entry 2 of the table, at slot $9 \bmod 4 = 1$. A block holds rows of keys and values at no particular position of any sequence; the table gives it its meaning, so once its sequence has given it back, any table may list it.
3. Sequences that finished leave and waiting requests join, so the batch's members and size can change at every step. The step reads the weights once for all its rows, so a further decode row adds its own arithmetic and the reading of its own cache, but no further pass over the weights.
4. The running sequences grow: each takes a new block whenever it fills its last one, and an admission that spends those blocks on a newcomer leaves a decode with nowhere to write. The rule must count the newcomer's first new position and the blocks promised to the running sequences for their next positions.
5. Because the allocation is the damage: a server that sets aside memory for the declared length has spent it before it learns that the length is false. Sized at 512 bytes per position, the million positions would be 512,000,000 bytes for a message of 20 bytes, a ratio of 25.6 million.

Check your understanding.

1. Alone, a request is never preempted, never shares the pool and never meets a full queue. Such a test finds the misplaced slot of step 2 (only 2 of 10 replies right alone) and nothing else: the kept table, the two wrong rebuilds and the admission that counts only the prompt all answer every request correctly when it is alone, and fail only under load (8, 8 and 7 of 10 right, and a stop in step 7). The boundary's checks are not exercised by honest requests at all.
2. No reply changes: whatever admission lets in, the loop before the decode restores room by stopping the newest sequences, and a stopped sequence is rebuilt exactly. What changes is the work. At step 13 request 6 is admitted into blocks that the running sequences need for their next positions in the same step, is prefilled, and is stopped by the loop before its first decode; its prefill is paid again when it is rebuilt. The row shows 3 preemptions instead of 2 and 224,000 FLOPs of rebuilds instead of 159,104, in the same 21 passes.
3. Each sequence reserves blocks for its whole budget, and with budgets of 6 to 16 tokens over prompts of 4 to 8 a reservation is 3 to 6 blocks of a pool of 6, so rarely can two sequences run together, and the passes approach those of one sequence at a time. A reserved block is used only if the reply reaches the position it was reserved for, and here 7 of the 10 replies stop at a 0 after 1 to 4 tokens, well short of budgets of 6 to 16, so most reserved blocks are never written.
4. The last token of the reply is not in the cache: it is the input of the next decode, which writes its key and value. Rebuilding from the whole reply writes it into the cache and then feeds it again, so the sequence reads that token twice; it is the planted mistake of step 4, with 7 of 10 replies right.
5. The total bound fixes the memory of the queue (16 messages of at most 104 bytes) but lets one client fill it, so honest requests are refused: 3 of the 10 under the flood. The per-client bound needs to know which client sent a message, and it must take that from the connection, which the server knows, and not from a field, which the sender writes and could set to any client.
6. The amplification table: the message of 20 bytes that declares a million tokens would have such a server reserve 250,000 blocks, 512,000,000 bytes, before the check refused it.
