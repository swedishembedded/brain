<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Batching and Scheduling Requests

*How does a server keep the device busy while answering many users at once?*

Four requests share a batch. One wants 40 tokens, one 6, one 9 and one 30. The batch decodes them together, one token per sequence per step, and the request that wanted 6 tokens has its answer after six steps. Its row in the batch then has nothing left to do, and if the batch was formed once and runs until its longest member is done, the row stays empty for 34 more steps. So do the rows of the 9-token and the 30-token requests, for 31 and 10 steps. Of the 160 row-steps the batch occupies, 85 produce a token. Meanwhile eight more requests wait at the door, and the first of them gets its first token at step 41, although a row was free from step 7 on.

Nothing about the device made those rows idle. It was the decision of when a request may join and when it must leave. A server that answers many users is a device with a fixed amount of memory and a fixed rate of arithmetic, and a stream of requests of unknown length arriving whenever their senders like. Its scheduler decides, once per pass of the model, which requests take part in the next pass, how much of each one's work is done in it, and which requests are not taken at all. Each of the three is a decision about work: the bytes and the floating-point operations (FLOPs) a pass costs, and how much of that cost turns into tokens that a user is waiting for.

The decisions lean on four earlier results, and they are worth bringing back before the first table. Volume V, Chapter 12 ("The KV Cache") split generation into a prefill phase and a decode phase: which of the two is limited by the bytes it moves and which by its arithmetic, and from about how many prompt tokens does a prefill of the 7.5-billion-parameter decoder reach the ridge of the illustrative GPU? The same chapter decoded several sequences in one step: what is read once for all of them, what is read once for each, and why does adding sequences raise the arithmetic intensity without ever reaching the ridge at a context of 4,096 tokens? Chapter 3 of this volume ("Managing the KV Cache") stores a sequence's cache in fixed-size blocks named by a block table: with blocks of 16 tokens, how many blocks does a sequence of 97 cached positions hold, and what happens when it appends its 113th position? And Chapter 2 ("Memory Budgets and Model Residency") gives every device a budget with a reserved headroom: which bytes of it may a new allocation use?

## A batch that lets rows leave and others join

The opening's batch is **static batching**: a batch is formed from the requests waiting when the previous batch ended, and it runs, one decode step after another, until every member has finished. It is the obvious way to batch, and it is what a server does if it treats a batch as one job. Its defect is in its unit. The unit of the device's work is one pass of the model, an iteration, but the unit of the scheduler's decisions is the whole batch. A request that finishes early cannot leave, since its row is part of a shape fixed for the batch's life, and a request that arrives cannot enter, since the batch it would enter is still running.

**Continuous batching** moves the decision to the iteration. Before every pass, finished sequences give back their rows and waiting requests take the free ones, so the batch is a set that changes at every step, and a request waits at most until some row is free. The idea was published as iteration-level scheduling by Yu and colleagues (2022); in brain it has been the shape of the serving scheduler since its first version, whose message says that "finished sequences return their blocks immediately so new requests admit mid-flight" (commit `3ebcc2bf6`, July 2026).

The two policies can be compared on the opening's batch and the eight requests behind it, counting iterations and leaving out what an iteration costs.

<!-- worked: batching.static_vs_continuous -->
Twelve requests arrive together, numbered 0 to 11 in arrival order, and are served on 4 slots (a slot is a row of the batched decode step). Request i produces the number of tokens in the column 'tokens', one in every iteration it holds a slot; its prompt and first token are folded into its first iteration. Iterations are numbered from 1. Static: the first 4 waiting requests form a batch, which runs until its longest member is done; only then is the next batch formed. Continuous: at the start of every iteration each free slot takes the oldest waiting request, the lowest-numbered free slot first; a request that produces its last token in iteration t frees its slot for iteration t + 1.

| request | tokens | static: first | static: last | continuous: first | continuous: last |
|---|---:|---:|---:|---:|---:|
| 0 | 40 | 1 | 40 | 1 | 40 |
| 1 | 6 | 1 | 6 | 1 | 6 |
| 2 | 9 | 1 | 9 | 1 | 9 |
| 3 | 30 | 1 | 30 | 1 | 30 |
| 4 | 5 | 41 | 45 | 7 | 11 |
| 5 | 12 | 41 | 52 | 10 | 21 |
| 6 | 8 | 41 | 48 | 12 | 19 |
| 7 | 25 | 41 | 65 | 20 | 44 |
| 8 | 7 | 66 | 72 | 22 | 28 |
| 9 | 10 | 66 | 75 | 29 | 38 |
| 10 | 6 | 66 | 71 | 31 | 36 |
| 11 | 20 | 66 | 85 | 37 | 56 |

Slot-iterations are slots times iterations; the useful ones are the tokens produced. The mean wait is the mean, over the 12 requests, of the first iteration minus 1; both percentages and means are rounded half away from zero to two decimals.

| policy | iterations | slot-iterations | useful | useful (%) | mean wait (iterations) |
|---|---:|---:|---:|---:|---:|
| static | 85 | 340 | 178 | 52.35 | 35.00 |
| continuous | 56 | 224 | 178 | 79.46 | 13.33 |
<!-- /worked -->

The static schedule takes 85 iterations: three batches of 40, 25 and 20 iterations, each as long as its longest member. Of its 340 slot-iterations, 178 produce a token, 52.35 percent. The continuous schedule produces the same 178 tokens in 56 iterations, 79.46 percent of its 224 slot-iterations. The mean wait before a request's first token falls from 35 iterations to 13.33. Neither is at 100 percent, and the continuous schedule shows why: in its last twelve iterations only request 11 is left, and three of the four rows have nobody to serve. Continuous batching removes the idleness that the scheduler causes, not the idleness of an empty queue.

![Two timelines of four slots each. Static: requests 0 to 3, then 4 to 7, then 8 to 11, each batch running until its longest member is done, with dashed idle stretches after the short requests. Continuous: a new request starts in a slot as soon as the previous one ends, and the last request ends at iteration 56.](img/slot-occupancy.svg)

The two schedules are the routines below. A slot is a row of the batched decode step, and a request of $n_{\mathrm{gen}}$ tokens holds its slot for $n_{\mathrm{gen}}$ iterations, its prompt folded into the first.

```text
function static_slots(lengths: list<int>, slots: int) -> list<(int, int, int)>:
    require slots >= 1
    out = []                                   # (slot, first, last iteration) per request
    t = 1
    b = 0
    while b < len(lengths):
        batch = lengths[b : min(b + slots, len(lengths))]
        for j in 0 .. len(batch):
            append (j, t, t + batch[j] - 1) to out
        t = t + max(batch)                     # the next batch starts after the longest member
        b = b + slots
    return out

function continuous_slots(lengths: list<int>, slots: int) -> list<(int, int, int)>:
    require slots >= 1
    free_at = [1 for j in 0 .. slots]          # the first iteration at which each slot is free
    out = []
    for n in lengths:
        j = argmin(free_at)                    # earliest free slot; ties: lowest number
        append (j, free_at[j], free_at[j] + n - 1) to out
        free_at[j] = free_at[j] + n
    return out
```

A batch that changes at every step needs a cache that can change at every step. A cache laid out as one rectangle, a row of positions per sequence and as many columns as the longest sequence may grow, has to be rebuilt or padded whenever a sequence joins or leaves. The block table of the recall question is what makes the set cheap to change: a sequence that joins takes free blocks from the pool, a sequence that leaves returns its blocks, and nothing else moves. Continuous batching and the paged cache are two halves of one design, and the scheduler described here runs over the paged cache of Chapter 3.

## The price of a row in a pass

Counting iterations treats every iteration as equal, and they are not. An iteration is one or more forward passes, and a pass costs the bytes it must move and the FLOPs it must do. Volume IV, Chapter 8 folded the two into one number, the equivalent bytes $B_{\mathrm{eq}} = \max(N_{\mathrm{byte}}, N_{\mathrm{flop}} / \mathcal{I}_{\mathrm{ridge}})$: the roofline time multiplied by the bandwidth, so that a cost carries no seconds and holds on any device with the same ridge point. Everything below is priced in it, on the illustrative GPU of Volume IV, whose ridge $\mathcal{I}_{\mathrm{ridge}}$ is 50 FLOP per byte. These costs are a roofline model written from the arithmetic and the bytes of the pass, not measurements of any machine: they say what a pass requires of a device with that ridge, and a scheduler on real hardware has to learn what its passes actually take, as brain's does below.

The pass is the one of the 7.5-billion decoder of Volume V, Chapter 12, and its FLOPs are that chapter's $F(c, n)$: the FLOPs of $n$ new positions on a cache of $c$, the head applied once. A pass can carry two kinds of rows. A decode row of a sequence with $c$ positions cached costs $F(c, 1)$ and reads the $c$ cached positions. A prompt slice of $n$ rows that starts at position $s$ costs $F(s, n)$, without the head unless it is the last slice of its prompt, since only the last position of a prompt is turned into logits; it reads the $s$ positions before it and writes its own $n$. The weights, $B_{\mathrm{w}}$ bytes, are read once per pass whatever rows it carries, and every cached position is $B_{\mathrm{kv}}$ bytes of keys and values.

```text
function forward_work(ctx: list<int>, slices: list<(int, int, bool)>) -> (int, int):
    # FLOPs and bytes of ONE pass: a decode row per entry of ctx (positions cached),
    # a prompt slice per (start, rows, last of its prompt)
    if len(ctx) == 0 and len(slices) == 0:
        return (0, 0)
    flops = 0
    nbytes = Bw                                # the weights, once for the pass
    for c in ctx:
        flops = flops + F(c, 1)
        nbytes = nbytes + Bkv * c
    for (s, n, last) in slices:
        flops = flops + F(s, n) - (0 if last else 2 * D * V)
        nbytes = nbytes + Bkv * (s + n)
    return (flops, nbytes)

function forward_cost50(ctx: list<int>, slices: list<(int, int, bool)>) -> int:
    (flops, nbytes) = forward_work(ctx, slices)
    return max(50 * nbytes, flops)             # 50 B_eq: an int, so sums of costs are exact

function free_rows(batch: int, c: int) -> int:
    n = 0
    loop:
        (flops, nbytes) = forward_work([c for k in 0 .. batch], [(0, n + 1, false)])
        if flops > 50 * nbytes:
            return n
        n = n + 1
```

`forward_cost50` returns 50 times $B_{\mathrm{eq}}$ so that the schedules below add integers and every printed digit can be reproduced; the tables divide by 50 at the end. The costs of the passes a scheduler chooses between:

<!-- worked: batching.step_costs -->
The 7.5-billion decoder of Volume V, Chapter 12 (15,009,316,864 bytes of bf16 weights, 131,072 bytes of keys and values per token) on the illustrative GPU of Volume IV, whose ridge is 50 FLOP per byte. FLOPs are counted as in that chapter (a multiply-add is 2, only matrix products count): a decode row at context c costs F(c, 1), the head included; a prompt slice of n rows starting at position s costs F(s, n), without the head unless it is the last slice of its prompt. Bytes: the weights once per forward, the c cached positions every decode row reads, the s earlier positions a slice reads and the n it writes. The cost of a forward is B_eq = max(bytes, FLOPs / 50) in equivalent bytes (Volume IV, Chapter 8), given in units of 10^9 and rounded half away from zero to three decimals; FLOP per byte to two.

| forward | FLOPs | bytes | FLOP per byte | B_eq (10^9) |
|---|---:|---:|---:|---:|
| decode, 1 sequence at context 2,048 | 16,083,582,976 | 15,277,752,320 | 1.05 | 15.278 |
| decode, 16 sequences at context 2,048 | 257,337,327,616 | 19,304,284,160 | 13.33 | 19.304 |
| decode, 64 sequences at context 2,048 | 1,029,349,310,464 | 32,189,186,048 | 31.98 | 32.189 |
| prefill, 2,048 tokens in one pass | 29,688,401,494,016 | 15,277,752,320 | 1,943.24 | 593.768 |
| prefill slice, tokens 0 to 255 | 3,590,659,768,320 | 15,042,871,296 | 238.70 | 71.813 |
| prefill slice, tokens 1,792 to 2,047 (the last) | 3,832,228,610,048 | 15,277,752,320 | 250.84 | 76.645 |
| decode of 16 at 2,048 with the slice 0 to 255 | 3,847,997,095,936 | 19,337,838,592 | 198.99 | 76.960 |

The most prompt rows (a slice from position 0, not the prompt's last) that a decode step at context 2,048 can carry while its FLOPs stay at or below 50 times its bytes:

| sequences decoding | free prompt rows |
|---|---:|
| 1 | 53 |
| 4 | 52 |
| 16 | 50 |
| 64 | 41 |

Past that point each further prompt row adds 0.279 x 10^9 equivalent bytes for its weights (2 x 6,979,321,856 FLOPs over 50), plus its attention.
<!-- /worked -->

Four facts in the table carry what follows. A decode step of one sequence at context 2,048 costs 15.278 $\times 10^9$ equivalent bytes, almost exactly the weights; sixteen sequences cost 19.304, only 26 percent more for sixteen times the tokens; sixty-four cost 32.189. This is the lower table of Volume V, Chapter 12 seen from the scheduler's side: a decode step is bandwidth-bound, so a row added to it costs its own cache read and not another read of the weights, and every row left empty is a token the device could have produced for almost nothing. That is the work behind the opening's 85 idle row-steps.

A prefill of one 2,048-token prompt costs 593.768 $\times 10^9$, at 1,943.24 FLOP per byte, deep in the compute-bound region. That is 30.76 decode steps of sixteen sequences, or 38.86 decode steps of one. A scheduler that runs such a prefill as a pass of its own between two decode steps makes every running sequence wait that long for its next token.

A slice of 256 prompt rows costs 71.813 from the first position and 76.645 for the last slice, which reads the 1,792 earlier positions and computes the head. Cutting a prompt into slices is nearly free in work: the eight slices of a 2,048-token prompt read their earlier positions again, $256 \times (0 + 1 + \cdots + 7) \times 131{,}072 = 939{,}524{,}096$ bytes in all, and since each slice is compute-bound those bytes do not even show in its cost. What costs is attention, which a later slice computes against more keys, and nothing else.

And the last row: the decode step of sixteen sequences carrying the first slice in the same pass costs 76.960, against $71.813 + 19.304 = 91.117$ for the two as separate passes. The weights are read once for both, and the decode rows' arithmetic hides under the slice's. The second table says how far that goes: a decode step at context 2,048 can carry 53 prompt rows with one sequence decoding, 50 with sixteen and 41 with sixty-four, before its FLOPs reach 50 times its bytes. Up to that point a prompt row costs only its cache write; past it, each further row costs its 0.279 $\times 10^9$ equivalent bytes of weight FLOPs.

![The cost of a decode step of 16 sequences at context 2,048 against the prompt rows it carries, from 0 to 512: carried in the same pass the cost is flat at 19.3 billion equivalent bytes up to 50 rows and then rises in a straight line; run as a separate pass it is about 15 billion higher throughout.](img/step-cost.svg)

The figure is the shape a scheduler has to know. Below the knee, prompt rows ride on a decode step for nothing; above it, the cost grows in a straight line with the rows, plus a fixed part that the separate pass pays twice. The scheduler of brain models its iteration with exactly that shape, a fixed part plus a part per row, but it does not compute it from FLOPs: `StepCost` in `crates/model/src/serve/budget.rs` fits the measured seconds of the mixed iterations that ran (the scheduler reads its clock around each one and calls `StepCost::observe` with the rows and the seconds) against their rows by a least-squares line whose history decays by a factor of 0.95 per observation, and treats the cost as proportional to the rows until the row counts it has seen have a standard deviation of at least 8 rows, below which a slope cannot be told apart. A measurement and a model can disagree about where the fixed part comes from, and knowledge #379 records that they did: on the engine it was fitted for, the fixed part of a mixed pass was the number of kernel launches and the host's work to record them, not the weight read the roofline predicts. The scheduler is indifferent to which: a fitted line works whatever made it straight.

## A long prompt in front of everyone else

The price table already says what goes wrong when a burst of long prompts arrives. Request 0 is decoding when eight requests arrive together, each with a prompt of 2,048 tokens and wanting 32 tokens. Every policy below produces the same tokens for every request, so they differ only in when the tokens exist. Four policies take the prompts in, and the next block runs them through the costs of the table.

<!-- worked: batching.burst -->
Request 0 is decoding (a prompt of 512 tokens, 512 positions cached, one token produced) when 8 requests, 1 to 8, arrive together, each with a prompt of 2,048 tokens and wanting 32 tokens (the first from its prompt's last pass, the others from decode steps). The batch and the cache have room for all of them. An iteration does the prompt work of the policy and then one decode row for every sequence that still wants tokens, a new sequence included from the iteration its prompt finished, except under the mixed pass, where it joins in the next one. Whole prompts, no budget: every waiting prompt is prefilled whole, each in a forward of its own, in the first iteration. One prompt an iteration: the oldest waiting prompt is prefilled whole. Chunks of 256, own pass: 256 rows of the one prompt being read in, oldest first, in a forward of their own before the decode forward; the next prompt starts in the iteration after the previous one is in. Chunks of 256, mixed pass: the same rows carried in the decode forward. Every forward costs B_eq of the cost table; an iteration costs the sum of its forwards, and a token exists at the cumulative cost at the end of the iteration that computes it. The run ends when requests 1 to 8 have finished. The first new token is the earliest cumulative cost at which any of requests 1 to 8 has its first token; the longest gap of request 0 is the largest difference between the cumulative costs at the ends of two consecutive iterations in which it decodes, the first measured from cost 0 because its first token exists at time 0; its tokens are the number of those iterations. Costs in 10^12 and in 10^9 equivalent bytes, rounded half away from zero.

| policy | iterations | first new token (10^12) | mean first token (10^12) | last first token (10^12) | all new done (10^12) | longest gap of request 0 (10^9) | tokens of request 0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| whole prompts, no budget | 31 | 4.767 | 4.767 | 4.767 | 5.285 | 4,767.4 | 31 |
| one prompt an iteration | 38 | 0.609 | 2.744 | 4.880 | 5.390 | 611.0 | 38 |
| chunks of 256, own pass | 94 | 0.715 | 3.235 | 5.763 | 6.235 | 92.8 | 94 |
| chunks of 256, mixed pass | 95 | 0.596 | 2.706 | 4.825 | 5.313 | 77.9 | 95 |
<!-- /worked -->

The first policy is the one a scheduler has if it admits every request that fits and prefills it before the next decode step. All eight prompts are prefilled in the first iteration, eight passes of 593.768 each, and request 0 waits 4,767.4 $\times 10^9$ equivalent bytes between two tokens, more than 300 times a decode step of one sequence (15.278 at context 2,048, less at request 0's shorter context). Every new request gets its first token at the same moment, 4.767 $\times 10^{12}$, because none of them can be ahead of the others. This is head-of-line blocking, work at the front of the line holding up everything behind it, and the line here is not even a line of requests: it is the running sequences, which had nothing to do with the prompts that stalled them.

The second policy gives each iteration a budget of prompt tokens and prefills at most one whole prompt per iteration. The longest gap of request 0 falls to 611.0, one prompt and one decode step, and the first new request has its first token at 0.609 instead of 4.767: it no longer waits for the seven prompts behind it. The mean first token improves from 4.767 to 2.744, and the total work hardly changes (5.390 against 5.285, the difference lying in the decode passes of the seven extra iterations). This was brain's first fix. Commit `4d756f585` (28 July 2026) gave the scheduler a per-iteration prefill budget, two prefill chunks by default, and kept one guarantee explicit: at least one waiting request is admitted per iteration, so that a budget can never starve the queue. The test `prefill_budget_spreads_admissions_without_changing_outputs` in `crates/qwen3/src/serve.rs` runs the same requests with an unlimited and with a tight budget and asserts the same tokens; the message's own summary is that the budget "changes WHEN tokens appear, never WHICH". The message also stated what was left: a single long prompt still blocks one iteration, and the cure would be to interleave a prompt across iterations, which needs the scheduler to keep requests that are partly prefilled.

That cure is **chunked prefill**: a prompt is read into the cache in slices of a bounded number of rows over several iterations, and the running sequences decode between the slices. The third row of the table does it with slices of 256 rows in a pass of their own, and request 0's longest gap falls to 92.8, one slice and one decode step. The price is in the other columns. The first new token comes at 0.715 instead of 0.609, the last first token at 5.763 instead of 4.880, and all new requests are done at 6.235 instead of 5.390. A prompt read in eight slices is in the cache only after eight iterations, and each of those iterations also pays for a separate decode pass whose weight read the slice did not share. Chunking bounded the gap between the running sequences' tokens and made every new request wait longer.

The fourth row puts the slice into the decode pass, as the price table suggested: one pass per iteration, carrying every running sequence's decode row and 256 prompt rows. Request 0's longest gap is 77.9, the first new token comes at 0.596, earlier than under any other policy, the mean first token is the best of the four at 2.706, and the work to finish is 5.313, within 0.6 percent of the policy that blocked everyone. Nothing about the prompts changed between the third and fourth rows; only where the slice's rows went. Agrawal and colleagues (2023) built a serving system on this observation, that decode rows can ride on the compute-bound pass of a prompt chunk.

![The device work done so far against the tokens of request 0 for the four policies: whole prompts jump to 4.8 trillion equivalent bytes at the first token, one prompt an iteration climbs steeply over eight tokens, and the two chunked policies climb gently over 64 tokens and then flatten, the mixed pass staying below the separate one.](img/burst-work.svg)

In the figure every token of request 0 is a step to the right, and the height of the step is what that token cost the device. The two policies that prefill whole prompts climb nearly vertically: request 0 produces a handful of tokens while almost all of the burst's work is done. The chunked policies read the eight prompts in over the first 64 tokens of request 0, one prompt per 8 iterations, and after that only decode rows are left; the mixed one does it with less work, the gap between the two curves being one weight read per iteration.

The routine of the four policies, in the order an iteration does its work:

```text
record Seq:
    id: int
    cached: int                                # positions in its cache
    made: int                                  # tokens produced so far
    wants: int                                 # tokens it produces in all

function run_burst(policy: text, P: int, n_new: int, chunk: int, n_req: int)
        -> (list<int>, list<int>, list<int>, list<int>):
    running = [Seq(0, 512, 1, 10^9)]           # request 0, already decoding
    waiting = [i for i in 1 .. n_req + 1]
    pre = -1                                   # the prompt being read in by slices; -1: none
    pre_done = 0
    costs = []
    first = [0 for i in 0 .. n_req + 1]
    last = [0 for i in 0 .. n_req + 1]
    long_iters = []
    t = 0
    while len(waiting) > 0 or pre >= 0 or any([r.id != 0 for r in running]):
        t = t + 1
        cost = 0
        started = []
        sl = []
        if policy == "whole":
            for i in waiting:
                cost = cost + forward_cost50([], [(0, P, true)])
                append i to started
            waiting = []
        else if policy == "one":
            if len(waiting) > 0:
                cost = cost + forward_cost50([], [(0, P, true)])
                append waiting[0] to started
                waiting = waiting[1 : len(waiting)]
        else:                                  # "own" or "mixed": a slice of one prompt
            if pre < 0 and len(waiting) > 0:
                pre = waiting[0]
                pre_done = 0
                waiting = waiting[1 : len(waiting)]
            if pre >= 0:
                n = min(chunk, P - pre_done)
                sl = [(pre_done, n, pre_done + n == P)]
                pre_done = pre_done + n
                if pre_done == P:
                    append pre to started
                    pre = -1
            if policy == "own":
                cost = cost + forward_cost50([], sl)
                sl = []
        if policy != "mixed":                  # a finished prompt decodes in this iteration
            for i in started:
                append Seq(i, P, 1, n_new) to running
                first[i] = t
        act = [j for j in 0 .. len(running) if running[j].made < running[j].wants]
        cost = cost + forward_cost50([running[j].cached for j in act], sl)
        for j in act:
            running[j].cached = running[j].cached + 1
            running[j].made = running[j].made + 1
            if running[j].id == 0:
                append t to long_iters
            else if running[j].made == running[j].wants:
                last[running[j].id] = t
        if policy == "mixed":                  # first token here; decodes next iteration
            for i in started:
                append Seq(i, P, 1, n_new) to running
                first[i] = t
        running = [r for r in running if r.made < r.wants]
        append cost to costs
    return (costs, first, last, long_iters)
```

The table's run is `run_burst(policy, 2048, 32, 256, 8)` for the four policies `"whole"`, `"one"`, `"own"` and `"mixed"`, and its columns are read off the returned lists: a token computed in iteration $k$ exists at the sum of the first $k$ costs, divided by 50.

The scheduler of brain took the same three steps, in that order. Commit `c025dd891` (17 October 2026) let a decoder prefill a prompt in slices: a decoder that declares `prefill_interleave_tokens` is handed at most that many prompt tokens per iteration through `prefill_chunk`, the scheduler keeps a queue of partly prefilled requests ahead of the running ones, and a cancelled request that is half prefilled returns its blocks (the test `a_cancelled_half_prefilled_request_returns_its_blocks`). Its message describes the failure it removed in these terms: sixteen long prompts arriving together, a short answer needing one iteration per token, and "every short answer waited for every prompt behind it and the whole wave finished at one instant". The test `short_requests_finish_at_their_own_time_not_after_the_queue_of_prompts_behind_them` in `crates/model/tests/it/serve_continuous.rs` is that burst on a toy decoder with a virtual clock (one unit per prompt token, ten per decode step): without slices the first short answer waits at least 500 units, with slices of 16 tokens it is back before three quarters of that, every short answer leaves before half of the long request's time, they leave in arrival order, and every answer's tokens are unchanged. The next day's commit, `f738e7661`, let a decoder take the slices of several prompts in one pass, `prefill_chunks`, and the test beside it counts the passes: eight 100-token prompts at 64 rows a pass take 13 passes instead of the 50 that one slice of 16 rows a pass would take.

The mixed pass came last. Commit `f74979a12` (29 October 2026) added `PagedDecoder::mixed_step` and `forward_mixed`, one pass carrying the decode rows of the running sequences and slices of the prompts being read in, and sized the prompt part from a latency target through the fitted `StepCost`. Two later commits adjusted the size of the slice to the shape of the cost. Commit `53264280b` made the smallest prompt part of a mixed pass 512 rows on the engine that declares it, and lets a pass take a quarter more than the smallest slice's predicted cost when the target is out of reach even for that slice: a pass whose cost is mostly fixed should carry many rows, since the fixed part is paid anyway. Commit `fc6586893` rounds the prompt rows up so that the rows of the pass fill the last tile of its matrix products, since a product costs the same for every row count within a tile; the test `the_rows_fill_the_last_tile_of_the_products` takes a pass that the target leaves at 250 rows and checks that it is filled to 256. `StepCost::prompt_rows` holds the guarantee that matches the one-admission rule of the budget: whatever the target says, a waiting prompt gets at least its minimum slice in every pass (`a_prompt_advances_every_iteration_however_busy_decode_is` in `crates/model/tests/it/serve_mixed_step.rs`).

## Admitting a request on the blocks it will need

A scheduler that takes requests in as soon as there is a row for them still has to decide whether there is room for them, and room is not only rows. **Admission control** is that decision: at the door of the batch, before any work is spent on a request, decide whether to take it in now, to let it wait, or to refuse it, from what it will need and what the server has. The first resource it guards is the cache. A sequence's cache grows by one position per token it produces, and the pool of blocks that holds every sequence's cache has a fixed size, set by the memory budget of Chapter 2.

Some requests can be refused at once, before they wait at all, because no amount of waiting will make them fit. The scheduler of brain refuses three kinds (`queued_refusal` and `adapter_verdict` in `crates/model/src/serve.rs`): a prompt that holds a token outside the vocabulary, whose embedding lookup would read out of bounds; a request whose prompt plus its ceiling of new tokens, less one, exceeds what one sequence may hold, which is the smaller of the decoder's sequence limit and the whole pool (the last token produced is never cached, hence the one); and a request for an adapter the decoder does not serve, since answering it with the base model's weights would be a wrong answer and not a slow one. The test `a_request_that_cannot_fit_the_whole_pool_is_rejected_not_queued_forever` in `crates/model/tests/it/serve_kv_pressure.rs` is the second kind. A request that only does not fit yet is a different matter: it waits, and it holds the line, so that a large request is not overtaken forever by smaller ones behind it.

For a request that can fit, the question is how much to charge it. A request names a ceiling on its new tokens, but most requests stop well before it, at a stop token. Charging the ceiling is safe: a sequence admitted with its whole future reserved can never run the pool dry. Charging only the prompt admits more sequences at once, but sequences admitted that way can outgrow the pool together, and then something must give way. The next block runs eight requests against a pool of 48 blocks under both charges, and under a third that charges the prompt and has no plan for the overflow.

<!-- worked: batching.pool_admission -->
A pool of 48 blocks of 16 tokens (768 tokens), at most 8 sequences in flight, and eight requests, 0 to 7, arriving together, each with a prompt of 96 tokens and a ceiling of 400 new tokens; they produce 40, 380, 70, 20, 250, 120, 60, 300 tokens and then stop (1,240 in all). A sequence holding c positions holds ceil(c / 16) blocks. Every iteration: (1) admit waiting requests oldest first while fewer than 8 sequences run and the free blocks cover what the policy asks, stopping at the first that does not fit; an admitted request is prefilled whole in a forward of its own, and a fresh one gets its first token from it; (2) make room for the decode step; (3) decode one row for every sequence that still wants tokens, each adding one position and taking a block when that position starts one; (4) the finished ones give their blocks back. Reserve the ceiling: admission asks for the blocks of prompt + ceiling - 1 positions and holds them to the end. Prompt only: admission asks for the blocks of the prompt and its first token, plus the block (if any) that each running sequence's next position needs, and a sequence grows a block at a time. Recompute: while the free blocks cannot back every sequence's next position and more than one sequence decodes, the newest gives back all its blocks and goes to the head of the queue, to be prefilled again over its prompt and every token it produced except the last. Work is the sum of B_eq over every forward (the rules of the cost table), in 10^12 equivalent bytes, rounded half away from zero to three decimals; the mean batch is decode rows over iterations, to two.

| policy | iterations | mean batch | prompt tokens prefilled | preemptions | work (10^12) |
|---|---:|---:|---:|---:|---:|
| reserve the ceiling | 1,232 | 1.00 | 768 | 0 | 18.743 |
| prompt only, recompute | 470 | 2.62 | 1,606 | 5 | 7.541 |
| prompt only, no preemption | no free block at iteration 33, 6 sequences decoding |  |  |  |  |
<!-- /worked -->

Reserving the ceiling admits one request at a time: each asks for $\lceil 495 / 16 \rceil = 31$ of the 48 blocks, so the second can never join the first. The eight requests take 1,232 iterations, one decode row each, and the work is 18.743 $\times 10^{12}$ equivalent bytes, almost all of it weight reads for steps that carry one row. Charging the prompt lets several sequences run, 2.62 on average, and finishes in 470 iterations with 7.541 $\times 10^{12}$, 40 percent of the work, although five sequences were preempted and 838 prompt tokens were prefilled again. Charging the prompt without a plan stops at iteration 33, when six decoding sequences each need a new block and the pool has none to give.

The plan in the second row is preemption. When the free blocks cannot back every running sequence's next position, the newest sequence gives back all of its blocks and returns to the head of the queue, and when it is admitted again its prompt and every token it had produced except the last are prefilled anew, so that it continues from where it stopped. It is called preemption by recompute, since the cache that was given up is computed again. The newest goes first because it is the furthest from finishing and from giving its blocks back, so the oldest sequences keep running and release theirs soonest. Kwon and colleagues (2023), in the system that introduced the paged cache, name the same two ways of giving a sequence's blocks up when the pool runs dry, copying its cache out to host memory and recomputing it later, and also give up the newest sequences first. The routine, in the order of an iteration:

```text
record Req:
    id: int
    ctx: int                                   # tokens to prefill when it is admitted
    made: int                                  # tokens it produced before it was preempted

record Run:
    id: int
    cached: int                                # positions in its cache
    held: int                                  # blocks it holds
    made: int                                  # tokens produced so far

function grow(r: Run, reserve: bool, blk: int) -> int:
    return 0 if reserve else (r.cached + blk) // blk - r.held   # for its next position

function run_pool(policy: text, lengths: list<int>, P: int, ceiling: int, pool: int,
        max_run: int, blk: int) -> (bool, int, int, int, int, int):
    reserve = policy == "reserve"
    full = (P + ceiling - 1 + blk - 1) // blk
    waiting = [Req(i, P, 0) for i in 0 .. len(lengths)]
    running = []
    free = pool
    t = 0
    work50 = 0
    prefilled = 0
    preempted = 0
    rows = 0
    while len(waiting) > 0 or len(running) > 0:
        t = t + 1
        while len(waiting) > 0 and len(running) < max_run:          # 1. admit, oldest first
            w = waiting[0]
            want = full
            if reserve == false:
                want = (w.ctx + blk) // blk + sum([grow(r, reserve, blk) for r in running])
            if free < want:
                break                                                # it holds the line
            waiting = waiting[1 : len(waiting)]
            held = full if reserve else (w.ctx + blk - 1) // blk
            free = free - held
            work50 = work50 + forward_cost50([], [(0, w.ctx, true)])
            prefilled = prefilled + w.ctx
            append Run(w.id, w.ctx, held, max(w.made, 1)) to running   # fresh: first token
        act = [j for j in 0 .. len(running) if running[j].made < lengths[running[j].id]]
        need = sum([grow(running[j], reserve, blk) for j in act])      # 2. room to decode
        if policy == "recompute":
            while free < need and len(act) > 1:
                j = act[len(act) - 1]                  # the newest active sequence
                act = act[0 : len(act) - 1]
                v = running[j]
                need = need - grow(v, reserve, blk)
                free = free + v.held
                preempted = preempted + 1
                running = concat(running[0 : j], running[j + 1 : len(running)])
                waiting = concat([Req(v.id, P + v.made - 1, v.made)], waiting)
        else if policy == "none" and free < need:
            return (true, t, len(act), 0, 0, 0)        # the pool cannot back the step
        if len(act) > 0:                                               # 3. one decode row each
            work50 = work50 + forward_cost50([running[j].cached for j in act], [])
            rows = rows + len(act)
        for j in act:
            g = grow(running[j], reserve, blk)
            free = free - g
            running[j].held = running[j].held + g
            running[j].cached = running[j].cached + 1
            running[j].made = running[j].made + 1
        for r in running:                              # 4. the finished give blocks back
            if r.made >= lengths[r.id]:
                free = free + r.held
        running = [r for r in running if r.made < lengths[r.id]]
    return (false, t, rows, prefilled, preempted, work50)
```

The table is `run_pool(policy, [40, 380, 70, 20, 250, 120, 60, 300], 96, 400, 48, 8, 16)` for `"reserve"`, `"recompute"` and `"none"`. Two details of the admission test matter. A request is charged the blocks of its prompt and its first token, $\lceil (p + 1) / B_{\mathrm{blk}} \rceil$, and not only of its prompt, since its first decode step writes the first token's position; and it is charged, on top, the blocks that every running sequence needs for its next position in this same iteration. Admitting into blocks that the running sequences are about to take would only get the newcomer preempted before it produced anything.

Both details are brain's, and both came from failures. Commit `fd8c0a29c` (2 September 2026) found that the scheduler admitted on whether a prompt fit and that "nothing checked that the pool could still back the batch's next token": the message reproduces it with eight sessions against a pool sized for two, which reached the decoder's `expect("KV pool exhausted")` and took the serving lane down, the third row of the table. The fix was preemption to host memory, which copies a sequence's cache out instead of dropping it, newest first, with two rules against thrashing (a sequence brought back in an iteration is not that iteration's victim, and a sequence is brought back only when the pool can also back the batch's next step). Copying needs a host budget and a decoder that can copy, so commit `a7d14c830` (19 October 2026) added recompute for the case where neither is there, and the charge for the running sequences' next positions at admission. The comment on `ensure_decode_room` gives the same reason for recompute over reserving the ceiling: the ceiling "is a ceiling, not a forecast", and reserving it would keep the batch "far below what the pool can actually hold, on every request, to avoid a cost paid only under overload". The tests in `crates/model/tests/it/serve_kv_pressure.rs` run six requests whose prompts fit together and whose lives do not, against a decoder whose next token is a function of the whole cache, so a cache rebuilt wrongly gives other tokens; `sequences_outgrowing_the_pool_without_offload_finish_with_their_solo_tokens` asserts that every request ends with exactly the tokens it produces alone, and `a_preempted_sampled_sequence_keeps_its_seeded_stream` that a sampled one does too, since the preempted sequence carries its random stream to the head of the queue with its tokens.

Charging is subtler than counting a prompt's blocks when the prompt shares a prefix with a conversation already in the cache. Commit `51e99f712` (29 October 2026) found the scheduler charging the next turn of a conversation for its whole prompt, although most of it was cached history that the new turn would adopt and not allocate. When the history filled more than the pool less the prompt, the turn was never admitted, "even on an idle server": a livelock. Admission now asks the decoder how many cached blocks a prompt would adopt (`cached_prefix_blocks`) and charges only the rest, and when it must reclaim cached blocks to make room it spares the ones the admitted prompt is about to reuse (`reclaim_prefix_keeping`). The test `the_next_turn_of_a_conversation_filling_most_of_the_pool_is_admitted_on_its_cached_history` sets up a pool of 16 blocks, a history of 10 cached blocks and a next turn of 13 blocks of which 3 are new, with 6 free.

The pool is not the only memory a pass needs, and the scheduler cannot see the rest in advance. A pass also allocates activation scratch, and a hybrid decoder holds recurrent state beside the cache; when those run out, the device refuses an allocation in the middle of a pass. Commit `0ade6e92a` (6 November 2026) changed what such a failure does. It used to fail every request the pass carried; now the scheduler sends them back to the head of the queue to be recomputed, halves the number of sequences it lets in at once, and gives one back after every 32 iterations without a failure (`MemoryPressure` in `crates/model/src/serve.rs`). A request is refused only when it fails three times while it is alone on the device, after the prefix cache has been given back. The device's refusal is treated as the measurement of its real capacity, which the pool's accounting could not provide. The test `a_decode_step_that_runs_the_card_out_of_memory_loses_no_sequence` in `crates/model/tests/it/serve_device_oom.rs` checks that every request still ends with the tokens of a device that never ran out.

## Refusing work that would be answered too late

Room in the cache is one limit; the device's rate is the other. When requests arrive faster than the device can serve them, a queue grows without bound, and a request at the back of a long queue gets an answer after its sender has stopped waiting. The device still does the work. Admission control at the door of the queue decides how much of that to take on.

The next block puts a deliberately simple server under load: requests arrive at a fixed spacing of $a_{\mathrm{gap}}$ units of device work, the device serves one at a time for $s_{\mathrm{svc}}$ units each, and a request is useful only if it starts within $D_{\mathrm{dl}}$ units of its arrival. Three policies stand at the door. The unbounded queue takes everyone. The depth rule takes a request only while fewer than $q_{\mathrm{max}}$ requests wait. The deadline-aware rule takes a request only if the work still ahead of it, the queued requests and what is left of the one being served, ends within the deadline.

```text
function run_overload(gap: int, policy: text, arrivals: int, service: int, deadline: int,
        depth: int) -> (int, int, int):
    starts = []
    busy_until = 0
    in_time = 0
    late = 0
    refused = 0
    for k in 0 .. arrivals:
        t = k * gap
        q = len([s for s in starts if s > t])  # admitted, not started by t
        wait = max(t, busy_until) - t
        if policy == "unbounded":
            ok = true
        else if policy == "depth":
            ok = q < depth
        else:
            ok = wait <= deadline
        if ok == false:
            refused = refused + 1
            continue
        append t + wait to starts
        busy_until = t + wait + service
        if wait <= deadline:
            in_time = in_time + 1
        else:
            late = late + 1
    return (in_time, late, refused)
```

The block is `run_overload(g, policy, 300, 6, 60, 4)` for every spacing and policy.

<!-- worked: batching.overload -->
300 requests arrive, request k (from 0) at k x g units of device work, for g = 12, 6, 4, 3; the device serves one request at a time, first come first served, each for 6 units, so the offered load 6 / g is 0.5, 1, 1.5 and 2. A request is useful when it starts at most 60 units after it arrives. At its arrival a policy sees q, the admitted requests that have not started (one whose start equals the arrival has started), and the wait the request would have, the work still ahead of it. The unbounded queue admits every request; the depth rule admits when q < 4; the deadline-aware rule admits when the wait is at most 60. A refused request is answered at once and costs no work.

| offered load | policy | in time | late | refused | work on late answers (units) |
|---|---|---:|---:|---:|---:|
| 0.50 | unbounded queue | 300 | 0 | 0 | 0 |
| 0.50 | queue depth below 4 | 300 | 0 | 0 | 0 |
| 0.50 | deadline-aware | 300 | 0 | 0 | 0 |
| 1.00 | unbounded queue | 300 | 0 | 0 | 0 |
| 1.00 | queue depth below 4 | 300 | 0 | 0 | 0 |
| 1.00 | deadline-aware | 300 | 0 | 0 | 0 |
| 1.50 | unbounded queue | 31 | 269 | 0 | 1,614 |
| 1.50 | queue depth below 4 | 204 | 0 | 96 | 0 |
| 1.50 | deadline-aware | 210 | 0 | 90 | 0 |
| 2.00 | unbounded queue | 21 | 279 | 0 | 1,674 |
| 2.00 | queue depth below 4 | 154 | 0 | 146 | 0 |
| 2.00 | deadline-aware | 160 | 0 | 140 | 0 |
<!-- /worked -->

Below and at the device's capacity the three policies are the same: nobody waits long enough to matter, and nobody is refused. At an offered load of 1.5 the unbounded queue answers all 300 requests and only 31 of them in time. The wait of request $k$ grows by $s_{\mathrm{svc}} - a_{\mathrm{gap}} = 2$ units per arrival, so from request 31 on every request waits more than 60 units, and the device spends 1,614 units, nearly all of what it does after the first few hundred units, on answers nobody will read. The deadline-aware rule answers 210 in time and refuses 90, and the depth rule 204 in time and refuses 96. Refusing a third of the requests turned 31 useful answers into 210. At a load of 2 the numbers are 21, 160 and 154. The depth rule refuses more than it must, because four waiting requests are only 24 units of waiting against a deadline of 60; a deeper queue would serve more in time, and a much deeper one would let the late answers back in. The deadline rule needs no such tuning, because it asks the question the deadline asks, but it needs to know what the work ahead of a request will take.

The scheduler of brain has the same three policies and leaves the choice to its caller. Commit `57fb1c7b7` (28 July 2026) added the `AdmissionPolicy` seam, consulted when a request is submitted, with `UnboundedQueue` as the default, `MaxQueueDepth` and `DeadlineAware`; its message states the measurement that motivated it, a queue without bound under twice the load it could serve, where compute "was spent on answers past their deadlines", and its principle, that a request is refused only when its lateness is proven. `DeadlineAware` cannot know the work ahead exactly, as the table's rule does, and estimates it. At submission it is given the number of requests in the waiting queue (`QueueState::queued_ahead` is `waiting.len()`, which does not include the running sequences) and an observed mean in milliseconds, and admits the request when their product is at most the deadline; it admits everything while nothing has been observed. The mean is updated each time a request completes: the first completion sets it, and each later one makes it $0.8$ times the previous mean plus $0.2$ times that request's time in service, from the iteration that took it off the queue to its completion. The queue wait must stay out of the mean, since the product already counts the queue: commit `8bb3474d5` (14 November 2026) moved the start of the clock from submission to admission for that reason (`mean_service_ms` in `crates/model/src/serve.rs`, tests `deadline_aware_uses_measured_service_time` in `crates/qwen3/src/serve.rs` and `mean_service_time_excludes_queue_wait` in `crates/model/tests/it/serve_mixed_step.rs`). A policy is a pure function of a `QueueState`, so it can be tested without a model.

A server has a door in front of the scheduler too, and the same question is asked there in coarser terms. The residency layer, which the `serving` crate assembles into the one shared executor of `brain serve` (`build_executor` in `crates/serving/src/executor.rs`), defines two limits that the transports apply (`crates/residency/src/admission.rs`): a concurrency ceiling at the edge, `EDGE_CONCURRENCY` (256 requests, enforced by the HTTP router of the `apiserve` crate), beyond which requests are shed at once, and a deadline on how long a request may wait to start on a lane, 15 minutes by default. That deadline is long by design, and commit `e910fd88f` (16 October 2026) says why: with a short default, the third and fourth concurrent long requests of an agent were refused while the first two were still making progress. "Waiting for running sequences is queueing, not overload." A request that waits behind running work that will end is not a request the server cannot serve, and refusing it turns a slow answer into none.

## What an iteration does, and in which order

Every policy so far was a rule about one decision. **Request scheduling** is the whole of them together: the order in which, in each iteration, a server resumes, admits, slices, preempts, decodes and releases, and the rules that choose among waiting and running requests at each of those points. The order is not a detail. Each step consumes something the next one needs, and putting them in another order changes who waits.

`Scheduler::iterate` in `crates/model/src/serve.rs` runs one iteration as five steps. It first brings back sequences that were preempted to host memory, oldest demotion first, as long as the pool can hold them, because work already in flight outranks work still queued; while any such sequence is still waiting, nothing new is admitted. It then admits and starts prefilling, in one of five ways chosen by what the decoder declares: whole prompts within a token budget, slices of one prompt, slices of several prompts in one pass, prompts read in by the decoder in the background while it decodes, or slices carried in the mixed pass. It then makes room for the decode step, preempting newest first, a sequence still prefilling before one that decodes, since it has produced nothing. It then runs one decode step over every running sequence, or the mixed pass. And it releases the blocks of every sequence that finished. The conformance suite in `crates/model/src/serve/conformance.rs`, which every decoder registered for `brain serve` must pass, states what the order buys from the outside: sequences decoded together produce what each produces alone, and a short request admitted beside a long decode finishes in about its own number of steps while the long one is still running.

Inside that order, each choice among requests has its own rule, and each rule has a reason:

- Admission is in arrival order, and a request that does not fit the pool yet holds the line, so a large request cannot be overtaken forever. A request whose adapter is still loading is the one exception: it is stepped over and asked again in the next iteration, so a cold adapter cannot delay requests for the base model behind it (`a_cold_adapter_does_not_delay_base_requests_behind_it` in `crates/model/tests/it/serve_adapters.rs`). Its own wait ends when the adapter is loaded, which the decoder has started.
- A request that shares a long beginning with a prompt still being read in (a shared system prompt or document) may be held back until that prefill has passed the shared part, so that the shared part is computed once and reused, instead of once per request (`waits_for_shared_prefill`). The wait is bounded by the progress of that one prefill, and the scheduler asks only when the shared part is at least two slices long.
- In a mixed pass, the prompt rows go to the prompt with the fewest rows left, ties to the older, each taking all it can use before the next gets a row (`allot_prefill_shortest_first`): a prompt read in whole starts decoding sooner, which lowers the mean wait for a first token at the same total work. The oldest prompt is given its minimum slice first, so a long prompt behind a stream of short ones still advances.
- Preemption takes the newest sequence first, since it is the furthest from finishing and from giving its blocks back.
- When nothing waits and every running sequence is greedy, the decode step runs a window of several tokens per host round trip, as many as the shortest remaining budget and the decoder's window allow (`decode_window_capacity`), and falls back to one token per step as soon as anything waits, so that admission is never delayed by a window.

The last rule is the one place where the scheduler trades the latency of waiting requests for the cost of the running ones, and it does so only when there is no waiting request to pay for it.

## From a connection to a row of the batch

Between a client's connection and a row of a decode step stand two more layers, and continuous batching has to hold across both of them, or a scheduler that batches perfectly is fed requests one batch at a time.

The `llmserve` crate is the glue between the residency layer and the scheduler for every text model brain serves (`crates/llmserve/src/lib.rs`). A model family supplies a `Frontend`, which turns a request into a prompt, names its stop tokens and its adapter, and phrases its refusals; `run_batch_scheduled` and `run_open_scheduled` do the rest for every family: submit each request to the scheduler with its own sampling parameters and seed, stream every open request's new tokens once per iteration, cancel a request whose client stopped or whose stop string appeared (the scheduler returns its blocks at once), and turn a refusal from the step report into an answer for the client instead of leaving the client waiting. A request's prompt is rendered and tokenized on the front end's own intake threads, so that a long prompt's tokenization does not leave the device idle.

The residency layer's dispatcher groups queued requests for one model into a call to that model's instance, and originally a group took only the requests queued when it started: a request arriving while a long one decoded waited for the whole group to end, however continuous the scheduler inside it was. Commit `9faf2a5bc` (16 October 2026) made the group open: a model that declares a batch limit gets later requests for the same instance moved into the running group, up to the limit, and admitted between steps through `Instance::run_open` and a `JobFeed`, and each request is answered the moment it finishes. The dispatcher stops feeding a running group as soon as a request for another model that is not running waits, so that one busy model cannot hold the device for as long as its traffic keeps coming.

Knowledge #333 records how easily the layers hide each other. On a server with ten streaming clients, short requests all finished at the same instant as one long one, which looked exactly like a server without continuous batching. The scheduler was innocent. The open group had kept every request's progress channel until the group ended, so every stream ended with the longest request of its group; the same checkpoint had been registered as two residents, so requests for the base model and for its adapter never shared a batch; and the cache pool was sized for two long contexts, so beyond four clients every turn prefilled its whole history again. The lesson is the one the conformance suite encodes from the other side: when the symptoms of a missing scheduler appear, test the scheduler in isolation first, and then look at what sits between it and the client.

## Summary

A server's device does its work in passes, and every decision about requests is a decision about what the next pass carries. **Static batching** fixes a batch for its whole life and leaves rows idle from the moment their requests finish; **continuous batching** lets requests join and leave at every iteration, over a paged cache whose blocks a sequence takes and returns on its own, and on the example it served the same 178 tokens in 56 iterations instead of 85. Priced in equivalent bytes, a decode step is bandwidth-bound, so its rows cost little and an empty row is waste, while a prefill is compute-bound and, as a pass of its own, makes every running sequence wait for it. **Chunked prefill** reads a prompt in bounded slices across iterations and bounds that wait, at the price of later first tokens when the slices run as passes of their own; carried in the decode pass, the slices share its weight read, and on the burst of eight prompts that gave the shortest gap for the running sequence, the earliest first token and nearly the least work. **Admission control** refuses at once what can never fit, charges a request the blocks its prompt and first token need beside what the running sequences need next, and lets sequences grow, giving back the newest by recompute when they outgrow the pool together; and at the door of the queue it refuses requests that would be served too late, which on the overload example turned 31 useful answers into 210. **Request scheduling** puts the steps of an iteration in an order in which work in flight outranks work queued, and chooses among requests with rules that each come with a bound on how long anyone can be held back.

The limits are those of the model: the costs are a roofline model of one illustrative device, not measurements; the overload server serves one request at a time with a known cost, which a batched server does not; and the schedules are deterministic, while real arrivals are not. The scheduler described here produces exactly one token per sequence per pass. Speculative decoding breaks that: a small model proposes several tokens, and the large model checks them all in one pass, without changing what it would have said.

## Check your understanding

1. A colleague proposes to keep static batching but to sort the waiting requests by the number of tokens they will produce before forming batches. Why can a server not do this, and what is left of the idea that it can do?
2. The burst table shows that the policy "chunks of 256, own pass" finishes the eight new requests later than the policy that prefilled all eight prompts in the first iteration. Explain where the extra work comes from, using the price table.
3. Predict, without running anything, what happens to the longest gap of request 0 under the mixed pass if the slices are made 64 rows instead of 256, and what happens to the last first token.
4. Find the error: "Admission charges a request for the blocks of its prompt. A prompt of 96 tokens with blocks of 16 tokens fills exactly 6 blocks, so admitting it needs 6 free blocks."
5. Under preemption by recompute, a preempted sequence that had produced 50 tokens of a 96-token prompt is prefilled again over 145 tokens, not 146. Why one fewer, and where does that token go?
6. A server with a deadline-aware door estimates the work ahead of a request as the number of waiting requests times a mean taken from completed requests. In which direction does the estimate err if completed requests waited in the queue before they ran, and what does that do to the number of requests refused?
7. A request's adapter is still loading. Why does the scheduler step over it instead of making it hold the line, as it does for a request that does not fit the pool, and why is the same treatment wrong for the request that does not fit?

## References

- Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim and Byung-Gon Chun, *Orca: A Distributed Serving System for Transformer-Based Generative Models*, 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI), 2022. Iteration-level scheduling: the batch is rebuilt before every pass of the model.
- Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang and Ion Stoica, *Efficient Memory Management for Large Language Model Serving with PagedAttention*, 29th ACM Symposium on Operating Systems Principles (SOSP), 2023. The paged cache over which a continuously changing batch is cheap, and preemption by swapping or recomputation when it runs out.
- Amey Agrawal, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani and Ramachandran Ramjee, *SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills*, arXiv preprint 2308.16369, 2023. Prompts cut into chunks, with decode rows carried in the chunk's pass.

## Exercises

1. **[Calculate]** Six requests that produce 12, 3, 7, 5, 9 and 2 tokens arrive together at a server with 2 slots. Find, for static and for continuous batching with the rules of `batching.static_vs_continuous`, the number of iterations, the percentage of useful slot-iterations and the mean wait.
2. **[Calculate]** With the cost rules of `batching.step_costs`, find the FLOPs, the bytes and $B_{\mathrm{eq}}$ of a decode step of 32 sequences at context 1,024, and explain why its bytes equal those of 16 sequences at context 2,048.
3. **[Calculate]** A 2,048-token prompt is read in slices of 256 rows and, separately, of 64 rows. How many bytes of earlier positions do the slices read in each case, and how do they compare with one read of the weights?
4. **[Calculate]** A pool holds 40 blocks of 16 tokens. Is a request with a prompt of 100 tokens and a ceiling of 600 new tokens refused at once? With a ceiling of 500? How many blocks is the second charged at admission when nothing else runs?
5. **[Calculate]** Under the unbounded queue of `batching.overload`, derive in closed form the index of the first request that is late at an offered load of 1.5 and at 2, and check both against the table.
6. **[Implement]** Reproduce `batching.static_vs_continuous` from `static_slots` and `continuous_slots`, then run `continuous_slots` on the twelve requests sorted by the number of tokens they produce (shortest first, ties by request number), an order a real server cannot know, and report its iterations and mean wait.
7. **[Implement]** Reproduce `batching.step_costs` and `batching.burst` from `forward_work`, `forward_cost50` and `run_burst`, using the $F(c, n)$ of Volume V, Chapter 12. Every printed digit must agree.
8. **[Implement]** Reproduce `batching.pool_admission` from `run_pool`, then run the three policies again with a pool of 64 blocks and report, for each, the iterations, the mean batch, the prompt tokens prefilled, the preemptions and the work, or the iteration at which the pool runs dry.
9. **[Implement]** Reproduce `batching.overload` from `run_overload`, and add a depth rule with a depth of 10. Explain why it agrees with the deadline-aware rule in this model and in which kind of server it would not.
10. **[Investigate]** Read `Scheduler::admit_one` and `Scheduler::ensure_decode_room` in `crates/model/src/serve.rs` and list every term of the number of blocks a request must find free to be admitted. For each, name the failure that would follow if it were left out, and the test in `crates/model/tests/it/` that would catch it.
11. **[Investigate]** Read `StepCost::prompt_rows` in `crates/model/src/serve/budget.rs`. For a fitted fixed part of 20 units, 0.1 units per row, 16 decode rows, a target of 45 units, a minimum slice of 64 rows, room for 1,024 rows and a tile of 128 rows, compute the prompt rows it returns, and compare with the test `the_rows_fill_the_last_tile_of_the_products`.

## Key terms

- **static batching**: forming a batch from the requests waiting when the previous one ended and running it until its longest member is done, so that rows of finished requests stay idle and new requests wait for the whole batch. <!-- concept: static-batching -->
- **continuous batching**: rebuilding the batch before every pass of the model, so that finished sequences leave and waiting requests take their rows at every iteration; also called iteration-level scheduling. <!-- concept: continuous-batching -->
- **chunked prefill**: reading a prompt into the cache in slices of a bounded number of rows over several iterations, with the running sequences decoding between or beside the slices, so that a long prompt cannot stall them. <!-- concept: chunked-prefill -->
- **admission control**: deciding, before any work is spent on a request, whether to take it in now, let it wait or refuse it, from what it will need (blocks of cache, a deadline) and what the server has. <!-- concept: admission-control -->
- **request scheduling**: the order of the steps of each iteration (resume, admit, prefill, make room, decode, release) and the rules that choose among waiting and running requests at each step, each with a bound on how long a request can be held back. <!-- concept: request-scheduling -->

## Answers

**Recall.**

1. The prefill phase is limited by its arithmetic once the prompt is long, the decode phase by the bytes of the weights and the cache. A prefill of the 7.5-billion decoder reaches the ridge of 50 FLOP per byte from 54 prompt tokens.
2. The weights are read once for the whole step; each sequence reads its own cache. Every added sequence brings its own cache read, so the intensity tends to one sequence's FLOPs over its own cache bytes, 31.96 FLOP per byte at context 4,096, below the ridge of 50.
3. $\lceil 97 / 16 \rceil = 7$ blocks. Position 113 (the 113th, index 112) is the first of a new block, since 112 is a multiple of 16, so the append takes one more block from the pool, and the sequence holds 8 (`blocks_for` in `crates/qwen3/src/serve.rs` is the division rounded up; the block table is in `crates/model/src/paged.rs`).
4. What is left after the reserved headroom and the bytes already accounted to residents: total minus reserved minus used (`Budget::free` in `crates/residency/src/budget.rs`).

**Check your understanding.**

1. A server does not know how many tokens a request will produce: it stops at a stop token the model chooses, and the request names only a ceiling. What is left is to stop needing to know: continuous batching lets each request leave when it ends, whatever its length, which captures the gain without the forecast.
2. Each of the 64 iterations that read a slice runs the slice as a pass of its own, compute-bound, and then a decode pass, bandwidth-bound, which reads the weights again (about $15 \times 10^9$ equivalent bytes). The policy that prefilled all prompts at once paid the decode passes only for the iterations it ran, 31 of them; the chunked policy runs 94. The prompt work itself is nearly the same.
3. The longest gap falls almost to a plain decode step, since 64 rows are only a little past the knee of about 50 rows (the table below gives 19.8 against 77.9). The last first token comes later, 4.899 $\times 10^{12}$ against 4.825 (the table below): each prompt needs 32 slices instead of 8, and every one of the extra iterations pays the decode rows' bytes, which with 64 prompt rows the slice no longer hides.
4. The request is charged for its prompt and its first token, 97 positions, which is $\lceil 97 / 16 \rceil = 7$ blocks, plus the blocks the running sequences need for their next positions. With only the prompt's 6 blocks, the first decode step would find no block for position 96.
5. The last token produced has no position in the cache yet: it is the input of the next decode step, which writes its keys and values. Recompute rebuilds the cache over the prompt and the first 49 produced tokens, 145 positions, and the 50th token travels with the sequence as its next input.
6. It overestimates the work ahead, since a completed request's time from submission to completion includes its own wait in the queue, which grows with the queue. The rule then refuses requests that would have started in time, more of them the longer the queue, which is why brain's mean starts when a request leaves the queue.
7. A loading adapter is something the scheduler has started and that will end without anyone else's help, and the requests behind it may be for the base model, which nothing stops; holding them back would make an unrelated request wait for the adapter. A request that does not fit the pool waits for blocks that the requests behind it would also take: if smaller ones were let past, they would keep taking the blocks it waits for, and it could wait forever.

**Exercises.**

1. Static: batches (12, 3), (7, 5), (9, 2) take 12, 7 and 9 iterations, 28 in all; 38 useful of 56 slot-iterations, 67.86 percent; first iterations 1, 1, 13, 13, 20, 20, mean wait $(0 + 0 + 12 + 12 + 19 + 19) / 6 = 10.33$. Continuous: requests start at 1, 1, 4, 11, 13, 16 and the last ends at 21; 38 useful of 42, 90.48 percent; mean wait $(0 + 0 + 3 + 10 + 12 + 15) / 6 = 6.67$.
2. FLOPs $32 \times F(1024, 1) = 497{,}494{,}786{,}048$; bytes $15{,}009{,}316{,}864 + 32 \times 1{,}024 \times 131{,}072 = 19{,}304{,}284{,}160$; $B_{\mathrm{eq}} = \max(19{,}304{,}284{,}160,\ 497{,}494{,}786{,}048 / 50) = 19{,}304{,}284{,}160$, about $19.304 \times 10^9$. The cache bytes depend only on the total number of cached positions read, $32 \times 1{,}024 = 16 \times 2{,}048 = 32{,}768$.
3. Slices of 256: $256 \times (0 + 1 + \cdots + 7) \times 131{,}072 = 939{,}524{,}096$ bytes, 6.3 percent of one weight read of 15,009,316,864. Slices of 64: $64 \times (0 + 1 + \cdots + 31) \times 131{,}072 = 4{,}160{,}749{,}568$ bytes, 27.7 percent. For a prompt of $p$ tokens in $k$ equal slices the re-read is $p(k - 1)/2$ positions, so it grows in proportion to the number of slices.
4. The pool holds 640 positions. $100 + 600 - 1 = 699 > 640$: refused at once. $100 + 500 - 1 = 599 \le 640$: admitted. It is charged $\lceil 101 / 16 \rceil = 7$ blocks.
5. Request $k$ arrives at $k a_{\mathrm{gap}}$ and, with the device always busy, starts at $k s_{\mathrm{svc}}$, so it waits $k(s_{\mathrm{svc}} - a_{\mathrm{gap}})$. It is late when that exceeds 60: at load 1.5 ($a_{\mathrm{gap}} = 4$) when $2k > 60$, first at $k = 31$, so 31 requests (0 to 30) are in time; at load 2 ($a_{\mathrm{gap}} = 3$) when $3k > 60$, first at $k = 21$, so 21 are in time. Both match the table.
6. Implementation: the printed tables are the check. Shortest first, the schedule takes 59 iterations, three more than arrival order, because the 40-token request starts last and runs alone at the end, while the mean wait falls from 13.33 to 7.25 iterations (the table below): knowing lengths would help the mean wait, not the total.
7. Implementation: the printed tables are the check.
8. Implementation: the printed table is the check. With 64 blocks (the last table below): reserving the ceiling runs two sequences at once, and the prompt-only charge with recompute needs one preemption instead of five; without preemption the pool still runs dry, at iteration 241. A larger pool narrows the gap, and does not remove the need for a plan.
9. Implementation: the printed table is the check, and the depth of 10 gives exactly the deadline-aware rows. When $q > 0$ the device is busy and the request in service has $r$ units left, $0 < r \le 6$, so the wait is $6q + r$: for $q \le 9$ it is at most 60, for $q \ge 10$ it is more than 60, and the two rules admit the same requests. In a server whose requests have different costs, or that serves many at once, the count of waiting requests no longer says how long the wait is.
10. The blocks of the prompt and its first token (without the first token's block, the first decode step can find the pool empty); the blocks of a hybrid decoder's state row (`sequence_state_blocks`; without it a burst of admissions promises the same blocks to prompts and to state rows, `a_burst_of_admissions_is_charged_for_the_state_rows_it_will_take`); the next-token blocks of every running sequence and the blocks still owed to prompts being prefilled (`committed_blocks`; without them the newcomer is preempted before it produces anything, or the decode step runs dry, `serve_kv_pressure.rs`); less the blocks the prompt adopts from the prefix cache (`cached_prefix_blocks`; without it a long conversation's next turn is never admitted, `serve_prefix_admission.rs`).
11. The target leaves $(45 - 20) / 0.1 = 250$ rows, 234 of them prompt rows beside the 16 decode rows, above the minimum of 64 and below the room of $1{,}024 - 16$. Rounded up to fill the last tile of 128 rows, $250 \to 256$, so it returns 240 prompt rows, which is what the test asserts.

The runs behind answer 3 of the questions and answers 6 and 8 of the exercises, printed by the module so that they can be checked against it:

<!-- worked: batching.variations -->
The burst of `batching.burst` under the mixed pass with the slice of 256 rows and with a slice of 64 rows (the last first token is the cumulative cost at which the last of requests 1 to 8 has its first token; costs as in that block):

| slice rows | iterations | longest gap of request 0 (10^9) | last first token (10^12) | all new done (10^12) |
|---:|---:|---:|---:|---:|
| 256 | 95 | 77.9 | 4.825 | 5.313 |
| 64 | 287 | 19.8 | 4.899 | 5.375 |

The twelve requests of `batching.static_vs_continuous` on 4 slots under the continuous rule, in arrival order and sorted by tokens produced, shortest first, ties by request number:

| order | iterations | mean wait (iterations) |
|---|---:|---:|
| arrival order | 56 | 13.33 |
| shortest first | 59 | 7.25 |

The pool of `batching.pool_admission` made 64 blocks instead of 48, with every other rule unchanged:

| policy | iterations | mean batch | prompt tokens prefilled | preemptions | work (10^12) |
|---|---:|---:|---:|---:|---:|
| reserve the ceiling | 737 | 1.67 | 768 | 0 | 11.313 |
| prompt only, recompute | 379 | 3.25 | 1,104 | 1 | 6.034 |
| prompt only, no preemption | no free block at iteration 241, 3 sequences decoding |  |  |  |  |
<!-- /worked -->
