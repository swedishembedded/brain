<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Benchmarking and Performance Claims

*What makes a benchmark trustworthy, and what are the classic ways it lies?*

Serve 68 requests to a language-model server and count the tokens that come back. Submit all of them at once and the server produces 1,094.4 output tokens per second. Let only four be in flight at a time and it produces 515.0. The first number is more than twice the second, and a report that has room for one number will print the first. Now ask what the people who sent the requests experienced. In the first run the median request waited 2,241.3 milliseconds for its first token, and only 5 of the 64 measured requests got their first token within 600 milliseconds and never waited more than 60 between two tokens, the promise this server makes. In the second run every one of them did.

<!-- worked: benchmarking.opening_claims -->
The same engine and the same 68 requests (4 warm-up, 64 measured), served two ways. The engine is a model whose rates the author chose, and no number here is a measurement. Generator: the book's SplitMix64, seed 0xd49f7978cb897c2d (`seed_of` of the name `benchmarking.opening`, defined on the pseudocode page). Draw order: the 68 requests in order, the prompt length and then the output length of each. Rates in tokens per second and times in milliseconds are of the model engine that the block `benchmarking.request_timeline` specifies in full; 'goodput' counts only the tokens of requests that met the service-level objective.

| how the requests arrive | output tokens per second | goodput (tokens per second) | share meeting the objective | time to first token, median (milliseconds) | time to first token, 99th percentile | gap between tokens, 99th percentile |
|---|---:|---:|---:|---:|---:|---:|
| all 68 requests submitted at once | 1,094.4 | 78.0 | 0.078 | 2,241.3 | 4,230.2 | 52.6 |
| 4 requests in flight at a time | 515.0 | 515.0 | 1.000 | 27.3 | 49.6 | 42.8 |
<!-- /worked -->

The engine is a model whose every cost is derived from the floating-point operations (FLOPs) it does and the bytes it moves (it is specified below and nothing in it is a measurement), so neither row is noise and neither is a timing mistake. Each is a correct measurement of a different quantity. The first is the rate at which a full batch turns out tokens; the second, together with its goodput, is the rate at which the server keeps its promise. "The server does 1,094 tokens per second" is true and, offered as an answer to "how many users can this server keep happy", it is a lie.

That is the shape of almost every misleading benchmark. The arithmetic is right; the quantity is not the one the claim is about. Volume IV taught how to take one time honestly. The step after is deciding what a benchmark must compute, under which load, against which denominator and compared with what, so that its number means what its claim says. A handful of earlier ideas carry the weight, and it is worth bringing them back first.

Volume IV, Chapter 11 ("Measuring Performance Honestly") stopped the clock only after the device had finished. What does a clock read if it is stopped when the dispatch call returns, and what is the name for a timing that waits for the device or reads its own stamps? The same chapter judged a change by one number and not by the sum of a profile's rows: what is that number, and why can a kernel that got 14.9 percent faster by its own stamps make it worse? It also gave Amdahl's law: a part takes the share $p$ of the time and is made $g$ times as fast; what does the whole gain, and what is the most it can ever gain? And Volume II, Chapter 8 ("Experiments, Baselines and Claims") separated two questions about a difference: how large it is and whether chance could explain it. Which of the two does the effect size answer, and why can a difference of half a point be significant at five standard errors?

## What a benchmark number has to be about

A **benchmark** is a fixed procedure that runs a system on a stated workload and reduces what happened to a few numbers, so that two systems, or two versions of one, can be compared. The procedure is the instrument and the numbers are its readings. Everything that can go wrong with an instrument can go wrong here, and one thing more: a benchmark is usually built by the same people whose work it will judge.

**Benchmark validity** is the property that a benchmark's number measures the quantity the claim built on it is about. Four conditions make it up, and each classic lie breaks one of them.

1. *The computation is the right one.* The system produced the outputs it is supposed to produce. A faster run that returns different answers is a measurement of a different program.
2. *The workload is the one of the claim.* The requests, their sizes, the order and timing of their arrival, the state of the caches and the machine all match the use the claim speaks for, or the claim names the workload it does speak for.
3. *The denominator is stated and appropriate.* A rate is divided by the right time window; a fraction of peak is divided by a peak the device can reach, of the right kind of arithmetic or memory.
4. *The comparison is like for like.* Two numbers that are compared differ only in the thing the claim is about, and the difference is larger than what the procedure itself produces when nothing changes.

Validity is a different property from reproducibility. A benchmark can give the same number to four digits every time and still measure the wrong thing; Volume IV, Chapter 11 told the case in brain's history in which a serving benchmark reported a stable speedup over four runs because both arms ran the same attention kernel over sequences of length zero (commit a8cadf8c2), and the commit's own words were "reproducibility is not validity". A stable number is necessary and nowhere near sufficient.

## Fast and wrong: the output check that comes before any speed

The cheapest way to make a program faster is to make it do less, and the easiest way to make it do less is by mistake. Drop a normalization, skip a term of a recurrence, round the weights to fewer bits than the model tolerates: each of those reduces the FLOPs or the bytes of a step, so each one improves every speed number a benchmark prints. A benchmark that does not look at the outputs rewards breaking the model.

To see how cheaply, take a toy decoder small enough to print. Its state $\mathbf{h}$ has 8 numbers; each token $x$ updates it as $\mathbf{h} \leftarrow \mathrm{clamp}(\mathbf{A}\mathbf{h} + \mathbf{E}[x], -1, 1)$, and the next token is the index of the largest entry of $\mathbf{U}\mathbf{h}$. Greedy decoding has no randomness in it, so two correct implementations produce the same tokens, and any difference in the stream is a difference in the computation. The table runs the same probes in five forms and compares each stream with the reference position by position.

<!-- worked: benchmarking.fidelity_gate -->
A toy greedy decoder with random weights (a model, not a measurement), the same work in five forms, each checked against the reference output. Vocabulary 16, width 8: E [16, 8], A [8, 8], U [16, 8], every entry (integer(512) - 256) / 256, drawn E, then A, then U, each in row-major order, then the 3 prompts of 3 tokens, each integer(16). Generator: the book's SplitMix64, seed 0x4352f1db5c0863b2 (`seed_of` of the name `benchmarking.fidelity`, defined on the pseudocode page). The state h starts at zero; each token x (the prompt's, then each generated one) sets h = clamp(A h + E[x], -1, 1), the sum over j taken in ascending order and E[x] added last; after the prompt and after each generated token the next token is argmax(U h) (sum over j ascending, ties to the lowest id). 24 tokens per probe, arithmetic in double precision, one operation at a time, no fused multiply-add. The 8-bit form rounds every weight to a multiple of 1/64, halves away from zero. Weights read per generated token: one row of E (8), A (64) and U (128); stored in 2 bytes each (they are multiples of 1/256 in [-1, 1)) or 1 byte in the 8-bit form. FLOPs per generated token, a multiply-add counted as 2 and the clamp's comparisons not counted: 128 for A h, 8 for adding E[x], 256 for U h. The descending form reverses only the order of the sum over j in A h. Token numbers in the first-divergence column count the generated tokens from 0 (token 5 is the sixth). Agreement counts positions over the longer of each pair of streams; the gate passes only on exact agreement with at least one position compared. 'Common prefix' scores only the positions both streams have.

| form | bytes of weights per token | FLOPs per token | tokens agreeing / positions compared | first divergence | exact gate | common-prefix score |
|---|---:|---:|---:|---|---|---|
| the reference itself | 400 | 392 | 72 / 72 | - | pass | 1.000 |
| sums over j in descending order | 400 | 392 | 72 / 72 | - | pass | 1.000 |
| weights rounded to 8 bits | 200 | 392 | 59 / 72 | probe 1, token 5 | fail | 0.819 |
| recurrent term dropped | 272 | 264 | 16 / 72 | probe 0, token 1 | fail | 0.222 |
| stops after 16 tokens | 400 | 392 | 48 / 72 | probe 0, token 16 | fail | 1.000 |

The reference streams: probe 0: prompt [1, 3, 2], reference [13, 8, 14, 9, 9, 9, 11, 8, 14, 9, 9, 9, 11, 8, 14, 9, 9, 9, 11, 8, 14, 9, 9, 9]; probe 1: prompt [6, 9, 0], reference [9, 9, 9, 9, 8, 14, 14, 9, 12, 11, 8, 14, 14, 9, 9, 11, 8, 14, 0, 9, 3, 5, 8, 14]; probe 2: prompt [4, 0, 9], reference [8, 14, 9, 9, 9, 11, 8, 14, 9, 9, 9, 11, 8, 14, 9, 9, 9, 11, 8, 14, 9, 9, 9, 11].
<!-- /worked -->

Read the work columns as a speed report would. The 8-bit form reads 200 bytes of weights per token where the reference reads 400, so on a decoder whose steps are bound by bandwidth (Volume IV, Chapter 3) it is about twice as fast. The form that drops the recurrent term does 264 FLOPs and reads 272 bytes, less of both. Neither is the same program: the 8-bit form agrees on 59 of 72 positions and departs at the sixth token of the second probe, the other on 16. A speed table without the last three columns would rank the two broken forms first.

The second row is the other half of the lesson. Summing the eight products in the opposite order changes the last bits of some sums (addition of floating-point numbers is not associative, Volume IV, Chapter 1), and here it changes no token. A check that demanded identical bits of every intermediate would fail that row and every honest batching or tiling change with it; a check on the tokens passes it and fails the two forms that changed what the program computes. The gate has to sit at the level of the result the user receives.

The last row shows why the comparison has to run over every position either stream has. The early-stopping form produced 16 tokens of each 24, all of them right. Scored over the common prefix it agrees perfectly, 1.000; scored over the longer of each pair it agrees on 48 of 72 and fails. A run that stops early did produce different behaviour, and on a benchmark it also produced fewer tokens to time, which makes it look faster per request. The check, written out:

```text
function agreement(cand: list<list<int>>, ref: list<list<int>>) -> (int, int, bool):
    # tokens that agree, positions compared, and whether the exact gate passes
    agree = 0
    compared = 0
    for k in 0 .. min(len(cand), len(ref)):
        c = cand[k]
        r = ref[k]
        for p in 0 .. max(len(c), len(r)):
            compared = compared + 1
            if p < len(c) and p < len(r) and c[p] == r[p]:
                agree = agree + 1
    passed = compared > 0 and agree == compared and len(cand) == len(ref)
    return (agree, compared, passed)
```

Two clauses of the last line carry most of its value. A different number of streams fails even if the common ones match, and a comparison of zero positions fails, because a check that compared nothing has verified nothing.

### Where brain keeps the check

`crates/perf/src/fidelity.rs` is that function. `Fidelity::greedy` compares token streams and `Fidelity::exact_bytes` compares byte streams (the text a served endpoint sent back, an encoded image), and both are one generic `verdict`, so that "a second comparator with weaker semantics", in the file's own words, cannot creep in. It counts over the longer of each pair, records the first divergence, fails a different stream count, and fails an empty comparison. The default threshold, `EXACT`, is 1.0, for the reason the toy showed: greedy decoding is deterministic, so anything short of exact is a behavioural change.

The schema in `crates/perf/src/schema.rs` makes the verdict part of the result. A run whose check fails is written with `valid: false` and the reason, and `crates/perf/src/report.rs` excludes it from every comparison, printing it under "EXCLUDED". A run that never ran a check has `correctness.passed` written as `null`, not `true`, so that silence can never read as verification. Commit e60ec8151 (July 2026) records the order in which this was built: the batched-against-sequential check "landed FIRST as the safety net", before the three decode kernels the same commit replaced, so that every Tier-1 scenario of the suite (the core shapes run for every model and device) ran the check after its measurement from then on. The order is the lesson. A gate added after the optimization it should have judged has already let one number through.

How a server checks itself without a second implementation is the subtle part, and `crates/perf/src/targets.rs` has three answers. The paged engine runs the same three probe prompts one at a time and then all together through the same scheduler and demands identical greedy tokens, because batching is exactly the code path an optimization changes. The executor and the HTTP (Hypertext Transfer Protocol) targets do the same sequentially and concurrently, and `two_path_verdict` adds a step for an honest complication: if the two paths disagree, the sequential path is run again, and if it disagrees with itself the action has no deterministic output to gate, so the target reports no verdict rather than "manufacturing a verdict out of the model's own noise". The adapter for an arbitrary capability provider has only one code path and so has no check, deliberately: running one path twice compares an optimization with itself, passes by construction, "and a gate that cannot fail is worse than an absent one because it reads as verification". Commit fddf866a8 (August 2026) made that gap loud: the comparison report names every compared run whose check never ran, and the regression gate says its floors then "bound SPEED only".

One more case shows how a check can fail without the computation being wrong. Commit 6c3222ac4 (August 2026) found a run marked invalid with the message "greedy_token_match 1.0000 < 1.0000". The probe's own requests had been refused at admission, because the key-value pool had been sized for a smaller measured workload than the fixed probe, so zero positions were compared; an empty comparison scores 1.0 by convention and fails, and the message then printed an inequality between two equal numbers. The fix sized the pool to fit the probe and made `failure_reason` say "compared 0 positions" instead. The check was right to fail. Its message pointed at the wrong cause, which for a check whose readers are deciding whether to trust a number is a defect of its own.

## Requests in, tokens out: the end-to-end benchmark

The output check settles condition 1. Condition 2, the workload, needs a different kind of benchmark from the ones of Volume IV. A whole-pass measurement there timed one unit of work from request to result, with everything inside the bracket. A server never does one unit of work. It does many at once, and how long one request takes depends on how many others are in the machine with it and when they arrived.

An **end-to-end benchmark** drives a system through the entry point its users use, with a stream of requests whose sizes and arrival times are a stated workload, records for every request the moments at which it was submitted, admitted, produced each piece of output and finished, and reports distributions of those per-request times together with throughput, the goodput that met a stated service-level objective, and the output check. The word that matters is *distributions*: the user of a server is a request, and requests do not all have the same experience.

The timeline of one request is enough to define every quantity. A request is *submitted*; the scheduler *admits* it when there is room; its first output (for a decoder, the first token) arrives at *first*; further outputs follow; it is *done* with the last. Then:

- the **time to first token** (time to first artifact, for a model whose output is a frame or an audio chunk) is first minus submitted, which includes the time the request waited for admission;
- the **gap** between two consecutive tokens of one request is their difference, and a request has one fewer gap than tokens;
- the **end-to-end time** is done minus submitted;
- a **service-level objective** is a promise per request, here a budget for the time to first token and a budget for *every* gap;
- **throughput** is the output produced in a time window divided by the window, and **goodput** is the output of the requests that met the objective, divided by the same window.

The model engine that produced the opening table is now worth stating in full, since everything that follows runs on it. Its costs are counted work at stated rates, the way Volume IV taught speed to be stated: a prefill of a prompt is compute-bound (two FLOPs per parameter per token), a decode step is bandwidth-bound at small batches (the whole of the weights is read once per step, whatever the batch) and grows with the key-value bytes of the sequences it serves. The table lists the first ten requests of the run with four in flight.

<!-- worked: benchmarking.request_timeline -->
The model engine of this chapter, every parameter labelled, chosen by the author and none a measurement. A decoder of 1.5e9 parameters held in 16-bit numbers (3.0e9 bytes of weights) on the model GPU unit of Volume IV, whose kernels reach 20,000 (GFLOP/s) and 520 (GB/s) (0.58 and 0.80 of its measured roofline, 34,469 and 646). An iteration costs 0.4 model milliseconds of host work plus max(FLOPs / 20,000 (GFLOP/s), bytes / 520 (GB/s)). A prefill of a prompt of n tokens does 2 x 1.5e9 x n FLOPs and moves 3.0e9 + 32,768 n bytes and yields the request's first token; a decode step over b running sequences does 2 x 1.5e9 x b FLOPs and moves 3.0e9 bytes plus 32,768 bytes for every token of context each sequence holds (its prompt and the tokens it has produced); the arithmetic of attention is left out of the count. At most 16 sequences run at once; a waiting request is admitted (oldest first) when a place is free and the previous iteration was not a prefill, so at most one prefill falls between two decode steps. The first iteration of the engine's life costs 300 model milliseconds more (pipelines are built). Each run serves 4 warm-up requests and then 64 measured ones; a request's prompt is 64 + integer(257) tokens and its output 32 + integer(97), drawn in request order with the prompt first. Statistics use the measured requests only, over the window from the first measured submission to the last measured completion; percentiles interpolate as the pseudocode page's `percentile`; gaps are pooled over all measured requests. A request meets its service-level objective when its first token comes within 600 model milliseconds of its submission and no gap between two of its tokens exceeds 60.

The first 10 requests of the run with 4 in flight of `benchmarking.opening_claims` (the same 68 requests, the same seed). Requests are numbered from 0; times are model milliseconds from the start of the run.

| request | kind | prompt tokens | output tokens | submitted | admitted | first token | done | time to first token | longest gap | end to end | meets the objective |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | warm-up | 224 | 126 | 0.0 | 0.0 | 334.0 | 1,280.8 | 334.0 | 53.6 | 1,280.8 | yes |
| 1 | warm-up | 267 | 80 | 0.0 | 340.2 | 380.6 | 962.0 | 380.6 | 53.6 | 962.0 | yes |
| 2 | warm-up | 313 | 72 | 0.0 | 386.8 | 434.2 | 905.3 | 434.2 | 34.2 | 905.3 | yes |
| 3 | warm-up | 184 | 81 | 0.0 | 440.4 | 468.4 | 1,007.6 | 468.4 | 33.2 | 1,007.6 | yes |
| 4 | measured | 85 | 67 | 905.3 | 905.3 | 918.4 | 1,402.9 | 13.1 | 41.3 | 497.6 | yes |
| 5 | measured | 177 | 113 | 962.0 | 962.0 | 989.0 | 1,780.6 | 27.0 | 41.3 | 818.5 | yes |
| 6 | measured | 78 | 63 | 1,007.6 | 1,007.6 | 1,019.7 | 1,450.9 | 12.1 | 41.3 | 443.3 | yes |
| 7 | measured | 231 | 92 | 1,280.8 | 1,280.8 | 1,315.9 | 1,949.7 | 35.0 | 32.1 | 668.9 | yes |
| 8 | measured | 69 | 43 | 1,402.9 | 1,402.9 | 1,413.7 | 1,700.8 | 10.8 | 32.1 | 297.9 | yes |
| 9 | measured | 170 | 80 | 1,450.9 | 1,450.9 | 1,476.8 | 2,018.6 | 25.9 | 26.1 | 567.6 | yes |
<!-- /worked -->

Follow request 0. It is admitted at once and its first token arrives at 334.0 model milliseconds: 300 for the cold start and 34.0 for a prefill of 224 tokens, whose $2 \times 1.5 \times 10^9 \times 224 = 6.72 \times 10^{11}$ FLOPs take 33.6 milliseconds at 20,000 (GFLOP/s), plus 0.4 of host work. That is a warm-up request, and the cold start is exactly why it is one: no statistic sees it. Request 1 was submitted at 0.0 but admitted at 340.2, after the first prefill and one decode step (the scheduler never runs two prefills in a row while something is running), so its time to first token includes 340.2 milliseconds of waiting. The longest gap of request 0, 53.6, is a decode step with a prefill of another request in front of it: while the engine prefills a newcomer, every running sequence waits. That one rule of the scheduler is what an objective on *every* gap exists to catch. Request 4 is the first measured one; it is submitted at 905.3, the moment request 2 finishes, which is how a closed loop with four in flight works: a new request enters when one leaves.

The engine as pseudocode, with `prefill_ms` and `decode_ms` the two costs stated in the block above (host work plus the larger of FLOPs over the FLOP rate and bytes over the byte rate, in milliseconds):

```text
record Engine:
    n_warm: int          # leading requests of a run that no statistic sees (4)
    b_max: int           # sequences that may run at once (16)
    t_cold: real         # extra cost of the first iteration of the engine's life (300.0)

record Request:
    n_in: int            # prompt tokens
    n_out: int           # output tokens to produce
    warm: bool
    submit: real
    admit: real
    first: real
    done: real           # -1.0 until finished
    arts: list<real>     # the time of each output token, in order
    made: int            # output tokens produced so far

function serve_run(e: Engine, reqs: list<(int, int)>, mode: text, conc: int, arrivals: list<real>) -> list<Request>:
    require mode == "saturated" or mode == "closed" or mode == "rate"
    require mode != "closed" or conc >= 1
    require mode != "rate" or len(arrivals) == len(reqs)
    n = len(reqs)
    recs = []
    waiting = []                      # indices into recs, oldest first
    running = []
    t = 0.0
    nxt = 0
    in_flight = 0
    cold = true
    last_prefill = false
    if mode == "saturated":
        while nxt < n:
            append Request(reqs[nxt][0], reqs[nxt][1], nxt < e.n_warm, 0.0, -1.0, -1.0, -1.0, [], 0) to recs
            append nxt to waiting
            nxt = nxt + 1
            in_flight = in_flight + 1
    loop:
        if mode == "closed":
            while nxt < n and in_flight < conc:
                append Request(reqs[nxt][0], reqs[nxt][1], nxt < e.n_warm, t, -1.0, -1.0, -1.0, [], 0) to recs
                append nxt to waiting
                nxt = nxt + 1
                in_flight = in_flight + 1
        if mode == "rate":
            while nxt < n and arrivals[nxt] <= t:
                append Request(reqs[nxt][0], reqs[nxt][1], nxt < e.n_warm, arrivals[nxt], -1.0, -1.0, -1.0, [], 0)
                    to recs
                append nxt to waiting
                nxt = nxt + 1
                in_flight = in_flight + 1
        if len(waiting) == 0 and len(running) == 0:
            if nxt >= n:
                break
            if mode == "rate":
                t = arrivals[nxt]     # idle until the next arrival
                continue
        extra = 0.0
        if cold:
            extra = e.t_cold
        cold = false
        if len(waiting) > 0 and len(running) < e.b_max and (not last_prefill or len(running) == 0):
            i = waiting[0]
            waiting = waiting[1:len(waiting)]
            recs[i].admit = t
            t = t + prefill_ms(recs[i].n_in) + extra
            recs[i].first = t
            append t to recs[i].arts
            recs[i].made = 1
            if recs[i].made == recs[i].n_out:
                recs[i].done = t
                in_flight = in_flight - 1
            else:
                append i to running
            last_prefill = true
        else:
            ctx = [recs[i].n_in + recs[i].made for i in running]
            t = t + decode_ms(ctx) + extra
            still = []
            for i in running:
                recs[i].made = recs[i].made + 1
                append t to recs[i].arts
                if recs[i].made == recs[i].n_out:
                    recs[i].done = t
                    in_flight = in_flight - 1
                else:
                    append i to still
            running = still
            last_prefill = false
    return recs
```

and the reduction of the records to numbers, which carries most of the rules that keep a serving benchmark honest:

```text
function meets(r: Request, first_ms: real, gap_ms: real) -> bool:
    if r.done < 0.0:
        return false                      # an unfinished request is never good output
    if r.first - r.submit > first_ms:
        return false
    for k in 1 .. len(r.arts):
        if r.arts[k] - r.arts[k - 1] > gap_ms:
            return false                  # one stall fails the request, whatever the mean gap
    return true

function summarize(recs: list<Request>, first_ms: real, gap_ms: real) -> (real, real, real):
    m = [r for r in recs if not r.warm]   # warm-up requests never enter a statistic
    require len(m) >= 1
    t0 = min([r.submit for r in m])       # the window starts at the first measured submission ...
    t1 = max([r.done for r in m])         # ... and ends at the last measured completion
    wall = (t1 - t0) / 1000.0             # seconds of the model
    out = sum([r.made for r in m])
    good = sum([r.made for r in m if meets(r, first_ms, gap_ms)])
    met = len([r for r in m if meets(r, first_ms, gap_ms)])
    return (out / wall, good / wall, met / len(m))
```

`summarize` returns the throughput, the goodput and the share of requests that met the objective; the percentiles of the tables are `percentile` of the lists of times to first token, of pooled gaps and of end-to-end times of `m`. In this model every request finishes, and `summarize` therefore needs the guard of `meets` only in principle; a real server's failed or refused requests stay in the denominator (`len(m)`).

### What brain measures, and how it refuses to flatter

`crates/perf` is brain's end-to-end suite, and its header answers "how fast, and at what cost, and is it still correct?" for "models times hardware times configuration". Its seam, `PerfTarget` in `crates/perf/src/target.rs`, is the timeline above: a target accepts requests and reports *emissions*, each an admission, an artifact, a completion, a failure or a refusal, with the moment it happened. Because the unit is an artifact and not a token, the same suite measures a decoder (time to first token, gaps, time per output token), an image generator (each denoising step an artifact) and a one-shot detector, where time to first artifact collapses into end-to-end time with no special case.

`crates/perf/src/metrics.rs` turns emissions into numbers and enforces the rules of `summarize` rather than trusting callers with them. `Summary::build` drops the warm-up records itself, "so no caller can accidentally include them". `ReqRecord::meets` fails a failed or unfinished request whatever its latencies, and fails a request with any single gap over budget, "one long stall is exactly the user-visible failure an averaged ITL would hide" (ITL, inter-token latency, is the gap). Goodput counts only the output of requests that met their objective. A failed request stays in the denominator. A request the engine *refused* at admission is counted apart, neither completed nor failed, and leaves the denominator of the objective's attainment, so that a policy that sheds hopeless work under overload is not scored as missing deadlines it never accepted. `crates/perf/src/driver.rs` takes the window from the first measured submission to the last measured completion, so that the warm-up never stretches the denominator, and is single-threaded on purpose: an asynchronous runtime in the harness would fold its own scheduling noise into the engine's numbers. The latency distributions are kept as every sample (`Dist` in `crates/stats/src/latency.rs`) and their percentiles are nearest-rank, the sample at rank $\lceil qn \rceil$; the pseudocode page's `percentile` interpolates instead, so the two can differ at the tail of a small sample, and the tables of this chapter use the interpolated one. And `Workload::ignore_stop` in `crates/perf/src/workload.rs` forces every request to its full output length in synthetic runs, because requests that stop early "silently shorten the workload and inflate the rate", the early-stopping row of the toy decoder in another form.

## The arrival process is part of the workload

The opening table differed in one thing only, how the requests arrived. That is a choice with more than two answers, and it changes which engine is being measured. A **closed loop** keeps a fixed number of requests in flight and submits a new one when one finishes: the arrival rate follows the engine's speed, so the engine can never be overloaded. An **open loop** submits requests at times drawn independently of the engine, here a Poisson process (exponential gaps between arrivals): if the engine is slower than the arrivals, the queue grows without limit. **Saturated** is the extreme closed case in which everything is available at time zero. Schroeder, Wierman and Harchol-Balter (2006) showed that the choice between closed and open load generators can change measured response times and even which scheduling policy wins, and that most load generators then in common use were closed. The same 68 requests under six processes:

<!-- worked: benchmarking.arrival_processes -->
The 68 requests of `benchmarking.opening_claims` under six arrival processes. The open-loop rows draw their arrival times from a second generator: Generator: the book's SplitMix64, seed 0x131b075a37fdc2af (`seed_of` of the name `benchmarking.arrivals`, defined on the pseudocode page). Draw order: 68 gaps for the rate of 8 per second, then 68 for the rate of 20; a gap is -ln(1 - u) / rate seconds with u = rng.uniform() and ln the library natural logarithm, multiplied by 1000 to the model milliseconds that serve_run uses, and request j arrives at the sum of the first j + 1 gaps. Rates per second and times in milliseconds of the model engine, whose rates the author chose (no number here is a measurement).

| arrival process | requests per second | output tokens per second | goodput (tokens per second) | share meeting the objective | first token, median (milliseconds) | first token, 99th percentile | gap, 99th percentile | end to end, median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| saturated: all at once | 13.20 | 1,094.4 | 78.0 | 0.078 | 2,241.3 | 4,230.2 | 52.6 | 3,259.5 |
| closed loop, 1 in flight | 1.87 | 155.0 | 155.0 | 1.000 | 26.9 | 47.2 | 6.2 | 528.5 |
| closed loop, 8 in flight | 9.25 | 767.4 | 767.4 | 1.000 | 29.9 | 554.6 | 49.7 | 720.2 |
| closed loop, 16 in flight | 13.20 | 1,094.4 | 982.6 | 0.891 | 36.4 | 768.0 | 52.6 | 1,116.4 |
| open loop, 8 requests per second | 8.13 | 674.5 | 674.5 | 1.000 | 40.6 | 144.4 | 49.6 | 722.2 |
| open loop, 20 requests per second | 13.74 | 1,139.5 | 782.4 | 0.672 | 467.5 | 1,201.9 | 52.6 | 1,485.5 |
<!-- /worked -->

One request at a time is the engine's latency floor: a gap of 6.2 milliseconds, one decode step of $3.0 \times 10^9$ bytes at 520 (GB/s) plus a few kilobytes of context and the host's 0.4, and 155.0 tokens per second, because each step reads all of the weights to produce a single token. Batching reads the weights once for many sequences, so the output rate climbs to 767.4 at eight in flight and to 1,094.4 at sixteen, where the batch reaches its cap of 16 sequences. The saturated run reaches the same output rate: once the batch is full, more waiting requests add only waiting. What changes is who waits. At 8 in flight every request meets the objective; at 16, 0.891 do; saturated, 0.078.

The open-loop rows show what a closed loop can never show. At 8 requests per second, below what the engine can sustain, every request meets the objective and the median first token comes after 40.6 milliseconds. At 20 per second the engine cannot keep up: its output rate is near the full-batch maximum, 1,139.5 (slightly above the closed rows only because its measured window is shaped differently), and the queue grows for as long as requests keep coming, so the median request waits 467.5 milliseconds for its first token and a third of them miss. A closed loop at 16 in flight and an open loop at 20 per second have about the same throughput and differ in everything a user notices.

So the arrival process is part of the workload, recorded with the result and never averaged across. `crates/perf/src/scenarios/mod.rs` says it in one sentence, "requests that all exist at t=0 exercise a different engine than requests arriving over time", and its three core scenarios differ "**only** in their arrival process": `throughput` is saturated and documented as "the efficiency ceiling", `latency` is a closed loop with a fixed batch and nothing queued behind it, `serve` and `sweep` are closed loops at a level of concurrency, and the `overload` scenario offers load past capacity to see whether the engine sheds it or collapses. `Arrival::Rate` in `crates/perf/src/workload.rs` draws gamma-distributed gaps, Poisson when its burstiness parameter is 1, from the workload's seed, and the comparison report warns when two results used different processes.

Commit dff98364d (July 2026) is a small story about a benchmark that measured nothing real. The first adapter for resident models ran each request synchronously, so a run at a concurrency above 1 serialized on a lock and reported, in the commit's words, "nothing real": the concurrency knob was turned and the engine never saw two requests at once. The commit added `ExecutorTarget`, which drives the scheduler, budgets and device lanes that production serving uses, and recorded a second correction in the same breath: the latency floor is a run at concurrency 1, because the default of 8 "measures queueing, not service". Both are condition 2. The workload the claim names has to be the workload the system receives.

## The deliverable is the knee, not the peak

If the output rate is not the answer, what is? A server's operator wants the largest load at which the promise holds. Run a closed-loop ladder and read it off:

<!-- worked: benchmarking.concurrency_sweep -->
A closed-loop ladder over the same 68 requests of `benchmarking.opening_claims` (the same seed, the same requests at every level, the engine reset between levels so that each level's first iteration pays the cold start, which falls on a warm-up request). Rates per second and times in milliseconds of the model engine, whose rates the author chose (no number here is a measurement).

| requests in flight | output tokens per second | goodput (tokens per second) | share meeting the objective | first token, 99th percentile (milliseconds) | gap, 99th percentile (milliseconds) |
|---:|---:|---:|---:|---:|---:|
| 1 | 155.0 | 155.0 | 1.000 | 47.2 | 6.2 |
| 2 | 287.3 | 287.3 | 1.000 | 47.2 | 22.0 |
| 4 | 515.0 | 515.0 | 1.000 | 49.6 | 42.8 |
| 8 | 767.4 | 767.4 | 1.000 | 554.6 | 49.7 |
| 12 | 960.1 | 921.7 | 0.953 | 657.1 | 51.3 |
| 16 | 1,094.4 | 982.6 | 0.891 | 768.0 | 52.6 |
| 20 | 1,094.4 | 905.1 | 0.828 | 1,248.1 | 52.6 |
| 24 | 1,094.4 | 741.0 | 0.688 | 1,446.6 | 52.6 |
| 32 | 1,094.4 | 78.0 | 0.078 | 1,915.7 | 52.6 |
| 48 | 1,094.4 | 78.0 | 0.078 | 3,018.2 | 52.6 |

Peak output rate: 1,094.4 tokens per second at 16 in flight. Highest goodput among the levels where at least 0.99 of the requests met the objective: 767.4 tokens per second at 8 in flight.
<!-- /worked -->

![Output tokens per second and goodput of the model engine against the number of requests in flight on a logarithmic axis: both rise together to 767 at 8; the output rate keeps rising to 1,094 at 16 and stays flat, while goodput peaks near 983 at 16 and falls to 78 at 32 and 48.](img/concurrency-sweep.svg)

Two numbers come out of the same ladder. The peak output rate is 1,094.4 tokens per second at 16 in flight. The highest goodput at which the objective held for at least 0.99 of the requests is 767.4 at 8 in flight, 0.70 of the peak. Between them is the knee, where the time to first token's 99th percentile passes the budget of 600 milliseconds: at 8 it is 554.6 and at 12 already 657.1. Goodput peaks at 16 (982.6) with 0.891 of the requests served on time, and an operator who promised the objective to 99 percent of requests cannot run there. Above 16 the batch cannot grow, every extra request is a request in a queue, and goodput falls to 78.0, the five requests of the 64 that happened to be early in it.

The gap budget is never what fails in this model, and it is worth seeing why. A gap is a decode step plus at most one prefill in front of it, because the scheduler interleaves; the longest prompt is 320 tokens, whose prefill does $9.6 \times 10^{11}$ FLOPs in 48.0 milliseconds, and with a decode step that stays under the budget of 60. A scheduler that ran prefills back to back would raise the output rate a little and show its cost only in the per-request latencies, the first-token waits and the longest gaps; that is the reason `meets` judges each request on its own gaps and wait and not on a mean.

`run_sweep` in `crates/perf/src/scenarios/mod.rs` reports both numbers apart on purpose. Its doc comment says that the deliverable is not the peak rate but the "**highest goodput that still satisfies the SLO**". It records `max_sustainable_concurrency` and `max_goodput_per_s` among the levels where attainment reached 0.99, and it records `peak_output_artifacts_per_s` beside them, and the test `sweep_reports_max_sustainable_not_just_peak` keeps the two from being merged. When no level holds the objective the renderer says so ("the workload exceeds this configuration") rather than printing a peak. The arithmetic behind goodput and the attainment of an objective has a recent literature of its own in language-model serving; Zhong and coauthors (DistServe, OSDI 2024) used goodput under a latency objective as the measure by which to design a server that separates prefill from decoding, which is a direct answer to the stall above.

## A kernel's gain seen from a request

Condition 3, the denominator, has a version that is not about peaks at all: a change to one kernel is reported with the kernel as its denominator, and the claim is about the server. Amdahl's law (1967) converts one into the other, and it needs the shares of the server's time, not of a profile. Suppose the decode kernels are made to read half the bytes, by storing the weights in 8 bits (the output check of the first section would then have to pass, which this model takes for granted). A decode step's dominant term halves. What does a request see?

<!-- worked: benchmarking.amdahl_request -->
The engine's busy time (the model engine of `benchmarking.request_timeline`, whose rates the author chose; no number here is a measurement) for all 68 requests split into the host's 0.4 model milliseconds per iteration, the prefill work and the decode work (the larger of its FLOP time and byte time, without the host's 0.4; the cold start left out), for the closed-loop runs with 1 and with 16 in flight of `benchmarking.concurrency_sweep`. The change: weights stored in 8-bit numbers and nothing else, so a decode step reads 1.5e9 bytes of weights in place of 3.0e9 (a prefill is compute-bound and its count is unchanged). g is the decode work before over after, from a rerun of the changed engine; A(p, g) = 1 / ((1 - p) + p / g) with p the decode share; 'rerun' is the output rate of the changed engine over the original's, measured over each run's own window.

| requests in flight | host share | prefill share | decode share p | g on decode | Amdahl A(p, g) | rerun: new output rate / old | ceiling 1 / (1 - p) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.062 | 0.053 | 0.885 | 1.995 | 1.790 | 1.791 | 8.698 |
| 16 | 0.042 | 0.426 | 0.532 | 1.933 | 1.346 | 1.317 | 2.137 |
<!-- /worked -->

The same kernel change is worth two different things. With one request in flight, decoding is 0.885 of the engine's work, the change makes it 1.995 times as fast (not quite 2, since the key-value bytes read at each step did not shrink), and the law predicts 1.790 for the server, which the rerun confirms at 1.791. With sixteen in flight, the batch shares each read of the weights among sixteen sequences, so decoding is only 0.532 of the work and prefill, untouched by the change, is 0.426: the law predicts 1.346 and the rerun gives 1.317. The rerun differs from the prediction there because the law assumes the rest of the work is unchanged, and in a server it is not quite: faster decode steps finish requests sooner, so the next prefills come sooner and interleave differently, and the queue changes shape. That is the general caveat of Volume IV turned into a serving fact. Amdahl's law is the right first estimate and the end-to-end rerun is the measurement.

So a claim of the form "decode is twice as fast" is a claim about a kernel. A claim about the server names its load, because the same change is worth 1.79 times at one request in flight and 1.32 times at sixteen, and at sixteen the ceiling for any decode improvement whatever is 2.137.

## Paper peaks and probed peaks

The denominator that most performance claims share is "peak": a kernel "reaches 80 percent of peak", an engine "uses 60 percent of the hardware". The phrase has two very different meanings. A **nominal peak** is the rate computed from the device's specification: for arithmetic, the number of lanes times 2 FLOPs per fused multiply-add times the highest (boost) clock; for memory, the width of the bus in bytes times the data rate of each pin. It assumes every lane does a fused multiply-add on every cycle at the boost clock, or every pin transfers on every beat, which nothing does for long: a device under sustained load lowers its clock, and memory refresh and protocol overheads take their share of the beats. A **measured peak** is the rate a probe reaches on the device that ran the work: a chain of fused multiply-adds held in registers for arithmetic, a stream over buffers far larger than any cache for bandwidth, the measured roofline of Volume IV, Chapter 10 (the roofline model is that of Williams, Waterman and Patterson, 2009). **Measured versus nominal peak** names the distinction and the rule that follows from it: a fraction of peak states which peak it divides by, divides by a peak of the same kind of work as the kernel (the same arithmetic format, the same level of memory), and compares devices only through peaks measured on them.

Volume IV, Chapter 10 showed what the nominal denominator does to the ranking of kernels on one device: it shifts every row by the same factor, so the ranking survives and the distance to the roof is overstated. Across devices the factor is not the same, and the ranking can turn over. Two model devices and five kernels:

<!-- worked: benchmarking.peak_denominators -->
Two model devices, X (the model GPU unit of Volume IV) and Y, none of it a measurement. The nominal arithmetic peak is lanes x 2 FLOPs per fused multiply-add x the boost clock in GHz; the nominal bandwidth is the bus width in bytes x the data rate per pin in Gb/s; the nominal 8-bit integer rate is 4 times the nominal fp32 rate (four 8-bit products per lane and cycle). The measured rates are the probes of Volume IV, Chapter 10: X's are that chapter's calibration, Y's are parameters of this example. X also has a measured 8-bit rate of 128,000 (GOP/s) and a measured cache-resident bandwidth of 2,400 (GB/s); nothing nominal is published for the cache.

| device | lanes | boost clock (GHz) | nominal fp32 (GFLOP/s) | measured fp32 | measured / nominal | bus bits x Gb/s per pin | nominal bandwidth (GB/s) | measured bandwidth | measured / nominal |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| X | 10,240 | 1.953125 | 40,000 | 34,469 | 0.862 | 256 x 25 | 800 | 646 | 0.807 |
| Y | 6,144 | 1.953125 | 24,000 | 22,800 | 0.950 | 384 x 20 | 960 | 590 | 0.615 |

Five kernels, their counted work and model times. The rate is the work over the time, in G per second of the work's own unit (bytes, FLOPs or integer operations). Each fraction divides by the roof of the kernel's own class (main-memory bandwidth, fp32 arithmetic, 8-bit integer arithmetic, cache bandwidth); the last column divides by the roof that a report with only fp32 and main-memory roofs would use.

| kernel | device | counted work | time (microseconds) | rate (G per second) | fraction of nominal peak | fraction of measured roof | against a roof of the wrong class |
|---|---|---:|---:|---:|---:|---:|---|
| decode step, weights streamed | X | 3.000e+09 bytes | 6,000.0 | 500 | 0.625 | 0.774 | - |
| decode step, weights streamed | Y | 3.000e+09 bytes | 5,555.6 | 540 | 0.562 | 0.915 | - |
| fp32 matrix product 4096^3 | X | 1.374e+11 FLOPs | 6,247.2 | 22,000 | 0.550 | 0.638 | - |
| int8 matrix product 4096^3 | X | 1.374e+11 integer operations | 11,453.2 | 12,000 | 0.075 | 0.094 | 0.348 (measured fp32 roof) |
| reduction over a 4 MiB working set, 600 passes | X | 2.517e+09 bytes | 1,677.7 | 1,500 | - | 0.625 | 2.322 (measured main-memory roof) |
<!-- /worked -->

The nominal figures come from arithmetic anyone can check: $10{,}240 \times 2 \times 1.953125 = 40{,}000$ GFLOP per second for X, and a 256-bit bus at 25 (Gb/s) per pin moves $32 \times 25 = 800$ (GB/s). The measured rates are what the probes reached: 0.862 and 0.807 of nominal on X, 0.950 and 0.615 on Y, whose memory falls much further short of its specification than its arithmetic does.

Now the first two kernel rows, the same decode step on the two devices. Against the nominal bandwidth, X's kernel is at 0.625 and Y's at 0.562, and a report would say X's engine is the more efficient. Against each device's measured bandwidth, X's is at 0.774 and Y's at 0.915: Y's engine is within 9 percent of what its device can move and X's leaves more than a fifth of its device unused. The order turned over because the denominators carry each device's own shortfall, 0.807 and 0.615, which have nothing to do with either engine. Neither nominal fraction is an error of arithmetic; both are statements about a specification sheet rather than about the engines.

The last two rows are the lies in the other direction, and they are more dangerous because they flatter. The 8-bit matrix product does integer operations, and the device does 8-bit products four to a lane. Graded against the fp32 roof, the only arithmetic roof a simple report measures, it is at 0.348, a third of the machine and a kernel in reasonable shape; against the roof of its own class it is at 0.094, a tenth, the largest gap in the table. The reduction over a 4 MiB working set reads its data from cache 600 times. Graded against main-memory bandwidth it is at 2.322, above the roof, and looks like a broken measurement; graded against the cache's bandwidth it is at 0.625, which is what it is. The roof of the wrong class makes one kernel look nearly four times better than it is and another look impossible.

Commit b500a8094 (August 2026) is the real case of the fourth row. The profiler of brain had graded its int8 matrix product against the only compute roof it had measured, the fp32 one, and the commit measured a second, with a probe of packed 8-bit dot products built like the fp32 chain:

<!-- perf-number: historical measurement, commit b500a8094 -->
In August 2026, on one older discrete GPU, the commit reports a probed 8-bit dot-product rate 4.15 times the probed fp32 rate, and the int8 product at 8.5 percent of the 8-bit roof where the table had shown 36.1 percent of the fp32 one.

The commit's own summary is that grading 8-bit work against the fp32 roof "turned the single largest gap on the served path into an apparent success". The code it left is in `crates/gpu-core/src/roof.rs`: `Roofs` carries `int8_gops` (absent where the device has no 8-bit dot path) and `cache_gbs`, the bandwidth of a cache-resident working set measured with the same triad shape, whose doc comment names the kernel that once read above the main-memory roof because its data fit in cache; `utilisation_of` takes the floating-point and integer operation counts apart so that the right roof can be chosen, a substituted roof is marked in the profile table, and `real_work_can_never_beat_the_roof` is the test of the invariant. The header of the file states the general reason: every "percent of peak" brain printed had once divided by a datasheet literal copied into each benchmark, and "a wrong roof silently invalidates the method rather than producing an obviously wrong answer". `DeviceCaps` in `crates/backend-api/src/lib.rs` holds the measured rates as `peak_gflops` and `peak_bandwidth_gbs`, documented as "measured, never queried and never derived from a marketing figure", and leaves them `None` until something has measured them: an unknown denominator is no denominator, not a guessed one. Commit 73466fee0, which replaced those literals with the probe and compared the two on one card, is told in Volume IV, Chapter 3.

## Heat, order and a drift that looks like an effect

Condition 4 begins with an uncomfortable fact: the machine changes while it is measured. A device under continuous load heats up, and when it reaches its power or thermal limit it lowers its clock, and the work of the next minute runs slower than the work of the last. A benchmark that measures its cases one after another in a fixed order assigns that drift to whatever it measured later. Here is a ladder of batch sizes whose true gain is small, 4.4 percent from a batch of 1 to a batch of 8, on a model device whose clock settles at 0.72 of its cold value with a time constant of 150 seconds of load:

<!-- worked: benchmarking.order_drift -->
A batch-size ladder on a device that slows as it heats, a model and not a measurement. The steady rate of batch size B relative to B = 1 is 1.000, 1.020, 1.035 and 1.044 for B = 1, 2, 4, 8. Under continuous load the clock factor is m(t) = 0.72 + 0.28 exp(-t / 150), t the model seconds of load since the device was cold, exp the library function; a level is measured for 60 model seconds and reads its steady rate times the mean of m over its interval, 0.72 + 0.28 (150 / 60) (exp(-t0 / 150) - exp(-t1 / 150)), times (1 + 0.005 z) with z = rng.normal(). Generator: the book's SplitMix64, seed 0x442a29ee2989ac97 (`seed_of` of the name `benchmarking.order`, defined on the pseudocode page). Draw order: the columns from left to right, and within a column the levels in the order the column measures them, one normal each. 'Ascending' runs B = 1, 2, 4, 8 back to back from cold; 'descending' runs 8, 4, 2, 1; 'rest' lets the device cool fully before each level; 'pre-heated' runs 900 model seconds of load first and then the ascending ladder. Each column is divided by its own B = 1 reading.

| batch size B | steady rate (truth) | ascending from cold | descending from cold | cooled before each level | pre-heated, ascending |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2 | 1.020 | 0.937 | 1.060 | 1.024 | 1.025 |
| 4 | 1.035 | 0.896 | 1.134 | 1.036 | 1.038 |
| 8 | 1.044 | 0.862 | 1.258 | 1.044 | 1.036 |
<!-- /worked -->

![The rate of each batch size relative to a batch of 1 against the batch size: the truth rises gently to 1.044; the ladder run ascending from cold falls to 0.862, the ladder run descending rises to 1.258, and the ladder that cools before each level lies on the truth.](img/order-drift.svg)

Measured ascending from cold, batching *loses* 13.8 percent at a batch of 8; measured descending, it *gains* 25.8 percent. The truth is a gain of 4.4 percent, and both orders report an effect several times larger than the truth, with opposite signs. The drift of the clock between the first and the last level, from the cold value down toward 0.72, is the effect the ladder measures. Two designs recover the truth: let the device return to the same state before each level (the "cooled" column, 1.044), or bring it to its steady hot state first and measure there (the "pre-heated" column, 1.036, within the noise of a single reading). Interleaving the cases and repeating them in more than one order does the same job when neither is practical, and it has a useful side effect: if the forward and reverse orders disagree, the benchmark has detected its own drift.

This is the story of commit adba36da4 (July 2026). It made a diffusion image model run a batch of requests in one device pass, bit for bit equal to running them one at a time (the test is `crates/flux2/tests/it/batch_parity.rs`, which demands a largest absolute difference of exactly zero), and then measured what batching was worth:

<!-- perf-number: historical measurement, commit adba36da4 -->
In July 2026, on one passively cooled older GPU, a batch of 8 took 0.958 of the time per image of a batch of 1 (the commit's 1.044 times), while a continuous concurrency ladder heated the card from 83 to 90 degrees Celsius and lowered its clock from 1,531 to 923 MHz, an effect the commit puts at 1.32 times.

<!-- perf-number: historical measurement, commit adba36da4 -->
The commit is explicit about both. Batching was not the throughput multiplier that had been hoped for, because the matrix products that were 80 percent of the forward pass were already at their plateau at one sample; it is a capacity feature, one resident set of weights serving several requests. And the ladder "run as one continuous sweep reported its own thermal ramp: reversing it inverts the result", an effect of 1.32 times, "larger than everything the sweep exists to measure". The effect size of the claim, 4.4 percent, had to be read against the effect size of the instrument's drift, 32 percent. A claim smaller than the drift of its own procedure is not a claim until the procedure has been fixed.

The suite now records the state of the machine with every result. `crates/perf/src/devicetel.rs`, added by commit 4dce625d4 (August 2026), samples the GPU's frequency, its idle-state residency and its throttle reasons before and after the measured region, from the sysfs files of the first Intel (i915) graphics device it finds, together with a package temperature read from the processor's `coretemp` sensor in `hwmon`. Every field is optional, and where no Intel graphics device is found (a discrete NVIDIA or AMD card, say) every field is written as `null` rather than as a reassuring zero. The commit's reason is one sentence: "A number without frequency/throttle context is not reproducible on a box that throttles under load." The static frequency ceiling it reads joins the comparison axes of `Env` in `crates/perf/src/env.rs`, since two runs at different ceilings are not comparable, and a reading of throttling that could not be made stays "unknown", never "not throttling".

## When two runs may be compared at all

The last part of condition 4 is the comparison itself. A difference between two runs is evidence only when the runs differ in nothing else and the difference exceeds what the procedure produces when nothing changes. The workload's random draws are part of "nothing else". Take variant B of the engine with 0.3 milliseconds of host work per iteration in place of 0.4, a real improvement, and measure it against A on eight different request streams, each drawn from the same distribution with its own seed:

<!-- worked: benchmarking.stream_seeds -->
Variant A is the engine of `benchmarking.opening_claims`; variant B is the same engine with 0.3 model milliseconds of host work per iteration in place of 0.4, nothing else changed. Both serve 8 in flight, closed loop. Stream k (k = 0 .. 7) is 68 requests drawn as before from its own generator, seeded with `seed_of` of the name `benchmarking.streams.k`, the seed printed in each row. 'Paired' compares B and A on the same stream; 'unpaired' compares B on stream k + 1 (stream 0 after stream 7) with A on stream k. Output rates in tokens per second of the model engine, whose rates the author chose (no number here is a measurement).

| stream k | seed | A: output tokens per second | B: output tokens per second | paired B / A - 1 | unpaired B(k + 1) / A(k) - 1 |
|---:|---|---:|---:|---:|---:|
| 0 | 0x2e1bdecc8d14fbec | 770.5 | 779.8 | +0.0120 | -0.0372 |
| 1 | 0x9524c529a3c10c4b | 733.2 | 741.8 | +0.0118 | +0.0558 |
| 2 | 0xc985c0730c7620b1 | 765.2 | 774.1 | +0.0116 | -0.0219 |
| 3 | 0x2c19330468c6c126 | 739.9 | 748.5 | +0.0117 | -0.0092 |
| 4 | 0x78fd7a0fa84d24fd | 725.0 | 733.1 | +0.0112 | +0.0738 |
| 5 | 0xcb266aaa27f40340 | 769.3 | 778.5 | +0.0118 | -0.0003 |
| 6 | 0xd353df5f0dc15942 | 760.2 | 769.1 | +0.0116 | +0.0066 |
| 7 | 0xc056816d116367e5 | 756.2 | 765.2 | +0.0120 | +0.0312 |

Paired: mean +0.0117, sample standard deviation 0.0003. Unpaired: mean +0.0123, sample standard deviation 0.0384. Spread of A over the 8 streams, (largest - smallest) / largest: 0.0591. (Divisor n - 1 = 7.)
<!-- /worked -->

The model engine has no timing noise at all, and still the output rate of the same engine moves by 5.91 percent of its largest value between streams, because one stream happens to hold longer prompts or shorter answers than another. Against that, B's real gain is 1.17 percent. Measured on the same stream, the gain is +0.0117 with a standard deviation of 0.0003 over the eight streams: the effect is 39 standard deviations of its own measurement and nobody could miss it. Measured with each variant on its own stream, the eight comparisons have a standard deviation of 0.0384 and range from -0.0372 to +0.0738; four of the eight say B is slower. This is the paired design of Volume II, Chapter 8 applied to a workload: the stream is the unit of pairing, and pairing removes almost all of the stream's variation from the difference.

The rule that follows is mechanical, and `crates/perf/src/gate.rs` enforces it with the rest of the conditions as refusals rather than as warnings. Before comparing a candidate run with a committed baseline, `gate` refuses when the scenarios differ, when the units of output differ (tokens per second and frames per second are not one axis), when the workload seeds differ (naming both and the flag to rerun with), when either run failed its output check ("a broken run neither sets nor passes a floor"), when the candidate is a smoke run sized for continuous integration ("not a measurement"), when any axis of the environment fingerprint differs (device set, backend, adapter, build, cores, frequency ceiling: "a baseline from different hardware is not a floor"), and when the pair has no metric in common, because "a gate that checked NOTHING must not read as a pass". A metric missing on one side is skipped and said to be skipped. The comparison report of `crates/perf/src/report.rs` applies the same tests in a softer form: it refuses to rank results in different units and prints a warning line for every other axis that differs, asking for comparisons "deliberately along ONE axis".

Two commits show where those refusals came from. Commit 5dcd85408 (October 2026) made each run draw its workload seed by default and made the gate refuse runs of different request streams, with the gate scripts replaying the seed recorded in their baseline, which is the table above made into a rule: a different stream is a different workload. Commit c77ce9e58 (August 2026) closed a gate that could pass vacuously: a floor fraction of 0 made every throughput bound 0 and every latency bound infinite, "a green PASS that checked nothing, defeatable with one flag", while the message claimed the range was (0, 1]. The test `floor_zero_refuses_instead_of_passing_vacuously` now holds the code to its message.

## A floor that neither flaps nor sleeps

A regression gate turns a comparison into a decision, and the decision has the two errors of any test. It can fail a candidate that did not regress (a false alarm), and it can pass one that did. On a quiet machine the first is rare; on a machine another process shares, each run is also slowed by whatever share of it the other process took, and the false alarms of a tight floor become routine. The table judges a gate that compares the best of 3 runs of the candidate with the best of 3 of the baseline, at three floors, on both kinds of machine:

<!-- worked: benchmarking.gate_floors -->
A throughput gate: the candidate passes when its best of 3 runs is at least the floor times the baseline's best of 3. On the quiet machine a run's throughput is its true rate over the noise factor of Volume IV, Chapter 10, (1 + 0.04 u0) (1 - 0.25 ln(1 - u2) if u1 < 0.2, else 1), which only ever slows a run; on the shared machine the same factor is multiplied by (1 + 0.15 u3), the share of the run another tenant took. The baseline's true rate is 1 and the candidate's is 1 minus the regression; the noise rates are chosen by the author and this is a simulation, not a measurement. Generator: the book's SplitMix64, seed 0xdd03e986bafaec74 (`seed_of` of the name `benchmarking.gate`, defined on the pseudocode page). Draw order: for each regression in the order of the table, 2,000 trials; a trial draws the baseline's 3 runs and then the candidate's 3, four uniforms per run (u0, u1, u2, u3); both machines and every floor are judged on the same draws (the quiet machine ignores u3). Each cell is the share of the 2,000 trials in which the gate failed.

| true regression | quiet: floor 0.95 | quiet: 0.90 | quiet: 0.85 | shared: floor 0.95 | shared: 0.90 | shared: 0.85 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.0050 | 0.0020 | 0.0015 | 0.1460 | 0.0145 | 0.0025 |
| 0.05 | 0.4930 | 0.0035 | 0.0015 | 0.4930 | 0.1250 | 0.0085 |
| 0.10 | 0.9960 | 0.5040 | 0.0020 | 0.8745 | 0.5000 | 0.1145 |
| 0.15 | 0.9965 | 0.9940 | 0.4965 | 0.9890 | 0.8655 | 0.4910 |
| 0.20 | 0.9990 | 0.9965 | 0.9915 | 0.9980 | 0.9920 | 0.8875 |
| 0.30 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9995 |
<!-- /worked -->

The first row is the false alarm. On the shared machine a floor of 0.95 fails 0.1460 of the candidates that did not regress at all, about one run in seven. A gate that fails one run in seven for no reason is a gate that people learn to rerun until it passes, and then to switch off, after which it detects nothing. At 0.85 the false alarms fall to 0.0025. The price is in the columns below: a floor detects a regression with probability about one half when the regression equals one minus the floor (0.4965 at a true 0.15 against the floor 0.85 on the quiet machine), and reliably only beyond it (0.9915 at 0.20). The floor is the smallest effect size the gate can claim to catch. A gate with a floor of 0.85 is a statement that regressions of 15 percent and more will be stopped, and it says nothing about a regression of 5 percent; that one needs the paired, repeated comparison of the previous section, with its effect size and its standard error.

That is the trade brain's gates make on purpose. Commit 030cdfbbb (July 2026), which added the gate, chose "hard floors, generous on purpose, because tight deltas flap on shared boxes and a flapping gate gets deleted", with 0.85 as the documented example; throughput is gated as a floor (candidate at least the baseline times the fraction) and latency as a ceiling (candidate at most the baseline divided by it). `scripts/gates/perf-floor-gate.sh`, the per-backend gate of the hottest serving kernels, follows the same reasoning in its own terms: floors recorded on a rested card of each device class, a default tolerance of 0.25 to absorb the spread of a shared one, and a device class without a floor file reported as ungated "so a different machine is never judged against this one's numbers". The best of several runs is the statistic where repeats are used (`brain perf` takes it with `--best-of N`, whose default is 1, and the per-backend floor test times the minimum over repeated runs), for the reason Volume IV, Chapter 11 gave (noise only slows a run, so the best is the least contaminated), and `BestOf` in `crates/stats/src/latency.rs` records the spread of the runs beside it, $(\max - \min)/\max$, so that a reader knows how noisy the machine was.

The pseudocode of the decision, with the refusals first:

```text
record Run:
    scenario: text
    unit: text
    seed: int
    valid: bool          # its output check passed (or none was run)
    checked: bool        # an output check ran at all
    smoke: bool
    axes: list<text>     # the environment fingerprint, in a fixed order
    rate: real           # output per second, best of its repeats

function perf_gate(cand: Run, base: Run, floor: real) -> text:
    if not (floor > 0.0 and floor <= 1.0):
        return "refused: floor outside (0, 1]"
    if cand.scenario != base.scenario or cand.unit != base.unit:
        return "refused: not the same measurement"
    if cand.seed != base.seed:
        return "refused: different request streams"
    if not cand.valid or not base.valid:
        return "refused: a run failed its output check"
    if cand.smoke:
        return "refused: a smoke run is not a measurement"
    if cand.axes != base.axes:
        return "refused: different hardware or build"
    verdict = "pass"
    if cand.rate < floor * base.rate:
        verdict = "fail"
    if not cand.checked or not base.checked:
        verdict = concat(verdict, " (speed only: a run was never checked for correctness)")
    return verdict
```

## Benchmarks of what a model can learn

Everything so far has been about speed, but the four conditions are not. A benchmark of what a model can do is an instrument with the same failure: its score measures the claim only if the task cannot be solved by something other than the ability it names. The other benchmark crate of brain is built around that question. `crates/bench` asks, in the words of `crates/perf/src/lib.rs`, "can this architecture **learn** task X?", and compares architectures where `perf` compares models, hardware and configurations; Volume II, Chapter 8 described its battery, its per-capability axes and how seed noise made one of its probes informational. Its reference benchmark shows the first condition, a task that cannot be faked, in a few lines.

Multi-query associative recall, from Arora and coauthors (the Zoology study, 2023), writes a few key-value bindings into a sequence and then queries the keys; answering requires finding the earlier occurrence of the queried key in *this* sequence and copying what followed it. `crates/bench/src/mqar.rs` builds it with keys from one half of the content vocabulary and values from the other, distinct keys within a sequence, and values drawn afresh for every binding, and its documentation states the reason: the task "cannot be solved by n-gram statistics". Whether a design has that property can be checked without training anything, by scoring the strongest predictors that do not read the sequence:

<!-- worked: benchmarking.recall_shortcut -->
An associative-recall task in two designs and three predictors that are not trained networks, on synthetic sequences (a model, not a measurement). A sequence binds two distinct keys to values and then queries both keys, k1 v1 k2 v2 SEP q1 a1 q2 a2; keys are ids 0 .. 7 and values ids 8 .. 15. Generator: the book's SplitMix64, seed 0xd54eaa694acd5708 (`seed_of` of the name `benchmarking.recall`, defined on the pseudocode page). Draw order: 6,000 training then 200 held-out sequences of the first design, then the fixed table (one integer(8) per key 0 .. 7), then 6,000 and 200 sequences of the second design. A sequence draws its first key integer(8), its second key from integer(7) skipping the first (an index j at or above the first key becomes j + 1), its two values integer(8) each in the first design (offset by 8; the second design reads the table and draws none), and which key is queried first, integer(2). 'Most common value' always answers the value most frequent at the answer positions of training; 'value seen most often after this key' answers, for a queried key, the value that followed it most often at the answer positions of training (ties to the lowest id); 'look the key up in this sequence' copies the value bound to the key earlier in the same sequence. Accuracy is over the 400 answer positions of the held-out sequences.

| design | answer positions | chance, 1 / 8 | most common value | value seen most often after this key | look the key up in this sequence |
|---|---:|---:|---:|---:|---:|
| values drawn afresh in every sequence | 400 | 0.125 | 0.142 | 0.135 | 1.000 |
| each key always bound to the same value | 400 | 0.125 | 0.552 | 1.000 | 1.000 |
<!-- /worked -->

In the design as built, the predictor that remembers which value usually follows a key scores 0.135 and the one that always answers the most common value 0.142, both within the noise of chance (0.125, with 400 answers a standard error of about 0.017). Only reading the sequence scores 1.000. A model that scores well has learned recall. In the second design, which looks almost the same, each key is bound to the same value in every sequence. The predictor that never reads the sequence now scores 1.000 by memorizing a table of eight entries, and a model that scored 1.000 might have learned recall or might have memorized the table, and the benchmark cannot tell which. The second design is a valid benchmark of memorization and an invalid one of in-context recall. The check measured the shortcut *before* any model was trained, and a benchmark that publishes the score of its strongest shortcut beside its chance level cannot tell this lie by accident.

The comparisons of `crates/bench` meet condition 4 by construction in places. The registry of `crates/bench/src/arch.rs` holds a pair of architectures, `qwen-cmp` and `glm-cmp`, sized so that the dense model and the mixture-of-experts model activate about the same number of parameters per token (about 71,000 and 69,000, its comment says), and capped at the same step budget, `CMP_STEPS`, so that a difference between them is a difference of architecture and not of size or of training. The reduced registry that the integration tests run, `registry_smoke`, is documented as scores that "are *not* meaningful as architecture quality", and the artifact records `smoke` beside the scores and the seed and the commit, so that a leaderboard can tell a measurement from a test of the plumbing. The same two words, *smoke* and *seed*, are refusals in `perf`'s gate. The instrument for capability and the instrument for speed are kept honest by the same four conditions.

## Check your understanding

1. A colleague reports that a change to the attention kernel "makes the server 1.6 times faster" and shows a table of kernel times. Name the two things you ask for before believing the claim, and say what Amdahl's law with the shares of `benchmarking.amdahl_request` predicts for a 1.6 times faster decode at 16 requests in flight if decode is 0.532 of the work.
2. Predict: the 8-bit form of the toy decoder is benchmarked on a bandwidth-bound device. What does a speed table report for it, and what does the output check report? Which of the two numbers decides whether the speed number can be published?
3. Find the error: "Our server sustains 1,139 tokens per second at 20 requests per second, more than the 1,094 of a closed loop at 16, so open-loop load is handled better."
4. Two engines are compared by "fraction of peak bandwidth": 0.625 on one device and 0.562 on another, both against the vendor's numbers. Explain why this does not say which engine is better, and what you would divide by instead.
5. A batch-size ladder is measured once, smallest batch first, on a passively cooled card, and shows that larger batches are slower. Give two cheap ways to tell whether the result is real.
6. Find the error: "The gate at a floor of 0.95 passed, so the change did not regress performance by more than 5 percent."
7. A recall benchmark's best model scores 0.97. What one number, which needs no trained model at all, would you ask for next, and why?
8. A run of the perf suite shows `correctness.passed: null`. Is it a verified result, a failed one or neither, and what may and may not be concluded from it in a comparison?

## Summary

A benchmark is valid when its number measures what the claim says, and that needs four things: the right computation, the claim's workload, a stated and appropriate denominator, and a like-for-like comparison larger than the procedure's own variation. Reproducibility is not among them. The output check comes first: in the toy decoder the two broken forms read fewer bytes and do fewer FLOPs than the reference, a speed table would rank them first, and only a position-by-position comparison over the longer of each pair (59 of 72 and 16 of 72) catches them, while the reordered sums of an honest change pass. A server is measured end to end, as distributions of per-request times under a stated arrival process; the same 68 requests give 1,094.4 output tokens per second saturated with 0.078 of the requests on time, and 767.4 at the knee where the objective held for all of them, and the deliverable of a ladder is that knee and not the peak. Amdahl's law converts a kernel's gain into a server's, and the same halving of weight bytes is worth 1.79 times at one request in flight and 1.32 at sixteen. A fraction of peak names its peak: nominal peaks reverse the ranking of two engines on two devices (0.625 against 0.562 nominal, 0.774 against 0.915 measured), and a roof of the wrong class makes an 8-bit product look at 0.348 instead of 0.094. The machine drifts while it is measured, enough to turn a 4.4 percent gain into a 13.8 percent loss or a 25.8 percent gain by order alone. A comparison pairs what it can (the request stream moved the rate by 5.91 percent, the paired gain of 1.17 percent had a standard deviation of 0.0003) and refuses what it cannot, and a regression gate's floor is the smallest effect it can claim to catch. The same conditions make a capability benchmark: a task is a test of recall only if no predictor that ignores the sequence can solve it.

A trustworthy number is the end of a result and not the whole of it. The next chapter, [From Hypothesis to Paper](04-research.md), takes a result from a hypothesis that could fail through to a report that others can check, reproduce and build on.

## References

- Gene M. Amdahl, "Validity of the single processor approach to achieving large scale computing capabilities", 1967.
- Simran Arora, Sabri Eyuboglu, Aman Timalsina, Isys Johnson, Michael Poli, James Zou, Atri Rudra and Christopher Ré, "Zoology: Measuring and Improving Recall in Efficient Language Models", 2023.
- Bianca Schroeder, Adam Wierman and Mor Harchol-Balter, "Open Versus Closed: A Cautionary Tale", 2006.
- Samuel Williams, Andrew Waterman and David Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures", 2009.
- Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin and Hao Zhang, "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving", OSDI 2024.

## Exercises

1. **[Calculate]** From the engine of `benchmarking.request_timeline`, compute the cost of a decode step with one running sequence that holds 300 tokens of context, and with 16 sequences of 300 tokens each. Which term of the maximum wins in each case, and what output rate per second does each step give?
2. **[Calculate]** Compute the nominal fp32 peak and nominal bandwidth of a device with 4,096 lanes at a boost clock of 2.4 GHz and a 192-bit bus at 18 (Gb/s) per pin. A kernel moves $2 \times 10^9$ bytes in 9.0 milliseconds; the probe measured 300 (GB/s) on this device. Give both fractions.
3. **[Calculate]** With the shares of `benchmarking.amdahl_request` at 16 requests in flight, what speedup of the server does halving the prefill work give, and what is the ceiling for any prefill improvement? Compare with the decode change of the table and say which you would work on first.
4. **[Calculate]** A gate's floor is 0.90. Using the quiet columns of `benchmarking.gate_floors`, estimate the probability that a true regression of 12 percent is stopped, and explain why the answer lies between two rows.
5. **[Implement]** Reproduce `benchmarking.fidelity_gate`: the draws of the weights and prompts, the toy decoder in its five forms, `agreement` and the common-prefix score. Then add a sixth form that runs the three probes in the order 2, 1, 0 and returns the streams in that order, and say what `agreement` reports and why.
6. **[Implement]** Reproduce `benchmarking.request_timeline`, `benchmarking.arrival_processes` and `benchmarking.concurrency_sweep` with `serve_run` and `summarize` in your own language, from the same seeds and draw order. Then change the scheduler so that it may run any number of prefills in a row while a place is free, and report how the 99th percentile of the gap and the share meeting the objective change at 16 in flight.
7. **[Implement]** Reproduce `benchmarking.order_drift`, including the closed form of the mean clock factor over a level. Add a column, drawn after the existing four, for the order 1, 8, 2, 4, 4, 2, 8, 1 (each level measured twice, back to back from cold, one normal for each measurement, the two readings averaged) and say how close it comes to the truth.
8. **[Implement]** Reproduce `benchmarking.stream_seeds` and `benchmarking.gate_floors`. Then compute, for the paired design, how small a gain of host work could be detected at two standard errors with the eight streams.
9. **[Implement]** Reproduce `benchmarking.recall_shortcut`. Then design a third variant in which values are drawn afresh but keys and values share one vocabulary of 16 ids, and score the three predictors on it.
10. **[Investigate]** Read `gate` in `crates/perf/src/gate.rs` and the tests below it. For each refusal, write one sentence on the lie it prevents, and note which of the refusals the tests below it exercise.
11. **[Investigate]** Read `Summary::build` and `ReqRecord::meets` in `crates/perf/src/metrics.rs`. Construct by hand a set of four request records for which the mean gap is under the gap budget and the share meeting the objective is 0.5, and check your answer against the definitions.
12. **[Investigate]** Read the header of `crates/gpu-core/src/roof.rs` and the fields of `Roofs`. List each roof it measures, the kind of kernel that must be graded against it, and what the profile table shows when a kernel's own roof was not measured.

## Key terms

- **benchmark validity**: the property that a benchmark's number measures the quantity the claim built on it is about: the computation is the right one (its outputs are checked), the workload is the claim's, the denominator is stated and of the right kind, and a comparison differs only in the thing claimed, by more than the procedure's own variation. <!-- concept: benchmark-validity -->
- **measured versus nominal peak**: the distinction between the rate a device's specification implies (lanes times 2 times the boost clock, bus bytes times the pin rate) and the rate a probe reaches on the device that ran the work, with the rule that a fraction of peak names its peak, divides by a peak of the kernel's own kind of arithmetic or memory, and compares devices only through measured peaks. <!-- concept: measured-versus-nominal-peak -->
- **end-to-end benchmark**: a benchmark that drives a system through its users' entry point with a stated stream of requests and arrival process, records each request's submission, admission, outputs and completion, and reports the distributions of time to first output, gaps and end-to-end time with throughput, goodput under a stated service-level objective, and an output check. <!-- concept: end-to-end-benchmark -->

## Answers

**Recall questions.**

1. The host's cost of handing the work over, since the call returns before the device has run it. Device-synchronized timing stops the clock only after the device has finished, by waiting for it or by reading timestamps the device wrote itself.
2. The whole-pass measurement: the time of the whole unit of work a user waits for, with every dispatch, wait, copy and host step inside the bracket. A kernel can get faster by its own stamps while the change adds dispatches or synchronizations whose cost appears only in the whole pass (Volume IV's variant added two waits and three dispatches and made the pass 2.44 percent slower).
3. The speedup is $A(p, g) = 1/((1 - p) + p/g)$, and however large $g$ becomes it is below $1/(1 - p)$, because the rest of the time is untouched.
4. The effect size answers how large the difference is, in the metric's units or standardized; the test answers whether chance could explain it. With enough examples the standard error shrinks as $1/\sqrt{n}$, so a tiny but real difference reaches any number of standard errors: Volume II's table had a gain of 0.42 points at $z = 5.20$ with 100,000 questions.

**Check your understanding.**

1. Ask for the whole-pass (here end-to-end) number at a stated load, not kernel times, and for the output check. For decode only: $A(0.532, 1.6) = 1/(0.468 + 0.3325) = 1.249$, so even a decode 1.6 times as fast makes the server about 1.25 times as fast at 16 in flight, and an attention change that is only part of decode gives less.
2. Half the weight bytes per token, so on a bandwidth-bound device a speed table reports it nearly twice as fast; the output check reports 59 of 72 positions in agreement and fails. The check decides: a speed number is publishable only for a computation that passed it.
3. The higher rate belongs to an overloaded server: at 20 per second only 0.672 of the requests met the objective and the median first token waited 467.5 milliseconds, against goodput of 982.6 and 0.891 on time at 16 in flight. The rate is also measured over a differently shaped window. Throughput without goodput and attainment does not say which load is handled better.
4. Each fraction is divided by its device's specification, and the two devices fall short of their specifications by different factors (0.807 and 0.615 in the model), which have nothing to do with the engines. Divide each by the bandwidth measured by a probe on the device that ran it; in the model the order then reverses, 0.774 against 0.915.
5. Run the ladder in the opposite order (if the result reverses, it is drift), or let the card cool to the same state before each level, or pre-heat it to its steady state and measure there; record the clock and throttle state beside each level.
6. A floor of 0.95 passes a true regression of 5 percent about half the time (0.4930 in the table) and even somewhat larger ones some of the time, so passing says only that a regression well beyond 5 percent is unlikely. Its own false-alarm rate on a shared machine (0.1460) also means a pass and a fail are both weak evidence there.
7. The score of the strongest predictor that does not read the sequence (and chance). If that shortcut also scores near 0.97, the benchmark cannot distinguish recall from memorization, as the second design of `benchmarking.recall_shortcut` showed.
8. Neither: no output check ran. It may be compared for speed, and the comparison must say that its correctness was never verified; it may not be read as a verified result, and brain's report and gate both say so in words.
