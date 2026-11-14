# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Speculative Decoding".

Plain Python 3 on floats (IEEE double precision), standard library only, written from the chapter's own description and
independent of any engine code. Each public function without a leading underscore that takes no arguments returns a
markdown string that the chapter embeds between `<!-- worked: speculative_decoding.<name> -->` markers; the helpers with
arguments are shared with the chapter's figures, so a picture and the table beside it cannot disagree.

Nothing here is trained or sampled. The language models are the counting models of Volume V, Chapter 4 (the order-1 to
order-4 interpolated models over its 40 printed training sentences), imported through `decoding`; the order-4 model is
the target and the smaller orders are drafts. The work counts reuse the 7.5-billion-parameter decoder and the
illustrative GPU of `kv_cache` and `roofline`. Every distribution that a sampled verification produces is computed by
exact enumeration, so no generator is needed:

* what scoring several rows in one pass costs, in FLOPs and bytes and in units of one plain decode step;
* greedy verification of a draft's proposals, round by round, and its summary over 21 prompts for three drafts;
* the expected tokens of a round under a per-position acceptance, and the tokens per unit of work it buys;
* the marginal window rule and the decayed acceptance estimate a scheduler keeps per sequence;
* sampled verification: the general accept-or-residual rule and the point-mass case, with and without a nucleus;
* prompt lookup, and the parameter inventory of a published block-diffusion draft.

Swedish Embedded AB implements solutions for generating text from large models at a fraction of the work per token,
without changing what the model says, for its clients. If your team needs expertise in speculative decoding and its
verification, you can procure our services by sending an email to info@swedishembedded.com.
"""
import math

import decoding as dec
import kv_cache as kc
import language_modeling as lm
import roofline

# --------------------------------------------------------------------------------------------- formatting


def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _f(x, nd=4):
    s = f"{x:.{nd}f}"
    return f"{0.0:.{nd}f}" if float(s) == 0 else s


def _int(n):
    return f"{n:,}"


def _noise(x):
    """A difference that is exact arithmetic zero prints as a bound: its digits are rounding noise that depends on summation order."""
    return "below 1e-15" if x < 1e-15 else f"{x:.1e}"


def _w(ids):
    return " ".join(lm.TOKENS[i] for i in ids) if ids else "(none)"


# --------------------------------------------------------------------------------------------- the work of one pass

SHAPE = kc.DECODER_8B          # the 7.5-billion-parameter decoder of Volume V, Chapter 12
CONTEXT = 1024                 # tokens already in each sequence's cache
PEAK, BW = roofline.MACHINES["GPU"]
RIDGE = PEAK / BW              # FLOP per byte at which the illustrative GPU stops being bandwidth-bound


def pass_work(batch, rows, ctx=CONTEXT):
    """(FLOPs, bytes) of one pass that scores `rows` new rows for each of `batch` sequences at context `ctx`: every weight
    multiplies every row (the head included: every row needs its logits), each row attends to the ctx cached keys and to
    the new rows up to its own; the weights are read once for the pass and every sequence reads its own cache once. The
    new rows' keys and values are computed in the pass and their writes are left out of the bytes."""
    s = SHAPE
    flops = batch * (kc.flops_new(s, ctx, rows) + 2 * s["D"] * s["V"] * (rows - 1))
    nbytes = kc.weight_params(s) * kc.BYTES + batch * kc.cache_bytes_per_token(s) * ctx
    return flops, nbytes


def pass_units(batch, rows, ctx=CONTEXT):
    """The roofline cost of the pass in units of one plain decode step of the same batch: the larger of FLOPs / ridge and
    bytes, divided by the same quantity for one row per sequence."""
    f, b = pass_work(batch, rows, ctx)
    f1, b1 = pass_work(batch, 1, ctx)
    return max(f / RIDGE, b) / max(f1 / RIDGE, b1)


def verify_cost():
    rows = []
    for batch in (1, 8, 32, 64):
        for k in (0, 1, 3, 7):
            f, b = pass_work(batch, k + 1)
            rows.append((batch, k, k + 1, _int(f), _int(b), _f(f / b, 2), _f(pass_units(batch, k + 1), 3)))
    return (f"The decoder of Volume V, Chapter 12: D = {SHAPE['D']}, L = {SHAPE['L']}, {SHAPE['H']} query heads and {SHAPE['kvh']} key-value "
            f"heads of width {SHAPE['dh']}, F = {_int(SHAPE['F'])} (SwiGLU), V = {_int(SHAPE['V'])}, {_int(kc.weight_params(SHAPE))} parameters "
            f"in matrix products, every value in bf16 ({kc.BYTES} bytes). Every sequence holds {_int(CONTEXT)} tokens in its cache. A pass scores "
            "k + 1 rows per sequence (the token the step consumes and k proposals). FLOPs: 2 per parameter per row, the head included for every "
            "row, plus 4 L H d_h per key a row attends to (the cached keys and the new rows up to its own). Bytes: the weights once per pass plus "
            "each sequence's cached keys and values once; activations and the writes of the new rows are not counted. The unit of the last "
            f"column is one plain decode step of the same batch: the larger of FLOPs / {RIDGE:g} (the ridge of the illustrative GPU of Volume IV, "
            "in FLOP per byte) and bytes, divided by that of one row per sequence.\n\n"
            + _table(["sequences", "proposals k", "rows each", "FLOPs", "bytes", "FLOP per byte", "cost in plain steps"], rows))


# --------------------------------------------------------------------------------------------- the toy target and drafts

TARGET = 4                      # the order of the target model
MAX_NEW = 12                    # a generation stops at the full stop or after 12 tokens
DOT = dec.DOT


def best(order, ctx):
    """The greedy token of the order-`order` model after `ctx`: the largest probability, the lowest id on a tie."""
    return dec.argmax(dec.probs(order, ctx))


def plain_greedy(prompt, order=TARGET, max_new=MAX_NEW):
    out = []
    while len(out) < max_new and (not out or out[-1] != DOT):
        out.append(best(order, list(prompt) + out))
    return out


def draft_greedy(order, ctx, k):
    """Up to k greedy proposals of the draft after ctx, ending after a full stop."""
    props, c = [], list(ctx)
    for _ in range(k):
        t = best(order, c)
        props.append(t)
        c.append(t)
        if t == DOT:
            break
    return props


def verify_greedy(props, best_rows):
    """(accepted, next): the longest prefix of props that equals the target's best tokens, and the target's token after it."""
    assert len(best_rows) == len(props) + 1
    a = 0
    while a < len(props) and props[a] == best_rows[a]:
        a += 1
    return a, best_rows[a]


def spec_greedy(prompt, draft_order, k, max_new=MAX_NEW, window=None):
    """Greedy speculative generation. Returns (tokens, rounds), each round a dict. `window(state)` may choose k per round."""
    out, rounds = [], []
    while len(out) < max_new and (not out or out[-1] != DOT):
        ctx = list(prompt) + out
        want = k if window is None else window(rounds)
        kk = min(want, max_new - len(out) - 1)
        props = draft_greedy(draft_order, ctx, kk)
        rows = [best(TARGET, ctx + props[:j]) for j in range(len(props) + 1)]
        a, nxt = verify_greedy(props, rows)
        emit = props[:a] + [nxt]
        if DOT in emit:
            emit = emit[:emit.index(DOT) + 1]
        emit = emit[:max_new - len(out)]
        rounds.append({"ctx": ctx, "asked": want, "props": props, "rows": rows, "accepted": a, "emit": emit})
        out += emit
    return out, rounds


def prompts():
    """The distinct prefixes of 1, 2 and 3 tokens of the 30 test sentences of Volume V, Chapter 4, in order of first appearance."""
    seen = []
    for s in lm.test_set():
        for n in (1, 2, 3):
            p = tuple(s[:n])
            if p not in seen:
                seen.append(p)
    return seen


TRACE_PROMPT = ("the", "dog")


def trace_rounds():
    """The rounds of the greedy trace: prompt `the dog`, order-2 draft, k = 3."""
    return spec_greedy([dec.ID[w] for w in TRACE_PROMPT], 2, 3)


def greedy_trace():
    prompt = [dec.ID[w] for w in TRACE_PROMPT]
    out, rounds = trace_rounds()
    rows = []
    for i, r in enumerate(rounds):
        rows.append((i + 1, _w(r["ctx"]), _w(r["props"]), _w(r["rows"]), r["accepted"], _w(r["emit"])))
    same = out == plain_greedy(prompt)
    return ("Target: the order-4 interpolated model of Volume V, Chapter 4; draft: the order-2 model of the same chapter, both over the "
            "same 18 tokens and with that chapter's smoothing ratios. Greedy means the largest probability, the lowest id on a tie. A round "
            "asks the draft for up to k = 3 greedy proposals (fewer when only 12 - generated - 1 tokens remain, and none after a full stop), "
            "scores the context and each proposal with the target in one pass (row j is the target's best token after the context and the "
            "first j proposals), accepts the longest prefix that equals the target's best tokens and emits it followed by the target's own "
            "token at the first disagreement, or after the last proposal; a full stop ends the generation and anything emitted after it is "
            f"dropped. The prompt is `{_w(prompt)}`.\n\n"
            + _table(["round", "context", "proposals", "target's best at each row", "accepted", "emitted"], rows, "rlllrl")
            + f"\n\nThe speculative text is `{_w(out)}` in {len(rounds)} target passes; plain greedy decoding of the target gives "
            f"`{_w(plain_greedy(prompt))}` in {len(plain_greedy(prompt))} passes. The two are {'identical' if same else 'DIFFERENT'}.")


def run_summary(draft_order, k):
    toks = passes = proposed = accepted = rejected_rounds = 0
    identical = True
    for p in prompts():
        out, rounds = spec_greedy(list(p), draft_order, k)
        identical &= out == plain_greedy(list(p))
        toks += len(out)
        passes += len(rounds)
        proposed += sum(len(r["props"]) for r in rounds)
        accepted += sum(r["accepted"] for r in rounds)
        rejected_rounds += sum(1 for r in rounds if r["accepted"] < len(r["props"]))
    return {"tokens": toks, "passes": passes, "proposed": proposed, "accepted": accepted, "rejections": rejected_rounds,
            "identical": identical}


def draft_orders():
    ps = prompts()
    rows = []
    for order in (1, 2, 3):
        for k in (1, 3, 5):
            s = run_summary(order, k)
            frac = s["accepted"] / s["proposed"]
            alpha = s["accepted"] / (s["accepted"] + s["rejections"])
            rows.append((f"order {order}", k, s["passes"], s["tokens"], s["proposed"], s["accepted"], s["rejections"], _f(frac, 3),
                         _f(alpha, 3), _f(s["tokens"] / s["passes"], 3), "yes" if s["identical"] else "no"))
    plain_tokens = sum(len(plain_greedy(list(p))) for p in ps)
    return (f"The {len(ps)} prompts are the distinct prefixes of 1, 2 and 3 tokens of the 30 test sentences of Volume V, Chapter 4, in order "
            "of first appearance: " + "; ".join(f"`{_w(p)}`" for p in ps) + f". Plain greedy decoding of the target generates {plain_tokens} "
            f"tokens for them in {plain_tokens} passes. The models are those of the previous example, with the interpolation ratios of Volume V, "
            "Chapter 4 for the orders 1, 2, 3 and 4 of " + ", ".join(f"{dec._ratio(o):g}" for o in (1, 2, 3, 4))
            + f" and its uniform floor of {lm.EPS_UNIFORM}. Each generation follows the rules of the previous example with the draft order and "
            "the k of the row. 'rejections' counts the rounds in which some proposal was not accepted; 'accepted / proposed' is the share of "
            "proposals accepted; the per-position estimate is accepted / (accepted + rejections); 'tokens per pass' is tokens / passes.\n\n"
            + _table(["draft", "k", "target passes", "tokens", "proposed", "accepted", "rejections", "accepted / proposed",
                      "per-position estimate", "tokens per pass", "same text as plain greedy"], rows, "lrrrrrrrrrl"))


# --------------------------------------------------------------------------------------------- what a round is worth

def expected_tokens(alpha, k):
    """Tokens a round of k proposals emits when each proposal is accepted with probability alpha given that the ones before
    it were: 1 + alpha + ... + alpha^k (the target's own token is always emitted)."""
    total, reach = 0.0, 1.0
    for _ in range(k + 1):
        total += reach
        reach *= alpha
    return total


ALPHAS = (0.5, 0.7, 0.9)


def expected_table():
    ks = (1, 2, 3, 4, 5, 7)
    rows = [(_f(a, 1),) + tuple(_f(expected_tokens(a, k), 3) for k in ks) + (_f(1 / (1 - a), 3),) for a in ALPHAS]
    return ("Expected tokens of one round, 1 + alpha + alpha^2 + ... + alpha^k = (1 - alpha^(k+1)) / (1 - alpha), for a per-position "
            "acceptance alpha that is the same at every depth; the last column is the limit 1 / (1 - alpha) of an unbounded window.\n\n"
            + _table(["alpha"] + [f"k = {k}" for k in ks] + ["k unbounded"], rows))


R_DRAFT = 0.1                   # bytes of the draft model's weights over the target's (and the same ratio for its FLOPs)
BLOCK_ROWS = 16                 # rows of a block drafter's one pass: the context's last token and 15 masks


def draft_units(kind, batch, k):
    """The work of drafting k proposals for each of `batch` sequences, in plain target steps of the same batch. The draft
    is the target scaled by R_DRAFT in FLOPs and in bytes, so a pass of it costs R_DRAFT times the same pass of the target."""
    if kind == "free" or k == 0:
        return 0.0
    if kind == "small decoder":
        return k * R_DRAFT * pass_units(batch, 1)
    if kind == "block drafter":
        return R_DRAFT * pass_units(batch, BLOCK_ROWS)
    raise ValueError(kind)


def tokens_per_unit(kind, batch, alpha, k):
    return expected_tokens(alpha, k) / (pass_units(batch, k + 1) + draft_units(kind, batch, k))


KINDS = ("free", "small decoder", "block drafter")
KS = (1, 2, 3, 5, 7)


def economics():
    rows = []
    for batch in (1, 32):
        for kind in KINDS:
            for a in ALPHAS:
                vals = [tokens_per_unit(kind, batch, a, k) for k in range(1, BLOCK_ROWS)]
                kbest = max(range(len(vals)), key=lambda i: (vals[i], -i)) + 1
                rows.append((batch, kind, _f(a, 1)) + tuple(_f(tokens_per_unit(kind, batch, a, k), 3) for k in KS)
                            + (kbest, _f(vals[kbest - 1], 3)))
    return ("Tokens per sequence per unit of work, the unit being one plain decode step of the same batch (which emits 1 token per "
            "sequence), for the decoder and context of the first table. A round of k proposals costs one target pass of k + 1 rows per "
            "sequence (the last column of the first table) plus the drafting, and emits the expected tokens of the previous table. "
            f"Drafts: 'free' costs nothing (prompt lookup); 'small decoder' is the target scaled down to {R_DRAFT} of its FLOPs and bytes, "
            "run once per proposal (k passes of one row); 'block drafter' is the same scaled model run once over a block of "
            f"{BLOCK_ROWS} rows whatever k is (at most {BLOCK_ROWS - 1} proposals). The best k is searched over 1 to "
            f"{BLOCK_ROWS - 1} for every drafter, the smaller k on a tie. Above 1 the round is worth more than plain decoding.\n\n"
            + _table(["sequences", "drafter", "alpha"] + [f"k = {k}" for k in KS] + ["best k", "at best k"], rows, "llrrrrrrrr"))


# --------------------------------------------------------------------------------------------- the window a scheduler asks for

MAX_WINDOW = 7


def marginal_window(alpha, cost, cap=MAX_WINDOW):
    """The largest k <= cap such that every proposal j = 1 .. k is expected to earn at least its cost: alpha^j >= cost, with
    the cost of a proposal (one more verify row plus one more draft step) in units of the work of one plain token."""
    k, reach = 0, 1.0
    while k < cap:
        reach *= alpha
        if reach < cost:
            break
        k += 1
    return k


def window_rule():
    costs = (0.02, 0.1, 0.3)
    alphas = (0.3, 0.5, 0.7, 0.8, 0.9)
    rows = []
    for a in alphas:
        best_k = max(range(1, 8), key=lambda k: (tokens_per_unit("small decoder", 1, a, k), -k))
        rows.append((_f(a, 1),) + tuple(marginal_window(a, c) for c in costs) + (best_k,))
    return ("The window of the marginal rule: the largest k (at most 7) for which alpha^j >= c for every j = 1 .. k, where c is the "
            "work of one more proposal (a verify row and a draft step) over the work of one plain token. The last column is the k in 1 to 7 that "
            f"maximises the tokens per unit of the previous table for the small decoder at one sequence (the smaller k on a tie), where one more proposal costs "
            f"c = {R_DRAFT} (a verify row is free there and a draft step costs {R_DRAFT}).\n\n"
            + _table(["alpha"] + [f"c = {c}" for c in costs] + ["best k for tokens per unit, c = 0.1"], rows))


DECAY = 0.75                    # weight kept of the past at each round
PRIOR = 1.0                     # the estimate starts as one accepted proposal and one rejection
PROBE_EVERY = 8                 # plain steps between probes of a switched-off drafter


def estimator_trace():
    """The adaptive loop: each round asks for the marginal window at the current estimate, the target verifies, and the
    estimate is updated from what was accepted."""
    state = {"acc": PRIOR, "rej": PRIOR}
    cost = R_DRAFT
    rows = []

    def window(_rounds):
        return marginal_window(state["acc"] / (state["acc"] + state["rej"]), cost)

    out, ctxs = [], [dec.THE]
    max_new = 40
    rounds = 0
    while len(out) < max_new:
        alpha_before = state["acc"] / (state["acc"] + state["rej"])
        k = window(None)
        kk = min(k, max_new - len(out) - 1)
        ctx = ctxs + out
        # the draft may propose past a full stop here: the generation runs 40 tokens without stopping
        props = []
        c = list(ctx)
        for _ in range(kk):
            t = best(2, c)
            props.append(t)
            c.append(t)
        brs = [best(TARGET, ctx + props[:j]) for j in range(len(props) + 1)]
        a, nxt = verify_greedy(props, brs)
        emit = (props[:a] + [nxt])[:max_new - len(out)]
        if props:
            state["acc"] = DECAY * state["acc"] + a
            state["rej"] = DECAY * state["rej"] + (1.0 if a < len(props) else 0.0)
        rounds += 1
        rows.append((rounds, _f(alpha_before, 3), k, len(props), a, _w(emit), _f(state["acc"], 4), _f(state["rej"], 4)))
        out += emit
    plain = []
    while len(plain) < max_new:
        plain.append(best(TARGET, ctxs + plain))
    return ("One sequence generated greedily for 40 tokens after `the`, without stopping at the full stop, by the order-4 target with "
            f"the order-2 draft. Before each round the estimate is a = A / (A + R), with A and R starting at {PRIOR}; the round asks for "
            f"the marginal window at a with c = {cost} (the previous table's rule, at most 7; the 'window' column is that rule's value), and "
            "proposes the smaller of the window and 40 - generated - 1 tokens (the 'proposed' column, which is why the last round has window 7 "
            "and 3 proposals). After a round "
            f"with proposals, A becomes {DECAY} A + (accepted) and R becomes {DECAY} R + 1 if some proposal was rejected, else {DECAY} R; "
            "a round with no proposals changes neither. The draft proposes greedily and the target verifies as before.\n\n"
            + _table(["round", "estimate before", "window", "proposed", "accepted", "emitted", "A after", "R after"], rows, "rrrrrlrr")
            + f"\n\n{rounds} target passes produce the 40 tokens; the text is {'identical to' if out == plain else 'DIFFERENT from'} plain "
            "greedy decoding of the target, which takes 40 passes.")


# --------------------------------------------------------------------------------------------- sampled verification

def residual(p, q):
    """max(0, p - q) renormalised: the distribution a rejected proposal is replaced from."""
    r = [max(0.0, a - b) for a, b in zip(p, q)]
    s = sum(r)
    return [x / s for x in r] if s > 0 else [0.0] * len(p)


def emitted_first(p, q):
    """The distribution of the first token a sampled round emits: the proposal d ~ q is kept with probability
    min(1, p(d) / q(d)); otherwise a token is drawn from the residual."""
    keep = [min(qa, pa) for pa, qa in zip(p, q)]
    rej = 1.0 - sum(keep)
    r = residual(p, q)
    return [kp + rej * ra for kp, ra in zip(keep, r)], sum(keep)


P_LOGITS = (1.2, 0.7, 0.1, -0.4)


def four_tokens():
    p = dec.softmax(list(P_LOGITS))
    q = [0.25] * 4
    out, acc = emitted_first(p, q)
    r = residual(p, q)
    rows1 = [(f"t{i}", _f(P_LOGITS[i], 1), _f(p[i]), _f(q[i]), _f(min(1.0, p[i] / q[i])), _f(min(p[i], q[i])), _f(r[i]), _f(out[i]))
             for i in range(4)]
    d = 1
    pm = [1.0 if i == d else 0.0 for i in range(4)]
    out2, acc2 = emitted_first(p, pm)
    r2 = residual(p, pm)
    rows2 = [(f"t{i}", _f(p[i]), _f(pm[i], 0), _f(r2[i]), _f(out2[i])) for i in range(4)]
    return ("Four tokens t0 to t3 with logits 1.2, 0.7, 0.1 and -0.4; the target's distribution p is their softmax (temperature 1, "
            "nothing cut). First a draft that samples its proposal from the uniform q; then a draft that always proposes t1 (a point mass). "
            "A proposal d is kept with probability min(1, p(d) / q(d)); a rejected one is replaced by a draw from the residual, "
            "max(0, p - q) renormalised. 'emitted' is the exact probability that the round's first token is that token, "
            "q(t) min(1, p(t) / q(t)) + (1 - sum of min(p, q)) residual(t).\n\n"
            + _table(["token", "logit", "p", "q", "kept if proposed", "min(p, q)", "residual", "emitted"], rows1)
            + f"\n\nProbability that the proposal is kept: {_f(acc)} (the sum of min(p, q)). Largest difference between 'emitted' and p: "
            f"{_noise(max(abs(a - b) for a, b in zip(out, p)))}.\n\n"
            + _table(["token", "p", "q (always t1)", "residual", "emitted"], rows2)
            + f"\n\nProbability that t1 is kept: {_f(acc2)}, which is p(t1). The residual is p with t1 removed and renormalised. Largest "
            f"difference between 'emitted' and p: {_noise(max(abs(a - b) for a, b in zip(out2, p)))}.")


def _nucleus(p, top_p):
    if top_p >= 1.0:
        return list(p)
    q = dec.cut(p, top_p=top_p)
    return [q.get(i, 0.0) for i in range(len(p))]


CONTEXTS = (("the",), ("the", "cat"), ("the", "dogs", "chase"), ("the", "fox", "near"))


def sampled_contexts():
    rows = []
    for words in CONTEXTS:
        ctx = [dec.ID[w] for w in words]
        for top_p in (1.0, 0.9):
            p = _nucleus(dec.probs(TARGET, ctx), top_p)
            q = _nucleus(dec.probs(2, ctx), top_p)
            out, acc = emitted_first(p, q)
            d = dec.argmax(q)
            pm = [1.0 if i == d else 0.0 for i in range(len(p))]
            out2, acc2 = emitted_first(p, pm)
            tv = 0.5 * sum(abs(a - b) for a, b in zip(p, q))
            kept_p = sum(1 for x in p if x > 0)
            rows.append((_w(ctx), "none" if top_p >= 1 else _f(top_p, 1), kept_p, _f(acc, 4), _f(1 - tv, 4), _w([d]), _f(acc2, 4),
                         _noise(max(max(abs(a - b) for a, b in zip(out, p)), max(abs(a - b) for a, b in zip(out2, p))))))
    return ("Target p: the order-4 model's distribution after the context at temperature 1; draft q: the order-2 model's. With a "
            "top-p of 0.9 both are cut to their own nucleus (the shortest prefix in order of decreasing probability, ties to the lower id, "
            "whose cumulative probability reaches 0.9) and renormalised, as Volume V, Chapter 11 defines it; 'tokens in p' counts the "
            "tokens the target can emit. 'kept, sampled draft' is the sum of min(p, q), the probability that a proposal drawn from q is "
            "kept, beside 1 - TV(p, q) (TV is half the sum of |p - q|); 'kept, greedy draft' is p(d) for the draft's greedy token d. The "
            "last column is the largest difference between the exact distribution of the emitted token and p, over both drafts; it is zero "
            "in exact arithmetic and printed as the bound below which the floating-point sums leave it.\n\n"
            + _table(["context", "top-p", "tokens in p", "kept, sampled draft", "1 - TV(p, q)", "greedy draft proposes", "kept, greedy draft",
                      "largest error"], rows, "llrrrlrr"))


def two_token_joint():
    """Exact joint distribution of the first two tokens emitted after `the` by rounds of k = 2 sampled proposals."""
    ctx = [dec.THE]
    V = dec.V
    p1, q1 = dec.probs(TARGET, ctx), dec.probs(2, ctx)
    acc1 = [min(a, b) for a, b in zip(p1, q1)]
    rej1 = 1.0 - sum(acc1)
    r1 = residual(p1, q1)
    joint = [[0.0] * V for _ in range(V)]
    target = [[0.0] * V for _ in range(V)]
    draft = [[0.0] * V for _ in range(V)]
    both_kept = 0.0
    for t1 in range(V):
        p2 = dec.probs(TARGET, ctx + [t1])
        q2 = dec.probs(2, ctx + [t1])
        for t2 in range(V):
            target[t1][t2] = p1[t1] * p2[t2]
            draft[t1][t2] = q1[t1] * q2[t2]
        # rejected at depth 1: t1 from the residual, the round ends; the next round's first token follows p2 exactly
        for t2 in range(V):
            joint[t1][t2] += rej1 * r1[t1] * p2[t2]
        # kept at depth 1 (probability acc1[t1]): the second proposal is tested against p2
        acc2 = [min(a, b) for a, b in zip(p2, q2)]
        rej2 = 1.0 - sum(acc2)
        r2 = residual(p2, q2)
        for t2 in range(V):
            joint[t1][t2] += acc1[t1] * (acc2[t2] + rej2 * r2[t2])
        both_kept += acc1[t1] * sum(acc2)
    err = max(abs(joint[a][b] - target[a][b]) for a in range(V) for b in range(V))
    tv_draft = 0.5 * sum(abs(draft[a][b] - target[a][b]) for a in range(V) for b in range(V))
    tv_spec = 0.5 * sum(abs(joint[a][b] - target[a][b]) for a in range(V) for b in range(V))
    return ("Rounds of k = 2 sampled proposals after `the`: the draft draws d1 from its order-2 distribution and d2 from its distribution "
            "after d1; the target keeps d1 with probability min(1, p(d1) / q(d1)) and, if it did, d2 with min(1, p(d2 given d1) / q(d2 given d1)); a "
            "rejection is replaced from the residual and ends the round, whose next round starts after the emitted token. The joint "
            f"distribution of the first two emitted tokens over the 18 x 18 pairs is computed exactly. The two differences that are zero in exact arithmetic are printed as the bound below which the floating-point sums "
            "leave them.\n\n"
            + _table(["quantity", "value"], [
                ("probability that both proposals are kept", _f(both_kept, 4)),
                ("largest difference, emitted pairs against the target's p(t1) p(t2 given t1)", _noise(err)),
                ("total variation, emitted pairs against the target", _noise(tv_spec)),
                ("total variation, the draft's own pairs against the target", _f(tv_draft, 4)),
            ], "lr"))


# --------------------------------------------------------------------------------------------- drafters

def prompt_lookup(context, n, k):
    """The latest earlier occurrence of the context's last n tokens (starting at most len - n - 1) and the k tokens after it."""
    if k == 0 or len(context) <= n:
        return [], None
    suffix = context[-n:]
    for i in range(len(context) - n - 1, -1, -1):
        if context[i:i + n] == suffix:
            return context[i + n:i + n + k], i
    return [], None


LOOKUP_TEXT = "the cat sees the dog . the dog chases the cat . the cat sees the"


def lookup_table():
    ctx = [dec.ID[w] for w in LOOKUP_TEXT.split()]
    rows = []
    for n in (1, 2, 3):
        props, at = prompt_lookup(ctx, n, 3)
        rows.append((n, f"`{_w(ctx[-n:])}`", "none" if at is None else at, _w(props)))
    return (f"The context is the {len(ctx)} tokens `{LOOKUP_TEXT}`, positions counted from 0. For each n the drafter finds the latest "
            "earlier start of the context's last n tokens (a start at most len - n - 1, so the suffix itself is not a match) and proposes "
            "the k = 3 tokens that followed it.\n\n"
            + _table(["n", "suffix", "latest earlier start", "proposals"], rows, "rlrl"))


# The configuration of a published DFlash2 draft, as a test of `crates/dflash2/src/config.rs` holds it.
DRAFT_CFG = {"layers": 5, "d": 5120, "ff": 40960, "heads": 80, "kv_heads": 8, "head_dim": 128, "block": 16, "conv_kernel": 4,
             "conv_group": 16, "rank": 256, "top_k": 32, "taps": 5, "vocab": 100352, "window": 4096,
             "kinds": ("sliding", "sliding", "sliding", "sliding", "full")}


def draft_inventory(cfg=DRAFT_CFG):
    """(name, shape, count) of every tensor of the draft, in the order the chapter lists them."""
    d, ff, hq, hkv = cfg["d"], cfg["ff"], cfg["heads"] * cfg["head_dim"], cfg["kv_heads"] * cfg["head_dim"]
    dyn = 2 * cfg["conv_kernel"] * (d // cfg["conv_group"])
    per_layer = [("q", (hq, d)), ("k", (hkv, d)), ("v", (hkv, d)), ("o", (d, hq)), ("gate", (ff, d)), ("up", (ff, d)),
                 ("down", (d, ff)), ("attention conv projection", (dyn, d)), ("MLP conv projection", (dyn, d)),
                 ("attention conv base", (2, cfg["conv_kernel"], d)), ("MLP conv base", (2, cfg["conv_kernel"], d)),
                 ("input norm", (d,)), ("post-attention norm", (d,)), ("q norm", (cfg["head_dim"],)), ("k norm", (cfg["head_dim"],))]
    items = [("fc (fuses the taps)", (d, cfg["taps"] * d))]
    items += [(f"each layer: {n}", s) for n, s in per_layer]
    items += [("hidden norm", (d,)), ("final norm", (d,)), ("selector projection", (cfg["rank"], d)),
              ("selector predecessor codebook", (cfg["vocab"], cfg["rank"])), ("selector successor codebook", (cfg["vocab"], cfg["rank"]))]
    out = []
    for name, shape in items:
        n = math.prod(shape)
        out.append((name, shape, n * (cfg["layers"] if name.startswith("each layer") else 1)))
    return out


def cache_rows(kind, cap, cfg=DRAFT_CFG):
    """Rows one sequence's draft cache keeps for a layer: a sliding layer the window plus one block (at most the context plus
    one block), a full layer every position plus one block."""
    if kind == "sliding":
        return min(cap, cfg["window"]) + cfg["block"]
    return cap + cfg["block"]


def dflash2_inventory():
    cfg = DRAFT_CFG
    inv = draft_inventory()
    rows = [(n, " x ".join(str(x) for x in s), _int(c)) for n, s, c in inv]
    total = sum(c for _, _, c in inv)
    linear = sum(c for n, _, c in inv if "norm" not in n and "base" not in n and "selector" not in n)
    kv_row = 2 * cfg["kv_heads"] * cfg["head_dim"] * 4
    cache = [(cap,) + tuple(cache_rows(k, cap) for k in ("sliding", "full")) for cap in (1024, 32768, 1048576)]
    crow = [(_int(c), _int(s), _int(f), _int((4 * s + f) * kv_row)) for c, s, f in cache]
    return (f"The draft has {cfg['layers']} layers of width {cfg['d']} (feed-forward width {_int(cfg['ff'])}, {cfg['heads']} query heads and "
            f"{cfg['kv_heads']} key-value heads of width {cfg['head_dim']}); four layers attend within a window of {_int(cfg['window'])} "
            f"positions and the fifth to every position; it reads the target's residual stream at {cfg['taps']} layers, denoises blocks of "
            f"{cfg['block']} rows, wraps each sublayer in a dynamic convolution of {cfg['conv_kernel']} taps per group of {cfg['conv_group']} "
            f"channels, and its selector has rank {cfg['rank']} over the top {cfg['top_k']} candidates of a vocabulary of {_int(cfg['vocab'])}. "
            "'each layer' rows are summed over the five layers. The draft holds no embedding table and no output head: it uses the target's.\n\n"
            + _table(["tensor", "shape", "parameters"], rows, "llr")
            + f"\n\nIn all {_int(total)} parameters ({total * 2 / 1e9:.2f} GB in bf16), of which {_int(linear)} are in the matrices a pass "
            "multiplies on the device. A draft cache row of one layer holds the keys and values of one position, "
            f"2 x {cfg['kv_heads']} x {cfg['head_dim']} values of 4 bytes = {_int(kv_row)} bytes. A sliding layer keeps min(positions, {_int(cfg['window'])}) + {cfg['block']} rows and the "
            f"full layer positions + {cfg['block']} rows, so a sequence shorter than the window fills its sliding layers as well; "
            "one sequence's cache holds:\n\n"
            + _table(["positions in the sequence", "rows of a sliding layer", "rows of the full layer", "bytes, all five layers"], crow))
