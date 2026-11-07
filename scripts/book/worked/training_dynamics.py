# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Training Dynamics and Debugging".

Plain Python 3 on floats (IEEE double precision), written from the chapter's own
description and independent of any engine code. Each public function returns a
markdown string that the chapter embeds between `<!-- worked: training_dynamics.<name> -->`
markers. The same module supplies the data of the chapter's figures.

Two small networks carry the whole chapter.

  the gallery network   x in R^8 -> 32 -> 32 -> 3 logits, ReLU, softmax cross-entropy,
                        trained by minibatch gradient descent on a synthetic 3-class task
  the deep stack        x in R^8 -> 16 -> 24 weight matrices of 16 x 16 -> 3 logits, built plain
                        (a = g(W a + b)) or residual (a = a + s g(W a + b)), the same 24 matrices

Everything random comes from the book's SplitMix64 (`bookrng.generator(name)`). The
module does the arithmetic of a training loop by hand: a forward pass that keeps what
the backward pass needs, the backward pass of chapters 4 and 5, plain gradient descent
with the optional global-norm clipping of chapter 8, and the diagnostics the chapter
teaches (gradient norms, update ratios, dead fractions, finiteness scans).

Swedish Embedded AB implements solutions for finding out why a training run misbehaves.
If your team needs expertise in training diagnostics for neural networks, you can procure
our services by sending an email to info@swedishembedded.com.
"""
import math
from functools import lru_cache
from operator import mul

import bookrng

# ------------------------------------------------------------------ constants

IN_DIM = 8              # inputs of the gallery task
CLASSES = 3             # classes of the gallery task
HIDDEN = 32             # units per hidden layer of the gallery network
N_TRAIN = 512           # training examples of the healthy gallery runs
N_VAL = 400             # validation examples (the teacher's labels, no flips)
STEPS = 400             # gradient steps of a gallery run
BATCH = 32              # minibatch size of a gallery run
ETA = 0.1               # learning rate of the healthy run
EVAL_EVERY = 20         # steps between two evaluations on the full sets
PROBE = 256             # training examples used to decide whether a unit is dead

DATA_NAME = "training_dynamics.data"
SMALL_DATA_NAME = "training_dynamics.small_data"
INIT_NAME = "training_dynamics.init"
BATCH_NAME = "training_dynamics.batches"

# the deep stack
STACK_WIDTH = 16
STACK_MATRICES = 24     # 16 x 16 weight matrices between the input layer and the read-out
STACK_BATCH = 32
STACK_NAME = "training_dynamics.stack"

LN3 = math.log(CLASSES)


# ------------------------------------------------------------------ formatting

def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _sci(v, d=2):
    if v != v:
        return "NaN"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return f"{v:.{d}e}"


def _num(v, d=4):
    """Fixed notation for moderate values, scientific for the rest, NaN and inf by name."""
    if v != v:
        return "NaN"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    if v == 0.0:
        return f"{0:.{d}f}"
    return f"{v:.{d}f}" if 1e-3 <= abs(v) < 1e4 else _sci(v)


def _gen_note(name):
    s = bookrng.seed_of(name)
    return f"Generator: the book's SplitMix64, seed 0x{s:016x} ({s})"


# ------------------------------------------------------- IEEE arithmetic helpers
# Python raises where IEEE arithmetic returns inf or NaN; the chapter's NaN hunt needs the
# IEEE answers, so these three functions give them.

def ieee_exp(x):
    """exp(x) as IEEE double arithmetic gives it: inf above 709.78, NaN for NaN."""
    if x != x:
        return x
    try:
        return math.exp(x)
    except OverflowError:
        return math.inf


def ieee_log(x):
    """log(x): -inf at 0, NaN below 0 or for NaN."""
    if x != x or x < 0.0:
        return math.nan
    if x == 0.0:
        return -math.inf
    return math.log(x)


def ieee_div(a, b):
    """a / b with the IEEE rules for a zero divisor: 0/0 is NaN, a/0 is +-inf."""
    if b == 0.0:
        if a == 0.0 or a != a:
            return math.nan
        return math.copysign(math.inf, a) * math.copysign(1.0, b)
    return a / b


def is_finite(x):
    return x == x and abs(x) != math.inf


def dot(a, b):
    return sum(map(mul, a, b))


# --------------------------------------------------------------------- the task

@lru_cache(maxsize=None)
def make_data(name, n_train, n_val, flip):
    """Synthetic 3-class task. Draw order from bookrng.generator(name):

    1. the teacher: the three vectors u_0, u_1, u_2 and then the three vectors v_0, v_1, v_2, each of 8 normal()
       draws, so 48 draws;
    2. n_train inputs of 8 normal() draws, row by row; 3. n_val inputs likewise;
    4. for each training example in order, one uniform(); when it is below `flip`, one integer(2) more.

    The teacher's score for class k is s_k(x) = 2 (u_k . x) + (v_k . x)^2, the label is the index of the largest
    score (the first on a tie), and a flipped training label is moved by 1 + integer(2) classes (mod 3)."""
    g = bookrng.generator(name)
    U = [[g.normal() for _ in range(IN_DIM)] for _ in range(CLASSES)]
    V = [[g.normal() for _ in range(IN_DIM)] for _ in range(CLASSES)]

    def label(x):
        s = [2.0 * dot(u, x) + dot(v, x) ** 2 for u, v in zip(U, V)]
        return max(range(CLASSES), key=lambda k: s[k])

    Xt = [[g.normal() for _ in range(IN_DIM)] for _ in range(n_train)]
    Xv = [[g.normal() for _ in range(IN_DIM)] for _ in range(n_val)]
    yt = [label(x) for x in Xt]
    yv = [label(x) for x in Xv]
    for i in range(n_train):
        if g.uniform() < flip:
            yt[i] = (yt[i] + 1 + g.integer(2)) % CLASSES
    return Xt, yt, Xv, yv


def gallery_data():
    return make_data(DATA_NAME, N_TRAIN, N_VAL, 0.0)


# ------------------------------------------------------------------ activations

LEAK = 0.1


def act_fwd(name, z):
    if name == "relu":
        return z if z > 0.0 else 0.0
    if name == "leaky":
        return z if z > 0.0 else LEAK * z
    if name == "tanh":
        return math.tanh(z)
    if name == "sigmoid":
        return 1.0 / (1.0 + math.exp(-z)) if z >= 0.0 else math.exp(z) / (1.0 + math.exp(z))
    if name == "identity":
        return z
    raise ValueError(name)


def act_slope(name, z, a):
    """g'(z) given z and a = g(z); a ReLU's slope at z <= 0 is 0."""
    if name == "relu":
        return 1.0 if z > 0.0 else 0.0
    if name == "leaky":
        return 1.0 if z > 0.0 else LEAK
    if name == "tanh":
        return 1.0 - a * a
    if name == "sigmoid":
        return a * (1.0 - a)
    return 1.0


# ---------------------------------------------------------- layers and networks
# A layer is a dict. "dense": a = g(W a_in + b). "res": a = a_in + s g(W a_in + b) with a fixed scalar s.
# "P" holds its parameters (matrices as lists of rows, vectors as lists), "G" its gradients.

def dense(W, b, act):
    return {"type": "dense", "act": act, "P": {"W": W, "b": b}}


def residual(W, b, act, s):
    return {"type": "res", "act": act, "scale": s, "P": {"W": W, "b": b}}


def _affine(W, b, A):
    return [[dot(w, a) + bj for w, bj in zip(W, b)] for a in A]


def forward(layers, X):
    """Run X through the layers; returns (logits, caches)."""
    A, caches = X, []
    for L in layers:
        P = L["P"]
        if L["type"] == "dense":
            Z = _affine(P["W"], P["b"], A)
            Out = [[act_fwd(L["act"], z) for z in r] for r in Z]
            caches.append((A, Z, Out))
        else:
            Z = _affine(P["W"], P["b"], A)
            sc = L["scale"]
            H = [[act_fwd(L["act"], z) for z in r] for r in Z]
            Out = [[x + sc * h for x, h in zip(ra, rh)] for ra, rh in zip(A, H)]
            caches.append((A, Z, H))
        A = Out
    return A, caches


def _outer_sum(D, A):
    """sum over examples of d[n] a[n]^T as a list of rows: out[j][i] = sum_n D[n][j] A[n][i]."""
    Dt, At = list(zip(*D)), list(zip(*A))
    return [[dot(d, a) for a in At] for d in Dt]


def backward(layers, caches, dOut, mask_bug=False):
    """Chapters 4 and 5. dOut[n][j] is d loss / d logits (already divided by the batch size).

    Fills L["G"] for every layer. Returns the gradient with respect to the network input and the list
    of the norms of the gradient arriving at the output of each layer (for the diagnostics)."""
    arrive = []
    for L, (A, Z, Out) in zip(reversed(layers), reversed(caches)):
        P = L["P"]
        arrive.append(math.sqrt(sum(v * v for r in dOut for v in r)))
        if L["type"] == "dense":
            if mask_bug:
                dZ = dOut
            else:
                dZ = [[d * act_slope(L["act"], z, o) for d, z, o in zip(rd, rz, ro)] for rd, rz, ro in zip(dOut, Z, Out)]
            L["G"] = {"W": _outer_sum(dZ, A), "b": [sum(c) for c in zip(*dZ)]}
            Wt = list(zip(*P["W"]))
            dOut = [[dot(c, dz) for c in Wt] for dz in dZ]
        else:
            sc = L["scale"]
            H = Out  # for a residual layer the cache holds g(Z), not the layer output
            dZ = [[sc * d * act_slope(L["act"], z, h) for d, z, h in zip(rd, rz, rh)] for rd, rz, rh in zip(dOut, Z, H)]
            L["G"] = {"W": _outer_sum(dZ, A), "b": [sum(c) for c in zip(*dZ)]}
            Wt = list(zip(*P["W"]))
            dOut = [[d + dot(c, dzr) for d, c in zip(rd, Wt)] for rd, dzr in zip(dOut, dZ)]
    arrive.reverse()
    return dOut, arrive


def n_params(layers):
    return sum(len(v) * (len(v[0]) if isinstance(v[0], list) else 1) for L in layers for v in L["P"].values())


def _flat(v):
    return [x for r in v for x in r] if isinstance(v[0], list) else list(v)


def grad_vector(layers):
    return [x for L in layers for name in L["P"] for x in _flat(L["G"][name])]


def param_vector(layers):
    return [x for L in layers for name in L["P"] for x in _flat(L["P"][name])]


def _each(layers, fn):
    """Call fn(param_list, grad_list) for every parameter row (a vector counts as one row)."""
    for L in layers:
        for name, p in L["P"].items():
            g = L["G"][name]
            if isinstance(p[0], list):
                for pr, gr in zip(p, g):
                    fn(pr, gr)
            else:
                fn(p, g)


def sgd_update(layers, eta, scale=1.0, sign=-1.0):
    """w <- w - eta * scale * g for every parameter (sign +1 is the injected bug of the debugging table)."""
    c = sign * eta * scale

    def f(p, g):
        for j in range(len(p)):
            p[j] += c * g[j]
    _each(layers, f)


def global_norm(layers):
    return math.sqrt(sum(x * x for x in grad_vector(layers)))


def layer_grad_norms(layers):
    """Frobenius norm of the gradient of each layer's parameters."""
    return [math.sqrt(sum(x * x for name in L["P"] for x in _flat(L["G"][name]))) for L in layers]


# ------------------------------------------------------------- losses (softmax CE)

def ce_stable(logits, y):
    """Mean softmax cross-entropy through log-sum-exp (chapter 3) and the gradient (p - t) / B."""
    n = len(logits)
    total, d = 0.0, []
    for r, c in zip(logits, y):
        m = max(r)
        if not is_finite(m):
            total += math.nan if m != m else math.inf
            d.append([math.nan] * len(r))
            continue
        e = [math.exp(v - m) for v in r]
        s = sum(e)
        total += math.log(s) + m - r[c]
        p = [v / s for v in e]
        p[c] -= 1.0
        d.append([v / n for v in p])
    return total / n, d


def ce_naive(logits, y):
    """The textbook formula p = exp(z) / sum exp(z), loss -log p_c, with IEEE overflow and 0/0 rules."""
    n = len(logits)
    total, d = 0.0, []
    for r, c in zip(logits, y):
        e = [ieee_exp(v) for v in r]
        s = sum(e)
        p = [ieee_div(v, s) for v in e]
        total += -ieee_log(p[c])
        p[c] -= 1.0
        d.append([v / n for v in p])
    return total / n, d


def ce_batch_axis(logits, y):
    """The injected bug: softmax taken over the examples of the batch, one column (class) at a time."""
    n, k = len(logits), len(logits[0])
    cols = list(zip(*logits))
    P = []
    for col in cols:
        m = max(col)
        e = [math.exp(v - m) for v in col]
        s = sum(e)
        P.append([v / s for v in e])
    counts = [sum(1 for c in y if c == j) for j in range(k)]
    total = 0.0
    for i, c in enumerate(y):
        total += -math.log(P[c][i])
    d = [[(counts[j] * P[j][i] - (1.0 if y[i] == j else 0.0)) / n for j in range(k)] for i in range(n)]
    return total / n, d


def accuracy(logits, y):
    return sum(1 for r, c in zip(logits, y) if max(range(len(r)), key=lambda j: r[j]) == c) / len(y)


def evaluate(layers, X, y):
    logits, _ = forward(layers, X)
    loss, _ = ce_stable(logits, y)
    return loss, accuracy(logits, y)


# ---------------------------------------------------------------- initialisation

def init_mlp(dims, act="relu", scale=1.0, name=INIT_NAME, bias=None):
    """Hidden weights ~ scale * sqrt(2 / fan_in) * normal() (He), the read-out ~ scale * sqrt(1 / fan_in) * normal();
    layer by layer, row by row (row j of W has fan_in draws); biases 0 except those named in `bias`,
    a dict {layer index: value} that sets every bias of that layer."""
    g = bookrng.generator(name)
    layers = []
    n_layers = len(dims) - 1
    for l, (fi, fo) in enumerate(zip(dims[:-1], dims[1:])):
        last = l == n_layers - 1
        s = scale * math.sqrt((1.0 if last else 2.0) / fi)
        W = [[s * g.normal() for _ in range(fi)] for _ in range(fo)]
        b = [(bias or {}).get(l, 0.0)] * fo
        layers.append(dense(W, list(b), "identity" if last else act))
    return layers


def copy_layers(layers):
    return [{"type": L["type"], "act": L["act"], **({"scale": L["scale"]} if "scale" in L else {}),
             "P": {k: [list(r) for r in v] if isinstance(v[0], list) else list(v) for k, v in L["P"].items()}}
            for L in layers]


# ------------------------------------------------------------ training: one loop

def dead_fraction(layers, Xp):
    """Share of the hidden units (layers with a ReLU or leaky ReLU) whose pre-activation is <= 0 on every probe input."""
    dead = total = 0
    A = Xp
    for L in layers:
        if L["type"] != "dense":
            break
        Z = _affine(L["P"]["W"], L["P"]["b"], A)
        if L["act"] in ("relu", "leaky"):
            for col in zip(*Z):
                total += 1
                if all(z <= 0.0 for z in col):
                    dead += 1
        A = [[act_fwd(L["act"], z) for z in r] for r in Z]
    return dead / total if total else 0.0


def train(layers, data, *, eta=ETA, batch=BATCH, steps=STEPS, clip=None, loss="stable", bug=None,
          batch_name=BATCH_NAME, eval_every=EVAL_EVERY, probe=None, n_pool=None, input_scale=1.0, hook=None):
    """Minibatch gradient descent. Per step t: draw `batch` indices with integer(n_pool) from the stream
    `batch_name` (with replacement), forward, loss, backward, optional global-norm clip (chapter 8), update.

    Returns a record dict: step_loss[t] (the minibatch loss before the update), grad_norm[t] (global norm before
    clipping), max_logit[t] (largest |logit| of the minibatch), update_ratio[t] (norm of the update over norm of
    the weights), evals [(t, train_loss, val_loss, train_acc, val_acc)] on the full sets after t updates (t = 0,
    every eval_every steps, and the last), dead [(t, fraction)] at the same moments when `probe` inputs are given,
    and first_nonfinite (the first step whose minibatch loss is not finite, or None). `batch=None` uses the whole
    pool, in order, at every step; `hook(t, layers, X, y)` is called before the forward pass of step t."""
    Xt, yt, Xv, yv = data
    if input_scale != 1.0:
        Xt = [[input_scale * v for v in r] for r in Xt]
        Xv = [[input_scale * v for v in r] for r in Xv]
    n = n_pool or len(Xt)
    g = bookrng.generator(batch_name)
    lossfn = {"stable": ce_stable, "naive": ce_naive, "batch_axis": ce_batch_axis}[loss]
    rec = {"step_loss": [], "grad_norm": [], "max_logit": [], "update_ratio": [], "evals": [], "dead": [],
           "first_nonfinite": None, "layers": layers}

    def snapshot(t):
        trl, tra = evaluate(layers, Xt[:n], yt[:n])
        val, vaa = evaluate(layers, Xv, yv)
        rec["evals"].append((t, trl, val, tra, vaa))
        if probe is not None:
            rec["dead"].append((t, dead_fraction(layers, probe)))

    for t in range(steps):
        if t % eval_every == 0:
            snapshot(t)
        if batch is None:
            X, y = Xt[:n], yt[:n]
        else:
            idx = [g.integer(n) for _ in range(batch)]
            X = [Xt[i] for i in idx]
            y = [yt[i] for i in idx]
        if hook is not None:
            hook(t, layers, X, y)
        if bug == "labels":
            y = list(y)
            g.shuffle(y)
        logits, caches = forward(layers, X)
        L, d = lossfn(logits, y)
        backward(layers, caches, d, mask_bug=(bug == "nomask"))
        if bug == "zero1":
            for name, v in layers[0]["G"].items():
                if isinstance(v[0], list):
                    for r in v:
                        for j in range(len(r)):
                            r[j] = 0.0
                else:
                    for j in range(len(v)):
                        v[j] = 0.0
        gn = global_norm(layers)
        scale = 1.0
        if clip is not None:
            scale = min(1.0, clip / (gn + 1e-6)) if is_finite(gn) else 1.0
        wn = math.sqrt(sum(x * x for x in param_vector(layers)))
        rec["step_loss"].append(L)
        rec["grad_norm"].append(gn)
        rec["max_logit"].append(max(abs(v) for r in logits for v in r))
        rec["update_ratio"].append(eta * scale * gn / wn if wn > 0.0 and is_finite(wn) else math.nan)
        if rec["first_nonfinite"] is None and not is_finite(L):
            rec["first_nonfinite"] = t
        sgd_update(layers, eta, scale, sign=1.0 if bug == "sign" else -1.0)
    snapshot(steps)
    return rec


# ================================================================== the gallery

GALLERY = [
    # key, label, what is changed, train() keyword arguments
    ("healthy", "healthy", "nothing: He start, rate 0.1, batch 32", {}),
    ("diverging", "diverging", "rate 10 instead of 0.1", {"eta": 10.0}),
    ("plateau", "plateau", "every weight 0.02 times its He scale", {"scale": 0.02}),
    ("slow", "too slow", "rate 0.003 instead of 0.1", {"eta": 0.003}),
    ("noisy", "noisy", "batch 2 instead of 32", {"batch": 2}),
]


@lru_cache(maxsize=None)
def gallery_run(key, nudge=0.0):
    """One run of the gallery on the shared data, from the shared draws of the start (`nudge` is added to the first weight:
    the test of whether a printed digit is reproducible)."""
    spec = {k: kw for k, _, _, kw in GALLERY}[key]
    data = gallery_data()
    layers = init_mlp([IN_DIM, HIDDEN, HIDDEN, CLASSES], scale=spec.get("scale", 1.0))
    layers[0]["P"]["W"][0][0] += nudge
    kw = {k: v for k, v in spec.items() if k != "scale"}
    return train(layers, data, **kw)


# ================================================================ the deep stack

def build_stack(kind, act, gain, matrices=STACK_MATRICES, omega=None, name=STACK_NAME):
    """Input layer (8 -> 16), `matrices` weight matrices of 16 x 16, read-out (16 -> 3). Draws from
    bookrng.generator(name), all normal(), in the order: input layer, the hidden matrices in order, read-out;
    rows are drawn one after the other (row j has fan_in draws), biases are 0.

    kind "plain":    matrices dense layers a = g(W a + b), W ~ gain * c / sqrt(fan_in) * normal(),
                     with c = sqrt(2) for relu and leaky and 1 otherwise.
    kind "residual": matrices layers a <- a + s g(W a + b) with the same W and the branch scale s = sqrt(omega),
                     omega = 1 / (2 * matrices) unless given (chapter 9's scaled branch).
    The input layer uses the same gain and activation; the read-out ~ normal() / sqrt(fan_in)."""
    g = bookrng.generator(name)
    w = STACK_WIDTH
    c = math.sqrt(2.0) if act in ("relu", "leaky") else 1.0

    def mat(fo, fi, s):
        return [[s * g.normal() for _ in range(fi)] for _ in range(fo)]

    layers = [dense(mat(w, IN_DIM, gain * c / math.sqrt(IN_DIM)), [0.0] * w, act)]
    om = omega if omega is not None else 1.0 / (2 * matrices)
    for _ in range(matrices):
        W = mat(w, w, gain * c / math.sqrt(w))
        if kind == "plain":
            layers.append(dense(W, [0.0] * w, act))
        else:
            layers.append(residual(W, [0.0] * w, act, math.sqrt(om)))
    layers.append(dense(mat(CLASSES, w, 1.0 / math.sqrt(w)), [0.0] * CLASSES, "identity"))
    return layers



# ============================================================== gallery tables

def _entropy(y):
    n = len(y)
    return -sum((y.count(k) / n) * math.log(y.count(k) / n) for k in range(CLASSES) if y.count(k))


def small_data():
    """The overfitting task: 24 training examples with a quarter of the labels flipped, the usual validation set."""
    return make_data(SMALL_DATA_NAME, 24, N_VAL, 0.25)


@lru_cache(maxsize=None)
def overfit_run(nudge=0.0):
    layers = init_mlp([IN_DIM, HIDDEN, HIDDEN, CLASSES])
    layers[0]["P"]["W"][0][0] += nudge
    return train(layers, small_data(), eta=ETA, batch=8, steps=600, n_pool=24)


@lru_cache(maxsize=None)
def nan_run():
    """The diverging run again, with the textbook softmax cross-entropy instead of the log-sum-exp form."""
    layers = init_mlp([IN_DIM, HIDDEN, HIDDEN, CLASSES])
    return train(layers, gallery_data(), eta=10.0, steps=12, loss="naive")


def log_slope(evals, start=100):
    """Least-squares slope of log10(train loss) against step, per 100 steps, over the evaluations at step >= start."""
    pts = [(t, math.log10(l)) for t, l, *_ in evals if t >= start and is_finite(l) and l > 0.0]
    if len(pts) < 3:
        return math.nan
    mt = sum(t for t, _ in pts) / len(pts)
    ml = sum(v for _, v in pts) / len(pts)
    return 100.0 * sum((t - mt) * (v - ml) for t, v in pts) / sum((t - mt) ** 2 for t, _ in pts)


def gallery_setup():
    Xt, yt, Xv, yv = gallery_data()
    sx, sy, _, _ = small_data()
    P = IN_DIM * HIDDEN + HIDDEN + HIDDEN * HIDDEN + HIDDEN + HIDDEN * CLASSES + CLASSES
    counts = [yt.count(k) for k in range(CLASSES)]
    rows = [
        ("network", f"{IN_DIM} -> {HIDDEN} -> {HIDDEN} -> {CLASSES}, ReLU, softmax cross-entropy, {P} parameters"),
        ("start", "hidden weights sqrt(2 / fan_in) * normal(), read-out weights sqrt(1 / fan_in) * normal(), biases 0"),
        ("start draws", _gen_note(INIT_NAME) + "; layer by layer, row by row"),
        ("task", "3 classes, label = argmax_k [2 (u_k . x) + (v_k . x)^2] (first index on a tie), the teacher vectors u_k, v_k drawn as below"),
        ("task draws", _gen_note(DATA_NAME) + "; order: u_0, u_1, u_2, v_0, v_1, v_2 (8 normal() each), "
                       f"{N_TRAIN} training inputs, {N_VAL} validation inputs (8 normal() each)"),
        ("training set", f"{N_TRAIN} examples, class counts {counts}, entropy of the label distribution {_num(_entropy(yt))}; "
                         f"ln 3 = {_num(LN3)}"),
        ("validation set", f"{N_VAL} examples, teacher labels"),
        ("step", f"plain gradient descent, minibatch of {BATCH} indices drawn with replacement by integer({N_TRAIN}) from the "
                 f"stream {bookrng.seed_of(BATCH_NAME):#018x}; {STEPS} steps; learning rate {ETA}"),
        ("overfitting task", f"{len(sx)} training examples, {N_VAL} validation examples with the teacher's labels; generator seed "
                             f"{bookrng.seed_of(SMALL_DATA_NAME):#018x}; the draws are the 48 of the teacher, the {len(sx)} training inputs, the {N_VAL} validation inputs, "
                             f"then for each training example in order one uniform() and, when it is below 0.25, one integer(2): the label moves by 1 + integer(2) classes (mod 3); "
                             f"the run uses the same start, minibatches of 8 indices drawn with replacement by integer(24) from the batch stream above, 600 steps at rate {ETA}"),
        ("evaluation", f"'loss after t updates' is the mean cross-entropy (log-sum-exp form) of the whole training set of the run, or of the {N_VAL} validation examples, "
                       "with the weights after t gradient steps; accuracy takes the first largest logit"),
    ]
    return _table(["item", "value"], rows, "ll")


NUDGE = 1e-12      # change of the first weight used to test which printed digits another implementation can reproduce


def _numm(v, w, d=4):
    """_num(v, d) when the nudged value w agrees to the third significant digit; otherwise 'about x' with the digits on which the two
    agree (two significant digits if they differ by at most 2%, else one), or 'unstable'. Non-finite values are printed by name."""
    if not (is_finite(v) and is_finite(w)):
        return _num(v, d)
    if abs(v - w) <= 1e-3 * max(abs(v), 1e-300):
        return _num(v, d)
    if abs(v - w) <= 0.02 * max(abs(v), 1e-300):
        return f"about {(v + w) / 2:.2g}"
    if f"{v:.1g}" == f"{w:.1g}":
        return f"about {v:.1g}"
    return "unstable"


def gallery_table():
    entries = [(k, lab, chg) for k, lab, chg, _ in GALLERY]
    runs = {k: gallery_run(k) for k, *_ in entries}
    twins = {k: gallery_run(k, NUDGE) for k, *_ in entries}
    entries.append(("overfit", "overfitting", "24 training examples, 25% wrong labels"))
    runs["overfit"] = overfit_run()
    twins["overfit"] = overfit_run(NUDGE)
    rows = []
    for key, lab, chg in entries:
        r, q = runs[key], twins[key]
        ev = {t: l for t, l, *_ in r["evals"]}
        ew = {t: l for t, l, *_ in q["evals"]}
        cells = [_numm(ev[m], ew[m], 3) for m in (0, 20, 100, 200, 400)]
        rows.append((lab, chg, *cells, _numm(max(r["step_loss"]), max(q["step_loss"]), 3), _numm(log_slope(r["evals"]), log_slope(q["evals"]), 3),
                     _numm(r["evals"][-1][4], q["evals"][-1][4], 3)))
    nan = nan_run()
    rows.append(("NaN", "the diverging run with the textbook softmax", _num(nan["evals"][0][1], 3), "NaN", "NaN", "NaN", "NaN",
                 "NaN", "NaN", "-"))
    note = ("A number that changes in its third significant digit when the first weight of the first layer is changed by 1e-12 is printed as 'about x' with the digits that survive, or as 'unstable'; "
            "the diverging run, whose loss reaches 1e22, is chaotic after its first steps and can only be reproduced to those digits. "
            "'Log10 slope per 100 steps' is the least-squares slope of log10 of the training loss against the step over the evaluations at steps 100, 120, ..., 400, times 100.\n\n")
    return note + _table(["run", "what differs from the healthy run", "step 0", "step 20", "step 100", "step 200", "step 400",
                   "largest step loss", "log10 slope per 100 steps", "val acc"], rows, "llrrrrrrrr")


# ============================================================ reading a curve

EMA_BETA = 0.9
NOISE_NAME = "training_dynamics.noise"


def ema(values, beta=EMA_BETA):
    """Bias-corrected exponential average: s_t = beta s_{t-1} + (1 - beta) x_t with s_{-1} = 0, reported as
    s_t / (1 - beta^(t+1)), the correction of chapter 7."""
    out, s = [], 0.0
    for t, v in enumerate(values):
        s = beta * s + (1.0 - beta) * v
        out.append(s / (1.0 - beta ** (t + 1)))
    return out


def smoothing_table():
    r = gallery_run("noisy")
    smooth = ema(r["step_loss"])
    full = {t: l for t, l, *_ in r["evals"]}
    rows = [(t, _num(r["step_loss"][t], 3), _num(smooth[t], 3), _num(full[t], 3)) for t in range(100, 201, 20)]
    marks = [t for t in range(20, STEPS, 20)]
    raw_dev = sum(abs(r["step_loss"][t] - full[t]) for t in marks) / len(marks)
    sm_dev = sum(abs(smooth[t] - full[t]) for t in marks) / len(marks)
    return (f"The noisy run (batch 2). Smoothing: beta = {EMA_BETA}, s_t = beta s_(t-1) + (1 - beta) loss_t with s_(-1) = 0, "
            f"reported divided by 1 - beta^(t+1). Full-set loss: the {N_TRAIN} training examples after t updates.\n\n"
            + _table(["step t", "minibatch loss at step t", "smoothed s_t", "full-set training loss"], rows)
            + f"\n\nOver the {len(marks)} steps t = 20, 40, ..., 380, the mean absolute difference from the full-set loss is {_num(raw_dev, 3)} for the "
              f"minibatch loss and {_num(sm_dev, 3)} for the smoothed loss.")


def noise_by_batch_table():
    """Spread of the minibatch loss at fixed weights: the weights are the healthy run's final ones."""
    layers = gallery_run("healthy")["layers"]
    Xt, yt, _, _ = gallery_data()
    logits, _ = forward(layers, Xt)
    per = [ce_stable([r], [c])[0] for r, c in zip(logits, yt)]
    mean = sum(per) / len(per)
    sd_ex = math.sqrt(sum((v - mean) ** 2 for v in per) / len(per))
    g = bookrng.generator(NOISE_NAME)
    rows = []
    n_draw = 2000
    for B in (1, 2, 8, 32, 128):
        vals = []
        for _ in range(n_draw):
            vals.append(sum(per[g.integer(len(per))] for _ in range(B)) / B)
        m = sum(vals) / n_draw
        sd = math.sqrt(sum((v - m) ** 2 for v in vals) / n_draw)
        rows.append((B, _num(m, 4), _num(sd, 4), _num(sd * math.sqrt(B), 4), _num(sd_ex / math.sqrt(B), 4)))
    note = (f"Fixed weights: the healthy run after {STEPS} steps; loss of one example, mean {_num(mean, 4)}, standard deviation "
            f"{_num(sd_ex, 4)} over the {N_TRAIN} training examples. {_gen_note(NOISE_NAME)}; for each B, {n_draw} minibatches in turn, "
            f"each of B indices from integer({N_TRAIN}) drawn with replacement, B = 1, 2, 8, 32, 128 in that order. "
            "The spread is the standard deviation (divisor n) of the minibatch means.")
    return note + "\n\n" + _table(["batch B", "mean of minibatch loss", "standard deviation", "std * sqrt(B)", "predicted: example std / sqrt(B)"], rows)


# ====================================================== causes, one run at a time

def _scim(v, w, d=2):
    """Scientific notation of v with d decimals if the nudged value w agrees to the third significant digit, else 'about' / 'unstable' (see _numm)."""
    if not (is_finite(v) and is_finite(w)):
        return _sci(v, d)
    if abs(v - w) <= 1e-3 * max(abs(v), 1e-300):
        return _sci(v, d)
    if abs(v - w) <= 0.02 * max(abs(v), 1e-300):
        return f"about {(v + w) / 2:.2g}"
    if f"{v:.1g}" == f"{w:.1g}":
        return f"about {v:.1g}"
    return "unstable"


def divergence_table():
    r = gallery_run("diverging")
    q = gallery_run("diverging", NUDGE)
    n, m = r["grad_norm"], q["grad_norm"]
    rows = []
    for t in range(0, 8):
        ratio = "" if t == 0 else _numm(n[t] / n[t - 1], m[t] / m[t - 1], 2)
        rows.append((t, _numm(r["step_loss"][t], q["step_loss"][t], 3), _numm(n[t], m[t], 3), ratio, _numm(r["max_logit"][t], q["max_logit"][t], 3)))
    return ("A number that changes in its third significant digit when the first weight of the first layer is changed by 1e-12 is printed as 'about x' or 'unstable' (as in the gallery table).\n\n" + _table(["step t", "minibatch loss", "global gradient norm", "norm / previous norm", "largest |logit|"], rows))


SWEEP_RATES = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 2.0, 3.0, 5.0, 10.0)


@lru_cache(maxsize=None)
def sweep_runs():
    data = gallery_data()
    out = []
    for eta in SWEEP_RATES:
        layers = init_mlp([IN_DIM, HIDDEN, HIDDEN, CLASSES])
        twin = init_mlp([IN_DIM, HIDDEN, HIDDEN, CLASSES])
        twin[0]["P"]["W"][0][0] += NUDGE
        out.append((eta, train(layers, data, eta=eta, steps=100, eval_every=100), train(twin, data, eta=eta, steps=100, eval_every=100)))
    return out


def lr_sweep_table():
    rows = []
    for eta, r, q in sweep_runs():
        l0, q0 = r["step_loss"][0], q["step_loss"][0]
        rows.append((f"{eta:g}", _numm(max(r["step_loss"]) / l0, max(q["step_loss"]) / q0, 3), _numm(r["evals"][-1][1], q["evals"][-1][1], 3),
                     _numm(r["evals"][-1][3], q["evals"][-1][3], 3)))
    return ("Each row is a fresh run of the healthy configuration with only the rate changed, 100 steps, the same start and the "
            "same batches. Largest step loss: the largest minibatch loss of the run divided by the loss at step 0. A number that changes in its third significant digit when the first weight "
            "of the first layer is changed by 1e-12 is printed as 'about x' or 'unstable'.\n\n"
            + _table(["rate", "largest step loss / step-0 loss", "full-set training loss after 100 steps", "training accuracy"], rows))


def clip_table():
    data = gallery_data()
    rows = []
    for eta, clip in ((10.0, None), (10.0, 1.0), (10.0, 0.1), (10.0, 0.01), (0.1, None), (0.1, 0.1)):
        layers = init_mlp([IN_DIM, HIDDEN, HIDDEN, CLASSES])
        twin = init_mlp([IN_DIM, HIDDEN, HIDDEN, CLASSES])
        twin[0]["P"]["W"][0][0] += NUDGE
        r = train(layers, data, eta=eta, clip=clip, steps=100, eval_every=100)
        q = train(twin, data, eta=eta, clip=clip, steps=100, eval_every=100)
        clipped = sum(1 for g in r["grad_norm"] if clip is not None and g > clip)
        clipped_q = sum(1 for g in q["grad_norm"] if clip is not None and g > clip)
        rows.append((f"{eta:g}", "none" if clip is None else f"{clip:g}", f"{clipped}" if clipped == clipped_q else f"about {clipped}",
                     _numm(max(r["step_loss"]), max(q["step_loss"]), 3), _numm(r["evals"][-1][1], q["evals"][-1][1], 3)))
    return ("A number that changes in its third significant digit when the first weight of the first layer is changed by 1e-12 is printed as 'about x' or 'unstable'.\n\n" + _table(["rate", "clip threshold (global norm)", "steps clipped of 100", "largest step loss", "full-set training loss after 100 steps"], rows))


def plateau_table():
    data = gallery_data()
    Xt, yt, _, _ = data
    rows = []
    for label, scale in (("healthy start", 1.0), ("0.02 times the He scale", 0.02)):
        layers = init_mlp([IN_DIM, HIDDEN, HIDDEN, CLASSES], scale=scale)
        logits, caches = forward(layers, Xt[:BATCH])
        loss, d = ce_stable(logits, yt[:BATCH])
        backward(layers, caches, d)
        gn = layer_grad_norms(layers)
        gb = math.sqrt(sum(x * x for x in layers[2]["G"]["b"]))
        rows.append((label, _num(loss, 4), _sci(gn[0], 2), _sci(gn[1], 2), _sci(gn[2], 2), _sci(gb, 2), _sci(gn[0] / gn[2], 2)))
    return (f"Loss and gradient on the first {BATCH} training examples at step 0; the norm of a layer is that of its weights and biases together. "
            f"ln 3 = {_num(LN3, 4)}; entropy of the training labels = {_num(_entropy(yt), 4)}.\n\n"
            + _table(["start", "loss at step 0", "gradient norm, layer 1", "layer 2", "layer 3 (read-out)", "of which the read-out bias", "layer 1 / layer 3"], rows))


def update_ratio_table():
    rows = []
    for key in ("healthy", "slow", "diverging"):
        r, q = gallery_run(key), gallery_run(key, NUDGE)
        meds = []
        for run in (r, q):
            srt = sorted(run["update_ratio"][:100])
            meds.append(0.5 * (srt[49] + srt[50]))      # the book's median of 100 values: the mean of the two middle ones
        ur, uq = r["update_ratio"], q["update_ratio"]
        rows.append((key, {"healthy": "0.1", "slow": "0.003", "diverging": "10"}[key], _scim(ur[0], uq[0]), _scim(ur[1], uq[1]), _scim(ur[2], uq[2]),
                     _scim(meds[0], meds[1])))
    return ("The update ratio of a step is (rate * gradient norm) / (norm of all weights before the step), the relative size of "
            "the step, with the gradient norm before any clipping. The median is the book's median of the 100 ratios of steps 0 to 99 (the mean of the 50th and 51st smallest). A number that changes in its third significant digit when the first weight of the first layer is changed by 1e-12 is printed as 'about x' or 'unstable'.\n\n"
            + _table(["run", "rate", "ratio at step 0", "step 1", "step 2", "median over steps 0 to 99"], rows))


def overfit_table():
    r = overfit_run()
    rows = [(t, _num(tl, 3), _num(vl, 3), _num(ta, 3), _num(va, 3)) for t, tl, vl, ta, va in r["evals"] if t in (0, 20, 40, 60, 100, 200, 400, 600)]
    best = min(r["evals"], key=lambda e: e[2])
    return (f"Training set of 24 examples (batch 8), validation set of {N_VAL}. Lowest validation loss {_num(best[2], 3)} after "
            f"{best[0]} updates.\n\n" + _table(["updates", "training loss", "validation loss", "training accuracy", "validation accuracy"], rows))


# ================================================================ NaN hunting

def nan_scan(layers, X, y):
    """Run the textbook loss op by op and report, for each intermediate array, its largest magnitude and how many
    entries are not finite. Rows: (name, max |value| over the finite entries, count of non-finite entries)."""
    rows = []

    def rec(name, arrays):
        vals = [v for a in arrays for v in a]
        bad = sum(1 for v in vals if not is_finite(v))
        fin = [abs(v) for v in vals if is_finite(v)]
        rows.append((name, max(fin) if fin else math.nan, bad))
    A = X
    n_layers = len(layers)
    for i, L in enumerate(layers):
        Z = _affine(L["P"]["W"], L["P"]["b"], A)
        rec(f"z{i + 1} = W{i + 1} a{i} + b{i + 1}", Z)
        A = [[act_fwd(L["act"], z) for z in r] for r in Z]
        if i < n_layers - 1:
            rec(f"a{i + 1} = relu(z{i + 1})", A)
    logits = A
    E = [[ieee_exp(v) for v in r] for r in logits]
    S = [[sum(r)] for r in E]
    P = [[ieee_div(v, s[0]) for v in r] for r, s in zip(E, S)]
    LP = [[-ieee_log(r[c])] for r, c in zip(P, y)]
    rec("e = exp(z3)", E)
    rec("s = sum of e", S)
    rec("p = e / s", P)
    rec("-log p[c]", LP)
    return rows


def nan_hunt_table():
    captured = {}

    def hook(t, layers, X, y):
        if t == 3:
            captured["rows"] = nan_scan(layers, X, y)
    layers = init_mlp([IN_DIM, HIDDEN, HIDDEN, CLASSES])
    r = train(layers, gallery_data(), eta=10.0, steps=5, loss="naive", hook=hook)
    stable = gallery_run("diverging")
    steps = _table(["step t", "textbook softmax: minibatch loss", "log-sum-exp form: minibatch loss", "largest |logit| (log-sum-exp run)"],
                   [(t, _num(r["step_loss"][t], 3), _num(stable["step_loss"][t], 3), _num(stable["max_logit"][t], 3)) for t in range(5)])
    scan = _table(["array at step 3, in the order computed", "largest finite |value|", "non-finite entries"],
                  [(name, _num(m, 3), c) for name, m, c in captured["rows"]], "lrr")
    return (f"The diverging run (rate 10) with the textbook loss. exp overflows above {math.log(1.7976931348623157e308):.2f}.\n\n"
            + steps + "\n\nA scan of every array of the forward pass of step 3, with the weights as they are at the start of that step:\n\n" + scan)


def nan_sources_table():
    big = [3e7 + 0.1] * 3

    def one_pass_var(x):
        n = len(x)
        m = sum(x) / n
        return sum(v * v for v in x) / n - m * m

    def two_pass_var(x):
        n = len(x)
        m = sum(x) / n
        return sum((v - m) ** 2 for v in x) / n
    z = [710.0, 0.0, -5.0]
    naive_p = [ieee_div(ieee_exp(v), sum(ieee_exp(u) for u in z)) for v in z]
    m = max(z)
    stable_e = [math.exp(v - m) for v in z]
    stable_p = [v / sum(stable_e) for v in stable_e]
    p, t = [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]
    full_ce = -sum(ti * ieee_log(pi) for ti, pi in zip(t, p))
    x = [3.0, 3.0, 3.0, 3.0]
    mu = sum(x) / 4
    var = sum((v - mu) ** 2 for v in x) / 4
    ln0 = [ieee_div(v - mu, math.sqrt(var + 0.0)) if var + 0.0 > 0 else ieee_div(v - mu, 0.0) for v in x]
    ln5 = [(v - mu) / math.sqrt(var + 1e-5) for v in x]
    v1 = one_pass_var(big)
    sq = math.sqrt(v1) if v1 >= 0 else math.nan
    rows = [
        ("exp overflow in a softmax", "z = (710, 0, -5); p = exp(z) / sum(exp(z))", _num(naive_p[0]), "subtract max(z) first (log-sum-exp)",
         _num(stable_p[0])),
        ("log of zero in a cross-entropy", "p = (1, 0, 0), t = (1, 0, 0); L = -sum t_i log p_i", _num(full_ce), "L = -log p_c (only the true class)", _num(-math.log(1.0))),
        ("0 / 0 in a layer norm", "x = (3, 3, 3, 3); (x - mean) / sqrt(var + eps) with eps = 0", _num(ln0[0]), "eps = 1e-5 under the root", _num(ln5[0])),
        ("negative variance", "x = (30000000.1, 30000000.1, 30000000.1); var = mean(x^2) - mean(x)^2, then sqrt",
         f"{_num(v1)} -> {_num(sq)}", "var = mean((x - mean(x))^2)", _num(math.sqrt(two_pass_var(big)), 4)),
        ("0 * infinity in a mask", "a = (inf, 2); mask = (0, 1); a * mask summed", _num(sum(ai * mi for ai, mi in zip([math.inf, 2.0], [0.0, 1.0]))),
         "select: take a_i where mask_i = 1, else 0", _num(sum(ai if mi == 1.0 else 0.0 for ai, mi in zip([math.inf, 2.0], [0.0, 1.0])))),
    ]
    return _table(["source", "computation (IEEE double)", "result", "guard", "guarded result"], rows, "lllll")


# ===================================================== vanishing and exploding

SPECTRAL_NAME = "training_dynamics.spectral"


def spectral_norm(W, iters=60, name=SPECTRAL_NAME):
    """Largest singular value of W by power iteration on W^T W: 60 iterations from a vector of normal() draws."""
    g = bookrng.generator(name)
    v = [g.normal() for _ in range(len(W[0]))]
    Wt = list(zip(*W))
    for _ in range(iters):
        u = [dot(w, v) for w in W]
        v = [dot(c, u) for c in Wt]
        nv = math.sqrt(sum(x * x for x in v))
        v = [x / nv for x in v]
    return math.sqrt(sum(dot(w, v) ** 2 for w in W))


JAC_DEPTHS = (1, 2, 4, 8, 16, 24)
JAC_CONFIGS = (("sigmoid", 1.0), ("tanh", 0.5), ("tanh", 1.0), ("tanh", 2.5))
JAC_PAIRS = 16
JAC_NAME = "training_dynamics.jacobian"


def backward_gain(layers, x, v, k):
    """Push the unit vector v (at the output of hidden layer 24) back through the last k hidden layers of a plain stack for the
    input x; returns the final norm (the product of the Jacobians' transposes applied to v), the product of the largest slope of
    each layer times its spectral norm, and the mean of g'^2 over the units of those layers."""
    out, caches = forward(layers[:-1], [x])
    d = [list(v)]
    bound, slope2, count = 1.0, 0.0, 0
    for L, (A, Z, Out) in list(zip(layers[:-1], caches))[::-1][:k]:
        slopes = [act_slope(L["act"], z, o) for z, o in zip(Z[0], Out[0])]
        bound *= max(slopes) * spectral_norm(L["P"]["W"])
        slope2 += sum(s * s for s in slopes)
        count += len(slopes)
        dz = [a * s for a, s in zip(d[0], slopes)]
        Wt = list(zip(*L["P"]["W"]))
        d = [[dot(c, dz) for c in Wt]]
    return math.sqrt(sum(a * a for a in d[0])), bound, slope2 / count


def jacobian_table():
    """For each configuration, over JAC_PAIRS (example, vector) pairs: the geometric mean of the measured norm after k layers,
    the prediction (n sigma^2 E[g'^2])^(k/2) with E[g'^2] measured over the same layers, and the product-of-norms bound."""
    Xt, _, _, _ = gallery_data()
    g = bookrng.generator(JAC_NAME)
    pairs = []
    for i in range(JAC_PAIRS):
        v = [g.normal() for _ in range(STACK_WIDTH)]
        nv = math.sqrt(sum(a * a for a in v))
        pairs.append((Xt[i], [a / nv for a in v]))
    rows = []
    for act, gain in JAC_CONFIGS:
        layers = build_stack("plain", act, gain)
        sigma2 = gain * gain / STACK_WIDTH
        meas = {k: 0.0 for k in JAC_DEPTHS}
        pred = {k: 0.0 for k in JAC_DEPTHS}
        bnd = {k: 0.0 for k in JAC_DEPTHS}
        for x, v in pairs:
            for k in JAC_DEPTHS:
                norm, bound, s2 = backward_gain(layers, x, v, k)
                meas[k] += math.log(norm)
                bnd[k] += math.log(bound)
                pred[k] += 0.5 * k * math.log(STACK_WIDTH * sigma2 * s2)
        label = f"{act}, gain {gain:g}"
        rows.append((label, "measured", *[_sci(math.exp(meas[k] / JAC_PAIRS), 2) for k in JAC_DEPTHS]))
        rows.append(("", "predicted", *[_sci(math.exp(pred[k] / JAC_PAIRS), 2) for k in JAC_DEPTHS]))
        rows.append(("", "norm bound", *[_sci(math.exp(bnd[k] / JAC_PAIRS), 2) for k in JAC_DEPTHS]))
    note = (f"Plain stack of {STACK_MATRICES} layers of width {STACK_WIDTH} (start of training). {_gen_note(STACK_NAME)}; draws in the order: input layer "
            f"({STACK_WIDTH} x {IN_DIM}), the {STACK_MATRICES} hidden matrices, read-out (rows one after the other, biases 0). Weights of the input layer and of the hidden matrices = gain * c / sqrt(fan_in) * normal() "
            f"with c = 1 for sigmoid and tanh (fan_in {IN_DIM} for the input layer, {STACK_WIDTH} for the hidden matrices); the read-out is normal() / sqrt({STACK_WIDTH}), without gain. "
            f"The spectral norm of a matrix (for the norm bound) is its largest singular value, found by 60 power iterations on W^T W from a start vector of {STACK_WIDTH} normal() draws of the generator `{SPECTRAL_NAME}` "
            f"(seed {bookrng.seed_of(SPECTRAL_NAME):#018x}), restarted for each matrix, each iteration normalising the vector. "
            f"Slopes use the activation's derivative at the pre-activation (sigmoid s(1 - s), tanh 1 - a^2). {JAC_PAIRS} pairs (first {JAC_PAIRS} training inputs, unit vector at the output of the last hidden layer "
            f"from normal() draws normalised, stream {bookrng.seed_of(JAC_NAME):#018x}). Entries are geometric means over the pairs of the norm "
            f"of the gradient after passing back through k layers, started from a vector of norm 1.")
    return note + "\n\n" + _table(["configuration", "quantity"] + [f"k = {k}" for k in JAC_DEPTHS], rows, "llrrrrrr")


def stack_matrix_norms(layers):
    """Gradient norm of each of the hidden weight matrices, from the input side to the output side."""
    return [math.sqrt(sum(x * x for r in L["G"]["W"] for x in r)) for L in layers[1:-1]]


GRAD_CONFIGS = (
    ("plain", "sigmoid", 1.0, "plain, sigmoid, gain 1"),
    ("plain", "tanh", 0.5, "plain, tanh, gain 0.5"),
    ("plain", "tanh", 1.0, "plain, tanh, gain 1"),
    ("plain", "tanh", 2.5, "plain, tanh, gain 2.5"),
    ("plain", "relu", 1.0, "plain, ReLU, He"),
    ("residual", "relu", 1.0, "residual, ReLU, He"),
)


@lru_cache(maxsize=None)
def stack_norms(kind, act, gain):
    """(loss, norms of dW for the 24 hidden matrices, norm of the gradient arriving at the output of hidden layer 1 and of 24)."""
    layers = build_stack(kind, act, gain)
    Xt, yt, _, _ = gallery_data()
    logits, caches = forward(layers, Xt[:STACK_BATCH])
    loss, d = ce_stable(logits, yt[:STACK_BATCH])
    _, arrive = backward(layers, caches, d)
    return loss, stack_matrix_norms(layers), arrive[1], arrive[STACK_MATRICES]


def stack_gradient_table():
    rows = []
    for kind, act, gain, label in GRAD_CONFIGS:
        loss, norms, a1, a24 = stack_norms(kind, act, gain)
        rows.append((label, _num(loss, 3), *[_sci(norms[i], 1) for i in (0, 5, 11, 17, 23)], _sci(norms[0] / norms[-1], 1), _sci(a1 / a24, 1)))
    note = (f"Gradient of the loss (mean over the batch of the cross-entropy) on the first {STACK_BATCH} training examples at the start; 'dW norm' is the Euclidean (Frobenius) norm of the gradient of the "
            f"{STACK_WIDTH} x {STACK_WIDTH} weight matrix, the whole batch summed in the gradient and the bias gradient not included. Weights: input layer and hidden matrices gain * c / sqrt(fan_in) * normal() "
            f"with c = sqrt(2) for ReLU and 1 for sigmoid and tanh (the gain is 1 for the ReLU rows), the read-out normal() / sqrt({STACK_WIDTH}) without gain, biases 0. Matrix 1 is the one nearest the input layer and "
            f"matrix 24 the one nearest the read-out. Same normal() draws for every row ({_gen_note(STACK_NAME)}), only the scales differ; the "
            f"residual stack adds s = sqrt(1 / 48) times the layer's output to its input. Last column: the norm of the gradient arriving at "
            f"the output of hidden layer 1 divided by the one arriving at the output of hidden layer 24, which is the product of the 23 layers' Jacobians "
            f"seen from above.")
    return note + "\n\n" + _table(["stack", "loss", "dW norm, matrix 1", "matrix 6", "matrix 12", "matrix 18", "matrix 24", "matrix 1 / matrix 24",
                                  "gradient at layer 1 / at layer 24"], rows)


def stack_grad_curves():
    """(label, [log10 of the gradient norm of matrix 1 .. 24]) for the figure."""
    return [(label, [math.log10(v) for v in stack_norms(kind, act, gain)[1]]) for kind, act, gain, label in GRAD_CONFIGS]


# ==================================================== residual connections

RES_DEPTHS = (1, 2, 4, 8, 16, 24)


def residual_backward(layers, x, v, k):
    """Push the unit vector v back through the last k residual layers of a stack for the input x. Returns the norm of the
    result, the lower bound prod(1 - s_i) and the upper bound prod(1 + s_i) with s_i = the spectral norm of the branch Jacobian
    s D_i W_i, and the largest difference between the backward pass and the formula d + J^T d with J built explicitly."""
    out, caches = forward(layers[:-1], [x])
    d = [list(v)]
    lower = upper = 1.0
    worst = 0.0
    for L, (A, Z, H) in list(zip(layers[:-1], caches))[::-1][:k]:
        sc = L["scale"]
        slopes = [act_slope(L["act"], z, h) for z, h in zip(Z[0], H[0])]
        W = L["P"]["W"]
        J = [[sc * sl * w for w in row] for sl, row in zip(slopes, W)]   # J[j][i] = d f_j / d x_i
        sj = spectral_norm(J)
        lower *= 1.0 - sj
        upper *= 1.0 + sj
        Jt = list(zip(*J))
        d_new = [a + dot(c, d[0]) for a, c in zip(d[0], Jt)]
        dz = [sc * a * sl for a, sl in zip(d[0], slopes)]
        Wt = list(zip(*W))
        d_back = [a + dot(c, dz) for a, c in zip(d[0], Wt)]
        worst = max(worst, max(abs(p - q) for p, q in zip(d_new, d_back)))
        d = [d_new]
    return math.sqrt(sum(a * a for a in d[0])), lower, upper, worst


def residual_identity_table():
    Xt, _, _, _ = gallery_data()
    g = bookrng.generator(JAC_NAME)
    pairs = []
    for i in range(JAC_PAIRS):
        v = [g.normal() for _ in range(STACK_WIDTH)]
        nv = math.sqrt(sum(a * a for a in v))
        pairs.append((Xt[i], [a / nv for a in v]))
    rows = []
    worst_all = 0.0
    for act, gain in (("tanh", 0.5), ("sigmoid", 1.0)):
        plain = build_stack("plain", act, gain)
        res = build_stack("residual", act, gain)
        mp = {k: 0.0 for k in RES_DEPTHS}
        mr = dict(mp)
        lo = dict(mp)
        up = dict(mp)
        for x, v in pairs:
            for k in RES_DEPTHS:
                mp[k] += math.log(backward_gain(plain, x, v, k)[0])
                nr, l, u, w = residual_backward(res, x, v, k)
                mr[k] += math.log(nr)
                lo[k] += math.log(l)
                up[k] += math.log(u)
                worst_all = max(worst_all, w)
        rows.append((f"{act}, gain {gain:g}", "plain: measured", *[_sci(math.exp(mp[k] / JAC_PAIRS), 2) for k in RES_DEPTHS]))
        rows.append(("", "residual: measured", *[_sci(math.exp(mr[k] / JAC_PAIRS), 2) for k in RES_DEPTHS]))
        rows.append(("", "residual: lower bound", *[_sci(math.exp(lo[k] / JAC_PAIRS), 2) for k in RES_DEPTHS]))
        rows.append(("", "residual: upper bound", *[_sci(math.exp(up[k] / JAC_PAIRS), 2) for k in RES_DEPTHS]))
    note = (f"The same {JAC_PAIRS} (example, vector) pairs and the same weights as the plain stack of the previous table; the residual stack adds "
            f"s = sqrt(1 / 48) = {math.sqrt(1 / 48):.4f} times each layer's output to its input. Entries: geometric means over the pairs of the norm "
            f"of the gradient after passing back through k layers, from a vector of norm 1, and the bounds prod(1 - s_i) and prod(1 + s_i) with s_i the "
            f"spectral norm of the branch Jacobian s D_i W_i (power iteration, 60 steps). The largest difference over all pairs and depths between the "
            f"backward pass and the formula d + J^T d, with J built explicitly, is {_sci(worst_all, 1)}.")
    return note + "\n\n" + _table(["configuration", "quantity"] + [f"k = {k}" for k in RES_DEPTHS], rows, "llrrrrrr")


def residual_scale_table():
    """Forward growth of the residual stream and backward growth of the gradient for several branch gains omega."""
    Xt, yt, _, _ = gallery_data()
    rows = []
    n = STACK_MATRICES
    for label, om in (("1", 1.0), ("1/4", 0.25), ("1/(2L) = 1/48", 1.0 / (2 * n))):
        layers = build_stack("residual", "tanh", 1.0, omega=om)
        X = Xt[:STACK_BATCH]
        h0, _ = forward(layers[:1], X)
        out, caches = forward(layers, X)
        h24, _ = forward(layers[:-1], X)
        q0 = sum(v * v for r in h0 for v in r) / (len(h0) * STACK_WIDTH)
        q24 = sum(v * v for r in h24 for v in r) / (len(h24) * STACK_WIDTH)
        loss, d = ce_stable(out, yt[:STACK_BATCH])
        _, arrive = backward(layers, caches, d)
        rows.append((label, _num((1 + om) ** n, 4), _num(q24 / q0, 4), _num(arrive[1] / arrive[n], 4), _num(loss, 3)))
    return ("Residual tanh stack, 24 layers, gain 1 start, branch s = sqrt(omega). Predicted growth (1 + omega)^24 of the second moment of "
            "the stream (chapter 9, upper bound for a branch that preserves the second moment); measured growth: mean square of the "
            "stream after layer 24 over its mean square after the input layer. Backward growth: norm of the gradient arriving at hidden layer 1's "
            "output over the one at layer 24's output.\n\n"
            + _table(["omega", "predicted growth (1 + omega)^24", "measured forward growth", "backward growth", "loss at step 0"], rows))


DEEP_CONFIGS = (
    ("plain", "tanh", 0.5, "plain, tanh, gain 0.5"),
    ("residual", "tanh", 0.5, "residual, tanh, gain 0.5"),
    ("plain", "relu", 1.0, "plain, ReLU, He"),
    ("residual", "relu", 1.0, "residual, ReLU, He"),
)


@lru_cache(maxsize=None)
def deep_runs():
    data = gallery_data()
    out = {}
    for kind, act, gain, label in DEEP_CONFIGS:
        layers = build_stack(kind, act, gain)
        out[label] = train(layers, data, eta=ETA, steps=300, eval_every=25)
    return out


def deep_training_table():
    rows = []
    for label, r in deep_runs().items():
        ev = {t: l for t, l, *_ in r["evals"]}
        rows.append((label, *[_num(ev[t], 3) for t in (0, 50, 100, 200, 300)], _num(r["evals"][-1][3], 3), _num(r["evals"][-1][4], 3)))
    return (f"24-matrix stacks of width {STACK_WIDTH}, rate {ETA}, batch {BATCH}, 300 steps, the gallery data and batch stream; the plain and the "
            f"residual stack of a pair have the same 24 matrices. Full-set training loss after t updates.\n\n"
            + _table(["stack", "step 0", "step 50", "step 100", "step 200", "step 300", "train acc", "val acc"], rows))


# ======================================================================= dead units

DEAD_STEPS = 200
DEAD_RUNS = (
    # label, activation, rate, bias start, weight scale (times the He scale)
    ("ReLU, rate 0.1", "relu", ETA, None, 1.0),
    ("ReLU, rate 2", "relu", 2.0, None, 1.0),
    ("leaky ReLU, rate 2", "leaky", 2.0, None, 1.0),
    ("ReLU, rate 0.1, hidden biases start at -2", "relu", ETA, {0: -2.0, 1: -2.0}, 1.0),
    ("ReLU, rate 0.1, weights 0.02 times He (the plateau start)", "relu", ETA, None, 0.02),
)


@lru_cache(maxsize=None)
def dead_runs():
    data = gallery_data()
    probe = data[0][:PROBE]
    out = {}
    for label, act, eta, bias, scale in DEAD_RUNS:
        layers = init_mlp([IN_DIM, HIDDEN, HIDDEN, CLASSES], act=act, bias=bias, scale=scale)
        out[label] = train(layers, data, eta=eta, steps=DEAD_STEPS, eval_every=10, probe=probe)
    return out


def dead_units_table():
    rows = []
    marks = (0, 10, 20, 50, 100, 200)
    for label, r in dead_runs().items():
        fr = dict(r["dead"])
        rows.append((label, *[_num(fr[m], 3) for m in marks], _num(r["evals"][-1][1], 3)))
    return (f"Fraction of the 64 hidden units that are not positive on any of the first {PROBE} training inputs, after t updates; the same start, data and "
            f"batches as the gallery, {DEAD_STEPS} steps. The leaky ReLU is x for x > 0 and {LEAK} x otherwise. Last column: full-set training loss after 200 "
            f"updates (the healthy run reaches {_num(gallery_run('healthy')['evals'][10][1], 3)} at step 200).\n\n"
            + _table(["run"] + [f"t = {m}" for m in marks] + ["loss at t = 200"], rows))


def dead_gradient_table():
    """For the rate-2 ReLU run after 200 updates: the gradient of the probe batch with respect to the incoming weights of
    dead and of alive units, and their biases."""
    r = dead_runs()["ReLU, rate 2"]
    layers = r["layers"]
    Xt, yt, _, _ = gallery_data()
    X, y = Xt[:PROBE], yt[:PROBE]
    logits, caches = forward(layers, X)
    loss, d = ce_stable(logits, y)
    backward(layers, caches, d)
    rows = []
    for li in (0, 1):
        L = layers[li]
        A, Z, Out = caches[li]
        dead = [j for j, col in enumerate(zip(*Z)) if all(z <= 0.0 for z in col)]
        alive = [j for j in range(HIDDEN) if j not in dead]
        gmax_dead = max((max(abs(x) for x in L["G"]["W"][j]) for j in dead), default=0.0)
        bmax_dead = max((abs(L["G"]["b"][j]) for j in dead), default=0.0)
        norms = sorted(math.sqrt(sum(x * x for x in L["G"]["W"][j])) for j in alive)
        mb = lambda idx: _num(sum(L["P"]["b"][j] for j in idx) / len(idx), 3) if idx else "-"   # noqa: E731
        rows.append((f"hidden layer {li + 1}", len(dead), len(alive), _sci(gmax_dead, 1), _sci(bmax_dead, 1), _sci(norms[len(norms) // 2], 2),
                     mb(dead), mb(alive)))
    return (f"Gradient of the loss on the {PROBE} probe examples, rate-2 ReLU run after {DEAD_STEPS} updates (loss {_num(loss, 3)}). Every bias started at 0.\n\n"
            + _table(["layer", "dead units", "alive units", "largest |dW| entry in a dead unit's row", "largest |db| of a dead unit",
                      "median row norm of an alive unit", "mean bias of dead units", "mean bias of alive units"], rows))


def dead_curves():
    """(label, [(t, fraction)]) for the figure: the first three runs."""
    return [(label, r["dead"]) for label, r in list(dead_runs().items())[:3]]


# ============================================================= overfit one batch

OVERFIT_N = 8
OVERFIT_STEPS = 300
OVERFIT_ETA = 0.1
CHECK_NAME = "training_dynamics.direction"
LABEL_NAME = "training_dynamics.random_labels"

BUG_ROWS = (
    # key, description, where the defect lives
    (None, "no defect", "-"),
    ("randlabels", "no defect, labels replaced by random ones", "data"),
    ("sign", "the update adds the gradient instead of subtracting it", "update"),
    ("labels", "the labels are reshuffled against the inputs at every step", "data"),
    ("batch_axis", "the softmax runs over the batch instead of over the classes", "loss"),
    ("nomask", "backward forgets the ReLU slope (passes the gradient of every unit)", "backward"),
    ("zero1", "the gradient of the first layer (weights and biases) is dropped", "backward"),
)


def _batch_loss_fn(bug):
    return ce_batch_axis if bug == "batch_axis" else ce_stable


def directional_check(layers, X, y, bug, h=1e-5):
    """Relative error between the directional derivative from backward and the central difference, along one random
    direction over all parameters (chapter 6), for a layer stack with the defect `bug` in its backward pass or loss."""
    lossfn = _batch_loss_fn(bug)
    logits, caches = forward(layers, X)
    _, d = lossfn(logits, y)
    backward(layers, caches, d, mask_bug=(bug == "nomask"))
    if bug == "zero1":
        for name, v in layers[0]["G"].items():
            for r in (v if isinstance(v[0], list) else [v]):
                for j in range(len(r)):
                    r[j] = 0.0
    grad = grad_vector(layers)
    g = bookrng.generator(CHECK_NAME)
    v = [g.normal() for _ in grad]
    nv = math.sqrt(sum(a * a for a in v))
    v = [a / nv for a in v]
    ana = dot(grad, v)

    def loss_at(sgn):
        M = copy_layers(layers)
        k = 0
        for L in M:
            for p in L["P"].values():
                for r in (p if isinstance(p[0], list) else [p]):
                    for j in range(len(r)):
                        r[j] += sgn * h * v[k]
                        k += 1
        return lossfn(forward(M, X)[0], y)[0]
    num = (loss_at(1.0) - loss_at(-1.0)) / (2.0 * h)
    return abs(ana - num) / max(abs(ana), abs(num), 1e-300)


def overfit_one_batch_table():
    Xt, yt, _, _ = gallery_data()
    X = Xt[:OVERFIT_N]
    gl = bookrng.generator(LABEL_NAME)
    rand_y = [gl.integer(CLASSES) for _ in range(OVERFIT_N)]
    rows = []
    for bug, desc, where in BUG_ROWS:
        y = rand_y if bug == "randlabels" else yt[:OVERFIT_N]
        layers = init_mlp([IN_DIM, HIDDEN, HIDDEN, CLASSES])
        err = directional_check(copy_layers(layers), X, y, None if bug == "randlabels" else bug)
        r = train(layers, (X + [], y + [], X, y), eta=OVERFIT_ETA, batch=None, steps=OVERFIT_STEPS, n_pool=OVERFIT_N,
                  loss="batch_axis" if bug == "batch_axis" else "stable", bug=None if bug in ("randlabels", "batch_axis") else bug,
                  eval_every=OVERFIT_STEPS)
        l0 = r["evals"][0][1]
        l1, a1 = r["evals"][-1][1], r["evals"][-1][3]
        # for the batch-axis loss the reported loss is the usual one on the same weights (evaluate uses the correct loss)
        gc = "fails" if err > 1e-4 else "passes"
        of = "passes" if (l1 < 0.01 * l0 and a1 == 1.0) else "fails"
        rows.append((desc, where, "below 1e-8" if err < 1e-8 else _sci(err, 1), gc, _num(l0, 3), _num(l1, 4), _num(a1, 3), of))
    return (f"The first {OVERFIT_N} training examples as one batch (or random labels), network 8 -> 32 -> 32 -> 3 from the gallery start, {OVERFIT_STEPS} steps "
            f"of plain gradient descent at rate {OVERFIT_ETA} on the whole batch. Gradient check: one random unit direction over all {n_params(init_mlp([IN_DIM, HIDDEN, HIDDEN, CLASSES]))} "
            f"parameters (normal() draws, stream {bookrng.seed_of(CHECK_NAME):#018x}), central difference with h = 1e-5, at the start; 'fails' when the relative "
            f"error exceeds 1e-4. Random labels: {OVERFIT_N} draws of integer(3) from stream {bookrng.seed_of(LABEL_NAME):#018x}. "
            "Loss and accuracy are the usual (correct) cross-entropy and argmax accuracy on the batch after the last update; overfit one batch passes "
            "when that loss is below 1% of the loss at step 0 and the accuracy is 1. A relative error under 1e-8 is rounding noise of the central difference and is printed as a bound. "
            f"'The labels are reshuffled at every step': at each step the 8 labels, in their original order, are shuffled with the book's shuffle by the generator of the gallery's batch stream (seed {bookrng.seed_of(BATCH_NAME):#018x}), "
            "which continues from step to step. 'The gradient of the first layer is dropped': the whole gradient of layer 1, weights and bias vector, is set to zero before the update.\n\n"
            + _table(["defect", "lives in", "gradient check: relative error", "gradient check", "loss at step 0", "loss after 300 steps", "accuracy",
                      "overfit one batch"], rows, "llrlrrrl"))


# ================================================================ what monitoring costs

def _macs(dims):
    return sum(a * b for a, b in zip(dims[:-1], dims[1:]))


def step_flops(dims, batch):
    """FLOPs of one training step: forward 2 MACs, backward dW 2 MACs and dA 2 MACs (not for the first layer), per example.
    Biases, activations and the loss are not counted (a few hundred per example, under 5% of the total)."""
    m = _macs(dims)
    return batch * (2 * m + 2 * m + 2 * (m - dims[0] * dims[1]))


def monitoring_cost_table():
    dims = [IN_DIM, HIDDEN, HIDDEN, CLASSES]
    P = 1443
    U = 2 * HIDDEN
    step = step_flops(dims, BATCH)
    items = [
        ("global gradient norm (and each layer's share of it)", "P multiplies and P - 1 adds", 2 * P),
        ("update ratio", "the norm of the weights: P multiplies and adds (the update norm is the rate times the gradient norm)", 2 * P),
        ("activation rms of each hidden layer", f"B * U squares and adds, U = {U} hidden units", 2 * BATCH * U),
        ("fraction of dead units on the minibatch", "B * U comparisons", BATCH * U),
        ("finiteness scan of the gradient", "P comparisons", P),
    ]
    rows = [(a, b, f"{c:,}", _num(100.0 * c / step, 2)) for a, b, c in items]
    total = sum(c for _, _, c in items)
    rows.append(("all five", "", f"{total:,}", _num(100.0 * total / step, 2)))
    rows.append(("all five, every 10th step", "", f"{total // 10:,}", _num(100.0 * total / 10.0 / step, 2)))
    return (f"Network {dims}, P = {P} parameters, batch B = {BATCH}. One training step costs {step:,} FLOPs: per example 2M for the forward pass, 2M for the "
            f"weight gradients and 2(M - {IN_DIM * HIDDEN}) for the gradients passed down, with M = {_macs(dims)} multiply-adds. Every monitor reads arrays "
            "the step has already produced.\n\n"
            + _table(["monitor", "operations", "FLOPs per step", "% of a training step"], rows, "llrr"))


@lru_cache(maxsize=None)
def baseline_run():
    """A linear softmax model (8 -> 3) on the gallery data: the cheapest model that is not the prior."""
    return train(init_mlp([IN_DIM, CLASSES]), gallery_data(), eta=ETA, steps=STEPS)


def checklist_table():
    dims = [IN_DIM, HIDDEN, HIDDEN, CLASSES]
    m = _macs(dims)
    P = 1443
    full = STEPS * step_flops(dims, BATCH)
    _, yt, _, yv = gallery_data()
    maj_acc = max(yv.count(k) for k in range(CLASSES)) / len(yv)
    prior_val_loss = -sum(math.log(yt.count(c) / len(yt)) for c in yv) / len(yv)
    base = baseline_run()
    monitors = STEPS * (2 * P + 2 * P + 3 * BATCH * 2 * HIDDEN + P)

    def fwd(b):
        return 2 * m * b
    items = [
        ("1. shapes, types, finiteness of the inputs", "every array has the expected shape and no NaN enters", 0, ""),
        ("2. initial loss against ln C", f"one forward pass of {BATCH} examples", fwd(BATCH), ""),
        ("3. the label prior", "counting the labels", 0, f"validation loss {_num(prior_val_loss, 4)}, validation accuracy {_num(maj_acc, 3)}"),
        ("4. a gradient check", "one backward pass of 8 examples and 4 directions of 2 forward passes",
         step_flops(dims, 8) + 4 * 2 * fwd(8), ""),
        ("5. a linear baseline", f"a linear softmax model, {STEPS} steps of {BATCH} examples", STEPS * step_flops([IN_DIM, CLASSES], BATCH),
         f"validation loss {_num(base['evals'][-1][2], 4)}, validation accuracy {_num(base['evals'][-1][4], 3)}"),
        ("6. monitoring during the run", "five monitors at every step (previous table)", monitors, ""),
        ("7. overfit one batch", f"{OVERFIT_STEPS} steps of 8 examples", OVERFIT_STEPS * step_flops(dims, 8), ""),
        ("8. a rate sweep", f"{len(SWEEP_RATES)} rates of 100 steps of {BATCH} examples", len(SWEEP_RATES) * 100 * step_flops(dims, BATCH), ""),
        ("the run itself", f"{STEPS} steps of {BATCH} examples", full, ""),
    ]
    rows = [(a, b, f"{c:,}", _num(100.0 * c / full, 2), n) for a, b, c, n in items]
    return (f"Network {dims}. Cost in FLOPs (counted as in the previous table) and as a share of the {STEPS}-step run. The monitoring row is spread over "
            f"the run, the others are paid before or beside it.\n\n"
            + _table(["check", "what it costs", "FLOPs", "% of the run", "what it tells"], rows, "llrrl"))


# ====================================================== a data bug that survives it

def misalignment_table():
    """Labels permanently shuffled against the inputs: overfit one batch still passes, a baseline does not."""
    Xt, yt, Xv, yv = gallery_data()
    g = bookrng.generator("training_dynamics.misalignment")
    perm = g.shuffle(list(yt))
    rows = []
    for label, labels in (("correct labels", yt), ("labels shuffled once, before training", perm)):
        layers = init_mlp([IN_DIM, HIDDEN, HIDDEN, CLASSES])
        X8, y8 = Xt[:OVERFIT_N], labels[:OVERFIT_N]
        r = train(layers, (X8, y8, X8, y8), eta=OVERFIT_ETA, batch=None, steps=OVERFIT_STEPS, n_pool=OVERFIT_N, eval_every=OVERFIT_STEPS)
        base = train(init_mlp([IN_DIM, CLASSES]), (Xt, labels, Xv, yv), eta=ETA, steps=STEPS, eval_every=STEPS)
        rows.append((label, _num(r["evals"][-1][1], 4), _num(r["evals"][-1][3], 3), _num(base["evals"][-1][3], 3), _num(base["evals"][-1][4], 3)))
    return (f"Shuffle: Fisher-Yates of the {N_TRAIN} training labels with the stream {bookrng.seed_of('training_dynamics.misalignment'):#018x} "
            f"(validation labels untouched). Overfit one batch: the first {OVERFIT_N} examples, {OVERFIT_STEPS} steps at rate {OVERFIT_ETA}. Baseline: the "
            f"linear softmax model of the checklist table, {STEPS} steps, trained on all {N_TRAIN} (possibly shuffled) labels and scored against the true validation labels.\n\n"
            + _table(["training labels", "overfit one batch: loss", "overfit one batch: accuracy", "linear baseline: training accuracy",
                      "linear baseline: validation accuracy"], rows))


# ============================================================ two small accounting tables

def log_scale_table():
    rows = []
    for key, label in (("healthy", "healthy"), ("slow", "too slow"), ("noisy", "noisy")):
        sl = log_slope(gallery_run(key)["evals"])
        rows.append((label, _num(sl, 3), _num(100.0 / -sl, 0)))
    sl = log_slope(overfit_run()["evals"], start=40)
    rows.append(("overfitting (training loss, steps 40 to 600)", _num(sl, 3), _num(100.0 / -sl, 0)))
    return ("Slope: least-squares slope of log10 of the full-set training loss against the step, per 100 steps, over the evaluations at step >= 100 "
            "(at step >= 40 for the overfitting run). Steps per decade: 100 divided by minus the slope, the steps the loss needs to fall by a factor of 10 "
            "if the slope persisted.\n\n" + _table(["run", "slope per 100 steps", "steps per decade"], rows))


def evaluation_cost_table():
    dims = [IN_DIM, HIDDEN, HIDDEN, CLASSES]
    m = _macs(dims)
    step = step_flops(dims, BATCH)
    full = (N_TRAIN + N_VAL) * 2 * m
    rows = []
    for k in (1, 10, 20, 100):
        rows.append((k, f"{full:,}", f"{k * step:,}", _num(100.0 * full / (k * step), 1)))
    return (f"One evaluation of both full sets is a forward pass over {N_TRAIN} + {N_VAL} examples: {full:,} FLOPs (2 FLOPs per multiply-add, "
            f"M = {m} multiply-adds per example). One training step costs {step:,} FLOPs.\n\n"
            + _table(["evaluate every k steps, k =", "FLOPs per evaluation", "FLOPs of k training steps", "evaluation as % of training"], rows))
