# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "First Complete Models".

Plain Python 3 on floats (IEEE double precision), written from the chapter's own
description and independent of any engine code. Each public function returns a
markdown string that the chapter embeds between `<!-- worked: first_models.<name> -->`
markers; the same module supplies the data of the chapter's figures.

Two complete builds carry the chapter.

  the autoencoder   16 sensor readings -> 12 -> 3 (the bottleneck) -> 12 -> 16, GELU after the
                    two hidden layers, trained to reproduce its input under the mean squared error
  the controller    (error, previous error, previous command) -> 16 -> 16 -> command, GELU, trained to
                    imitate a pole-placed velocity-form PI controller on a first-order plant

Everything random comes from the book's SplitMix64 (`bookrng.generator(name)`); every
stream name is printed by the chapter. Parameters are one flat vector, layer by layer, each
layer a weight matrix of shape [n_out, n_in] in row-major order followed by its bias; the
gradient has the same layout. The training loop is the one the chapter writes in pseudocode:
shuffle, batch, forward, loss, backward, global-norm clip, AdamW, schedule, log, evaluate,
checkpoint.

Counting rule used for work: one multiplication, addition, subtraction, division or square
root is one FLOP; the work of a network is that of its matrix products plus the optimizer's
elementwise arithmetic (16 FLOPs per parameter for AdamW as in chapter 7, 3 for the clip), and
all arrays are counted in 32-bit floats (4 bytes) even though this module computes in doubles.

Swedish Embedded AB implements solutions for building, training and verifying compact neural
networks end to end. If your team needs expertise in learned compression and learned control
for embedded systems, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import hashlib
import math
import struct
from functools import lru_cache
from operator import mul

import bookrng

# ------------------------------------------------------------------ constants

D = 16                                  # sensors
POSITIONS = [j / (D - 1) for j in range(D)]
PATTERNS = [[math.cos(math.pi * p) for p in POSITIONS],
            [math.sin(2.0 * math.pi * p) for p in POSITIONS],
            [math.cos(2.0 * math.pi * p) for p in POSITIONS]]
SCALES = (1.0, 0.5, 0.25)               # how strongly each cause drives the sensors
SENSOR_NOISE = 0.05
N_TRAIN, N_VAL, N_TEST = 256, 128, 128

AE_SPEC = [(16, 12, "gelu"), (12, 3, None), (3, 12, "gelu"), (12, 16, None)]
AE_RUN = dict(epochs=300, batch=32, peak=1e-2, tau=1.0, lam=0.01)
LIN_RUN = dict(epochs=400, batch=32, peak=3e-2, tau=1.0, lam=0.0)
DAE_SIGMA = 0.3                         # corruption of the denoising variant

CURVE_SPEC = [(16, 12, "gelu"), (12, 1, None), (1, 12, "gelu"), (12, 16, None)]
CURVE_QUAD = 1.5                        # weight of the s^2 pattern of the curved data
CURVE_RUN = dict(epochs=300, batch=32, peak=1e-2, tau=1.0, lam=0.0)

# the plant and the expert of the controller (the numbers of crates/toypid)
DT = 0.05
WN, ZETA = 3.5, 1.0
NOMINAL = (0.65, 1.25)                  # (tau, gain)
HELD_OUT = [(0.55, 1.125), (0.55, 1.375), (0.75, 1.125), (0.75, 1.375)]
HOLDS = [(0, 0.70), (45, -0.60), (90, 0.40), (135, -0.85)]
ROLLOUT_STEPS = 180
CTRL_SPEC = [(3, 16, "gelu"), (16, 16, "gelu"), (16, 1, None)]
CTRL_RUN = dict(epochs=100, batch=72, peak=1e-2, tau=1.0, lam=0.01)
CTRL_TRAJ, CTRL_LEN, CTRL_VAL_TRAJ = 24, 90, 8

BATCH_STREAM = "first_models.batches"
GATE_H, GATE_ATOL, GATE_RTOL = 1e-5, 1e-8, 1e-5

SQRT2 = math.sqrt(2.0)
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


# ------------------------------------------------------------------ formatting

def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _fx(v, d=4):
    out = f"{v:.{d}f}"
    return out[1:] if out.startswith("-") and not any(c in "123456789" for c in out) else out


def _sci(v, d=2):
    return f"{v:.{d}e}"


def _int(v):
    return f"{v:,}"


def _seed(name):
    s = bookrng.seed_of(name)
    return f"`{name}` (seed 0x{s:016x})"


# ------------------------------------------------------------------ the network

def gelu(z):
    return 0.5 * z * (1.0 + math.erf(z / SQRT2))


def gelu_slope(z):
    return 0.5 * (1.0 + math.erf(z / SQRT2)) + z * INV_SQRT_2PI * math.exp(-0.5 * z * z)


def n_params(spec):
    return sum(o * i + o for i, o, _ in spec)


def macs(spec):
    return sum(o * i for i, o, _ in spec)


def init_theta(spec, gen):
    """Weights N(0, 1/n_in) layer by layer in row-major order, biases zero. Draws: the weights only."""
    theta = []
    for n_in, n_out, _ in spec:
        s = 1.0 / math.sqrt(n_in)
        theta.extend(gen.normal() * s for _ in range(n_out * n_in))
        theta.extend([0.0] * n_out)
    return theta


def forward(spec, theta, X):
    A, tape, off = X, [], 0
    for n_in, n_out, act in spec:
        W = [theta[off + j * n_in: off + (j + 1) * n_in] for j in range(n_out)]
        off += n_out * n_in
        b = theta[off: off + n_out]
        off += n_out
        Z = [[sum(map(mul, w, a)) + bj for w, bj in zip(W, b)] for a in A]
        Y = [[gelu(z) for z in row] for row in Z] if act == "gelu" else Z
        tape.append((A, Z, W))
        A = Y
    return A, tape


def mse(Y, T):
    s, n = 0.0, 0
    for y, t in zip(Y, T):
        for a, b in zip(y, t):
            s += (a - b) ** 2
            n += 1
    return s / n


def backward(spec, tape, Y, T):
    B, O = len(Y), len(Y[0])
    dY = [[2.0 * (a - b) / (B * O) for a, b in zip(y, t)] for y, t in zip(Y, T)]
    grads = []
    for (n_in, n_out, act), (A_in, Z, W) in zip(reversed(spec), reversed(tape)):
        dZ = [[d * gelu_slope(z) for d, z in zip(dr, zr)] for dr, zr in zip(dY, Z)] if act == "gelu" else dY
        dZt = list(zip(*dZ))
        A_t = list(zip(*A_in))
        dW = [sum(map(mul, dzj, ai)) for dzj in dZt for ai in A_t]
        db = [sum(c) for c in dZt]
        grads.append(dW + db)
        if len(grads) < len(spec):
            Wt = list(zip(*W))
            dY = [[sum(map(mul, dz, wc)) for wc in Wt] for dz in dZ]
    out = []
    for g in reversed(grads):
        out.extend(g)
    return out


def loss_grad(spec, theta, X, T):
    Y, tape = forward(spec, theta, X)
    return mse(Y, T), backward(spec, tape, Y, T)


def loss_only(spec, theta, X, T):
    return mse(forward(spec, theta, X)[0], T)


# ------------------------------------------------------------------ optimizer, clip, schedule

def schedule_rate(t, peak, floor, warmup, total):
    if t < warmup:
        return peak * (t + 1) / warmup
    u = min(1.0, (t - warmup) / (total - warmup))
    return floor + (peak - floor) * 0.5 * (1.0 + math.cos(math.pi * u))


def global_norm(g):
    return math.sqrt(sum(a * a for a in g))


def adamw_step(theta, g, m, v, t, eta, lam, beta1=0.9, beta2=0.999, eps=1e-8):
    bc1, bc2 = 1.0 - beta1 ** t, 1.0 - beta2 ** t
    for i in range(len(theta)):
        gi = g[i]
        m[i] = beta1 * m[i] + (1.0 - beta1) * gi
        v[i] = beta2 * v[i] + (1.0 - beta2) * gi * gi
        theta[i] -= eta * lam * theta[i]
        theta[i] -= eta * (m[i] / bc1) / (math.sqrt(v[i] / bc2) + eps)


# ------------------------------------------------------------------ the end-to-end training loop

def fit(spec, X, T, Xv, Tv, run, init_name, corrupt=None, on_epoch=None, steps_override=None):
    """The chapter's loop. Returns a dict with the final and the best-validation parameters and the log.

    corrupt(rows_of_X, gen) -> inputs to feed (denoising); the target stays the clean row.
    on_epoch(epoch, theta) -> dict of extra columns for the log."""
    theta = init_theta(spec, bookrng.generator(init_name))
    P = len(theta)
    m, v = [0.0] * P, [0.0] * P
    shuf = bookrng.generator(BATCH_STREAM)
    noise = bookrng.generator(corrupt.stream) if corrupt else None
    B, E = run["batch"], run["epochs"]
    assert len(X) % B == 0
    per_epoch = len(X) // B
    total = E * per_epoch
    warmup = total // 10
    floor = run["peak"] / 100.0
    order = list(range(len(X)))
    log, t = [], 0
    best_val, best_theta, best_epoch = loss_only(spec, theta, Xv, Tv), list(theta), 0
    start = {"train": loss_only(spec, theta, X, T), "val": best_val}
    for epoch in range(1, E + 1):
        shuf.shuffle(order)
        norms, clipped = [], 0
        for k in range(per_epoch):
            rows = order[k * B:(k + 1) * B]
            xb, tb = [X[i] for i in rows], [T[i] for i in rows]
            fed = corrupt(xb, noise) if corrupt else xb
            loss, g = loss_grad(spec, theta, fed, tb)
            if not math.isfinite(loss):
                raise ValueError("non-finite loss")
            norm = global_norm(g)
            c = min(1.0, run["tau"] / (norm + 1e-6))
            clipped += c < 1.0
            g = [c * a for a in g]
            adamw_step(theta, g, m, v, t + 1, schedule_rate(t, run["peak"], floor, warmup, total), run["lam"])
            norms.append(norm)
            t += 1
        row = {"epoch": epoch, "train": loss_only(spec, theta, X, T), "val": loss_only(spec, theta, Xv, Tv),
               "norm": sum(norms) / len(norms), "clipped": clipped,
               "lr": schedule_rate(t - 1, run["peak"], floor, warmup, total)}
        if on_epoch:
            row.update(on_epoch(epoch, theta))
        log.append(row)
        if row["val"] < best_val:
            best_val, best_theta, best_epoch = row["val"], list(theta), epoch
    return {"theta": theta, "best": best_theta, "best_epoch": best_epoch, "log": log, "start": start,
            "steps": t, "warmup": warmup, "floor": floor, "P": P}


def digest(theta):
    return hashlib.sha256(struct.pack(f"<{len(theta)}d", *theta)).hexdigest()


# ------------------------------------------------------------------ the sensor data

@lru_cache(maxsize=None)
def sensor_data():
    """512 readings: per example three N(0,1) causes, then 16 N(0,1) noise values (scaled by SENSOR_NOISE)."""
    gen = bookrng.generator("first_models.ae_data")
    X, S = [], []
    for _ in range(N_TRAIN + N_VAL + N_TEST):
        s = [gen.normal() for _ in range(3)]
        noise = [gen.normal() for _ in range(D)]
        X.append([sum(SCALES[k] * s[k] * PATTERNS[k][j] for k in range(3)) + SENSOR_NOISE * noise[j] for j in range(D)])
        S.append(s)
    return X[:N_TRAIN], X[N_TRAIN:N_TRAIN + N_VAL], X[N_TRAIN + N_VAL:], S


def mean_vec(X):
    return [sum(x[j] for x in X) / len(X) for j in range(len(X[0]))]


def mean_predictor_loss(X, ref):
    mu = mean_vec(ref)
    return mse([mu] * len(X), X)


@lru_cache(maxsize=None)
def ae_model():
    tr, va, te, _ = sensor_data()
    return fit(AE_SPEC, tr, tr, va, va, AE_RUN, "first_models.ae_init")


def ae_model_uncached():
    tr, va, te, _ = sensor_data()
    return fit(AE_SPEC, tr, tr, va, va, AE_RUN, "first_models.ae_init")


# ------------------------------------------------------------------ PCA (Jacobi eigenvalues of the covariance)

def jacobi_eigh(A, sweeps=60):
    n = len(A)
    A = [r[:] for r in A]
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(sweeps):
        if sum(A[i][j] ** 2 for i in range(n) for j in range(n) if i != j) < 1e-24:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(A[p][q]) < 1e-300:
                    continue
                th = (A[q][q] - A[p][p]) / (2.0 * A[p][q])
                t = (1.0 if th >= 0 else -1.0) / (abs(th) + math.sqrt(th * th + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    akp, akq = A[k][p], A[k][q]
                    A[k][p], A[k][q] = c * akp - s * akq, s * akp + c * akq
                for k in range(n):
                    apk, aqk = A[p][k], A[q][k]
                    A[p][k], A[q][k] = c * apk - s * aqk, s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = V[k][p], V[k][q]
                    V[k][p], V[k][q] = c * vkp - s * vkq, s * vkp + c * vkq
    ev = [A[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda i: -ev[i])
    vecs = []
    for i in order:
        col = [V[k][i] for k in range(n)]
        big = max(range(n), key=lambda k: abs(col[k]))
        vecs.append([c * (1.0 if col[big] >= 0 else -1.0) for c in col])   # sign: largest entry positive
    return [ev[i] for i in order], vecs


@lru_cache(maxsize=None)
def pca_fit():
    tr = sensor_data()[0]
    mu = mean_vec(tr)
    d = len(mu)
    C = [[sum((x[i] - mu[i]) * (x[j] - mu[j]) for x in tr) / len(tr) for j in range(d)] for i in range(d)]
    ev, vecs = jacobi_eigh(C)
    return mu, ev, vecs


def pca_reconstruct(x, k):
    mu, _, vecs = pca_fit()
    c = [a - b for a, b in zip(x, mu)]
    r = list(mu)
    for u in vecs[:k]:
        s = sum(a * b for a, b in zip(u, c))
        r = [ri + s * ui for ri, ui in zip(r, u)]
    return r


def pca_error(X, k):
    return mse([pca_reconstruct(x, k) for x in X], X)


# ------------------------------------------------------------------ gradient check

def gradient_check(spec, theta, X, T):
    """Central differences on every entry; returns (worst gate ratio, worst relative error, failing, per-tensor worst rel)."""
    _, g = loss_grad(spec, theta, X, T)
    bounds, off = [], 0
    for layer, (i, o, _) in enumerate(spec, 1):
        bounds.append((f"W{layer}", off, off + o * i))
        bounds.append((f"b{layer}", off + o * i, off + o * i + o))
        off += o * i + o
    worst_ratio, worst_rel, failing = 0.0, 0.0, 0
    per = {}
    th = list(theta)
    for name, lo, hi in bounds:
        w = 0.0
        for i in range(lo, hi):
            old = th[i]
            th[i] = old + GATE_H
            up = loss_only(spec, th, X, T)
            th[i] = old - GATE_H
            down = loss_only(spec, th, X, T)
            th[i] = old
            num = (up - down) / (2.0 * GATE_H)
            diff = abs(g[i] - num)
            scale = max(abs(g[i]), abs(num))
            ratio = diff / (GATE_ATOL + GATE_RTOL * scale)
            worst_ratio = max(worst_ratio, ratio)
            failing += ratio > 1.0
            if scale > 1e-4:
                w = max(w, diff / scale)
        per[name] = w
        worst_rel = max(worst_rel, w)
    return worst_ratio, worst_rel, failing, per


def overfit_one_batch(spec, theta0, X, T, steps=400, eta=1e-2):
    theta = list(theta0)
    m, v = [0.0] * len(theta), [0.0] * len(theta)
    first = loss_only(spec, theta, X, T)
    for t in range(steps):
        _, g = loss_grad(spec, theta, X, T)
        adamw_step(theta, g, m, v, t + 1, eta, 0.0)
    return first, loss_only(spec, theta, X, T)


# ------------------------------------------------------------------ work counts

def step_flops(spec, B):
    M = macs(spec)
    first = spec[0][0] * spec[0][1]
    fwd, wgrad, dgrad = 2 * M * B, 2 * M * B, 2 * (M - first) * B
    P = n_params(spec)
    return fwd, wgrad, dgrad, 3 * P, 16 * P


def saved_floats(spec, B):
    """Floats the backward pass reads from the forward: each layer's input and each GELU layer's pre-activation."""
    return B * sum(i + (o if a == "gelu" else 0) for i, o, a in spec)


# =================================================================== autoencoder: worked blocks

def ae_data():
    tr, va, te, S = sensor_data()
    rows = []
    for j in range(D):
        rows.append([str(j), _fx(POSITIONS[j], 4)] + [_fx(PATTERNS[k][j], 4) for k in range(3)] + [_fx(tr[0][j], 4), _fx(tr[1][j], 4)])
    mu = mean_vec(tr)
    var = sum(sum((x[j] - mu[j]) ** 2 for j in range(D)) for x in tr) / (len(tr) * D)
    head = (f"Sensor j sits at position pos_j = j/15. The patterns are pat1 = cos(pi pos), pat2 = sin(2 pi pos), pat3 = cos(2 pi pos). A reading is "
            f"x_j = 1.0 s1 pat1_j + 0.5 s2 pat2_j + 0.25 s3 pat3_j + 0.05 n_j with s1, s2, s3 and n_j drawn from N(0, 1). Generator: {_seed('first_models.ae_data')}. "
            f"Draw order, per example: s1, s2, s3, then n_0 .. n_15 (19 normals); 512 examples in all, split in order into "
            f"{N_TRAIN} training, {N_VAL} validation and {N_TEST} test examples. The last two columns are training examples 0 and 1 "
            f"(causes s = ({_fx(S[0][0])}, {_fx(S[0][1])}, {_fx(S[0][2])}) and ({_fx(S[1][0])}, {_fx(S[1][1])}, {_fx(S[1][2])})). "
            f"Mean of the squared readings about the training mean (per sensor): {_fx(var, 4)}.\n\n")
    return head + _table(["j", "pos_j", "pat1", "pat2", "pat3", "x (example 0)", "x (example 1)"], rows)


def ae_architecture():
    names = ["encoder", "bottleneck", "decoder", "output"]
    rows, tot_p, tot_m = [], 0, 0
    for name, (i, o, a) in zip(names, AE_SPEC):
        rows.append([name, f"{o} x {i}", str(o), str(o * i + o), str(o * i), a or "none"])
        tot_p += o * i + o
        tot_m += o * i
    rows.append(["total", "", "", str(tot_p), str(tot_m), ""])
    head = ("Layer shapes are [n_out, n_in] (weight matrix of `x @ W^T`), activations GELU (exact erf form) after the encoder and the decoder. "
            f"Initial weights N(0, 1/n_in), biases 0, stream {_seed('first_models.ae_init')}: the weights of layers 1 to 4 in order, row-major.\n\n")
    return head + _table(["layer", "W shape", "bias", "parameters", "multiply-adds per example", "activation"], rows)


def ae_prechecks():
    tr, va, te, _ = sensor_data()
    run = ae_model()
    theta0 = init_theta(AE_SPEC, bookrng.generator("first_models.ae_init"))
    X8 = tr[:8]
    ratio, rel, failing, per = gradient_check(AE_SPEC, theta0, X8, X8)
    first, last = overfit_one_batch(AE_SPEC, theta0, X8, X8)
    base_tr, base_va = mean_predictor_loss(tr, tr), mean_predictor_loss(va, tr)
    finite = all(math.isfinite(a) for x in tr + va + te for a in x)
    rows = [
        ["1. shapes and finiteness", f"inputs {len(tr)} x {D}, targets the same; every value finite: {finite}", ""],
        ["2. initial loss", f"training {_fx(run['start']['train'], 4)}, validation {_fx(run['start']['val'], 4)}", f"the mean predictor scores {_fx(base_tr, 4)} (training), {_fx(base_va, 4)} (validation)"],
        ["3. gradient check, all 499 entries", f"worst gate ratio abs(ana - num) / (atol + rtol max(abs(ana), abs(num))) = {_fx(ratio, 4)}, failing {failing}", "gate: h = 1e-5, atol = 1e-8, rtol = 1e-5"],
        ["   worst relative error per tensor", ", ".join(f"{k} {_sci(w, 1)}" for k, w in per.items()), "entries with max(abs(ana), abs(num)) above 1e-4"],
        ["4. overfit one batch (8 examples, 400 steps, rate 0.01)", f"loss {_fx(first, 4)} -> {_sci(last, 2)}", "no clip, no decay"],
    ]
    return _table(["check", "result", "reference"], rows, "lll")


def ae_training_log():
    run = ae_model()
    shown = [1, 2, 5, 10, 20, 40, 60, 80, 100, 150, 200, 250, 300]
    rows = [["0", _fx(run["start"]["train"], 5), _fx(run["start"]["val"], 5), "", "", ""]]
    for r in run["log"]:
        if r["epoch"] in shown:
            rows.append([str(r["epoch"]), _fx(r["train"], 5), _fx(r["val"], 5), _fx(r["norm"], 4), str(r["clipped"]), _sci(r["lr"], 2)])
    tr, va, te, _ = sensor_data()
    best = run["best"]
    head = (f"{AE_RUN['epochs']} epochs of {N_TRAIN // AE_RUN['batch']} steps at batch {AE_RUN['batch']} ({run['steps']} steps). AdamW (0.9, 0.999, 1e-8), decay {AE_RUN['lam']}, clip {AE_RUN['tau']}, "
            f"warmup {run['warmup']} steps then cosine from {AE_RUN['peak']} to {_sci(run['floor'], 1)}. Shuffle stream {_seed(BATCH_STREAM)}, one shuffle of 0 .. 255 per epoch. "
            f"Loss columns are the mean squared error over the whole training and validation sets after the epoch; 'grad norm' is the mean over the epoch's steps of the global norm before the clip; "
            f"'clipped' counts the steps of the epoch whose norm exceeded the threshold; 'rate' is the rate of the epoch's last step. "
            f"The best validation loss {_fx(min(r['val'] for r in run['log']), 5)} is at epoch {run['best_epoch']}; the checkpoint kept is that epoch's. Test loss of the checkpoint: {_fx(loss_only(AE_SPEC, best, te, te), 5)}.\n\n")
    return head + _table(["epoch", "training loss", "validation loss", "grad norm", "clipped", "rate"], rows)


def ae_reconstruction():
    tr, va, te, _ = sensor_data()
    run = ae_model()
    th = run["best"]
    Y, tape = forward(AE_SPEC, th, te[:1])
    z = None
    # the code: output of the second layer
    A1 = [[gelu(sum(map(mul, th[j * 16:(j + 1) * 16], te[0])) + th[192 + j]) for j in range(12)]]
    off = 204
    z = [sum(map(mul, th[off + j * 12: off + (j + 1) * 12], A1[0])) + th[off + 36 + j] for j in range(3)]
    p3 = pca_reconstruct(te[0], 3)
    rows = [[str(j), _fx(te[0][j], 4), _fx(Y[0][j], 4), _fx(Y[0][j] - te[0][j], 4), _fx(p3[j], 4)] for j in range(D)]
    sq = sum((a - b) ** 2 for a, b in zip(Y[0], te[0])) / D
    summary = [
        ["mean predictor (training means)", _fx(mean_predictor_loss(tr, tr), 5), _fx(mean_predictor_loss(va, tr), 5), _fx(mean_predictor_loss(te, tr), 5)],
        ["PCA with 3 components", _fx(pca_error(tr, 3), 5), _fx(pca_error(va, 3), 5), _fx(pca_error(te, 3), 5)],
        ["autoencoder, bottleneck 3 (checkpoint)", _fx(loss_only(AE_SPEC, th, tr, tr), 5), _fx(loss_only(AE_SPEC, th, va, va), 5), _fx(loss_only(AE_SPEC, th, te, te), 5)],
        ["noise floor: 13/16 of the sensor noise variance", _fx(13 / 16 * SENSOR_NOISE ** 2, 5), "", ""],
    ]
    head = (f"Test example 0 through the checkpoint. The code is z = ({_fx(z[0], 4)}, {_fx(z[1], 4)}, {_fx(z[2], 4)}): three numbers stand for sixteen. "
            f"Mean squared error of this one example: {_fx(sq, 5)}.\n\n")
    return head + _table(["sensor j", "reading x_j", "reconstruction", "difference", "PCA, 3 components"], rows) + "\n\n" + \
        "Mean squared error over each whole set:\n\n" + _table(["method", "training", "validation", "test"], summary)


# ------------------------------------------------------------------ PCA and the linear autoencoder

@lru_cache(maxsize=None)
def linear_model(k):
    tr, va, te, _ = sensor_data()
    return fit([(16, k, None), (k, 16, None)], tr, tr, va, va, LIN_RUN, f"first_models.lin_init.{k}")


def linear_spec(k):
    return [(16, k, None), (k, 16, None)]


def _orthonormal_columns(cols):
    q = []
    for c in cols:
        v = list(c)
        for u in q:
            s = sum(a * b for a, b in zip(v, u))
            v = [a - s * b for a, b in zip(v, u)]
        n = math.sqrt(sum(a * a for a in v))
        q.append([a / n for a in v])
    return q


def pca_spectrum():
    mu, ev, vecs = pca_fit()
    tr, va, te, _ = sensor_data()
    total = sum(ev)
    rows = []
    for k in range(0, 8):
        err_tr = sum(ev[k:]) / D
        rows.append([str(k), _fx(ev[k - 1], 5) if k else "", _fx(100 * sum(ev[:k]) / total, 3), _fx(err_tr, 5), _fx(pca_error(tr, k) if k else mean_predictor_loss(tr, tr), 5), _fx(pca_error(te, k) if k else mean_predictor_loss(te, tr), 5)])
    head = ("Covariance of the 256 training readings (divisor 256), its eigenvalues by the Jacobi rotation method. "
            "Error with k components = (sum of the discarded eigenvalues) / 16, which the next column confirms by reconstructing; the last column is the same error on the test set.\n\n")
    return head + _table(["k", "eigenvalue k", "variance kept (%)", "(sum of discarded) / 16", "training error", "test error"], rows)


def linear_vs_pca():
    mu, ev, vecs = pca_fit()
    tr, va, te, _ = sensor_data()
    rows = []
    for k in (1, 2, 3, 4):
        run = linear_model(k)
        th = run["best"]
        spec = linear_spec(k)
        rows.append([str(k), _fx(pca_error(tr, k), 5), _fx(loss_only(spec, th, tr, tr), 5), _fx(100 * (loss_only(spec, th, tr, tr) / pca_error(tr, k) - 1), 2),
                     _fx(pca_error(te, k), 5), _fx(loss_only(spec, th, te, te), 5)])
    sweep = _table(["bottleneck k", "PCA training error", "linear autoencoder training error", "excess over PCA (%)", "PCA test error", "linear autoencoder test error"], rows)
    th = linear_model(3)["best"]
    off = 16 * 3 + 3
    Wd = [th[off + j * 3: off + (j + 1) * 3] for j in range(D)]            # [16, 3]
    cols = [[Wd[j][c] for j in range(D)] for c in range(3)]
    Q = _orthonormal_columns(cols)
    inside = [math.sqrt(sum(sum(a * b for a, b in zip(u, q)) ** 2 for q in Q)) for u in vecs[:4]]
    unit = [[a / math.sqrt(sum(b * b for b in c)) for a in c] for c in cols]
    cos = [[abs(sum(a * b for a, b in zip(u, c))) for u in vecs[:3]] for c in unit]
    sub_rows = [[f"principal direction {j + 1}", _fx(inside[j], 5)] for j in range(4)]
    cos_rows = [[f"decoder column {c + 1}"] + [_fx(cos[c][j], 3) for j in range(3)] for c in range(3)]
    return ("Linear autoencoders 16 -> k -> 16 (no activation), trained by the loop of the autoencoder's training log: AdamW (0.9, 0.999, 1e-8), batch 32, 400 epochs of 8 steps, clip 1.0, decay 0, "
            "warmup of a tenth of the steps then cosine from 0.03 to 0.0003, the shuffle stream of that log (one shuffle of 0 .. 255 per epoch); weights N(0, 1/n_in) row by row, layer by layer, biases 0, from the generator `first_models.lin_init.k` with seeds "
            + ", ".join(f"k = {k}: {_seed(f'first_models.lin_init.{k}')}" for k in (1, 2, 3, 4)) + ". The reported weights are those of the epoch with the lowest validation loss; training, validation and test errors are the mean squared error per entry on the three splits. "
            "The PCA errors are those of the best rank-k reconstruction fitted on the training set.\n\n" + sweep +
            "\n\nBottleneck 3. Length of the projection of each principal direction (unit vector) onto the span of the decoder's three columns (1 means inside the span):\n\n" +
            _table(["direction", "length of projection"], sub_rows) +
            "\n\nAbsolute cosine between each decoder column and each principal direction (a column is not a principal direction, only the span is):\n\n" +
            _table(["", "PC 1", "PC 2", "PC 3"], cos_rows))


def pca_directions_note():
    mu, ev, vecs = pca_fit()
    return ", ".join(_fx(e, 4) for e in ev[:5])


# ------------------------------------------------------------------ curved data

@lru_cache(maxsize=None)
def curved_data():
    gen = bookrng.generator("first_models.curve_data")
    X, S = [], []
    for _ in range(N_TRAIN + N_VAL + N_TEST):
        s = -1.0 + 2.0 * gen.uniform()
        noise = [gen.normal() for _ in range(D)]
        X.append([s * PATTERNS[0][j] + CURVE_QUAD * (s * s - 1.0 / 3.0) * PATTERNS[1][j] + SENSOR_NOISE * noise[j] for j in range(D)])
        S.append(s)
    return X[:N_TRAIN], X[N_TRAIN:N_TRAIN + N_VAL], X[N_TRAIN + N_VAL:], S[N_TRAIN + N_VAL:]


@lru_cache(maxsize=None)
def curved_model():
    tr, va, te, _ = curved_data()
    return fit(CURVE_SPEC, tr, tr, va, va, CURVE_RUN, "first_models.curve_init")


def _pca1_error(X, ref):
    mu = mean_vec(ref)
    d = len(mu)
    C = [[sum((x[i] - mu[i]) * (x[j] - mu[j]) for x in ref) / len(ref) for j in range(d)] for i in range(d)]
    ev, vecs = jacobi_eigh(C)
    u = vecs[0]
    rec = []
    for x in X:
        s = sum((a - b) * c for a, b, c in zip(x, mu, u))
        rec.append([m + s * c for m, c in zip(mu, u)])
    return mse(rec, X), ev


def curved_table():
    tr, va, te, s_te = curved_data()
    run = curved_model()
    th = run["best"]
    e_pca, ev = _pca1_error(te, tr)
    e_ae = loss_only(CURVE_SPEC, th, te, te)
    codes = []
    for x in te:
        a1 = [gelu(sum(map(mul, th[j * 16:(j + 1) * 16], x)) + th[192 + j]) for j in range(12)]
        codes.append(sum(map(mul, th[204: 204 + 12], a1)) + th[216])
    n = len(codes)
    mz, ms = sum(codes) / n, sum(s_te) / n
    cov = sum((a - mz) * (b - ms) for a, b in zip(codes, s_te)) / n
    corr = cov / math.sqrt(sum((a - mz) ** 2 for a in codes) / n * sum((b - ms) ** 2 for b in s_te) / n)
    rows = [["mean predictor", _fx(mean_predictor_loss(te, tr), 5)],
            ["PCA, 1 component", _fx(e_pca, 5)],
            ["autoencoder, bottleneck 1", _fx(e_ae, 5)],
            ["noise floor 15/16 of the sensor noise variance", _fx(15 / 16 * SENSOR_NOISE ** 2, 5)]]
    head = ("Curved data: x_j = s pat1_j + 1.5 (s^2 - 1/3) pat2_j + 0.05 n_j, with s uniform on [-1, 1) (`bookrng` uniform, then 16 normals n_j per example), stream "
            f"{_seed('first_models.curve_data')}, same split sizes. Autoencoder 16-12-1-12-16, rate 0.01, 300 epochs, decay 0, stream {_seed('first_models.curve_init')}. "
            f"Mean squared error on the test set. Correlation of the learned one-number code with the true cause s on the test set: {_fx(corr, 4)}.\n\n")
    return head + _table(["method", "test error"], rows)


def curved_points():
    """(true s, code z, PCA score) of the test set, for the figure."""
    tr, va, te, s_te = curved_data()
    th = curved_model()["best"]
    out = []
    for x, s in zip(te, s_te):
        a1 = [gelu(sum(map(mul, th[j * 16:(j + 1) * 16], x)) + th[192 + j]) for j in range(12)]
        out.append((s, sum(map(mul, th[204: 204 + 12], a1)) + th[216]))
    return out


# ------------------------------------------------------------------ denoising

def _corrupt(sigma):
    def f(rows, gen):
        return [[a + sigma * gen.normal() for a in x] for x in rows]
    f.stream = "first_models.dae_train_noise"
    return f


@lru_cache(maxsize=None)
def dae_model():
    tr, va, te, _ = sensor_data()
    return fit(AE_SPEC, tr, tr, va, va, AE_RUN, "first_models.dae_init", corrupt=_corrupt(DAE_SIGMA))


def denoising_table():
    tr, va, te, _ = sensor_data()
    gen = bookrng.generator("first_models.dae_eval_noise")
    noisy = [[a + DAE_SIGMA * gen.normal() for a in x] for x in te]
    plain, dae = ae_model()["best"], dae_model()["best"]
    rows = [
        ["no model (use the noisy reading)", _fx(mse(noisy, te), 5)],
        ["PCA, 3 components, of the noisy reading", _fx(mse([pca_reconstruct(x, 3) for x in noisy], te), 5)],
        ["autoencoder trained on clean readings", _fx(mse(forward(AE_SPEC, plain, noisy)[0], te), 5)],
        ["denoising autoencoder", _fx(mse(forward(AE_SPEC, dae, noisy)[0], te), 5)],
        ["denoising autoencoder, on clean test readings", _fx(loss_only(AE_SPEC, dae, te, te), 5)],
    ]
    head = (f"Corruption: add {DAE_SIGMA} times a N(0, 1) draw to every one of the 16 values. The denoising autoencoder has the architecture, "
            f"start, shuffle and settings of the plain one; each step draws fresh noise from {_seed('first_models.dae_train_noise')} "
            f"(the batch's 32 examples in order, 16 values each, a corruption per step), start stream {_seed('first_models.dae_init')}. "
            f"The test noise is drawn once from {_seed('first_models.dae_eval_noise')} (128 examples in order, 16 values each). "
            "Every row is the mean squared error against the clean test readings.\n\n")
    return head + _table(["method", "test error against the clean reading"], rows)


# =================================================================== the controller

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def gains(tau, gain):
    return (2.0 * ZETA * WN * tau - 1.0) / gain, tau * WN * WN / gain


def plant_step(y, u, tau, gain):
    u = clamp(u, -1.0, 1.0)
    return y + DT * (-y + gain * u) / tau


def pid_command(u_prev, e, e_prev, kp, ki):
    return clamp(u_prev + kp * (e - e_prev) + ki * DT * e, -1.0, 1.0)


def setpoint(t):
    sp = HOLDS[0][1]
    for start, v in HOLDS:
        if t >= start:
            sp = v
    return sp


def gen_trajectories(n_traj, length, mode, stream, plant=NOMINAL):
    """Samples (e, e_prev, u_prev) -> expert command. mode 'expert': the expert drives. mode 'explore': 30 % noisy, 8 % random, rest expert."""
    gen = bookrng.generator(stream)
    tau, gain = plant
    kp, ki = gains(tau, gain)
    X, T = [], []
    for _ in range(n_traj):
        y = -0.6 + 1.2 * gen.uniform()
        sp, u_prev, e_prev = 0.0, 0.0, 0.0
        for t in range(length):
            if t == 0 or gen.uniform() < 0.10:
                sp = -1.0 + 2.0 * gen.uniform()
            e = sp - y
            u_star = pid_command(u_prev, e, e_prev, kp, ki)
            u_app = u_star
            if mode == "explore":
                r = gen.uniform()
                if r < 0.3:
                    u_app = clamp(u_star + 0.25 * gen.normal(), -1.0, 1.0)
                elif r < 0.38:
                    u_app = -1.0 + 2.0 * gen.uniform()
            X.append([e, e_prev, u_prev])
            T.append([u_star])
            y = plant_step(y, u_app, tau, gain)
            u_prev, e_prev = u_app, e
    return X, T


@lru_cache(maxsize=None)
def ctrl_sets():
    return {"expert": gen_trajectories(CTRL_TRAJ, CTRL_LEN, "expert", "first_models.ctrl_data_expert"),
            "explore": gen_trajectories(CTRL_TRAJ, CTRL_LEN, "explore", "first_models.ctrl_data_explore"),
            "val": gen_trajectories(CTRL_VAL_TRAJ, CTRL_LEN, "expert", "first_models.ctrl_val")}


def rollout(controller, plant=NOMINAL, steps=ROLLOUT_STEPS):
    """Closed loop from y = 0, u_prev = 0, e_prev = 0. Rows (t, sp, y, u, e); y is the measurement before the command acts."""
    tau, gain = plant
    y, u_prev, e_prev = 0.0, 0.0, 0.0
    rows = []
    for t in range(steps):
        sp = setpoint(t)
        e = sp - y
        u = clamp(controller(e, e_prev, u_prev), -1.0, 1.0)
        rows.append((t, sp, y, u, e))
        y = plant_step(y, u, tau, gain)
        u_prev, e_prev = u, e
    return rows


def pid_controller(plant=NOMINAL):
    kp, ki = gains(*plant)
    return lambda e, ep, up: pid_command(up, e, ep, kp, ki)


def net_controller(spec, theta):
    def f(e, ep, up):
        return forward(spec, theta, [[e, ep, up]])[0][0][0]
    return f


def tracking_mse(rows):
    return sum(r[4] ** 2 for r in rows) / len(rows)


def steady_error(rows):
    """Mean |e| over the last 10 steps of each 45-step hold (steps 35 .. 44 of the hold)."""
    errs = [abs(r[4]) for r in rows if r[0] % 45 >= 35]
    return sum(errs) / len(errs)


def hold_metrics(rows):
    """Per hold: overshoot (% of the step) and settling step (first step after which |e| <= 5 % of the step for the rest of the hold)."""
    out, prev = [], 0.0
    for start, sp in HOLDS:
        seg = [r for r in rows if start <= r[0] < start + 45]
        delta = sp - prev
        sign = 1.0 if delta >= 0 else -1.0
        over = max(0.0, max((r[2] - sp) * sign for r in seg)) / abs(delta) * 100.0
        settle = None
        for i in range(len(seg)):
            if all(abs(r[4]) <= 0.05 * abs(delta) for r in seg[i:]):
                settle = seg[i][0] - start
                break
        out.append((delta, over, settle))
        prev = sp
    return out


def action_errors(rows, plant=NOMINAL):
    """RMS difference between the commands used and the expert's command at the same states (on-policy)."""
    kp, ki = gains(*plant)
    sq = []
    u_prev, e_prev = 0.0, 0.0
    for r in rows:
        sq.append((r[3] - pid_command(u_prev, r[4], e_prev, kp, ki)) ** 2)
        u_prev, e_prev = r[3], r[4]
    return math.sqrt(sum(sq) / len(sq))


def _ctrl_epoch_metrics(spec):
    sets = ctrl_sets()
    Xv, Tv = sets["val"]
    pid_rows = rollout(pid_controller())

    def f(epoch, theta):
        rows = rollout(net_controller(spec, theta))
        return {"open_rms": math.sqrt(loss_only(spec, theta, Xv, Tv)), "closed_rms": action_errors(rows),
                "closed_mse": tracking_mse(rows), "ss": steady_error(rows),
                "dev": max(abs(a[2] - b[2]) for a, b in zip(rows, pid_rows))}
    return f


@lru_cache(maxsize=None)
def ctrl_model(mode):
    sets = ctrl_sets()
    X, T = sets[mode]
    Xv, Tv = sets["val"]
    return fit(CTRL_SPEC, X, T, Xv, Tv, CTRL_RUN, "first_models.ctrl_init", on_epoch=_ctrl_epoch_metrics(CTRL_SPEC))


def ctrl_model_uncached(mode):
    sets = ctrl_sets()
    X, T = sets[mode]
    Xv, Tv = sets["val"]
    return fit(CTRL_SPEC, X, T, Xv, Tv, CTRL_RUN, "first_models.ctrl_init")


def lstsq_baseline(X, T):
    """Least squares of the command on (e, e_prev, u_prev, 1) by Gaussian elimination."""
    A = [x + [1.0] for x in X]
    n = 4
    M = [[sum(a[i] * a[j] for a in A) for j in range(n)] for i in range(n)]
    b = [sum(a[i] * t[0] for a, t in zip(A, T)) for i in range(n)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[p] = M[p], M[i]
        b[i], b[p] = b[p], b[i]
        for r in range(i + 1, n):
            f = M[r][i] / M[i][i]
            M[r] = [a - f * c for a, c in zip(M[r], M[i])]
            b[r] -= f * b[i]
    w = [0.0] * n
    for i in reversed(range(n)):
        w[i] = (b[i] - sum(M[i][j] * w[j] for j in range(i + 1, n))) / M[i][i]
    return w


def ctrl_plant_expert():
    rows = []
    for plant in [NOMINAL] + HELD_OUT:
        kp, ki = gains(*plant)
        rows.append([f"{plant[0]}", f"{plant[1]}", _fx(kp, 4), _fx(ki, 4), _fx(ki * DT, 4)])
    kp, ki = gains(*NOMINAL)
    steps = rollout(pid_controller())
    show = [0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 44]
    rows2 = [[str(r[0]), _fx(r[1], 2), _fx(r[2], 4), _fx(r[4], 4), _fx(r[3], 4)] for r in steps if r[0] in show]
    m = hold_metrics(steps)
    head = (f"The plant is y(t+1) = y(t) + {DT} (-y(t) + gain u(t)) / tau with the command clamped to [-1, 1]. The expert is the velocity-form PI "
            f"u(t) = clamp(u(t-1) + kp (e(t) - e(t-1)) + ki {DT} e(t)) with kp = (2 {ZETA} {WN} tau - 1)/gain and ki = tau {WN}^2/gain "
            "(poles placed at damping 1 and natural frequency 3.5). Gains:\n\n")
    return head + _table(["tau", "gain", "kp", "ki", "ki dt"], rows) + \
        f"\n\nThe nominal plant (tau {NOMINAL[0]}, gain {NOMINAL[1]}) under the expert, from y = 0, u = 0, setpoint 0.70 (first hold, steps 0 to 44):\n\n" + \
        _table(["t", "setpoint", "y(t)", "e(t)", "u(t)"], rows2) + \
        f"\n\nOvershoot of the first hold: {_fx(m[0][1], 2)} % of the step; first step after which |e| stays within 5 % of the step: {m[0][2]}."


def ctrl_data():
    sets = ctrl_sets()
    X, T = sets["explore"]
    Xe, Te = sets["expert"]
    Xv, Tv = sets["val"]
    sat = lambda T_: sum(abs(t[0]) >= 1.0 for t in T_) / len(T_)
    w = lstsq_baseline(X, T)
    mean_u = sum(t[0] for t in T) / len(T)
    lin_val = mse([[sum(a * b for a, b in zip(w, x + [1.0]))] for x in Xv], Tv)
    mean_val = mse([[mean_u]] * len(Tv), Tv)
    head = (f"{CTRL_TRAJ} trajectories of {CTRL_LEN} steps on the nominal plant, so {len(X)} samples per training set, and {CTRL_VAL_TRAJ} validation trajectories ({len(Xv)} samples, the expert driving). "
            f"Per trajectory, in this draw order: y(0) = -0.6 + 1.2 u (one uniform u); the setpoint, the previous command u_prev and the previous error e_prev start at 0. At step t = 0 a new setpoint -1 + 2 u is drawn (one uniform) and no c is drawn; "
            "at every later step c = uniform and, if c < 0.10, a new setpoint -1 + 2 u (one more uniform). Then e = setpoint - y and u* is the expert's command from e, e_prev and u_prev (chapter section above); "
            "for the 'explore' set then r = uniform and, if r < 0.3, a normal draw g and the applied command clamp(u* + 0.25 g, -1, 1), "
            "else if r < 0.38 a uniform draw and the applied command -1 + 2 u, else the applied command u*. "
            "The plant then advances with the applied command, u_prev becomes the applied command and e_prev becomes e. "
            f"The sample is (e, e_prev, u_prev) -> u* (the expert's command at that state, whatever was applied). The mean predictor is the constant equal to the mean of the explore training targets. Streams: {_seed('first_models.ctrl_data_expert')}, "
            f"{_seed('first_models.ctrl_data_explore')}, {_seed('first_models.ctrl_val')}. The 'expert' set draws no r and no g.\n\n")
    rows = [
        ["samples in the training set", str(len(X)), str(len(Xe))],
        ["fraction of targets at the clamp (abs(u*) = 1)", _fx(sat(T), 3), _fx(sat(Te), 3)],
        ["mean predictor: validation error", _fx(mean_val, 5), ""],
        ["linear baseline on (e, e_prev, u_prev, 1): validation error", _fx(lin_val, 5), ""],
        ["the linear baseline's weights", ", ".join(_fx(a, 3) for a in w), ""],
    ]
    return head + _table(["", "explore set", "expert set"], rows)


def ctrl_prechecks():
    sets = ctrl_sets()
    X, T = sets["explore"]
    Xv, Tv = sets["val"]
    run = ctrl_model("explore")
    theta0 = init_theta(CTRL_SPEC, bookrng.generator("first_models.ctrl_init"))
    X8, T8 = X[:8], T[:8]
    ratio, rel, failing, per = gradient_check(CTRL_SPEC, theta0, X8, T8)
    first, last = overfit_one_batch(CTRL_SPEC, theta0, X8, T8)
    mean_u = sum(t[0] for t in T) / len(T)
    base = mse([[mean_u]] * len(Tv), Tv)
    finite = all(math.isfinite(a) for x in X + Xv for a in x)
    rows = [
        ["1. shapes and finiteness", f"inputs {len(X)} x 3, targets {len(T)} x 1; every value finite: {finite}", ""],
        ["2. initial loss", f"training {_fx(run['start']['train'], 4)}, validation {_fx(run['start']['val'], 4)}", f"the mean predictor scores {_fx(base, 4)} on validation"],
        [f"3. gradient check, all {n_params(CTRL_SPEC)} entries", f"worst gate ratio abs(ana - num) / (atol + rtol max(abs(ana), abs(num))) = {_fx(ratio, 4)}, failing {failing}", "gate: h = 1e-5, atol = 1e-8, rtol = 1e-5"],
        ["   worst relative error per tensor", ", ".join(f"{k} {_sci(w, 1)}" for k, w in per.items()), "entries with max(abs(ana), abs(num)) above 1e-4"],
        ["4. overfit one batch (8 samples, 400 steps, rate 0.01)", f"loss {_fx(first, 4)} -> {_sci(last, 2)}", "no clip, no decay"],
    ]
    return _table(["check", "result", "reference"], rows, "lll")


def ctrl_training_log():
    run = ctrl_model("explore")
    shown = [1, 2, 3, 5, 8, 12, 20, 40, 70, 100]
    rows = [["0", _fx(run["start"]["train"], 5), _fx(run["start"]["val"], 5), "", "", ""]]
    for r in run["log"]:
        if r["epoch"] in shown:
            rows.append([str(r["epoch"]), _fx(r["train"], 5), _fx(r["val"], 5), _fx(r["norm"], 4), str(r["clipped"]), _sci(r["lr"], 2)])
    X, T = ctrl_sets()["explore"]
    head = (f"{CTRL_RUN['epochs']} epochs of {len(X) // CTRL_RUN['batch']} steps at batch {CTRL_RUN['batch']} ({run['steps']} steps), on the explore set. AdamW (0.9, 0.999, 1e-8), decay {CTRL_RUN['lam']}, clip {CTRL_RUN['tau']}, "
            f"warmup {run['warmup']} steps then cosine from {CTRL_RUN['peak']} to {_sci(run['floor'], 1)}; init {_seed('first_models.ctrl_init')}, shuffle {_seed(BATCH_STREAM)} (one shuffle of 0 .. {len(X) - 1} per epoch). "
            "Columns as in the autoencoder's log. The validation set is the expert-driven one.\n\n")
    return head + _table(["epoch", "training loss", "validation loss", "grad norm", "clipped", "rate"], rows)


def ctrl_closed_loop():
    sets = ctrl_sets()
    rows = []
    pid_rows = rollout(pid_controller())
    entries = [("expert (PID)", pid_rows)]
    for mode, label in (("expert", "network, trained on expert states"), ("explore", "network, trained on explored states")):
        th = ctrl_model(mode)["best"]
        entries.append((label, rollout(net_controller(CTRL_SPEC, th))))
    for label, r in entries:
        m = hold_metrics(r)
        rows.append([label, _fx(tracking_mse(r), 5), _fx(steady_error(r), 5), " / ".join(_fx(x[1], 1) for x in m),
                     " / ".join("-" if x[2] is None else str(x[2]) for x in m), _fx(max(abs(x[3]) for x in r), 3), _fx(action_errors(r), 4),
                     _fx(max(abs(a[2] - b[2]) for a, b in zip(r, pid_rows)), 4)])
    head = ("Closed loop on the nominal plant, 180 steps, setpoint 0.70, -0.60, 0.40, -0.85 held for 45 steps each, from y = 0, u_prev = 0, e_prev = 0. "
            "Each network is its training run's best-validation checkpoint (the weights of the epoch with the lowest validation loss, as kept in the training log), not the weights of the last epoch. "
            "'Tracking error' is the mean of e(t)^2 over the 180 steps; 'steady error' the mean |e| over steps 35 .. 44 of each hold; overshoot is "
            "(largest excursion beyond the setpoint) / |step| in percent; settling is the first step of the hold after which |e| stays within 5 % of the step; "
            "'action error' is the RMS difference between the commands used and the expert's command at the states visited; 'largest deviation' is the maximum of |y - y_PID|.\n\n")
    return head + _table(["controller", "tracking error", "steady error", "overshoot % (4 holds)", "settling steps (4 holds)", "largest abs(u)", "action error, on its own states", "largest deviation from PID"], rows)


def ctrl_open_vs_closed():
    run = ctrl_model("explore")
    shown = [1, 2, 3, 5, 8, 12, 20, 40, 100]
    rows = []
    for r in run["log"]:
        if r["epoch"] in shown:
            rows.append([str(r["epoch"]), _fx(r["open_rms"], 4), _fx(r["closed_rms"], 4), _fx(r["closed_mse"], 5), _fx(r["ss"], 4), _fx(r["dev"], 3)])
    pid_rows = rollout(pid_controller())
    head = (f"The explore-trained network at the end of epoch n. 'one-step error' is the RMS error of the command on the validation samples (teacher forcing: every input is an expert state). "
            f"'action error, own states' is the same difference measured at the states the network itself visits in the 180-step closed loop. The expert's own closed loop has tracking error {_fx(tracking_mse(pid_rows), 5)} and steady error {_fx(steady_error(pid_rows), 5)}.\n\n")
    return head + _table(["epoch", "one-step error", "action error, own states", "tracking error", "steady error", "largest deviation from PID"], rows)


def ctrl_bias_amplification():
    kp, ki = gains(*NOMINAL)
    tau, gain = NOMINAL
    rows = []
    for eps in (0.0, 0.001, 0.005, 0.02, 0.05):
        y, u_prev, e_prev = 0.0, 0.0, 0.0
        for t in range(150):
            e = 0.70 - y
            u = clamp(pid_command(u_prev, e, e_prev, kp, ki) + eps, -1.0, 1.0)
            y = plant_step(y, u, tau, gain)
            u_prev, e_prev = u, e
        e_end = 0.70 - y
        rows.append([_fx(eps, 3), _fx(e_end, 5), _fx(-eps / (ki * DT), 5), _fx(e_end / eps, 3) if eps else ""])
    head = (f"The expert on the nominal plant with a constant error eps added to every command it makes (the added error is part of the stored u(t-1)), "
            f"setpoint held at 0.70 for 150 steps from y = 0. ki dt = {_fx(ki * DT, 4)}, so 1 / (ki dt) = {_fx(1 / (ki * DT), 3)}. "
            "Settled error e(150) = 0.70 - y(150) against the prediction -eps / (ki dt).\n\n")
    return head + _table(["added error eps", "settled error e(150)", "predicted -eps / (ki dt)", "ratio e / eps"], rows)


def ctrl_best_epochs():
    return ", ".join(f"{m}: {ctrl_model(m)['best_epoch']}" for m in ("expert", "explore"))


def ctrl_held_out():
    th = ctrl_model("explore")["best"]
    rows = []
    for plant in [NOMINAL] + HELD_OUT:
        a = rollout(pid_controller(plant), plant)
        b = rollout(net_controller(CTRL_SPEC, th), plant)
        c = rollout(pid_controller(NOMINAL), plant)
        rows.append([f"{plant[0]}", f"{plant[1]}", _fx(tracking_mse(a), 5), _fx(tracking_mse(c), 5), _fx(tracking_mse(b), 5), _fx(steady_error(a), 4), _fx(steady_error(c), 4), _fx(steady_error(b), 4)])
    head = ("The same 180-step test on the nominal plant and on the four plants between the nominal grid of crates/toypid (never used for training here). "
            "'plant PID' is the expert tuned for that plant; 'nominal PID' is the nominal plant's gains used on it; 'network' is the explore-trained network.\n\n")
    return head + _table(["tau", "gain", "tracking, plant PID", "tracking, nominal PID", "tracking, network", "steady, plant PID", "steady, nominal PID", "steady, network"], rows)


# ------------------------------------------------------------------ determinism, work

def determinism():
    a = ae_model()
    b = ae_model_uncached()
    c = ctrl_model("explore")
    d = ctrl_model_uncached("explore")
    rows = []
    for name, x, y in (("autoencoder", a, b), ("controller", c, d)):
        same_log = all(p["train"] == q["train"] and p["val"] == q["val"] for p, q in zip(x["log"], y["log"]))
        diff = max(abs(p - q) for p, q in zip(x["theta"], y["theta"]))
        rows.append([name, digest(x["theta"])[:16], digest(y["theta"])[:16], _sci(diff, 1), str(same_log)])
    return _table(["model", "run 1: SHA-256 of the final parameters (first 16 hex)", "run 2", "largest parameter difference", "all logged losses equal"], rows)


def work_counts():
    rows = []
    for name, spec, B, N, Nv, E in (("autoencoder", AE_SPEC, AE_RUN["batch"], N_TRAIN, N_VAL, AE_RUN["epochs"]),
                                    ("controller", CTRL_SPEC, CTRL_RUN["batch"], CTRL_TRAJ * CTRL_LEN, CTRL_VAL_TRAJ * CTRL_LEN, CTRL_RUN["epochs"])):
        P, M = n_params(spec), macs(spec)
        fwd, wg, dg, clip, adam = step_flops(spec, B)
        step = fwd + wg + dg + clip + adam
        spe = N // B
        evalf = 2 * M * (N + Nv)
        epoch = spe * step + evalf
        rows.append([name, _int(P), _int(M), _int(fwd), _int(wg), _int(dg), _int(clip + adam), _int(step), _int(epoch), _int(E * epoch)])
    first = _table(["model", "parameters P", "multiply-adds M per example", "forward", "weight gradients", "gradients passed down", "clip and AdamW", "FLOPs per step", "FLOPs per epoch (steps and two evaluations)", f"FLOPs per run"], rows)
    mem = []
    for name, spec, B in (("autoencoder", AE_SPEC, AE_RUN["batch"]), ("controller", CTRL_SPEC, CTRL_RUN["batch"])):
        P = n_params(spec)
        saved = saved_floats(spec, B)
        mem.append([name, _int(4 * P), _int(4 * P), _int(8 * P), _int(16 * P), _int(4 * saved), _int(28 * P), _fx(19 / 28, 3)])
    second = _table(["model", "parameter bytes", "gradient bytes", "AdamW state bytes (m and v)", "persistent total", "saved activations of one step (bytes)", "bytes moved by one optimizer step", "clip and AdamW FLOPs per byte (19 / 28)"], mem)
    head = ("Matrix-product FLOPs follow chapter 12's rule: forward 2 M B, weight gradients 2 M B, gradients passed down 2 (M - M_1) B with M_1 the first layer's multiply-adds; "
            "the clip costs 3 FLOPs and AdamW 16 per parameter (chapter 7). An epoch adds one forward pass over the training and the validation set. All arrays are counted as 4-byte floats. The last column divides the 19 FLOPs per parameter of the clip and AdamW together (3 + 16) by the 28 bytes the optimizer step moves; AdamW alone is 16 / 28 = 0.571 (chapter 7).\n\n")
    return head + first + "\n\nMemory and traffic of the same runs:\n\n" + second


def toypid_counts():
    d, ff, vocab, block, u_bins, layers = 32, 128, 263, 256, 81, 1
    params = vocab * d + block * d
    per_block = 2 * d + (3 * d * d + 3 * d) + (d * d + d) + 2 * d + 2 * (ff * d + ff) + (d * ff + d)
    params += layers * per_block + 2 * d + (u_bins * d + u_bins)
    per_token = layers * (3 * d * d + d * d + 3 * d * ff) + u_bins * d
    window = block
    attn = layers * 2 * window * window * d
    flops = 2 * (window * per_token + attn)
    mlp_flops = 2 * macs(CTRL_SPEC)
    rows = [
        ["parameters", _int(params), _int(n_params(CTRL_SPEC))],
        ["parameter bytes (4-byte floats)", _int(4 * params), _int(4 * n_params(CTRL_SPEC))],
        ["FLOPs for one control decision", _int(flops), _int(mlp_flops)],
        ["ratio of the FLOPs", _fx(flops / mlp_flops, 0), "1"],
    ]
    head = ("toypid's default model (`default_small`: vocabulary 263, window 256, one layer, width 32, four heads, feed-forward width 128, 81 command bins) against this chapter's controller. "
            "FLOPs of a decision count the matrix products of the whole window (the model recomputes its window, and applies its output head at every position): per token 3 d^2 + d^2 for the attention projections, "
            "3 d d_ff for the three feed-forward matrices and 81 d for the head, plus 2 T^2 d for the dense attention scores and weighted sums, each multiply-add as two FLOPs, T = 256.\n\n")
    return head + _table(["", "toypid transformer (window of 256 tokens)", "this chapter's controller"], rows)


def checklist_table():
    tr, va, te, _ = sensor_data()
    sets = ctrl_sets()
    rows = []
    for name, spec, run_cfg, N, Nv, Xb, label in (
            ("autoencoder", AE_SPEC, AE_RUN, N_TRAIN, N_VAL, 32, "reconstruction"),
            ("controller", CTRL_SPEC, CTRL_RUN, CTRL_TRAJ * CTRL_LEN, CTRL_VAL_TRAJ * CTRL_LEN, 72, "imitation")):
        B = run_cfg["batch"]
        M = macs(spec)
        fwd, wg, dg, clip, adam = step_flops(spec, B)
        step = fwd + wg + dg + clip + adam
        run_total = run_cfg["epochs"] * ((N // B) * step + 2 * M * (N + Nv))
        init_loss = fwd
        gc = 2 * n_params(spec) * (2 * M * 8) + (fwd + wg + dg) // B * 8
        overfit = 400 * ((fwd + wg + dg) // B * 8 + adam)
        rows.append([name, "2. initial loss: one forward pass of a batch", _int(init_loss), _fx(100 * init_loss / run_total, 3)])
        rows.append([name, "4. gradient check: 2 P forward passes of 8 examples and one backward", _int(gc), _fx(100 * gc / run_total, 3)])
        rows.append([name, "7. overfit one batch: 400 steps of 8 examples", _int(overfit), _fx(100 * overfit / run_total, 3)])
        rows.append([name, "the run itself", _int(run_total), "100"])
    return _table(["model", "check", "FLOPs", "% of the run"], rows, "llrr")
