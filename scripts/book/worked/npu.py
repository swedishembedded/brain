# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Neural Processing Units and Graph Compilers".

Plain Python 3, standard library only, written from the chapter's own
description. Each public function without a leading underscore that takes no
arguments returns a markdown string that the chapter embeds between
`<!-- worked: npu.<name> -->` markers. Helpers with a leading underscore, or
that take arguments, are shared with the chapter's figures so a picture and the
table beside it cannot disagree.

Conventions fixed once for the whole module:

* the network is a two-layer perceptron, 6 inputs, 4 hidden units with a
  rectifier, 3 outputs; its weights come from the book's generator;
* every float forward pass accumulates a dot product from index 0 upward and
  adds the bias last, in binary64;
* an activation is quantized per tensor with signed codes -128 .. 127 and a
  zero point, a weight per output channel with symmetric codes -127 .. 127;
  every quotient is rounded to nearest, ties to even (Python's `round` on a
  float, `Fraction.__round__` on an exact quotient);
* an integer accumulator is an exact Python int; the fixed-point requantizer
  takes a 15-bit multiplier and a shift, and its rounding shift is the exact
  quotient rounded to nearest, ties to even;
* the error of a result is the decoded integer output minus the float output;
  SQNR is 10 log10(mean y^2 / MSE) in decibels over every output of the test set;
* a random draw comes from the book's generator, `bookrng.generator(name)`, and
  each function states the order of its draws.

The device-format facts used at the end (the 16-lane layout, the 6144-atom
input buffer, the 126-register task profile, the 64-bit command word) are
those of the Rockchip RV1103 compiler in `crates/rknpu`, labelled as stated by
that source wherever they appear.

Swedish Embedded AB implements solutions for compiling whole neural networks
into integer programs for fixed-function accelerators. If your team needs
expertise in post-training quantization and graph compilers for edge NPUs, you
can procure our services by sending an email to info@swedishembedded.com.
"""
import math
from fractions import Fraction

import bookrng

# ---------------------------------------------------------------- formatting


def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _seed_line(name):
    return f"seed {bookrng.seed_of(name)} (seed_of '{name}', defined on the pseudocode page)"


def _v(values, nd=2):
    return "[" + ", ".join(f"{x:+.{nd}f}" for x in values) + "]"


# ---------------------------------------------------------------- the network

IN, HID, OUT = 6, 4, 3
W_SCALE, B_SCALE = 0.6, 0.2


def network():
    """(W1 [4][6], b1 [4], W2 [3][4], b2 [3]): normal() draws in this order,
    W1 row-major, b1, W2 row-major, b2; weights 0.6 z and biases 0.2 z, each
    rounded to 2 decimals."""
    g = bookrng.generator("npu.network")
    w1 = [[round(W_SCALE * g.normal(), 2) for _ in range(IN)] for _ in range(HID)]
    b1 = [round(B_SCALE * g.normal(), 2) for _ in range(HID)]
    w2 = [[round(W_SCALE * g.normal(), 2) for _ in range(HID)] for _ in range(OUT)]
    b2 = [round(B_SCALE * g.normal(), 2) for _ in range(OUT)]
    return w1, b1, w2, b2


def _dot(w, x):
    s = 0.0
    for a, b in zip(w, x):
        s += a * b
    return s


def forward(x, net=None):
    """Float forward pass: (h1, a1, y) with h1 = W1 x + b1, a1 = relu(h1), y = W2 a1 + b2."""
    w1, b1, w2, b2 = net or network()
    h1 = [_dot(w1[o], x) + b1[o] for o in range(HID)]
    a1 = [max(0.0, v) for v in h1]
    y = [_dot(w2[o], a1) + b2[o] for o in range(OUT)]
    return h1, a1, y


def _inputs(name, n, scale=1.0):
    """n inputs of 6 values: round(scale * normal(), 3), drawn in order."""
    g = bookrng.generator(name)
    return [[round(scale * g.normal(), 3) for _ in range(IN)] for _ in range(n)]


_CAL = {}


def _calibration(n):
    """The first n inputs of the calibration stream: x_i = round(normal(), 3), 128 inputs in draw order."""
    if "inputs" not in _CAL:
        _CAL["inputs"] = _inputs("npu.calibration", N_CAL_MAX)
    return _CAL["inputs"][:n]


def _spiky(name, n, p=0.02, factor=6.0):
    """n inputs of 6 values; for each: six normal() draws, then one uniform();
    when the uniform is below p every value is multiplied by factor. Rounded to 3 decimals."""
    g = bookrng.generator(name)
    out = []
    for _ in range(n):
        z = [g.normal() for _ in range(IN)]
        k = factor if g.uniform() < p else 1.0
        out.append([round(k * v, 3) for v in z])
    return out


N_CAL = 32
N_CAL_MAX = 128
N_TEST = 256
EXAMPLE = [round(v, 3) for v in (lambda g: [g.normal() for _ in range(IN)])(bookrng.generator("npu.example"))]


def network_weights():
    """The seeded weights (the initializers of the exported graph)."""
    w1, b1, w2, b2 = network()
    rows1 = [[f"W1[{o}]"] + [f"{v:+.2f}" for v in w1[o]] + [f"{b1[o]:+.2f}"] for o in range(HID)]
    rows2 = [[f"W2[{o}]"] + [f"{v:+.2f}" for v in w2[o]] + [f"{b2[o]:+.2f}"] for o in range(OUT)]
    t1 = _table(["row", "i=0", "i=1", "i=2", "i=3", "i=4", "i=5", "bias"], rows1)
    t2 = _table(["row", "i=0", "i=1", "i=2", "i=3", "bias"], rows2)
    return (f"The weights of the network, {_seed_line('npu.network')}. Draw order: normal() for W1[0][0], W1[0][1], ... "
            f"W1[3][5] (row-major, 24 draws), then b1[0..3], then W2[0][0] ... W2[2][3] (12 draws), then b2[0..2]; weights are "
            f"round(0.6 z, 2), biases round(0.2 z, 2). The printed values are the inputs of everything below.\n\n"
            f"Layer 1, W1 of shape [4, 6] and b1 of shape [4]:\n\n{t1}\n\nLayer 2, W2 of shape [3, 4] and b2 of shape [3]:\n\n{t2}")


def float_forward():
    h1, a1, y = forward(EXAMPLE)
    rows = [["x", f"{_v(EXAMPLE, 3)}"],
            ["h1 = W1 x + b1", _v(h1, 4)],
            ["a1 = relu(h1)", _v(a1, 4)],
            ["y = W2 a1 + b2", _v(y, 4)]]
    return (f"One input, x_i = round(normal(), 3) for i = 0 .. 5 from the generator, {_seed_line('npu.example')}. "
            f"The float forward pass of the network above (binary64, dot products summed from index 0, bias added last).\n\n"
            + _table(["step", "value"], rows, "ll"))


# ---------------------------------------------------------------- graph export


def graph_listing():
    """The exported graph as an ONNX-style listing plus a per-node cost table."""
    w1, b1, w2, b2 = network()
    listing = "\n".join([
        "graph tiny_mlp",
        "  input   x : f32[1, 6]",
        "  output  y : f32[1, 3]",
        "  initializer W1 : f32[4, 6]    b1 : f32[4]",
        "  initializer W2 : f32[3, 4]    b2 : f32[3]",
        "  node 0  Gemm(x, W1, b1)   attr transB = 1   -> h1 : f32[1, 4]",
        "  node 1  Relu(h1)                             -> a1 : f32[1, 4]",
        "  node 2  Gemm(a1, W2, b2)  attr transB = 1   -> y  : f32[1, 3]",
        "  opset 13, ir_version 8, every dimension static",
    ])
    rows = []
    for name, op, ins, out, shape, params, flops, rd, wr in (
            ("0", "Gemm", "x, W1, b1", "h1", "[1, 4]", 4 * 6 + 4, 2 * 4 * 6 + 4, (6 + 24 + 4) * 4, 4 * 4),
            ("1", "Relu", "h1", "a1", "[1, 4]", 0, 4, 4 * 4, 4 * 4),
            ("2", "Gemm", "a1, W2, b2", "y", "[1, 3]", 3 * 4 + 3, 2 * 3 * 4 + 3, (4 + 12 + 3) * 4, 3 * 4)):
        rows.append([name, op, ins, f"{out} {shape}", params, flops, rd, wr])
    total = [sum(r[i] for r in rows) for i in (5, 6, 7)]
    rows.append(["total", "", "", "", sum(r[4] for r in rows[:3]), total[0], total[1], total[2]])
    n_init = HID * IN + HID + OUT * HID + OUT
    table = _table(["node", "op", "inputs", "output", "initializer values", "FLOPs", "bytes read", "bytes written"], rows,
                   "lllllrrr")
    return (f"```text\n{listing}\n```\n\n"
            f"Counted as in Chapter 8: a Gemm of a vector with an [out, in] weight does 2 x out x in FLOPs for the products and sums plus out for the bias; "
            f"every distinct operand is read once and the result written once, in float32 (4 bytes). The exported file carries {n_init} initializer values, "
            f"{4 * n_init} bytes of float32.\n\n{table}")


# ---------------------------------------------------------------- calibration


def _flat(vs):
    return [x for v in vs for x in v]


def _tensors(inputs, net=None):
    """The values of x, h1, a1, y over `inputs`, each flattened in input order."""
    xs, hs, as_, ys = [], [], [], []
    for x in inputs:
        h1, a1, y = forward(x, net)
        xs += x
        hs += h1
        as_ += a1
        ys += y
    return {"x": xs, "h1": hs, "a1": as_, "y": ys}


def _percentile(a, q):
    v = sorted(a)
    pos = q * (len(v) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (pos - lo) * (v[hi] - v[lo])


def _range(vals, tail=0.0):
    """(lo, hi) of the values: min and max, or the tail and 1 - tail percentiles."""
    if tail == 0.0:
        return min(vals), max(vals)
    return _percentile(vals, tail), _percentile(vals, 1.0 - tail)


def calibration_ranges():
    cal = _calibration(N_CAL)
    t = _tensors(cal)
    shapes = {"x": "[1, 6]", "h1": "[1, 4]", "a1": "[1, 4]", "y": "[1, 3]"}
    rows = [[k, shapes[k], len(v), f"{min(v):+.4f}", f"{max(v):+.4f}"] for k, v in t.items()]
    return (f"A calibration set of {N_CAL} inputs, each x_i = round(normal(), 3) for i = 0 .. 5, drawn in order from the generator, "
            f"{_seed_line('npu.calibration')}. Each float tensor is recorded for every input; 'values' counts all numbers recorded for the tensor.\n\n"
            + _table(["tensor", "shape", "values", "min", "max"], rows, "llrrr"))


# ---------------------------------------------------------------- quantization


def _rne(x):
    return round(x)


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def affine_params(lo, hi):
    """(delta, q0) for signed codes -128 .. 127: the range is widened to contain 0,
    delta = (hi - lo) / 255 and q0 = round_even(-128 - lo / delta), clamped to the codes."""
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    delta = (hi - lo) / 255.0 if hi > lo else 1.0
    q0 = _clamp(_rne(-128 - lo / delta), -128, 127)
    return delta, q0


def quantize_act(r, delta, q0):
    """Code of the real number r and whether it had to be clamped."""
    q = _rne(r / delta) + q0
    return _clamp(q, -128, 127), not -128 <= q <= 127


def weight_params(w):
    """Per-output-channel symmetric int8: codes -127 .. 127, delta_o = max|w_o| / 127."""
    codes, deltas = [], []
    for row in w:
        d = max(abs(v) for v in row) / 127.0
        codes.append([_rne(v / d) for v in row])
        deltas.append(d)
    return codes, deltas


def exponent_of(x):
    """e with 2^e <= x < 2^(e+1), exactly."""
    return math.frexp(x)[1] - 1


def fixed_point(m, bits=15):
    """(multiplier, shift) with m ~ multiplier / 2^shift and 2^(bits-1) <= multiplier < 2^bits:
    shift = exponent_of((2^bits - 1) / m), multiplier = round_even(m 2^shift)."""
    if not 0 < m < 2 ** bits:
        raise ValueError("factor outside the range of the multiplier")
    shift = exponent_of(((1 << bits) - 1) / m)
    return _rne(m * 2.0 ** shift), shift


def round_shift(x, n):
    """x / 2^n rounded to the nearest integer, ties to even, exactly (x an int, n >= 0)."""
    if n == 0:
        return x
    q, r = divmod(x, 1 << n)
    half = 1 << (n - 1)
    if r > half or (r == half and q % 2 == 1):
        q += 1
    return q


class QuantizedMLP:
    """The network after post-training quantization from a calibration set.

    `tail` is the fraction clipped from each end of every activation range (0 =
    min and max). `requant` is 'channel' (a multiplier and shift per output
    channel) or 'two_stage' (a Q14 factor per channel, then one multiplier and
    shift per layer, the structure of crates/rknpu)."""

    def __init__(self, cal_inputs, tail=0.0, requant="channel", net=None):
        self.net = net or network()
        w1, b1, w2, b2 = self.net
        t = _tensors(cal_inputs, self.net)
        self.dx, self.qx = affine_params(*_range(t["x"], tail))
        self.da, self.qa = affine_params(*_range(t["a1"], tail))
        self.dy, self.qy = affine_params(*_range(t["y"], tail))
        self.c1, self.dw1 = weight_params(w1)
        self.c2, self.dw2 = weight_params(w2)
        self.bq1 = [_rne(b1[o] / (self.dx * self.dw1[o])) for o in range(HID)]
        self.bq2 = [_rne(b2[o] / (self.da * self.dw2[o])) for o in range(OUT)]
        self.kind = requant
        self.m1 = [self.dx * self.dw1[o] / self.da for o in range(HID)]
        self.m2 = [self.da * self.dw2[o] / self.dy for o in range(OUT)]
        if requant == "channel":
            self.f1 = [fixed_point(m) for m in self.m1]
            self.f2 = [fixed_point(m) for m in self.m2]
        else:
            self.f1 = self._two_stage(self.dx, self.dw1, self.da)
            self.f2 = self._two_stage(self.da, self.dw2, self.dy)
        # zero point folded into the bias: sum (q - z) w = sum q w - z sum w
        self.fb1 = [self.bq1[o] - self.qx * sum(self.c1[o]) for o in range(HID)]
        self.fb2 = [self.bq2[o] - self.qa * sum(self.c2[o]) for o in range(OUT)]

    @staticmethod
    def _two_stage(d_in, dw, d_out):
        """(q14 factors per channel, layer multiplier, layer shift)."""
        dmax = max(dw)
        q14 = [_rne(d / dmax * 16384) for d in dw]
        mult, shift = fixed_point(d_in * dmax / d_out)
        return q14, mult, shift

    def _requant(self, acc, params, o, zero):
        """Output code of accumulator `acc` of channel o, and whether it was clamped."""
        if self.kind == "channel":
            m, n = params[o]
            v = round_shift(m * acc, n) + zero
        else:
            q14, mult, shift = params
            v = round_shift(round_shift(acc * q14[o], 14) * mult, shift) + zero
        return _clamp(v, -128, 127), not -128 <= v <= 127

    def quantize_input(self, x):
        pairs = [quantize_act(v, self.dx, self.qx) for v in x]
        return [p[0] for p in pairs], sum(p[1] for p in pairs)

    def run(self, x):
        """Compiled form: raw codes, folded bias, relu on the accumulator.
        Returns (input codes, acc1, a1 codes, acc2, y codes, clamped count)."""
        qx, clipped = self.quantize_input(x)
        acc1 = [sum(qx[i] * self.c1[o][i] for i in range(IN)) + self.fb1[o] for o in range(HID)]
        a, c = [], 0
        for o in range(HID):
            q, cl = self._requant(max(acc1[o], 0), self.f1, o, self.qa)
            a.append(q)
            c += cl
        acc2 = [sum(a[i] * self.c2[o][i] for i in range(HID)) + self.fb2[o] for o in range(OUT)]
        y = []
        for o in range(OUT):
            q, cl = self._requant(acc2[o], self.f2, o, self.qy)
            y.append(q)
            c += cl
        return qx, acc1, a, acc2, y, clipped + c

    def reference(self, x):
        """Direct reference: zero points subtracted before the products, requantization by exact rationals."""
        qx = [quantize_act(v, self.dx, self.qx)[0] for v in x]
        a = []
        for o in range(HID):
            acc = sum((qx[i] - self.qx) * self.c1[o][i] for i in range(IN)) + self.bq1[o]
            a.append(self._ref_requant(max(acc, 0), self.f1, o, self.qa))
        y = []
        for o in range(OUT):
            acc = sum((a[i] - self.qa) * self.c2[o][i] for i in range(HID)) + self.bq2[o]
            y.append(self._ref_requant(acc, self.f2, o, self.qy))
        return y

    def _ref_requant(self, acc, params, o, zero):
        if self.kind == "channel":
            m, n = params[o]
            v = round(Fraction(m * acc, 2 ** n)) + zero
        else:
            q14, mult, shift = params
            v = round(Fraction(round(Fraction(acc * q14[o], 2 ** 14)) * mult, 2 ** shift)) + zero
        return _clamp(v, -128, 127)

    def decode(self, y_codes):
        return [self.dy * (q - self.qy) for q in y_codes]


def _sqnr_of(model, tests):
    """(SQNR dB, largest |error|, share of values clamped, number of tests) of the compiled net on `tests`."""
    se = sp = 0.0
    worst = 0.0
    clamped = 0
    for x in tests:
        run = model.run(x)
        yq = model.decode(run[4])
        clamped += run[5]
        y = forward(x, model.net)[2]
        for a, b in zip(yq, y):
            se += (a - b) ** 2
            sp += b * b
            worst = max(worst, abs(a - b))
    total = len(tests) * (IN + HID + OUT)
    return 10 * math.log10(sp / se), worst, clamped / total


def activation_params():
    cal = _calibration(N_CAL)
    t = _tensors(cal)
    rows = []
    for name in ("x", "a1", "y"):
        lo, hi = min(t[name]), max(t[name])
        d, q0 = affine_params(lo, hi)
        rows.append([name, f"{min(lo, 0):+.4f}", f"{max(hi, 0):+.4f}", f"{d:.6f}", q0,
                     f"{d * (-128 - q0):+.4f}", f"{d * (127 - q0):+.4f}"])
    return ("The calibration ranges of the previous block (min and max over the 32 inputs), widened to contain 0, turned into signed int8 codes: "
            "Delta = (hi - lo) / 255, q0 = round_even(-128 - lo / Delta), clamped to -128 .. 127. The last two columns are the real numbers "
            "of the lowest and highest code, Delta (-128 - q0) and Delta (127 - q0). h1 is not stored (the rectifier is fused into layer 1), so it has no row.\n\n"
            + _table(["tensor", "lo", "hi", "Delta", "q0", "code -128 means", "code 127 means"], rows))


def weight_codes():
    w1, b1, w2, b2 = network()
    cal = _calibration(N_CAL)
    m = QuantizedMLP(cal)
    rows = []
    for name, codes, deltas, ws in (("W1", m.c1, m.dw1, w1), ("W2", m.c2, m.dw2, w2)):
        for o, row in enumerate(codes):
            rows.append([f"{name}[{o}]", f"{max(abs(v) for v in ws[o]):.2f}", f"{deltas[o]:.6f}", " ".join(f"{c:+d}" for c in row)])
    brow = []
    for layer, bq, b, d_in, dws in (("1", m.bq1, b1, m.dx, m.dw1), ("2", m.bq2, b2, m.da, m.dw2)):
        for o in range(len(bq)):
            brow.append([f"b{layer}[{o}]", f"{b[o]:+.2f}", f"{d_in * dws[o]:.3e}", bq[o]])
    return ("Per-output-channel symmetric int8 weights, Delta_o = max(abs(row)) / 127, code = round_even(w / Delta_o). Biases are stored as int32 codes at the scale "
            "of the accumulator, Delta_in Delta_o: bq = round_even(b / (Delta_in Delta_o)), with Delta_in = Delta_x for layer 1 and Delta_a for layer 2 "
            "(values of the activation table above).\n\n"
            + _table(["row", "max abs", "Delta_o", "codes"], rows, "lrrl") + "\n\n"
            + _table(["bias", "value", "accumulator scale", "bq"], brow, "lrrr"))


def fixed_point_table():
    cal = _calibration(N_CAL)
    m = QuantizedMLP(cal)
    rows = []
    for layer, ms, fs in (("1", m.m1, m.f1), ("2", m.m2, m.f2)):
        for o, (M, (mult, sh)) in enumerate(zip(ms, fs)):
            approx = mult / 2 ** sh
            rows.append([f"layer {layer}, channel {o}", f"{M:.8f}", sh, mult, f"{approx:.8f}", f"{abs(approx - M) / M:.2e}"])
    return ("The requantization factor of every output channel, M = Delta_in Delta_o / Delta_out, as a 15-bit multiplier and a shift: "
            "shift = exponent_of(32767 / M), multiplier = round_even(M 2^shift), so that 16384 <= multiplier <= 32767. "
            "'relative error' is the absolute difference of multiplier 2^-shift and M, divided by M.\n\n"
            + _table(["channel", "M", "shift", "multiplier", "multiplier / 2^shift", "relative error"], rows))


def one_input():
    cal = _calibration(N_CAL)
    m = QuantizedMLP(cal)
    qx, acc1, a, acc2, y, _ = m.run(EXAMPLE)
    ref_unfolded1 = [sum((qx[i] - m.qx) * m.c1[o][i] for i in range(IN)) + m.bq1[o] for o in range(HID)]
    ref_unfolded2 = [sum((a[i] - m.qa) * m.c2[o][i] for i in range(HID)) + m.bq2[o] for o in range(OUT)]
    h1, a1, yf = forward(EXAMPLE)
    rows = [
        ["input codes q_x", f"{qx}"],
        ["accumulators of layer 1, raw codes + folded bias", f"{acc1}"],
        ["the same, (q_x - q0_x) codes + bias code", f"{ref_unfolded1}"],
        ["after relu, requantized: a1 codes", f"{a}"],
        ["a1 decoded, Delta_a (q - q0_a)", _v([m.da * (q - m.qa) for q in a], 4)],
        ["float a1", _v(a1, 4)],
        ["accumulators of layer 2, raw codes + folded bias", f"{acc2}"],
        ["the same, (q - q0_a) codes + bias code", f"{ref_unfolded2}"],
        ["y codes", f"{y}"],
        ["y decoded, Delta_y (q - q0_y)", _v(m.decode(y), 4)],
        ["float y", _v(yf, 4)],
        ["error, decoded - float", _v([p - q for p, q in zip(m.decode(y), yf)], 4)],
    ]
    return (f"The input of the first example, x = {_v(EXAMPLE, 3)}, through the network quantized from the 32-input calibration set "
            f"(Delta_x = {m.dx:.6f}, q0_x = {m.qx}; Delta_a = {m.da:.6f}, q0_a = {m.qa}; Delta_y = {m.dy:.6f}, q0_y = {m.qy}), "
            f"one multiplier and shift per output channel. Folded bias of layer 1: {m.fb1}; of layer 2: {m.fb2}.\n\n"
            + _table(["step", "value"], rows, "ll"))


def exactness():
    cal = _calibration(N_CAL)
    tests = _inputs("npu.test", N_TEST)
    m = QuantizedMLP(cal)
    same = diff_ideal = total = 0
    max_a1 = max_a2 = 0
    for x in tests:
        _, acc1, a, acc2, y, _ = m.run(x)
        same += m.reference(x) == y
        max_a1 = max(max_a1, max(abs(v) for v in acc1))
        max_a2 = max(max_a2, max(abs(v) for v in acc2))
        # ideal requantizer: the real factor, no fixed point
        a_ideal = [_clamp(_rne(m.m1[o] * max(acc1[o], 0)) + m.qa, -128, 127) for o in range(HID)]
        acc2_ideal = [sum(a_ideal[i] * m.c2[o][i] for i in range(HID)) + m.fb2[o] for o in range(OUT)]
        y_ideal = [_clamp(_rne(m.m2[o] * acc2_ideal[o]) + m.qy, -128, 127) for o in range(OUT)]
        diff_ideal += sum(1 for p, q in zip(y_ideal, y) if p != q)
        total += OUT
    sq, worst, clamped = _sqnr_of(m, tests)
    ymax = max(abs(v) for x in tests for v in forward(x)[2])
    clean = [x for x in tests if m.run(x)[5] == 0]
    worst_clean = max(abs(a - b) for x in clean for a, b in zip(m.decode(m.run(x)[4]), forward(x)[2]))
    rows = [
        ["test inputs", N_TEST],
        ["outputs compared (3 per input)", total],
        ["inputs where compiled form == direct reference, all 3 codes", f"{same} of {N_TEST}"],
        ["output codes that differ from an ideal real-valued requantizer", f"{diff_ideal} of {total}"],
        ["largest absolute accumulator, layer 1 / layer 2", f"{max_a1} / {max_a2}"],
        ["int32 limit 2^31 - 1", 2 ** 31 - 1],
        ["largest absolute error of the decoded output", f"{worst:.4f}"],
        ["largest absolute value of the float output", f"{ymax:.4f}"],
        ["inputs with no clamped value, and their largest absolute error", f"{len(clean)} of {N_TEST}, {worst_clean:.4f}"],
        ["half a step of y, Delta_y / 2", f"{m.dy / 2:.4f}"],
        ["SQNR of the decoded output against the float output, dB", f"{sq:.2f}"],
        ["values clamped when quantized (inputs, a1, y)", f"{100 * clamped:.2f}%"],
    ]
    return (f"The network quantized from the 32-input calibration set run on {N_TEST} test inputs, x_i = round(normal(), 3) drawn in order, "
            f"{_seed_line('npu.test')}. 'Compiled form' is the integer program of the chapter (raw codes, folded bias, fixed-point requantizer); "
            f"'direct reference' subtracts the zero point before each product and rounds the requantization as an exact rational: with the same integer multiplier and shift as the compiled form, round half to even of (multiplier x accumulator) / 2^shift (so the equality in the table tests the folding of the zero points into the bias and the integer arithmetic, not the approximation of M by a 15-bit multiplier). 'Ideal requantizer': the same with the real factor M in place of the multiplier and the shift, rounded half to even.\n\n"
            + _table(["quantity", "value"], rows, "lr"))


def calibration_curve(sizes=(1, 2, 4, 8, 16, 32, 64, 128)):
    """[(n, SQNR dB representative, clamped share, SQNR dB narrow, clamped share)]: each calibration set is the first n
    inputs of its stream; the test set is the one of `exactness`."""
    tests = _inputs("npu.test", N_TEST)
    rep = _calibration(max(sizes))
    nar = _inputs("npu.calibration_narrow", max(sizes), 0.25)
    out = []
    for n in sizes:
        a = _sqnr_of(QuantizedMLP(rep[:n]), tests)
        b = _sqnr_of(QuantizedMLP(nar[:n]), tests)
        out.append((n, a[0], a[2], b[0], b[2]))
    return out


def calibration_size():
    rows = [[n, f"{a:.2f}", f"{100 * ca:.1f}%", f"{b:.2f}", f"{100 * cb:.1f}%"] for n, a, ca, b, cb in calibration_curve()]
    return ("Output SQNR (dB, against the float network) on the 256 test inputs of the previous block, as a function of the size of the calibration set. "
            f"'representative' sets are the first n inputs of the calibration stream (128 inputs, " + _seed_line('npu.calibration') + "); "
            f"'narrow' sets are the first n inputs of a stream of x_i = round(0.25 normal(), 3), {_seed_line('npu.calibration_narrow')} "
            "(an input a quarter of the size of the deployed ones). 'clamped' is the share of the quantized values (inputs, hidden activations, outputs) "
            "that fell outside the calibrated range and were clamped.\n\n"
            + _table(["calibration inputs", "SQNR representative", "clamped", "SQNR narrow", "clamped"], rows))


TAILS = (0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05)


def clipping_curve():
    """[(tail, clamped share, median relative error, 99th percentile of it, SQNR dB)] on spiky inputs."""
    cal = _spiky("npu.spiky_calibration", 512)
    tests = _spiky("npu.spiky_test", 1000)
    out = []
    for tail in TAILS:
        m = QuantizedMLP(cal, tail=tail)
        rel = []
        for x in tests:
            yq = m.decode(m.run(x)[4])
            y = forward(x, m.net)[2]
            rel.append(math.sqrt(sum((a - b) ** 2 for a, b in zip(yq, y))) / math.sqrt(sum(b * b for b in y)))
        sq, _, clamped = _sqnr_of(m, tests)
        out.append((tail, clamped, _percentile(rel, 0.5), _percentile(rel, 0.99), sq))
    return out


def clipping():
    rows = [[f"{100 * t:.2f}%", f"{100 * c:.2f}%", f"{med:.4f}", f"{p99:.4f}", f"{sq:.2f}"] for t, c, med, p99, sq in clipping_curve()]
    return ("A calibration range that ignores the tails of the recorded values. Inputs are 'spiky': for each, six normal() draws then one uniform(), "
            "and when the uniform is below 0.02 all six values are multiplied by 6; each value is then rounded once to 3 decimals, x_i = round(k z_i, 3) with k = 6 or 1 (it is the product that is rounded, not z_i beforehand). The calibration set is 512 such inputs, "
            f"{_seed_line('npu.spiky_calibration')}; the test set 1000, {_seed_line('npu.spiky_test')}. Each activation range is [percentile(v, tail), percentile(v, 1 - tail)] "
            "of the values recorded for the tensor (the percentile of the book's pseudocode page; tail 0 is the minimum and maximum). "
            "'relative error' of an input is the norm of (decoded - float) over the norm of the float output; 'median' and '99th percentile' are over the 1000 test inputs; "
            "SQNR is over all their outputs.\n\n"
            + _table(["tail clipped per end", "values clamped", "median relative error", "99th percentile of it", "SQNR dB"], rows))


def two_stage():
    cal = _calibration(N_CAL)
    tests = _inputs("npu.test", N_TEST)
    a = QuantizedMLP(cal, requant="channel")
    b = QuantizedMLP(cal, requant="two_stage")
    differ = sum(1 for x in tests for p, q in zip(a.run(x)[4], b.run(x)[4]) if p != q)
    sa, _, _ = _sqnr_of(a, tests)
    sb, _, _ = _sqnr_of(b, tests)
    q14_1, mult1, sh1 = b.f1
    q14_2, mult2, sh2 = b.f2
    rows = [
        ["layer 1 Q14 factors (16384 = 1.0)", f"{q14_1}"],
        ["layer 1 multiplier, shift", f"{mult1}, {sh1}"],
        ["layer 2 Q14 factors", f"{q14_2}"],
        ["layer 2 multiplier, shift", f"{mult2}, {sh2}"],
        ["output codes that differ from the per-channel form", f"{differ} of {len(tests) * OUT}"],
        ["SQNR per-channel multiplier and shift, dB", f"{sa:.2f}"],
        ["SQNR two-stage, dB", f"{sb:.2f}"],
    ]
    return ("The requantizer with one multiplier and shift per layer. Channel o first scales its accumulator by a 14-bit fraction "
            "c_o = round_even(Delta_o / max(Delta) 2^14) and rounds (ties to even) to an integer, then the layer multiplies by the multiplier of "
            "M = Delta_in max(Delta) / Delta_out and rounds again. Same network, calibration set and 256 test inputs as before.\n\n"
            + _table(["quantity", "value"], rows, "ll"))


# ---------------------------------------------------------------- compiler: fusion, layout, tiling


def _ceil_div(a, b):
    return -(-a // b)


def fusion_bytes():
    h, w, ic, oc, k = 40, 40, 128, 64, 1
    n_out = h * w * oc
    n_in = h * w * ic
    wts = oc * ic * k * k
    macs = n_out * ic * k * k
    i8, i32 = 1, 4
    bias = oc * i32
    designs = []
    # 4 ops: conv -> int32, + bias -> int32, relu -> int32, requantize -> int8
    r = [n_in * i8 + wts, n_out * i32 + bias, n_out * i32, n_out * i32]
    wr = [n_out * i32, n_out * i32, n_out * i32, n_out * i8]
    designs.append(("conv, bias, relu, requantize as four ops", 4, sum(r), sum(wr)))
    # 2 ops: conv+bias+relu keeping the int32 accumulator, then requantize
    r2 = [n_in * i8 + wts + bias, n_out * i32]
    wr2 = [n_out * i32, n_out * i8]
    designs.append(("conv + bias + relu, then requantize", 2, sum(r2), sum(wr2)))
    # 1 op
    designs.append(("one op: accumulator stays on chip", 1, n_in * i8 + wts + bias, n_out * i8))
    base = designs[0][2] + designs[0][3]
    rows = [[name, n, f"{r:,}", f"{w_:,}", f"{r + w_:,}", f"{base / (r + w_):.3f}"] for name, n, r, w_ in designs]
    return (f"One layer of the 320 x 320 detector of the board check: input {h} x {w} x {ic} int8, kernel {k} x {k}, {oc} output channels, stride 1. "
            f"It has {n_out:,} outputs, {wts:,} int8 weights, {oc} int32 biases, and does {macs:,} multiply-adds = {2 * macs:,} operations in every design. "
            f"An int32 accumulator is 4 bytes, an int8 activation 1. Each op reads its distinct operands once and writes its result once "
            f"(the accounting of Chapter 8; the requantization constants are not counted).\n\n"
            + _table(["design", "ops", "bytes read", "bytes written", "total bytes", "total fewer by"], rows))


LANES = 16
ALIGN = 64


def native16_bytes(h, w, c):
    """Bytes of a feature map in 16-lane planes: planes of ceil(c/16), each h w 16 bytes rounded up to 64."""
    plane = _ceil_div(h * w * LANES, ALIGN) * ALIGN
    return _ceil_div(c, LANES) * plane


def layout_bytes():
    shapes = [("detector input", 320, 320, 3), ("the same image after 2 x 2 space-to-depth", 160, 160, 12),
              ("a feature map", 160, 160, 32), ("a head output", 40, 40, 255), ("a deep feature map", 20, 20, 256)]
    rows = []
    for name, h, w, c in shapes:
        logical = h * w * c
        n16 = native16_bytes(h, w, c)
        rows.append([name, f"{h} x {w} x {c}", f"{logical:,}", f"{n16:,}", f"{n16 / logical:.3f}"])
    saving = native16_bytes(320, 320, 3) - native16_bytes(160, 160, 12)
    return ("Bytes of an int8 feature map stored densely (h w c) and in 16-lane planes: ceil(c / 16) planes, each h w 16 bytes rounded up to a multiple of 64. "
            f"The first two rows hold the same pixels; the difference between their 16-lane sizes is {saving:,} bytes.\n\n"
            + _table(["tensor", "h x w x c", "dense bytes", "16-lane bytes", "ratio"], rows))


INPUT_ATOMS = 6144


def bands(ih, iw, c, k, stride, pad, capacity=INPUT_ATOMS):
    """Split a convolution into bands of output rows whose input rows fit `capacity` atoms of 16 bytes.
    Returns a list of (out_lo, out_hi, in_lo, in_hi). Band starts must sit on a multiple of 4 atoms (64 bytes) in
    the input and in the output plane."""
    t, l, b, r = pad
    tiles = _ceil_div(c, LANES)
    out_h = (ih + t + b - k) // stride + 1
    out_w = (iw + l + r - k) // stride + 1
    if ih * iw * tiles <= capacity:
        return [(0, out_h, 0, ih)]
    max_rows = capacity // (iw * tiles)
    if max_rows < k:
        raise ValueError("one kernel window of rows does not fit")
    result = []
    oy0 = 0
    while oy0 < out_h:
        for oy1 in range(out_h, oy0, -1):
            first = oy0 * stride - t
            last = (oy1 - 1) * stride - t + k - 1
            iy0, iy1 = max(first, 0), min(last + 1, ih)
            if iy1 - iy0 > max_rows or (iy0 * iw) % 4 or (oy0 * out_w) % 4:
                continue
            if oy1 < out_h:
                nxt = max(oy1 * stride - t, 0)
                if (nxt * iw) % 4 or (oy1 * out_w) % 4:
                    continue
            result.append((oy0, oy1, iy0, iy1))
            oy0 = oy1
            break
        else:
            raise ValueError("no aligned tiling")
    return result


BOARD_LAYERS = [
    ("smoke test", 8, 8, 16, 16, 3, 1, (1, 1, 1, 1)),
    ("stem (kernel 2 grown to 3)", 320, 320, 3, 12, 3, 2, (0, 0, 1, 1)),
    ("conv", 160, 160, 12, 32, 3, 1, (1, 1, 1, 1)),
    ("conv", 160, 160, 32, 64, 3, 2, (1, 1, 1, 1)),
    ("conv", 40, 40, 128, 64, 1, 1, (0, 0, 0, 0)),
    ("conv", 20, 20, 256, 256, 3, 1, (1, 1, 1, 1)),
    ("conv", 10, 10, 1024, 512, 1, 1, (0, 0, 0, 0)),
    ("head", 40, 40, 128, 255, 1, 1, (0, 0, 0, 0)),
]


def tiling():
    rows = []
    for name, ih, iw, c, oc, k, s, pad in BOARD_LAYERS:
        tiles = _ceil_div(c, LANES)
        atoms = ih * iw * tiles
        bs = bands(ih, iw, c, k, s, pad)
        read_rows = sum(b[3] - b[2] for b in bs)
        lanes = tiles * LANES
        wbytes = oc * lanes * k * k
        rows.append([f"{ih} x {iw} x {c} -> {oc}, k{k} s{s}", f"{atoms:,}", "yes" if atoms <= INPUT_ATOMS else "no", len(bs),
                     f"{read_rows / ih:.3f}", f"{wbytes:,}"])
    return ("Height tiling of eight convolutions against an input buffer of 6144 atoms of 16 bytes (the capacity stated by the source of `crates/rknpu`, "
            "98,304 bytes). A layer is written h x w x c -> oc, kernel k, stride s; padding is 1 for 3 x 3 kernels except the stem, and 0 for the 1 x 1 kernels, whose padding is [0, 0, 1, 1] (top, left, bottom, right). "
            "'atoms' is h w ceil(c / 16); 'rows read' is the input rows summed over the bands divided by h (1.000 = no row is read twice); "
            "'weight bytes' is oc ceil(c / 16) 16 k^2.\n\n"
            + _table(["layer", "input atoms", "fits", "bands", "rows read", "weight bytes"], rows))


def tiling_detail():
    ih, iw, c, k, s, pad = 10, 10, 1024, 1, 1, (0, 0, 0, 0)
    bs = bands(ih, iw, c, k, s, pad)
    max_rows = INPUT_ATOMS // (iw * _ceil_div(c, LANES))
    rows = [[i, f"{b[0]}..{b[1] - 1}", f"{b[2]}..{b[3] - 1}", (b[2] * iw) % 4, (b[0] * iw) % 4] for i, b in enumerate(bs)]
    return (f"The layer 10 x 10 x 1024 -> 512 with kernel 1 has 10 x 10 x 64 = 6400 atoms, more than 6144. A row of the input is 10 x 64 = 640 atoms, so at most "
            f"{max_rows} rows fit. Nine output rows would need the band starting at row 9 to begin at atom 9 x 10 = 90, which is not a multiple of 4, so the first band is "
            f"cut at 8 rows.\n\n"
            + _table(["band", "output rows", "input rows", "start atom of input band mod 4", "start atom of output band mod 4"], rows))


def stem_bands():
    return len(bands(320, 320, 3, 3, 2, (0, 0, 1, 1)))


# ---------------------------------------------------------------- compiler: memory plan

# (name, bytes, first op, last op): a graph of five ops on 16-lane tensors; see memory_plan()
PLAN_OPS = [("conv 3 x 3, 16 -> 32", "t0", "t1"), ("conv 3 x 3, 32 -> 32, stride 2", "t1", "t2"),
            ("conv 1 x 1, 32 -> 32", "t2", "t3"), ("add", "t2,t3", "t4"), ("conv 1 x 1, 32 -> 16", "t4", "t5")]
PLAN_SHAPES = {"t0": (16, 16, 16), "t1": (16, 16, 32), "t2": (8, 8, 32), "t3": (8, 8, 32), "t4": (8, 8, 32), "t5": (8, 8, 16)}


def lifetimes():
    """{tensor: (first, last)}: an op touches its inputs and its output; the graph input is touched at op 0 and
    the graph output after the last op. Intervals are inclusive."""
    live = {}

    def touch(t, at):
        a, b = live.get(t, (at, at))
        live[t] = (min(a, at), max(b, at))
    touch("t0", 0)
    for i, (_, ins, out) in enumerate(PLAN_OPS):
        touch(out, i)
        for t in ins.split(","):
            touch(t, i)
    touch("t5", len(PLAN_OPS))
    return live


def plan_memory(sizes, live):
    """First fit, largest first: offsets such that tensors with intersecting lifetimes do not overlap."""
    order = sorted(sizes, key=lambda t: -sizes[t])
    placed, offset = [], {}
    for t in order:
        a, b = live[t]
        busy = sorted((offset[u], offset[u] + sizes[u]) for u in placed if live[u][0] <= b and a <= live[u][1])
        at = 0
        for lo, hi in busy:
            if at + sizes[t] <= lo:
                break
            at = max(at, hi)
        offset[t] = at
        placed.append(t)
    return offset


def memory_plan_data():
    live = lifetimes()
    sizes = {t: native16_bytes(*s) for t, s in PLAN_SHAPES.items()}
    offset = plan_memory(sizes, live)
    n_ops = len(PLAN_OPS)
    at_op = [sum(sizes[t] for t in sizes if live[t][0] <= i <= live[t][1]) for i in range(n_ops + 1)]
    return live, sizes, offset, at_op


def memory_plan():
    live, sizes, offset, at_op = memory_plan_data()
    rows = [[t, "x".join(map(str, PLAN_SHAPES[t])), f"{sizes[t]:,}", f"{live[t][0]}..{live[t][1]}", f"{offset[t]:,}",
             f"{offset[t] + sizes[t]:,}"] for t in sorted(sizes)]
    total = sum(sizes.values())
    top = max(offset[t] + sizes[t] for t in sizes)
    live_rows = [[i, ", ".join(t for t in sorted(sizes) if live[t][0] <= i <= live[t][1]), f"{at_op[i]:,}"] for i in range(len(PLAN_OPS) + 1)]
    ops = _table(["op", "computation", "reads", "writes"], [[i, n, a, b] for i, (n, a, b) in enumerate(PLAN_OPS)], "llll")
    return (f"A graph of five ops on 16-lane tensors (sizes by the 16-lane rule of the layout block, in bytes). Lifetimes are inclusive op intervals: an op touches the tensors it reads and the one it writes, "
            f"the input is touched by op 0 and the output after the last op (op 5).\n\n{ops}\n\n"
            + _table(["tensor", "h x w x c", "bytes", "lifetime (ops)", "offset", "end"], rows) + "\n\n"
            + _table(["op", "tensors alive", "bytes alive"], live_rows, "llr") + "\n\n"
            f"Without reuse the plan needs {total:,} bytes (every tensor its own bytes). The largest total alive at one op is {max(at_op):,} bytes, a lower bound for any plan. "
            f"First fit, largest first, ends at byte {top:,}.")


# ---------------------------------------------------------------- compiler: command stream

TARGET = {"CNA": 0x0201, "CORE": 0x0801, "DPU": 0x1001}
PROFILE_WORDS, TAIL_WORDS = 126, 4


def command_word(target, register, value):
    """64-bit command word: target in bits 63..48, value in bits 47..16, register in bits 15..0."""
    return (target << 48) | ((value & 0xFFFFFFFF) << 16) | register


def command_stream():
    cal = _calibration(N_CAL)
    m = QuantizedMLP(cal, requant="two_stage")
    _, mult, shift = m.f1
    ic, oc, k = IN, HID, 1
    lanes = _ceil_div(ic, LANES) * LANES
    oc_lanes = _ceil_div(oc, LANES) * LANES
    regs = [
        ("CNA", "CNA_DATA_SIZE0", 0x1020, (1 << 16) | 1, "input width 1 and height 1: width shifted left by 16, or the height"),
        ("CNA", "CNA_DATA_SIZE1", 0x1024, ((ic - 1) << 16) | lanes, f"channels - 1 shifted left by 16, or the lanes: {ic} channels in {lanes} lanes"),
        ("CNA", "CNA_WEIGHT_SIZE0", 0x1030, oc * lanes * k * k, f"weight table bytes: {oc} x {lanes} x {k * k}"),
        ("CORE", "CORE_DATAOUT_SIZE_1", 0x3018, oc_lanes - 1, f"output lanes - 1: {oc_lanes} - 1"),
        ("DPU", "DPU_OUT_CVT_OFFSET", 0x4080, m.qa, f"output zero point q0_a = {m.qa}, as a 32-bit two's complement value"),
        ("DPU", "DPU_OUT_CVT_SCALE", 0x4084, mult, "layer multiplier of the requantizer"),
        ("DPU", "DPU_OUT_CVT_SHIFT", 0x4088, shift, "layer shift of the requantizer"),
    ]
    rows = [[name, f"0x{reg:04x}", f"0x{(val & 0xFFFFFFFF):08x}", f"0x{command_word(TARGET[eng], reg, val):016x}", note]
            for eng, name, reg, val, note in regs]
    words = PROFILE_WORDS + TAIL_WORDS
    nbytes = _ceil_div(words * 8, ALIGN) * ALIGN
    return (f"Layer 1 of the tiny network as a convolution of kernel 1 on a 1 x 1 image of {ic} channels, {oc} output channels, with the two-stage requantizer "
            f"(multiplier {mult}, shift {shift}). Seven of the registers one task writes, with the register addresses, the targets (CNA 0x0201, CORE 0x0801, DPU 0x1001) and the "
            f"field formulas of `crates/rknpu/src/conv.rs`, encoded as target << 48 | value << 16 | register.\n\n"
            + _table(["register", "address", "value", "command word", "meaning"], rows, "lllll") + "\n\n"
            f"A task writes the {PROFILE_WORDS} registers of the profile (the source's count) and then {TAIL_WORDS} tail words that link to the next task: {words} words of 8 bytes = "
            f"{words * 8} bytes, rounded up to a multiple of {ALIGN}: {nbytes} bytes per task. A network of 2 such tasks is {2 * nbytes:,} bytes of commands; a network of 18 tasks "
            f"(the stem above) {18 * nbytes:,}; the 172 tasks the commit cited below counts, {172 * nbytes:,}.")


def pipeline_sizes():
    """Bytes of the tiny network at each stage of the figure."""
    n_init = HID * IN + HID + OUT * HID + OUT
    fp32 = 4 * n_init
    w_i8 = HID * IN + OUT * HID
    bias_i32 = 4 * (HID + OUT)
    requant = (HID + OUT) * 3
    quantized = w_i8 + bias_i32 + requant
    words = PROFILE_WORDS + TAIL_WORDS
    task = _ceil_div(words * 8, ALIGN) * ALIGN
    return {"fp32": fp32, "quantized": quantized, "task": task, "tasks": 2, "stream": 2 * task}
