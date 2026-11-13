# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Counting Parameters, FLOPs and Memory".

Plain Python 3 (integers for every count, IEEE double precision for ratios and the regression), standard library only,
written from the chapter's own description and independent of any engine code. Earlier worked modules are imported, never
copied: `modern_decoder` (the decoder shapes of Volume VI, Chapter 1 and their parameter and decode-FLOP counts, used here as
the cross-check of the general formula), `mixture_of_experts` (Chapter 4's counts of Qwen3.5-35B-A3B).

What this module adds:

* the parameters of a stated small decoder tensor by tensor, the general formula P that replaces the tensor list, and its
  agreement with the counts of Chapter 1;
* the same count for a mixture-of-experts feed-forward layer (total and active) and for a stack of two layer types (the text
  tower of Gemma-4, as `crates/gemma4` lists its tensors), and for the packed diffusion transformer of MiniMax-H3 (as
  `crates/minimaxh3` lists its tensors), each checked against the numbers the crates pin in their tests;
* FLOPs per token, 2 N_act + 4 L H d_h T, with the attention-to-parameter ratio as the context grows;
* inference memory (weights by format, the KV cache per layer type, the prefill workspace, the allocator headroom rule) and
  training memory (bytes per parameter by recipe, saved activations per token with and without recomputation);
* a log-log regression of a power law on stated synthetic points, and the compute-optimal split of a budget.

Every random quantity comes from the book's SplitMix64 (`bookrng.generator(name)`); each block prints the seed. Sums are exact
integer sums or plain left-to-right additions. Sizes in bytes use GiB = 2^30 bytes unless a table says GB (10^9).

Swedish Embedded AB implements solutions for sizing language models before they are run. If your team needs expertise in
parameter, FLOP and memory budgeting for large models, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math

import bookrng
import mixture_of_experts as moe
import modern_decoder as md

GIB = 2 ** 30

# ------------------------------------------------------------------------------------------------ formatting


def _int(n):
    return f"{n:,}"


def _fx(v, d=3):
    out = f"{v:.{d}f}"
    return out[1:] if out.startswith("-") and not any(c in "123456789" for c in out) else out


def _sig(v, d=4):
    return f"{v:.{d}g}"


def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _seed_line(name):
    return f"Generator: the book's SplitMix64, seed {bookrng.seed_of(name):#018x} (`seed_of` of the name `{name}`)."


def _gib(b):
    return b / GIB


# ------------------------------------------------------------------------------------------------ the stated small decoder

# The decoder of the opening sections: vocabulary 1,000, width 64, 4 layers, 8 query heads and 2 key-value heads of width 8
# (so H d_h = D), a SwiGLU network of hidden width 160, RMS norms with a gain only, a gain of d_h numbers for the QK-norm of
# queries and again for keys, no biases, a separate output matrix.
SMALL = dict(V=1000, D=64, L=4, H=8, hkv=2, dh=8, F=160)


def small_shape(tied=False):
    s = SMALL
    return md.shape("small decoder", s["V"], s["D"], s["L"], s["H"], s["hkv"], s["dh"], s["F"], 3, False, False, 0, tied, True, 1024)


def parameters_formula(V, D, L, H, hkv, dh, F, untied=True, qk_norm=True):
    """P = V D (1 + untied) + D + L (2 D H d_h + 2 D h_kv d_h + 3 D F + 2 d_h + 2 D); the 2 d_h only with QK-norm."""
    per_layer = 2 * D * H * dh + 2 * D * hkv * dh + 3 * D * F + (2 * dh if qk_norm else 0) + 2 * D
    return V * D * (2 if untied else 1) + D + L * per_layer


def matmul_parameters(V, D, L, H, hkv, dh, F):
    """N_p: the parameters that take part in a matrix product for one token (norm gains do not; the head counts once, the
    embedding lookup not at all)."""
    return L * (2 * D * H * dh + 2 * D * hkv * dh + 3 * D * F) + V * D


def small_tensor_table():
    s = SMALL
    V, D, L, H, hkv, dh, F = s["V"], s["D"], s["L"], s["H"], s["hkv"], s["dh"], s["F"]
    q, kv = H * dh, hkv * dh
    layer = [("input norm gain", [D]), ("W_Q", [D, q]), ("W_K", [D, kv]), ("W_V", [D, kv]), ("W_O", [q, D]), ("QK-norm gain of queries", [dh]),
             ("QK-norm gain of keys", [dh]), ("post-attention norm gain", [D]), ("W_gate", [D, F]), ("W_up", [D, F]), ("W_down", [F, D])]

    def numel(shape):
        n = 1
        for x in shape:
            n *= x
        return n

    rows = [("token embedding table", [V, D], numel([V, D]), 1, numel([V, D]))]
    per_layer = 0
    for name, shape in layer:
        rows.append((name, shape, numel(shape), L, numel(shape) * L))
        per_layer += numel(shape)
    rows.append(("final norm gain", [D], D, 1, D))
    rows.append(("output matrix (untied head)", [V, D], numel([V, D]), 1, numel([V, D])))
    total = sum(r[4] for r in rows)
    assert total == V * D + L * per_layer + D + V * D == parameters_formula(V, D, L, H, hkv, dh, F)
    body = [(n, "[" + ", ".join(str(x) for x in sh) + "]", _int(c), _int(t), _int(c * t)) for n, sh, c, t, _ in rows]
    tied = total - V * D
    return (f"The small decoder: vocabulary V = {V}, width D = {D}, L = {L} layers, H = {H} query heads and h_kv = {hkv} key-value heads of "
            f"width d_h = {dh} (so H d_h = {q} = D and a key or value matrix has {kv} columns), a SwiGLU network of hidden width F = {F}, RMS "
            "norms with a gain and no shift, QK-norm gains of d_h numbers for queries and again for keys, no biases and a separate output "
            f"matrix. A layer holds {_int(per_layer)} numbers.\n\n"
            + _table(["tensor", "shape", "numbers in one", "how many", "numbers in all"], body, "llrrr")
            + f"\n\nTotal {_int(total)}; with the head tied to the embedding table the output matrix goes and the total is {_int(tied)}.")


def formula_agrees():
    s_small = small_shape()
    shapes = [s_small, md.QWEN3_0_6B, md.QWEN3_8B, md.LLAMA2_13B]
    rows = []
    for s in shapes:
        formula = parameters_formula(s["V"], s["D"], s["L"], s["H"], s["hkv"], s["dh"], s["F"], untied=not s["tied"], qk_norm=s["qkn"])
        counted = md.parameters(s)
        lead = s["V"] * s["D"] * (1 if s["tied"] else 2) + s["L"] * (2 * s["D"] * s["H"] * s["dh"] + 2 * s["D"] * s["hkv"] * s["dh"] + 3 * s["D"] * s["F"])
        rows.append((s["name"], _int(formula), _int(counted), "yes" if formula == counted else "NO", _int(lead),
                     _fx(100.0 * (counted - lead) / counted, 3) + "%"))
    return ("The formula P = V D (1 + e_h) + D + L (2 D H d_h + 2 D h_kv d_h + 3 D F + 2 d_h + 2 D), with e_h = 1 for a separate output matrix "
            "and e_h = 0 for a tied one and the 2 d_h only with QK-norm, against the counts of Chapter 1 (`modern_decoder.parameters`, "
            "tensor by tensor, from the shapes of that chapter's table). The last two columns drop the norm gains and the final norm: "
            "the leading terms V D (1 + e_h) + L (2 D H d_h + 2 D h_kv d_h + 3 D F) and the share of P that they leave out.\n\n"
            + _table(["decoder", "P by the formula", "P by the tensors of Chapter 1", "equal", "leading terms", "left out"], rows)
            + "\n\nThe shapes that went in (Qwen3 0.6B ties its head, the others do not; only Llama-2 13B has no QK-norm; all four use a "
            "SwiGLU network and no biases):\n\n"
            + _table(["decoder", "V", "D", "L", "H", "h_kv", "d_h", "F", "head", "QK-norm"],
                     [(s["name"], _int(s["V"]), _int(s["D"]), s["L"], s["H"], s["hkv"], s["dh"], _int(s["F"]),
                       "tied" if s["tied"] else "separate", "yes" if s["qkn"] else "no") for s in shapes], "l" + "r" * 7 + "ll"))


# ------------------------------------------------------------------------------------------------ mixtures and the unified count

# The small decoder with its network replaced by a mixture: E = 8 routed experts of hidden width 40 (so the experts together hold
# the 320 = 2 F of hidden width), k = 2 used by a token, one shared expert of the same width, a router of D x E.
MOE = dict(E=8, k=2, Fe=40, shared=1)


def small_mixture_counts():
    s, m = SMALL, MOE
    V, D, L, H, hkv, dh, F = s["V"], s["D"], s["L"], s["H"], s["hkv"], s["dh"], s["F"]
    attn = 2 * D * H * dh + 2 * D * hkv * dh
    ffn_dense_p, ffn_dense_f = moe.layer_costs(D, F, 1, 1)
    ffn_moe_p, ffn_moe_f = moe.layer_costs(D, m["Fe"], m["E"], m["k"], m["shared"])
    norms = 2 * D + 2 * dh
    rows = []
    for label, ffn_p, ffn_f in (("dense network, F = 160", ffn_dense_p, ffn_dense_f),
                                ("mixture: 8 experts of width 40, 2 used, 1 shared, router", ffn_moe_p, ffn_moe_f)):
        layer_p = attn + norms + ffn_p
        P = V * D * 2 + D + L * layer_p
        # N_act: the products of one token: attention matrices, the network's used weights (FLOPs / 2), the head.
        n_act = L * (attn + ffn_f // 2) + V * D
        rows.append((label, _int(ffn_p), _int(ffn_f // 2), _int(P), _int(n_act), _fx(P / n_act, 3)))
    return (f"The small decoder with its SwiGLU network (D = {D}, F = {F}) replaced by a mixture of E = {m['E']} experts of hidden width "
            f"{m['Fe']}, k = {m['k']} used per token, {m['shared']} shared expert and a router of D by E weights (an expert is 3 D F_e numbers; "
            "`mixture_of_experts.layer_costs` counts the network). N_act is the number of parameters that take part in the products of "
            "one token: the attention matrices, the network's used weights and the head; the embedding lookup is not a product.\n\n"
            + _table(["feed-forward layer", "parameters of the network", "used per token", "P (whole model)", "N_act", "P / N_act"], rows))


def earlier_counts_unified():
    """GPT-2 small, Qwen3 8B, Qwen3.5-35B-A3B, as Chapters 1 and 4 counted them, in the vocabulary of this chapter."""
    rows = []
    for s in (md.GPT2, md.QWEN3_8B):
        P = md.parameters(s)
        n_act = s["L"] * (s["D"] * s["H"] * s["dh"] + 2 * s["D"] * s["hkv"] * s["dh"] + s["H"] * s["dh"] * s["D"] + s["mlp"] * s["D"] * s["F"]) + s["V"] * s["D"]
        rows.append((s["name"], "dense", _int(P), _int(n_act), _fx(P / n_act, 3), _int(2 * n_act)))
    c = moe.qwen_counts_data()
    rows.append(("Qwen3.5-35B-A3B", "mixture, hybrid layers", _int(c["total"]), _int(c["used"]), _fx(c["total"] / c["used"], 3), _int(2 * c["used"])))
    return ("Three models of earlier chapters in the quantities of this one: P stored parameters, N_act the parameters a token's "
            "products touch (Chapter 1's count of the matrices and the head, Chapter 4's `qwen_counts_data` for the mixture), and 2 N_act, "
            "the parameter term of the forward FLOPs per token. For GPT-2 small the product with the tied head is counted once, "
            "so N_act is the N_p of Volume V, Chapter 10.\n\n"
            + _table(["model", "layers", "P", "N_act", "P / N_act", "2 N_act"], rows, "llrrrr"))


# ------------------------------------------------------------------------------------------------ the ledger as a sum over layers

NO_WINDOW = 10 ** 18   # a window larger than any context: full attention


def ledger(layers, embedding, tail_stored, tail_used, T, b):
    """The routine of the chapter's pseudocode. A layer is (mixer, ffn_stored, ffn_used, gains, kv_numbers_per_token, window,
    score_width). Returns (P, FLOPs per token for a token that sees T keys, cache bytes of one sequence of T tokens)."""
    P = embedding + tail_stored
    n_act = tail_used
    attn = 0
    cache = 0
    for mixer, ffn_stored, ffn_used, gains, kv, window, width in layers:
        P = P + mixer + ffn_stored + gains
        n_act = n_act + mixer + ffn_used
        keys = min(T, window)
        attn = attn + 4 * width * keys
        cache = cache + kv * keys * b
    return P, 2 * n_act + attn, cache


def small_layers():
    s = SMALL
    D, H, hkv, dh, F = s["D"], s["H"], s["hkv"], s["dh"], s["F"]
    layer = (2 * D * H * dh + 2 * D * hkv * dh, 3 * D * F, 3 * D * F, 2 * D + 2 * dh, 2 * hkv * dh, NO_WINDOW, H * dh)
    return [layer] * s["L"]


def g4_layers():
    g = G4
    out = []
    for i in range(g["L"]):
        full = g4_layer_is_full(i)
        t = g4_layer_tensors(full)
        dh, hkv = (g["dh_g"], g["hkv_g"]) if full else (g["dh_s"], g["hkv_s"])
        mixer = sum(n for name, n in t if name in ("q_proj", "k_proj", "v_proj", "o_proj"))
        ffn = sum(n for name, n in t if name in ("gate_proj", "up_proj", "down_proj"))
        gains = sum(n for name, n in t) - mixer - ffn
        out.append((mixer, ffn, ffn, gains, 2 * hkv * dh, NO_WINDOW if full else g["window"], g["H"] * dh))
    return out


def ledger_agrees():
    s = SMALL
    rows = []
    T = 1024
    P, f, c = ledger(small_layers(), s["V"] * s["D"], s["V"] * s["D"] + s["D"], s["V"] * s["D"], T, 2)
    n_p = matmul_parameters(s["V"], s["D"], s["L"], s["H"], s["hkv"], s["dh"], s["F"])
    ok = (P, f, c) == (parameters_formula(s["V"], s["D"], s["L"], s["H"], s["hkv"], s["dh"], s["F"]),
                       flops_per_token(n_p, s["L"], s["H"], s["dh"], T), s["L"] * 2 * s["hkv"] * s["dh"] * T * 2)
    rows.append(("small decoder, T = 1,024, bf16", _int(P), _int(f), _int(c), "yes" if ok else "NO"))
    g = G4
    agg_in = g["D"] * g["agg_states"]
    tail_used = agg_in * (g["agg_video"] + g["agg_audio"])
    tail_stored = tail_used + g["agg_video"] + g["agg_audio"]
    for T in (8192, 131072):
        P, f, c = ledger(g4_layers(), g["V"] * g["D"], tail_stored + g["D"], tail_used, T, 2)
        pt, agg, sl, fu = g4_flops(T)
        ok = (P, f, c) == (g4_total(), pt + agg + sl + fu, g4_kv_bytes(T))
        rows.append((f"Gemma-4 text tower, T = {_int(T)}, bf16", _int(P), _int(f), _int(c), "yes" if ok else "NO"))
    return ("The routine `ledger` of the chapter, run on a list of layer records, against the closed forms of the earlier tables (the parameter formula, "
            "2 N_p + 4 L H d_h T, and the cache formula with the window). FLOPs per token are those of a token that sees T keys. For Gemma-4 the 40 sliding "
            f"layers carry a window of {_int(G4['window'])} and the 8 full layers none; the tail is the aggregate projection (used: its weights; stored: weights and biases) and "
            "the final norm gain.\n\n"
            + _table(["case", "P", "FLOPs per token", "cache bytes", "equal to the closed forms"], rows, "lrrrr"))


# ------------------------------------------------------------------------------------------------ Gemma-4's text tower

# `crates/gemma4/src/config.rs`, `Gemma4Config::gemma4_12b`: 48 layers; the layer i is full attention when (i + 1) is a multiple
# of 6 and sliding-window attention otherwise.
G4 = dict(V=262144, D=3840, F=15360, L=48, H=16, hkv_s=8, dh_s=256, hkv_g=1, dh_g=512, window=1024, pattern=6, k_eq_v=True,
          agg_states=49, agg_video=4096, agg_audio=2048)


def g4_layer_is_full(i):
    return (i + 1) % G4["pattern"] == 0


def g4_layer_tensors(full):
    """(name, numbers) of the tensors of one decoder layer, the order of the crate's tensor manifest."""
    g = G4
    D, F, H = g["D"], g["F"], g["H"]
    dh, hkv = (g["dh_g"], g["hkv_g"]) if full else (g["dh_s"], g["hkv_s"])
    q, kv = H * dh, hkv * dh
    t = [("input_layernorm", D), ("post_attention_layernorm", D), ("pre_feedforward_layernorm", D), ("post_feedforward_layernorm", D),
         ("layer_scalar", 1), ("q_proj", q * D), ("k_proj", kv * D)]
    if not (full and g["k_eq_v"]):
        t.append(("v_proj", kv * D))
    t += [("o_proj", D * q), ("q_norm", dh), ("k_norm", dh), ("gate_proj", F * D), ("up_proj", F * D), ("down_proj", D * F)]
    return t


def g4_groups():
    g = G4
    D = g["D"]
    attn_names = {"q_proj", "k_proj", "v_proj", "o_proj"}
    groups = {"embedding table": g["V"] * D, "attention matrices": 0, "QK-norm gains": 0, "feed-forward matrices": 0,
              "layer norms and scalars": D, "aggregate projection": 0}
    n_tensors = 2
    for i in range(g["L"]):
        t = g4_layer_tensors(g4_layer_is_full(i))
        n_tensors += len(t)
        for name, n in t:
            if name in attn_names:
                groups["attention matrices"] += n
            elif name in ("q_norm", "k_norm"):
                groups["QK-norm gains"] += n
            elif name in ("gate_proj", "up_proj", "down_proj"):
                groups["feed-forward matrices"] += n
            else:
                groups["layer norms and scalars"] += n
    agg_in = D * g["agg_states"]
    groups["aggregate projection"] = agg_in * g["agg_video"] + g["agg_video"] + agg_in * g["agg_audio"] + g["agg_audio"]
    n_tensors += 4
    return groups, n_tensors


def g4_total():
    return sum(g4_groups()[0].values())


def g4_matmul_layer_parameters():
    """Parameters of the attention and network matrices, the ones a token multiplies by (all the layer matrices)."""
    groups = g4_groups()[0]
    return groups["attention matrices"] + groups["feed-forward matrices"]


def gemma4_parameters():
    groups, n_tensors = g4_groups()
    total = sum(groups.values())
    rows = [(k, _int(v), _fx(100.0 * v / total, 2) + "%") for k, v in groups.items()]
    rows.append(("**total**", _int(total), "100%"))
    full = [i for i in range(G4["L"]) if g4_layer_is_full(i)]
    n_s, n_f = len(g4_layer_tensors(False)), len(g4_layer_tensors(True))
    per = {k: sum(n for name, n in g4_layer_tensors(k) if name in ("q_proj", "k_proj", "v_proj", "o_proj")) for k in (False, True)}
    pins = [
        ("tensors in a sliding layer / a full layer", f"{n_s} / {n_f}", "14 / 13", "`manifest_matches_the_real_12b_checkpoint_header_count`"),
        ("importable tensors", _int(n_tensors), "670", "the same test"),
        ("importable + 11 out of scope + 5 assets", _int(n_tensors + 11 + 5), "686", "the same test"),
        ("full-attention layers (0-indexed)", ", ".join(str(i) for i in full), "5, 11, 17, 23, 29, 35, 41, 47", "`gemma4_12b_layer_types_match_the_real_checkpoints_explicit_array`"),
        ("file at bf16 (2 bytes per number)", f"{total * 2 / 1e9:.2f} GB", "26 GB", "doc of `crates/gemma4/src/lib.rs`"),
        ("the same numbers as f32 on the host", f"{total * 4 / 1e9:.2f} GB = {_gib(total * 4):.2f} GiB", "~49 GB", "doc of `crates/gemma4/src/gguf_src.rs`"),
    ]
    return (f"The text tower of Gemma-4 as `Gemma4Config::gemma4_12b` configures it: V = {_int(G4['V'])}, D = {_int(G4['D'])}, F = {_int(G4['F'])}, "
            f"L = {G4['L']} layers, H = {G4['H']} query heads. Layers 5, 11, ..., 47 (0-indexed) are full attention with 1 key-value head of width "
            f"{G4['dh_g']} and no V matrix (the key output serves as the value); the other 40 are sliding-window attention with {G4['hkv_s']} "
            f"key-value heads of width {G4['dh_s']} (window {_int(G4['window'])}). Each layer has four norm gains and a layer scalar; the tower has no output matrix; "
            f"after the last layer an aggregate projection maps the {G4['agg_states']} hidden states (embedding output and 48 layers) "
            f"of one token, {_int(G4['D'] * G4['agg_states'])} numbers, to {_int(G4['agg_video'])} and to {_int(G4['agg_audio'])} numbers, each with a bias. "
            f"The attention matrices of one sliding layer hold {_int(per[False])} numbers, those of a full layer {_int(per[True])}.\n\n"
            + _table(["group", "numbers", "share"], rows)
            + "\n\nWhat the crate's tests and docs state, against what the tensor list above gives.\n\n"
            + _table(["quantity", "counted here", "stated in the crate", "where"], pins, "llll"))


def opening_numbers():
    groups, _ = g4_groups()
    total = sum(groups.values())
    without = total - groups["aggregate projection"]
    rows = [("numbers in the tower and its aggregate projection", _int(total), "the crate's constructor is named `gemma4_12b`"),
            ("the same without the aggregate projection", _int(without), ""),
            ("bytes at bf16 (2 per number)", f"{_int(total * 2)} = {total * 2 / 1e9:.2f} GB = {_gib(total * 2):.2f} GiB", "\"26 GB\""),
            ("bytes as f32 on the host (4 per number)", f"{_int(total * 4)} = {total * 4 / 1e9:.2f} GB = {_gib(total * 4):.2f} GiB", "\"~49 GB\"")]
    return ("Counted from the tensor list of the Gemma-4 text tower (the sum of the shapes of the 670 tensors that `gemma4_tensor_manifest` lists "
            "for `Gemma4Config::gemma4_12b`), against what the crate's documentation and a commit message say. GB is 10^9 bytes, GiB is 2^30.\n\n"
            + _table(["quantity", "counted from the tensors", "said in the crate and its history"], rows, "lrl"))


# ------------------------------------------------------------------------------------------------ MiniMax-H3's DiT

# `crates/minimaxh3/src/config.rs`, `H3TransformerConfig::real`.
H3 = dict(heads=56, hd=128, hidden=5376, layers=50, refiner=2, ffn=14336, c_video=24, c_audio=32, patch=(1, 2, 2), text=5120,
          freq=256, t_hidden=5376, t_embed=2688)


def h3_inner():
    return H3["heads"] * H3["hd"]


def h3_block_parameters():
    """One transformer block without the AdaLN projection: Q, K, V, O, the two head norms, the fused gate and value matrix of the network
    (2 F by D), the down matrix (D by F) and two norm gains."""
    hid, inner, hd, ffn = H3["hidden"], h3_inner(), H3["hd"], H3["ffn"]
    attn = 3 * inner * hid + hid * inner
    return attn, 2 * hd, 2 * ffn * hid + hid * ffn, 2 * hid


def h3_adaln_parameters():
    """adaln_proj of one block: a linear map from the time embedding (2,688) to 18 D numbers (six modulation vectors for each of three
    modalities), weight and bias."""
    return 18 * H3["hidden"] * H3["t_embed"] + 18 * H3["hidden"]


def h3_groups():
    g = H3
    hid, L, R = g["hidden"], g["layers"], g["refiner"]
    attn, qkn, ffn, norms = h3_block_parameters()
    video_patch = g["c_video"] * g["patch"][0] * g["patch"][1] * g["patch"][2]
    groups = {
        "attention matrices (blocks)": L * attn,
        "feed-forward matrices (blocks)": L * ffn,
        "head norms and block norms": L * (qkn + norms),
        "adaln_proj (blocks)": L * h3_adaln_parameters(),
        "token refiner (2 blocks, no adaln_proj)": R * (attn + qkn + ffn + norms) + hid,
        "input projections and text embedder": hid * video_patch + hid + hid * g["c_audio"] + hid + hid * g["text"] + hid,
        "time embedder": g["t_hidden"] * g["freq"] + g["t_hidden"] + g["t_embed"] * g["t_hidden"] + g["t_embed"],
        "output norm and heads": hid + 2 * hid * g["t_embed"] + 2 * hid + video_patch * hid + video_patch + g["c_audio"] * hid + g["c_audio"],
    }
    return groups


def h3_shape_rows():
    """The shapes behind the groups of `h3_groups`, as the crate's fixture `rand_tensors` lists them, with the numbers each adds up to
    (asserted against the group totals)."""
    g = H3
    D, inner, hd, F, te = g["hidden"], h3_inner(), g["hd"], g["ffn"], g["t_embed"]
    vp = g["c_video"] * g["patch"][0] * g["patch"][1] * g["patch"][2]
    S = lambda *d: "[" + ", ".join(str(x) for x in d) + "]"
    attn = f"to_q, to_k, to_v {S(inner, D)} each; to_out {S(D, inner)}"
    norms = f"norm_q, norm_k {S(hd)} each; norm1, norm2 {S(D)} each"
    ffn = f"ff.net.0.proj {S(2 * F, D)}; ff.net.2 {S(D, F)}"
    rows = [("attention matrices (blocks)", attn, g["layers"]),
            ("feed-forward matrices (blocks)", ffn, g["layers"]),
            ("head norms and block norms", norms + f" (2 x {D} + 2 x {hd} = {2 * D + 2 * hd} per block)", g["layers"]),
            ("adaln_proj (blocks)", f"linear.weight {S(18 * D, te)}; linear.bias {S(18 * D)}", g["layers"]),
            ("token refiner (2 blocks, no adaln_proj)",
             f"per block the attention, norm and feed-forward tensors above (no adaln_proj); once more final_norm {S(D)}", g["refiner"]),
            ("input projections and text embedder",
             f"proj_in.weight {S(D, vp)} and bias {S(D)}; audio_proj_in.weight {S(D, g['c_audio'])} and bias {S(D)}; "
             f"context_embedder.weight {S(D, g['text'])} and bias {S(D)}", 1),
            ("time embedder",
             f"linear_1.weight {S(g['t_hidden'], g['freq'])} and bias {S(g['t_hidden'])}; linear_2.weight {S(te, g['t_hidden'])} and bias {S(te)}", 1),
            ("output norm and heads",
             f"norm_out.norm.weight {S(D)}; norm_out.linear.weight {S(2 * D, te)} and bias {S(2 * D)}; proj_out.weight {S(vp, D)} and bias {S(vp)}; "
             f"audio_proj_out.weight {S(g['c_audio'], D)} and bias {S(g['c_audio'])}", 1)]
    return rows


def h3_total():
    return sum(h3_groups().values())


def minimaxh3_parameters():
    groups = h3_groups()
    total = sum(groups.values())
    attn, qkn, ffn, norms = h3_block_parameters()
    block = attn + qkn + ffn + norms
    ada = h3_adaln_parameters()
    inner, hid = h3_inner(), H3["hidden"]
    rows = [(k, _int(v), _fx(100.0 * v / total, 2) + "%") for k, v in groups.items()]
    rows.append(("**total**", _int(total), "100%"))
    resident = total - H3["layers"] * ada
    pins = [
        ("QKV width H d_h", _int(inner), "7,168 (56 x 128), not equal to D", "`real_config_matches_the_checkpoint_header_numbers`"),
        ("adaln_proj output width 18 D", _int(18 * hid), "96,768", "the same test"),
        ("one block without adaln_proj", f"{block / 1e6:.1f} M", "within 3.5e8 to 4.5e8 (the comment says ~385M)", "`the_cost_model_ranks_a_block_above_the_endpoint_weights`"),
        ("adaln_proj of one block", f"{ada / 1e6:.1f} M", "260M per block", "doc of `crates/minimaxh3/src/lib.rs`"),
        ("adaln_proj of the 50 blocks", f"{H3['layers'] * ada / 1e9:.2f} B of {total / 1e9:.2f} B", "13.0B of 33B", "the same doc"),
        ("adaln_proj share of a block's bytes", _fx(100.0 * ada / (ada + block), 1) + "%", "40%", "doc of `StreamingDit`"),
        ("adaln_proj at f32", f"{H3['layers'] * ada * 4 / 1e9:.1f} GB", "52 GB", "test comment in `resident_minimaxh3.rs`"),
        ("all parameters at f32", f"{total * 4 / 1e9:.1f} GB", "~132 GB", "doc of `dit_shard.rs`"),
        ("parameters without adaln_proj (the resident backbone)", f"{resident / 1e9:.2f} B", "~20B", "doc of `crates/minimaxh3/src/lib.rs`"),
    ]
    return (f"The diffusion transformer of MiniMax-H3 as `H3TransformerConfig::real` configures it: D = {_int(hid)}, {H3['layers']} blocks plus "
            f"{H3['refiner']} token-refiner blocks, {H3['heads']} heads of width {H3['hd']} (plain multi-head attention, no sharing, QKV width "
            f"{_int(inner)} larger than D), a gated network of hidden width {_int(H3['ffn'])} whose gate and value matrices are one tensor of 2 F by D, "
            f"and for every block an `adaln_proj` from the {_int(H3['t_embed'])}-number time embedding to 18 D numbers, six modulation vectors for "
            "each of three modalities. The tensor list is the one the crate's own fixture `rand_tensors` builds.\n\n"
            + _table(["group", "numbers", "share"], rows)
            + "\n\nThe tensors behind each group, as the fixture lists them (shapes are [rows, columns], a vector has one dimension; "
            "all linear maps are stored as [out, in]).\n\n"
            + _table(["group", "tensors in one instance and their shapes", "instances"], h3_shape_rows(), "llr")
            + "\n\nThe same tensor list against what the crate pins or states.\n\n"
            + _table(["quantity", "counted here", "stated in the crate", "where"], pins, "llll"))


def params_share_data():
    """(label, total, [(component, share)]) for the stacked bars of the parameter figure."""
    out = []
    s = small_shape()
    groups = md.parameter_groups(s)
    total = md.parameters(s)

    def pick(gr, keys):
        return sum(c for n, c in gr if n in keys)

    def split(gr, total):
        emb = pick(gr, ("token embedding table", "separate output matrix"))
        att = pick(gr, ("attention matrices W_Q, W_K, W_V, W_O",))
        ffn = pick(gr, ("feed-forward matrices",))
        return [("embedding and head", emb / total), ("attention matrices", att / total), ("feed-forward matrices", ffn / total),
                ("norms and other", (total - emb - att - ffn) / total)]

    out.append(("small decoder", total, split(groups, total)))
    q8 = md.QWEN3_8B
    out.append(("Qwen3 8B", md.parameters(q8), split(md.parameter_groups(q8), md.parameters(q8))))
    gg = g4_groups()[0]
    gt = sum(gg.values())
    out.append(("Gemma-4 text tower", gt, [("embedding and head", gg["embedding table"] / gt), ("attention matrices", gg["attention matrices"] / gt),
                                           ("feed-forward matrices", gg["feed-forward matrices"] / gt),
                                           ("norms and other", (gg["QK-norm gains"] + gg["layer norms and scalars"] + gg["aggregate projection"]) / gt)]))
    hg = h3_groups()
    ht = sum(hg.values())
    att = hg["attention matrices (blocks)"]
    ffn = hg["feed-forward matrices (blocks)"]
    ada = hg["adaln_proj (blocks)"]
    out.append(("MiniMax-H3 DiT", ht, [("embedding and head", 0.0), ("attention matrices", att / ht), ("feed-forward matrices", ffn / ht),
                                       ("norms and other", (ht - att - ffn - ada) / ht), ("adaln_proj", ada / ht)]))
    return out


# ------------------------------------------------------------------------------------------------ FLOPs per token


def flops_per_token(n_act, L, H, dh, T):
    """Forward FLOPs of a token that sees T keys: 2 N_act for the products with weights and 4 L H d_h T for the scores and the
    weighted sum of values (norms, softmax, rotation, gates and adds not counted)."""
    return 2 * n_act + 4 * L * H * dh * T


def flops_small_decoder():
    s = SMALL
    V, D, L, H, hkv, dh, F = s["V"], s["D"], s["L"], s["H"], s["hkv"], s["dh"], s["F"]
    n_p = matmul_parameters(V, D, L, H, hkv, dh, F)
    crossover = n_p / (2 * L * H * dh)
    sh = small_shape()
    rows = []
    for T in (1, 16, 128, 445, 1024, 8192):
        f = flops_per_token(n_p, L, H, dh, T)
        assert f == md.decode_flops(sh, T - 1), (f, md.decode_flops(sh, T - 1))
        att = 4 * L * H * dh * T
        rows.append((_int(T), _int(2 * n_p), _int(att), _int(f), _fx(att / (2 * n_p), 4)))
    return (f"The small decoder (V = {V}, D = {D}, L = {L}, H = {H}, d_h = {dh}, h_kv = {hkv}, F = {F}, separate head): N_p = L (2 D H d_h + 2 D h_kv d_h "
            f"+ 3 D F) + V D = {_int(n_p)}. A token that sees T keys (itself included) costs 2 N_p + 4 L H d_h T FLOPs; the last column is the attention "
            f"term over the parameter term, 4 L H d_h T / (2 N_p), equal to 1 at T = N_p / (2 L H d_h) = {_fx(crossover, 1)}. Every row agrees with "
            "`modern_decoder.decode_flops` for a token at cache length T - 1.\n\n"
            + _table(["T", "parameter term 2 N_p", "attention term 4 L H d_h T", "FLOPs per token", "attention / parameter"], rows))


def g4_flops(T):
    """(parameter term of the layers, aggregate projection, attention term of sliding layers, attention term of full layers) for a
    token that sees T keys."""
    g = G4
    layer_params = g4_matmul_layer_parameters()
    agg = 2 * g["D"] * g["agg_states"] * (g["agg_video"] + g["agg_audio"])
    n_full = sum(1 for i in range(g["L"]) if g4_layer_is_full(i))
    n_slide = g["L"] - n_full
    slide = n_slide * 4 * g["H"] * g["dh_s"] * min(T, g["window"])
    full = n_full * 4 * g["H"] * g["dh_g"] * T
    return 2 * layer_params, agg, slide, full


def h3_flops_per_row(R):
    """Main stack of the DiT for one packed row among R rows (every row attends to all R, no mask): 2 per weight of the blocks'
    matrices plus 4 H d_h R per block. adaln_proj does no per-row work (its output is cached per timestep)."""
    attn, _, ffn, _ = h3_block_parameters()
    L = H3["layers"]
    return 2 * L * (attn + ffn), 4 * L * h3_inner() * R


def flops_models():
    rows = []
    q8 = md.QWEN3_8B
    n_q8 = q8["L"] * (2 * q8["D"] * q8["H"] * q8["dh"] + 2 * q8["D"] * q8["hkv"] * q8["dh"] + 3 * q8["D"] * q8["F"]) + q8["V"] * q8["D"]
    for T in (1024, 8192, 131072):
        pt = 2 * n_q8
        at = 4 * q8["L"] * q8["H"] * q8["dh"] * T
        rows.append(("Qwen3 8B (full attention)", _int(T), _fx(pt / 1e9, 3), _fx(at / 1e9, 3), _fx(at / pt, 3)))
    for T in (1024, 8192, 131072):
        pt, agg, sl, fu = g4_flops(T)
        rows.append(("Gemma-4 text tower (5:1 sliding : full)", _int(T), _fx((pt + agg) / 1e9, 3), _fx((sl + fu) / 1e9, 3), _fx((sl + fu) / (pt + agg), 3)))
    for R in (4096, 13312, 32768):
        pt, at = h3_flops_per_row(R)
        rows.append(("MiniMax-H3 blocks (no mask)", _int(R), _fx(pt / 1e9, 3), _fx(at / 1e9, 3), _fx(at / pt, 3)))
    return ("GFLOPs (10^9) per token, or per packed row for the diffusion transformer, split into the parameter term and the attention term. "
            "Qwen3 8B: 2 N_p with N_p the matrices and the head, 4 L H d_h T. Gemma-4: 2 times the attention and network matrices of 48 layers "
            "plus the aggregate projection of 188,160 inputs to 6,144 outputs (2 x 188,160 x 6,144), and an attention term in which the 40 sliding "
            "layers see min(T, 1,024) keys with H d_h = 4,096 and the 8 full layers see T keys with H d_h = 8,192. MiniMax-H3: 2 times the "
            "matrices of 50 blocks (`adaln_proj` excluded: it does nothing per row), and 4 H d_h R per block over R packed rows, all rows "
            "attending to all rows.\n\n"
            + _table(["model", "T or R", "parameter term (G)", "attention term (G)", "attention / parameter"], rows, "lrrrr"))


def ratio_data():
    """(label, [(T, ratio)]) for the figure of the attention term over the parameter term."""
    Ts = [2 ** k for k in range(6, 18)]
    s = SMALL
    n_s = matmul_parameters(s["V"], s["D"], s["L"], s["H"], s["hkv"], s["dh"], s["F"])
    q8 = md.QWEN3_8B
    n_q8 = q8["L"] * (2 * q8["D"] * q8["H"] * q8["dh"] + 2 * q8["D"] * q8["hkv"] * q8["dh"] + 3 * q8["D"] * q8["F"]) + q8["V"] * q8["D"]
    out = [("small decoder", [(T, 4 * s["L"] * s["H"] * s["dh"] * T / (2 * n_s)) for T in Ts]),
           ("Qwen3 8B", [(T, 4 * q8["L"] * q8["H"] * q8["dh"] * T / (2 * n_q8)) for T in Ts])]
    pts = []
    for T in Ts:
        pt, agg, sl, fu = g4_flops(T)
        pts.append((T, (sl + fu) / (pt + agg)))
    out.append(("Gemma-4 text tower", pts))
    out.append(("MiniMax-H3 blocks", [(R, h3_flops_per_row(R)[1] / h3_flops_per_row(R)[0]) for R in Ts]))
    return out


# ------------------------------------------------------------------------------------------------ inference memory

# Bytes per stored parameter by format. int8 and 4-bit carry one f32 scale per group of 32 along the input axis
# (`Weight::bytes_for` in `crates/model/src/ops.rs`): 32 one-byte codes plus 4 bytes of scale is 36 / 32 bytes per weight.
FORMATS = (("f32", 4.0), ("bf16", 2.0), ("int8, group 32", 1.0 + 4.0 / 32), ("4-bit, group 32", 0.5 + 4.0 / 32))


def weights_bytes_table():
    cases = (("Qwen3 8B", md.parameters(md.QWEN3_8B)), ("Gemma-4 text tower", g4_total()), ("MiniMax-H3 DiT", h3_total()))
    rows = []
    for fmt, b in FORMATS:
        rows.append((fmt, _fx(b, 4), *[_fx(_gib(p * b), 2) for _, p in cases]))
    return ("Weight memory in GiB (2^30 bytes) with every parameter stored at the format's rate (the norm gains and biases, under 0.1% of "
            "the numbers, are left at the same rate). The packed formats carry one f32 scale per 32 weights along the input axis: "
            "(32 x 8 + 32) / 32 bits = 9 bits for int8, 5 bits for 4-bit. P: "
            + ", ".join(f"{n} {_int(p)}" for n, p in cases) + ".\n\n"
            + _table(["format", "bytes per parameter"] + [n + " (GiB)" for n, _ in cases], rows))


def g4_kv_bytes(T, b=2):
    """KV cache bytes for one sequence of T tokens in a hypothetical decode with a cache: each layer stores K and V rows of
    h_kv x d_h numbers; a sliding layer keeps the last `window` rows."""
    g = G4
    total = 0
    for i in range(g["L"]):
        full = g4_layer_is_full(i)
        hkv, dh = (g["hkv_g"], g["dh_g"]) if full else (g["hkv_s"], g["dh_s"])
        rows = T if full else min(T, g["window"])
        total += 2 * hkv * dh * rows * b
    return total


def g4_kv_all_sliding_geometry_global(T, b=2):
    """Every layer a global layer with the sliding layers' 8 key-value heads of width 256."""
    return G4["L"] * 2 * G4["hkv_s"] * G4["dh_s"] * T * b


def kv_cache_gemma4():
    rows = []
    for T in (1024, 8192, 32768, 131072):
        rows.append((_int(T), _fx(_gib(g4_kv_all_sliding_geometry_global(T)), 3), _fx(_gib(g4_kv_bytes(T)), 3),
                     _fx(g4_kv_all_sliding_geometry_global(T) / g4_kv_bytes(T), 2)))
    per_slide = 2 * G4["hkv_s"] * G4["dh_s"] * 2
    per_full = 2 * G4["hkv_g"] * G4["dh_g"] * 2
    n_full = sum(1 for i in range(G4["L"]) if g4_layer_is_full(i))
    n_slide = G4["L"] - n_full
    return (f"If the Gemma-4 text tower decoded with a cache (brain's `crates/gemma4` encodes a whole prompt in one forward and keeps none, so this "
            f"is the cache a decoder of this shape would need), a sliding layer would store 2 x {G4['hkv_s']} x {G4['dh_s']} numbers per token = "
            f"{_int(per_slide)} bytes in bf16 and keep the last {_int(G4['window'])} tokens, and a full layer 2 x {G4['hkv_g']} x {G4['dh_g']} numbers = "
            f"{_int(per_full)} bytes and keep all of them ({n_slide} sliding layers, {n_full} full). The comparison column gives every one of the "
            f"48 layers the sliding layers' geometry and no window (48 x {_int(per_slide)} bytes per token). GiB for one sequence.\n\n"
            + _table(["tokens", "48 layers, 8 heads of 256, no window", "Gemma-4 layout", "ratio"], rows))


def h3_block_workspace_bytes(rows):
    """Peak transient bytes of one block's forward over `rows` packed rows, from the buffer inventory of the chapter."""
    hid, inner, ffn = H3["hidden"], h3_inner(), H3["ffn"]
    fp32 = (5 * hid + 5 * inner + ffn + max(3 * inner, 2 * ffn)) * 4
    int8_inputs = 2 * hid + inner + ffn
    return rows * (fp32 + int8_inputs)


def workspace_tables():
    hid, inner, ffn, heads = H3["hidden"], h3_inner(), H3["ffn"], H3["heads"]
    per_row = (5 * hid + 5 * inner + ffn + max(3 * inner, 2 * ffn)) * 4 + 2 * hid + inner + ffn
    rows = []
    for R in (1, 512, 4096, 8192, 13312):
        ws = h3_block_workspace_bytes(R)
        mat = 2 * heads * R * R * 4
        rows.append((_int(R), _int(ws), _fx(ws / GIB, 3), _int(mat), _fx(mat / GIB, 3), _fx(3 * R * inner * 4 / GIB, 3)))
    g4rows = []
    for T in (512, 1024, 4096, 8192):
        g4rows.append((_int(T), _fx(G4["H"] * T * T * 4 / GIB, 3)))
    return (f"One block of the MiniMax-H3 DiT over R packed rows. The inventory behind the first column: five fp32 buffers of width D = {_int(hid)}, "
            f"five of width H d_h = {_int(inner)}, one of width F = {_int(ffn)} and one of max(3 H d_h, 2 F) = {_int(max(3 * inner, 2 * ffn))}, and the int8 copies "
            f"of the linears' inputs (2 D + H d_h + F bytes), which is {_int(per_row)} bytes per row; with the halves of the block sharing buffers "
            "it is linear in R. The columns headed materialized are what two score matrices (scores and probabilities, H x R x R fp32 numbers each) "
            "would add; the last column is the packed query-key-value slab (3 R H d_h fp32 numbers) the fused kernel uses instead.\n\n"
            + _table(["rows R", "block workspace (bytes)", "(GiB)", "scores and probabilities materialized (bytes)", "(GiB)", "fused slab (GiB)"], rows)
            + "\n\nThe score matrix of the Gemma-4 tower is materialized in `crates/gemma4` (H x T x T fp32 numbers for every layer's forward):\n\n"
            + _table(["tokens T", "scores of one layer, 16 heads (GiB)"], g4rows))


def headroom_rule():
    """`capacity::headroom(free)`: a budget b of the free bytes holds when b x 1.14 + 512 MiB <= free; the floor is 2 GiB."""
    MIB = 2 ** 20
    free = 24433 * MIB
    context = 512 * MIB
    overhead = 14
    floor = 2 * GIB
    budget = (free - context) // (100 + overhead) * 100
    headroom = max(floor, free - budget)
    old_budget = free - floor
    requested = 21232 * MIB
    rows = [("free bytes reported by the card", _int(free // MIB) + " MiB"),
            ("budget with the old fixed headroom of 2 GiB", _int(old_budget // MIB) + " MiB"),
            ("budget with (free - 512 MiB) / 1.14", _int(round(budget / MIB)) + " MiB"),
            ("buffers the int8 load requested (when it failed)", _int(requested // MIB) + " MiB"),
            ("old budget fits the request", "yes" if requested <= old_budget else "no"),
            ("new budget fits the request", "yes" if requested <= budget else "no"),
            ("headroom taken off the free bytes by the new rule", _int(round(headroom / MIB)) + " MiB")]
    return ("A weights-only estimate leaves out what the allocator adds. The rule of `gpu_core::capacity::headroom`: a budget b of the free "
            "bytes holds when 1.14 b + 512 MiB is at most the free bytes, and the headroom is never less than 2 GiB. The case that set "
            "the terms, an idle 24 GB card with an int8 diffusion transformer:\n\n" + _table(["quantity", "value"], rows, "lr"))


# ------------------------------------------------------------------------------------------------ training memory

# The four independent choices of one trained parameter, as `paramstore::precision::ParamPrecision` accounts them:
# (label, value bytes, gradient bytes, master bytes, bytes of one moment slot, slots). `int8 blocks` is 1 byte per element plus an f32
# scale per block of 256 elements.
def recipe_bytes(value, grad, master, moment, slots=2):
    return value + grad + master + slots * moment


def int8_block_bytes(numel, block=256):
    return numel + 4 * (-(-numel // block))


def training_bytes_per_parameter():
    n = 1000
    rows = []
    recipes = (("f32 values, f32 gradients, f32 moments, no master", 4 * n, 4 * n, 0, 4 * n),
               ("bf16 values and gradients, f32 master, f32 moments", 2 * n, 2 * n, 4 * n, 4 * n),
               ("bf16 values and gradients, f32 master, bf16 moments", 2 * n, 2 * n, 4 * n, 2 * n),
               ("bf16 values and gradients, f32 master, 8-bit block moments", 2 * n, 2 * n, 4 * n, int8_block_bytes(n)))
    for label, v, g, m, mom in recipes:
        total = v + g + m + 2 * mom
        rows.append((label, _int(v), _int(g), _int(m), _int(2 * mom), _int(total), _fx(total / n, 3)))
    # The pinned case of `bytes_are_physical_and_components_independent`: 1,000 numbers, two slots.
    assert (2 * n, 2 * n, 2 * int8_block_bytes(n), 4 * n) == (2000, 2000, 2 * (1000 + 16), 4000)
    return ("Bytes of the training state of n = 1,000 parameters under AdamW (two moment slots). A moment slot in 8-bit blocks is "
            "n + 4 ceil(n / 256) = 1,016 bytes. The last row of the first table is the case that `bytes_are_physical_and_components_independent` "
            "pins in `crates/paramstore/src/precision.rs` (values 2,000, gradients 2,000, moments 2 x 1,016, master 4,000).\n\n"
            + _table(["recipe", "values", "gradients", "master copy", "two moments", "total", "bytes per parameter"], rows, "lrrrrrr"))


def saved_floats_per_token_layer(D, H, hkv, dh, F, n_ffn, attn_state, extra_norms=0):
    """Numbers one block keeps per token for its backward pass: the block input, the output of each norm that a matrix reads,
    the residual after attention, queries, keys, values and the attention output, the attention state, and the network's
    intermediate tensors (n_ffn of width F). Rule of Volume V, Chapter 10: every tensor that a backward product or norm reads."""
    return 4 * D + 2 * (H + hkv) * dh + attn_state + n_ffn * F + extra_norms * D


def activation_rule_check():
    """The rule reproduces the 36,864 numbers per token and block of Volume V, Chapter 10 for GPT-2 small at T = 1,024."""
    f = saved_floats_per_token_layer(768, 12, 12, 64, 3072, 2, 2 * 12 * 1024)
    assert f == 36864, f
    return f


def g4_saved_floats(T, flash=True):
    g = G4
    total = 0
    per = []
    for i in range(g["L"]):
        full = g4_layer_is_full(i)
        hkv, dh = (g["hkv_g"], g["dh_g"]) if full else (g["hkv_s"], g["dh_s"])
        vis = T if full else min(T, g["window"])
        state = g["H"] if flash else 2 * g["H"] * vis
        f = saved_floats_per_token_layer(g["D"], g["H"], hkv, dh, g["F"], 3, state, extra_norms=2)
        per.append(f)
        total += f
    return total, per


def h3_saved_floats(R, flash=True):
    hid, inner, ffn, heads = H3["hidden"], h3_inner(), H3["ffn"], H3["heads"]
    state = heads if flash else 2 * heads * R
    # The gated network keeps the fused gate and value pre-activations (2 F) and the product (F).
    return saved_floats_per_token_layer(hid, heads, heads, H3["hd"], ffn, 3, state)


def training_memory_models():
    activation_rule_check()
    gemma = g4_total()
    h3 = h3_total()
    rows = []
    tokens = 8192
    b_act = 2   # activations kept in bf16
    g_keep, per = g4_saved_floats(tokens)
    g_rec = G4["L"] * G4["D"] + max(per)
    h_rows = 13312
    h_layer = h3_saved_floats(h_rows)
    h_keep = H3["layers"] * h_layer
    h_rec = H3["layers"] * H3["hidden"] + h_layer
    recipes = (("f32 everywhere, AdamW", 4, 4, 0, 4), ("bf16 values and gradients, f32 master, f32 moments", 2, 2, 4, 4),
               ("bf16 values and gradients, f32 master, bf16 moments", 2, 2, 4, 2))
    out = []
    for name, P, keep, rec, T in (("Gemma-4 text tower", gemma, g_keep, g_rec, tokens), ("MiniMax-H3 DiT", h3, h_keep, h_rec, h_rows)):
        for label, v, g, m, mo in recipes:
            state = P * (v + g + m + 2 * mo)
            out.append((name, label, _fx(_gib(state), 1), _fx(_gib(keep * T * b_act), 1), _fx(_gib(rec * T * b_act), 1),
                        _fx(_gib(state + keep * T * b_act), 1), _fx(_gib(state + rec * T * b_act), 1)))
    return ("Training memory of two models if they were trained, in GiB: brain trains neither at this size (the Gemma-4 crate is an inference "
            "port with no trainer; MiniMax-H3 has a host f64 reference trainer for gradient checks), so these are what the formula says, not "
            "what a run measured. State = P x (value + gradient + master + two moments) bytes. Activations are kept in bf16 (2 bytes) for "
            f"{_int(tokens)} tokens of Gemma-4 and {_int(h_rows)} packed rows of the DiT; the attention state is a logsumexp per head (a fused "
            f"attention kernel). Numbers kept per token: Gemma-4 {_int(g_keep)} (all layers) or {_int(g_rec)} with recomputation (the 48 residual snapshots and "
            f"one full layer); DiT {_int(h_keep)} or {_int(h_rec)}.\n\n"
            + _table(["model", "recipe", "state", "activations kept", "activations, recomputed", "total, kept", "total, recomputed"], out, "llrrrrr"))


def activations_per_layer_table():
    g = G4
    rows = []
    sliding = saved_floats_per_token_layer(g["D"], g["H"], g["hkv_s"], g["dh_s"], g["F"], 3, g["H"], extra_norms=2)
    full = saved_floats_per_token_layer(g["D"], g["H"], g["hkv_g"], g["dh_g"], g["F"], 3, g["H"], extra_norms=2)
    items = [("block input, norm outputs read by matrices, residual after attention (4 D)", 4 * g["D"], 4 * g["D"]),
             ("two more norms of the layer (2 D)", 2 * g["D"], 2 * g["D"]),
             ("queries and attention output (2 H d_h)", 2 * g["H"] * g["dh_s"], 2 * g["H"] * g["dh_g"]),
             ("keys and values (2 h_kv d_h)", 2 * g["hkv_s"] * g["dh_s"], 2 * g["hkv_g"] * g["dh_g"]),
             ("logsumexp per head (H)", g["H"], g["H"]),
             ("gate, up and their product (3 F)", 3 * g["F"], 3 * g["F"])]
    for n, a, b in items:
        rows.append((n, _int(a), _int(b)))
    assert sum(a for _, a, _ in items) == sliding and sum(b for _, _, b in items) == full
    rows.append(("**one layer**", _int(sliding), _int(full)))
    return ("Numbers a Gemma-4 layer keeps per token for the backward pass under the rule of Volume V, Chapter 10 (every tensor a backward product or "
            "norm reads), with a fused attention kernel that keeps one logsumexp per head. The rule reproduces the 36,864 numbers per token and block "
            "of that chapter's table for GPT-2 small (checked when this table is built).\n\n"
            + _table(["tensor", "sliding layer", "full layer"], rows))


def h3_activations_per_block_table():
    hid, inner, ffn, heads, hd = H3["hidden"], h3_inner(), H3["ffn"], H3["heads"], H3["hd"]
    items = [("block input, the two norm outputs the matrices read, the residual after attention (4 D)", 4 * hid),
             ("queries and attention output (2 H d_h)", 2 * inner),
             ("keys and values (2 H d_h, no sharing)", 2 * inner),
             ("logsumexp per head (H)", heads),
             ("gate and value pre-activations and the product (3 F)", 3 * ffn)]
    total = sum(n for _, n in items)
    assert total == h3_saved_floats(13312)
    rows = [(n, _int(v)) for n, v in items] + [("**one block, per row**", _int(total))]
    return ("Numbers a MiniMax-H3 block keeps per packed row for the backward pass under the same rule (D = 5,376, H = 56 heads of "
            f"d_h = {hd} so H d_h = {_int(inner)}, F = {_int(ffn)}), with a fused attention kernel that keeps one logsumexp per head. The "
            "block has two norms and no post-norms, so the 2 D of extra norms that the Gemma-4 layer keeps are absent; the AdaLN tables are "
            f"side inputs and are not counted per row. Over the {H3['layers']} blocks one row keeps {_int(H3['layers'] * total)} numbers.\n\n"
            + _table(["tensor", "numbers per row"], rows))


def memory_breakdown_data():
    """(label, [(component, GiB)]) for the stacked bars of the memory figure."""
    gemma = g4_total()
    h3 = h3_total()
    T = 32768
    out = []
    out.append(("inference, Gemma-4 tower, bf16, one cache of 32,768 tokens",
                [("weights", _gib(gemma * 2)), ("KV cache", _gib(g4_kv_bytes(T))), ("workspace", 0.0)]))
    out.append(("inference, Gemma-4 tower, int8, same cache",
                [("weights", _gib(gemma * (1 + 4 / 32))), ("KV cache", _gib(g4_kv_bytes(T))), ("workspace", 0.0)]))
    backbone = h3 - H3["layers"] * h3_adaln_parameters()
    out.append(("inference, MiniMax-H3 backbone, int8, 13,312 rows",
                [("weights", _gib(backbone * (1 + 4 / 32))), ("KV cache", 0.0), ("workspace", _gib(h3_block_workspace_bytes(13312)))]))
    tokens = 8192
    keep, per = g4_saved_floats(tokens)
    rec = G4["L"] * G4["D"] + max(per)
    state_w = gemma * 2
    state_g = gemma * 2
    state_o = gemma * (4 + 8)
    out.append(("training, Gemma-4 tower, mixed precision, activations kept (8,192 tokens)",
                [("weights", _gib(state_w)), ("gradients", _gib(state_g)), ("optimizer state and master", _gib(state_o)), ("activations", _gib(keep * tokens * 2))]))
    out.append(("training, same, activations recomputed",
                [("weights", _gib(state_w)), ("gradients", _gib(state_g)), ("optimizer state and master", _gib(state_o)), ("activations", _gib(rec * tokens * 2))]))
    return out


# T5-XXL encoder, `T5Config::xxl` in `crates/t5encoder`: vocabulary 32,128, width 4,096, three network matrices of 4,096 by 10,240, 24 layers,
# 64 heads of width 64, one shared relative-position bias table of 32 buckets by 64 heads.
T5 = dict(V=32128, D=4096, F=10240, dkv=64, L=24, H=64, buckets=32)


def t5_undercount():
    t = T5
    qkvo = 4 * (t["H"] * t["dkv"]) * t["D"]
    per_layer = qkvo + 3 * t["D"] * t["F"] + 2 * t["D"]
    manifest = t["V"] * t["D"] + t["L"] * per_layer + t["D"] + t["buckets"] * t["H"]
    old_bytes = (2 * t["D"] * t["F"] + 4 * 64 * t["H"] * t["D"] + 2 * t["D"]) * t["L"] * 4 + t["D"] * t["V"] * 4
    new_bytes = manifest * 4
    missing = new_bytes - old_bytes
    rows = [("numbers by the tensor list", _int(manifest), ""),
            ("bytes at f32 by the tensor list", _int(new_bytes), f"{_gib(new_bytes):.2f} GiB"),
            ("bytes by the hand-written formula (two network matrices per layer)", _int(old_bytes), f"{_gib(old_bytes):.2f} GiB"),
            ("bytes missing", _int(missing), f"{_gib(missing):.3f} GiB"),
            ("one network matrix per layer, 24 x 4,096 x 10,240 x 4 bytes", _int(t["L"] * t["D"] * t["F"] * 4), f"{_gib(t['L'] * t['D'] * t['F'] * 4):.3f} GiB"),
            ("final norm gain and position table, (4,096 + 32 x 64) x 4 bytes", _int((t["D"] + t["buckets"] * t["H"]) * 4), ""),
            ("the two lines above added", _int(t["L"] * t["D"] * t["F"] * 4 + (t["D"] + t["buckets"] * t["H"]) * 4), "")]
    assert missing == t["L"] * t["D"] * t["F"] * 4 + (t["D"] + t["buckets"] * t["H"]) * 4
    return ("The weights of the T5-XXL encoder at f32 (`T5Config::xxl`): V = 32,128, D = 4,096, F = 10,240, 24 layers, 64 heads of width 64, three network "
            "matrices per layer (two input matrices and one output matrix), one shared relative-position table of 32 by 64, and per layer two norm gains of 4,096 (before attention and before the network) "
            "with a final norm gain of 4,096; QKVO are four matrices of 4,096 by 4,096. The hand-written formula counted 2 D F per layer for the network, "
            "and counted the embedding table, the four attention matrices and the two norm gains of every layer, but not the final norm gain or the "
            "position table.\n\n" + _table(["quantity", "value", "in GiB"], rows, "lrr"))


def activation_estimates():
    """The planner's calibrated constant of `memplan::calibrated_activation_bytes` against the counts of this chapter."""
    per_token_8k_16k = (78 - 67) * GIB / (16384 - 8192)
    const = 1.4 * 1048576.0 / (64.0 * 5120.0)
    g = G4
    tokens = 8192
    planner = const * tokens * g["L"] * g["D"]
    keep, per = g4_saved_floats(tokens)
    rec = g["L"] * g["D"] + max(per)
    rows = [("two points of the 27B run: (78 - 67) GiB over (16,384 - 8,192) tokens", f"{per_token_8k_16k / 1048576:.3f} MiB per token", ""),
            ("the constant in the code, 1.4 MiB per token over 64 layers of width 5,120", f"{const:.4f} bytes per token, layer and unit of width", ""),
            ("planner estimate for the Gemma-4 tower at 8,192 tokens (48 layers, width 3,840)", _int(round(planner)), f"{_gib(planner):.2f} GiB"),
            ("this chapter's count, every layer kept, bf16", _int(keep * tokens * 2), f"{_gib(keep * tokens * 2):.2f} GiB"),
            ("this chapter's count, recomputed, bf16", _int(rec * tokens * 2), f"{_gib(rec * tokens * 2):.2f} GiB")]
    return ("Activation memory of one training step: the planner's calibrated estimate against the counts of this chapter. The code's own description "
            "of the constant is \"calibrated on the 27B (64 layers, width 5120): 78 GiB at 16K tokens against 67 GiB at 8K for the bf16 base\"; "
            "it prices the Gemma-4 tower here only to set the scales side by side.\n\n" + _table(["quantity", "value", "in GiB"], rows, "lrr"))


# ------------------------------------------------------------------------------------------------ scaling laws

NAME_FIT = "model_cost.power_law"
FIT_SIZES = [10 ** 6 * 4 ** i for i in range(8)]     # 1e6 .. 1.6384e10 parameters
FIT_C, FIT_A, FIT_NOISE = 15.0, 0.08, 0.01


def fit_line(xs, ys):
    """Least squares y = a + b x by the closed form of Volume II, Chapter 2, sums added left to right. Returns (a, b, standard
    error of b, R^2)."""
    n = len(xs)
    mx = 0.0
    my = 0.0
    for x, y in zip(xs, ys):
        mx += x
        my += y
    mx /= n
    my /= n
    sxx = sxy = syy = 0.0
    for x, y in zip(xs, ys):
        sxx += (x - mx) * (x - mx)
        sxy += (x - mx) * (y - my)
        syy += (y - my) * (y - my)
    b = sxy / sxx
    a = my - b * mx
    rss = 0.0
    for x, y in zip(xs, ys):
        r = y - (a + b * x)
        rss += r * r
    se = math.sqrt(rss / (n - 2) / sxx)
    return a, b, se, 1.0 - rss / syy


def power_law_points():
    g = bookrng.generator(NAME_FIT)
    pts = []
    for p in FIT_SIZES:
        z = g.normal()
        pts.append((p, FIT_C * p ** (-FIT_A) * (1.0 + FIT_NOISE * z)))
    return pts


def power_law_fit():
    pts = power_law_points()
    xs = [math.log(p) for p, _ in pts]
    ys = [math.log(l) for _, l in pts]
    a, b, se, r2 = fit_line(xs, ys)
    rows = [(_int(p), _fx(l, 4), _fx(math.exp(a + b * math.log(p)), 4), _fx(100.0 * (l / math.exp(a + b * math.log(p)) - 1.0), 2) + "%") for p, l in pts]
    big = FIT_SIZES[-1] * 10
    pred = math.exp(a + b * math.log(big))
    truth = FIT_C * big ** (-FIT_A)
    return (_seed_line(NAME_FIT) + f" Synthetic points, not measurements of any model: loss = {FIT_C} P^(-{FIT_A}) (1 + {FIT_NOISE} z) at the eight sizes "
            "P = 1,000,000 x 4^i, i = 0 .. 7, with one standard normal z drawn per size in increasing order of size (the generator's `normal()`, "
            "Box-Muller). The fit is ordinary least squares of ln(loss) on ln(P) (natural logarithms, the library `log`), closed form, sums added "
            "left to right.\n\n"
            + _table(["P", "loss", "fitted", "deviation"], rows)
            + f"\n\nSlope {_fx(b, 4)} with standard error {_fx(se, 4)}, so the exponent is estimated as {_fx(-b, 4)} against the true {FIT_A}; "
            f"intercept {_fx(a, 4)} (true ln {FIT_C} = {_fx(math.log(FIT_C), 4)}); R^2 = {_fx(r2, 5)}. At 10 times the largest size the fit "
            f"predicts {_fx(pred, 4)} and the generating law gives {_fx(truth, 4)} ({_fx(100.0 * (pred / truth - 1.0), 2)}% apart).")


FLOOR_E, FLOOR_C, FLOOR_A = 1.7, 40.0, 0.3
FLOOR_SIZES = [10 ** 6 * 4 ** i for i in range(8)]


def floor_points():
    return [(p, FLOOR_E + FLOOR_C * p ** (-FLOOR_A)) for p in FLOOR_SIZES]


def power_law_with_floor():
    pts = floor_points()
    xs = [math.log(p) for p, _ in pts]
    ys = [math.log(l) for _, l in pts]
    rows = []
    for lo, hi in ((0, 4), (2, 6), (4, 8), (0, 8)):
        a, b, se, r2 = fit_line(xs[lo:hi], ys[lo:hi])
        top = FLOOR_SIZES[hi - 1] * 10
        pred = math.exp(a + b * math.log(top))
        truth = FLOOR_E + FLOOR_C * top ** (-FLOOR_A)
        rows.append((f"{_int(FLOOR_SIZES[lo])} to {_int(FLOOR_SIZES[hi - 1])}", _fx(-b, 4), _fx(r2, 5), _fx(pred, 4), _fx(truth, 4),
                     _fx(100.0 * (pred / truth - 1.0), 1) + "%"))
    pt = ", ".join(f"{_fx(l, 4)}" for _, l in pts)
    return (f"Noise-free synthetic points of a law with an irreducible term: loss = {FLOOR_E} + {FLOOR_C} P^(-{FLOOR_A}) at the same eight sizes (losses {pt}). "
            "A straight line on the log-log plot is fitted to a window of the points; the table gives the exponent it finds, R^2, its prediction "
            "at 10 times the window's largest size and the true value there.\n\n"
            + _table(["window of sizes", "fitted exponent", "R^2", "prediction at 10x", "true value at 10x", "prediction error"], rows))


def compute_optimal():
    rows = []
    for C in (1e19, 1e21, 1e23):
        P = math.sqrt(C / 120.0)
        n = 20.0 * P
        rows.append((f"{C:.0e}", f"{P:.3e}", f"{n:.3e}", f"{6 * P * n:.3e}"))
    gemma = g4_total()
    n_g = 20.0 * gemma
    rows.append(("(a model of Gemma-4's size)", f"{gemma:.3e}", f"{n_g:.3e}", f"{6 * gemma * n_g:.3e}"))
    gpt2 = md.parameters(md.GPT2)
    return ("The compute-optimal split under C = 6 P n and n = 20 P: P = sqrt(C / 120) parameters and n = 20 P tokens use all of C. The first three rows "
            "are round budgets; the last row is the budget that 20 tokens per parameter would ask for a model with the parameter count of the "
            "Gemma-4 text tower (this says nothing about how that model was trained). For comparison, the GPT-2 small run of Volume V, Chapter 10 "
            f"trained {_int(gpt2)} parameters on 10,000,000,000 tokens, {10_000_000_000 / gpt2:.1f} tokens per parameter, where the rule asks 20.\n\n"
            + _table(["budget C (FLOPs)", "parameters P", "tokens n", "6 P n"], rows))


def power_law_figure_data():
    pts = power_law_points()
    xs = [math.log(p) for p, _ in pts]
    ys = [math.log(l) for _, l in pts]
    a, b, _, _ = fit_line(xs, ys)
    fl = floor_points()
    fx = [math.log(p) for p, _ in fl]
    fy = [math.log(l) for _, l in fl]
    a_lo, b_lo, _, _ = fit_line(fx[0:4], fy[0:4])
    a_hi, b_hi, _, _ = fit_line(fx[4:8], fy[4:8])
    return dict(points=pts, fit=(a, b), floor=fl, low=(a_lo, b_lo), high=(a_hi, b_hi))
