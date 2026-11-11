# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Speech Synthesis".

Plain Python 3, standard library only, written from the chapter's own description and independent of any engine code. Each public
function without arguments returns a markdown string that the chapter embeds between `<!-- worked: speech_synthesis.<name> -->`
markers; the functions with arguments (or a leading underscore) supply the numbers of the chapter's figures, so a picture and the
table beside it cannot disagree. Earlier modules are imported, never copied: the table helper of `sound`, the cost rows of the
codec in `audio_codecs` and of the two speaker encoders in `speaker_embeddings`.

* the representations a synthesiser passes through and what each carries per second;
* the prompt of a codec language model (rows before the first frame, the text read one row per frame) and the label alignment of
  its training;
* the parameters and the multiply-adds of a frame of the 16-codebook synthesiser, counted from its configuration, and the memory
  its attention cache grows by;
* the logit chain that draws one acoustic token, and what a repetition penalty applied once per token or once per occurrence does;
* finite scalar quantisation, the geometry of a speech tokenizer on a reference clip, and the prompt, budget and sampler of a
  speech-token language model;
* the cosine time grid, the assembly of the conditions of a flow decoder and an exactly solvable flow-matching decoder with
  classifier-free guidance;
* the parameters and the multiply-adds of the two flow decoders and of the vocoder, counted from their configurations;
* the chunk arithmetic of streaming, and the work done before the first sample of each system.

Arithmetic is the language's 64-bit floating point. The only random numbers are the Gaussian draws of the flow examples, from the
book's SplitMix64 (`bookrng.generator(name)`); every stream name is printed by the chapter.

Swedish Embedded AB implements solutions for speech synthesis that is counted before it is run: token rates, work per second of
speech and first-sound latency stated as multiply-adds. If your team needs expertise in neural text-to-speech, you can procure our
services by sending an email to info@swedishembedded.com.
"""
import math

import audio_codecs as ac
import bookrng
import sound
import speaker_embeddings as se

_table = sound._table


def _i(n):
    return f"{int(round(n)):,}"


def _f(x, d=4):
    out = f"{x:.{d}f}"
    return out[1:] if out.startswith("-") and not any(c in "123456789" for c in out) else out


def _g(x, d=3):
    return f"{x:,.{d}f}"


def _m(x, d=1):
    """multiply-adds in millions"""
    return f"{x / 1e6:,.{d}f}"


def _gm(x, d=1):
    """multiply-adds in billions"""
    return f"{x / 1e9:,.{d}f}"


def _seed(name):
    return f"`{name}` (seed 0x{bookrng.seed_of(name):016x})"


def _ceil_div(a, b):
    return -(-a // b)


FS = 24000
FRAME_RATE = ac.FRAME_RATE            # 12.5 codec frames a second
SAMPLES_PER_FRAME = ac.HOP            # 1920
NQ = ac.NQ                            # 16 codes per frame
CODE_BITS = ac.CODE_BITS              # 11 bits per code

# ---------------------------------------------------------------- Qwen3-TTS synthesiser (the fields of its configuration)
T_LAYERS, T_D, T_HEADS, T_KV_HEADS, T_HEAD_DIM, T_FF = 28, 1024, 16, 8, 128, 3072
T_VOCAB = 3072                        # codebook-0 entries plus specials
T_TEXT_VOCAB, T_TEXT_HIDDEN = 151936, 2048
M_LAYERS, M_VOCAB = 5, 2048           # the code predictor: layers, entries per residual codebook
SUPPRESS_WINDOW = 1024

# ---------------------------------------------------------------- CosyVoice (the fields of its configuration)
S_TOK = 6561                          # 3 ** 8 speech tokens
S_RATE = 25                           # speech tokens a second
S_VOCAB_V2 = S_TOK + 3                # speech-token embedding rows of the second generation
LM_D, LM_LAYERS, LM_HEADS, LM_KV_HEADS, LM_HEAD_DIM, LM_FF, LM_TEXT_VOCAB = 896, 24, 14, 2, 64, 4864, 151936
MEL_RATE, MEL_BINS = 50, 80           # mel frames a second, mel bins
HIFT_UP = (8, 5, 3)                   # vocoder upsampling stages
HIFT_HOP, HIFT_FFT = 4, 16            # inverse short-time transform of the vocoder


# ================================================================== interfaces

def interface_rates():
    rows = [
        ("waveform, 16-bit samples", FS, 1, 16),
        ("log-mel spectrogram, 32-bit floats", MEL_RATE, MEL_BINS, 32),
        ("codec codes of the 16-codebook synthesiser", FRAME_RATE, NQ, CODE_BITS),
        ("only the first codebook of those", FRAME_RATE, 1, CODE_BITS),
        ("speech tokens of the single-token synthesiser", S_RATE, 1, math.log2(S_TOK)),
    ]
    body = []
    for name, rate, per, bits in rows:
        body.append((name, f"{rate:g}", per, f"{bits:.3f}".rstrip("0").rstrip("."), _g(rate * per, 1).rstrip("0").rstrip("."),
                     _g(rate * per * bits, 1).rstrip("0").rstrip(".")))
    return ("The representations a text-to-speech system passes through, per second of speech: steps a second, numbers per step, "
            "bits per number (a codebook of 2048 entries is 11 bits, 6561 speech tokens are log2 6561 bits), and the product of "
            "each.\n\n"
            + _table(["representation", "steps per second", "numbers per step", "bits per number", "numbers per second",
                      "bits per second"], body, "lrrrrr"))


# ================================================================== the prompt of the codec language model

def _prefix_rows(language, speaker):
    """Rows before the text: the role header of 3, then one row for each of the codec prefix rows but the last, then the first
    text row (non-streaming prompt of the x-vector form)."""
    tag = 4 if language else 3
    m = tag + (1 if speaker else 0) + 2
    return tag, m, 3 + (m - 1) + 1


def text_channel():
    n_text, ref_frames, ref_text = 6, 125, 20
    rows = []
    for name, language, speaker in (("no voice, language 'auto'", False, False), ("x-vector voice, a named language", True, True)):
        tag, m, p = _prefix_rows(language, speaker)
        rows.append((name, tag, int(speaker), m, p, n_text, f"frames 0 .. {n_text - 1}"))
    tag, m, _ = _prefix_rows(True, True)
    codec_lens = 1 + ref_frames
    text_lens = ref_text + n_text + 1
    pre = 3 + (m - 1)
    rows.append((f"in-context clone, {ref_frames} reference frames", tag, 1, m, pre + codec_lens, 1,
                 f"text rows ({text_lens}) are summed onto the {codec_lens} reference rows, the rest is padding"))
    head = ("A prompt in the order the code builds it. The codec side is m rows: the think or no-think tag (3 rows, or 4 with a "
            "language row), an optional speaker row, then the padding row and the begin-of-speech row. The text side is the role "
            "header of 3 rows, then the first m - 1 codec rows each summed with a text-side row (padding, and the begin row of the "
            "text on the last of them), then the first text token summed with the codec side's begin-of-speech row. That is "
            "3 + (m - 1) + 1 = m + 3 rows before frame 0. The in-context form puts 3 + (m - 1) rows first and then the reference "
            "block, one row for each reference frame and one for the begin row. 'Trailing rows' are the text rows still to be fed, "
            "one per frame. The target text has 6 tokens, the reference text 20.\n\n")
    t1 = _table(["form", "tag rows", "speaker rows", "m", "rows before frame 0", "trailing rows", "text read by"], rows, "lrrrrrl")
    frames = []
    trailing = n_text  # tokens 2 .. n_text, then the end-of-text row
    for s in range(0, 9):
        if s < n_text - 1:
            src = f"text token {s + 2}"
        elif s == n_text - 1:
            src = "end-of-text row"
        else:
            src = "padding row"
        frames.append((s, f"the sum of the {NQ} codec rows of frame {s}", src))
    t2 = _table(["frame fed back", "codec part of the row", "text part of the row"], frames, "lll")
    return (head + t1 + "\n\nThe row the Talker reads after frame s is generated, for the x-vector form above (target text of 6 "
            f"tokens, so {trailing} trailing rows):\n\n" + t2)


def label_alignment():
    codes = [[5, 1, 2, 3], [7, 4, 0, 6], [9, 8, 1, 2]]
    nq = 4
    rows = []
    for p in range(len(codes) - 1):
        rows.append((f"Talker position {p}", f"sum of the codes of frame {p}", f"first code of frame {p + 1}", codes[p + 1][0]))
    rows.append(("Talker position 2", "sum of the codes of frame 2", "nothing (the end row has no label here)", "-"))
    mtp = []
    for f in range(len(codes)):
        for k in range(1, nq):
            prev = f"code {k - 1} of frame {f}"
            if k == 1:
                prev += " (after the Talker's hidden state at position 0)"
            mtp.append((f"frame {f}, position {k}", prev, f"code {k} of frame {f}", codes[f][k]))
    doubled = [codes[p + 2][0] for p in range(len(codes) - 2)]
    return ("Three frames of four codes, codes[f][q] with the first code q = 0: frame 0 is (5, 1, 2, 3), frame 1 (7, 4, 0, 6), frame 2 "
            "(9, 8, 1, 2). The first-code model reads one summed row per frame and is scored once against the next frame's first code "
            "(one shift). The residual model of a frame reads a short sequence whose position k holds code k - 1 of the frame (position 0 holds "
            "the Talker's hidden state) and is scored against code k of the same frame (no shift).\n\n"
            + _table(["row read", "its input", "its target", "target value"], rows, "llll") + "\n\n"
            + _table(["row read", "input at this position", "its target", "target value"], mtp, "llll")
            + f"\n\nA second shift in the loss would score Talker position 0 against frame 2's first code, {doubled[0]}, and leave "
              "position 1 with no target; the first-code targets would be a list of length 1 instead of 2.")


# ================================================================== counts of the 16-codebook synthesiser

def _layer_params(d, hq, hkv, ff, hd, qk_norm=True):
    return d * hq + 2 * d * hkv + hq * d + 3 * d * ff + 2 * d + (2 * hd if qk_norm else 0)


def _layer_macs(d, hq, hkv, ff):
    return d * hq + 2 * d * hkv + hq * d + 3 * d * ff


def _talker_numbers():
    hq, hkv = T_HEADS * T_HEAD_DIM, T_KV_HEADS * T_HEAD_DIM
    per_p = _layer_params(T_D, hq, hkv, T_FF, T_HEAD_DIM)
    per_m = _layer_macs(T_D, hq, hkv, T_FF)
    return hq, hkv, per_p, per_m


def talker_counts():
    hq, hkv, per_p, per_m = _talker_numbers()
    # the Talker: 28 layers, its tables, its two text-side matrices
    tab_codec = T_VOCAB * T_D
    tab_text = T_TEXT_VOCAB * T_TEXT_HIDDEN
    proj = T_TEXT_HIDDEN * T_TEXT_HIDDEN + T_TEXT_HIDDEN + T_TEXT_HIDDEN * T_D + T_D
    talker_params = T_LAYERS * per_p + T_D + 2 * tab_codec + tab_text + proj
    mtp_params = M_LAYERS * per_p + T_D + (NQ - 1) * 2 * M_VOCAB * T_D
    talker_layer_macs = T_LAYERS * per_m
    talker_head = T_D * T_VOCAB
    mtp_steps = NQ                      # one decoder step per position of a frame: the Talker's hidden state, then codes 0 .. 14
    mtp_layer_macs = mtp_steps * M_LAYERS * per_m
    mtp_heads = (NQ - 1) * T_D * M_VOCAB
    frame = talker_layer_macs + talker_head + mtp_layer_macs + mtp_heads
    rows = [
        ("Talker: 28 layers", _i(T_LAYERS * per_p), "1", _m(talker_layer_macs)),
        ("Talker: codebook-0 head (1024 -> 3072)", _i(tab_codec), "1", _m(talker_head)),
        ("Talker: embedding tables and text projection (not matrix products per frame)",
         _i(tab_codec + T_D + tab_text + proj), "0", "0.0"),
        ("code predictor: 5 layers and final norm", _i(M_LAYERS * per_p + T_D), f"{mtp_steps}", _m(mtp_layer_macs)),
        ("code predictor: 15 heads (1024 -> 2048) and 15 embedding tables", _i((NQ - 1) * 2 * M_VOCAB * T_D), "15", _m(mtp_heads)),
    ]
    tot_rows = [("one frame (80 milliseconds of speech)", _i(talker_params + mtp_params), _m(frame)),
                ("one second of speech (12.5 frames)", "", _m(frame * FRAME_RATE)),
                ("one second of speech, billions", "", _gm(frame * FRAME_RATE, 2))]
    cached = mtp_steps * M_LAYERS
    uncached = (NQ - 1) * NQ * M_LAYERS
    return ("Parameters from the tensors of the configuration (a 1024-wide, 28-layer Talker with 16 query heads and 8 key-value heads "
            "of 128, feed-forward 3072; a 5-layer code predictor of the same layer shape; layer = q, k, v and output "
            "projections, gate, up and down matrices, two norms and the two query-key norm vectors). Multiply-adds per frame count "
            "each matrix product once per row it is applied to; attention scores and norms are left out. "
            f"A layer holds q ({T_D} x {hq}), k and v ({T_D} x {hkv} each), o ({hq} x {T_D}), gate and up ({T_D} x {T_FF} each) and "
            f"down ({T_FF} x {T_D}), with no bias on any, two RMSNorm weights of {T_D} and a query norm and a key norm of {T_HEAD_DIM}: "
            f"{_i(per_p)} parameters ({_i(per_m)} of them in the matrices, which is also the layer's multiply-adds a row). The Talker's "
            f"embedding row is the codec table ({T_VOCAB} x {T_D}), the Talker's final norm ({T_D}), the text table "
            f"({T_TEXT_VOCAB} x {T_TEXT_HIDDEN}) and the text projection ({T_TEXT_HIDDEN} -> {T_TEXT_HIDDEN} -> {T_D}, both with "
            f"bias); the head row is the {T_D} x {T_VOCAB} codebook-0 head, counted separately from the codec table. The code "
            f"predictor has {M_LAYERS} such layers, a final norm, and for each of the {NQ - 1} residual codes a head and an "
            f"embedding table of {M_VOCAB} x {T_D}.\n\n"
            + _table(["component", "parameters", "decoder steps per frame", "multiply-adds per frame (millions)"], rows, "lrrr")
            + "\n\n" + _table(["total", "parameters", "multiply-adds (millions)"], tot_rows, "lrr")
            + f"\n\nThe Talker holds {_i(talker_params)} parameters and the code predictor {_i(mtp_params)}. "
              f"The code predictor runs {mtp_steps} decoder steps of {M_LAYERS} layers, {cached} layer-steps, per frame; the Talker "
              f"runs {T_LAYERS}. Re-running the whole 16-row sequence for each of the 15 codes, as the first version did, is "
              f"{uncached} layer-rows per frame, {uncached / cached:.0f} times as many.")


def _talker_frame_macs():
    hq, hkv, per_p, per_m = _talker_numbers()
    talker = T_LAYERS * per_m
    mtp = NQ * M_LAYERS * per_m
    return talker, mtp, T_D * T_VOCAB, (NQ - 1) * T_D * M_VOCAB, per_m


def kv_growth():
    talker_pos = 2 * T_LAYERS * T_KV_HEADS * T_HEAD_DIM * 4
    lm_pos = 2 * LM_LAYERS * LM_KV_HEADS * LM_HEAD_DIM * 4
    rows = [
        ("16-codebook synthesiser's Talker", FRAME_RATE, talker_pos, talker_pos * FRAME_RATE),
        ("single-token synthesiser's language model", S_RATE, lm_pos, lm_pos * S_RATE),
    ]
    body = [(n, f"{r:g}", _i(b), _i(bs), _g(bs * 10 / 1e6, 2)) for n, r, b, bs in rows]
    ctx = 1024
    return ("Bytes of attention keys and values kept per position, 32-bit floats (2 tensors x layers x key-value heads x head width "
            "x 4 bytes), per second of speech, and for 10 seconds.\n\n"
            + _table(["model", "positions per second", "bytes per position", "bytes per second of speech", "megabytes for 10 seconds"],
                     body, "lrrrr")
            + f"\n\nA cache of {ctx} Talker positions is {_i(talker_pos * ctx)} bytes and holds {ctx / FRAME_RATE:.2f} seconds of "
              "speech, the prompt included.")


# ================================================================== the logit chain

_V, _W, _EOS = 12, 4, 10                       # toy vocabulary, suppress window, end-of-speech id
_LOGITS = [2.0, 1.0, 0.5, 4.0, 0.0, 3.5, -1.0, 0.8, 1.5, 0.3, 5.0, 2.5]
_HIST = [3, 3, 5, 3]
_THETA, _TEMP, _TOPK, _TOPP = 1.25, 0.9, 5, 0.9


def _chain(allow_eos):
    z = list(_LOGITS)
    eos_logit = z[_EOS]
    masked = list(z)
    for i in range(_V - _W, _V):
        masked[i] = -math.inf
    if allow_eos:
        masked[_EOS] = eos_logit
    pen = list(masked)
    for t in sorted(set(_HIST)):
        if pen[t] != -math.inf:
            pen[t] = pen[t] / _THETA if pen[t] > 0 else pen[t] * _THETA
    scaled = [x / _TEMP for x in pen]
    order = sorted(range(_V), key=lambda i: -scaled[i])
    thr = scaled[order[_TOPK - 1]]
    topk = [x if x >= thr else -math.inf for x in scaled]
    mx = max(topk)
    e = [(math.exp(x - mx) if x != -math.inf else 0.0) for x in topk]
    z_sum = sum(e)
    ranked = sorted([i for i in range(_V) if e[i] > 0], key=lambda i: -e[i])
    cum, cut = 0.0, len(ranked)
    for r, i in enumerate(ranked):
        cum += e[i] / z_sum
        if cum >= _TOPP:
            cut = r + 1
            break
    keep = set(ranked[:cut])
    topp = [topk[i] if i in keep else -math.inf for i in range(_V)]
    mx = max(topp)
    e2 = [(math.exp(x - mx) if x != -math.inf else 0.0) for x in topp]
    s2 = sum(e2)
    return masked, pen, scaled, topk, topp, [x / s2 for x in e2]


def _show(x):
    return "-inf" if x == -math.inf else f"{x:.3f}"


def _draw(p, u):
    acc = 0.0
    last = 0
    r = u * sum(p)
    for i, pi in enumerate(p):
        if pi <= 0:
            continue
        acc += pi
        last = i
        if acc >= r:
            return i
    return last


def sampling_chain():
    out = []
    drawn = {}
    for allow in (False, True):
        masked, pen, scaled, topk, topp, prob = _chain(allow)
        rows = []
        for i in range(_V):
            rows.append((i, f"{_LOGITS[i]:.1f}", _show(masked[i]), _show(pen[i]), _show(scaled[i]), _show(topk[i]), _show(topp[i]),
                         _f(prob[i], 4)))
        out.append(_table(["id", "logit", "after the mask", "after the penalty", "after temperature", "after top-k", "after top-p",
                           "probability"], rows, "lrrrrrrr"))
        drawn[allow] = (_draw(prob, 0.40), _draw(prob, 0.97))
    return ("A toy chain: 12 ids, the last 4 (ids 8 .. 11) are the suppressed window and id 10 is the end-of-speech id; the "
            f"generated history is {_HIST}; penalty {_THETA}, temperature {_TEMP}, top-k {_TOPK}, top-p {_TOPP}. The real default "
            "penalty is 1.05, and the real window is the last 1024 of 3072 ids. The order is the code's: mask, penalty "
            "(once per distinct id, positive logits divided and negative ones multiplied), division by the temperature, top-k "
            "(entries below the k-th largest are removed, a tie at the threshold stays), top-p (the shortest prefix of the "
            "ranked probabilities whose sum reaches the cut), then a softmax over what is left and an inverse-CDF draw with one "
            "uniform u (scan the ids in order, stop at the first whose running sum reaches u times the total).\n\n"
            "End-of-speech not yet allowed (fewer than the minimum number of frames so far):\n\n" + out[0]
            + f"\n\nWith u = 0.40 the draw is id {drawn[False][0]}, with u = 0.97 it is id {drawn[False][1]}.\n\n"
            "End-of-speech allowed:\n\n" + out[1]
            + f"\n\nWith u = 0.40 the draw is id {drawn[True][0]}, with u = 0.97 it is id {drawn[True][1]}.")


def penalty_compounding():
    th = 1.05
    rows = []
    for n in (1, 2, 5, 10, 50, 100):
        f = th ** n
        rows.append((n, _g(f, 4), _g(5.0 / th, 4), _g(5.0 / f, 4)))
    shifts = [(z, _g(z * (1 - 1 / th), 3)) for z in (1, 2, 5, 10, 20)]
    return ("Repetition penalty 1.05 on a positive logit of 5.0 for a token seen n times: once per distinct token (divide by "
            "1.05 however many times it was said) against once per occurrence (divide by 1.05 n times, that is by 1.05^n).\n\n"
            + _table(["n occurrences", "1.05^n", "logit, once per token", "logit, once per occurrence"], rows, "lrrr")
            + "\n\nThe handicap of the once-per-token rule on a positive logit z is z (1 - 1/1.05), in nats:\n\n"
            + _table(["logit z", "logit lost"], shifts, "lr"))


# ================================================================== finite scalar quantisation and the tokenizer

FSQ_SCALE = 0.999
FSQ_DIMS = 8


def fsq_digit(z):
    """round-half-even(tanh(z) * 0.999) + 1"""
    return round(math.tanh(z) * FSQ_SCALE) + 1      # Python's round() is round-half-even


def fsq_index(zs):
    return sum(fsq_digit(z) * 3 ** i for i, z in enumerate(zs))


def fsq_toy():
    c = math.atanh(0.5 / FSQ_SCALE)
    rows = []
    for z in (-2.0, -0.6, -0.54, 0.0, 0.54, 0.56, 3.0):
        t = math.tanh(z)
        rows.append((z, _f(t, 5), _f(t * FSQ_SCALE, 5), fsq_digit(z) - 1, fsq_digit(z)))
    vecs = [(0.8, -0.3, 1.9, -1.2, 0.1, 0.6, -0.7, 2.4), (-0.2, -0.2, 0.3, 0.45, -3.0, 0.7, 1.0, -0.56)]
    body = []
    for v in vecs:
        ds = [fsq_digit(z) for z in v]
        body.append((", ".join(f"{z:g}" for z in v), " ".join(str(d) for d in ds), " + ".join(f"{d}x{3 ** i}" for i, d in enumerate(ds)),
                     fsq_index(v)))
    return ("One coordinate: h = round(tanh(z) x 0.999) + 1, the rounding to the nearest integer with ties to even (the "
            f"library tanh in double precision). The digit changes where 0.999 tanh z crosses 1/2, at |z| = atanh(0.5/0.999) = {c:.6f}.\n\n"
            + _table(["z", "tanh z", "0.999 tanh z", "rounded", "digit h"], rows, "lrrrr")
            + "\n\nEight coordinates make one token, the digit of coordinate i weighted by 3^i (so the first coordinate is "
              "the least significant):\n\n"
            + _table(["z (8 coordinates)", "digits h", "sum", "token"], body, "llll")
            + f"\n\nThe token space has 3^8 = {3 ** 8:,} entries, log2 of that is {math.log2(3 ** 8):.4f} bits, no table is stored and "
              "a nearest-entry search is replaced by eight roundings.")


def _mel16(n):
    return n // 160


def _tokens(frames):
    f = frames
    for _ in range(2):
        f = (f - 1) // 2 + 1
    return f


def _mel24(n):
    return n // 480


def tokenizer_geometry():
    rows = []
    for sec in (1.0, 2.5, 3.3, 7.0, 10.0):
        n16, n24 = round(sec * 16000), round(sec * 24000)
        m16, m24 = _mel16(n16), _mel24(n24)
        tk = _tokens(m16)
        keep = min(m24 // 2, tk)
        rows.append((f"{sec:g}", _i(n16), m16, tk, _i(n24), m24, m24 // 2, keep, 2 * keep, tk - keep))
    return ("A reference clip of the given length is resampled twice. At 16 kHz a 400-sample window at hop 160 (reflect padding "
            "of 200, the last frame dropped) gives floor(n / 160) frames, and two convolutions of kernel 3, stride 2 and padding 1 "
            "(output (l - 1) // 2 + 1) give the tokens. At 24 kHz a 1920-sample window at hop 480 with reflect padding of 720 "
            "on each side gives floor(n / 480) frames. The prompt keeps min(mel frames // 2, tokens) tokens and twice as many "
            "mel frames.\n\n"
            + _table(["seconds", "samples at 16 kHz", "mel frames", "tokens", "samples at 24 kHz", "mel frames", "mel // 2",
                      "tokens kept", "mel frames kept", "tokens dropped"], rows, "lrrrrrrrrr"))


def tokenizer_counts():
    d, mel, heads = 1280, 128, 20
    hd, mlp, layers = d // heads, 4 * d, 6
    conv1 = d * mel * 3 + d
    conv2 = d * d * 3 + d
    per_block = (d * d + d) + d * d + (d * d + d) + (d * d + d) + d * 31 + 2 * d * 2 + (d * mlp + mlp) + (mlp * d + d)
    head = d * 8 + 8
    total = conv1 + conv2 + layers * per_block + head
    mac_conv1 = 50 * d * mel * 3
    mac_conv2 = 25 * d * d * 3
    per_tok_block = 4 * d * d + d * 31 + 2 * d * mlp
    mac_blocks = 25 * layers * per_tok_block
    mac_head = 25 * d * 8
    rows = [
        ("two convolutions, 128 -> 1280 -> 1280 (k 3, stride 2)", _i(conv1 + conv2), "50 and 25", _m(mac_conv1 + mac_conv2)),
        (f"{layers} blocks (attention, depthwise filter k 31, MLP 1280 -> 5120 -> 1280)", _i(layers * per_block), "25", _m(mac_blocks)),
        ("projection to 8 digits", _i(head), "25", _m(mac_head)),
    ]
    total_macs = mac_conv1 + mac_conv2 + mac_blocks + mac_head
    att = []
    for sec in (3, 10, 30):
        t = sec * 25
        att.append((sec, t, _m(layers * 2 * heads * hd * t * t)))
    return ("Counted from the tensors of the released tokenizer (102 of them): weights and biases and the layer-norm gains, in the "
            "layer shapes above; one second of audio is 100 mel frames, 50 after the first convolution and 25 after the second. "
            f"Both convolutions have k 3, stride 2 and a bias. A block has width {d} and {heads} heads of {hd}: attention q, v and "
            f"output matrices with a bias, the k matrix without (each {d} x {d}), a depthwise filter of {d} x 31 along time without "
            f"bias (added to the values, counted as {d} x 31 multiply-adds a position), two LayerNorms with gain and shift, and an "
            f"MLP {d} -> {mlp} -> {d} with both biases; the projection to 8 digits has a bias.\n\n"
            + _table(["part", "parameters", "positions per second", "multiply-adds per second of audio (millions)"], rows, "lrrr")
            + f"\n\nIn all {_i(total)} parameters and {_m(total_macs)} million multiply-adds per second of reference audio "
              "before the attention scores. The scores cost 2 x heads x head width per query and key pair of each block "
              "(the encoder attends in both directions over the whole clip):\n\n"
            + _table(["seconds", "tokens", "attention multiply-adds of the clip (millions)"], att, "rrr"))


# ================================================================== the language model over speech tokens

def _lm_numbers():
    d, hq, hkv, ff = LM_D, LM_HEADS * LM_HEAD_DIM, LM_KV_HEADS * LM_HEAD_DIM, LM_FF
    per_p = d * hq + hq + 2 * (d * hkv + hkv) + hq * d + 3 * d * ff + 2 * d
    per_m = d * hq + 2 * d * hkv + hq * d + 3 * d * ff
    return per_p, per_m


def speech_lm_counts():
    per_p, per_m = _lm_numbers()
    embed = LM_TEXT_VOCAB * LM_D
    backbone = LM_LAYERS * per_p + embed + LM_D
    speech_emb = S_VOCAB_V2 * LM_D
    dec = S_VOCAB_V2 * LM_D + S_VOCAB_V2
    llm_emb = 2 * LM_D
    total = backbone + speech_emb + dec + llm_emb
    macs_tok = LM_LAYERS * per_m + S_VOCAB_V2 * LM_D
    rows = [
        ("24 layers (grouped-query attention 14 query heads, 2 of 64, feed-forward 4864)", _i(LM_LAYERS * per_p)),
        ("text embedding table, tied (151,936 x 896) and final norm", _i(embed + LM_D)),
        ("speech embedding (6564 x 896), speech head (896 -> 6564 with bias), start and task rows", _i(speech_emb + dec + llm_emb)),
        ("total", _i(total)),
    ]
    # the example prompt
    ref_sec, ref_text, tgt = 3.3, 10, 8
    ref_tokens = min(_mel24(round(ref_sec * 24000)) // 2, _tokens(_mel16(round(ref_sec * 16000))))
    prefix = 1 + (ref_text + tgt) + 1 + ref_tokens
    mx = _round_half_up(tgt * 20.0)
    mn = _round_half_up(tgt * 2.0)
    ctx = prefix + mx + 8
    bud = []
    for n in (1, 4, 8, 20, 60):
        a, b = _round_half_up(n * 2.0), _round_half_up(n * 20.0)
        bud.append((n, a, f"{a / S_RATE:.2f}", b, f"{b / S_RATE:.2f}"))
    return ("Counted from the tensors of the released speech-token model (295 in the file; the 151,936-wide output matrix of the "
            "text backbone is a copy of the embedding table and is not read). "
            f"A layer has width {LM_D}, {LM_HEADS} query heads and {LM_KV_HEADS} key-value heads of {LM_HEAD_DIM}: q ({LM_D} x "
            f"{LM_HEADS * LM_HEAD_DIM}), k and v ({LM_D} x {LM_KV_HEADS * LM_HEAD_DIM}) each with a bias, o without, gate, up and down of "
            f"{LM_D} x {LM_FF} without bias, two RMSNorm weights of {LM_D}, no query or key norm; the 'start and task rows' are two "
            f"rows of {LM_D}.\n\n"
            + _table(["part", "parameters"], rows, "lr")
            + f"\n\nOne step of the model is {_m(macs_tok)} million multiply-adds (24 layers and the 6564-way head), so a second of "
              f"speech at 25 tokens costs {_gm(macs_tok * S_RATE, 2)} billion, attention scores aside.\n\n"
            f"A prompt: reference of {ref_sec} seconds ({ref_tokens} speech tokens after the truncation of the last table), "
            f"{ref_text} reference-text tokens and {tgt} target-text tokens. The rows it feeds before the first draw are "
            f"1 + {ref_text + tgt} + 1 + {ref_tokens} = {prefix}; the model may draw at most {mx} tokens "
            f"({mx / S_RATE:.1f} seconds) and is held from stopping before {mn}; the cache is sized for {prefix} + {mx} + 8 = {ctx} "
            "positions (the 8 is a margin that the pipeline adds to the cache; the rows actually written never exceed "
            f"{prefix} + {mx}).\n\nThe budget as a function of the number of target-text tokens n (minimum round(2 n), maximum "
            "round(20 n), ties upward):\n\n"
            + _table(["target text tokens n", "minimum tokens", "seconds", "maximum tokens", "seconds"], bud, "rrrrr"))


def _round_half_up(x):
    return int(math.floor(x + 0.5))


_RAS_LOGITS = [3.0, 2.2, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0]
_RAS_HIST = [4, 1, 6, 1, 2, 0, 5, 1, 7, 3]


def _softmax(x):
    m = max(x)
    e = [math.exp(v - m) if v != -math.inf else 0.0 for v in x]
    s = sum(e)
    return [v / s for v in e]


def _multinomial(p, u):
    total = sum(p)
    r = u * total
    acc = 0.0
    for i, pi in enumerate(p):
        acc += pi
        if r <= acc:
            return i
    return len(p) - 1


def _nucleus(logits, top_p, top_k):
    p = _softmax(logits)
    order = sorted(range(len(p)), key=lambda i: -p[i])      # stable: equal probabilities keep their order
    sel, cum = [], 0.0
    for i in order:
        if cum < top_p and len(sel) < top_k:
            cum += p[i]
            sel.append(i)
        else:
            break
    return sel, [p[i] for i in sel]


def ras_demo():
    top_p, top_k, win, tau = 0.8, 25, 10, 0.1
    p = _softmax(_RAS_LOGITS)
    sel, sp = _nucleus(_RAS_LOGITS, top_p, top_k)
    rows = [(i, f"{_RAS_LOGITS[i]:.1f}", _f(p[i], 4), "yes" if i in sel else "no", _RAS_HIST[-win:].count(i)) for i in range(8)]
    out = []
    for u1, u2 in ((0.05, 0.60), (0.70, 0.60), (0.90, 0.20)):
        first = sel[_multinomial(sp, u1)]
        count = _RAS_HIST[-win:].count(first)
        if count >= win * tau:
            logits2 = list(_RAS_LOGITS)
            logits2[first] = -math.inf
            second = _multinomial(_softmax(logits2), u2)
            out.append((f"{u1:.2f}", first, count, "yes", f"{u2:.2f}", second, second))
        else:
            out.append((f"{u1:.2f}", first, count, "no", "-", "-", first))
    return ("Eight speech tokens with the logits below (a softmax of the log-softmax is the softmax, so the logits are used "
            f"directly), the last {win} generated tokens {_RAS_HIST}, nucleus 0.8 over at most 25 candidates, and a repetition "
            f"window of {win} with threshold {win} x {tau} = {win * tau:g} occurrence. The nucleus takes ids in order of "
            "probability while the sum so far is below 0.8; a draw is one uniform u, scanned over the kept probabilities in "
            "that order (stop at the first whose running sum reaches u times the total).\n\n"
            + _table(["id", "logit", "probability", "in the nucleus", "times in the last 10"], rows, "lrrll")
            + f"\n\nThe nucleus is {sel} with probabilities {[round(x, 4) for x in sp]}.\n\n"
            + _table(["u1", "first draw", "its count in the window", "redrawn", "u2", "second draw (all ids, first one removed)",
                      "token emitted"], out, "rrrlrrr"))


# ================================================================== the flow decoder

def _cos_grid(n):
    return [1.0 - math.cos(i / n * 0.5 * math.pi) for i in range(n + 1)]


def cfm_grid():
    n = 10
    g = _cos_grid(n)
    rows = []
    for i in range(n + 1):
        dt = g[i + 1] - g[i] if i < n else None
        rows.append((i, f"{i / n:.1f}", _f(g[i], 6), _f(dt, 6) if dt is not None else "", "0.100000" if dt is not None else ""))
    first, last = g[1] - g[0], g[n] - g[n - 1]
    return ("The grid t_i = 1 - cos(pi (i / 10) / 2), from t = 0 (noise) to t = 1 (data), against the uniform step 0.1.\n\n"
            + _table(["i", "i / 10", "t_i", "step t_(i+1) - t_i", "uniform step"], rows, "lrrrr")
            + f"\n\nThe last step is {last / first:.2f} times the first: {_f(first, 6)} against {_f(last, 6)}.")


_L = (-0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9)          # a 'mel level' for each of 8 toy tokens


def _spk(xvec):
    n = math.sqrt(sum(v * v for v in xvec))
    u = [v / n for v in xvec]
    w, b = (0.5, 0.25, -0.5), 0.1
    return u, sum(wi * ui for wi, ui in zip(w, u)) + b


def cfm_condition():
    name = "speech_synthesis.condition_noise"
    g = bookrng.generator(name)
    tokens = (5, 2, 7)           # the first is the prompt's
    mu = [_L[t] for t in tokens for _ in range(2)]
    prompt = [0.55, 0.45]
    cond = prompt + [0.0] * 4
    xvec = (3.0, 4.0, 0.0)
    u, s = _spk(xvec)
    x0 = [g.normal() for _ in range(6)]
    rows = []
    for f in range(6):
        rows.append((f, "prompt" if f < 2 else "generated", _f(x0[f], 4), _f(mu[f], 2), _f(s, 2), _f(cond[f], 2)))
    return (f"The conditions of the flow decoder for 3 toy tokens (5, 2, 7), the first of them the reference's: the level of token "
            f"id i is {list(_L)}[i], and each token becomes 2 frames. The 3-number speaker vector {xvec} is scaled to unit length, "
            f"{[round(v, 3) for v in u]}, and mapped by the weights (0.5, 0.25, -0.5) and the bias 0.1 to the one number "
            f"{s:.2f}, which is repeated at every frame. The two prompt frames carry their mel values {prompt} in the condition "
            f"channel and the rest carries 0. The starting values are 6 draws of normal() from {_seed(name)}, in frame order. "
            "At t = 0 the network reads one row of four channels per frame:\n\n"
            + _table(["frame", "role", "x (noise)", "mu (from the tokens)", "speaker", "condition"], rows, "llrrrr")
            + "\n\nOnly the last 4 frames are kept as the mel the vocoder will read.")


_TOY_MU, _TOY_S = 1.5, 0.3                   # the conditional target N(1.5, 0.3^2); the unconditional one is N(0, 1)
_GUIDE = 1.7                                  # 1 + the guidance rate 0.7


def _velocity(x, t, mu, s):
    """exact marginal velocity of the straight path x_t = (1 - t) z + t y, z ~ N(0,1), y ~ N(mu, s^2): E[y - z | x_t = x]"""
    var = (1 - t) ** 2 + t * t * s * s
    return mu + (t * s * s - (1 - t)) / var * (x - t * mu)


def _guided(x, t):
    uc = _velocity(x, t, _TOY_MU, _TOY_S)
    un = _velocity(x, t, 0.0, 1.0)
    return un + _GUIDE * (uc - un)


def _euler(x, grid, f):
    for i in range(len(grid) - 1):
        x = x + (grid[i + 1] - grid[i]) * f(x, grid[i])
    return x


def cfm_toy():
    name = "speech_synthesis.flow_start"
    g = bookrng.generator(name)
    zs = [g.normal() for _ in range(6)]
    grid = _cos_grid(10)
    uni = [i / 10 for i in range(11)]
    fine = [i / 2000 for i in range(2001)]
    cond = lambda x, t: _velocity(x, t, _TOY_MU, _TOY_S)
    z = zs[0]
    xc = xg = z
    steps = []
    for i in range(10):
        dt = grid[i + 1] - grid[i]
        t = grid[i]
        vc, vg = cond(xc, t), _guided(xg, t)
        steps.append((i, _f(t, 4), _f(dt, 4), _f(xc, 4), _f(vc, 4), _f(xg, 4), _f(_velocity(xg, t, 0.0, 1.0), 4), _f(vg, 4)))
        xc, xg = xc + dt * vc, xg + dt * vg
    rows = []
    for z in zs:
        exact = _TOY_MU + _TOY_S * z
        rows.append((_f(z, 4), _f(exact, 4), _f(_euler(z, grid, cond), 4), _f(_euler(z, uni, cond), 4),
                     _f(_euler(z, grid, _guided), 4), _f(_euler(z, fine, _guided), 4)))
    g0, g1 = _euler(0.0, fine, _guided), _euler(1.0, fine, _guided)
    worst_c = max(abs(_euler(z, grid, cond) - (_TOY_MU + _TOY_S * z)) for z in zs)
    worst_u = max(abs(_euler(z, uni, cond) - (_TOY_MU + _TOY_S * z)) for z in zs)
    return (f"One mel value, one frame. The target given the tokens and the speaker is N({_TOY_MU}, {_TOY_S}^2) and the model "
            "without conditions knows only N(0, 1). Both fields are the exact marginal velocity of the straight path "
            "x_t = (1 - t) z + t y, u(x, t) = m + (t sd^2 - (1 - t)) / ((1 - t)^2 + t^2 sd^2) (x - t m) for a target of mean m and "
            "standard deviation sd, which makes the "
            f"example free of any trained network. Guidance is u_un + {_GUIDE} (u_c - u_un) (rate 0.7 in the form "
            "(1 + 0.7) u_c - 0.7 u_un). Starting values: 6 draws of normal() from "
            f"{_seed(name)}, in order; each row is one start, run through 10 Euler steps on the cosine grid.\n\n"
            "The first start, step by step (conditional only, then guided):\n\n"
            + _table(["step", "t", "dt", "x, conditional", "u_c", "x, guided", "u_un at the guided x", "u guided"], steps, "rrrrrrrr")
            + "\n\nEnd points of all six starts:\n\n"
            + _table(["start z", "exact map m + sd z", "10 steps, cosine grid", "10 steps, uniform grid",
                      "guided, 10 steps, cosine grid", "guided, 2000 uniform steps"], rows, "rrrrrr")
            + f"\n\nThe largest distance of the 10-step conditional run from the exact map is {worst_c:.4f} on the cosine grid and "
              f"{worst_u:.4f} on the uniform grid. The field is linear in x, so the guided map from start to end point is a "
              f"straight line, found here from the starts 0 and 1 with 2000 steps: end = {g0:.4f} + {g1 - g0:.4f} z, against "
              f"{_TOY_MU} + {_TOY_S} z without guidance. Guided runs cost 20 evaluations of the field, conditional ones 10.")


# ================================================================== counts of the flow decoders and the vocoder

def _lin(i, o, b=True):
    return i * o + (o if b else 0)


def _conv(ci, co, k):
    return ci * co * k + co


def _ln(d):
    return 2 * d


def _cv2_parts():
    d, ffe = 512, 2048
    conf = 4 * _lin(d, d) + d * d + 2 * d + _lin(d, ffe) + _lin(ffe, d) + _ln(d) + _ln(d)

    def resnet(ci, co):
        return _lin(1024, co) + _conv(ci, co, 3) + _ln(co) + _conv(co, co, 3) + _ln(co) + _conv(ci, co, 1)

    def block():
        return _ln(256) + 3 * 256 * 512 + _lin(512, 256) + _ln(256) + _lin(256, 1024) + _lin(1024, 256)

    enc = (_conv(d, d, 4) + _conv(d, d, 3) + _lin(d, d) + _ln(d) + 6 * conf + _conv(d, d, 5) + _lin(d, d) + _ln(d) + 4 * conf + _ln(d))
    resnets = resnet(320, 256) + 12 * resnet(256, 256) + resnet(512, 256)
    blocks = 56 * block()
    rest = _lin(320, 1024) + _lin(1024, 1024) + 2 * _conv(256, 256, 3) + _conv(256, 256, 3) + _ln(256) + _conv(256, 80, 1)
    head = S_TOK * 512 + _lin(192, 80) + _lin(512, 80)
    return head, enc, blocks, resnets, rest


def _cv2_frame_macs():
    """matrix products per mel frame per evaluation of the estimator, and the encoder's per second of speech (run once)"""
    def resnet(ci, co):
        return 3 * ci * co + 3 * co * co + ci * co
    blocks = 56 * (3 * 256 * 512 + 512 * 256 + 2 * 256 * 1024)
    resnets = resnet(320, 256) + 12 * resnet(256, 256) + resnet(512, 256)
    convs = 3 * 256 * 256 * 3 + 256 * 80
    per_eval_fixed = (320 * 1024 + 1024 * 1024) + 14 * 1024 * 256
    d = 512
    layer = 4 * d * d + 2 * d * d + 2 * d * 2048          # q, k, v, o, the 2T-row position projection, feed-forward
    tok_rate = d * d * 4 + d * d * 3 + d * d + 6 * layer
    up_rate = d * d * 5 + d * d + 4 * layer + d * 80
    return blocks + resnets + convs, per_eval_fixed, S_RATE * tok_rate + MEL_RATE * up_rate, blocks


def _cv3_parts():
    D = 1024
    blk_ada, blk_rest = _lin(D, 6 * D), 4 * _lin(D, D) + _lin(D, 2048) + _lin(2048, D)
    head = S_TOK * 80 + _lin(192, 80) + _conv(80, 1024, 4) + _conv(1024, 80, 3)
    inp = _lin(320, D) + 2 * (D // 16 * D * 31 + D)
    time = _lin(256, D) + _lin(D, D)
    fin = _lin(D, 2 * D) + _lin(D, 80)
    return head, time, inp, 22 * blk_ada, 22 * blk_rest, fin


def _cv3_frame_macs():
    D = 1024
    blocks = 22 * (4 * D * D + 2 * D * 2048)
    inp = 320 * D + 2 * (D // 16 * D * 31)
    per_frame = blocks + inp + D * 80
    fixed = 22 * D * 6 * D + D * 2 * D + 256 * D + D * D
    return per_frame, fixed


def flow_layout():
    """The layers behind every row of flow_counts' parameter tables."""
    d, ffe = 512, 2048
    rows2 = [
        ("token embedding, speaker map, output projection", f"embedding {S_TOK} x {d}; speaker vector 192 -> 80; encoder output {d} -> 80",
         "the two linear maps have a bias"),
        ("conformer encoder", f"a linear {d} -> {d} with a LayerNorm; a look-ahead pair of convolutions {d} -> {d} with k 4 (reading 3 "
         f"later rows) and k 3, added to its input; 6 layers at 25 Hz; every row repeated twice (50 Hz) and a convolution "
         f"{d} -> {d} with k 5, a linear {d} -> {d} with a LayerNorm; 4 layers at 50 Hz; a final LayerNorm. A layer, with its "
         f"LayerNorm before the attention and before the feed-forward (no convolution module): {d // 64} heads of 64, q, k, v and o "
         f"of {d} x {d} with bias, a relative-position matrix {d} x {d} without bias, two vectors u and v of {d}, one "
         f"feed-forward {d} -> {ffe} -> {d} with both biases, two LayerNorms (weight and bias)", "every convolution and linear map biased except the position matrix"),
        ("estimator", "input 320 channels (the noisy mel, the token mel, the speaker vector repeated and the reference mel, 80 each); "
         "a time embedding of a sinusoid of width 320 through 320 -> 1024 -> 1024; 14 residual blocks (one going down, 12 in the "
         "middle, one going up, whose input is 512 channels, the middle output and the skip from the down stage), each followed by "
         "4 transformer blocks, so 56 in all; a convolution 256 -> 256 (k 3) after the down stage and one after the up stage, "
         "a final convolution 256 -> 256 (k 3), a LayerNorm, Mish and a convolution 256 -> 80 (k 1)", "all convolutions and linear maps biased"),
        ("residual block (ci -> co)", "convolution ci -> co (k 3), LayerNorm, Mish; the time embedding through a projection 1024 -> co "
         "added to every position; convolution co -> co (k 3), LayerNorm, Mish; shortcut convolution ci -> co (k 1) added; "
         "320 -> 256 once, 256 -> 256 twelve times, 512 -> 256 once",
         "every convolution, LayerNorm and the time projection with bias"),
        ("transformer block of the estimator", "LayerNorm 256; attention with 8 heads of 64, q, k, v 256 -> 512 without bias, o 512 -> 256 "
         "with bias; LayerNorm 256; feed-forward 256 -> 1024 -> 256", "LayerNorm weight and bias, the feed-forward and o biased"),
    ]
    D = 1024
    rows3 = [
        ("token embedding, speaker map, look-ahead layer", f"embedding {S_TOK} x 80; speaker vector 192 -> 80; convolution 80 -> {D} (k 4) "
         f"and convolution {D} -> 80 (k 3)", "biased"),
        ("time embedding", f"a sinusoid of width 256 through 256 -> {D} -> {D}", "biased"),
        ("input projection and position convolutions", f"320 -> {D} (the same four 80-channel pieces); then two convolutions of "
         f"{D} channels with k 31 in 16 groups (each channel group of {D // 16} reads {D // 16} channels)", "all three biased"),
        ("22 blocks", f"width {D}, 16 heads of 64 with rotary positions, a LayerNorm without gain or shift before attention and before the feed-forward, "
         f"whose scale, shift and gate come from the time embedding by one matrix {D} -> {6 * D} per block; attention q, k, v, o "
         f"({D} x {D}); feed-forward {D} -> 2048 -> {D}", "all matrices biased; the LayerNorms have no parameters"),
        ("final scale-and-shift and output", f"a matrix {D} -> {2 * D} from the time embedding, then {D} -> 80", "biased"),
    ]
    return ("The layers behind the counts below: widths, heads, kernels, groups and which tensors carry a bias. A LayerNorm "
            "has a weight and a bias of its width unless said otherwise.\n\nSecond generation\n\n"
            + _table(["part", "layout", "biases and norms"], rows2, "lll")
            + "\n\nThird generation\n\n" + _table(["part", "layout", "biases and norms"], rows3, "lll"))


def flow_counts():
    h2, enc, blocks, resnets, rest = _cv2_parts()
    tot2 = h2 + enc + blocks + resnets + rest
    h3, time3, inp3, ada3, blk3, fin3 = _cv3_parts()
    tot3 = h3 + time3 + inp3 + ada3 + blk3 + fin3
    t1 = _table(["part of the second-generation decoder (convolutional estimator)", "parameters"], [
        ("token embedding (6561 x 512), speaker map (192 -> 80), output projection (512 -> 80)", _i(h2)),
        ("conformer encoder: 6 layers at 25 Hz, upsampling by 2, 4 layers at 50 Hz", _i(enc)),
        ("estimator: 56 transformer blocks of width 256", _i(blocks)),
        ("estimator: 14 residual blocks", _i(resnets)),
        ("estimator: time embedding and final convolutions", _i(rest)),
        ("total", _i(tot2))], "lr")
    t2 = _table(["part of the third-generation decoder (transformer estimator)", "parameters"], [
        ("token embedding (6561 x 80), speaker map, look-ahead layer (80 -> 1024 -> 80)", _i(h3)),
        ("time embedding (256 -> 1024 -> 1024)", _i(time3)),
        ("input projection (320 -> 1024) and two grouped position convolutions", _i(inp3)),
        ("22 blocks: scale-and-shift matrices (1024 -> 6144) from the time embedding", _i(ada3)),
        ("22 blocks: attention (4 matrices of 1024 x 1024) and feed-forward (1024 -> 2048 -> 1024)", _i(blk3)),
        ("final scale-and-shift and output projection (1024 -> 80)", _i(fin3)),
        ("total", _i(tot3))], "lr")
    est2, fixed2, enc2, _ = _cv2_frame_macs()
    est3, fixed3 = _cv3_frame_macs()
    steps, cfg = 10, 2
    evals = steps * cfg
    T = 650
    att2 = 56 * 2 * T * 512
    att3 = 22 * 2 * T * 1024
    rows = []
    for name, est, fixed, att, once in (("second generation (convolutional estimator)", est2, fixed2, att2, enc2),
                                        ("third generation (transformer estimator)", est3, fixed3, att3, 0)):
        per_s = MEL_RATE * evals * est
        rows.append((name, _m(est), _m(att), _m(fixed), evals, _gm(per_s, 1), _gm(MEL_RATE * evals * att, 1),
                     _gm(once, 2) if once else "0"))
    t3 = _table(["decoder", "matrix products per frame per evaluation (millions)",
                 "attention scores and values per frame at T = 650 (millions)",
                 "per evaluation, independent of T (millions)", "evaluations", "billions per second of speech",
                 "attention, billions per second at T = 650", "encoder, once (billions per second)"], rows, "lrrrrrrr")
    return ("Parameters counted from the tensor shapes of the two released flow files (the first matches the total of the "
            f"released file's 1121 tensors, {_i(tot2)}; the second matches its 330 tensors apart from a 32-entry rotary-frequency buffer "
            f"that is recomputed, {_i(tot3)} + 32 = {_i(tot3 + 32)}).\n\n" + t1 + "\n\n" + t2
            + "\n\nWork per second of speech: 50 mel frames a second, 10 Euler steps, and two evaluations of the estimator per step "
              "(the conditional one and the unconditioned one), so "
              f"{evals} evaluations of every frame. Matrix products are counted once per row they are applied to; attention costs "
              "2 x width per pair of frames and block, where width is heads x head width: 8 x 64 = 512 inside each of the 56 "
              "blocks of the second generation's estimator (the block's own width is 256) and 16 x 64 = 1024 in each of the 22 blocks "
              "of the third generation's, so a pair costs 2 x 512 or 2 x 1024 multiply-adds, scores and weighted values together, "
              "and the attention column is blocks x 2 x width x T; norms and activations are left out. The conformer's "
              "relative-position projection is counted over 2 rows per frame. The 'independent of T' column per evaluation is, "
              "for the second generation, the time embedding (320 -> 1024 -> 1024) and the 14 time projections (1024 -> 256), "
              "and for the third the 22 scale-and-shift matrices (1024 -> 6144), the final one (1024 -> 2048) and the time "
              "embedding (256 -> 1024 -> 1024); the third's 'matrix products per frame' are its 22 blocks (4 attention matrices of "
              "1024 x 1024 and the feed-forward 1024 -> 2048 -> 1024), the input projection with the two grouped position "
              "convolutions and the output projection.\n\n" + t3)


def vocoder_geometry():
    t = MEL_RATE                                  # one second of mel
    lens = [t]
    for u in HIFT_UP:
        lens.append(lens[-1] * u)
    lens[-1] += 1                                 # the reflection pad of the last stage
    frames = lens[-1]
    samples = (frames - 1) * HIFT_HOP
    rows = [("mel frames in", t, f"{MEL_RATE} per second"),
            ("after upsampling by 8", lens[1], f"{lens[1]} per second"),
            ("after upsampling by 5", lens[2], f"{lens[2]} per second"),
            ("after upsampling by 3 and one mirrored sample", lens[3], f"{lens[3]} frames of the inverse transform"),
            ("inverse transform, window 16, hop 4", samples, f"{FS} samples per second")]
    f0 = 200.0
    harm = [(h, _g(f0 * h, 0), _f(f0 * h / FS, 5)) for h in range(1, 10)]
    return ("One second of mel (50 frames) through the vocoder's trunk. Each upsampling stage is a transposed convolution whose "
            "kernel (16, 11, 7) minus its stride (8, 5, 3) is even, padded by half of that difference, so it multiplies the length "
            "by the stride exactly; the last stage mirrors one sample in front, and an inverse short-time transform "
            "with a 16-point window and hop 4 turns the 6001 frames of 9 complex bins into (6001 - 1) x 4 samples.\n\n"
            + _table(["stage", "length", "rate"], rows, "lrl")
            + "\n\nThe harmonic source for a pitch of 200 Hz: 9 sinusoids at multiples of the pitch, each advancing its phase by "
              "(frequency / 24000) cycles per sample, and the excitation rule of the source: a frame is voiced when the pitch is "
              "above 10 Hz, a voiced sinusoid has amplitude 0.1 and noise of standard deviation 0.003, an unvoiced sample is noise "
              "of standard deviation 0.1 / 3 = 0.0333.\n\n"
            + _table(["harmonic", "frequency (Hz)", "cycles per sample"], harm, "rrr"))


def _vocoder_rows():
    """(part, parameters, multiply-adds per second of speech)"""
    base = 512
    chs = (256, 128, 64)
    ups = ((512, 256, 16), (256, 128, 11), (128, 64, 7))
    srcd, src_k, res_k = (30, 6, 1), (7, 7, 11), (3, 7, 11)
    rin = (MEL_RATE, MEL_RATE * 8, MEL_RATE * 40)
    rout = (MEL_RATE * 8, MEL_RATE * 40, MEL_RATE * 120)

    def res_p(c, k):
        return 6 * (c * c * k + c) + 6 * c
    return [
        ("pitch predictor: 5 convolutions (k 3) at 50 Hz, then a linear read-out",
         _conv(80, base, 3) + 4 * _conv(base, base, 3) + base + 1, MEL_RATE * (80 * base * 3 + 4 * base * base * 3 + base)),
        ("first convolution (80 -> 512, k 7) at 50 Hz", _conv(80, base, 7), MEL_RATE * 80 * base * 7),
        ("3 transposed convolutions (512 -> 256, 256 -> 128, 128 -> 64)", sum(_conv(ci, co, k) for ci, co, k in ups),
         sum(r * ci * co * k for r, (ci, co, k) in zip(rin, ups))),
        ("source branch: 3 strided convolutions of the excitation's 18 channels", sum(_conv(18, c, k) for c, k in zip(chs, srcd)),
         sum(r * 18 * c * k for r, c, k in zip(rout, chs, srcd))),
        ("source branch: 3 residual blocks (k 7, 7, 11), 6 convolutions and 6 gains each",
         sum(res_p(c, k) for c, k in zip(chs, src_k)), sum(r * 6 * c * c * k for r, c, k in zip(rout, chs, src_k))),
        ("trunk: 9 residual blocks (k 3, 7, 11 at each of 3 stages)", sum(res_p(c, k) for c in chs for k in res_k),
         sum(r * 6 * c * c * k for r, c in zip(rout, chs) for k in res_k)),
        ("output convolution (64 -> 18, k 7) at 6000 Hz", _conv(64, 18, 7), MEL_RATE * 120 * 64 * 18 * 7),
        ("harmonic merge (9 -> 1)", 10, FS * 9),
    ]


def vocoder_layout():
    """The layers behind every row of vocoder_counts."""
    rows = [
        ("pitch predictor", "convolution 80 -> 512 (k 3), four convolutions 512 -> 512 (k 3), each followed by an ELU, then a linear "
         "512 -> 1 whose absolute value is the pitch in Hz", "every convolution and the linear map biased"),
        ("first convolution", "80 -> 512, k 7", "biased"),
        ("upsampling stages", "before each, a leaky ReLU; transposed convolutions 512 -> 256 (k 16, stride 8), 256 -> 128 (k 11, "
         "stride 5), 128 -> 64 (k 7, stride 3), taking 50, 400 and 2000 positions a second in", "biased"),
        ("source branch", "the nine harmonic sinusoids at 24,000 samples a second are merged into one excitation by the linear map "
         "9 -> 1, and a 16-point transform with hop 4 makes of it 9 complex bins, 18 real channels, at 6000 frames a second; one "
         "strided convolution for each stage, 18 -> 256 (k 30, stride 15), 18 -> 128 (k 6, stride 3) and 18 -> 64 (k 1, stride "
         "1); a residual block on its output with k 7, 7 and 11; the result is added to the trunk after the stage's "
         "transposed convolution", "biased"),
        ("residual block (c channels, kernel k)", "three branches in sequence on the same signal, each Snake, convolution c -> c "
         "(kernel k, dilation 1, 3 or 5), Snake, convolution c -> c (kernel k, dilation 1), the branch added to its input; a "
         "Snake multiplies by a learned per-channel alpha", "six convolutions with bias and six alpha vectors of c: 6 (c x c x k + c) + 6 c"),
        ("trunk", "after each stage three residual blocks with k 3, 7 and 11 on the same input, their outputs averaged: 9 blocks in all",
         ""),
        ("output convolution", "after a leaky ReLU, 64 -> 18, k 7: for each of the 9 bins a logarithm of the magnitude and a "
         "phase argument, at 6000 frames a second, then the inverse transform", "biased"),
        ("harmonic merge", "linear map 9 -> 1 at each of the 24,000 samples a second", "biased (10 numbers)"),
    ]
    return ("The layers behind the counts below: widths, kernels, strides and which tensors carry a bias or a gain.\n\n"
            + _table(["part", "layout", "biases and gains"], rows, "lll"))


def vocoder_counts():
    rows = _vocoder_rows()
    tp = sum(r[1] for r in rows)
    tm = sum(r[2] for r in rows)
    body = [(n, _i(p), _m(m)) for n, p, m in rows]
    body.append(("total", _i(tp), _m(tm)))
    return ("Counted from the tensors of the released second-generation vocoder, after the weight-normalisation gains are folded "
            "into the kernels (the file stores one extra gain per convolution, 14,738 numbers, which are not parameters of the "
            "folded model). A convolution costs c_out c_in k multiply-adds per output position and a transposed convolution c_in "
            "c_out k per input position, at the rates 50, 400, 2000 and 6000 positions a second; the inverse transform and the "
            "activations are not counted.\n\n"
            + _table(["part", "parameters", "multiply-adds per second of speech (millions)"], body, "lrr")
            + f"\n\nThat is {_gm(tm, 1)} billion multiply-adds for each second of speech.")


# ================================================================== streaming and the work before the first sound

def _decode_frame_macs():
    return sum(m for _, _, m in ac._decoder_rows()) / FRAME_RATE


def stream_schedule():
    rows = []
    first = 8
    for n in (6, 20, 24, 38, 125, 250):
        calls, nxt = [], first
        for f in range(1, n + 1):
            if f >= nxt:
                calls.append(f)
                nxt *= 3
        final_repeats = bool(calls) and calls[-1] == n
        calls.append(n)
        total = sum(calls)
        new = [calls[0]] + [b - a for a, b in zip(calls, calls[1:])]
        rows.append((n, f"{n / FRAME_RATE:.2f}", ", ".join(str(c) for c in calls), len(calls), total, f"{total / n:.2f}",
                     f"{(calls[0] / FRAME_RATE):.2f}", "yes" if final_repeats else "no"))
    per = _decode_frame_macs()
    return ("The streaming schedule of the device engine: the codec decodes the codes so far when the first 8 frames exist, "
            "then each time the frame count reaches 3 times the last trigger (8, 24, 72, 216), and once more on the finished clip; "
            "every decode restarts at frame 0 and only the samples not yet sent are handed out. A clip shorter than 8 frames "
            "is decoded once. One decode of n frames costs n x "
            f"{_m(per)} million multiply-adds (the decoder of the codec chapter).\n\n"
            + _table(["frames n", "seconds", "frames decoded in each call", "calls", "frames decoded in all", "ratio to n",
                      "seconds of speech before the first sound", "last call repeats the one before"], rows, "rrlrrrrl"))


def first_sound():
    per_dec = _decode_frame_macs()
    talker, mtp, head, mtp_heads, per_layer = _talker_frame_macs()
    frame = talker + head + mtp + mtp_heads
    rows = []
    ref_s, ref_frames = 10, 125
    enc = sum(m for _, _, m in ac._encoder_rows())
    ecapa = se.ecapa_counts()
    ecapa_macs = sum(r[4] for r in ecapa) * 93.75 * ref_s + sum(r[5] for r in ecapa)
    for name, prefix, extra in (("x-vector voice, vector computed now", 10, ecapa_macs),
                                ("x-vector voice, vector kept from an earlier request", 10, 0.0),
                                ("in-context clone, reference seen for the first time", 135, enc * ref_s + ecapa_macs),
                                ("in-context clone, reference conditioning kept", 135, 0.0)):
        pre = prefix * talker
        gen = 8 * frame
        dec = 8 * per_dec
        rows.append((name, prefix, _gm(extra, 1), _gm(pre, 1), _gm(gen, 1), _gm(dec, 1), _gm(extra + pre + gen + dec, 1)))
    return ("Multiply-adds before the first sound of the device engine, in billions: the reference's codes and speaker vector "
            "(a 10-second reference through the first speaker encoder, and for the in-context form also the codec's encoder), the prefix rows through the 28 "
            "layers of the Talker, 8 frames of generation (one Talker step with its head and the code predictor's 16 steps and "
            "15 heads each), and the first decode of 8 frames. Attention scores and the text projections of the prompt are not "
            "counted.\n\n"
            + _table(["case", "prefix rows", "reference analysis", "prefix", "8 frames generated", "first decode", "total"],
                     rows, "lrrrrrr"))


def stream_reference():
    per_dec_lm = LM_LAYERS * _lm_numbers()[1] + S_VOCAB_V2 * LM_D
    est3, fixed3 = _cv3_frame_macs()
    D = 1024

    def flow_macs(t_frames):
        return 20 * (t_frames * est3 + 22 * 2 * t_frames * t_frames * D + fixed3)

    pad_rows = []
    for prompt in (25, 75, 82, 150):
        pad = _ceil_div(prompt, 25) * 25 - prompt
        need = 25 + pad + 3
        pad_rows.append((prompt, pad, 25 + pad, need, f"{need / S_RATE:.2f}"))
    prompt, n_tok = 75, 250
    hop, offset, rows = 25, 0, []
    pad = _ceil_div(prompt, 25) * 25 - prompt
    cum = 0.0
    chunks = []
    this = hop + pad
    while n_tok - offset >= this + 3:
        t = 2 * (prompt + offset + this + 3)
        chunks.append((offset, this, prompt + offset + this + 3, t))
        offset += this
        hop = min(100, hop * 2)
        this = hop
    t_final = 2 * (prompt + n_tok)
    chunks.append((offset, n_tok - offset, prompt + n_tok, t_final))
    tot_frames = 0
    for k, (off, hp, toks, t) in enumerate(chunks, start=1):
        m = flow_macs(t)
        cum += m
        tot_frames += t
        rows.append((k if k < len(chunks) else f"{k} (final)", off, hp, toks, t, _gm(m, 0), _gm(cum, 0)))
    one = flow_macs(2 * (prompt + n_tok))
    first_lm = (1 + 18 + 1 + prompt) * per_dec_lm + 28 * per_dec_lm
    return ("The chunk rule of the released reference's streaming loop (the third generation's class), which brain's pipeline "
            "does not implement. A chunk is cut when the tokens generated so far exceed the tokens already sent by the hop plus the "
            "3 tokens of look-ahead the flow decoder reads; the first hop is 25 tokens plus the padding that rounds the prompt up "
            "to a multiple of 25; after each chunk the hop doubles, up to 100. Each chunk runs the flow decoder again on the "
            "prompt tokens and every token up to the end of the chunk, and keeps only its new frames.\n\n"
            + _table(["prompt tokens", "padding", "first hop", "tokens needed before the first chunk", "seconds of speech generated"],
                     pad_rows, "rrrrr")
            + f"\n\nThe chunks of a 250-token utterance (10 seconds) after a prompt of {prompt} tokens. A flow evaluation of "
              "T frames costs 20 x (T x matrix products + attention + the fixed part) multiply-adds, with the "
              "third-generation numbers of the flow table: the matrix products per frame, the attention 22 blocks x 2 x 1024 x T x T "
              "(each of T frames scores T keys, 2 x 1024 multiply-adds a pair), and the fixed part per evaluation; "
              "'billions' are billions of multiply-adds. The cumulative column adds the unrounded flows, so it can differ "
              "by one from the sum of the printed ones.\n\n"
            + _table(["chunk", "tokens already sent", "hop", "tokens through the flow", "frames T", "flow billions",
                      "cumulative billions"], rows, "lrrrrrr")
            + f"\n\nThe same utterance in one pass is T = {2 * (prompt + n_tok)} frames and {_gm(one, 0)} billion; the streamed "
              f"run runs the flow on {tot_frames} frames in all, {tot_frames / (2 * (prompt + n_tok)):.2f} times as many. Before "
              f"the first sound the language model has run its {1 + 18 + 1 + prompt}-row prompt and 28 tokens, "
              f"{_gm(first_lm, 1)} billion, and the first chunk's flow is {_gm(flow_macs(chunks[0][3]), 0)} billion.")


def guidance_sweep():
    fine = [i / 2000 for i in range(2001)]
    rows = []
    for g in (1.0, 1.5, 1.7, 2.0, 2.5, 3.0):
        def field(x, t, g=g):
            uc = _velocity(x, t, _TOY_MU, _TOY_S)
            un = _velocity(x, t, 0.0, 1.0)
            return un + g * (uc - un)
        a, b = _euler(0.0, fine, field), _euler(1.0, fine, field)
        rows.append((f"{g:g}", _f(a, 4), _f(b - a, 4)))
    return ("The toy flow decoder with the guidance scale g of the first column, 2000 uniform Euler steps. The field is linear in x, "
            "so the map from the start z to the end point is a line: 'centre' is its value at z = 0 and 'slope' its change per "
            "unit of z, which is the standard deviation of the end points for a standard normal start.\n\n"
            + _table(["g", "centre", "slope"], rows, "lrr"))


# ================================================================== where the work of a second of speech goes

def work_per_second():
    talker, mtp, head, mtp_heads, _ = _talker_frame_macs()
    gen = (talker + head + mtp + mtp_heads) * FRAME_RATE
    dec = sum(m for _, _, m in ac._decoder_rows())
    per_lm = LM_LAYERS * _lm_numbers()[1] + S_VOCAB_V2 * LM_D
    lm = per_lm * S_RATE
    est2, _, enc2, _ = _cv2_frame_macs()
    flow2 = MEL_RATE * 20 * est2 + enc2
    est3, _ = _cv3_frame_macs()
    flow3 = MEL_RATE * 20 * est3
    voc = sum(r[2] for r in _vocoder_rows())
    rows = [
        ("16-codebook synthesiser", "Talker and code predictor, 12.5 frames", _gm(gen, 1)),
        ("", "codec decoder", _gm(dec, 1)),
        ("", "total", _gm(gen + dec, 1)),
        ("single-token synthesiser, second generation", "speech-token model, 25 tokens", _gm(lm, 1)),
        ("", "flow decoder, 20 evaluations of 50 frames, with its encoder", _gm(flow2, 1)),
        ("", "vocoder", _gm(voc, 1)),
        ("", "total", _gm(lm + flow2 + voc, 1)),
        ("single-token synthesiser, third generation", "flow decoder, 20 evaluations of 50 frames", _gm(flow3, 1)),
    ]
    return ("Multiply-adds for one second of speech in billions, from the tables of this chapter (attention scores, norms and "
            "activations not counted; the third generation's vocoder is not counted here).\n\n"
            + _table(["system", "stage", "billions per second of speech"], rows, "llr"))


# ================================================================== figure data

def toy_paths(starts=(-0.4186, 0.3, 1.0636), n=400):
    """(t grid, {start: (conditional path, guided path)}) on a fine uniform grid"""
    out = {}
    ts = [i / n for i in range(n + 1)]
    for z in starts:
        c = [z]
        g = [z]
        for i in range(n):
            dt = ts[i + 1] - ts[i]
            c.append(c[-1] + dt * _velocity(c[-1], ts[i], _TOY_MU, _TOY_S))
            g.append(g[-1] + dt * _guided(g[-1], ts[i]))
        out[z] = (c, g)
    return ts, out
