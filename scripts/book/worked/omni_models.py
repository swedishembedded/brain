# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Models That Listen, See and Speak".

Plain Python 3, standard library only, written from the chapter's own description and independent of any engine code. Each public
function without arguments returns a markdown string that the chapter embeds between `<!-- worked: omni_models.<name> -->`
markers; the functions with arguments (or a leading underscore) supply the numbers of the chapter's figures, so a picture and the
table beside it cannot disagree. Earlier modules are imported, never copied: the table helper of `sound`, the encoder geometry of
`speech_recognition` (chunks of 100 frames, windows of 104 rows), the rounding rule of `vlms` (`smart_resize`), the decoder rows of
`audio_codecs` (the upsampling stages and the SEANet decoder that the Code2Wav shares with the codec of Chapter 3) and the layer
formulas of `speech_synthesis` (the code predictor).

* how many rows a second of sound, a picture and a pair of video frames become, and what a minute of video with its sound asks of
  the decoder: rows, cache bytes and multiply-adds, stage by stage;
* two ways of giving the rows of a clip their positions, on a clip small enough to list: the time-aligned rule that the model's
  configuration declares, and the frame-ordinal rule of brain's crate;
* the parameters of every component of the omni model counted from its configuration (and checked against the number of tensors
  the crate records), the parameters a token of each role uses, and the two kinds of expert layer side by side;
* what crosses from the Thinker to the Talker: the rows of the Talker's prompt, and what each frame of speech is fed;
* the way from sixteen codes a frame to samples: lengths stage by stage, and the work of the code-to-wave decoder per frame;
* the work that must be done before the first sound, in brain's order and in the order the interfaces allow, and the bytes that the
  Talker's weights move per frame in brain's validation-tier path.

Everything is deterministic integer arithmetic on printed constants (ratios in 64-bit floating point): no random number is drawn, so
no generator or seed is involved, and nothing is trained, so every digit printed is reproducible. A multiply-add counts once; the
weight matrices are counted, the attention scores only where the table says so, and normalisations and activations never.

Swedish Embedded AB implements solutions for multimodal models whose token budget, parameter count and first-sound work are counted
before anything is served. If your team needs expertise in omni-modal model engineering, you can procure our services by sending
an email to info@swedishembedded.com.
"""
import audio_codecs as ac
import sound
import speech_recognition as sr
import speech_synthesis as ss
import vlms

_table = sound._table


def _i(n):
    return f"{int(n):,}"


def _g(x, d=2):
    return f"{x:,.{d}f}"


def _m(x, d=1):
    return f"{x / 1e6:,.{d}f}"


def _b(x, d=2):
    return f"{x / 1e9:,.{d}f}"


def _t(x, d=2):
    return f"{x / 1e12:,.{d}f}"


# ------------------------------------------------------------------------------------------------ the configuration
# thinker_config.audio_config (the audio tower)
A_MEL, A_D, A_HEADS, A_FFN, A_LAYERS, A_CONV, A_OUT = 128, 1280, 20, 5120, 32, 480, 2048
A_CHUNK, A_WINDOW_FRAMES = 100, 800
A_HOP = 160                      # samples between mel frames at 16 kHz
# thinker_config.vision_config (the vision tower)
V_DEPTH, V_D, V_HEADS, V_FFN, V_PATCH, V_TEMPORAL, V_MERGE = 27, 1152, 16, 4304, 16, 2, 2
V_POS, V_OUT, V_CH, V_TAPS = 2304, 2048, 3, 3
# thinker_config.text_config (the Thinker decoder)
T_LAYERS, T_D, T_HEADS, T_KV, T_HD, T_FF, T_EXPERTS, T_TOPK, T_VOCAB = 48, 2048, 32, 4, 128, 768, 128, 8, 152064
# talker_config.text_config (the Talker decoder) and its code predictor
K_LAYERS, K_D, K_HEADS, K_KV, K_HD, K_FF, K_EXPERTS, K_TOPK, K_SHARED, K_VOCAB = 20, 1024, 16, 2, 128, 384, 128, 6, 768, 3072
K_PROJ_IN, K_PROJ_MID = 2048, 2048   # the two projections; the middle width is an assumption, see the chapter
NQ, M_LAYERS, M_VOCAB = 16, 5, 2048
# code2wav_config
C_HIDDEN, C_FF, C_LAYERS, C_HEADS, C_WINDOW = 1024, 3072, 8, 16, 72
C_UP, C_RATES, C_DEC = (2, 2), (8, 5, 4, 3), 1536
C_CODEBOOK = 2048
FRAME_RATE = 12.5                # code frames a second
POSITION_ID_PER_SECOND, SECONDS_PER_CHUNK = 13, 2
CHECKPOINT_TENSORS = 28010       # tensors in the released checkpoint's index, as recorded by the crate's import test


# ================================================================================================ rows and what they cost


def _audio_rows(seconds):
    """Rows of the audio tower for a clip of `seconds` seconds at 16 kHz (whole chunks of 100 frames, 13 rows each; the same stem as the
    recogniser of Chapter 4)."""
    return sr._qwen_chunk_rows(int(seconds * 16000) // A_HOP)


def _picture(h, w):
    """(resized h, resized w, patch grid (rows, cols), merged rows) of a picture or a video frame."""
    hb, wb = vlms.smart_resize(h, w, vlms.QWEN_FACTOR, vlms.QWEN_MIN, vlms.QWEN_MAX)
    gh, gw = hb // V_PATCH, wb // V_PATCH
    return hb, wb, (gh, gw), gh * gw // (V_MERGE * V_MERGE)


def rows_per_medium():
    rows = []
    for secs in (0.5, 1, 2, 10, 60):
        rows.append(("sound", f"{secs} second{'' if secs == 1 else 's'} at 16 kHz", _i(int(secs * 16000) // A_HOP), _i(_audio_rows(secs)), f"{_audio_rows(secs) / secs:.1f}"))
    for h, w in ((64, 64), (448, 448), (480, 640), (1080, 1920)):
        hb, wb, (gh, gw), n = _picture(h, w)
        rows.append(("picture", f"{h} x {w}, resized to {hb} x {wb}", f"{gh} x {gw} patches", _i(n), "-"))
    hb, wb, (gh, gw), n = _picture(448, 448)
    rows.append(("video, 2 frames a second", "448 x 448, one pair of frames a second", f"{gh} x {gw} patches", _i(n), f"{n:.1f}"))
    first = ("Rows that each medium hands to the decoder. Sound: the audio tower cuts the 100-frames-a-second log-mel picture (hop 160 samples) into chunks "
             "of 100 frames, each of which gives 13 rows (a partial chunk fewer, by three halvings that round up). Picture: both sides are resized by the "
             "rounding rule of Volume VII, Chapter 9 (called `smart_resize` there): each side goes to the nearest multiple of 32, a half way value to the "
             "even multiple, and at least 32; if the area is then over 3584 x 3584 both original sides are divided by sqrt(h w / 3584^2) and rounded "
             "down to a multiple of 32 (at least 32), and if it is under 56 x 56 both are multiplied by sqrt(56^2 / (h w)) and rounded up; none of the "
             "printed pictures reaches a tie or a limit. Then a patch of 16 x 16 pixels, and 2 x 2 patches merged into one row. "
             "A pair of video frames is one picture's worth of rows.\n\n"
             + _table(["medium", "input", "frames or patches", "rows", "rows a second"], rows, "llrrr"))
    out_rows = [
        ("sound in, one second", f"{_audio_rows(1)} rows", "of 2,048 numbers"),
        ("picture in, 448 x 448, one pair of frames a second", f"{_picture(448, 448)[3]} rows", "of 2,048 numbers"),
        ("speech out, codes", f"{FRAME_RATE} frames x {NQ} codes = {int(FRAME_RATE * NQ)} integers", "of 11 bits: 2,200 bits"),
        ("speech out, waveform", "24,000 samples", "16 bits: 384,000 bits"),
    ]
    second = ("Per second, what goes in and what comes out (rows and integers are counted, not timed).\n\n"
              + _table(["stream", "per second", "each"], out_rows, "lll"))
    return first + "\n\n" + second


# the minute of video of the chapter
MIN_SECONDS, MIN_H, MIN_W, MIN_FPS, MIN_TEXT, MIN_MARKERS = 60, 448, 448, 2, 50, 4
KV_BYTES_PER_VALUE = 4


def _audio_tower_macs(seconds):
    rows = _audio_rows(seconds)
    chunks = int(seconds * 100) // A_CHUNK
    c1 = 64 * 50 * A_CONV * 1 * 9
    c2 = 32 * 25 * A_CONV * A_CONV * 9
    c3 = 16 * 13 * A_CONV * A_CONV * 9
    cout = 13 * (A_CONV * 16) * A_D
    window, spans = sr._qwen_spans(rows)
    stem = chunks * (c1 + c2 + c3 + cout)
    dense = A_LAYERS * rows * (4 * A_D * A_D + 2 * A_D * A_FFN)
    attn = A_LAYERS * sum(s * s for s in spans) * 2 * A_D
    proj = rows * (A_D * A_D + A_D * A_OUT)
    return {"stem": stem, "dense": dense, "attn": attn, "proj": proj, "total": stem + dense + attn + proj, "rows": rows, "spans": spans}


def _vision_group_macs(h, w):
    hb, wb, (gh, gw), n_rows = _picture(h, w)
    n = gh * gw
    embed = n * (V_CH * V_TEMPORAL * V_PATCH * V_PATCH) * V_D
    dense = V_DEPTH * n * (3 * V_D * V_D + V_D * V_D + 2 * V_D * V_FFN)
    attn = V_DEPTH * n * n * 2 * V_D
    merged = V_D * V_MERGE * V_MERGE
    merge = n_rows * (merged * merged + merged * V_OUT)
    # The three DeepStack mergers have the main merger's two linear maps and run on the rows of the tower's taps.
    return {"embed": embed, "dense": dense, "attn": attn, "merge": merge, "deepstack": V_TAPS * merge,
            "total": embed + dense + attn + merge, "rows": n_rows}


def _thinker_layer_params():
    attn = T_D * T_HEADS * T_HD + 2 * T_D * T_KV * T_HD + T_HEADS * T_HD * T_D
    return attn, T_D * T_EXPERTS, 3 * T_D * T_FF


def _thinker_row_macs():
    """Weight multiply-adds of one row through the 48 layers (attention, router, 8 experts), without the output head."""
    attn, router, expert = _thinker_layer_params()
    return T_LAYERS * (attn + router + T_TOPK * expert)


def _thinker_head_macs():
    return T_D * T_VOCAB


def _thinker_attention_macs(n):
    """Scores and weighted sums of a causal prefill of n rows: row p reads p positions, 2 x heads x head width a position, in every layer."""
    return T_LAYERS * 2 * T_HEADS * T_HD * (n * (n + 1) // 2)


def _kv_bytes_per_token():
    return T_LAYERS * 2 * T_KV * T_HD * KV_BYTES_PER_VALUE


def minute_of_video():
    groups = MIN_SECONDS * MIN_FPS // V_TEMPORAL
    hb, wb, (gh, gw), per_group = _picture(MIN_H, MIN_W)
    a_rows = _audio_rows(MIN_SECONDS)
    v_rows = groups * per_group
    total = MIN_TEXT + MIN_MARKERS + a_rows + v_rows
    rows = [("text of the request (an input of the example)", _i(MIN_TEXT)),
            ("start and end markers, one pair for the sound and one for the video", _i(MIN_MARKERS)),
            (f"sound: {MIN_SECONDS} chunks of 13 rows", _i(a_rows)),
            (f"video: {MIN_SECONDS * MIN_FPS} frames, {groups} pairs of {per_group} rows", _i(v_rows)),
            ("sequence the Thinker prefills", _i(total))]
    first = (f"A clip of {MIN_SECONDS} seconds with its sound, the video sampled at {MIN_FPS} frames a second at {MIN_H} x {MIN_W} pixels (resized to "
             f"{hb} x {wb}, a grid of {gh // V_MERGE} x {gw // V_MERGE} merged rows), preceded by a request of {MIN_TEXT} tokens.\n\n"
             + _table(["part", "rows"], rows))
    a = _audio_tower_macs(MIN_SECONDS)
    v = _vision_group_macs(MIN_H, MIN_W)
    thinker_w = total * _thinker_row_macs() + _thinker_head_macs()
    thinker_a = _thinker_attention_macs(total)
    stages = [("audio tower: convolution stem", a["stem"]), ("audio tower: 32 blocks, weights", a["dense"]), ("audio tower: attention in windows of 104 rows", a["attn"]),
              ("audio tower: projector", a["proj"]),
              (f"vision tower: patch embedding, {groups} pairs", groups * v["embed"]), (f"vision tower: 27 blocks, weights, {groups} pairs", groups * v["dense"]),
              ("vision tower: attention inside each pair's grid", groups * v["attn"]), ("vision tower: merger", groups * v["merge"]),
              ("vision tower: three DeepStack mergers", groups * v["deepstack"]),
              ("Thinker: weights of 48 layers, every row, and the head once", thinker_w), ("Thinker: attention scores and weighted sums", thinker_a)]
    all_total = sum(m for _, m in stages)
    body = [(n, _t(m, 3), f"{100 * m / all_total:.1f}") for n, m in stages]
    body.append(("everything before the first reply token", _t(all_total, 3), "100.0"))
    second = ("Multiply-adds, in trillions, that the clip costs before the first token of the reply (the vision tower's three DeepStack mergers are extra copies "
              "of the final merger that the checkpoint carries; their outputs are added to the visual rows after the Thinker's first three layers, "
              "so each costs the main merger's two linear maps; norms, activations, the additions and the position-table resampling are left out).\n\n" + _table(["stage", "trillions of multiply-adds", "percent"], body))
    kv = _kv_bytes_per_token()
    third = (f"The cache that the prefill leaves behind: 48 layers x 2 x 4 key-value heads x 128 x {KV_BYTES_PER_VALUE} bytes is {_i(kv)} bytes a position (32-bit floats, "
             f"as brain's caches are), so {_i(total)} positions hold {_i(kv * total)} bytes, {kv * total / 2**30:.2f} GiB.")
    return first + "\n\n" + second + "\n\n" + third


# ================================================================================================ positions on two clocks


def time_aligned(prefix, seconds, grid, p_rate=POSITION_ID_PER_SECOND, chunk_s=SECONDS_PER_CHUNK, groups_per_s=1, audio_rows_per_s=13):
    """The time-aligned rule: the clip is cut into chunks of `chunk_s` seconds; each chunk lists the rows of its video groups, then the rows of
    its sound. A video group g (one second of video when `groups_per_s` is 1) has T = prefix + p_rate * g / groups_per_s, and the row (i, j) of
    its grid has H = prefix + i, W = prefix + j. Sound row k has (prefix + k) on all three axes. Returns (segments, counter after the clip),
    a segment being (label, first index, rows, T range, H range, W range)."""
    gh, gw = grid
    segs, idx = [], prefix
    for c in range(seconds // chunk_s):
        for gi in range(chunk_s * groups_per_s):
            g = c * chunk_s * groups_per_s + gi
            t = prefix + p_rate * g // groups_per_s
            segs.append((f"video group {g}", idx, gh * gw, (t, t), (prefix, prefix + gh - 1), (prefix, prefix + gw - 1)))
            idx += gh * gw
        k0 = c * chunk_s * audio_rows_per_s
        n = chunk_s * audio_rows_per_s
        segs.append((f"sound rows {k0} to {k0 + n - 1}", idx, n, (prefix + k0, prefix + k0 + n - 1), (prefix + k0, prefix + k0 + n - 1), (prefix + k0, prefix + k0 + n - 1)))
        idx += n
    top = max(max(s[3][1], s[4][1], s[5][1]) for s in segs)
    return segs, top + 1


def frame_ordinal(prefix, seconds, grid, groups_per_s=1, audio_rows_per_s=13):
    """Brain's rule: the sound is one block of rows with the diagonal positions of a text run, then the video is one block, a group g at
    T = start + g and the grid on H and W from start; after each block the counter moves past its largest extent."""
    gh, gw = grid
    n_audio = seconds * audio_rows_per_s
    groups = seconds * groups_per_s
    segs = [("sound block", prefix, n_audio, (prefix, prefix + n_audio - 1), (prefix, prefix + n_audio - 1), (prefix, prefix + n_audio - 1))]
    start = prefix + max(n_audio, 1)
    segs.append(("video block", prefix + n_audio, groups * gh * gw, (start, start + groups - 1), (start, start + gh - 1), (start, start + gw - 1)))
    top = start + max(groups, gh, gw)
    return segs, top


def _rng(a):
    return f"{a[0]}" if a[0] == a[1] else f"{a[0]} to {a[1]}"


def interleave_toy():
    prefix, seconds, grid = 6, 4, (2, 2)
    segs, after = time_aligned(prefix, seconds, grid)
    rows = [(s[0], f"{s[1]} to {s[1] + s[2] - 1}", _i(s[2]), _rng(s[3]), _rng(s[4]), _rng(s[5])) for s in segs]
    first = (f"A clip of {seconds} seconds, video at 2 frames a second (one pair, one group, a second) with a grid of 2 x 2 merged rows, sound at 13 rows a "
             f"second, after a text prefix of {prefix} tokens (positions 0 to {prefix - 1}). Positions per the time-aligned rule with 13 positions a second and "
             "chunks of 2 seconds; within a chunk the video groups come first, then the sound. The counter after the clip is one more than the largest position "
             f"on any axis: {after}.\n\n"
             + _table(["segment", "sequence index", "rows", "T", "H", "W"], rows, "llrrrr"))
    segs2, after2 = frame_ordinal(prefix, seconds, grid)
    rows2 = [(s[0], f"{s[1]} to {s[1] + s[2] - 1}", _i(s[2]), _rng(s[3]), _rng(s[4]), _rng(s[5])) for s in segs2]
    second = ("The same clip under brain's rule: the sound as one block, then the video as one block, a pair at T equal to the block's start plus its ordinal "
              "(markers and the text after the clip are left out, as above). The counter after both blocks is "
              f"{after2}.\n\n" + _table(["segment", "sequence index", "rows", "T", "H", "W"], rows2, "llrrrr"))
    t_g3 = segs[4][3][0]                    # video group 3: chunk 0 is two groups and a sound segment, chunk 1 starts at index 3
    third = ("The sound row that starts second 3 (the fourth, counting from 0) is row 39 and the video group of that second is group 3. Time-aligned: the sound row sits at "
             f"{prefix + 39} on every axis and the group at T = {t_g3}. Frame-ordinal: the sound row sits at {prefix + 39} and the group at T = "
             f"{segs2[1][3][0] + 3}, with the whole sound block of {segs2[0][2]} rows between them in the sequence.")
    return first + "\n\n" + second + "\n\n" + third


def clock_data(seconds=MIN_SECONDS):
    """Series of the clock figure: for each second of a minute, the T position of the video group and of the first sound row of that second,
    under the two rules, for a clip with no text prefix."""
    _, ta_after = time_aligned(0, seconds, (14, 14))
    fo, after = frame_ordinal(0, seconds, (14, 14))
    secs = list(range(seconds))
    return {"seconds": secs,
            "aligned_video": [POSITION_ID_PER_SECOND * s for s in secs],
            "aligned_sound": [POSITION_ID_PER_SECOND * s for s in secs],
            "ordinal_video": [fo[1][3][0] + s for s in secs],
            "ordinal_sound": [fo[0][3][0] + POSITION_ID_PER_SECOND * s for s in secs],
            "after_aligned": ta_after, "after_ordinal": after}


def position_clocks():
    d = clock_data()
    rows = []
    for s in (0, 1, 10, 30, 59):
        rows.append((s, d["aligned_sound"][s], d["aligned_video"][s], d["ordinal_sound"][s], d["ordinal_video"][s],
                     d["ordinal_video"][s] - d["ordinal_sound"][s]))
    return ("The minute of the chapter (60 seconds, video groups of 14 x 14 merged rows), no text prefix: the position on the T axis of the first sound row "
            "of second s and of the video group of second s, under each rule. Frame-ordinal: the video block starts after the 780 sound rows, at position "
            f"780; the counter after the clip is {d['after_ordinal']} against {d['after_aligned']} for the time-aligned rule.\n\n"
            + _table(["second s", "time-aligned: sound", "time-aligned: video", "frame-ordinal: sound", "frame-ordinal: video", "video minus sound"], rows))


# ================================================================================================ parameters


def _thinker_text_counts():
    attn_w = T_D * T_HEADS * T_HD + 2 * T_D * T_KV * T_HD + T_HEADS * T_HD * T_D
    norms = 2 * T_D + 2 * T_HD
    router = T_D * T_EXPERTS
    experts = T_EXPERTS * 3 * T_D * T_FF
    per_layer = attn_w + norms + router + experts
    return {"layer": per_layer, "attn": attn_w, "norms": norms, "router": router, "experts": experts, "embed": T_VOCAB * T_D, "head": T_VOCAB * T_D,
            "final": T_D, "total": T_LAYERS * per_layer + 2 * T_VOCAB * T_D + T_D}


def _talker_counts():
    hq, hkv = K_HEADS * K_HD, K_KV * K_HD
    attn_w = K_D * hq + 2 * K_D * hkv + hq * K_D
    norms = 2 * K_D + 2 * K_HD
    router = K_D * K_EXPERTS
    experts = K_EXPERTS * 3 * K_D * K_FF
    shared = 3 * K_D * K_SHARED
    gate = K_D
    per_layer = attn_w + norms + router + experts + shared + gate
    proj = K_PROJ_IN * K_PROJ_MID + K_PROJ_MID + K_PROJ_MID * K_D + K_D
    return {"layer": per_layer, "attn": attn_w, "norms": norms, "router": router, "experts": experts, "shared": shared, "gate": gate, "proj": proj,
            "embed": K_VOCAB * K_D, "head": K_VOCAB * K_D, "final": K_D, "total": K_LAYERS * per_layer + 2 * K_VOCAB * K_D + K_D + 2 * proj}


def _predictor_counts():
    per = ss._layer_params(K_D, 16 * 128, 8 * 128, 3072, 128)
    return {"layers": M_LAYERS * per + K_D, "heads_tables": (NQ - 1) * 2 * M_VOCAB * K_D, "total": M_LAYERS * per + K_D + (NQ - 1) * 2 * M_VOCAB * K_D}


def _audio_counts():
    d = A_D
    attn = 4 * (d * d + d)
    mlp = d * A_FFN + A_FFN + A_FFN * d + d
    norms = 4 * d
    block = attn + mlp + norms
    conv = [A_CONV * 9 + A_CONV, A_CONV * A_CONV * 9 + A_CONV, A_CONV * A_CONV * 9 + A_CONV]
    conv_out = d * A_CONV * 16
    tail = 2 * d + (d * d + d) + (A_OUT * d + A_OUT)
    return {"block": block, "conv": sum(conv), "conv_out": conv_out, "tail": tail, "total": sum(conv) + conv_out + A_LAYERS * block + tail}


def _vision_counts():
    d = V_D
    block = 4 * d + (d * 3 * d + 3 * d) + (d * d + d) + (d * V_FFN + V_FFN) + (V_FFN * d + d)
    merged = d * V_MERGE * V_MERGE
    mlp = merged * merged + merged + merged * V_OUT + V_OUT
    merger = 2 * d + mlp
    deepstack = 2 * merged + mlp
    patch = d * V_CH * V_TEMPORAL * V_PATCH * V_PATCH + d
    pos = V_POS * d
    return {"block": block, "patch": patch, "pos": pos, "merger": merger, "deepstack": deepstack, "total": patch + pos + V_DEPTH * block + merger + V_TAPS * deepstack}


def _code2wav_counts():
    d, ff = C_HIDDEN, C_FF
    table = C_CODEBOOK * NQ * d
    layer = 4 * d * d + 3 * d * ff + 2 * d + 2 * d     # q k v o, gate up down, two norms, two layer scales; no biases
    transformer = C_LAYERS * layer + d
    # the decoder from the upsampling stages on is the one counted in Chapter 3 (same widths): take its rows
    dec = ac._decoder_rows()
    rest = dec[4:]
    return {"table": table, "transformer": transformer, "layer": layer, "rest": sum(p for _, p, _ in rest), "rest_rows": rest,
            "total": table + transformer + sum(p for _, p, _ in rest)}


def _tensor_counts():
    thinker = 3 + T_LAYERS * (8 + 1 + 3 * T_EXPERTS)
    talker = 3 + 4 + 4 + K_LAYERS * (8 + 1 + 3 * K_EXPERTS + 3 + 1)
    predictor = M_LAYERS * 11 + 1 + 2 * (NQ - 1)
    audio = 13 + A_LAYERS * 16
    vision = 3 + V_DEPTH * 12 + 6 + V_TAPS * 6
    code2wav = 1 + C_LAYERS * 11 + 1 + len(C_UP) * 11 + 2 + len(C_RATES) * (4 + 3 * 8) + 4
    return {"thinker": thinker, "talker": talker, "predictor": predictor, "audio": audio, "vision": vision, "code2wav": code2wav}


def _prod(dims):
    n = 1
    for x in dims:
        n *= x
    return n


def _shape(dims):
    return " x ".join(f"{x:,}" for x in dims)


def audio_rows():
    """Every tensor group of the audio tower as (name, dims, tensors): one stored tensor per copy, shapes written outputs first
    (a convolution as outputs x inputs x kernel height x kernel width)."""
    d, ff, c, n = A_D, A_FFN, A_CONV, A_LAYERS
    return [("stem convolution 1 weight (stride 2, padding 1)", (c, 1, 3, 3), 1), ("stem convolution 1 bias", (c,), 1),
            ("stem convolution 2 weight", (c, c, 3, 3), 1), ("stem convolution 2 bias", (c,), 1),
            ("stem convolution 3 weight", (c, c, 3, 3), 1), ("stem convolution 3 bias", (c,), 1),
            ("stem output linear, 480 channels x 16 mel bins = 7,680 to 1,280 (no bias)", (d, c * 16), 1),
            ("per block: attention q, k, v, out weights", (d, d), 4 * n), ("per block: their four biases", (d,), 4 * n),
            ("per block: feed-forward up weight", (ff, d), n), ("per block: its bias", (ff,), n),
            ("per block: feed-forward down weight", (d, ff), n), ("per block: its bias", (d,), n),
            ("per block: two LayerNorms, gain and bias each", (d,), 4 * n),
            ("final LayerNorm gain and bias", (d,), 2), ("projector, first linear 1,280 to 1,280, weight", (d, d), 1),
            ("its bias", (d,), 1), ("projector, second linear 1,280 to 2,048, weight", (A_OUT, d), 1), ("its bias", (A_OUT,), 1)]


def vision_rows():
    """Every tensor group of the vision tower as (name, dims, tensors)."""
    d, ff, n = V_D, V_FFN, V_DEPTH
    merged = d * V_MERGE * V_MERGE
    return [("patch embedding, a 3-D convolution over 3 channels, 2 frames, 16 x 16 pixels: weight", (d, V_CH, V_TEMPORAL, V_PATCH, V_PATCH), 1),
            ("its bias", (d,), 1), ("position table, a 48 x 48 grid of learned positions (resampled to the picture's grid)", (V_POS, d), 1),
            ("per block: two LayerNorms, gain and bias each", (d,), 4 * n), ("per block: fused q, k, v weight", (3 * d, d), n),
            ("per block: its bias", (3 * d,), n), ("per block: attention output weight", (d, d), n), ("per block: its bias", (d,), n),
            ("per block: feed-forward up weight", (ff, d), n), ("per block: its bias", (ff,), n),
            ("per block: feed-forward down weight", (d, ff), n), ("per block: its bias", (d,), n),
            ("main merger: LayerNorm over 1,152, gain and bias", (d,), 2),
            ("main merger: first linear 4,608 to 4,608, weight", (merged, merged), 1), ("its bias", (merged,), 1),
            ("main merger: second linear 4,608 to 2,048, weight", (V_OUT, merged), 1), ("its bias", (V_OUT,), 1),
            ("three DeepStack mergers: LayerNorm over 4,608 (the four rows merged), gain and bias", (merged,), 2 * V_TAPS),
            ("three DeepStack mergers: first linear weight", (merged, merged), V_TAPS), ("their bias", (merged,), V_TAPS),
            ("three DeepStack mergers: second linear weight", (V_OUT, merged), V_TAPS), ("their bias", (V_OUT,), V_TAPS)]


def _moe_attention_rows(d, heads, kv, hd, n, tag):
    return [(f"{tag}: W_Q", (heads * hd, d), n), (f"{tag}: W_K", (kv * hd, d), n), (f"{tag}: W_V", (kv * hd, d), n),
            (f"{tag}: W_O", (d, heads * hd), n), (f"{tag}: query-norm and key-norm gains", (hd,), 2 * n),
            (f"{tag}: input and post-attention RMSNorm gains", (d,), 2 * n)]


def thinker_rows():
    """The Thinker's tensors as (name, dims, tensors); every expert is three tensors of its own."""
    d, n, e, ff = T_D, T_LAYERS, T_EXPERTS, T_FF
    return ([("token embedding table", (T_VOCAB, d), 1), ("output matrix", (T_VOCAB, d), 1), ("final RMSNorm gain", (d,), 1)]
            + _moe_attention_rows(d, T_HEADS, T_KV, T_HD, n, "per layer")
            + [("per layer: router", (e, d), n), ("per layer: expert gate and up weights", (ff, d), 2 * e * n),
               ("per layer: expert down weights", (d, ff), e * n)])


def talker_rows():
    """The Talker's tensors, the two projections included (their middle width of 2,048 is the chapter's one assumption)."""
    d, n, e, ff = K_D, K_LAYERS, K_EXPERTS, K_FF
    return ([("codec embedding table", (K_VOCAB, d), 1), ("codec head", (K_VOCAB, d), 1), ("final RMSNorm gain", (d,), 1)]
            + _moe_attention_rows(d, K_HEADS, K_KV, K_HD, n, "per layer")
            + [("per layer: router", (e, d), n), ("per layer: expert gate and up weights", (ff, d), 2 * e * n),
               ("per layer: expert down weights", (d, ff), e * n),
               ("per layer: shared expert gate and up weights", (K_SHARED, d), 2 * n),
               ("per layer: shared expert down weight", (d, K_SHARED), n), ("per layer: shared expert's gate, one row", (d,), n),
               ("text_projection and hidden_projection: first linear 2,048 to 2,048, weights", (K_PROJ_MID, K_PROJ_IN), 2),
               ("their biases", (K_PROJ_MID,), 2), ("the same two: second linear 2,048 to 1,024, weights", (d, K_PROJ_MID), 2),
               ("their biases", (d,), 2)])


def predictor_rows():
    """The code predictor: 5 layers of 16 query and 8 key-value heads of 128, a feed-forward of 3,072, 15 tables and 15 heads."""
    d, n = K_D, M_LAYERS
    return ([("final RMSNorm gain", (d,), 1)] + _moe_attention_rows(d, 16, 8, 128, n, "per layer")
            + [("per layer: gate and up weights", (3072, d), 2 * n), ("per layer: down weight", (d, 3072), n),
               ("codec embedding tables, one per code 1 to 15", (M_VOCAB, d), NQ - 1), ("output heads, one per code 1 to 15", (M_VOCAB, d), NQ - 1)])


def code2wav_front_rows():
    """The Code2Wav's code table and transformer: 8 layers of 16 heads of 64, width 1,024, a feed-forward of 3,072, no biases."""
    d, ff, n = C_HIDDEN, C_FF, C_LAYERS
    return [("code table, 16 codes x 2,048 values", (C_CODEBOOK * NQ, d), 1), ("per layer: q, k, v, out weights", (d, d), 4 * n),
            ("per layer: gate and up weights", (ff, d), 2 * n), ("per layer: down weight", (d, ff), n),
            ("per layer: two RMSNorm gains", (d,), 2 * n), ("per layer: two layer-scale vectors", (d,), 2 * n),
            ("final RMSNorm gain", (d,), 1)]


def code2wav_layout():
    """(layer, parameter formula, parameters, tensors, multiply-adds per code frame, positions per frame) of the part of Code2Wav after
    its transformer: two upsampling stages and the SEANet decoder. A convolution is outputs x inputs x kernel, a transposed
    convolution inputs x outputs x kernel; its multiply-adds are counted per input position, every other layer per output
    position; `positions per frame` is the length of that axis for one code frame (the crop of Chapter 3's lengths ignored)."""
    c = ac.LATENT
    rows = []
    pos = 1
    for k, u in enumerate(ac.UP):
        rows.append((f"upsampling stage {k + 1}: transposed convolution (k = {u}, stride {u})", f"{c:,} x {c:,} x {u} + {c:,}",
                     c * c * u + c, 2, c * c * u, pos))
        pos *= u
        convnext = ac._conv_params(c, c, 7, groups=c) + 2 * c + ac._conv_params(c, 4 * c, 1) + ac._conv_params(4 * c, c, 1) + c
        rows.append((f"upsampling stage {k + 1}: ConvNeXt block", f"depthwise {c:,} x 7 + {c:,}, norm 2 x {c:,}, "
                     f"{c:,} x {4 * c:,} + {4 * c:,}, {4 * c:,} x {c:,} + {c:,}, scale {c:,}", convnext, 9,
                     c * 7 + c * 4 * c * 2, pos))
    d0 = ac.DEC_DIM
    rows.append(("SEANet first convolution (k = 7)", f"{c:,} x {d0:,} x 7 + {d0:,}", ac._conv_params(c, d0, 7), 2, c * d0 * 7, pos))
    cin = d0
    for i, r in enumerate(ac.RATES):
        cout = d0 >> (i + 1)
        up = 2 * cin + cin * cout * 2 * r + cout
        rows.append((f"SEANet block {i + 1}: Snake (alpha and beta) and transposed convolution (k = {2 * r}, stride {r})",
                     f"2 x {cin:,} + {cin:,} x {cout:,} x {2 * r} + {cout:,}", up, 4, cin * cout * 2 * r, pos))
        pos *= r
        unit = 2 * (2 * cout) + ac._conv_params(cout, cout, 7) + ac._conv_params(cout, cout, 1)
        rows.append((f"SEANet block {i + 1}: three residual units (7-tap and 1-tap convolutions, two Snakes each)",
                     f"3 x (2 x 2 x {cout:,} + {cout:,}^2 x 7 + {cout:,} + {cout:,}^2 + {cout:,})", 3 * unit, 24,
                     3 * (cout * cout * 7 + cout * cout), pos))
        cin = cout
    rows.append(("SEANet tail: Snake and convolution (k = 7, 96 to 1)", f"2 x {cin} + {cin} x 7 + 1", 2 * cin + ac._conv_params(cin, 1, 7), 4,
                 cin * 7, pos))
    return rows


def _check_layouts():
    """The layout tables must add up to the counts used everywhere else in the module, parameters and tensors alike."""
    def tot(rows):
        return sum(_prod(d) * k for _, d, k in rows), sum(k for _, _, k in rows)

    au, vi, th, tk = _audio_counts(), _vision_counts(), _thinker_text_counts(), _talker_counts()
    pr, cw, tc = _predictor_counts(), _code2wav_counts(), _tensor_counts()
    assert tot(audio_rows()) == (au["total"], tc["audio"])
    assert tot(vision_rows()) == (vi["total"], tc["vision"])
    assert tot(thinker_rows()) == (th["total"], tc["thinker"])
    assert tot(talker_rows()) == (tk["total"], tc["talker"])
    assert tot(predictor_rows()) == (pr["total"], tc["predictor"])
    front = tot(code2wav_front_rows())
    layout = code2wav_layout()
    assert front[0] + sum(r[2] for r in layout) == cw["total"]
    assert front[1] + sum(r[3] for r in layout) == tc["code2wav"]
    per_frame = sum(r[4] * r[5] for r in layout)
    assert abs(per_frame - sum(m for _, _, m in ac._decoder_rows()[4:]) / FRAME_RATE) < 1e-6 * per_frame


def architecture_layouts():
    """The tensors behind every parameter and tensor count of `parameter_counts`."""
    _check_layouts()

    def table(rows):
        body = [(n, _shape(dims), _i(k), _i(_prod(dims) * k)) for n, dims, k in rows]
        body.append(("total", "", _i(sum(k for _, _, k in rows)), _i(sum(_prod(d) * k for _, d, k in rows))))
        return _table(["tensor group", "shape (outputs x inputs)", "tensors", "parameters"], body, "llrr")

    cw = code2wav_layout()
    cw_body = [(n, f, _i(p), _i(t), _i(m), _i(u)) for n, f, p, t, m, u in cw]
    cw_body.append(("total, after the transformer", "", _i(sum(r[2] for r in cw)), _i(sum(r[3] for r in cw)), _i(sum(r[4] * r[5] for r in cw)), ""))
    return ("The tensors behind the next table, so that every parameter count and every tensor count in it can be re-derived. A shape lists outputs "
            "before inputs; a convolution is outputs x inputs x kernel; each copy is one tensor stored under its own name, so the experts of a "
            "layer are 128 gate, 128 up and 128 down tensors, and a weight and its bias are two tensors. A group's parameters are the product of "
            "its shape and its tensors. Where a bias is not listed the map has none.\n\n"
            "Audio tower (the stem takes the 128-bin log-mel picture of a chunk of 100 frames to 13 rows of 7,680 numbers; 32 blocks of width "
            "1,280, 20 heads, a feed-forward of 5,120; every linear map of the blocks and of the projector has a bias):\n\n"
            + table(audio_rows())
            + "\n\nVision tower (27 blocks of width 1,152, 16 heads, a feed-forward of 4,304, fused query-key-value maps with bias):\n\n"
            + table(vision_rows())
            + "\n\nThinker decoder (48 layers of width 2,048, 32 query and 4 key-value heads of 128, 128 experts of width 768; no biases):\n\n"
            + table(thinker_rows())
            + "\n\nTalker (20 layers of width 1,024, 16 query and 2 key-value heads of 128, 128 experts of width 384 and a shared expert of "
              "width 768):\n\n" + table(talker_rows())
            + "\n\nCode predictor (5 layers of width 1,024, 16 query and 8 key-value heads of 128, a feed-forward of 3,072):\n\n"
            + table(predictor_rows())
            + "\n\nCode2Wav, first the code table and the transformer (8 layers of width 1,024, 16 heads of 64, a feed-forward of 3,072, "
              "no biases, two layer scales a layer):\n\n" + table(code2wav_front_rows())
            + "\n\nand then the two upsampling stages and the SEANet decoder, which have the widths of Chapter 3's codec decoder. Every convolution "
              "has a bias, and a Snake activation has two vectors, alpha and beta, of one number per channel. The multiply-adds are per position "
              "of the layer's axis (a transposed convolution at its input positions), and the last column is how many such positions one code "
              "frame has there:\n\n"
            + _table(["layer", "parameters, as a sum of tensors", "parameters", "tensors", "multiply-adds a position", "positions a frame"],
                     cw_body, "llrrrr"))


def parameter_counts():
    th, tk, pr, au, vi, cw = _thinker_text_counts(), _talker_counts(), _predictor_counts(), _audio_counts(), _vision_counts(), _code2wav_counts()
    tc = _tensor_counts()
    rows = [
        ("Thinker: audio tower (conv stem, 32 blocks, projector)", tc["audio"], au["total"]),
        ("Thinker: vision tower (patch embedding, 27 blocks, main merger)", tc["vision"] - V_TAPS * 6, vi["total"] - V_TAPS * vi["deepstack"]),
        ("Thinker: three DeepStack mergers", V_TAPS * 6, V_TAPS * vi["deepstack"]),
        ("Thinker: embedding table and output matrix (152,064 x 2,048 each)", 2, th["embed"] + th["head"]),
        ("Thinker: 48 decoder layers and the final norm", tc["thinker"] - 3 + 1, T_LAYERS * th["layer"] + th["final"]),
        ("Talker: 20 decoder layers and the final norm", tc["talker"] - 3 - 8 + 1, K_LAYERS * tk["layer"] + tk["final"]),
        ("Talker: codec embedding and codec head (3,072 x 1,024 each)", 2, tk["embed"] + tk["head"]),
        ("Talker: text and hidden projections (middle width assumed 2,048)", 8, 2 * tk["proj"]),
        ("code predictor: 5 layers, 15 heads, 15 tables", tc["predictor"], pr["total"]),
        ("Code2Wav: code embedding, 8-layer transformer", 1 + C_LAYERS * 11 + 1, cw["table"] + cw["transformer"]),
        ("Code2Wav: upsampling stages and SEANet decoder", tc["code2wav"] - (1 + C_LAYERS * 11 + 1), cw["rest"]),
    ]
    total_params = sum(p for _, _, p in rows)
    total_tensors = sum(t for _, t, _ in rows)
    body = [(n, _i(t), _i(p)) for n, t, p in rows]
    body.append(("**the whole model**", _i(total_tensors), _i(total_params)))
    thinker_total = au["total"] + vi["total"] + th["total"]
    talker_total = tk["total"] + pr["total"]
    return ("Parameters counted from the tensors that the configuration and the code of the crate imply (weights and the biases that exist, norm gains, layer scales, "
            "the Snake parameters of the vocoder), with the number of tensors beside them. The Talker's two projections are the one assumption: their middle width "
            "is not a field the crate parses, and 2,048 is taken.\n\n"
            + _table(["component", "tensors", "parameters"], body)
            + f"\n\nThe tensors sum to {total_tensors:,}, the number of names that the crate's import test records for the released checkpoint ({CHECKPOINT_TENSORS:,}). "
              f"The Thinker with its towers holds {_i(thinker_total)} parameters, the speech side (Talker, code predictor and Code2Wav) "
              f"{_i(talker_total + cw['total'])}.")


def _per_role():
    th, tk, pr = _thinker_text_counts(), _talker_counts(), _predictor_counts()
    attn, router, expert = _thinker_layer_params()
    thinker_row = _thinker_row_macs()
    head = _thinker_head_macs()
    talker_layer = (K_D * K_HEADS * K_HD + 2 * K_D * K_KV * K_HD + K_HEADS * K_HD * K_D) + K_D * K_EXPERTS + K_TOPK * 3 * K_D * K_FF + 3 * K_D * K_SHARED + K_D
    talker_frame = K_LAYERS * talker_layer + K_VOCAB * K_D
    per = ss._layer_params(K_D, 16 * 128, 8 * 128, 3072, 128)
    predictor_frame = NQ * M_LAYERS * ss._layer_macs(K_D, 16 * 128, 8 * 128, 3072) + (NQ - 1) * K_D * M_VOCAB
    return {"thinker_row": thinker_row, "head": head, "talker_frame": talker_frame, "predictor_frame": predictor_frame, "talker_layer": talker_layer}


def code2wav_frame_macs():
    """Multiply-adds of the Code2Wav per code frame: the mean over the 16 embeddings (additions), the 8-layer transformer without its
    attention scores, and the upsampling stages and the SEANet decoder, each layer's multiply-adds a position times its positions a frame."""
    d, ff = C_HIDDEN, C_FF
    transformer = C_LAYERS * (4 * d * d + 3 * d * ff)
    rest = sum(m * u for _, _, _, _, m, u in code2wav_layout())
    return {"transformer": transformer, "rest": rest, "total": transformer + rest}


def active_counts():
    r = _per_role()
    th = _thinker_text_counts()
    c = code2wav_frame_macs()
    a = _audio_tower_macs(1)
    v = _vision_group_macs(448, 448)
    rows = [
        ("Thinker, one prompt row (attention, router, 8 of 128 experts; no head)", _i(r["thinker_row"])),
        ("Thinker, one generated token (the same and the 152,064-way head)", _i(r["thinker_row"] + r["head"])),
        ("audio tower, one second (13 rows)", _i(a["total"])),
        ("vision tower, one pair of 448 x 448 frames (196 rows)", _i(v["total"])),
        ("Talker, one frame (20 layers with 6 of 128 experts and the shared expert, and the codec head)", _i(r["talker_frame"])),
        ("code predictor, one frame (16 steps of 5 layers, 15 heads)", _i(r["predictor_frame"])),
        ("Code2Wav, one frame", _i(c["total"])),
    ]
    thinker_total = th["total"]
    first = ("Multiply-adds of one unit of each role, weights only, no attention scores except where the row says: the audio tower's one second and the vision "
             "tower's pair of frames are complete counts, with their windowed attention.\n\n" + _table(["unit", "multiply-adds"], rows, "lr"))
    speech = r["talker_frame"] + r["predictor_frame"] + c["total"]
    second = (f"A frame of speech is {_i(speech)} multiply-adds in all, {_g(speech / (r['thinker_row'] + r['head']), 2)} times the generated token of the Thinker, "
              f"and a second of speech is 12.5 of them, {_b(12.5 * speech, 2)} billion. The Thinker's decoder holds {_i(T_LAYERS * th['layer'] + th['final'])} "
              f"parameters in its layers, {_g((T_LAYERS * th['layer'] + th['final']) / r['thinker_row'], 2)} times what a row uses; with the matrices of the table it holds "
              f"{_i(thinker_total)}, and a generated token uses {_i(r['thinker_row'] + r['head'])} of them, {_g(thinker_total / (r['thinker_row'] + r['head']), 2)} times fewer.")
    return first + "\n\n" + second


def moe_layers():
    th, tk = _thinker_text_counts(), _talker_counts()
    attn, router, expert = _thinker_layer_params()
    rows = [
        ("attention (q, k, v, o)", _i(th["attn"]), _i(th["attn"]), _i(tk["attn"]), _i(tk["attn"])),
        ("norms (two gains, two query-key norms)", _i(th["norms"]), _i(th["norms"]), _i(tk["norms"]), _i(tk["norms"])),
        ("router", _i(th["router"]), _i(th["router"]), _i(tk["router"]), _i(tk["router"])),
        ("routed experts", f"128 x {_i(expert)} = {_i(th['experts'])}", f"8 x {_i(expert)} = {_i(T_TOPK * expert)}",
         f"128 x {_i(3 * K_D * K_FF)} = {_i(tk['experts'])}", f"6 x {_i(3 * K_D * K_FF)} = {_i(K_TOPK * 3 * K_D * K_FF)}"),
        ("shared expert and its gate", "0", "0", _i(tk["shared"] + tk["gate"]), _i(tk["shared"] + tk["gate"])),
    ]
    tot_t = th["layer"]
    use_t = th["attn"] + th["norms"] + th["router"] + T_TOPK * expert
    tot_k = tk["layer"]
    use_k = _talker_used_params()
    rows.append(("**a layer**", _i(tot_t), _i(use_t), _i(tot_k), _i(use_k)))
    rows.append(("held / used", "", f"{tot_t / use_t:.2f}", "", f"{tot_k / use_k:.2f}"))
    return ("One decoder layer of the Thinker (width 2,048, experts of 768) and of the Talker (width 1,024, experts of 384, a shared expert of 768): parameters held, "
            "and parameters whose products one token takes part in.\n\n"
            + _table(["part", "Thinker: held", "Thinker: used", "Talker: held", "Talker: used"], rows, "lrrrr"))


# ================================================================================================ the Thinker-to-Talker interface


def talker_prompt_rows():
    n_user, media = 6, (2, 3, 4)
    n_asst = 7
    rows = []
    for i in range(n_user):
        rows.append((i, "user turn", "hidden_projection(thinker hidden at layer 24, position " + str(i) + ")" if i in media
                     else f"text_projection(thinker embedding of user token {i})", "-"))
    text = ["a0", "a1", "a2", "pad", "pad", "pad", "pad", "bos", "a3"]
    codec = ["-", "-", "-", "nothink", "think_bos", "think_eos", "speaker", "codec_pad", "codec_bos"]
    for j in range(9):
        left = text[j]
        left = f"text_projection(embedding of {left if left in ('a0', 'a1', 'a2', 'a3') else 'tts_' + left})"
        rows.append((n_user + j, "assistant segment", left, "codec_embedding(" + codec[j] + ")" if codec[j] != "-" else "0"))
    first = (f"The prefill of the Talker for a user turn of {n_user} tokens (positions 2 to 4 are the spliced media rows) and a reply of {n_asst} text tokens a0 to a6. "
             "Each row is the sum of a text side and a codec side; `text_projection` and `hidden_projection` each map 2,048 numbers to the Talker's 1,024. "
             "`pad`, `bos` and `eos` are the special text ids whose embeddings go through `text_projection`.\n\n"
             + _table(["row", "segment", "text side", "codec side"], rows, "rlll"))
    trailing = [f"text_projection(embedding of a{k})" for k in range(4, n_asst)] + ["text_projection(embedding of tts_eos)"]
    feed = []
    for s in range(8):
        feed.append((s, trailing[s] if s < len(trailing) else "text_projection(embedding of tts_pad)"))
    second = (f"After the prefill ({n_user + 9} rows), every frame s that the Talker writes is fed back as one row: the sum of the 16 codec embeddings of the frame "
              f"and one more row, the one in the table (the text of the reply reaches the Talker at one token a frame, then the end marker, then padding).\n\n"
              + _table(["frame s", "text row added to the frame's codes"], feed, "rl")
              + f"\n\nThe reply has {n_asst} tokens, so {n_asst - 4 + 1} rows trail the prefill ({n_asst - 4} tokens and the end marker); a reply of fewer than 4 tokens cannot "
                "fill the template.")
    return first + "\n\n" + second


# ================================================================================================ from codes to samples


def _samples(frames):
    """Output samples of Code2Wav for `frames` code frames: after the two upsampling stages (x2 x2) the length is 4T, and each SEANet block's
    transposed convolution (kernel 2r, stride r) crops its padding off both sides, (L - 1) r."""
    n = frames * C_UP[0] * C_UP[1]
    stages = [n]
    for r in C_RATES:
        n = (n - 1) * r
        stages.append(n)
    return n, stages


def code_to_wave():
    n, stages = _samples(8)
    names = ["codes [8, 16] -> mean of 16 embeddings -> transformer -> after the two upsampling stages", "after block 1 (x8, crop)", "after block 2 (x5, crop)",
             "after block 3 (x4, crop)", "after block 4 (x3, crop)"]
    rows = [(names[0], _i(stages[0])), (names[1], _i(stages[1])), (names[2], _i(stages[2])), (names[3], _i(stages[3])), (names[4], _i(stages[4]))]
    first = ("Positions along the time axis for 8 code frames: the two upsampling stages (each by 2) turn 8 into 32, and each of the four SEANet blocks upsamples by "
             "its rate $\\mathsf{u}_j$ in 8, 5, 4, 3 with a transposed convolution of kernel $2\\mathsf{u}_j$ whose output is cropped by $\\mathsf{u}_j$ on both sides, $L \\mapsto (L - 1)\\,\\mathsf{u}_j$.\n\n"
             + _table(["stage", "length"], rows, "lr"))
    rows2 = []
    for t in (1, 2, 8, 38, 155, 256, 300):
        s, _ = _samples(t)
        rows2.append((t, _i(1920 * t), _i(s), _i(1920 * t - s), f"{s / 24000:.3f}"))
    second = ("Samples for T frames: the product 1,920 T that the total upsampling factor suggests, and the length that the stages give.\n\n"
              + _table(["frames T", "1,920 T", "samples", "missing", "seconds at 24,000 Hz"], rows2))
    return first + "\n\n" + second


def code2wav_work():
    c = code2wav_frame_macs()
    rows = [("mean of 16 embeddings (additions, not counted)", "", "", "0"),
            ("8-layer transformer, weights: 8 x (4 x 1,024^2 + 3 x 1,024 x 3,072)", _i(c["transformer"]), "1", _m(c["transformer"], 1))]
    for name, _, _, _, m, u in code2wav_layout():
        rows.append((name, _i(m), _i(u), _m(m * u, 1)))
    rows.append(("**one frame**", "", "", _m(c["total"], 1)))
    rows.append(("one second (12.5 frames)", "", "", _m(12.5 * c["total"], 1)))
    return ("Multiply-adds of the code-to-wave decoder per code frame: the multiply-adds a position of each layer, as in the layout above (the "
            "transformer's weights are counted per frame, with no attention scores), the positions that one frame has on that layer's axis, and "
            "their product in millions. A frame is 1 position at the transformer, 2 and then 4 at the upsampling stages, and 1,920 at the "
            "last SEANet block, so the SEANet decoder works at the sample rate. Chapter 3 counts the same layers per second, at 12.5 frames "
            "a second; the figures here are those divided by 12.5 and are derived above from the shapes.\n\n"
            + _table(["layer", "multiply-adds a position", "positions a frame", "millions a frame"], rows, "lrrr"))


# ================================================================================================ first sound


N_USER, CHUNK = 20, 8


def _first_sound(p, n, f, user, chunk=300, order="brain"):
    r = _per_role()
    c = code2wav_frame_macs()
    layers = r["thinker_row"]
    th_prefill = p * layers + r["head"]
    th_decode = (n - 1) * (layers + r["head"])
    tk_rows = user + 9
    tk_prefill = tk_rows * (K_LAYERS * r["talker_layer"] + K_VOCAB * K_D)
    tk_frames = f * (r["talker_frame"] + r["predictor_frame"])
    front = f * c["transformer"]
    back = min(chunk, f) * c["rest"]
    return {"thinker_prefill": th_prefill, "thinker_decode": th_decode, "talker_prefill": tk_prefill, "talker_frames": tk_frames, "front": front, "back": back,
            "total": th_prefill + th_decode + tk_prefill + tk_frames + front + back}


def first_audio_work():
    prompt = 40
    rows = []
    for n, f in ((8, 20), (40, 100), (80, 200), (160, 256)):
        b = _first_sound(prompt, n, f, N_USER)
        # the order the interfaces allow: the Talker may start once the reply has 4 tokens, and the first chunk is CHUNK frames
        s = _first_sound(prompt, 4, CHUNK, N_USER, chunk=CHUNK)
        rows.append((n, f, _b(b["total"], 1), _b(s["total"], 1), f"{b['total'] / s['total']:.1f}"))
    first = (f"Multiply-adds, in billions, that must be done before the first sample of the reply exists, for a prompt of {prompt} rows, a user turn of {N_USER} tokens for the Talker, a "
             f"reply of n text tokens and F code frames. The crate's order runs the Thinker to the end of the reply, the Talker to the end of its frames and the code-to-wave front "
             f"over all frames, and hands out audio in chunks of up to 300 frames, so for F up to 300 the first chunk is the whole reply. The order the interfaces allow starts the "
             f"Talker when the reply has 4 tokens and hands out the first chunk after {CHUNK} frames (weights only; attention scores and the 25-layer second pass of a "
             "conversation are left out). The count is by these rules. The Thinker's prefill of the prompt gives the reply's first token, so it costs the "
             f"{prompt} rows through the 48 layers and the head once, and each of the n - 1 further tokens one row through the layers and the head; "
             "'4 tokens' is the prefill and 3 further steps. The Talker's prefill is the user turn's "
             f"{N_USER} rows and the template's 9, {N_USER + 9} rows, each through its 20 layers and the codec head (the crate computes the logits of every "
             "prefill row and reads the last); then each of the F frames costs a Talker step and the code predictor's 16 steps. The code-to-wave "
             "transformer costs its weights for each of the F frames, and the upsampling and SEANet back its per-frame count for each frame of the "
             "first chunk, the smaller of 300 and F frames.\n\n"
             + _table(["reply tokens n", "code frames F", "brain's order", f"first chunk of {CHUNK} frames", "ratio"], rows))
    b = _first_sound(prompt, 40, 100, N_USER)
    parts = [("Thinker: prefill of the prompt", b["thinker_prefill"]), ("Thinker: the rest of 40 reply tokens", b["thinker_decode"]),
             ("Talker: prefill (29 rows)", b["talker_prefill"]), ("Talker and code predictor: 100 frames", b["talker_frames"]),
             ("code-to-wave: transformer over 100 frames", b["front"]), ("code-to-wave: upsampling and SEANet over the first chunk (100 frames)", b["back"])]
    second = ("Where those multiply-adds of brain's order go for n = 40 and F = 100 (in billions).\n\n"
              + _table(["stage", "billions", "percent"], [(n_, _b(m, 1), f"{100 * m / b['total']:.1f}") for n_, m in parts] + [("**total**", _b(b["total"], 1), "100.0")]))
    return first + "\n\n" + second


def _talker_used_params():
    """Parameters whose products one token takes part in, in one Talker layer: attention, norms, router, 6 routed experts, the shared
    expert and its gate (the `used` column of `moe_layers`)."""
    tk = _talker_counts()
    return tk["attn"] + tk["norms"] + tk["router"] + K_TOPK * 3 * K_D * K_FF + tk["shared"] + tk["gate"]


def talker_stream_bytes():
    tk = _talker_counts()
    per_step = 4 * K_LAYERS * tk["layer"]
    used = 4 * K_LAYERS * _talker_used_params()
    rows = [("every weight of the 20 layers uploaded for one step (32-bit)", _i(per_step)),
            ("the weights that step's products use (32-bit)", _i(used)),
            ("held / used", f"{per_step / used:.2f}"),
            ("prefill and 100 frames: 101 steps", _i(101 * per_step)),
            ("the same in gibibytes", f"{101 * per_step / 2**30:,.1f}")]
    return ("Bytes that the Talker's weights move in brain's validation-tier path, where every step re-reads all 20 layers of its 128 experts each from the checkpoint "
            "(32-bit values), against the bytes that the step's products use. Both count what the `moe_layers` table counts, norms included: a layer holds "
            f"{_i(tk['layer'])} parameters and uses {_i(_talker_used_params())}.\n\n" + _table(["quantity", "bytes"], rows, "lr"))
