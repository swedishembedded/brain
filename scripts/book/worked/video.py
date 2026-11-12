# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Video as Data".

Plain Python 3, standard library only, written from the chapter's own description and independent of any engine code. Each public
function without arguments returns a markdown string that the chapter embeds between `<!-- worked: video.<name> -->` markers; the
functions with arguments (or a leading underscore) supply the numbers of the chapter's figures, so a picture and the table beside
it cannot disagree.

* frame rates as exact fractions, the time of frame i, and what a 1 ms container clock does to it;
* a wheel with six spokes filmed at 24 frames per second, by the folding rule of the sound chapter (`sound.alias_of`);
* a 5 by 3 frame in the planes of a video: limited-range luma, 2 by 2 chroma means, nearest-sample chroma back;
* the bytes of a minute of raw, planar and coded video;
* a clip of eight 16 by 16 frames that pans one pixel per frame: the ideal code length of the frames coded alone, as differences
  and with block motion vectors, and the error of a lossy loop with and without the encoder tracking the decoder;
* where the frames of a clip and the packets of its sound sit in a WebM and an MP4 file;
* frame selection by timestamp (a 30 frames per second clip read at 12, and 8 frames spread over a clip);
* a causal and a symmetric temporal convolution on one pixel's history;
* the latent grid of a causal video autoencoder: shapes through the encoder, the time each latent frame covers, and the values saved.

Arithmetic is the language's 64-bit floating point (correctly rounded library sqrt, log2) unless a block says otherwise. Sums are
accumulated left to right in index order. The only random numbers are the texture of the pan clip, from the book's SplitMix64
(`bookrng.generator`, stream name printed by the chapter).

Swedish Embedded AB implements solutions for reading, writing and conditioning video inside a model pipeline. If your team needs
expertise in video codecs, containers and video autoencoders, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math

import bookrng
import images as im
import sound as snd

_table = im._table


def half_away(x):
    """The integer nearest to x, ties away from zero (Rust's f32::round, and the pseudocode's round(x, 0))."""
    return math.floor(x + 0.5) if x >= 0 else -math.floor(-x + 0.5)


# ------------------------------------------------------------------------------------------------ frame rates and timestamps


def fps_rational(fps):
    """A frame rate (a decimal number) as (numerator, denominator) frames per second, by the rule of the chapter."""
    if not (math.isfinite(fps) and 0 < fps <= 1000):
        raise ValueError("a frame rate of %r" % fps)
    if abs(fps - round(fps)) < 1e-9:
        return round(fps), 1
    ntsc = fps * 1001 / 1000
    if abs(ntsc - round(ntsc)) < 1e-6:
        return round(ntsc) * 1000, 1001
    num = half_away(fps * 1000)
    g = math.gcd(num, 1000)
    return num // g, 1000 // g


def snap_fps(fps):
    """A rate read from an integer clock, snapped to the integer or n * 1000 / 1001 rate it is within a millionth of."""
    def close(x):
        return abs(fps - x) <= 1e-6 * fps
    if close(round(fps)):
        return float(round(fps))
    ntsc = round(fps * 1.001) / 1.001
    return ntsc if close(ntsc) else fps


def frame_ns(i, num, den):
    """The start of frame i in whole nanoseconds, the integer arithmetic of a Matroska writer."""
    return i * 1_000_000_000 * den // num


def mkv_ms(i, num, den):
    return frame_ns(i, num, den) // 1_000_000


def rate_rule():
    cases = [("24", 24.0), ("25", 25.0), ("30000 / 1001", 30000 / 1001), ("60000 / 1001", 60000 / 1001), ("29.97", 29.97),
             ("23.976", 23.976), ("12.5", 12.5)]
    rows = []
    for text, fps in cases:
        n, d = fps_rational(fps)
        rows.append([text, f"{fps:.8f}", f"{n} / {d}", f"{n / d:.8f}"])
    drift_frames = 107892
    ntsc = drift_frames * 1001 / 30000
    dec = drift_frames * 100 / 2997
    return ("The rule: an integer within 1e-9 is that integer over 1; else if the rate * 1001 / 1000 is within 1e-6 of a whole number n it is "
            "n * 1000 / 1001; else it is round(1000 * rate) / 1000 reduced to lowest terms. Column 2 is the input as a 64-bit number.\n\n"
            + _table(["input", "as a number", "fraction", "fraction as a number"], rows, "lrrr")
            + f"\n\nAfter {drift_frames} frames (one hour at 30000 / 1001) the time is {ntsc:.4f} seconds at 30000 / 1001 and {dec:.4f} seconds at 2997 / 100: "
            f"the two differ by {1000 * (dec - ntsc):.1f} thousandths of a second.")


def timestamp_table():
    rows = []
    for name, (num, den) in (("24", (24, 1)), ("30000 / 1001", (30000, 1001))):
        for i in (1, 2, 3, 5, 10, 100):
            exact = i * den / num * 1000
            ms = mkv_ms(i, num, den)
            rows.append([name, i, f"{exact:.4f}", ms, f"{exact - ms:.4f}"])
    out = ("Frame i starts at i * den / num seconds. A Matroska writer with a clock of thousandths of a second stores floor(floor(i * 10^9 * den / num) / 10^6), "
           "whole nanoseconds first, whole milliseconds second; an MP4 track whose clock ticks num times a second and whose frames last den "
           "ticks stores i * den, which is exact.\n\n"
           + _table(["rate", "i", "exact time (thousandths of a second)", "stored (thousandths)", "lost (thousandths)"], rows, "lrrrr"))
    dur = frame_ns(1, 24, 1)
    dur_n = frame_ns(1, 30000, 1001)
    out += (f"\n\nThe frame duration a Matroska track declares is floor(10^9 den / num) ns: {dur} for 24 and {dur_n} for 30000 / 1001. "
            f"Read back as 10^9 / duration these are {1e9 / dur:.7f} and {1e9 / dur_n:.7f}, and the snap rule (an integer, or n / 1.001 with "
            f"n an integer, when within a millionth) returns {snap_fps(1e9 / dur):g} and {snap_fps(1e9 / dur_n):.8f}.")
    return out


# ------------------------------------------------------------------------------------------------ the wheel

SPOKES = 6
WHEEL_FPS = 24


def wheel_apparent(rev_per_s, spokes=SPOKES, fps=WHEEL_FPS):
    """The signed apparent speed in turns per second: the spoke-passing frequency folded by the sound chapter's rule."""
    r, sign = snd.alias_of(spokes * rev_per_s, float(fps))
    return sign * r / spokes


def wheel_data(lo=0.0, hi=14.0, n=281):
    xs = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    return xs, [wheel_apparent(x) for x in xs]


def wheel_table():
    rows = []
    for f in (0.5, 1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.5, 8.0, 11.0):
        passes = SPOKES * f
        app = wheel_apparent(f)
        tie = abs(passes % WHEEL_FPS - WHEEL_FPS / 2) < 1e-9
        word = "either way" if tie else "forward" if app > 0 else "backward" if app < 0 else "still"
        rows.append([f"{f:g}", f"{passes:g}", f"{app * SPOKES:+g}", f"{app:+g}", word])
    out = ("A wheel of 6 equal spokes looks the same after every 60 degrees, so a wheel at f turns per second passes 6 f spokes per second "
           "past a fixed point. The camera samples at 24 frames per second; the folding rule of the sound chapter gives the spoke "
           "passes per second that the film shows, positive when the wheel seems to turn the way it does.\n\n"
           + _table(["true turns / s", "spoke passes / s", "apparent passes / s", "apparent turns / s", "looks"], rows, "lrrrl"))
    # the direct check by frames: where the marker spoke sits, folded into (-30, 30] degrees
    lines = []
    for f in (3.5, 4.5):
        step = 360 * f / WHEEL_FPS
        marks = []
        for n in range(4):
            a = (step * n) % 60.0
            if a > 30.0:
                a -= 60.0
            marks.append(f"{a:g}")
        lines.append(f"At {f:g} turns per second the wheel advances {step:g} degrees per frame, and the marked spoke is seen, folded into "
                     f"(-30, 30] degrees, at {', '.join(marks)} degrees on frames 0 to 3.")
    return out + "\n\n" + "\n\n".join(lines)


# ------------------------------------------------------------------------------------------------ the planes of a frame

KR, KB = 0.2126, 0.0722
FRAME_W, FRAME_H = 5, 3
FRAME_RGB = [
    [(200, 40, 40), (210, 50, 30), (190, 60, 50), (30, 40, 200), (20, 50, 210)],
    [(200, 120, 40), (180, 140, 60), (60, 150, 80), (40, 60, 190), (30, 70, 200)],
    [(90, 90, 90), (100, 100, 100), (120, 130, 140), (50, 80, 180), (60, 90, 170)],
]


def _luma(r, g, b):
    return KR * r + (1 - KR - KB) * g + KB * b


def to_planes(frame):
    """Limited-range BT.709 planes: luma 16 + 219 L, chroma 128 + 224 C over each 2 x 2 block (what exists of it at an edge)."""
    h, w = len(frame), len(frame[0])
    y = [[half_away(16 + 219 * _luma(*frame[r][c]) / 255) for c in range(w)] for r in range(h)]
    ch, cw = -(-h // 2), -(-w // 2)
    u = [[0] * cw for _ in range(ch)]
    v = [[0] * cw for _ in range(ch)]
    for cy in range(ch):
        for cx in range(cw):
            cb = cr = 0.0
            n = 0
            for r in range(cy * 2, min(cy * 2 + 2, h)):
                for c in range(cx * 2, min(cx * 2 + 2, w)):
                    rr, gg, bb = (v_ / 255 for v_ in frame[r][c])
                    ll = _luma(rr, gg, bb)
                    cb += (bb - ll) / (2 * (1 - KB))
                    cr += (rr - ll) / (2 * (1 - KR))
                    n += 1
            u[cy][cx] = half_away(128 + 224 * cb / n)
            v[cy][cx] = half_away(128 + 224 * cr / n)
    return y, u, v


def from_planes(y, u, v):
    """The inverse: the chroma sample of the 2 x 2 block that holds the pixel, then the matrix, rounded and clamped to 0..255."""
    co = im.yuv_coefficients(KR, KB)
    out = []
    for r, row in enumerate(y):
        line = []
        for c, yy in enumerate(row):
            ll = (yy - 16) / 219
            cb = (u[r // 2][c // 2] - 128) / 224
            cr = (v[r // 2][c // 2] - 128) / 224
            rr, gg, bb = ll + co["r_cr"] * cr, ll - co["g_cb"] * cb - co["g_cr"] * cr, ll + co["b_cb"] * cb
            line.append(tuple(min(max(half_away(255 * t), 0), 255) for t in (rr, gg, bb)))
        out.append(line)
    return out


def _margin_from_tie(frame):
    """The smallest distance of any value rounded in to_planes from a .5 tie: a tie would make 32-bit and 64-bit arithmetic disagree."""
    worst = 1.0
    h, w = len(frame), len(frame[0])
    for row in frame:
        for p in row:
            x = 16 + 219 * _luma(*p) / 255
            worst = min(worst, abs((x % 1.0) - 0.5))
    for cy in range(-(-h // 2)):
        for cx in range(-(-w // 2)):
            cb = cr = 0.0
            n = 0
            for r in range(cy * 2, min(cy * 2 + 2, h)):
                for c in range(cx * 2, min(cx * 2 + 2, w)):
                    rr, gg, bb = (v_ / 255 for v_ in frame[r][c])
                    ll = _luma(rr, gg, bb)
                    cb += (bb - ll) / (2 * (1 - KB))
                    cr += (rr - ll) / (2 * (1 - KR))
                    n += 1
            for q in (128 + 224 * cb / n, 128 + 224 * cr / n):
                worst = min(worst, abs((q % 1.0) - 0.5))
    return worst


def planes_table():
    y, u, v = to_planes(FRAME_RGB)
    back = from_planes(y, u, v)
    assert _margin_from_tie(FRAME_RGB) > 1e-3
    rows = []
    for r in range(FRAME_H):
        for c in range(FRAME_W):
            p, q = FRAME_RGB[r][c], back[r][c]
            rows.append([f"({c}, {r})", str(p), str(q), max(abs(a - b) for a, b in zip(p, q))])
    pl = lambda m: "\n".join("    " + " ".join(f"{x:3d}" for x in row) for row in m)
    out = ("The 5 by 3 frame has pixels (R, G, B), row by row: " + "; ".join(" ".join(str(p) for p in row) for row in FRAME_RGB) + ".\n\n"
           "BT.709 (K_r = 0.2126, K_b = 0.0722), limited range: Y = round(16 + 219 L), U = round(128 + 224 C_b), V = round(128 + 224 C_r), "
           "with L, C_b, C_r computed from R / 255, G / 255, B / 255, the chroma as the mean over the 2 by 2 block (an edge block averages "
           "the pixels it has), and round() half away from zero.\n\n"
           "```text\nY plane, 5 x 3 = 15 bytes\n" + pl(y) + "\nU plane, 3 x 2 = 6 bytes\n" + pl(u) + "\nV plane, 3 x 2 = 6 bytes\n" + pl(v) + "\n```\n\n"
           "Decoding takes the chroma sample of the block that holds the pixel (the nearest sample), L = (Y - 16) / 219, C = (U - 128) / 224, "
           "then R = L + 2 (1 - K_r) C_r, B = L + 2 (1 - K_b) C_b, G = (L - K_r R - K_b B) / K_g, each times 255 and rounded half away "
           "from zero, clamped to 0 .. 255.\n\n"
           + _table(["pixel (x, y)", "original", "decoded", "largest error"], rows, "llrr"))
    return out + f"\n\nThe frame is {FRAME_W * FRAME_H * 3} bytes as RGB and {FRAME_W * FRAME_H + 2 * 3 * 2} bytes in these planes."


# ------------------------------------------------------------------------------------------------ the cost of a minute


def minute_costs():
    rows = []
    for name, w, h in (("640 x 360", 640, 360), ("1024 x 512", 1024, 512), ("1280 x 720", 1280, 720), ("1920 x 1080", 1920, 1080)):
        frames = 24 * 60
        rgb = w * h * 3 * frames
        planar = (w * h + 2 * (w // 2) * (h // 2)) * frames
        rows.append([name, f"{w * h * 3:,}", f"{rgb:,}", f"{planar:,}"])
    coded_bps = 220_000
    coded = coded_bps * 60 // 8
    w, h = 1024, 512
    rgb = w * h * 3 * 1440
    planar = (w * h * 3 // 2) * 1440
    return ("A minute is 24 * 60 = 1440 frames at 24 frames per second. RGB has 3 bytes per pixel; the planar 4:2:0 layout has one luma "
            "byte per pixel and two chroma bytes per 2 by 2 block, 1.5 bytes per pixel for even sizes.\n\n"
            + _table(["size", "bytes per RGB frame", "bytes per minute, RGB", "bytes per minute, 4:2:0"], rows, "lrrr")
            + f"\n\nA coded stream of {coded_bps:,} bits per second holds a minute in {coded:,} bytes, {planar / coded:.1f} times fewer than the "
            f"4:2:0 minute of 1024 x 512 ({planar:,} bytes) and {rgb / coded:.1f} times fewer than its RGB minute ({rgb:,} bytes).")


# ------------------------------------------------------------------------------------------------ the toy clip

TOY_NAME = "video.toy_scene"
TOY_W, TOY_H, TOY_FRAMES, TOY_BLOCK, TOY_RANGE, TOY_SCENE_W = 16, 16, 8, 4, 2, 24
TOY_Q = 8


def toy_scene():
    g = bookrng.generator(TOY_NAME)
    return [[g.integer(256) for _ in range(TOY_SCENE_W)] for _ in range(TOY_H)]


def toy_frames():
    s = toy_scene()
    return [[[s[y][x + t] for x in range(TOY_W)] for y in range(TOY_H)] for t in range(TOY_FRAMES)]


def entropy_bits(symbols):
    """n times the plug-in entropy -sum p log2 p of the values in the list: an ideal code length for that list, in bits."""
    n = len(symbols)
    counts = {}
    for s in symbols:
        counts[s] = counts.get(s, 0) + 1
    return -sum(c * math.log2(c / n) for c in counts.values())


def _flat(m):
    return [x for row in m for x in row]


def _clamp(i, n):
    return min(max(i, 0), n - 1)


def _predict_block(prev, bx, by, dx, dy):
    return [[prev[_clamp(by + j + dy, TOY_H)][_clamp(bx + i + dx, TOY_W)] for i in range(TOY_BLOCK)] for j in range(TOY_BLOCK)]


def best_vector(cur, prev, bx, by, rng=TOY_RANGE):
    """The vector (dx, dy) in -rng .. rng with the least sum of absolute differences; ties go to the smaller |dx| + |dy|, then dy, then dx."""
    best = None
    block = [[cur[by + j][bx + i] for i in range(TOY_BLOCK)] for j in range(TOY_BLOCK)]
    cand = sorted(((dx, dy) for dx in range(-rng, rng + 1) for dy in range(-rng, rng + 1)), key=lambda v: (abs(v[0]) + abs(v[1]), v[1], v[0]))
    for dx, dy in cand:
        pred = _predict_block(prev, bx, by, dx, dy)
        sad = sum(abs(block[j][i] - pred[j][i]) for j in range(TOY_BLOCK) for i in range(TOY_BLOCK))
        if best is None or sad < best[0]:
            best = (sad, dx, dy)
    return best[1], best[2]


def toy_vectors(cur, prev, rng=TOY_RANGE):
    return [best_vector(cur, prev, bx, by, rng) for by in range(0, TOY_H, TOY_BLOCK) for bx in range(0, TOY_W, TOY_BLOCK)]


def _residual(cur, prev, vectors):
    res = [[0] * TOY_W for _ in range(TOY_H)]
    k = 0
    for by in range(0, TOY_H, TOY_BLOCK):
        for bx in range(0, TOY_W, TOY_BLOCK):
            dx, dy = vectors[k]
            k += 1
            pred = _predict_block(prev, bx, by, dx, dy)
            for j in range(TOY_BLOCK):
                for i in range(TOY_BLOCK):
                    res[by + j][bx + i] = cur[by + j][bx + i] - pred[j][i]
    return res


VECTOR_BITS = math.log2((2 * TOY_RANGE + 1) ** 2)


def toy_costs():
    fr = toy_frames()
    rows = []
    for t, cur in enumerate(fr):
        intra = entropy_bits([p - 128 for p in _flat(cur)])
        if t == 0:
            rows.append((t, intra, intra, intra, sum(1 for p in _flat(cur) if p != 128), None))
            continue
        prev = fr[t - 1]
        zero = entropy_bits(_flat(_residual(cur, prev, [(0, 0)] * 16)))
        vecs = toy_vectors(cur, prev)
        res = _residual(cur, prev, vecs)
        mv = entropy_bits(_flat(res)) + 16 * VECTOR_BITS
        rows.append((t, intra, zero, mv, sum(1 for x in _flat(res) if x), vecs))
    return rows


def toy_table():
    rows = toy_costs()
    out_rows = []
    for t, intra, zero, mv, nz, vecs in rows:
        if vecs is None:
            out_rows.append([t, f"{intra:.1f}", f"{intra:.1f}", f"{intra:.1f}", nz, "-"])
        else:
            uniq = sorted(set(vecs))
            vs = "all " + str(uniq[0]) if len(uniq) == 1 else "mixed: " + ", ".join(f"{u} x {vecs.count(u)}" for u in uniq)
            out_rows.append([t, f"{intra:.1f}", f"{zero:.1f}", f"{mv:.1f}", nz, vs])
    tot = [sum(r[i] for r in rows) for i in (1, 2, 3)]
    out_rows.append(["total bits", f"{tot[0]:.1f}", f"{tot[1]:.1f}", f"{tot[2]:.1f}", "", ""])
    out_rows.append(["total bytes", f"{tot[0] / 8:.1f}", f"{tot[1] / 8:.1f}", f"{tot[2] / 8:.1f}", "", ""])
    scene = toy_scene()
    return (f"The scene is a texture of {TOY_H} rows and {TOY_SCENE_W} columns of integers 0 .. 255, drawn by the book's SplitMix64 from the "
            f"stream `{TOY_NAME}` (seed {bookrng.seed_of(TOY_NAME)}) with `integer(256)`, row by row (row 0 first, the columns of a row left to right); "
            f"its first row is {scene[0]}. Frame t shows columns t .. t + 15, so the picture pans one pixel to the left per frame; there are "
            f"{TOY_FRAMES} frames. Each frame is cut into 16 blocks of 4 by 4. The cost of a list of values is n times their plug-in entropy "
            f"-sum p log2 p (p the share of each distinct value in the list), in bits. 'Alone': values minus 128. 'Difference': minus the previous "
            f"frame's pixel at the same place. 'Motion': minus the previous frame's pixel at the place the block's vector points to (clamped to "
            f"the frame), with one vector per block from -{TOY_RANGE} .. {TOY_RANGE} in each direction (25 choices, log2 25 = {VECTOR_BITS:.3f} "
            f"bits each), picked by the least sum of absolute differences, ties to the smaller |dx| + |dy|, then dy, then dx. Frame 0 has no "
            f"previous frame and is coded alone in all three.\n\n"
            + _table(["frame", "alone (bits)", "difference (bits)", "motion (bits)", "non-zero residuals", "vectors (dx, dy)"], out_rows, "lrrrrl"))


def toy_bytes_data():
    rows = toy_costs()
    return [r[0] for r in rows], [r[1] / 8 for r in rows], [r[2] / 8 for r in rows], [r[3] / 8 for r in rows]


def _quant(r, q=TOY_Q):
    m = math.floor(abs(r) / q + 0.5)
    return m if r >= 0 else -m


def _clip255(x):
    return min(max(x, 0), 255)


def toy_loops():
    """Max abs error per frame of the lossy loop, with the predictor taken from the decoder's frame (closed) or from the original (open)."""
    fr = toy_frames()
    vecs = [None] + [toy_vectors(fr[t], fr[t - 1]) for t in range(1, TOY_FRAMES)]
    rec_c = [[_clip255(128 + TOY_Q * _quant(p - 128)) for p in row] for row in fr[0]]
    rec_o = [row[:] for row in rec_c]
    out = [(0, max(abs(a - b) for a, b in zip(_flat(fr[0]), _flat(rec_c))), max(abs(a - b) for a, b in zip(_flat(fr[0]), _flat(rec_o))))]
    for t in range(1, TOY_FRAMES):
        res_closed = _residual(fr[t], rec_c, vecs[t])
        res_open = _residual(fr[t], fr[t - 1], vecs[t])
        pred_c = [[fr[t][y][x] - res_closed[y][x] for x in range(TOY_W)] for y in range(TOY_H)]
        new_c = [[_clip255(pred_c[y][x] + TOY_Q * _quant(res_closed[y][x])) for x in range(TOY_W)] for y in range(TOY_H)]
        # the decoder's prediction from its own previous frame, plus the residual the encoder computed against the original
        pred_o = _predict_frame(rec_o, vecs[t])
        new_o = [[_clip255(pred_o[y][x] + TOY_Q * _quant(res_open[y][x])) for x in range(TOY_W)] for y in range(TOY_H)]
        rec_c, rec_o = new_c, new_o
        out.append((t, max(abs(a - b) for a, b in zip(_flat(fr[t]), _flat(rec_c))), max(abs(a - b) for a, b in zip(_flat(fr[t]), _flat(rec_o)))))
    return out


def _predict_frame(prev, vectors):
    pred = [[0] * TOY_W for _ in range(TOY_H)]
    k = 0
    for by in range(0, TOY_H, TOY_BLOCK):
        for bx in range(0, TOY_W, TOY_BLOCK):
            dx, dy = vectors[k]
            k += 1
            blk = _predict_block(prev, bx, by, dx, dy)
            for j in range(TOY_BLOCK):
                for i in range(TOY_BLOCK):
                    pred[by + j][bx + i] = blk[j][i]
    return pred


def loop_table():
    rows = [[t, c, o] for t, c, o in toy_loops()]
    return (f"The same clip, with every residual replaced by {TOY_Q} times its quantised value, q(r) = sign(r) floor(|r| / {TOY_Q} + 1/2) "
            f"(ties away from zero), and every rebuilt pixel clamped to 0 .. 255. Frame 0 is coded alone with prediction 128. The vectors are "
            f"those of the table above, found on the original frames. Closed loop: the residual is taken against the decoder's previous frame. "
            f"Open loop: against the original previous frame, while the decoder still predicts from its own.\n\n"
            + _table(["frame", "largest error, closed loop", "largest error, open loop"], rows, "lrr"))


def access_table():
    rows = toy_costs()
    intra = rows[0][1] / 8
    p_mean = sum(r[3] for r in rows[1:]) / (len(rows) - 1) / 8
    out = []
    for g in (1, 8, 24, 120):
        last_key = 100 // g * g
        out.append([g, 100 - last_key + 1, f"{(intra + (g - 1) * p_mean) / g:.1f}"])
    return (f"To show frame 100 a decoder starts at the last key frame at or before it. With key frames every G frames that is frame "
            f"G floor(100 / G), and it decodes 100 - G floor(100 / G) + 1 frames. The last column prices a frame at the toy's costs: "
            f"{intra:.1f} bytes alone and {p_mean:.1f} bytes on average for frames 1 to 7 with motion; a group of G costs "
            f"(alone + (G - 1) motion) / G per frame.\n\n"
            + _table(["G", "frames decoded for frame 100", "toy bytes per frame"], out, "lrr"))


# ------------------------------------------------------------------------------------------------ audio and video in one file

OPUS_PACKET = 960
PRE_SKIP = 312


def container_av(fps=(24, 1), audio_rate=16000, seconds=1, pre_skip=PRE_SKIP):
    num, den = fps
    frames = seconds * num // den
    n48 = -(-audio_rate * seconds * 48000 // audio_rate)
    total = n48 + pre_skip
    packets = -(-total // OPUS_PACKET)
    return num, den, frames, n48, total, packets


def av_blocks():
    num, den, frames, n48, total, packets = container_av()
    blocks = [(mkv_ms(i, num, den), 1, f"video frame {i}") for i in range(frames)]
    blocks += [(20 * j, 2, f"audio packet {j}") for j in range(packets)]
    blocks.sort(key=lambda b: (b[0], b[1]))
    rows = [[t, trk, what] for t, trk, what in blocks[:14]]
    return rows


def av_table():
    num, den, frames, n48, total, packets = container_av()
    rows = av_blocks()
    mp4_audio_ms = (packets * OPUS_PACKET - PRE_SKIP) * 1000 // 48000
    mp4_video_ms = frames * den * 1000 // num
    mkv_video = mkv_ms(frames, num, den)
    mkv_audio = packets * OPUS_PACKET * 1000 // 48000
    late_ms = PRE_SKIP * 1000 / 48000
    return (f"One second of mono sound at 16 000 Hz and 24 video frames. The sound is brought to 48 000 Hz by linear interpolation "
            f"(ceil(16 000 * 48 000 / 16 000) = {n48} samples), Opus codes packets of {OPUS_PACKET} samples (a twentieth of a second) and its encoder reports a "
            f"look-ahead, here taken as {PRE_SKIP} samples, which is appended as {PRE_SKIP} more samples of coding so that dropping it keeps the end: "
            f"{total} samples, {packets} packets (ceil({total} / {OPUS_PACKET})).\n\n"
            "A WebM file writes every block of both tracks in the order of (time in thousandths of a second, track), track 1 the video, track 2 the sound; the first 14:\n\n"
            + _table(["time (ms)", "track", "block"], rows, "rrl")
            + f"\n\nVideo frame i is at floor(i 10^9 / 24 / 10^6) thousandths of a second and audio packet j at 20 j. The file's duration, in the same unit, is the later of the two ends: "
            f"video {mkv_video}, audio {mkv_audio} ({packets} packets of 20), so {max(mkv_video, mkv_audio)}. An MP4 file keeps "
            f"each track on its own clock instead: the video track ticks 24 times a second with every frame lasting 1 tick, the sound track ticks "
            f"48 000 times a second with every packet lasting {OPUS_PACKET} ticks, and an edit list starts the sound at tick {PRE_SKIP}. Its "
            f"duration is the later of video {mp4_video_ms} and sound ({packets} * {OPUS_PACKET} - {PRE_SKIP}) * 1000 / 48 000 = "
            f"{mp4_audio_ms} (whole thousandths, rounded down), so {max(mp4_video_ms, mp4_audio_ms)}. A player that ignored the {PRE_SKIP} "
            f"samples would play the sound {late_ms:.1f} thousandths of a second late, {late_ms / (1000 * den / num):.3f} of a frame.")


# ------------------------------------------------------------------------------------------------ frame selection

TIE_S = 1e-3


def pick_frames(times, fps, max_frames):
    """Output instants start + j / fps, each taking the source frame nearest to it, a tie (within 1 ms) the earlier; after the last
    frame an instant is covered until one more source interval has passed. Returns (instants, source indices)."""
    start = times[0]
    last = len(times) - 1
    gap = times[last] - times[last - 1] if last else 1.0 / fps
    out_t, out_i = [], []
    for j in range(max_frames):
        s = start + j / fps
        pick = None
        for i in range(last):
            if s <= (times[i] + times[i + 1]) / 2 + TIE_S:
                pick = i
                break
        if pick is None:
            if s < times[last] + gap - 1e-9:
                pick = last
            else:
                break
        out_t.append(s)
        out_i.append(pick)
    return out_t, out_i


def pick_without_tolerance(times, fps, max_frames):
    start = times[0]
    last = len(times) - 1
    out = []
    for j in range(max_frames):
        s = start + j / fps
        pick = next((i for i in range(last) if s <= (times[i] + times[i + 1]) / 2), last)
        out.append(pick)
    return out


def frame_pick():
    n30 = 30
    times = [mkv_ms(i, 30, 1) / 1000 for i in range(n30)]
    ts, idx = pick_frames(times, 12.0, 12)
    strict = pick_without_tolerance(times, 12.0, 12)
    rows = []
    for j, (s, i) in enumerate(zip(ts, idx)):
        rows.append([j, f"{s * 1000:.3f}", f"{j * 30 / 12:g}", i, strict[j]])
    out = ("A clip of 30 frames at 30 frames per second in a Matroska file, whose stored times, in thousandths of a second, are floor(i * 10^9 / 30 / 10^6) "
           f"({', '.join(str(mkv_ms(i, 30, 1)) for i in range(8))}, ...). Asked for 12 frames per second, output j is the instant "
           "(stored time of frame 0) + j / 12 and takes the source frame whose time is nearest, a tie to the earlier frame, where 'tie' "
           "means at most a thousandth of a second above the midpoint of two stored times (an instant below the midpoint is the earlier frame's anyway).\n\n"
           + _table(["j", "instant (thousandths)", "position in frames", "frame, with the tie rule", "frame, strict midpoint"], rows, "lrrrr"))
    n48 = 48
    t48 = [mkv_ms(i, 24, 1) / 1000 for i in range(n48)]
    duration = t48[-1] - t48[0] + 1 / 24
    n = 8
    rate = (n - 0.5) / duration
    ts2, idx2 = pick_frames(t48, rate, n)
    rows2 = [[j, f"{s * 1000:.3f}", i] for j, (s, i) in enumerate(zip(ts2, idx2))]
    out += (f"\n\nEight frames spread over a clip of {n48} frames at 24 frames per second: the clip's duration is the last stored time minus the "
            f"first plus 1 / 24 of a second = {duration:.6f} seconds, and asking for n = {n} frames sets the rate to (n - 1/2) / duration = {rate:.5f} per second, "
            f"which makes exactly n instants fall inside the clip: instants j / rate for j = 0 .. {n - 1} are before the end "
            f"(the last at {ts2[-1]:.4f} seconds) and the next, {n / rate:.4f} seconds, is after {duration:.4f} seconds.\n\n"
            + _table(["j", "instant (thousandths)", "source frame"], rows2, "rrr"))
    return out


# ------------------------------------------------------------------------------------------------ causal convolution in time

CAUSAL_W = (0.125, 0.25, 0.5)
CAUSAL_X = [8.0, 16.0, 4.0, 12.0, 20.0, 24.0]


def conv_time(x, mode):
    """Kernel of 3 taps over time, tap k times padded[t + k]; 'causal' pads two copies of the first frame before, 'symmetric' one
    copy of the first before and one of the last after."""
    if mode == "causal":
        p = [x[0], x[0]] + list(x)
    else:
        p = [x[0]] + list(x) + [x[-1]]
    return [sum(CAUSAL_W[k] * p[t + k] for k in range(3)) for t in range(len(x))]


def causal_table():
    x2 = CAUSAL_X[:-1] + [100.0]
    rows = []
    for name, mode in (("causal", "causal"), ("symmetric", "symmetric")):
        a, b = conv_time(CAUSAL_X, mode), conv_time(x2, mode)
        rows.append([name, "x ending 24"] + [f"{v:g}" for v in a])
        rows.append([name, "x ending 100"] + [f"{v:g}" for v in b])
    prefix = conv_time(CAUSAL_X[:3], "causal")
    full = conv_time(CAUSAL_X, "causal")[:3]
    return ("One pixel's history over six frames, x = " + str(CAUSAL_X) + ", and a temporal kernel of three taps w = " + str(list(CAUSAL_W))
            + ", y[t] = w[0] p[t] + w[1] p[t + 1] + w[2] p[t + 2] for the padded sequence p (sums in tap order). Causal padding puts two copies of "
            "x[0] before the sequence (so y[t] reads frames t - 2, t - 1, t); symmetric padding puts one copy of x[0] before and one of x[5] after "
            "(frames t - 1, t, t + 1). The second row of each pair changes only the last frame.\n\n"
            + _table(["padding", "input", "y[0]", "y[1]", "y[2]", "y[3]", "y[4]", "y[5]"], rows, "llrrrrrr")
            + f"\n\nCausal padding on the first three frames alone gives {[f'{v:g}' for v in prefix]}, the first three outputs of the full clip "
            f"{[f'{v:g}' for v in full]}.")


# ------------------------------------------------------------------------------------------------ the latent grid

PATCH = 4
LATENT_C = 128
# (kind, (stride_t, stride_h, stride_w), channel multiplier): the down blocks of the video encoder, resnet blocks leave the shape alone
ENC_DOWN = (("space", (1, 2, 2), 2), ("time", (2, 1, 1), 2), ("all", (2, 2, 2), 2), ("all", (2, 2, 2), 1))


def encoder_stages(frames, h, w):
    """(stage, channels, frames, height, width) after each step of the encoder, for a clip of `frames` x h x w pixels."""
    assert (frames - 1) % 8 == 0 and h % 32 == 0 and w % 32 == 0
    stages = [("pixels", 3, frames, h, w)]
    t, hh, ww = frames, h // PATCH, w // PATCH
    c = LATENT_C
    stages.append(("patchify 4 x 4, conv_in", c, t, hh, ww))
    for kind, (st, sh, sw), mult in ENC_DOWN:
        if st == 2:
            t += 1  # frame 0 is repeated once so the count is even
        t, hh, ww, c = t // st, hh // sh, ww // sw, c * mult
        stages.append((f"down: {kind}", c, t, hh, ww))
    stages.append(("conv_out: 128 means + 1 spread", LATENT_C + 1, t, hh, ww))
    return stages


def decoder_frames(lat):
    """Frames after the three time-doubling up blocks, each of which drops the first frame of its output."""
    seq = [lat]
    for _ in range(3):
        seq.append(2 * seq[-1] - 1)
    return seq


def latent_frames(f):
    return 1 + (f - 1) // 8 if f >= 1 and (f - 1) % 8 == 0 else None


def shape_table():
    f, h, w = 121, 512, 768
    st = encoder_stages(f, h, w)
    rows = [[n, c, t, hh, ww, f"{c * t * hh * ww:,}"] for n, c, t, hh, ww in st]
    dec = decoder_frames(st[-1][2])
    return (f"A clip of {f} frames of {h} rows and {w} columns. The encoder: patchify cuts every 4 by 4 patch of the 3 colour channels into "
            f"48 channels at a quarter of the height and width, and a convolution brings them to 128; four down blocks follow (the "
            f"others, the residual blocks, keep the shape). A down block that halves time first repeats frame 0 once so the count is even; its "
            f"output channels are the channels times 2, 2, 2, 1 for the four blocks. The final convolution writes 128 channels of means and one "
            f"of spread.\n\n"
            + _table(["after", "channels", "frames", "height", "width", "values"], rows, "lrrrrr")
            + f"\n\nThe decoder's three time-doubling blocks take {dec[0]} latent frames to {dec[1]}, {dec[2]} and {dec[3]} frames: each doubles the "
            f"count and drops the first frame of the result, 2 n - 1.")


def frame_rule_table():
    rows = []
    for f in (1, 8, 9, 16, 17, 24, 25, 97, 121, 122):
        lf = latent_frames(f)
        rows.append([f, "not allowed" if lf is None else lf])
    near = [(f, latent_frames(f)) for f in (17, 25)]
    return ("Latent frames for a clip of F frames: 1 + (F - 1) / 8 when F - 1 is a multiple of 8, otherwise no clip. 24 frames, a second at "
            "24 frames per second, is not allowed; 25 is.\n\n" + _table(["F", "latent frames"], rows, "lr"))


def window_table():
    f = 25
    lat = latent_frames(f)
    rows = []
    for k in range(lat):
        lo = max(0, 8 * k - 7)
        hi = max(0, 8 * (k + 1) - 7)
        rows.append([k, f"{lo} .. {hi - 1}", hi - lo, f"{lo / 24:.4f}", f"{hi / 24:.4f}"])
    return ("For the 25-frame clip at 24 frames per second, latent frame k covers the pixel frames lo .. hi - 1 with lo = max(0, 8k - 7) and "
            "hi = max(0, 8(k + 1) - 7), that is the half-open time window from lo / 24 to hi / 24 seconds.\n\n"
            + _table(["k", "pixel frames", "how many", "from (s)", "to (s)"], rows, "lrrrr"))


def saving_table():
    rows = []
    for name, f, h, w in (("768 x 512, 121 frames", 121, 512, 768), ("1024 x 512, 1441 frames (60.04 seconds)", 1441, 512, 1024)):
        pix = 3 * f * h * w
        planar = f * h * w * 3 // 2
        lf = latent_frames(f)
        lat = LATENT_C * lf * (h // 32) * (w // 32)
        pos = lf * (h // 32) * (w // 32)
        rows.append([name, f"{pix:,}", f"{planar:,}", f"{lat:,}", f"{pos:,}", f"{pos * pos:,}", f"{pix / lat:.1f}", f"{planar / lat:.1f}"])
    return ("Values of the pixels (3 per pixel), of the 4:2:0 planes (1.5 per pixel) and of the latent (128 per position), and the number of latent "
            "positions (latent frames * height / 32 * width / 32, where latent frames = 1 + (F - 1) / 8 for F video frames), and the square of that number, the pairs of positions an attention layer would score.\n\n"
            + _table(["clip", "RGB values", "4:2:0 values", "latent values", "latent positions", "pairs of positions", "RGB / latent", "4:2:0 / latent"], rows, "lrrrrrrr"))


def patch_order():
    x = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
    p = 2
    groups = []
    for iw in range(p):
        for ih in range(p):
            groups.append([x[ho * p + ih][wo * p + iw] for ho in range(2) for wo in range(2)])
    return ("The 4 by 4 frame of the numbers 0 .. 15, row by row, cut into 2 by 2 patches (patch size 2 for the example). Channel number "
            "c = iw * 2 + ih holds the pixel at offset (ih, iw) of every patch, with the patches in row-major order:\n\n"
            + _table(["channel", "offset (ih, iw)", "values of the 4 patches"],
                     [[iw * 2 + ih, f"({ih}, {iw})", str(groups[iw * 2 + ih])] for iw in range(p) for ih in range(p)], "lll"))


WHOLE_DECODE_MAX_PIXELS = 24_000_000


def tile_policy():
    rows = []
    for f, h, w in ((9, 1088, 1920), (25, 704, 1280), (25, 1088, 1920), (49, 704, 1280), (121, 512, 768)):
        vol = f * h * w
        rows.append([f"{f} x {h} x {w}", f"{vol:,}", f"{vol / 1e6:.1f}", "tiled" if vol > WHOLE_DECODE_MAX_PIXELS else "whole"])
    return (f"The output pixel volume of a decode is frames x height x width. A decode goes through the whole clip in one graph when the volume "
            f"is at most {WHOLE_DECODE_MAX_PIXELS:,} and through overlapping tiles above it.\n\n"
            + _table(["frames x height x width", "pixels", "megapixels", "decode"], rows, "lrrl"))
