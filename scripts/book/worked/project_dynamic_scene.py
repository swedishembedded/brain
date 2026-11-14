# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the Volume X project "Reconstructing and Predicting a Scene".

Plain Python 3, standard library only, written from the project's own description and independent of any engine code. Each public
function without arguments returns a markdown string that the chapter embeds between `<!-- worked: project_dynamic_scene.<name> -->`
markers. What the project reuses from the chapters of the volume is imported, never copied: the lens, the look-at camera and the
rotation conventions of Chapter 3 (`camera_geometry`), the eight-point algorithm with RANSAC, the triangulation, the DLT absolute pose
and the bundle adjustment of Chapter 4 (`structure_from_motion`), the Gaussian primitive, the differentiable renderer and its
analytic gradient of Chapter 6 (`gaussian_splatting`), and the least-squares fit of a linear world model of Chapter 7
(`world_models`), together with the book's generator and the stable-digits rule.

The world is a closed loop on a synthetic scene:

* a camera of f = 500 pixels (640 by 480) slides along an arc past a wall and a table, filmed at 18 times; three balls move on the
  table (a spring, a glider that slows down, and a slider that stops at the table's edge); the frames are renders of a true scene of
  Gaussians at 32 by 24 pixels (the lens scaled by 1/20), and the feature tracks are the exact projections plus pixel noise;
* structure from motion on the first 12 frames: an essential matrix by RANSAC from frames 0 and 4, triangulation, registration of the
  other frames by RANSAC and the DLT, a bundle adjustment with the baseline of frames 0 and 4 as the known scale, a horizontal floor
  plane by consensus among the points, and the three tracks the rigid model rejects, which are the balls;
* a small set of Gaussians (ten for the wall and the floor, one for each ball) fitted to the 12 frames through the recovered
  cameras by Adam on the pixel error, the balls' positions on the floor refined with the Gaussians;
* a second-order linear model of each ball, fitted by least squares to its reconstructed path and run free for six frames, with and
  without the clamp that keeps a ball on the table, and the held-out frames rendered from it and scored;
* three planted mistakes (a half pixel in the lens, a transposed rotation, no clamp), each measured by a number, and the cost of the
  loop in operations and bytes.

Trained numbers are printed with the digits that agree between the run and four copies whose first weight is changed by the nudges of
`stable.py`. The fifteen fits (three kinds of five) are trained in four worker processes.

Swedish Embedded AB implements solutions for reconstructing a scene from ordinary footage and predicting how it moves. If your team
needs expertise in structure from motion, Gaussian splatting and learned world models, you can procure our services by sending an
email to info@swedishembedded.com.
"""
import math
import multiprocessing
from functools import lru_cache

import bookrng
import camera_geometry as cg
import decompositions as dc
import gaussian_splatting as gs
import images as im
import stable
import structure_from_motion as sfm
import world_models as wmod

_g = dc._g
_table = im._table
mv, tr, norm, unit, sub = cg.mv, cg.tr, cg.norm, cg.unit, cg.sub

# ------------------------------------------------------------------------------------------------ the world

F, CX, CY = sfm.F, sfm.CX, sfm.CY            # 500, 320, 240: the lens of Chapter 4
WIDTH, HEIGHT = sfm.WIDTH, sfm.HEIGHT        # 640 by 480
SCALE = 20                                   # the splat images are the lens divided by 20
W, H = WIDTH // SCALE, HEIGHT // SCALE       # 32 by 24
FS = F / SCALE                               # 25
CXS, CYS = CX / SCALE, CY / SCALE            # 16, 12 (corner rule: c' = s c)
FRAMES, OBSERVED = 18, 12
HELD = FRAMES - OBSERVED                     # 6
TARGET, UP = [0.0, 0.0, 5.2], [0.0, -1.0, 0.0]
SKY = (0.55, 0.65, 0.80)
FLOOR_Y = 1.5
BALL_R = 0.3
BALL_Y = FLOOR_Y - BALL_R
TABLE = (-1.2, 1.2, 4.0, 6.4)                # x min, x max, z min, z max of the table
BASE_PAIR = (0, 4)
IMG_SIGMA = 0.01
TRACK_SIGMA = 0.25                           # pixels per coordinate

LAND_NAME = "project_dynamic_scene.landmarks"
NOISE_NAME = "project_dynamic_scene.track_noise"
TRUTH_NAME = "project_dynamic_scene.true_scene"
IMG_NOISE_NAME = "project_dynamic_scene.image_noise"
RANSAC_NAME = "project_dynamic_scene.ransac"
PNP_NAME = "project_dynamic_scene.registration"


def _vec(x, d=3):
    return "(" + ", ".join(_g(c, d) for c in x) + ")"


def eye(i):
    """The camera centre at frame i (metres, in the frame of camera 0)."""
    return [0.14 * i, 0.02 * i, 0.04 * i]


@lru_cache(maxsize=None)
def cameras():
    """World-to-camera (R, t) of the 18 cameras: all look at TARGET with up = (0, -1, 0); camera 0 is the identity."""
    return [cg.look_at(eye(i), TARGET, UP) for i in range(FRAMES)]


def splat_camera(R, t, mistaken=False):
    """The 32 by 24 camera of the splat renderer of Chapter 6 for the pose (R, t). The lens is the lens of 640 by 480 divided by 20
    with the corner rule c' = s c; the mistaken one applies the centre rule c' = s (c + 1/2) - 1/2 to a corner-convention lens."""
    if mistaken:
        cx, cy = (CX + 0.5) / SCALE - 0.5, (CY + 0.5) / SCALE - 0.5
    else:
        cx, cy = CXS, CYS
    return {"R": R, "t": t, "fx": FS, "fy": FS, "cx": cx, "cy": cy, "W": W, "H": H}


# --- the balls

def _spring(x0, v0, k, d, c, n):
    """Positions of a damped spring with the update v' = d (v - k (x - c)), x' = x + v' (symplectic Euler) at times 0 .. n-1."""
    out, x, v = [x0], x0, v0
    for _ in range(n - 1):
        v = d * (v - k * (x - c))
        x = x + v
        out.append(x)
    return out


def _glide(x0, v0, d, n):
    out, x, v = [x0], x0, v0
    for _ in range(n - 1):
        v = d * v
        x = x + v
        out.append(x)
    return out


SLIDER_X0, SLIDER_V, SLIDER_STOP = -0.9, 0.15, TABLE[1] - BALL_R


def ball_paths():
    """True floor positions (x, z) of the three balls at the 18 frames: A a spring in both axes, B a glider that slows by 0.9 a frame,
    C a slider at constant speed that is stopped by the table's edge (its centre cannot pass x = 0.9)."""
    ax = _spring(-0.7, 0.12, 0.12, 0.995, 0.0, FRAMES)
    az = _spring(4.6, 0.05, 0.08, 0.995, 5.2, FRAMES)
    bx = _glide(-0.9, 0.10, 0.9, FRAMES)
    bz = _glide(4.5, 0.11, 0.9, FRAMES)
    cx = [min(SLIDER_X0 + SLIDER_V * t, SLIDER_STOP) for t in range(FRAMES)]
    cz = [6.0] * FRAMES
    return [list(zip(ax, az)), list(zip(bx, bz)), list(zip(cx, cz))]


BALL_COLOURS = [(0.90, 0.20, 0.20), (0.20, 0.80, 0.30), (0.90, 0.80, 0.20)]
BALL_NAMES = ["A (spring)", "B (glider)", "C (slider)"]


def true_static():
    """The true wall and floor: 8 wall Gaussians (a 4 by 2 grid at z = 8) and 6 floor Gaussians (a 3 by 2 grid at y = 1.5), colours
    from `project_dynamic_scene.true_scene`: three `uniform()` draws a Gaussian in the order wall (column-major over y, x) then floor,
    wall colour 0.15 + 0.7 u, floor colour 0.3 + 0.4 u; scales (0.9, 0.8, 0.1) on the wall and (0.5, 0.04, 0.5) on the floor."""
    g = bookrng.generator(TRUTH_NAME)
    out = []
    for iy, y in enumerate((-0.8, 0.6)):
        for ix, x in enumerate((-1.4, -0.1, 1.2, 2.5)):
            col = [0.15 + 0.7 * g.uniform() for _ in range(3)]
            out.append(gs.gaussian([x, y, 8.0], [0.9, 0.8, 0.1], [1.0, 0.0, 0.0, 0.0], 0.95, col))
    for iz, z in enumerate((4.8, 5.8)):
        for ix, x in enumerate((-0.8, 0.0, 0.8)):
            col = [0.3 + 0.4 * g.uniform() for _ in range(3)]
            out.append(gs.gaussian([x, FLOOR_Y, z], [0.5, 0.04, 0.5], [1.0, 0.0, 0.0, 0.0], 0.95, col))
    return out


def ball_gaussian(x, z, colour, scale=BALL_R, opacity=0.95):
    return gs.gaussian([x, BALL_Y, z], [scale] * 3, [1.0, 0.0, 0.0, 0.0], opacity, colour)


def true_scene_at(t):
    """All 17 Gaussians of the true scene at frame t."""
    paths = ball_paths()
    return true_static() + [ball_gaussian(paths[j][t][0], paths[j][t][1], BALL_COLOURS[j]) for j in range(3)]


@lru_cache(maxsize=None)
def true_frames():
    """The 18 true frames: renders of the true scene through the true cameras at 32 by 24, sky background; the 12 observed ones get
    noise IMG_SIGMA per channel from `project_dynamic_scene.image_noise` (rows, columns, channels in order, one `normal()` each).
    Returns (clean, observed), observed having 12 entries."""
    cams = cameras()
    clean = [gs.render(true_scene_at(t), splat_camera(*cams[t]), SKY) for t in range(FRAMES)]
    n = bookrng.generator(IMG_NOISE_NAME)
    obs = []
    for t in range(OBSERVED):
        obs.append([[[v + IMG_SIGMA * n.normal() for v in px] for px in row] for row in clean[t]])
    return clean, obs


# --- the feature tracks

def landmarks():
    """40 static points: 20 on the wall (x = -1.8 + 4.2 u, y = -1.6 + 2.6 u, z = 7.8 + 0.4 u, three draws each), the 4 corners of the
    table and 16 on its top (x = -1.2 + 2.4 u, z = 4.0 + 2.4 u, y = 1.5), all from `project_dynamic_scene.landmarks`."""
    g = bookrng.generator(LAND_NAME)
    pts = []
    for _ in range(20):
        x = -1.8 + 4.2 * g.uniform()
        y = -1.6 + 2.6 * g.uniform()
        z = 7.8 + 0.4 * g.uniform()
        pts.append([x, y, z])
    for x in (TABLE[0], TABLE[1]):
        for z in (TABLE[2], TABLE[3]):
            pts.append([x, FLOOR_Y, z])
    for _ in range(16):
        x = TABLE[0] + (TABLE[1] - TABLE[0]) * g.uniform()
        z = TABLE[2] + (TABLE[3] - TABLE[2]) * g.uniform()
        pts.append([x, FLOOR_Y, z])
    return pts


NLAND = 40
NTRACK = NLAND + 3


def _project(R, t, X):
    c = [a + b for a, b in zip(mv(R, X), t)]
    return (CX + F * c[0] / c[2], CY + F * c[1] / c[2])


@lru_cache(maxsize=None)
def tracks():
    """pix[v][j] for the 12 observed views and 43 tracks (40 landmarks then the 3 ball centres), pixel noise TRACK_SIGMA from
    `project_dynamic_scene.track_noise`: view by view, track by track, `normal()` for u then `normal()` for v."""
    cams = cameras()
    pts = landmarks()
    paths = ball_paths()
    n = bookrng.generator(NOISE_NAME)
    pix = []
    for v in range(OBSERVED):
        R, t = cams[v]
        row = []
        for j in range(NTRACK):
            if j < NLAND:
                X = pts[j]
            else:
                X = [paths[j - NLAND][v][0], BALL_Y, paths[j - NLAND][v][1]]
            u, w = _project(R, t, X)
            row.append((u + TRACK_SIGMA * n.normal(), w + TRACK_SIGMA * n.normal()))
        pix.append(row)
    return pix


# ------------------------------------------------------------------------------------------------ structure from motion

IDENT = sfm.IDENT
PAIR_THRESH_PX = 1.5
PNP_THRESH_PX = 4.0
GATE_PX = 2.0                    # a track whose median residual after the adjustment exceeds this is not rigid
BASELINE = norm(sub(eye(BASE_PAIR[1]), eye(BASE_PAIR[0])))      # the one distance the world supplies, metres


def _norm_obs(pix, v, j):
    return sfm.normalized(pix[v][j])


def reconstruct(pose_convention="correct"):
    """Chapter 4's pipeline on the 12 observed views of the 43 tracks. Returns a dict with the poses [(R, t)] in the gauge of camera 0 and
    the baseline of frames 0 and 4 in metres, the points of the rigid tracks, the track flags, and counts for the ledger."""
    pix = tracks()
    a = [_norm_obs(pix, BASE_PAIR[0], j) for j in range(NTRACK)]
    b = [_norm_obs(pix, BASE_PAIR[1], j) for j in range(NTRACK)]
    two = sfm.ransac_essential(a, b, PAIR_THRESH_PX, 500, bookrng.generator(RANSAC_NAME))
    pair_in = [j for j in range(NTRACK) if two["mask"][j]]
    P0 = [(IDENT, [0.0, 0.0, 0.0]), (two["R"], unit(two["t"]))]
    X0 = {}
    for j in pair_in:
        X = sfm.triangulate(P0, [a[j], b[j]])
        if sfm.depth_in(*P0[0], X) > 0 and sfm.depth_in(*P0[1], X) > 0:
            X0[j] = X
    ids = sorted(X0)
    poses = {BASE_PAIR[0]: (IDENT, [0.0, 0.0, 0.0]), BASE_PAIR[1]: (two["R"], unit(two["t"]))}
    reg = {}
    for v in range(OBSERVED):
        if v in poses:
            continue
        Xs = [X0[j] for j in ids]
        xs = [_norm_obs(pix, v, j) for j in ids]
        pxs = [pix[v][j] for j in ids]
        R, t, iters, trace = sfm.ransac_pnp(Xs, xs, pxs, PNP_THRESH_PX, 2000, bookrng.generator(RANSAC_NAME + ".pnp." + str(v)))
        mask = [sfm.reproj_px(R, t, X, p) < PNP_THRESH_PX for X, p in zip(Xs, pxs)]
        inl = [k for k, m in enumerate(mask) if m]
        obs = [(0, n, pxs[k][0], pxs[k][1]) for n, k in enumerate(inl)]
        Pr = sfm.Problem([[R, t]], [Xs[k] for k in inl], obs, ["free"], free_points=False, delta=2.0)
        Pr.solve(30)
        poses[v] = (Pr.poses[0][0], Pr.poses[0][1])
        reg[v] = (iters, len(inl), len(ids))
    return dict(two=two, pair_in=pair_in, ids=ids, X0=X0, poses0=poses, reg=reg)


def adjust(rec):
    """The global adjustment. Points of the pair's inlier tracks are triangulated linearly from the observations of all 12 views whose
    reprojection under the registered poses is within 8 pixels (a looser gate than the registration's, since the structure is still
    coarse); frame 0 is held, frame 4 is the scale camera (its distance from frame 0 stays at the unit baseline), the other ten are
    free, with Huber delta 2 pixels; then the rigid test: a track whose median residual exceeds GATE_PX is dropped and the problem
    solved again. Returns the adjusted problem and the counts."""
    pix = tracks()
    poses = [rec["poses0"][v] for v in range(OBSERVED)]
    pts, obs, kept = [], [], []
    for j in rec["ids"]:
        lst = []
        for v in range(OBSERVED):
            if sfm.reproj_px(poses[v][0], poses[v][1], rec["X0"][j], pix[v][j]) < 8.0:
                lst.append(v)
        if len(lst) < 2:
            continue
        X = sfm.triangulate([poses[v] for v in lst], [_norm_obs(pix, v, j) for v in lst])
        lst = [v for v in lst if sfm.depth_in(poses[v][0], poses[v][1], X) > 0]
        if len(lst) < 2:
            continue
        for v in lst:
            obs.append((v, len(pts), pix[v][j][0], pix[v][j][1]))
        pts.append(X)
        kept.append(j)
    slots = ["free"] * OBSERVED
    slots[BASE_PAIR[0]] = "held"
    slots[BASE_PAIR[1]] = "scale"
    P = sfm.Problem([list(p) for p in poses], pts, obs, slots, anchor=BASE_PAIR[0], scale=BASE_PAIR[1], f=F, free_f=False, delta=2.0)
    hist1 = P.solve(60)
    rs = P.residuals()
    med = {}
    for (v, k, u, w), r in zip(P.obs, rs):
        med.setdefault(k, []).append(math.hypot(r[0], r[1]))
    rigid = [k for k in range(len(P.points)) if sorted(med[k])[len(med[k]) // 2] <= GATE_PX]
    new = {k: n for n, k in enumerate(rigid)}
    obs2 = [(v, new[k], u, w) for (v, k, u, w) in P.obs if k in new]
    Q = sfm.Problem(P.poses, [P.points[k] for k in rigid], obs2, P.slots, P.anchor, P.scale, F, False, True, 2.0)
    hist2 = Q.solve(60)
    return dict(P=Q, kept=[kept[k] for k in rigid], dropped=[kept[k] for k in range(len(kept)) if k not in new], hist1=hist1, hist2=hist2,
                nobs=len(obs2), rms0=P.rms())


# ------------------------------------------------------------------------------------------------ what the reconstruction gives

PLANE_TOL = 0.08                 # metres: a point is on the horizontal plane of height h when |y - h| is below this


def floor_plane(points):
    """The horizontal plane (y = h, camera 0 being level) with the most points within PLANE_TOL, over the candidates h = y of each
    point in turn (the first with the largest count wins); h is the mean y of its points. Returns (h, indices of the points on it)."""
    best = None
    for p in points:
        on = [k for k, q in enumerate(points) if abs(q[1] - p[1]) < PLANE_TOL]
        if best is None or len(on) > len(best):
            best = on
    return sum(points[k][1] for k in best) / len(best), best


def ray_on_plane(R, t, px, y_plane):
    """The point where the ray through pixel px of the camera (R, t) meets the horizontal plane y = y_plane: C + s d with C = -R^T t,
    d = R^T (K^-1 (u, v, 1)) and s = (y_plane - C_y) / d_y."""
    C = [-x for x in mv(tr(R), t)]
    d = mv(tr(R), sfm.normalized(px))
    s = (y_plane - C[1]) / d[1]
    return [C[k] + s * d[k] for k in range(3)]


def scaled(adj):
    """The adjusted cameras and points in metres: the unit baseline of frames 0 and 4 becomes BASELINE, so every translation and
    every point is multiplied by it (a similarity, which moves no reprojection)."""
    P = adj["P"]
    cams = [(R, [BASELINE * x for x in t]) for R, t in P.poses]
    pts = [[BASELINE * x for x in X] for X in P.points]
    return cams, pts


def structure(adj, transposed=False):
    """Everything the later stages take from structure from motion: the cameras and points in metres, the floor height h and the
    points on it, the table rectangle (the bounding box of those points in x and z), the clamp box (the rectangle inset by the ball's
    radius), the floor-plane height of a ball centre h - BALL_R, and each ball track's detection per frame: the ray through its pixel
    meets that plane. With transposed=True the cameras are exported with R^T where R belongs (the planted rotation-convention mistake)."""
    cams, pts = scaled(adj)
    if transposed:
        cams = [(tr(R), t) for R, t in cams]
    h, on = floor_plane(pts)
    xs = [pts[k][0] for k in on]
    zs = [pts[k][2] for k in on]
    table = (min(xs), max(xs), min(zs), max(zs))
    box = (table[0] + BALL_R, table[1] - BALL_R, table[2] + BALL_R, table[3] - BALL_R)
    pix = tracks()
    y_ball = h - BALL_R
    det = []
    for j in range(3):
        row = []
        for v in range(OBSERVED):
            X = ray_on_plane(cams[v][0], cams[v][1], pix[v][NLAND + j], y_ball)
            row.append((X[0], X[2]))
        det.append(row)
    return dict(cams=cams, pts=pts, floor_y=h, floor_ids=on, table=table, box=box, y_ball=y_ball, det=det)


# ------------------------------------------------------------------------------------------------ fitting the Gaussians

N_WALL, N_FLOOR = 6, 4
N_STATIC = N_WALL + N_FLOOR
FIT_STEPS = 120
# Adam rates: twice those of Chapter 6 (gaussian_splatting.FIT_LR: mean 0.02, log scale 0.03, quaternion 0.01, opacity logit 0.1,
# colour 0.03), since 120 updates stand where that chapter used 300; a ball's log scale (one number for its three axes), opacity logit
# and colour get 0.06, 0.2 and 0.06
STATIC_LR = [2.0 * r for r in gs.FIT_LR]
BALL_LR = [0.06, 0.2, 0.06, 0.06, 0.06]


def _chunks(order, n):
    return [order[k * len(order) // n:(k + 1) * len(order) // n] for k in range(n)]


def _mean_std(vals):
    m = sum(vals) / len(vals)
    return m, math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


def _pixel_colour(img, cam, X):
    """The colour of the pixel of the image that the point X projects to (the pixel containing it), or grey outside the image."""
    c = [a + b for a, b in zip(mv(cam["R"], X), cam["t"])]
    if c[2] <= 0.01:
        return [0.5, 0.5, 0.5]
    u, v = cam["fx"] * c[0] / c[2] + cam["cx"], cam["fy"] * c[1] / c[2] + cam["cy"]
    i, j = int(math.floor(u)), int(math.floor(v))
    if 0 <= i < W and 0 <= j < H:
        return [min(0.95, max(0.05, x)) for x in img[j][i]]
    return [0.5, 0.5, 0.5]


def init_scene(st, cams0, img0):
    """The starting Gaussians from the sparse structure. The points off the floor plane are the wall, sorted by x (ties by index) and cut
    into N_WALL chunks of equal count (boundaries k n / 6 rounded down); each chunk gives one Gaussian at the chunk mean with scales
    (max(1.2 std x, 0.3), max(1.2 std y, 0.3), 0.15). The floor points are cut by x into N_FLOOR chunks, each a Gaussian at the chunk mean
    with scales (max(1.2 std x, 0.3), 0.05, max(1.2 std z, 0.4)). Rotation the identity quaternion, opacity 0.8, and the colour of the
    pixel of frame 0 that the centre projects to. A ball starts with one log scale ln 0.25 (all three axes), opacity 0.8 and the colour
    of the pixel of frame 0 at its first detection."""
    pts = st["pts"]
    fl = set(st["floor_ids"])
    wall = sorted((k for k in range(len(pts)) if k not in fl), key=lambda k: (pts[k][0], k))
    floor = sorted(st["floor_ids"], key=lambda k: (pts[k][0], k))
    static = []
    for ch in _chunks(wall, N_WALL):
        mx, sx = _mean_std([pts[k][0] for k in ch])
        my, sy = _mean_std([pts[k][1] for k in ch])
        mz, _ = _mean_std([pts[k][2] for k in ch])
        mu = [mx, my, mz]
        static.append(gs.gaussian(mu, [max(1.2 * sx, 0.3), max(1.2 * sy, 0.3), 0.15], [1.0, 0.0, 0.0, 0.0], 0.8, _pixel_colour(img0, cams0[0], mu)))
    for ch in _chunks(floor, N_FLOOR):
        mx, sx = _mean_std([pts[k][0] for k in ch])
        mz, sz = _mean_std([pts[k][2] for k in ch])
        mu = [mx, st["floor_y"], mz]
        static.append(gs.gaussian(mu, [max(1.2 * sx, 0.3), 0.05, max(1.2 * sz, 0.4)], [1.0, 0.0, 0.0, 0.0], 0.8, _pixel_colour(img0, cams0[0], mu)))
    app = []
    for j in range(3):
        x0, z0 = st["det"][j][0]
        col = _pixel_colour(img0, cams0[0], [x0, st["y_ball"], z0])
        app.append([math.log(0.25), gs.logit(0.8)] + col)
    return static, app


def frame_params(static, app, pos, t, y_ball):
    """The Gaussians of frame t: the static ones, then each ball at its floor position (x, z) of that frame, height y_ball, one log scale
    on all three axes, the identity quaternion."""
    out = [list(p) for p in static]
    for j in range(3):
        ls, lo = app[j][0], app[j][1]
        out.append([pos[j][t][0], y_ball, pos[j][t][1], ls, ls, ls, 1.0, 0.0, 0.0, 0.0, lo] + list(app[j][2:5]))
    return out


def _adam(p, m, v, g, lr, step):
    for k in range(len(p)):
        m[k] = 0.9 * m[k] + 0.1 * g[k]
        v[k] = 0.999 * v[k] + 0.001 * g[k] * g[k]
        p[k] -= lr[k] * (m[k] / (1 - 0.9 ** step)) / (math.sqrt(v[k] / (1 - 0.999 ** step)) + 1e-8)


def fit_scene(cams, imgs, static, app, pos, y_ball, steps=FIT_STEPS, nudge=0.0, curve_at=()):
    """Full-batch Adam (beta 0.9 and 0.999, epsilon 1e-8, bias correction) on the 10 x 14 numbers of the static Gaussians and the 3 x 5
    appearance numbers of the balls, whose floor positions `pos` (3 balls by 12 frames) stay at the detections. The loss of an update is
    the mean over the 12 frames of the mean squared error of the frame over pixels and channels (Chapter 6). `nudge` is added to the
    first mean number of the first static Gaussian. Returns the fitted (static, app) and the loss at the update numbers of curve_at."""
    static = [list(p) for p in static]
    app = [list(a) for a in app]
    static[0][0] += nudge
    ms = [[0.0] * 14 for _ in static]
    vs = [[0.0] * 14 for _ in static]
    ma = [[0.0] * 5 for _ in app]
    va = [[0.0] * 5 for _ in app]
    n = len(imgs)
    seen = {}
    for step in range(steps + 1):
        Gs = [[0.0] * 14 for _ in static]
        Ga = [[0.0] * 5 for _ in app]
        loss = 0.0
        for t in range(n):
            l, gr = gs.view_gradient(frame_params(static, app, pos, t, y_ball), cams[t], gs._mse_upstream(imgs[t]), SKY)
            loss += l / n
            for i in range(N_STATIC):
                for k in range(14):
                    Gs[i][k] += gr[i][k] / n
            for j in range(3):
                g = gr[N_STATIC + j]
                Ga[j][0] += (g[3] + g[4] + g[5]) / n
                Ga[j][1] += g[10] / n
                for k in range(3):
                    Ga[j][2 + k] += g[11 + k] / n
        if step in curve_at:
            seen[step] = loss
        if step == steps:
            break
        for i in range(N_STATIC):
            _adam(static[i], ms[i], vs[i], Gs[i], STATIC_LR, step + 1)
        for j in range(3):
            _adam(app[j], ma[j], va[j], Ga[j], BALL_LR, step + 1)
    return static, app, seen


# ------------------------------------------------------------------------------------------------ learning the dynamics

RIDGE = 1e-4


def fit_ar2(series):
    """The second-order linear model of one coordinate of one ball, p' = c + a p + b q with p the current and q the previous value:
    ridge regression on the transitions (q, p) -> p' over the series (len - 2 equations), by the normal equations of the features
    (p, q, 1) with RIDGE added to the diagonal entries of the weights a and b (not of c), solved by Gauss-Jordan elimination with
    partial pivoting (Chapter 7, `world_models._solve`); each sum runs over the transitions in order. A ball at constant speed makes
    the three features linearly dependent (p = q + v), and the ridge is what picks one of the equivalent solutions. Returns (a, b, c)."""
    rows = [(series[t], series[t - 1], 1.0) for t in range(1, len(series) - 1)]
    ys = [series[t + 1] for t in range(1, len(series) - 1)]
    A = [[sum(r[i] * r[j] for r in rows) + (RIDGE if i == j and i < 2 else 0.0) for j in range(3)] for i in range(3)]
    y = [sum(r[i] * v for r, v in zip(rows, ys)) for i in range(3)]
    a, b, c = wmod._solve(A, y)
    return a, b, c


def fit_dynamics(pos):
    """One (a, b, c) for each of the 3 balls and each of the 2 floor coordinates, from the first OBSERVED positions."""
    return [[fit_ar2([pos[j][t][k] for t in range(OBSERVED)]) for k in range(2)] for j in range(3)]


def clamp_to(box, x, z):
    """Clip a floor position to the clamp box (x min, x max, z min, z max)."""
    return (min(max(x, box[0]), box[1]), min(max(z, box[2]), box[3]))


def rollout(model, pos, steps, box=None):
    """The free run of the model from the last two observed positions of every ball, `steps` frames long: each frame is
    p' = c + a p + b q on each coordinate, fed with the model's own output, and when `box` is given the position is clipped to it
    before it is used again (and returned)."""
    out = []
    for j in range(3):
        q, p = pos[j][OBSERVED - 2], pos[j][OBSERVED - 1]
        path = []
        for _ in range(steps):
            n = tuple(model[j][k][2] + model[j][k][0] * p[k] + model[j][k][1] * q[k] for k in range(2))
            if box is not None:
                n = clamp_to(box, *n)
            path.append(n)
            q, p = p, n
        out.append(path)
    return out


# ------------------------------------------------------------------------------------------------ predicting and scoring

CURVE_AT = (0, 10, 30, 60, FIT_STEPS)
KINDS = ("clean", "half", "transposed")


def frame_mse(a, b):
    return gs._mse(a, b)


@lru_cache(maxsize=None)
def base():
    """The deterministic part of the pipeline: structure from motion and what it gives (cached; the fork of the workers shares it)."""
    rec = reconstruct()
    adj = adjust(rec)
    return rec, adj, structure(adj), structure(adj, transposed=True)


def export_cameras(kind, cams):
    """The renderer's cameras for the poses `cams` under the way of exporting them that `kind` plants: the half-pixel mistake changes the
    lens, the transposed one the rotation; the clean run is the Chapter 3 pair as it is."""
    if kind == "transposed":
        return [splat_camera(tr(R), t) for R, t in cams]
    return [splat_camera(R, t, mistaken=(kind == "half")) for R, t in cams]


def structure_of(kind):
    rec, adj, st, st_t = base()
    return st_t if kind == "transposed" else st


def predictions(st):
    """The four predictions of the held-out ball positions: the clamped model, the unclamped model, the last detection held, and the
    true positions (the reference that has no prediction error): each a list over the 3 balls of HELD (x, z) pairs."""
    model = fit_dynamics(st["det"])
    paths = ball_paths()
    return dict(model=rollout(model, st["det"], HELD, st["box"]), free=rollout(model, st["det"], HELD),
                persist=[[st["det"][j][OBSERVED - 1]] * HELD for j in range(3)],
                truth=[paths[j][OBSERVED:] for j in range(3)])


def run(spec):
    """One fit and its scores: kind in KINDS, nudge added to the first static mean number. The Gaussians are initialised from the
    structure of that kind, fitted to the 12 observed frames through its cameras, and the held-out frames are rendered at the true
    cameras of frames 12 .. 17 (exported the same way) with the balls where each prediction puts them and scored against the clean true frames."""
    kind, nudge = spec
    st = structure_of(kind)
    clean, obs = true_frames()
    cams = export_cameras(kind, st["cams"])
    held_cams = export_cameras(kind, cameras()[OBSERVED:])
    static, app = init_scene(st, cams, obs[0])
    static, app, curve = fit_scene(cams, obs, static, app, st["det"], st["y_ball"], FIT_STEPS, nudge, CURVE_AT)
    pred = predictions(st)
    mse = {}
    for name, pos in pred.items():
        mse[name] = [frame_mse(gs.render(frame_params(static, app, _shift(pos, OBSERVED), OBSERVED + k, st["y_ball"]),
                                         held_cams[k], SKY), clean[OBSERVED + k]) for k in range(HELD)]
    mse["copy"] = [frame_mse(obs[OBSERVED - 1], clean[OBSERVED + k]) for k in range(HELD)]
    pairs = sum(render_pairs(frame_params(static, app, st["det"], t, st["y_ball"]), cams[t]) for t in range(OBSERVED))
    return dict(curve=curve, mse=mse, static=static, app=app, pairs=pairs)


def render_pairs(params, cam):
    """The (pixel, record) pairs one render of the frame visits: for every pixel the records looked at in the front-to-back walk, the
    skipped ones included (the `visits` of `gaussian_splatting.walk`)."""
    recs, _ = gs.sorted_records(params, cam)
    return sum(gs.walk(recs, x + 0.5, y + 0.5, SKY)[3] for y in range(cam["H"]) for x in range(cam["W"]))


def _shift(pos, offset):
    """A held-out prediction indexed by absolute frame number: positions of frame offset + k at index offset + k."""
    return [[None] * offset + list(p) for p in pos]


# ------------------------------------------------------------------------------------------------ the fits, in four workers

@lru_cache(maxsize=None)
def runs():
    """Fifteen fits: for each kind in KINDS the nominal run and the four copies nudged by `stable.NUDGES`, trained in four worker
    processes. Returns {kind: [run, ...]} with the nominal run first."""
    base()
    true_frames()
    specs = [(k, n) for k in KINDS for n in (0.0, *stable.NUDGES)]
    with multiprocessing.get_context("fork").Pool(4) as pool:
        res = pool.map(run, specs, chunksize=1)
    out = {k: [] for k in KINDS}
    for (k, _), r in zip(specs, res):
        out[k].append(r)
    return out


def cell(kind, get, fmt):
    rs = runs()[kind]
    vals = [get(r) for r in rs]
    return stable.text(vals[0], vals[1:], fmt)


def e3(x):
    return f"{x:.3e}"


def e2(x):
    return f"{x:.2e}"


def mean_mse(name):
    return lambda r: sum(r["mse"][name]) / HELD


def pos_errors(kind="clean"):
    """The distance (metres) of each prediction from the true position of each ball at each held-out frame, for each of the four position
    predictions: {name: [[error at frame 12 .. 17] for each ball]}."""
    st = structure_of(kind)
    pred = predictions(st)
    paths = ball_paths()
    return {name: [[norm(sub(pred[name][j][k], paths[j][OBSERVED + k])) for k in range(HELD)] for j in range(3)] for name in pred}


# ------------------------------------------------------------------------------------------------ the world (milestone 1)

def _rot_deg(R):
    return sfm.rot_angle(R)


def world_setup():
    cams = cameras()
    paths = ball_paths()
    clean, obs = true_frames()
    pix = tracks()
    ts = [0, 4, 11, 12, 17]
    rows = []
    for t in ts:
        R, tt = cams[t]
        rows.append([t, _vec(eye(t), 3), _g(_rot_deg(R), 4), "observed" if t < OBSERVED else "held out"])
    prow = []
    for t in (0, 6, 11, 12, 17):
        prow.append([t] + [f"({_g(paths[j][t][0], 4)}, {_g(paths[j][t][1], 4)})" for j in range(3)])
    sg = true_static()
    srows = [[k, "wall" if k < 8 else "floor", _vec(sg[k][0:3], 2), _vec([math.exp(x) for x in sg[k][3:6]], 2), _vec(sg[k][11:14], 4)]
             for k in (0, 1, 7, 8, 13)]
    s0 = sum(v for row in clean[0] for px in row for v in px)
    s11 = sum(v for row in clean[11] for px in row for v in px)
    n0 = sum(v for row in obs[0] for px in row for v in px)
    return "\n".join([
        f"Lens: f = {F:g} pixels, principal point ({CX:g}, {CY:g}), image {WIDTH} by {HEIGHT} pixels, pixel i covering [i, i + 1) (the corner "
        f"convention of Chapter 3), no distortion. The splat images are the same lens divided by {SCALE} with the corner rule c' = s c: "
        f"{W} by {H} pixels, f = {FS:g}, principal point ({CXS:g}, {CYS:g}). Camera i, i = 0 .. {FRAMES - 1}, has its centre at "
        f"(0.14 i, 0.02 i, 0.04 i) metres, looks at {_vec(TARGET, 2)} with up (0, -1, 0) (the `look_at` of Chapter 3), so camera 0 is the "
        f"identity and the frame of camera 0 is the world (x right, y down, z forward). The first {OBSERVED} frames are observed, the last {HELD} are held out.", "",
        _table(["frame", "centre C (metres)", "angle of R (degrees)", "role"], rows, "rrrl"), "",
        f"The table is the rectangle x in [{TABLE[0]:g}, {TABLE[1]:g}], z in [{TABLE[2]:g}, {TABLE[3]:g}] at height y = {FLOOR_Y:g}; a ball of "
        f"radius {BALL_R:g} rests on it, so its centre is at y = {BALL_Y:g}. Floor positions (x, z) of the three balls, from the recurrences "
        "(the recurrence v' = eta (v - k (x - c)), x' = x + v' of the chapter, with the settings below, and the stop of C at x = 0.9):", "",
        _table(["frame", "A: spring", "B: glider", "C: slider"], prow, "rlll"), "",
        "Ball A: x from x_0 = -0.7, v_0 = 0.12, k = 0.12, eta = 0.995, c = 0; z from z_0 = 4.6, v_0 = 0.05, k = 0.08, eta = 0.995, c = 5.2. Ball B (k = 0): "
        "x from -0.9 with v_0 = 0.10 and z from 4.5 with v_0 = 0.11, both with eta = 0.9. Ball C (k = 0, eta = 1): x = min(-0.9 + 0.15 t, 0.9), z = 6.0. "
        f"A frame is the render (`gaussian_splatting.render`, background {_vec(SKY, 2)}) of the true scene: 14 Gaussians for the wall and floor "
        f"(`{TRUTH_NAME}`, seed {bookrng.seed_of(TRUTH_NAME)}: for each Gaussian in turn, wall (y = -0.8 then 0.6, x = -1.4, -0.1, 1.2, 2.5) "
        "then floor (z = 4.8 then 5.8, x = -0.8, 0, 0.8), three `uniform()` draws for the colour, 0.15 + 0.7 u on the wall and 0.3 + 0.4 u on the "
        "floor), scales (0.9, 0.8, 0.1) on the wall at z = 8 and (0.5, 0.04, 0.5) on the floor at y = 1.5, opacity 0.95, identity quaternion, and "
        f"three balls of scale {BALL_R:g} on all axes, opacity 0.95 and colours (0.9, 0.2, 0.2), (0.2, 0.8, 0.3), (0.9, 0.8, 0.2).", "",
        _table(["Gaussian", "part", "mean", "scales", "colour"], srows, "rllll"), "",
        f"The observed frames carry noise: `{IMG_NOISE_NAME}` (seed {bookrng.seed_of(IMG_NOISE_NAME)}), frame by frame, row by row, column by "
        f"column, channel by channel, `normal()` times {IMG_SIGMA:g} added to the clean value. The sum of the 2,304 values of clean frame 0 is "
        f"{s0:.6f}, of clean frame 11 {s11:.6f}, of observed frame 0 {n0:.6f}.", "",
        f"The feature tracks: 40 landmarks and the 3 ball centres (the points (x_t, {BALL_Y:g}, z_t)). The landmarks come from `{LAND_NAME}` "
        f"(seed {bookrng.seed_of(LAND_NAME)}): 20 wall points, three `uniform()` draws each in the order x = -1.8 + 4.2 u, y = -1.6 + 2.6 u, "
        "z = 7.8 + 0.4 u; then the four corners of the table in the order (xmin, zmin), (xmin, zmax), (xmax, zmin), (xmax, zmax); then 16 floor "
        "points, two draws each, x = -1.2 + 2.4 u, z = 4.0 + 2.4 u. The pixels are the projections plus noise from "
        f"`{NOISE_NAME}` (seed {bookrng.seed_of(NOISE_NAME)}), {TRACK_SIGMA:g} pixel per coordinate, view by view, track by track, `normal()` for u "
        f"then `normal()` for v. Landmark 0 is at {_vec(landmarks()[0], 5)}; its pixel in view 0 is ({pix[0][0][0]:.4f}, {pix[0][0][1]:.4f}) and "
        f"in view 11 ({pix[11][0][0]:.4f}, {pix[11][0][1]:.4f}); the pixel of ball A in view 0 is ({pix[0][40][0]:.4f}, {pix[0][40][1]:.4f})."])


# ------------------------------------------------------------------------------------------------ structure from motion (milestone 2)

def sfm_errors(st=None):
    """Per frame: the rotation error (degrees) and the camera-centre error (metres) of the recovered cameras against the true ones, the
    depth ratio of the points (recovered z over true z, mean) and the mean distance of the ball detections from the true positions."""
    rec, adj, st0, _ = base()
    st = st or st0
    truth = cameras()
    rows = []
    for v in range(OBSERVED):
        R, t = st["cams"][v]
        Rt, tt = truth[v]
        rows.append((sfm.rot_error(R, Rt), norm(sub(sfm.centre(R, t), sfm.centre(Rt, tt)))))
    pts = landmarks()
    ratio = [st["pts"][k][2] / pts[j][2] for k, j in enumerate(adj["kept"])]
    return rows, sum(ratio) / len(ratio), min(ratio), max(ratio)


def sparse_structure():
    rec, adj, st, _ = base()
    two = rec["two"]
    pix = tracks()
    a = [_norm_obs(pix, BASE_PAIR[0], j) for j in range(NTRACK)]
    b = [_norm_obs(pix, BASE_PAIR[1], j) for j in range(NTRACK)]
    Et = sfm.essential_of(*[x for x in cameras()[BASE_PAIR[1]]])
    rows, ratio, rmin, rmax = sfm_errors()
    P = adj["P"]
    n = len(P.obs)
    npar = P.nc + 3 * len(P.points)
    floor = TRACK_SIGMA * math.sqrt(2.0) * math.sqrt(max(0.0, (2 * n - npar) / (2 * n)))
    reg = [[v, rec["reg"][v][0], f"{rec['reg'][v][1]} of {rec['reg'][v][2]}"] for v in sorted(rec["reg"])]
    h1, h2 = adj["hist1"], adj["hist2"]
    paths = ball_paths()
    det_err = [sum(norm(sub(st["det"][j][v], paths[j][v])) for v in range(OBSERVED)) / OBSERVED for j in range(3)]
    err_rows = [[v, "%.3f" % rows[v][0], "%.4f" % rows[v][1]] for v in range(1, OBSERVED)]
    nsamp = sfm.ransac_iterations(sum(two["mask"]) / NTRACK, 8, 500)
    return "\n".join([
        f"Frames {BASE_PAIR[0]} and {BASE_PAIR[1]} give the pair (the distance of their centres, {BASELINE:.4f} metres, is the one length the "
        f"reader is told). `structure_from_motion.ransac_essential` on the 43 normalized correspondences, threshold {PAIR_THRESH_PX:g} pixels, cap 500, "
        f"generator `{RANSAC_NAME}` (seed {bookrng.seed_of(RANSAC_NAME)}): {two['iters']} iterations (the adaptive bound at the end is {nsamp}), "
        f"{sum(two['mask'])} inliers of 43, the tracks outside the model are {[j for j in range(NTRACK) if not two['mask'][j]]} (tracks 40, 41, 42 are the "
        f"balls), the pose has the rotation angle {_rot_deg(two['R']):.3f} degrees (true {_rot_deg(cameras()[BASE_PAIR[1]][0]):.3f}) and the "
        f"translation direction {sfm.dir_error(two['t'], cameras()[BASE_PAIR[1]][1]):.3f} degrees from the true one. The inlier tracks are triangulated (`structure_from_motion."
        "triangulate`, linear) from the pair and kept if in front of both cameras. The sampling is exactly this. A sample is 8 distinct track indices: "
        "`integer(43)` draws are taken from the generator one after another and a draw equal to an index already in the sample is discarded and drawn again, until 8 are held, "
        "in draw order. The eight-point algorithm (without refinement) gives a matrix from the sample; every one of the 43 correspondences is scored by its Sampson distance in "
        f"pixels d, and the hypothesis scores the sum of min(d^2, {PAIR_THRESH_PX:g}^2) (lower is better); a hypothesis replaces the best only when its score is strictly lower, and "
        "each time it does the cap is lowered to min(cap, the adaptive bound of the inlier fraction of the new best, 8-point samples), with the loop ending when the number of hypotheses "
        "tried reaches the cap. The inliers of the best (Sampson distance below the threshold) are fitted again by the eight-point algorithm; the pose of that essential matrix is the one of "
        "the four that puts the most inlier points in front of both cameras; and two rounds of Levenberg-Marquardt on the inliers' Sampson distances refine it, the inliers being recomputed "
        "from the refined pose after each round (the second round's inliers are the mask above).", "",
        "The other ten frames are registered one at a time by `structure_from_motion.ransac_pnp` on those points (6-point DLT, threshold "
        f"{PNP_THRESH_PX:g} pixels, cap 2000, generator `{RANSAC_NAME}.pnp.<v>` for frame v), each refined by 30 Levenberg-Marquardt iterations on "
        "its inliers with the points held. Frames are registered in increasing order, 1, 2, 3, 5, ..., 11 (frames 0 and 4 are the pair), each from a fresh generator of its own name; "
        "a sample is 6 distinct indices of the pair's points drawn as above with `integer(n)`, n the number of triangulated points; the 6-point DLT gives a pose; "
        f"the score is the sum of min(d^2, {PNP_THRESH_PX:g}^2) of the reprojection distances d in pixels, replaced only by a strictly lower one, with the cap lowered as above "
        "(sample size 6); the best pose's inliers (reprojection below the threshold) are the points of the refinement:", "",
        _table(["frame", "iterations", "inliers"], reg, "rrl"), "",
        f"The adjustment (`structure_from_motion.Problem`: frame {BASE_PAIR[0]} held, frame {BASE_PAIR[1]} the scale camera, the other ten free, "
        f"Huber threshold 2 pixels, lambda starting at 1e-3). Each inlier track is triangulated linearly from the observations of the frames whose "
        f"reprojection under the registered pose is within 8 pixels. Then {len(h1) - 1} iterations (RMS reprojection error {h1[0][4]:.3f} to "
        f"{h1[-1][4]:.3f} pixels); the median residual of every track is then below {GATE_PX:g} pixels (the gate of this milestone), so none is "
        f"dropped and the second solve has {len(h2) - 1} accepted iteration(s). {len(P.points)} points, {n} observations, {P.nc} camera unknowns "
        f"and {3 * len(P.points)} point coordinates; the expected RMS of a fit with the noise {TRACK_SIGMA:g} pixel is "
        f"sigma sqrt(2) sqrt((2 n - p) / (2 n)) = {floor:.3f}, the adjustment reaches {h2[-1][4]:.3f}.", "",
        "The baseline of the pair is set to its length, so every translation and every point is multiplied by it. Against the truth:", "",
        _table(["frame", "rotation error (degrees)", "centre error (metres)"], err_rows, "rrr"), "",
        f"The depth of the points, recovered over true, has mean {ratio:.4f} (range {rmin:.4f} to {rmax:.4f}): the structure is about "
        f"{100 * (1 - ratio):.1f}% too near while the camera centres are within {max(r[1] for r in rows):.3f} metres: the one measured baseline fixes the scale "
        f"of the cameras, not the depth of the points. The floor plane (`floor_plane`: for each point in turn the points within {PLANE_TOL:g} metres in y; "
        f"the largest set, the first on a tie) holds {len(st['floor_ids'])} of the 40 points at y = {st['floor_y']:.4f} (true {FLOOR_Y:g}). Their "
        f"bounding box in x and z is the table, [{st['table'][0]:.4f}, {st['table'][1]:.4f}] by [{st['table'][2]:.4f}, {st['table'][3]:.4f}]; inset by the "
        f"ball radius {BALL_R:g} it is the clamp box [{st['box'][0]:.4f}, {st['box'][1]:.4f}] by [{st['box'][2]:.4f}, {st['box'][3]:.4f}]. A ball "
        f"centre rests {BALL_R:g} above the floor, y = {st['y_ball']:.4f}; the ray through a ball track's pixel meets that plane at the detected position "
        f"(`ray_on_plane`), and the mean distance of the detections from the true positions over the 12 frames is {det_err[0]:.3f}, {det_err[1]:.3f} and "
        f"{det_err[2]:.3f} metres for A, B and C."])


# ------------------------------------------------------------------------------------------------ the first Gaussians (milestone 3)

def initial_gaussians():
    rec, adj, st, _ = base()
    clean, obs = true_frames()
    cams = export_cameras("clean", st["cams"])
    static, app = init_scene(st, cams, obs[0])
    rows = []
    for k, p in enumerate(static):
        rows.append([k, "wall" if k < N_WALL else "floor", _vec(p[0:3], 3), _vec([math.exp(x) for x in p[3:6]], 3), _vec(p[11:14], 3)])
    arows = []
    for j in range(3):
        x0, z0 = st["det"][j][0]
        arows.append([BALL_NAMES[j], f"({x0:.3f}, {z0:.3f})", _g(math.exp(app[j][0]), 3), _vec(app[j][2:5], 3)])
    pairs = sum(render_pairs(frame_params(static, app, st["det"], t, st["y_ball"]), cams[t]) for t in range(OBSERVED))
    loss = sum(gs._mse(gs.render(frame_params(static, app, st["det"], t, st["y_ball"]), cams[t], SKY), obs[t]) for t in range(OBSERVED)) / OBSERVED
    return "\n".join([
        f"From the structure of milestone 2. The points off the floor plane are the wall: sorted by x (ties by index) and cut into {N_WALL} chunks "
        f"of consecutive points (chunk k holds the points from floor(k n / {N_WALL}) up to floor((k + 1) n / {N_WALL}), n = 20), one Gaussian at the mean of each. "
        f"The {len(st['floor_ids'])} floor points are cut by x into {N_FLOOR} chunks the same way. Scales (before the exponential): wall "
        "(max(1.2 s_x, 0.3), max(1.2 s_y, 0.3), 0.15), floor (max(1.2 s_x, 0.3), 0.05, max(1.2 s_z, 0.4)), with s the standard deviation of the chunk "
        "(divisor n). Identity quaternion, opacity 0.8 (logit ln 4), and the colour of the pixel of observed frame 0 that the centre projects to "
        "(clipped to [0.05, 0.95]). The balls start at the detection of frame 0 with scale 0.25 on all three axes, opacity 0.8 and the colour of the pixel "
        "of frame 0 under it.", "",
        _table(["Gaussian", "part", "mean", "scales", "colour"], rows, "rllll"), "",
        _table(["ball", "detection at frame 0 (x, z)", "scale", "colour"], arows, "llll"), "",
        f"The loss of the starting scene (the mean over the 12 observed frames of the mean squared error over pixels and channels) is "
        f"{loss:.6f}. One render of one frame visits every record at every pixel, {pairs // OBSERVED} pairs of the 768 pixels and 13 Gaussians, "
        f"{pairs} for the 12 frames."])


def fit_curve():
    runs()
    rs = runs()["clean"]
    rows = [[t, cell("clean", lambda r, t=t: r["curve"][t], e3)] for t in CURVE_AT]
    apps = []
    nom = rs[0]["app"]
    for j in range(3):
        sc = cell("clean", lambda r, j=j: math.exp(r["app"][j][0]), lambda x: f"{x:.4f}")
        op = cell("clean", lambda r, j=j: gs.sigmoid(r["app"][j][1]), lambda x: f"{x:.4f}")
        col = ", ".join(cell("clean", lambda r, j=j, k=k: r["app"][j][2 + k], lambda x: f"{x:.4f}") for k in range(3))
        apps.append([BALL_NAMES[j], sc, op, col, _vec(BALL_COLOURS[j], 2)])
    return "\n".join([
        f"Full-batch Adam (beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8, bias correction), {FIT_STEPS} updates, on the 10 x 14 numbers of the static "
        "Gaussians (rates twice those of Chapter 6: mean 0.04, log scale 0.06, quaternion 0.02, opacity logit 0.2, colour 0.06) and the 3 x 5 appearance "
        "numbers of the balls (log scale, opacity logit, colour, at 0.06, 0.2, 0.06). The loss of an update is the mean over the 12 observed frames of the "
        "frame's mean squared error over pixels and channels; the gradient is `gaussian_splatting.view_gradient`, summed over the frames and divided by 12; "
        "the three numbers of a ball's log scale share one gradient (the sum of the three), and the balls' positions stay at the detections. "
        "Under the stable-digits rule the run is repeated with the first mean number of the first Gaussian changed by +1e-12, -1e-12, +3e-12 and -3e-12. "
        + stable.RULE, "",
        _table(["updates", "training loss"], rows, "rr"), "",
        "The balls' fitted appearance (scale is the exponential of the log scale; the true ball has scale 0.3 and opacity 0.95):", "",
        _table(["ball", "scale", "opacity", "colour (r, g, b)", "true colour"], apps, "lrrll")])


# ------------------------------------------------------------------------------------------------ the dynamics (milestone 5)

def dynamics():
    rec, adj, st, _ = base()
    model = fit_dynamics(st["det"])
    pred = predictions(st)
    errs = pos_errors("clean")
    rows = []
    for j in range(3):
        for k, axis in enumerate("xz"):
            a, b, c = model[j][k]
            rows.append([BALL_NAMES[j], axis, f"{a:.4f}", f"{b:.4f}", f"{c:.4f}"])
    one = []
    for j in range(3):
        for k, axis in enumerate("xz"):
            a, b, c = model[j][k]
            ser = [st["det"][j][t][k] for t in range(OBSERVED)]
            res = [ser[t + 1] - (c + a * ser[t] + b * ser[t - 1]) for t in range(1, OBSERVED - 1)]
            one.append(f"{BALL_NAMES[j][0]}{axis} {math.sqrt(sum(r * r for r in res) / len(res)):.4f}")
    erows = []
    for name, label in (("model", "learned, clamped"), ("free", "learned, unclamped"), ("persist", "last position held")):
        for j in range(3):
            erows.append([label if j == 0 else "", BALL_NAMES[j][0]] + ["%.3f" % x for x in errs[name][j]])
    return "\n".join([
        "The model of one coordinate of one ball is p' = c + a p + b q, with p the position at frame t and q the position at frame t - 1. It is fitted "
        f"to the detections of the 12 observed frames (the 10 transitions of frames 1 to 10 into 2 to 11) by ridge regression on the features (p, q, 1): "
        f"the normal equations with {RIDGE:g} added to the diagonal entries of a and b, solved by Gauss-Jordan elimination with partial pivoting, each sum over "
        "the transitions in order. Fitted coefficients:", "",
        _table(["ball", "axis", "a", "b", "c"], rows, "llrrr"), "",
        "The root mean square one-step residual over the 10 transitions, in metres: " + "; ".join(one) + ". The spring's coefficients can be read against the recurrence of milestone 1 "
        "(in x, a = 1 + eta - eta k = 1.8756 and b = -eta = -0.995), up to the noise of the detections; a glider's would be a = 1.9 and "
        "b = -0.9, which the nearly dependent features of ball B do not recover.", "",
        f"Free run from the detections of frames {OBSERVED - 2} and {OBSERVED - 1}, {HELD} frames, each prediction fed back (and, for the clamped "
        f"model, clipped to the clamp box of milestone 2 after every step). Distance in metres of the predicted position from the true one at frames "
        f"{OBSERVED} .. {FRAMES - 1}:", "",
        _table(["prediction", "ball"] + [f"frame {OBSERVED + k}" for k in range(HELD)], erows, "llrrrrrr")])


# ------------------------------------------------------------------------------------------------ rendering the prediction (milestone 6)

PREDICTORS = (("model", "learned model, clamped"), ("free", "learned model, unclamped"), ("persist", "balls held at their last detection"),
              ("truth", "balls at their true positions"), ("copy", "the last observed frame, copied"))


def held_out():
    rows = []
    for name, label in PREDICTORS:
        row = [label]
        for k in range(HELD):
            row.append(cell("clean", lambda r, k=k, name=name: r["mse"][name][k], e2))
        row.append(cell("clean", mean_mse(name), e2))
        rows.append(row)
    return "\n".join([
        "Mean squared error over pixels and channels of each held-out frame (12 .. 17) against the clean true frame, rendered at the true camera of that "
        "frame (exported as in milestone 4), with the Gaussians of milestone 4 and the balls where each row puts them. Each number carries the digits "
        "that the run and its four nudged copies share. The last row is not a render: it scores the noisy observed frame 11 as if it were the prediction.", "",
        _table(["prediction"] + [f"frame {OBSERVED + k}" for k in range(HELD)] + ["mean"], rows, "lrrrrrrr")])


# ------------------------------------------------------------------------------------------------ three planted mistakes (milestone 7)

def half_pixel_measure():
    """The pixel shift of the mistaken lens and the squared error between two renders of the true scene, through the right and through the
    mistaken lens, at the true cameras of frames 11 and 17."""
    wrong = splat_camera(*cameras()[0], mistaken=True)
    shift = (CXS - wrong["cx"], CYS - wrong["cy"])
    out = []
    for t in (OBSERVED - 1, FRAMES - 1):
        sc = true_scene_at(t)
        a = gs.render(sc, splat_camera(*cameras()[t]), SKY)
        b = gs.render(sc, splat_camera(*cameras()[t], mistaken=True), SKY)
        out.append(frame_mse(a, b))
    return shift, out


def transposed_measure():
    """For the cameras of the adjustment exported with R^T where R belongs: the camera-centre error (metres) at each frame (the centre the
    renderer would use is -R t), and the root mean square reprojection error (pixels at 640 by 480) of the recovered points through them,
    over the observations of the 12 frames, with the share of observations that land in front of the camera."""
    rec, adj, st, st_t = base()
    pix = tracks()
    truth = cameras()
    cent = []
    for v in range(OBSERVED):
        R, t = st["cams"][v]
        wrong = [-x for x in mv(R, t)]                       # the centre of the camera (R^T, t) is -(R^T)^T t = -R t
        cent.append(norm(sub(wrong, sfm.centre(*truth[v]))))
    err, front, n = [], 0, 0
    for v in range(OBSERVED):
        Rw, tw = tr(st["cams"][v][0]), st["cams"][v][1]
        for k, j in enumerate(adj["kept"]):
            n += 1
            e = sfm.reproj_px(Rw, tw, st["pts"][k], pix[v][j])
            if math.isfinite(e):
                front += 1
                err.append(e)
    return cent, math.sqrt(sum(e * e for e in err) / len(err)), front / n


def planted():
    runs()
    shift, hm = half_pixel_measure()
    cent, rms, front = transposed_measure()
    rec, adj, st, _ = base()
    pred = predictions(st)
    paths = ball_paths()
    over = [pred["free"][2][k][0] - st["box"][1] for k in range(HELD)]
    direct = [
        ["1", "half a pixel", "shift of every projection in the 32 by 24 image (pixels)", f"{shift[0]:.3f} and {shift[1]:.3f}"],
        ["1", "", "the same shift at 640 by 480 (pixels); at depth 5.2 m (metres)", f"{20 * shift[0]:.1f}; {abs(shift[0]) * 5.2 / FS:.4f}"],
        ["1", "", "squared error between the two renders of the true scene, frames 11 and 17", f"{hm[0]:.3e}; {hm[1]:.3e}"],
        ["2", "transposed rotation", "camera-centre error at frames 1, 4, 11 (metres): the cameras recovered in milestone 2, scaled, with R transposed, against the true centres", ", ".join(f"{cent[v]:.3f}" for v in (1, 4, 11))],
        ["2", "", "RMS reprojection error of the 40 points, 12 frames (pixels at 640 by 480); share in front", f"{rms:.1f}; {front:.3f}"],
        ["3", "no clamp", "ball C's predicted x past the clamp box at frames 12 .. 17 (metres)", ", ".join(f"{max(0.0, o):.3f}" for o in over)],
    ]
    end = []
    for label, kind, name in (("as specified", "clean", "model"), ("1: half a pixel", "half", "model"), ("2: transposed rotation", "transposed", "model"),
                              ("3: no clamp", "clean", "free")):
        errs = pos_errors(kind)[name]
        end.append([label, cell(kind, lambda r: r["curve"][FIT_STEPS], e2), cell(kind, mean_mse(name), e2),
                    cell(kind, mean_mse("truth"), e2), "%.3f" % max(errs[j][HELD - 1] for j in range(3))])
    return "\n".join([
        "Direct measurements, each a number that needs no fitted model:", "",
        _table(["mistake", "name", "measurement", "value"], direct, "lllr"), "",
        f"The pipeline run with each mistake planted alone. Runs 1 and 2 repeat milestones 3, 4 and 6 with the cameras exported the wrong way (in run 2 "
        f"the rays of the ball tracks use the exported cameras too, so the detections move); run 3 changes only the rollout. Training loss after {FIT_STEPS} updates; the held-out mean squared error "
        "of the prediction of that row and of the balls at their true positions (the reference, which no dynamics can influence); the largest distance of "
        f"a predicted ball from the true one at frame {FRAMES - 1}:", "",
        _table(["pipeline", "training loss", "held-out error", "held-out error, true balls", f"worst ball error at frame {FRAMES - 1} (m)"], end, "lrrrr")])


# ------------------------------------------------------------------------------------------------ the opening

def opening():
    runs()
    rows = []
    specs = [("the pipeline as specified", "clean", "model"), ("the same without the clamp on the rollout", "clean", "free"),
             ("a principal point by the centre rule (half a pixel)", "half", "model"), ("cameras exported with R transposed", "transposed", "model"),
             ("balls held at their last detection", "clean", "persist"), ("balls at their true positions (reference)", "clean", "truth")]
    for label, kind, name in specs:
        errs = pos_errors(kind)[name]
        rows.append([label, cell(kind, lambda r: r["curve"][FIT_STEPS], e2), cell(kind, mean_mse(name), e2),
                     "%.3f" % max(errs[j][HELD - 1] for j in range(3))])
    rows.append(["the last observed frame, copied", "-", cell("clean", mean_mse("copy"), e2), "-"])
    return "\n".join([
        f"Six held-out frames (frames {OBSERVED} .. {FRAMES - 1}) of the synthetic scene of milestone 1, predicted in seven ways, scored by the mean over the "
        "six of the squared error per pixel and channel against the clean true frame. 'Training loss' is the loss of the fit after "
        f"{FIT_STEPS} updates on the 12 observed frames; the last column is the largest distance, over the three balls, of the predicted floor "
        f"position from the true one at frame {FRAMES - 1}. Every trained number carries the digits that the run and its four nudged copies share.", "",
        _table(["how the frames were predicted", "training loss", "held-out error", f"worst ball error at frame {FRAMES - 1} (m)"], rows, "lrrr")])


# ------------------------------------------------------------------------------------------------ the bill (milestone 8)

SAMPSON_FLOPS = 45          # a hand count of `structure_from_motion.sampson_px`: two matrix-vector products (15 each), 7 for the denominator, 5 for the numerator, 3
REPROJ_FLOPS = 40           # a hand count of `structure_from_motion.reproj_px`: a rotation and a translation (18), the division and focal length (6), the hypot (4), the rest
BYTES = 4                   # 32-bit values
RIDGE_FIT_OPS = 260         # a hand allowance per ridge fit: the sums of the normal equations are 10 transitions x 12 products and sums x 2 = 240, and 20 more cover the ridge and the 3 by 3 elimination


def bill_rows():
    rec, adj, st, _ = base()
    P = adj["P"]
    ncam, npt = OBSERVED, len(P.points)
    k = len(P.obs) / npt
    n, nc, dense, reduced, mem_d, mem_r = sfm.work_counts(ncam, npt, k)
    tries = sum(h[1] for h in adj["hist1"][1:]) + sum(h[1] for h in adj["hist2"][1:])
    reg_iters = sum(v[0] for v in rec["reg"].values())
    clean, obs = true_frames()
    cams = export_cameras("clean", st["cams"])
    static, app = init_scene(st, cams, obs[0])
    pairs_frame = render_pairs(frame_params(static, app, st["det"], 0, st["y_ball"]), cams[0])
    nG = N_STATIC + 3
    per_frame_fwd = nG * gs.PROJECT_FLOPS + gs.PAIR_FLOPS * pairs_frame
    updates = FIT_STEPS + 1
    rows = [
        ["tracks", f"{OBSERVED} frames x {NTRACK} tracks x 2 values", "-", "-", OBSERVED * NTRACK * 2 * BYTES],
        ["pair: RANSAC scoring", f"{rec['two']['iters']} hypotheses x {NTRACK} correspondences x {SAMPSON_FLOPS}",
         rec["two"]["iters"] * NTRACK * SAMPSON_FLOPS, "-", NTRACK * 4 * BYTES],
        ["registration: RANSAC scoring", f"{reg_iters} hypotheses (10 frames) x {len(rec['ids'])} points x {REPROJ_FLOPS}",
         reg_iters * len(rec["ids"]) * REPROJ_FLOPS, "-", len(rec["ids"]) * 5 * BYTES],
        ["adjustment", f"{tries} damped steps x 2 x {reduced:,.2f} multiply-adds (Schur, {nc} camera unknowns, {npt} points, {k:.0f} views each)",
         int(2 * reduced * tries), "-", int(mem_r / 8 * BYTES)],
        ["fit: forward", f"{updates} loss evaluations x 12 frames x ({nG} Gaussians x {gs.PROJECT_FLOPS} + {pairs_frame:,} pairs x {gs.PAIR_FLOPS})",
         updates * OBSERVED * per_frame_fwd, "-", OBSERVED * 2 * H * W * 3 * BYTES],
        ["fit: backward", f"{updates - 1} updates x 12 frames x {pairs_frame:,} pairs replayed (not counted in operations)", "-", "-", OBSERVED * nG * 14 * BYTES],
        ["dynamics", "6 ridge fits (10 transitions, 3 features) and 2 rollouts of 6 frames", 6 * RIDGE_FIT_OPS + 2 * 6 * 3 * 2 * 4, "-", 0],
        ["held-out frames", f"4 predictions x {HELD} frames x ({nG} x {gs.PROJECT_FLOPS} + {pairs_frame:,} x {gs.PAIR_FLOPS})", 4 * HELD * per_frame_fwd, "-", 4 * HELD * H * W * 3 * BYTES],
    ]
    return rows, dict(pairs_frame=pairs_frame, per_frame_fwd=per_frame_fwd, nG=nG, updates=updates, tries=tries, reduced=reduced, mem_r=mem_r,
                      nc=nc, npt=npt, k=k)


def bill():
    runs()
    rows, c = bill_rows()
    flops = [r[2] for r in rows if isinstance(r[2], int)]
    total = sum(flops)
    out_rows = []
    for r in rows:
        share = f"{100.0 * r[2] / total:.2f}%" if isinstance(r[2], int) else "-"
        out_rows.append([r[0], r[1], f"{r[2]:,}" if isinstance(r[2], int) else r[2], share, f"{r[4]:,}" if r[4] else "-"])
    pairs_end = cell("clean", lambda r: r["pairs"], lambda x: f"{x:,.0f}")
    img = OBSERVED * H * W * 3 * BYTES
    par = OBSERVED * c["nG"] * 14 * BYTES
    upd_flops = OBSERVED * c["per_frame_fwd"]
    return "\n".join([
        "Operations (a multiply or an add is one; the counts of `gaussian_splatting`: 260 per Gaussian for the projection, 21 per pair) and bytes "
        "(32-bit values) of every stage, counted from the loops. 'Bytes' is what one pass of the stage moves: the observations, the matrix that is factored, the images read and written, the gradients written. "
        f"Counting rules. Adjustment: one damped step costs P (54 k + 108 k^2) + nc^3 / 6 multiply-adds, with P = {c['npt']} points, k = {c['k']:.0f} views each and nc = {c['nc']} camera unknowns "
        f"(6 per camera but two, plus 5), which is {c['reduced']:,.4f} per step, and the table's count is 2 x that x {c['tries']} steps = {2 * c['reduced'] * c['tries']:,.2f}, cut to an integer; its bytes are "
        f"4 (nc^2 + 9 P) = {int(c['mem_r'] / 8 * BYTES):,} (the reduced matrix and the 3 by 3 block of each point). Dynamics: each of the 6 ridge fits (3 balls x 2 coordinates) is "
        f"{RIDGE_FIT_OPS} (a hand allowance: the sums are 10 x 12 x 2 = 240 and 20 cover the ridge and the elimination), and each of the 2 rollouts (clamped and unclamped) is 6 frames x 3 balls x 2 coordinates x 4 "
        "(p' = c + a p + b q is two multiplies and two adds) = 144, so 6 x 260 + 2 x 144 = 1,848. The tracks row is 12 x 43 x 2 values of 4 bytes.", "",
        _table(["stage", "count", "operations", "share", "bytes"], out_rows, "llrrr"), "",
        f"The total is {total:,} operations, of which the forward passes of the fit are {100.0 * rows[4][2] / total:.1f}%. A render visits every record at every "
        f"pixel, {c['pairs_frame']:,} pairs a frame for the starting scene (and {pairs_end} for the 12 frames of the fitted scenes: no Gaussian is "
        f"culled, so the count does not move); an image of 32 by 24 pixels is four tiles of 16 by 16, so the tiled count of Chapter 6 is no smaller. One update "
        f"reads the 12 target frames ({img:,} bytes) and the parameters ({par:,} bytes) for {upd_flops:,} forward operations: "
        f"{upd_flops / (img + par):.1f} operations per byte moved, and the backward pass replays the same {OBSERVED * c['pairs_frame']:,} pairs."])


# ------------------------------------------------------------------------------------------------ acceptance tests

def acceptance_tests():
    runs()
    rec, adj, st, _ = base()
    clean, obs = true_frames()
    pix = tracks()
    two = rec["two"]
    rows_e, ratio, _, _ = sfm_errors()
    P = adj["P"]
    shift, hm = half_pixel_measure()
    cent, rms, front = transposed_measure()
    model = fit_dynamics(st["det"])
    errs = pos_errors("clean")
    bill_r, bc = bill_rows()
    total = sum(r[2] for r in bill_r if isinstance(r[2], int))
    s0 = sum(v for row in clean[0] for px in row for v in px)
    n0 = sum(v for row in obs[0] for px in row for v in px)
    cams = export_cameras("clean", st["cams"])
    static, app = init_scene(st, cams, obs[0])
    loss0 = sum(gs._mse(gs.render(frame_params(static, app, st["det"], t, st["y_ball"]), cams[t], SKY), obs[t]) for t in range(OBSERVED)) / OBSERVED
    rows = [
        (1, "sum of the 2,304 values of clean frame 0; of observed frame 0", f"{s0:.6f}; {n0:.6f}", "1e-6"),
        (1, "pixel of landmark 0 in view 0 and 11; of ball A in view 0", f"({pix[0][0][0]:.4f}, {pix[0][0][1]:.4f}); ({pix[11][0][0]:.4f}, {pix[11][0][1]:.4f}); ({pix[0][40][0]:.4f}, {pix[0][40][1]:.4f})", "1e-4"),
        (2, "pair: iterations; inliers of 43; tracks outside the model", f"{two['iters']}; {sum(two['mask'])}; {[j for j in range(NTRACK) if not two['mask'][j]]}", "exact"),
        (2, "adjustment: observations; points; final RMS reprojection (pixels)", f"{len(P.obs)}; {len(P.points)}; {adj['hist2'][-1][4]:.3f}", "exact; exact; 1e-2"),
        (2, "largest rotation error (degrees); largest centre error (metres)", f"{max(r[0] for r in rows_e):.3f}; {max(r[1] for r in rows_e):.4f}", "1e-2; 1e-3"),
        (2, "depth ratio; floor height; clamp box x range", f"{ratio:.4f}; {st['floor_y']:.4f}; [{st['box'][0]:.4f}, {st['box'][1]:.4f}]", "1e-3"),
        (3, "loss of the starting scene; pairs per frame", f"{loss0:.6f}; {bc['pairs_frame']}", "1e-6; exact"),
        (4, "training loss after 10, 60 and 120 updates", ", ".join(cell("clean", lambda r, t=t: r["curve"][t], e3) for t in (10, 60, FIT_STEPS)), "the digits shown"),
        (5, "ball A, x: a, b; ball A, z: a, b", f"{model[0][0][0]:.4f}, {model[0][0][1]:.4f}; {model[0][1][0]:.4f}, {model[0][1][1]:.4f}", "1e-3"),
        (5, f"clamped model, error of A and C at frame {FRAMES - 1} (metres)", f"{errs['model'][0][HELD - 1]:.3f}, {errs['model'][2][HELD - 1]:.3f}", "1e-3"),
        (5, f"unclamped model, error of C at frames {OBSERVED + 1} and {FRAMES - 1} (metres)", f"{errs['free'][2][1]:.3f}, {errs['free'][2][HELD - 1]:.3f}", "1e-3"),
        (6, "held-out mean squared error: clamped, unclamped, held, true balls, copied", ", ".join(cell("clean", mean_mse(n), e2) for n in ("model", "free", "persist", "truth", "copy")), "the digits shown"),
        (7, "half pixel: shift (pixels); squared error of the two renders, frames 11 and 17", f"{shift[0]:.3f}; {hm[0]:.2e}, {hm[1]:.2e}", "1e-3; 1e-2 relative"),
        (7, "half pixel: held-out error; training loss", f"{cell('half', mean_mse('model'), e2)}; {cell('half', lambda r: r['curve'][FIT_STEPS], e2)}", "the digits shown"),
        (7, "transposed (milestone-2 cameras, not the true ones): centre error at frame 11 (metres); RMS reprojection (pixels)", f"{cent[11]:.3f}; {rms:.1f}", "1e-3; 0.1"),
        (7, "transposed: held-out error; training loss", f"{cell('transposed', mean_mse('model'), e2)}; {cell('transposed', lambda r: r['curve'][FIT_STEPS], e2)}", "the digits shown"),
        (8, "operations of the whole loop; forward passes of the fit; pairs per frame", f"{total:,}; {bill_r[4][2]:,}; {bc['pairs_frame']:,}", "exact"),
    ]
    return "\n".join([
        "Expected values of the project, computed by this module. 'Exact' means equal; a number is compared with the tolerance in the last column, "
        "absolute unless it says relative; 'the digits shown' means the digits on which the run and its four nudged copies agree, and no more can be "
        "asked of another implementation.", "",
        _table(["milestone", "quantity", "value", "tolerance"], [[a, b, c, d] for a, b, c, d in rows], "rlll")])


# ------------------------------------------------------------------------------------------------ data of the figures

def _floor_plan():
    """The table, the clamp box, the true paths, the detections and the two predictions, for the floor-plan figure."""
    rec, adj, st, _ = base()
    pred = predictions(st)
    return dict(table=st["table"], box=st["box"], truth=ball_paths(), det=st["det"], model=pred["model"], free=pred["free"])
