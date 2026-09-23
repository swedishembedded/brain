// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Watching the model decide, and watching the cube turn.
//!
//! The cube is drawn as what it is: 26 plastic cubies, each a black box with
//! a coloured sticker on the faces that reach the surface, projected through
//! a real camera and sorted back to front. A turn is ANIMATED by rotating the
//! moving layer through the same angle the engine's own move applies, derived
//! from the same geometry the engine turns - so the picture cannot drift out
//! of agreement with the state. That is the whole reason
//! [`crate::cube::facelet_geometry`] is public rather than this file owning a
//! second copy of the layout.
//!
//! [`draw`] renders into a [`Canvas`] and nothing else. The window, the PNG
//! dump and the MP4 recording all go through it, so a headless run on a
//! server writes exactly the frames a person would have watched.
//!
//! Swedish Embedded AB builds operator-facing views of systems that decide on
//! their own - the thing you need before anybody will let such a system near
//! production. If your team needs that, you can procure our services by
//! sending an email to info@swedishembedded.com.

use brain::viewport::Canvas;

use crate::cube::{facelet_geometry, Cube, Face, Move};

pub const WIDTH: u32 = 1180;
pub const HEIGHT: u32 = 700;

const BG: [u8; 3] = [14, 14, 18];
const PANEL: [u8; 3] = [24, 24, 32];
const INK: [u8; 3] = [222, 222, 232];
const DIM: [u8; 3] = [132, 132, 148];
const GOOD: [u8; 3] = [90, 200, 120];
const BAD: [u8; 3] = [214, 92, 82];
const PICK: [u8; 3] = [250, 190, 60];
const TRACK: [u8; 3] = [44, 44, 56];

/// Standard colour scheme, indexed the way [`Cube`] stores a sticker: the
/// face it belongs on when solved.
const STICKER: [[u8; 3]; 6] = [
    [245, 245, 245], // U white
    [200, 60, 50],   // R red
    [60, 170, 90],   // F green
    [245, 205, 60],  // D yellow
    [235, 140, 50],  // L orange
    [55, 110, 205],  // B blue
];

/// One option as the panel shows it.
pub struct Row {
    pub name: String,
    pub detail: String,
    pub probability: f32,
    pub admissible: bool,
}

/// Everything the right-hand panel reports about this turn.
#[derive(Default)]
pub struct Panel {
    pub model: String,
    pub cube_index: usize,
    pub cubes: usize,
    pub distance: u8,
    pub state: String,
    pub rows: Vec<Row>,
    pub picked: Option<usize>,
    pub played: Option<usize>,
    pub shielded: bool,
    pub thinking: bool,
    pub history: Vec<String>,
    pub turns: usize,
    pub top_admissible: usize,
    pub chance: f32,
    pub interventions: usize,
    pub solved: usize,
    pub unassisted: bool,
    /// Whether a planner is in the loop at all.
    ///
    /// The learned policy runs without one, so it has no notion of
    /// admissibility and no exact distance to report. Drawing those fields
    /// anyway would show a truthful-looking `0%` for a quantity that was
    /// never measured, which is worse than leaving it out.
    pub planner: bool,
    /// Stickers already on their home face, when there is no planner to give
    /// an exact distance. Progress that needs no search.
    pub home: usize,
    /// What is driving, when it is not a model scoring options.
    ///
    /// The macro library decides by a monotone measure and consults no
    /// network at all, so the lines about confidence and forward passes
    /// describe nothing. Rendering them anyway would put a number on screen
    /// that was never computed.
    pub driver: Option<String>,
}

/// A camera: how far the cube is turned towards the viewer, and how far the
/// animating layer has got.
pub struct Scene {
    pub yaw: f32,
    pub pitch: f32,
    pub turning: Option<(Move, f32)>,
}

impl Default for Scene {
    fn default() -> Scene {
        // Three faces visible, which is the view that shows a cube is a cube.
        Scene { yaw: -0.62, pitch: 0.48, turning: None }
    }
}

type V3 = [f32; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn dot(a: V3, b: V3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(v: V3) -> V3 {
    let l = dot(v, v).sqrt().max(1e-6);
    [v[0] / l, v[1] / l, v[2] / l]
}

/// Rotate about a coordinate axis by `angle`, right-handed.
fn rotate_axis(v: V3, axis: usize, angle: f32) -> V3 {
    let (s, c) = angle.sin_cos();
    let (a, b) = match axis {
        0 => (1, 2),
        1 => (2, 0),
        _ => (0, 1),
    };
    let mut out = v;
    out[a] = v[a] * c - v[b] * s;
    out[b] = v[a] * s + v[b] * c;
    out
}

/// The turn a face makes, as a continuous angle.
///
/// A clockwise quarter turn of a face, seen from outside it, is a rotation
/// about that face's OUTWARD normal by -90 degrees - the same convention
/// `cube::rotate` implements discretely. Deriving the animation from the same
/// rule is what makes the cube land exactly where the state says it is when
/// the animation reaches the end.
fn layer_angle(face: Face, quarters: u8, progress: f32) -> (usize, i8, f32) {
    let (axis, sign) = match face {
        Face::R => (0, 1i8),
        Face::L => (0, -1),
        Face::U => (1, 1),
        Face::D => (1, -1),
        Face::F => (2, 1),
        Face::B => (2, -1),
    };
    let quarter = std::f32::consts::FRAC_PI_2;
    (axis, sign, -(sign as f32) * quarters as f32 * quarter * progress)
}

/// One flat quad in cube space, with the colour it is drawn in.
struct Quad {
    p: [V3; 4],
    colour: [u8; 3],
    outline: bool,
}

/// Build every quad of the cube: each cubie is a black box, and each face of
/// it that reaches the surface carries an inset sticker.
fn quads(cube: &Cube, scene: &Scene) -> Vec<Quad> {
    let geom = facelet_geometry();
    let mut out = Vec::with_capacity(220);
    let moving = scene.turning.map(|(m, p)| (layer_angle(m.face, m.quarters, p), m));

    // Which sticker sits on (cell, normal), so a cubie face knows its colour.
    let sticker_at = |cell: [i8; 3], n: [i8; 3]| -> Option<u8> {
        (0..54).find(|&i| geom[i] == (cell, n)).map(|i| cube.0[i])
    };

    for x in -1..=1i8 {
        for y in -1..=1i8 {
            for z in -1..=1i8 {
                if x == 0 && y == 0 && z == 0 {
                    continue;
                }
                let cell = [x, y, z];
                for axis in 0..3usize {
                    for sign in [-1i8, 1] {
                        let mut n = [0i8; 3];
                        n[axis] = sign;
                        let centre = face_centre(cell, axis, sign);
                        let (u, v) = in_face_axes(axis, sign);
                        let body = quad(centre, u, v, 0.5);
                        let mut push = |p: [V3; 4], colour: [u8; 3], outline: bool| {
                            let p = p.map(|q| spin(q, cell, &moving));
                            out.push(Quad { p, colour, outline });
                        };
                        push(body, [16, 16, 20], false);
                        if cell[axis] == sign {
                            if let Some(c) = sticker_at(cell, n) {
                                // Lifted a hair off the plastic so the fill
                                // beneath never shows through at a seam.
                                let lifted = quad(offset(centre, axis, sign as f32 * 0.012), u, v, 0.41);
                                push(lifted, STICKER[c as usize], true);
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

fn face_centre(cell: [i8; 3], axis: usize, sign: i8) -> V3 {
    let mut c = [cell[0] as f32, cell[1] as f32, cell[2] as f32];
    c[axis] = cell[axis] as f32 + sign as f32 * 0.5;
    c
}

fn offset(mut p: V3, axis: usize, d: f32) -> V3 {
    p[axis] += d;
    p
}

/// The two in-face directions of a face, wound so that `u x v` is the
/// face's OUTWARD normal.
///
/// Getting this wrong is invisible in a still of one face and fatal overall:
/// a face whose winding produces an inward normal is culled as a backface,
/// and the cube renders with a hole where its top used to be. Which is
/// exactly what happened - hence the test below that checks all six.
fn in_face_axes(axis: usize, sign: i8) -> (V3, V3) {
    // y x z = +x, z x x = +y, x x y = +z.
    let (u, v) = match axis {
        0 => ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        1 => ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        _ => ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    };
    // A face on the negative side of the cube points the other way, so its
    // winding has to reverse with it.
    if sign < 0 {
        (v, u)
    } else {
        (u, v)
    }
}

fn quad(c: V3, u: V3, v: V3, h: f32) -> [V3; 4] {
    let add = |a: V3, b: V3, s: f32| [a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s];
    [
        add(add(c, u, -h), v, -h),
        add(add(c, u, h), v, -h),
        add(add(c, u, h), v, h),
        add(add(c, u, -h), v, h),
    ]
}

/// Apply the animating layer's rotation, if this cubie is in it.
fn spin(p: V3, cell: [i8; 3], moving: &Option<((usize, i8, f32), Move)>) -> V3 {
    match moving {
        Some(((axis, sign, angle), _)) if cell[*axis] == *sign => rotate_axis(p, *axis, *angle),
        _ => p,
    }
}

/// Camera: yaw about the vertical, then pitch, then a real perspective
/// divide - a cube drawn in orthographic looks like a hexagon of coloured
/// rhombi, and stops reading as an object.
fn to_camera(p: V3, scene: &Scene) -> V3 {
    rotate_axis(rotate_axis(p, 1, scene.yaw), 0, scene.pitch)
}

/// Camera at +z looking back at the origin, with a real perspective divide -
/// a cube drawn in orthographic looks like a hexagon of coloured rhombi and
/// stops reading as an object.
fn project(p: V3, scene: &Scene, cx: f32, cy: f32, scale: f32) -> (f32, f32, f32) {
    let p = to_camera(p, scene);
    let depth = (9.0 - p[2]).max(0.1);
    let f = scale / depth;
    (cx + p[0] * f, cy - p[1] * f, depth)
}

/// Fill a convex polygon, scanline by scanline.
fn fill_poly(canvas: &mut Canvas, pts: &[(f32, f32)], colour: [u8; 3]) {
    let (w, h) = (canvas.width() as i32, canvas.height() as i32);
    let ymin = pts.iter().fold(f32::MAX, |a, p| a.min(p.1)).floor().max(0.0) as i32;
    let ymax = pts.iter().fold(f32::MIN, |a, p| a.max(p.1)).ceil().min(h as f32 - 1.0) as i32;
    let px = canvas.pixels_mut();
    for y in ymin..=ymax {
        let yc = y as f32 + 0.5;
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for i in 0..pts.len() {
            let (x0, y0) = pts[i];
            let (x1, y1) = pts[(i + 1) % pts.len()];
            if (y0 <= yc && y1 > yc) || (y1 <= yc && y0 > yc) {
                let t = (yc - y0) / (y1 - y0);
                let x = x0 + (x1 - x0) * t;
                lo = lo.min(x);
                hi = hi.max(x);
            }
        }
        if lo > hi {
            continue;
        }
        let x0 = lo.round().max(0.0) as i32;
        let x1 = hi.round().min(w as f32 - 1.0) as i32;
        for x in x0..=x1 {
            let i = ((y * w + x) * 3) as usize;
            if i + 2 < px.len() {
                px[i] = colour[0];
                px[i + 1] = colour[1];
                px[i + 2] = colour[2];
            }
        }
    }
}

fn shade(c: [u8; 3], lambert: f32) -> [u8; 3] {
    let k = (0.42 + 0.58 * lambert.clamp(0.0, 1.0)).clamp(0.0, 1.0);
    [
        (c[0] as f32 * k) as u8,
        (c[1] as f32 * k) as u8,
        (c[2] as f32 * k) as u8,
    ]
}

/// Draw the cube itself into the left-hand area.
fn draw_cube(canvas: &mut Canvas, cube: &Cube, scene: &Scene, cx: f32, cy: f32, scale: f32) {
    let light = norm([0.45, 0.8, 0.6]);
    let mut prepared: Vec<(f32, [(f32, f32); 4], [u8; 3], bool)> = Vec::new();
    for q in quads(cube, scene) {
        let n = norm(cross(sub(q.p[1], q.p[0]), sub(q.p[2], q.p[0])));
        // Cull on the face's own normal in camera space, not on screen
        // winding: the screen's y axis points down, so a winding test reads
        // backwards and paints the inside of the cube over its front.
        if to_camera(n, scene)[2] <= 0.0 {
            continue;
        }
        let projected: Vec<(f32, f32, f32)> = q.p.iter().map(|p| project(*p, scene, cx, cy, scale)).collect();
        let depth = projected.iter().map(|p| p.2).sum::<f32>() / 4.0;
        let pts = [
            (projected[0].0, projected[0].1),
            (projected[1].0, projected[1].1),
            (projected[2].0, projected[2].1),
            (projected[3].0, projected[3].1),
        ];
        prepared.push((depth, pts, shade(q.colour, dot(n, light)), q.outline));
    }
    // Painter's algorithm: furthest first.
    prepared.sort_by(|a, b| b.0.total_cmp(&a.0));
    for (_, pts, colour, outline) in prepared {
        fill_poly(canvas, &pts, colour);
        if outline {
            // A thin dark rim reads as the bevel on a real sticker.
            let dark = [colour[0] / 3, colour[1] / 3, colour[2] / 3];
            for i in 0..4 {
                line(canvas, pts[i], pts[(i + 1) % 4], dark);
            }
        }
    }
}

fn line(canvas: &mut Canvas, a: (f32, f32), b: (f32, f32), colour: [u8; 3]) {
    let steps = ((b.0 - a.0).abs().max((b.1 - a.1).abs()) as i32).max(1);
    let (w, h) = (canvas.width() as i32, canvas.height() as i32);
    let px = canvas.pixels_mut();
    for s in 0..=steps {
        let t = s as f32 / steps as f32;
        let x = (a.0 + (b.0 - a.0) * t).round() as i32;
        let y = (a.1 + (b.1 - a.1) * t).round() as i32;
        if x < 0 || y < 0 || x >= w || y >= h {
            continue;
        }
        let i = ((y * w + x) * 3) as usize;
        px[i] = colour[0];
        px[i + 1] = colour[1];
        px[i + 2] = colour[2];
    }
}

/// One frame: the cube, and the decision beside it.
pub fn draw(canvas: &mut Canvas, cube: &Cube, scene: &Scene, panel: &Panel) {
    canvas.clear(BG);
    let panel_x = 700i32;
    // Sized so the whole cube stays inside its half of the frame at every
    // camera angle: the corner-to-centre distance is sqrt(3) * 1.5 units, and
    // a turning layer swings a little wider than that.
    draw_cube(canvas, cube, scene, 350.0, 350.0, 760.0);

    // What is being turned, under the cube.
    if let Some((m, _)) = scene.turning {
        let label = format!("{}   {}", m.notation(), m.describe());
        canvas.text(40, 648, &label, 2, PICK);
    }
    canvas.text(40, 24, "BRAIN - A DECISION MODEL TURNING A RUBIK'S CUBE", 1, DIM);
    canvas.text(40, 24 + lh(1), &format!("{} of 54 stickers home", cube.facelets_home()), 1, DIM);

    canvas.fill(panel_x, 0, WIDTH - panel_x as u32, HEIGHT, PANEL);
    let x = panel_x + 22;
    let mut y = 26i32;
    canvas.text(x, y, &format!("MODEL  {}", panel.model.to_uppercase()), 2, INK);
    y += lh(2) + 6;
    canvas.text(
        x,
        y,
        &if panel.planner {
            format!("cube {} of {}   {} from solved", panel.cube_index + 1, panel.cubes, moves(panel.distance))
        } else {
            format!("cube {} of {}   {} of 54 stickers home", panel.cube_index + 1, panel.cubes, panel.home)
        },
        1,
        DIM,
    );
    y += lh(1) + 12;

    canvas.text(x, y, "STATE - ALL THE MODEL IS TOLD", 1, DIM);
    y += lh(1) + 4;
    for chunk in wrap(&panel.state, 56) {
        canvas.text(x, y, &chunk, 1, INK);
        y += lh(1);
    }
    y += 12;

    canvas.text(x, y, "OPTIONS - SUPPLIED AT RUN TIME", 1, DIM);
    y += lh(1) + 6;
    for (i, row) in panel.rows.iter().enumerate() {
        let picked = panel.picked == Some(i);
        let played = panel.played == Some(i);
        let good = panel.planner && row.admissible;
        let ink = if picked { PICK } else if good { GOOD } else { INK };
        canvas.text(x, y, &row.name, 1, ink);
        canvas.text(x + 30, y, &truncate(&row.detail, 46), 1, if good { GOOD } else { DIM });
        canvas.bar(x, y + lh(1) + 1, 360, 6, row.probability.clamp(0.0, 1.0), ink, TRACK);
        canvas.text(x + 372, y + lh(1) - 1, &format!("{:.2}", row.probability), 1, DIM);
        if played {
            canvas.text(x - 14, y, ">", 1, INK);
        }
        y += lh(1) * 2 + 6;
    }

    y += 6;
    // Both verdicts below are the PLANNER's. Without one there is nothing
    // here that could know whether a pick was on a shortest path, and saying
    // so anyway would put an unmeasured claim on the screen.
    if panel.thinking {
        canvas.text(x, y, "ASKING THE MODEL ...", 1, DIM);
    } else if panel.planner && panel.shielded {
        canvas.text(x, y, "SHIELD - FIRST PICK WAS NOT ON A SHORTEST PATH", 1, BAD);
    } else if panel.planner && panel.picked.is_some() {
        canvas.text(x, y, "THE MODEL'S OWN PICK WAS ON A SHORTEST PATH", 1, GOOD);
    }
    y += lh(1) + 14;

    let rate = |n: usize| if panel.turns == 0 { 0.0 } else { 100.0 * n as f32 / panel.turns as f32 };
    canvas.text(x, y, "SO FAR", 1, DIM);
    y += lh(1) + 4;
    canvas.text(x, y, &format!("turns decided        {}", panel.turns), 1, INK);
    y += lh(1);
    if panel.planner {
        canvas.text(
            x,
            y,
            &format!("first pick admissible {:.0}%   (chance {:.0}%)", rate(panel.top_admissible), 100.0 * panel.chance),
            1,
            INK,
        );
        y += lh(1);
        if panel.unassisted {
            canvas.text(x, y, "no shield: the model plays its own pick", 1, BAD);
        } else {
            canvas.text(x, y, &format!("shield interventions {}", panel.interventions), 1, INK);
        }
    } else if let Some(note) = &panel.driver {
        canvas.text(x, y, note, 1, BAD);
        y += lh(1);
    } else {
        canvas.text(x, y, &format!("confidence in its pick {:.2}", panel.chance), 1, INK);
        y += lh(1);
        canvas.text(x, y, "no planner, no search: one forward pass per move", 1, BAD);
        y += lh(1);
    }
    y += lh(1);
    canvas.text(x, y, &format!("cubes solved         {}", panel.solved), 1, INK);
    y += lh(1) + 14;

    canvas.text(x, y, "PLAYED", 1, DIM);
    y += lh(1) + 4;
    for chunk in wrap(&panel.history.join(" "), 56) {
        canvas.text(x, y, &chunk, 1, INK);
        y += lh(1);
    }
}

/// One line of text at scale `px`, so the panel lays out against the font
/// rather than against numbers that were right once.
fn lh(px: u32) -> i32 {
    Canvas::line_height(px) as i32
}

fn truncate(s: &str, cols: usize) -> String {
    if s.chars().count() <= cols {
        return s.to_string();
    }
    s.chars().take(cols.saturating_sub(1)).collect::<String>() + "."
}

fn moves(d: u8) -> String {
    if d == 1 {
        "1 move".to_string()
    } else {
        format!("{d} moves")
    }
}

/// Break text on spaces at `cols` characters, because the canvas draws a
/// fixed-width font and a panel that runs off its own edge hides the number
/// somebody opened the window for.
fn wrap(s: &str, cols: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut line = String::new();
    for word in s.split_whitespace() {
        if !line.is_empty() && line.len() + 1 + word.len() > cols {
            out.push(std::mem::take(&mut line));
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(word);
    }
    if !line.is_empty() {
        out.push(line);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cube::Move;

    /// The animation has to END where the engine's own move lands, or the
    /// picture and the state disagree at every turn. At progress 1.0 the
    /// rotation of a quarter turn must map the cube's axes the way
    /// `cube::rotate` does - checked here on a corner of the R layer.
    #[test]
    fn a_finished_animation_lands_exactly_where_the_move_does() {
        for face in Face::ALL {
            for quarters in 1..=3u8 {
                let m = Move { face, quarters };
                let moving = Some((layer_angle(face, quarters, 1.0), m));
                let cell = [1i8, 1, 1];
                let start = [1.0f32, 1.0, 1.0];
                // Through `spin`, which is what the renderer actually calls -
                // including its decision about whether this cubie is in the
                // turning layer at all.
                let ended = spin(start, cell, &moving).map(|v| v.round());

                // The same corner, moved by the engine's own discrete rule,
                // applied `quarters` times - and only if it is in the layer.
                let (axis, sign, _) = layer_angle(face, quarters, 1.0);
                let mut want = cell;
                if cell[axis] == sign {
                    for _ in 0..quarters {
                        want = crate::cube::rotate_for_test(face, want);
                    }
                }
                let want = [want[0] as f32, want[1] as f32, want[2] as f32];
                assert_eq!(ended, want, "{}", m.notation());
            }
        }
    }

    #[test]
    fn a_turn_in_progress_is_between_the_two_states() {
        let (axis, _, angle) = layer_angle(Face::R, 1, 0.5);
        let half = rotate_axis([0.0, 1.0, 0.0], axis, angle);
        assert!(half[1] > 0.6 && half[1] < 0.8, "half a quarter turn is 45 degrees: {half:?}");
    }

    #[test]
    fn every_sticker_is_drawn_once_per_frame() {
        let scene = Scene::default();
        let qs = quads(&Cube::SOLVED, &scene);
        let coloured = qs.iter().filter(|q| q.outline).count();
        assert_eq!(coloured, 54, "one sticker quad per facelet");
        // 26 cubies x 6 faces of black plastic underneath them.
        assert_eq!(qs.len() - coloured, 26 * 6);
    }

    /// Every face must wind to its own outward normal, or it is culled as a
    /// backface and the cube renders with a hole in it.
    #[test]
    fn every_face_winds_outward() {
        for axis in 0..3usize {
            for sign in [-1i8, 1] {
                let (u, v) = in_face_axes(axis, sign);
                let n = cross(u, v);
                let mut want = [0.0f32; 3];
                want[axis] = sign as f32;
                assert_eq!(n, want, "axis {axis} sign {sign} winds inward");
            }
        }
    }

    #[test]
    fn wrapping_never_loses_a_word() {
        let s = "the quick brown fox jumps over the lazy dog";
        let joined = wrap(s, 12).join(" ");
        assert_eq!(joined, s);
    }

    #[test]
    fn a_frame_draws_without_panicking_and_is_not_blank() {
        let mut canvas = Canvas::new(WIDTH, HEIGHT);
        let panel = Panel {
            model: "laya".into(),
            cubes: 1,
            distance: 5,
            state: "A scrambled Rubik's cube with 30 of its 54 stickers in place.".into(),
            rows: vec![Row { name: "R2".into(), detail: "one step closer".into(), probability: 0.4, admissible: true }],
            picked: Some(0),
            played: Some(0),
            ..Panel::default()
        };
        let scene = Scene { turning: Some((Move { face: Face::R, quarters: 1 }, 0.5)), ..Scene::default() };
        draw(&mut canvas, &Cube::SOLVED, &scene, &panel);
        let lit = canvas.pixels().chunks(3).filter(|p| *p != BG).count();
        assert!(lit > 20_000, "a frame with a cube in it is not mostly background: {lit} pixels");
    }
}
