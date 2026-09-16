// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! What does one unit of `mjv_moveCamera` actually move?
//!
//! The camera can be driven without reading any of its fields, which is what
//! makes it usable from this binding at all - but "a fraction of the window"
//! is not a world distance, and a tracking shot needs the conversion. So
//! measure it: put a marker at a known place, pan by a known amount, and see
//! how far the marker moved on screen.

use mujoco::{Camera, Data, Model, MuJoCo, Renderer};

const SCENE: &str = r#"
<mujoco>
  <statistic extent="10" center="0 0 0"/>
  <visual><global offwidth="320" offheight="240"/></visual>
  <worldbody>
    <light pos="0 0 10" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="30 30 .1" rgba="0.25 0.25 0.3 1"/>
    <geom name="mark" type="sphere" size="0.4" pos="0 0 0.4" rgba="1 0.2 0.1 1"/>
    <geom name="mark_x" type="sphere" size="0.4" pos="2 0 0.4" rgba="0.1 1 0.2 1"/>
    <geom name="mark_y" type="sphere" size="0.4" pos="0 2 0.4" rgba="0.2 0.2 1 1"/>
  </worldbody>
</mujoco>
"#;

/// Screen centroid of one colour channel's marker, in pixels.
fn marker(rgb: &[u8], w: usize, channel: usize) -> Option<(f64, f64)> {
    let (mut sx, mut sy, mut n) = (0.0, 0.0, 0.0);
    for (i, p) in rgb.chunks_exact(3).enumerate() {
        let (a, b, c) = (p[channel] as u16, p[(channel + 1) % 3] as u16, p[(channel + 2) % 3] as u16);
        if a > 110 && a > b * 2 && a > c * 2 {
            sx += (i % w) as f64;
            sy += (i / w) as f64;
            n += 1.0;
        }
    }
    (n > 0.0).then(|| (sx / n, sy / n))
}

fn marker_x(rgb: &[u8], w: usize) -> Option<f64> {
    marker(rgb, w, 0).map(|(x, _)| x)
}

fn main() {
    let mj = MuJoCo::load().expect("MuJoCo");
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("pan.xml");
    std::fs::write(&path, SCENE).unwrap();
    let model = Model::from_xml(&mj, &path).expect("the scene compiles");
    let mut data = Data::new(&model).unwrap();
    data.forward(&model);
    let mut r = Renderer::new(&mj, &model, 320, 240, 2000).expect("a context");

    r.render(&model, &data).unwrap();
    let frame = r.rgb_top_down();
    let origin = marker(&frame, 320, 0).expect("the origin marker is visible");
    let px = marker(&frame, 320, 1).expect("the +x marker is visible");
    let py = marker(&frame, 320, 2).expect("the +y marker is visible");
    println!("extent 10");
    println!("  world (0,0) at screen {origin:?}");
    println!("  world (2,0) at screen {px:?}  ->  per world x: ({:+.2}, {:+.2}) px", (px.0 - origin.0) / 2.0, (px.1 - origin.1) / 2.0);
    println!("  world (0,2) at screen {py:?}  ->  per world y: ({:+.2}, {:+.2}) px", (py.0 - origin.0) / 2.0, (py.1 - origin.1) / 2.0);
    let before = origin.0;

    // Solve the 2x2 that a tracking camera needs: which pan moves the look-at
    // point along which world axis, and by how much. Both components are
    // measured; assuming the second matches the first is how a camera ends up
    // pointing at empty floor.
    let per_wx = ((px.0 - origin.0) / 2.0, (px.1 - origin.1) / 2.0);
    let per_wy = ((py.0 - origin.0) / 2.0, (py.1 - origin.1) / 2.0);
    for (label, rx, ry) in [("PanH.x", 0.1, 0.0), ("PanH.y", 0.0, 0.1)] {
        r.move_camera(Camera::PanH, rx, ry);
        r.render(&model, &data).unwrap();
        let after = marker(&r.rgb_top_down(), 320, 0).expect("marker in frame");
        let (sx, sy) = (after.0 - origin.0, after.1 - origin.1);
        // Screen displacement back to world, by inverting the two columns
        // measured above.
        let det = per_wx.0 * per_wy.1 - per_wx.1 * per_wy.0;
        let wx = (sx * per_wy.1 - sy * per_wy.0) / det;
        let wy = (per_wx.0 * sy - per_wx.1 * sx) / det;
        // The MARKER moved that way on screen, so the look-at point moved the
        // other way in the world.
        println!("  {label} +0.1 moves look-at by world ({:+.4}, {:+.4}) = extent * ({:+.4}, {:+.4})", -wx, -wy, -wx / 10.0, -wy / 10.0);
        r.move_camera(Camera::PanH, -rx, -ry);
    }

    for step in [0.05f64, 0.10, 0.20] {
        r.move_camera(Camera::PanH, step, 0.0);
        r.render(&model, &data).unwrap();
        match marker_x(r.rgb_top_down().as_slice(), 320) {
            Some(x) => println!("  after PanH {step:+.2}: x = {x:.1} px (moved {:+.1})", x - before),
            None => println!("  after PanH {step:+.2}: marker left the frame"),
        }
        r.move_camera(Camera::PanH, -step, 0.0);
    }
    for step in [0.05f64, 0.10, 0.20] {
        r.move_camera(Camera::PanH, 0.0, step);
        r.render(&model, &data).unwrap();
        match marker_x(r.rgb_top_down().as_slice(), 320) {
            Some(x) => println!("  after PanV-axis {step:+.2}: x = {x:.1} px (moved {:+.1})", x - before),
            None => println!("  after PanV-axis {step:+.2}: marker left the frame"),
        }
        r.move_camera(Camera::PanH, 0.0, -step);
    }
}
