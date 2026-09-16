// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Offscreen rendering gates.
//!
//! These use a three-line MJCF written here rather than the fly, so the
//! renderer is tested on any machine with MuJoCo and a GPU, whether or not the
//! flybody model is present. Like the rest of this crate's tests they skip
//! cleanly when the native dependency is missing.
//!
//! Each assertion below is paired with a control, because a renderer has an
//! unusually easy way to look correct: a black frame is a valid image, is
//! perfectly deterministic, and flips to itself. "The frame changed" only means
//! something if "the frame did not change" was also observed.

use std::sync::{Arc, Mutex, MutexGuard};

use mujoco::{Data, Model, MuJoCo, Renderer, StateSpec};

/// A red box on a checkered floor, lit, with the body free to be moved.
const SCENE: &str = r#"
<mujoco>
  <visual><global offwidth="320" offheight="240"/></visual>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.3 0.3 0.35 1"/>
    <body name="cube" pos="0 0 0.5">
      <freejoint/>
      <geom name="cube" type="box" size="0.2 0.2 0.2" rgba="0.9 0.1 0.1 1"/>
    </body>
  </worldbody>
</mujoco>
"#;

struct Scene {
    mj: Arc<MuJoCo>,
    model: Model,
    data: Data,
    // Held so the file outlives the model load.
    _dir: tempfile::TempDir,
}

fn scene() -> Option<Scene> {
    let mj = match MuJoCo::load() {
        Ok(m) => m,
        Err(e) => {
            brain_testutil::skip_unavailable(&format!("MuJoCo not loadable: {e}"));
            return None;
        }
    };
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("scene.xml");
    std::fs::write(&path, SCENE).expect("the scene writes");
    let model = Model::from_xml(&mj, &path).expect("the scene compiles");
    let data = Data::new(&model).expect("mj_makeData");
    Some(Scene { mj, model, data, _dir: dir })
}

/// Serialises the tests in this file. At most one [`Renderer`] may be live per
/// process, which the library enforces by refusing the second - so without this
/// the cargo test harness's parallelism turns into spurious failures.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn exclusive() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner())
}

/// A renderer, or a skip if this machine has no usable GPU context. Absence of
/// EGL is the same category as absence of MuJoCo: a missing capability.
fn renderer(s: &Scene, w: u32, h: u32) -> Option<Renderer> {
    match Renderer::for_model(&s.mj, &s.model, w, h) {
        Ok(r) => Some(r),
        Err(e) => {
            brain_testutil::skip_unavailable(&format!("no offscreen GL context: {e}"));
            None
        }
    }
}

/// How many bytes differ between two frames, and by the largest amount.
fn differing(a: &[u8], b: &[u8]) -> (usize, u8) {
    let n = a.iter().zip(b).filter(|(x, y)| x != y).count();
    let worst = a.iter().zip(b).map(|(x, y)| x.abs_diff(*y)).max().unwrap_or(0);
    (n, worst)
}

#[test]
fn a_rendered_frame_contains_the_scene_and_not_one_flat_colour() {
    let _one = exclusive();
    let Some(mut s) = scene() else { return };
    let Some(mut r) = renderer(&s, 320, 240) else { return };
    s.data.forward(&s.model);
    let frame = r.render(&s.model, &s.data).expect("a frame").to_vec();

    assert_eq!(frame.len(), 320 * 240 * 3, "the buffer is exactly the viewport");

    // A blank frame is the failure this is here to catch: a context that never
    // became current, or a scene that was never populated, both produce a
    // uniformly-cleared image that passes every weaker check.
    let mut colours: Vec<[u8; 3]> = frame.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    colours.sort_unstable();
    colours.dedup();
    assert!(colours.len() > 16, "the frame has only {} distinct colours, so nothing was drawn", colours.len());

    // The box is the only strongly red thing in the scene, so its presence is
    // evidence that geometry - not just a cleared background - reached the
    // framebuffer.
    let red = frame.chunks_exact(3).filter(|c| c[0] > 120 && c[0] as u16 > c[1] as u16 * 2 && c[0] as u16 > c[2] as u16 * 2).count();
    assert!(red > 200, "only {red} reddish pixels, so the box did not render");
}

#[test]
fn the_image_follows_the_state_and_holds_still_when_the_state_does() {
    let _one = exclusive();
    let Some(mut s) = scene() else { return };
    let Some(mut r) = renderer(&s, 320, 240) else { return };
    s.data.forward(&s.model);
    let first = r.render(&s.model, &s.data).expect("a frame").to_vec();

    // THE CONTROL. Without this, "moving the body changed the image" is also
    // satisfied by a renderer whose output is noise.
    //
    // Not bit-equality: GPU rasterisation is reproducible but not guaranteed
    // bit-reproducible, and measured, a repeat render of an unchanged state
    // differs in at most a single byte by a single level. Demanding exactness
    // would make this test fail on hardware rather than on a defect, and
    // demanding nothing would let noise through - so it is bounded on both
    // counts, which is what distinguishes it from the moved-body case below by
    // three orders of magnitude.
    let again = r.render(&s.model, &s.data).expect("a frame").to_vec();
    let (n, worst) = differing(&first, &again);
    assert!(
        n <= first.len() / 10_000 && worst <= 1,
        "re-rendering an unchanged state changed {n} of {} bytes by up to {worst}; that is more \
         than rasterisation rounding",
        first.len()
    );

    // Move the box far enough that it cannot overlap where it was.
    let mut qpos = s.data.get(&s.model, StateSpec::QPOS);
    qpos[0] += 2.0;
    qpos[2] += 1.0;
    s.data.set(&s.model, StateSpec::QPOS, &qpos).expect("the pose sets");
    s.data.forward(&s.model);
    let moved = r.render(&s.model, &s.data).expect("a frame").to_vec();

    let (d, _) = differing(&first, &moved);
    assert!(
        d > first.len() / 100,
        "moving the body changed only {d} of {} bytes; the renderer is not reading the state",
        first.len()
    );
}

#[test]
fn the_top_down_view_is_a_row_flip_of_the_raw_frame() {
    let _one = exclusive();
    let Some(mut s) = scene() else { return };
    let Some(mut r) = renderer(&s, 320, 240) else { return };
    s.data.forward(&s.model);
    let raw = r.render(&s.model, &s.data).expect("a frame").to_vec();
    let flipped = r.rgb_top_down();
    let row = 320 * 3;

    // THE CONTROL. A frame whose rows are all identical is flipped correctly by
    // any implementation, including one that does nothing, so establish first
    // that there is something to flip.
    assert_ne!(raw[..row], raw[raw.len() - row..], "the frame's top and bottom rows are identical; this cannot test a flip");

    assert_eq!(flipped[..row], raw[raw.len() - row..], "the first output row is not the last input row");
    assert_eq!(flipped[flipped.len() - row..], raw[..row], "the last output row is not the first input row");
}

#[test]
fn a_second_renderer_is_refused_while_the_first_is_alive() {
    let _one = exclusive();
    let Some(s) = scene() else { return };
    let Some(first) = renderer(&s, 64, 64) else { return };

    let second = Renderer::for_model(&s.mj, &s.model, 64, 64);
    let e = second.err().expect("a second live renderer must be refused, not silently allowed");
    assert!(e.contains("already live"), "the refusal must say why, got: {e}");

    // THE CONTROL. Without it, this test also passes against a Renderer that
    // can never be constructed twice for any reason at all - including one
    // permanently broken after the first use.
    drop(first);
    assert!(
        Renderer::for_model(&s.mj, &s.model, 64, 64).is_ok(),
        "the exclusion did not release when the first renderer was dropped"
    );
}
