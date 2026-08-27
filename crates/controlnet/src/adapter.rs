// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The seam: **named injection points**, and the two halves that meet at them.
//!
//! A ControlNet is a trainable copy of a diffusion backbone's early blocks
//! whose zero-conv outputs are *added as residuals* at named places in that
//! backbone. Nothing about that is UNet-specific — a FLUX ControlNet injects
//! into double-stream blocks the same way, and an SD3 one into its MMDiT
//! blocks. So the thing this crate exports first is not "the SDXL ControlNet"
//! but the contract:
//!
//! * a **[`ControlSource`]** (a control model) declares the points it produces
//!   residuals for, and produces them as [`Residuals`];
//! * a **[`ControlAdapter`]** (a diffusion backbone) declares the points it
//!   consumes residuals at, and says whether its recorded graph actually reads
//!   them;
//! * [`check_compatible`] and [`order_for`] are the only places the two are
//!   matched, **by name and by element count**, so a mismatch is an error that
//!   names the point instead of a silently-misaligned residual list.
//!
//! Ordering is the part that is easy to get wrong and impossible to notice:
//! diffusers passes `down_block_additional_residuals` as a bare tuple zipped
//! against the UNet's own `down_block_res_samples`, so a permutation of two
//! points with equal channel counts (SDXL has four 320-channel points and three
//! 640-channel ones) type-checks, runs, and produces a plausible image.
//! [`order_for`] is what makes that a named lookup instead of a zip.

use std::collections::HashMap;

/// How a residual is laid out at an injection point.
///
/// Two variants because the two backbone families differ here and nothing else
/// in this seam does: a UNet residual is a feature map, a DiT residual is a
/// token stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// NCHW feature map, batch 1 — a UNet skip or mid activation.
    Spatial { c: u32, h: u32, w: u32 },
    /// `[tokens, dim]` rows — a DiT stream (FLUX double/single blocks).
    Tokens { t: u32, d: u32 },
}

impl Layout {
    pub fn numel(&self) -> usize {
        match *self {
            Layout::Spatial { c, h, w } => (c as usize) * (h as usize) * (w as usize),
            Layout::Tokens { t, d } => (t as usize) * (d as usize),
        }
    }
}

/// One place a control residual is added, and what shape it must have there.
///
/// `name` is defined by the **backbone**, is stable across resolutions, and is
/// what a control model is matched against. SDXL's UNet names its points
/// `down.0` … `down.8` and `mid`; a FLUX backbone would name its own
/// `double.0` …, and the same `ControlNet`-shaped code would drive it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InjectionPoint {
    pub name: String,
    pub layout: Layout,
}

impl InjectionPoint {
    pub fn spatial(name: impl Into<String>, c: u32, h: u32, w: u32) -> InjectionPoint {
        InjectionPoint { name: name.into(), layout: Layout::Spatial { c, h, w } }
    }

    pub fn numel(&self) -> usize {
        self.layout.numel()
    }
}

/// A diffusion backbone that can consume control residuals.
pub trait ControlAdapter {
    /// Every injection point, **in the order the backbone's own control input
    /// list expects**. `order_for` relies on this being that order.
    fn injection_points(&self) -> Vec<InjectionPoint>;

    /// Does the *recorded graph* actually read control residuals? A backbone
    /// can describe its points (so a control model can be validated against it,
    /// or a manifest printed) without having recorded the adds.
    fn accepts_control(&self) -> bool;
}

/// A control model that produces residuals for a backbone's injection points.
///
/// Deliberately narrow: *how* a control model is evaluated is model-specific
/// (an SDXL ControlNet takes a latent + timestep + text + pooled + time_ids + a
/// conditioning image; a FLUX one takes different conditioning), and pretending
/// otherwise would be an abstraction that fits exactly one implementation. What
/// is genuinely common is the point list and the residual bundle.
pub trait ControlSource {
    fn injection_points(&self) -> Vec<InjectionPoint>;
}

/// A bundle of control residuals, keyed by injection-point name.
#[derive(Clone, Debug, Default)]
pub struct Residuals {
    order: Vec<String>,
    data: HashMap<String, Vec<f32>>,
}

impl Residuals {
    pub fn new() -> Residuals {
        Residuals::default()
    }

    /// Insert (or replace) the residual at `name`, keeping first-insert order.
    pub fn insert(&mut self, name: impl Into<String>, values: Vec<f32>) {
        let name = name.into();
        if !self.data.contains_key(&name) {
            self.order.push(name.clone());
        }
        self.data.insert(name, values);
    }

    pub fn get(&self, name: &str) -> Option<&[f32]> {
        self.data.get(name).map(|v| v.as_slice())
    }

    pub fn len(&self) -> usize {
        self.order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// Names in insertion order.
    pub fn names(&self) -> &[String] {
        &self.order
    }

    /// Every residual scaled by `s` — diffusers' `conditioning_scale`, which is
    /// a pure multiply of the zero-conv outputs and of nothing else (asserted
    /// inside `tools/goldens/controlnet_dump_reference.py`).
    pub fn scaled(&self, s: f32) -> Residuals {
        let mut r = Residuals::new();
        for n in &self.order {
            r.insert(n.clone(), self.data[n].iter().map(|v| v * s).collect());
        }
        r
    }
}

/// Do `source`'s points match `backbone`'s, by name and element count?
///
/// This is the whole compatibility question, and it is the reason the trait is
/// over *named* points: an SDXL ControlNet checkpoint dropped onto an SD-1.5
/// UNet has 12 points against 9, and a ControlNet built for a different latent
/// size has the right names and the wrong `numel`. Both are caught here, by
/// name, before a single buffer is allocated.
pub fn check_compatible(
    backbone: &dyn ControlAdapter,
    source: &dyn ControlSource,
) -> Result<(), String> {
    let want = backbone.injection_points();
    let got = source.injection_points();
    if want.len() != got.len() {
        return Err(format!(
            "control: backbone has {} injection points, source produces {}",
            want.len(),
            got.len()
        ));
    }
    let by_name: HashMap<&str, &InjectionPoint> = got.iter().map(|p| (p.name.as_str(), p)).collect();
    for p in &want {
        let Some(q) = by_name.get(p.name.as_str()) else {
            return Err(format!("control: source has no injection point named {:?}", p.name));
        };
        if q.layout != p.layout {
            return Err(format!(
                "control: {:?} is {:?} on the backbone and {:?} on the source",
                p.name, p.layout, q.layout
            ));
        }
    }
    Ok(())
}

/// `r`, ordered as `backbone`'s control input list expects.
///
/// Errors by NAME on a missing point or a length disagreement — never returns a
/// short or permuted list, which is what a `zip` over two tuples silently does.
pub fn order_for(backbone: &dyn ControlAdapter, r: &Residuals) -> Result<Vec<Vec<f32>>, String> {
    let mut out = Vec::with_capacity(r.len());
    for p in backbone.injection_points() {
        let v = r
            .get(&p.name)
            .ok_or_else(|| format!("control: no residual for injection point {:?}", p.name))?;
        if v.len() != p.numel() {
            return Err(format!(
                "control: residual {:?} is {} values, the backbone wants {} ({:?})",
                p.name,
                v.len(),
                p.numel(),
                p.layout
            ));
        }
        out.push(v.to_vec());
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// The SDXL UNet as a backbone
// ---------------------------------------------------------------------------

/// `crates/sdxlunet`'s `UNet2DConditionModel` is the first [`ControlAdapter`].
///
/// The impl lives here rather than in `crates/sdxlunet` because the trait does:
/// `unet` must not depend on this crate (this crate composes `sdxlunet::model::Rec`
/// for the trainable copy, so the dependency runs one way). Rust's orphan rule
/// permits it exactly because the trait is local.
///
/// The point names are derived from the UNet's own control input order —
/// `UNetConfig::skip_stack()` finest-first, then the mid block — so they cannot
/// drift from what `Unet::run_with_control` writes.
impl ControlAdapter for sdxlunet::Unet {
    fn injection_points(&self) -> Vec<InjectionPoint> {
        let shapes = self.control_shapes();
        shapes
            .iter()
            .enumerate()
            .map(|(k, &(c, h, w))| {
                let name =
                    if k + 1 == shapes.len() { "mid".to_string() } else { format!("down.{k}") };
                InjectionPoint::spatial(name, c, h, w)
            })
            .collect()
    }

    fn accepts_control(&self) -> bool {
        sdxlunet::Unet::accepts_control(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeBackbone(Vec<InjectionPoint>);
    impl ControlAdapter for FakeBackbone {
        fn injection_points(&self) -> Vec<InjectionPoint> {
            self.0.clone()
        }
        fn accepts_control(&self) -> bool {
            true
        }
    }
    struct FakeSource(Vec<InjectionPoint>);
    impl ControlSource for FakeSource {
        fn injection_points(&self) -> Vec<InjectionPoint> {
            self.0.clone()
        }
    }

    fn pts() -> Vec<InjectionPoint> {
        vec![
            InjectionPoint::spatial("down.0", 320, 8, 8),
            InjectionPoint::spatial("down.1", 320, 8, 8),
            InjectionPoint::spatial("mid", 1280, 2, 2),
        ]
    }

    #[test]
    fn compatible_when_names_and_shapes_agree() {
        let b = FakeBackbone(pts());
        let s = FakeSource(pts());
        check_compatible(&b, &s).expect("identical point sets are compatible");
    }

    #[test]
    fn a_shape_disagreement_names_the_point() {
        let b = FakeBackbone(pts());
        let mut other = pts();
        other[2] = InjectionPoint::spatial("mid", 1280, 4, 4);
        let e = check_compatible(&b, &FakeSource(other)).expect_err("shapes differ");
        assert!(e.contains("mid"), "{e}");
    }

    #[test]
    fn a_missing_point_is_named_not_counted() {
        let b = FakeBackbone(pts());
        let mut other = pts();
        other[1].name = "down.7".into();
        let e = check_compatible(&b, &FakeSource(other)).expect_err("names differ");
        assert!(e.contains("down.1"), "{e}");
    }

    /// The ordering property this seam exists for: two points with the SAME
    /// element count, supplied in the wrong order, must come back in the
    /// backbone's order — a `zip` would not notice.
    #[test]
    fn order_for_reorders_equal_sized_points() {
        let b = FakeBackbone(pts());
        let mut r = Residuals::new();
        r.insert("mid", vec![9.0; 1280 * 4]);
        r.insert("down.1", vec![1.0; 320 * 64]);
        r.insert("down.0", vec![0.0; 320 * 64]);
        let v = order_for(&b, &r).expect("all points present");
        assert_eq!(v.len(), 3);
        assert_eq!(v[0][0], 0.0);
        assert_eq!(v[1][0], 1.0);
        assert_eq!(v[2][0], 9.0);
    }

    #[test]
    fn order_for_rejects_a_wrong_length_residual() {
        let b = FakeBackbone(pts());
        let mut r = Residuals::new();
        r.insert("down.0", vec![0.0; 320 * 64]);
        r.insert("down.1", vec![1.0; 7]);
        r.insert("mid", vec![9.0; 1280 * 4]);
        let e = order_for(&b, &r).expect_err("wrong length");
        assert!(e.contains("down.1") && e.contains('7'), "{e}");
    }

    #[test]
    fn scaled_is_a_pure_multiply_and_keeps_order() {
        let mut r = Residuals::new();
        r.insert("a", vec![2.0, -4.0]);
        r.insert("b", vec![1.0]);
        let s = r.scaled(0.75);
        assert_eq!(s.names(), r.names());
        assert_eq!(s.get("a"), Some(&[1.5f32, -3.0][..]));
        assert_eq!(s.get("b"), Some(&[0.75f32][..]));
    }
}
