// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Loading a pretrained detector into a model that ADDS classes to it.
//!
//! Fine-tuning a COCO-pretrained `yolov8n` so it keeps its 80 classes and gains
//! an 81st needs more than the element-count-matched tensor copy a same-`nc`
//! reload gets away with: bumping `nc` changes the shape of some tensors, so a
//! plain copy silently leaves those at their random init and throws the
//! pretrained classes away.
//!
//! ## Only 6 tensors depend on `nc`
//!
//! In [`YoloConfig::yolov8n`](crate::YoloConfig::yolov8n) the class branch's
//! hidden width `cls_mid` is a FIXED 80 - it is not derived from `nc`. Walking
//! [`crate::config::YoloConfig::full_param_list`], `nc` therefore appears in
//! exactly two places per pyramid scale, both in the class branch's FINAL 1x1
//! conv (`head.{s}.cls.2`, Ultralytics' `model.22.cv3.{s}.2`):
//!
//! - `head.{s}.cls.2.weight` - `[nc, cls_mid]` (K=1), row-major, so output
//!   channel `o` owns `w[o*cls_mid .. (o+1)*cls_mid]`,
//! - `head.{s}.cls.2.bias`   - `[nc]`.
//!
//! Everything else - the whole backbone, the whole neck, the box/DFL (`reg`)
//! branch, and even the class branch's OWN two hidden convs - is byte-identical
//! whether `nc` is 80 or 81. So "add a class" is a 6-tensor operation: copy the
//! pretrained `nc_src` leading output channels in channel-for-channel and give
//! the appended ones a fresh init. No existing class index ever moves.
//!
//! ## What the appended channels are initialised to
//!
//! [`crate::init::init_params`] - the SAME initialiser a random head gets, not a
//! second scheme. That matters twice over:
//!
//! - the bias arm already implements Ultralytics' `Detect.bias_init` class
//!   prior (`b = -log((1-p)/p)`, `p = 0.01`), so an appended class starts at a
//!   ~1% prior probability and cannot out-shout 80 trained classes before it has
//!   seen a single gradient;
//! - the weight arm is a zero-mean Gaussian whose `std` is a parameter, so we
//!   pass the PRETRAINED tensor's own sample std ([`sample_std`]) rather than
//!   the from-scratch default. A new channel then has the same signal scale as
//!   its 80 trained neighbours instead of a scale picked for an untrained net.

use std::collections::HashMap;

use crate::model::Yolo;

/// What [`load_pretrained`] did, tensor by tensor. Reported rather than logged
/// so a caller (and a test) can assert on it: a fine-tune that silently copied
/// nothing is the failure mode this whole module exists to make impossible.
#[derive(Debug, Default, Clone)]
pub struct LoadReport {
    /// Tensors copied verbatim (shape unaffected by the class-count change).
    pub exact: Vec<String>,
    /// Class-final tensors whose pretrained channels were copied into a wider
    /// tensor, the remainder freshly initialised.
    pub expanded: Vec<String>,
    /// Tensors the checkpoint has but whose element count does not match and
    /// which are not a class-head expansion: `(name, model_numel, ckpt_numel)`.
    pub mismatched: Vec<(String, usize, usize)>,
    /// Model tensors absent from the checkpoint entirely (left at their init).
    pub missing: Vec<String>,
    /// Class count read off the checkpoint's own class-head tensors, if the
    /// head was expanded.
    pub src_nc: Option<u32>,
}

impl LoadReport {
    /// A one-line human summary for the CLI.
    pub fn summary(&self) -> String {
        let mut s = format!("{} tensors copied exactly", self.exact.len());
        if let Some(nc) = self.src_nc {
            s += &format!(", {} class-head tensors expanded from nc={nc}", self.expanded.len());
        }
        if !self.mismatched.is_empty() {
            s += &format!(", {} shape-mismatched (left at init)", self.mismatched.len());
        }
        if !self.missing.is_empty() {
            s += &format!(", {} missing from the checkpoint", self.missing.len());
        }
        s
    }

    /// Whether anything at all was transferred. A fine-tune whose report is
    /// empty is a from-scratch training run wearing a fine-tune's name.
    pub fn transferred_anything(&self) -> bool {
        !self.exact.is_empty() || !self.expanded.is_empty()
    }
}

/// Is `name` one of the (at most 6) tensors whose shape depends on `nc`?
/// See the module docs: only the class branch's final 1x1 conv does.
fn is_class_final(name: &str) -> bool {
    name.starts_with("head.") && (name.ends_with(".cls.2.weight") || name.ends_with(".cls.2.bias"))
}

/// Sample standard deviation of `v` (0 for fewer than 2 elements). Used to give
/// an appended class channel the same signal scale as the pretrained ones.
pub fn sample_std(v: &[f32]) -> f32 {
    if v.len() < 2 {
        return 0.0;
    }
    let n = v.len() as f32;
    let mean = v.iter().sum::<f32>() / n;
    (v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / (n - 1.0)).sqrt()
}

/// Copy `src`'s pretrained channels into `model`, widening the class head when
/// `model` has more classes than the checkpoint.
///
/// Every tensor whose element count matches is copied verbatim. The class-final
/// tensors (module docs) are copied channel-for-channel into the wider tensor
/// and the appended channels are initialised by [`crate::init::init_params`]
/// with the pretrained tensor's own std, deterministically for `seed`.
///
/// Class indices are APPEND-only: pretrained class `c` is still class `c`.
pub fn load_pretrained(model: &Yolo, src: &HashMap<String, Vec<f32>>, seed: u64) -> LoadReport {
    let mut rep = LoadReport::default();
    let cls_mid = model.cfg.cls_mid as usize;
    let nc = model.cfg.nc as usize;

    for name in <Yolo as model::Model>::param_names(model) {
        let Some(w) = src.get(&name) else {
            rep.missing.push(name);
            continue;
        };
        let want = model.ps.numel(&name);
        if w.len() == want {
            model.write_weight(&name, w);
            rep.exact.push(name);
            continue;
        }
        // Shapes differ. The only legitimate reason is an added class.
        if is_class_final(&name) {
            // Per-channel width: `cls_mid` for the weight, 1 for the bias.
            let per = if name.ends_with(".weight") { cls_mid } else { 1 };
            let src_nc = w.len() / per;
            if w.len() % per == 0 && src_nc < nc && want == nc * per {
                let mut merged = vec![0.0f32; want];
                // Pretrained channels, channel-for-channel (row-major [nc, per]).
                merged[..w.len()].copy_from_slice(w);
                // Appended channels: the crate's OWN init scheme, at the
                // pretrained tensor's scale (see the module docs).
                let extra = want - w.len();
                let fresh = crate::init::init_params(&[(name.clone(), extra)], seed, sample_std(w));
                merged[w.len()..].copy_from_slice(&fresh[&name]);
                model.write_weight(&name, &merged);
                rep.expanded.push(name);
                rep.src_nc = Some(src_nc as u32);
                continue;
            }
        }
        rep.mismatched.push((name, want, w.len()));
    }
    rep
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::YoloConfig;
    use model::Model;

    /// Build a `tiny(nc)` model at a small input, plus the init map it was built
    /// from. `tiny`'s `cls_mid` is `max(32, nc)`, so nc=3 and nc=4 share the
    /// exact same `cls_mid` (32) - the SAME class-head isolation the canonical
    /// `yolov8n` config has at nc=80/81, reproduced at gradcheck size.
    fn tiny_model(nc: u32, seed: u64) -> (Yolo, HashMap<String, Vec<f32>>, YoloConfig) {
        let mut cfg = YoloConfig::tiny(nc);
        cfg.input = 64;
        let init = crate::init::init_model(&cfg, seed);
        (Yolo::new(cfg.clone(), 1, 0, &init), init, cfg)
    }

    /// The class-head isolation claim itself, checked against the REAL canonical
    /// config rather than trusted: bumping `nc` 80 -> 81 must change exactly 6
    /// tensors, all of them `head.{s}.cls.2.{weight,bias}`.
    #[test]
    fn only_the_class_final_tensors_depend_on_nc() {
        let a = YoloConfig::yolov8n();
        let mut b = YoloConfig::yolov8n();
        b.nc = 81;
        assert_eq!(a.cls_mid, b.cls_mid, "cls_mid must be fixed, not derived from nc");
        let pa = a.full_param_list();
        let pb = b.full_param_list();
        assert_eq!(pa.len(), pb.len(), "adding a class must not add or remove tensors");
        let changed: Vec<String> = pa
            .iter()
            .zip(&pb)
            .filter(|((na, ca), (nb, cb))| {
                assert_eq!(na, nb, "tensor ORDER must not change");
                ca != cb
            })
            .map(|((n, _), _)| n.clone())
            .collect();
        assert_eq!(changed.len(), 6, "expected exactly 6 nc-dependent tensors, got {changed:?}");
        for n in &changed {
            assert!(is_class_final(n), "{n} changed shape but is not a class-final tensor");
        }
    }

    /// The expansion contract: every unaffected tensor arrives bit-for-bit, the
    /// pretrained class channels arrive bit-for-bit at their ORIGINAL indices,
    /// and the appended class gets a fresh, non-degenerate init.
    #[test]
    fn class_head_expansion_preserves_pretrained_classes_and_freshly_inits_the_new_one() {
        let (_src_model, src_init, src_cfg) = tiny_model(3, 7);
        let (dst, _dst_init, dst_cfg) = tiny_model(4, 11);
        assert_eq!(src_cfg.cls_mid, dst_cfg.cls_mid, "test premise: cls_mid unchanged by the nc bump");
        let cls_mid = dst_cfg.cls_mid as usize;

        let rep = load_pretrained(&dst, &src_init, 99);
        assert!(rep.mismatched.is_empty(), "unexpected shape mismatches: {:?}", rep.mismatched);
        assert!(rep.missing.is_empty(), "unexpected missing tensors: {:?}", rep.missing);
        assert_eq!(rep.expanded.len(), 6, "expected 6 expanded class-head tensors, got {:?}", rep.expanded);
        assert_eq!(rep.src_nc, Some(3));

        // Unaffected tensors: bit-for-bit.
        for name in <Yolo as Model>::param_names(&dst) {
            if is_class_final(&name) {
                continue;
            }
            assert_eq!(dst.read_weight(&name), src_init[&name], "{name} was not copied verbatim");
        }

        for s in 0..3 {
            // Weight: the 3 pretrained channels land at channels 0..3 unchanged.
            let wn = format!("head.{s}.cls.2.weight");
            let got = dst.read_weight(&wn);
            let want = &src_init[&wn];
            assert_eq!(got.len(), 4 * cls_mid);
            assert_eq!(&got[..3 * cls_mid], &want[..], "{wn}: pretrained channels must be untouched");
            let new_ch = &got[3 * cls_mid..];
            assert!(new_ch.iter().any(|v| *v != 0.0), "{wn}: the appended channel is degenerate (all zero)");
            assert!(new_ch.iter().all(|v| v.is_finite()), "{wn}: non-finite fresh init");
            // Fresh init is at the pretrained tensor's own scale, not an
            // arbitrary one (within a generous factor for a 32-sample std).
            let (s_new, s_old) = (sample_std(new_ch), sample_std(want));
            assert!(s_new > 0.25 * s_old && s_new < 4.0 * s_old, "{wn}: fresh std {s_new} vs pretrained {s_old}");

            // Bias: 3 pretrained values kept; the appended one is the class prior.
            let bn = format!("head.{s}.cls.2.bias");
            let got = dst.read_weight(&bn);
            assert_eq!(got.len(), 4);
            assert_eq!(&got[..3], &src_init[&bn][..]);
            let prior = -((1.0f32 - 0.01) / 0.01).ln();
            assert!((got[3] - prior).abs() < 1e-5, "{bn}: appended bias {} != class prior {prior}", got[3]);
        }
    }

    /// Run `steps` real detection training steps on a fixed synthetic image.
    fn train_a_few_steps_wd(m: &Yolo, cls: u32, steps: u32, wd: f32) {
        use crate::model::{GtBox, LossMode};
        let side = m.cfg.input as usize;
        // A deterministic non-uniform image: BN statistics must have something
        // to move toward, or "unchanged running stats" would prove nothing.
        let img: Vec<f32> = (0..3 * side * side).map(|i| ((i % 37) as f32 / 37.0) * 0.8 + 0.1).collect();
        let gts = vec![GtBox { img: 0, cls, cx: 0.5, cy: 0.5, w: 0.4, h: 0.6 }];
        m.set_mode(LossMode::Detection);
        m.set_eval(false);
        m.set_update_running(true);
        m.set_image(&img);
        m.set_targets(&gts);
        for step in 0..steps {
            m.zero_grads();
            m.forward();
            m.backward();
            m.adamw_step(step + 1, 1e-2, wd, Some(1.0), 1.0);
            m.poll_wait();
        }
    }

    fn train_a_few_steps(m: &Yolo, cls: u32, steps: u32) {
        train_a_few_steps_wd(m, cls, steps, 1e-2)
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// The freeze contract at model level, proved the only way that counts:
    /// train for real, then compare BITS. Covers both drift routes - the
    /// optimiser (conv weights, BN `gamma`/`beta`) and the forward pass (BN
    /// `run_mean`/`run_var`, which no gradient ever touches).
    #[test]
    fn freezing_the_backbone_leaves_every_trunk_tensor_bit_for_bit_unchanged() {
        let (mut m, init, _) = tiny_model(4, 7);
        let froze = m.freeze_backbone();
        assert!(froze > 0, "freeze_backbone froze nothing");
        m.freeze_reg_head();

        train_a_few_steps(&m, 3, 5);

        let mut checked = 0usize;
        for name in <Yolo as Model>::param_names(&m) {
            let is_trunk = name.starts_with("backbone.") || name.starts_with("neck.");
            let is_reg = name.starts_with("head.") && name.contains(".reg.");
            if !(is_trunk || is_reg) {
                continue;
            }
            assert_eq!(bits(&m.read_weight(&name)), bits(&init[&name]), "frozen tensor {name} moved during training");
            checked += 1;
        }
        assert!(checked > 100, "expected to check the whole trunk, only saw {checked} tensors");

        // ... and training actually happened: the class head DID move.
        let moved = (0..3).any(|s| {
            let n = format!("head.{s}.cls.2.weight");
            bits(&m.read_weight(&n)) != bits(&init[&n])
        });
        assert!(moved, "nothing trained - the freeze test would pass vacuously");
    }

    /// The class-gate contract: training an ADDED class must leave the classes
    /// that have no data in this dataset bit-for-bit intact, rather than driving
    /// them to "never fire" (which is what the unmasked BCE does to every class
    /// absent from a fine-tune set - see `Yolo::train_only_classes`).
    #[test]
    fn training_only_the_new_class_leaves_the_other_classes_bit_for_bit_intact() {
        let (mut m, init, cfg) = tiny_model(4, 7);
        m.freeze_backbone();
        m.freeze_reg_head();
        // Class 3 is the "added" one; 0..2 stand in for the pretrained classes.
        m.train_only_classes(&[3]);
        let cls_mid = cfg.cls_mid as usize;

        // wd = 0: decoupled weight decay is not a gradient and would shrink the
        // gated rows anyway (documented on `train_only_classes`).
        train_a_few_steps_wd(&m, 3, 5, 0.0);

        for s in 0..3 {
            let wn = format!("head.{s}.cls.2.weight");
            let (got, was) = (m.read_weight(&wn), &init[&wn]);
            assert_eq!(
                bits(&got[..3 * cls_mid]),
                bits(&was[..3 * cls_mid]),
                "{wn}: a gated class's weights moved"
            );
            assert_ne!(bits(&got[3 * cls_mid..]), bits(&was[3 * cls_mid..]), "{wn}: the trained class did not move");

            let bn = format!("head.{s}.cls.2.bias");
            let (got, was) = (m.read_weight(&bn), &init[&bn]);
            assert_eq!(bits(&got[..3]), bits(&was[..3]), "{bn}: a gated class's bias moved");
            assert_ne!(got[3].to_bits(), was[3].to_bits(), "{bn}: the trained class's bias did not move");
        }
    }

    /// The full preservation configuration, the only one where "the pretrained
    /// classes still work" is a PROOF rather than a measurement: with the trunk,
    /// the box head and the class branch's SHARED hidden convs all frozen and
    /// every pretrained class gated off, an added class leaves literally every
    /// pretrained number untouched - so the old classes cannot have changed
    /// behaviour, because nothing they own or read has changed.
    ///
    /// Gating the classes alone is NOT enough and this is the difference:
    /// `training_only_the_new_class_...` above keeps their own weights intact,
    /// but they READ the shared hidden convs, so without freezing those their
    /// detections collapse regardless.
    #[test]
    fn the_preservation_config_leaves_every_pretrained_number_bit_identical() {
        let (mut m, init, cfg) = tiny_model(4, 7);
        m.freeze_backbone();
        m.freeze_reg_head();
        assert!(m.freeze_cls_hidden() > 0, "freeze_cls_hidden froze nothing");
        m.train_only_classes(&[3]);
        let cls_mid = cfg.cls_mid as usize;

        train_a_few_steps_wd(&m, 3, 5, 0.0);

        for name in <Yolo as Model>::param_names(&m) {
            let (got, was) = (m.read_weight(&name), &init[&name]);
            if is_class_final(&name) {
                // Only the appended channel may move.
                let keep = if name.ends_with(".weight") { 3 * cls_mid } else { 3 };
                assert_eq!(bits(&got[..keep]), bits(&was[..keep]), "{name}: a pretrained class channel moved");
                assert_ne!(bits(&got[keep..]), bits(&was[keep..]), "{name}: the added class did not train");
            } else {
                assert_eq!(bits(&got), bits(was), "{name} moved under the preservation config");
            }
        }
    }

    /// Without the gate, those same classes DO get destroyed - the regression
    /// this mechanism exists to prevent, pinned so it cannot silently return.
    #[test]
    fn without_the_class_gate_absent_classes_are_driven_down() {
        let (mut m, init, _) = tiny_model(4, 7);
        m.freeze_backbone();
        m.freeze_reg_head();
        train_a_few_steps_wd(&m, 3, 5, 0.0);
        // Every class absent from the data is pushed toward "never fire", which
        // shows up as its bias dropping.
        for s in 0..3 {
            let bn = format!("head.{s}.cls.2.bias");
            let (got, was) = (m.read_weight(&bn), &init[&bn]);
            for c in 0..3 {
                assert!(got[c] < was[c], "{bn}[{c}]: expected an absent class to be driven down, {} -> {}", was[c], got[c]);
            }
        }
    }

    /// A same-`nc` reload must still be the plain verbatim copy it always was -
    /// the expansion path must not perturb the case it generalises.
    #[test]
    fn same_class_count_load_is_a_verbatim_copy() {
        let (_a, src_init, _) = tiny_model(3, 7);
        let (dst, _, _) = tiny_model(3, 11);
        let rep = load_pretrained(&dst, &src_init, 99);
        assert!(rep.expanded.is_empty(), "same nc must not expand anything");
        assert!(rep.mismatched.is_empty() && rep.missing.is_empty());
        assert!(rep.transferred_anything());
        for name in <Yolo as Model>::param_names(&dst) {
            assert_eq!(dst.read_weight(&name), src_init[&name], "{name} differs after a same-nc load");
        }
    }
}
