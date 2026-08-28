// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Forward parity for SAM 2's VIDEO path - the temporal memory bank - against
//! goldens dumped from the official `facebookresearch/sam2` package running the
//! released `sam2.1_hiera_*.pt` (`tools/goldens/sam2_video_dump_reference.py`).
//!
//! ```text
//! testdata/sam2/hiera-tiny/
//!   memattn.safetensors    MemoryAttention on seeded random inputs, per layer
//!   memenc.safetensors     MemoryEncoder on seeded random inputs, per stage
//!   video.safetensors      end-to-end propagation over a 6-frame clip
//!   video_manifest.json    the reference config + the weight-health report
//! ```
//!
//! # Every gate is cosine AND relative L2
//!
//! [`Table`] is used, never [`Report`]: cosine alone cannot see a scale error,
//! and it is nearly blind to a small systematic one. Temporal code is exactly
//! where that bites - a dropped clamp on a neighbouring model scored cosine
//! 0.999999884 (above a 0.999999 floor) while its relative L2 of 4.8e-4 caught
//! it. So both bounds are asserted on every tap here, and both were checked to
//! FIRE by mutating the port (see the commit message for the per-gate results).
//!
//! Each test SKIPS ITSELF when its fixture is absent, through
//! `brain_testutil::skip`, so `BRAIN_REQUIRE_FIXTURES=1` turns a skip into a
//! failure and a green run under that flag means every comparison really ran.

use std::path::{Path, PathBuf};

use sam2::{Sam2, Sam2Config, Scope, Tracker};

use brain_testutil::parity::{load, Table};
use brain_testutil::testdata_path as testdata;

/// Cosine floor. The reference dump is fp32 on CPU and brain's forward is fp32,
/// so anything below this is a real disagreement, not accumulated rounding.
const COS: f64 = 0.999999;
/// Relative-L2 ceiling, the bound cosine cannot see. Deliberately tighter than
/// the accumulated fp32 noise of a 4-layer attention stack, and loose enough
/// that a reordered reduction does not trip it.
const REL: f64 = 1e-5;
/// The end-to-end propagation compounds six Hiera passes, four memory-attention
/// layers per frame and a memory encoder whose output feeds the NEXT frame, so
/// its budget is wider than a single submodule's - but it is still a rel_l2
/// bound, not cosine alone, and it is set from the MEASURED worst row (2.5e-4
// perf-number: a reviewed tolerance-headroom multiplier on a fixed test bound, not a measured runtime speedup
/// on the object pointer) with about 2x headroom, not at a round number that
/// would wave a real defect through.
const REL_E2E: f64 = 5e-4;

/// `torch.Tensor.to(bfloat16)`: keep the top 16 bits, round to nearest even.
fn to_bf16(x: f32) -> f32 {
    let b = x.to_bits();
    if x.is_nan() || x.is_infinite() {
        return x;
    }
    f32::from_bits((b + 0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
}

fn checkpoint_path(base: &Path, dir: &str, ckpt: &str) -> Option<PathBuf> {
    let local = base.join(ckpt);
    if local.exists() {
        return Some(local);
    }
    let store = PathBuf::from(brain_testutil::model_dir(&format!("facebook/sam2.1-{dir}"))?).join(ckpt);
    store.exists().then_some(store)
}

/// The whole checkpoint, at [`Scope::Video`] - which is itself the two-way
/// coverage gate: an unmatched key on either side is an error naming it, so a
/// module cannot silently stay at its random init.
fn build(base: &Path, dir: &str, ckpt: &str, cfg: Sam2Config) -> Option<Sam2> {
    let pt = checkpoint_path(base, dir, ckpt)?;
    let raw = checkpoint::torchpt::read(pt.to_str().unwrap()).expect("read .pt");
    let tensors: Vec<(String, Vec<usize>, Vec<f32>)> =
        raw.into_iter().filter_map(|t| t.name.strip_prefix("model.").map(|n| (n.to_string(), t.shape, t.data))).collect();
    let (weights, rep) = sam2::import_scoped(tensors, &cfg, Scope::Video).expect("import at Scope::Video");
    println!("  import: {} source, {} imported, {} skipped", rep.source, rep.imported, rep.skipped_video);
    assert_eq!(rep.skipped_video, 0, "Scope::Video must skip nothing");
    assert_eq!(rep.imported + 1, rep.source, "two-way coverage must close exactly");
    let gpu = gpu_core::testgpu::dev(sam2::PIPELINES);
    Some(Sam2::new_video(gpu, cfg, &weights))
}

// ===========================================================================
// weight health: the checkpoint's video half is TRAINED, not default init
// ===========================================================================

/// Excess-kurtosis-and-range test for "this tensor is at PyTorch default init".
///
/// `nn.Linear` / `nn.Conv2d` initialise from `U(-k, k)` with `k = 1/sqrt(fan_in)`:
/// unadjusted kurtosis 1.8, and for any tensor of real size `max|w|` lands
/// essentially AT `k`. A trained tensor is heavier-tailed and its max is a
/// fraction of `k`. Both conditions must hold for a tensor to be called
/// untrained, because either alone has honest false positives.
///
/// This exists because a sibling port nearly shipped an adapter whose published
/// weights were all at exactly this distribution - dropped at load by a key-name
/// mismatch under `strict=False`, and silent. `Scope::Video` makes the key-name
/// half impossible; this makes the values half checked.
fn looks_like_default_init(w: &[f32], fan_in: usize) -> bool {
    if w.len() < 256 || fan_in == 0 {
        return false;
    }
    let n = w.len() as f64;
    let mean = w.iter().map(|v| *v as f64).sum::<f64>() / n;
    let var = w.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / n;
    if var <= 0.0 {
        return true; // an all-constant tensor is not a trained one either
    }
    let sd = var.sqrt();
    let kurt = w.iter().map(|v| ((*v as f64 - mean) / sd).powi(4)).sum::<f64>() / n;
    let max_abs = w.iter().fold(0.0f64, |a, v| a.max((*v as f64).abs()));
    let bound = 1.0 / (fan_in as f64).sqrt();
    let ratio = max_abs / bound;
    (0.97..=1.03).contains(&ratio) && kurt < 2.0
}

#[test]
fn video_weights_are_trained_not_at_default_init() {
    let base = testdata("sam2/hiera-tiny");
    let cfg = Sam2Config::hiera_tiny();
    let Some(pt) = checkpoint_path(&base, "hiera-tiny", "sam2.1_hiera_tiny.pt") else {
        return brain_testutil::skip("sam2/hiera-tiny: need sam2.1_hiera_tiny.pt");
    };
    let raw = checkpoint::torchpt::read(pt.to_str().unwrap()).expect("read .pt");
    let tensors: Vec<(String, Vec<usize>, Vec<f32>)> =
        raw.into_iter().filter_map(|t| t.name.strip_prefix("model.").map(|n| (n.to_string(), t.shape, t.data))).collect();
    let (w, _) = sam2::import_scoped(tensors, &cfg, Scope::Video).expect("import");

    let mut checked = 0usize;
    let mut suspicious: Vec<String> = Vec::new();
    for (name, shape) in cfg.video_tensor_manifest() {
        let (_, data) = &w[&name];
        // torch's `fan_in`: `weight[1]` for Linear, `Cin/groups * kh * kw` for
        // Conv2d. A 1-d tensor (a norm gain, a bias) has none.
        let fan_in = match shape.len() {
            4 => shape[1] * shape[2] * shape[3],
            2 => shape[1],
            _ => 0,
        };
        if fan_in == 0 || data.len() < 256 {
            continue;
        }
        checked += 1;
        if looks_like_default_init(data, fan_in) {
            suspicious.push(name);
        }
    }
    assert!(checked >= 50, "only {checked} video tensors were large enough to test");
    assert!(suspicious.is_empty(), "{} video tensor(s) are at PyTorch default init: {suspicious:?}", suspicious.len());
    println!("  weight health: {checked} video tensors, none at default init");

    // The detector itself must fire, or it proves nothing: a real `U(-k, k)`
    // draw at k = 1/sqrt(fan_in) has to be flagged.
    let fan_in = 512usize;
    let k = 1.0 / (fan_in as f64).sqrt();
    let mut s: u64 = 0x2545F491_4F6CDD1D;
    let fake: Vec<f32> = (0..8192)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = (s >> 11) as f64 / (1u64 << 53) as f64;
            ((2.0 * u - 1.0) * k) as f32
        })
        .collect();
    assert!(looks_like_default_init(&fake, fan_in), "the default-init detector does not fire on default init");
}

// ===========================================================================
// submodule parity
// ===========================================================================

#[test]
fn memory_attention_parity() {
    let base = testdata("sam2/hiera-tiny");
    let path = base.join("memattn.safetensors");
    if !path.exists() {
        return brain_testutil::skip(&format!("{} absent - regenerate with tools/goldens/sam2_video_dump_reference.py", path.display()));
    }
    let cfg = Sam2Config::hiera_tiny();
    let Some(m) = build(&base, "hiera-tiny", "sam2.1_hiera_tiny.pt", cfg.clone()) else {
        return brain_testutil::skip("sam2/hiera-tiny: need sam2.1_hiera_tiny.pt");
    };
    let g = load(&path);
    let d = cfg.d_model;
    let md = cfg.mem_dim;
    let tq = cfg.image_embedding_size().pow(2);
    let n_ptr = g["num_obj_ptr_tokens"].data[0] as u32;
    let tk = (g["memory"].data.len() / md as usize) as u32;
    assert_eq!(g["curr"].data.len(), (tq * d) as usize, "golden curr is [H*W, d_model]");

    let curr = m.gpu.storage_init("t_curr", &g["curr"].data);
    let curr_pos = m.gpu.storage_init("t_curr_pos", &g["curr_pos"].data);
    let memory = m.gpu.storage_init("t_mem", &g["memory"].data);
    let memory_pos = m.gpu.storage_init("t_mem_pos", &g["memory_pos"].data);

    let consts = sam2::VideoConsts::read(&m);
    let mut taps = Vec::new();
    let out = m.memory_attention(&curr, &curr_pos, &memory, &memory_pos, tq, tk, n_ptr, &consts, &mut taps);

    let mut t = Table::new(COS, REL);
    for (i, tap) in taps.iter().enumerate() {
        let want = &g[&format!("memattn_layer{i}_out")];
        t.check(&format!("memattn_layer{i}_out"), &m.gpu.read(tap, want.data.len()), &want.data);
    }
    let want = &g["memattn_out"];
    t.check("memattn_out", &m.gpu.read(&out, want.data.len()), &want.data);
    t.print();
    println!("  [sam2 memory attention] {} taps, worst cosine {:.10} at {}, worst rel_l2 {:.3e} at {}",
        t.rows.len(), t.worst_cosine().1, t.worst_cosine().0, t.worst_rel_l2().1, t.worst_rel_l2().0);
    t.assert_clean();
}

#[test]
fn memory_encoder_parity() {
    let base = testdata("sam2/hiera-tiny");
    let path = base.join("memenc.safetensors");
    if !path.exists() {
        return brain_testutil::skip(&format!("{} absent", path.display()));
    }
    let cfg = Sam2Config::hiera_tiny();
    let Some(m) = build(&base, "hiera-tiny", "sam2.1_hiera_tiny.pt", cfg.clone()) else {
        return brain_testutil::skip("sam2/hiera-tiny: need sam2.1_hiera_tiny.pt");
    };
    let g = load(&path);
    // The scale/bias the tracker applies before calling the encoder is part of
    // the contract, so it is asserted rather than assumed.
    assert_eq!(g["sigmoid_scale"].data[0], cfg.sigmoid_scale_for_mem_enc);
    assert_eq!(g["sigmoid_bias"].data[0], cfg.sigmoid_bias_for_mem_enc);

    let pix = m.gpu.storage_init("t_pix", &g["pix_feat"].data);
    let mask = m.gpu.storage_init("t_mask", &g["mask_for_mem"].data);
    let consts = sam2::VideoConsts::read(&m);
    let mut taps = Vec::new();
    let feats = m.memory_encoder(&pix, &mask, &consts, &mut taps);

    let mut t = Table::new(COS, REL);
    let names = ["mask_downsampled", "pix_feat_proj", "fuser_layer0_out", "fuser_layer1_out"];
    for (tap, name) in taps.iter().zip(names) {
        let want = &g[name];
        t.check(name, &m.gpu.read(tap, want.data.len()), &want.data);
    }
    let want = &g["maskmem_features"];
    t.check("maskmem_features", &m.gpu.read(&feats, want.data.len()), &want.data);

    // The memory's positional encoding is a CONSTANT table per grid, built on
    // the host - so it is gated here rather than dumped from a device buffer.
    let grid = cfg.image_embedding_size();
    let host = sam2::hostpe::sine(cfg.mem_pos_sine_num_pos_feats, cfg.pos_sine_temperature, grid, grid);
    t.check("maskmem_pos_enc", &host, &g["maskmem_pos_enc"].data);
    t.print();
    println!("  [sam2 memory encoder] {} taps, worst cosine {:.10} at {}, worst rel_l2 {:.3e} at {}",
        t.rows.len(), t.worst_cosine().1, t.worst_cosine().0, t.worst_rel_l2().1, t.worst_rel_l2().0);
    t.assert_clean();
}

/// `get_1d_sine_pe(Δt / t_diff_max, d_model)` then `obj_ptr_tpos_proj` - the
/// object pointer's temporal encoding, which is host arithmetic in this port.
#[test]
fn object_pointer_temporal_encoding_parity() {
    let base = testdata("sam2/hiera-tiny");
    let path = base.join("video.safetensors");
    if !path.exists() {
        return brain_testutil::skip(&format!("{} absent", path.display()));
    }
    let cfg = Sam2Config::hiera_tiny();
    let Some(m) = build(&base, "hiera-tiny", "sam2.1_hiera_tiny.pt", cfg.clone()) else {
        return brain_testutil::skip("sam2/hiera-tiny: need sam2.1_hiera_tiny.pt");
    };
    let g = load(&path);
    let d = cfg.d_model as usize;
    let md = cfg.mem_dim as usize;
    let t_max = g["objptr_t_diff_max"].data[0];
    assert_eq!(t_max, (cfg.max_obj_ptrs_in_encoder - 1) as f32);

    let mut sine = Vec::new();
    for &p in g["objptr_tpos_input"].data.iter() {
        sine.extend_from_slice(&sam2::hostpe::sine_1d(p / t_max, cfg.d_model, cfg.pos_sine_temperature));
    }
    let w = m.ps.read_weight(&m.gpu, "obj_ptr_tpos_proj.weight");
    let b = m.ps.read_weight(&m.gpu, "obj_ptr_tpos_proj.bias");
    let proj = sam2::hostpe::linear_rows(&sine, &w, &b, d, md);

    let mut t = Table::new(COS, REL);
    t.check("objptr_tpos_sine", &sine, &g["objptr_tpos_sine"].data);
    t.check("objptr_tpos_proj", &proj, &g["objptr_tpos_proj"].data);
    t.print();
    println!("  [sam2 object-pointer temporal encoding] {} taps, worst cosine {:.10} at {}, worst rel_l2 {:.3e} at {}",
        t.rows.len(), t.worst_cosine().1, t.worst_cosine().0, t.worst_rel_l2().1, t.worst_rel_l2().0);
    t.assert_clean();
}

// ===========================================================================
// end to end: a point on frame 0 propagated through the clip
// ===========================================================================

#[test]
fn video_propagation_parity() {
    let base = testdata("sam2/hiera-tiny");
    let path = base.join("video.safetensors");
    if !path.exists() {
        return brain_testutil::skip(&format!("{} absent", path.display()));
    }
    let cfg = Sam2Config::hiera_tiny();
    let Some(m) = build(&base, "hiera-tiny", "sam2.1_hiera_tiny.pt", cfg.clone()) else {
        return brain_testutil::skip("sam2/hiera-tiny: need sam2.1_hiera_tiny.pt");
    };
    let g = load(&path);
    let n_frames = g["num_frames"].data[0] as usize;
    let prompt_frame = g["prompt_frame"].data[0] as usize;
    let side = cfg.image_size as usize;
    let per = 3 * side * side;
    assert_eq!(g["images"].data.len(), n_frames * per, "golden holds one normalized frame each");

    // The prompt is in SOURCE pixels; the reference normalises by the video
    // size and rescales to `image_size`. Here the clip IS `image_size` square.
    let (px, py) = (g["point_xy"].data[0], g["point_xy"].data[1]);
    let (vh, vw) = (g["video_hw"].data[0], g["video_hw"].data[1]);
    let prompt = sam2::Prompt {
        coords: vec![(px / vw * cfg.image_size as f32, py / vh * cfg.image_size as f32)],
        labels: vec![g["point_label"].data[0]],
        mask_lowres: None,
        multimask_output: true,
    };

    // The fixture MUST contain a frame the reference calls occluded, or the
    // `object_score <= 0` branch (the `no_obj_embed_spatial` add) is never
    // executed and no gate here can see a change to it. That is not a
    // hypothetical: a mutation deleting that branch survived an earlier,
    // occlusion-free version of this clip.
    let occluded: Vec<usize> =
        (0..n_frames).filter(|f| g[&format!("f{f}_object_score_logits")].data[0] <= 0.0).collect();
    assert!(
        !occluded.is_empty(),
        "the fixture clip never occludes the subject, so the object-absent path is untested -          regenerate with tools/goldens/sam2_video_dump_reference.py --occlude <frames>"
    );
    println!("  fixture: {n_frames} frames, reference reports the object absent on {occluded:?}");

    let mut tr = Tracker::new(&m, n_frames, 1);
    let mut t = Table::new(COS, REL_E2E);
    for f in 0..n_frames {
        let img = m.gpu.storage_init("t_frame", &g["images"].data[f * per..(f + 1) * per]);
        let enc = m.encode(&img);
        let step = if f == prompt_frame { tr.prompt(&enc, f, &prompt) } else { tr.track(&enc, f) };
        assert_eq!(step.is_cond, g[&format!("f{f}_is_cond")].data[0] > 0.5, "frame {f}: conditioning flag");
        assert_eq!(
            step.object_score <= 0.0,
            g[&format!("f{f}_object_score_logits")].data[0] <= 0.0,
            "frame {f}: the object-present decision disagrees with the reference, which is a \
             DIFFERENT memory entry (no_obj_embed_spatial), not a small numeric difference"
        );

        let want = &g[&format!("f{f}_low_res_masks")];
        t.check(&format!("f{f}_low_res_masks"), &m.gpu.read(&step.low_res_mask, want.data.len()), &want.data);
        let want = &g[&format!("f{f}_obj_ptr")];
        t.check(&format!("f{f}_obj_ptr"), &m.gpu.read(&step.decoded.obj_ptr, want.data.len()), &want.data);
        let want = &g[&format!("f{f}_object_score_logits")];
        t.check(&format!("f{f}_object_score"), &[step.object_score], &want.data);
        let mem = tr.memory(f).expect("every tracked frame records a memory");
        // The memory encoder's FP32 output. The reference predictor stores this
        // cast to bfloat16 (`_run_single_frame_inference`) purely to shrink the
        // inference state, so gating against the STORED value would charge this
        // port ~1.6e-3 of relative error that is the reference's own
        // quantisation, not a disagreement. The bf16 claim is checked below
        // rather than taken on trust.
        let got = m.gpu.read(&mem.features, g[&format!("f{f}_maskmem_features_fp32")].data.len());
        let want = &g[&format!("f{f}_maskmem_features_fp32")];
        t.check(&format!("f{f}_maskmem_features"), &got, &want.data);
        let stored = &g[&format!("f{f}_maskmem_features")];
        let rounded: Vec<f32> = got.iter().map(|v| to_bf16(*v)).collect();
        t.check(&format!("f{f}_maskmem_features_bf16"), &rounded, &stored.data);
        // On an occluded frame the memory carries `no_obj_embed_spatial` on top
        // of the encoder's output; on every other frame it does not. The golden
        // dumps both sides of that add, so this asserts the branch ran in the
        // reference exactly where it ran here - the encoder tap alone cannot
        // see it, and an earlier fixture with no occlusion could not either.
        let pre = &g[&format!("f{f}_maskmem_features_pre_no_obj")];
        let delta = pre.data.iter().zip(&want.data).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        if step.object_score <= 0.0 {
            assert!(delta > 1e-3, "frame {f} is occluded but the reference's memory carries no no_obj_embed_spatial");
        } else {
            assert_eq!(delta, 0.0, "frame {f} is visible but the reference's memory carries a no-object term");
        }
        // The encoder's own two inputs, so a future divergence here is
        // attributable to the mask or to the FPN feature without a bisect.
        let want = &g[&format!("f{f}_memenc_pix_feat")];
        t.check(&format!("f{f}_memenc_pix_feat"), &m.gpu.read(&enc.fpn[2], want.data.len()), &want.data);
        let want = &g[&format!("f{f}_maskmem_pos_enc")];
        t.check(&format!("f{f}_maskmem_pos_enc"), &m.gpu.read(&mem.pos_enc, want.data.len()), &want.data);
    }
    t.print();
    println!("  [sam2 video propagation] {} taps, worst cosine {:.10} at {}, worst rel_l2 {:.3e} at {}",
        t.rows.len(), t.worst_cosine().1, t.worst_cosine().0, t.worst_rel_l2().1, t.worst_rel_l2().0);
    t.assert_clean();

    // The clip's disc MOVES, so the tracked mask must move with it: a port that
    // ignored the memory and re-segmented each frame from nothing would still
    // pass a per-tap comparison on a static clip. The reference's own masks are
    // the witness.
    let area = |f: usize| -> usize {
        g[&format!("f{f}_video_res_masks")].data.iter().filter(|v| **v > 0.0).count()
    };
    assert!(area(0) > 1000, "the reference clip's frame 0 mask is empty - the fixture is wrong, not the port");
    assert!(area(n_frames - 1) > 1000, "the reference lost the object by the last frame");
}
