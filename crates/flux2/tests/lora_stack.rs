// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec gates for folding **several** LoRA adapters into one FLUX.2
//! generation - a face-identity adapter plus a style adapter in the same run.
//!
//! [`flux2::lora::fold_adapters`] is the seam every builder goes through
//! (`Pipeline::build_sized` and friends hand it exactly the `&[AdapterSpec]`
//! they were given), so everything a stacked run does to the weights can be
//! asserted here, on synthetic tiny tensors, without a single byte of real
//! checkpoint.
//!
//! Three properties, in the order they matter:
//!
//! 1. **Zero adapters is today's unadapted build, byte for byte.** Every
//!    generation with no `--adapter` takes this path, so it must not clone,
//!    rebuild or perturb the tensor map at all.
//! 2. **One adapter is today's single-adapter build, byte for byte** - for
//!    both families (brain's own trained container and a third-party
//!    ai-toolkit/ComfyUI `.safetensors`), measured against the pre-existing
//!    single-adapter entry points.
//! 3. **N adapters fold in order, each at its own strength** - asserted on the
//!    resulting weight values, over both overlapping and distinct targets.

use flux2::lora::{fold_adapters, fold_external_adapter, save_adapter, LoraAdapter, LoraCfg};
use flux2::modelgrad::Cfg;
use flux2::AdapterSpec;

fn rng(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f64 / (1u64 << 24) as f64 - 0.5) * 2.0
    }
}

/// The tiny [`flux2::Flux2Config`] whose manifest matches [`Cfg::tiny`].
fn tiny_fc() -> flux2::Flux2Config {
    flux2::Flux2Config {
        in_channels: 4,
        context_in_dim: 6,
        hidden: 16,
        n_heads: 2,
        depth_double: 2,
        depth_single: 2,
        mlp_ratio: 0.75,
        axes_dim: [2, 2, 2, 2],
        txt_len: 3,
        ..flux2::Flux2Config::klein_4b()
    }
}

fn manifest_tensors(fc: &flux2::Flux2Config, seed: u64) -> flux2::Tensors {
    let mut r = rng(seed);
    let mut ts = flux2::Tensors::new();
    for (name, shape) in fc.tensor_manifest() {
        let n: usize = shape.iter().product();
        let (base, scale) = if name.ends_with("norm.scale") { (1.0, 0.1) } else { (0.0, 0.2) };
        let data: Vec<f32> = (0..n).map(|_| (base + r() * scale) as f32).collect();
        ts.insert(name, (shape, data));
    }
    ts
}

fn tmp(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("brain-flux2-lora-stack-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d.join(name)
}

/// A trained-looking brain-native adapter: a fresh one is `B = 0` (a no-op
/// fold, which would prove nothing), so `B` is filled from `seed` directly.
fn brain_adapter(c: &Cfg, seed: u64, name: &str) -> String {
    let mut ad = LoraAdapter::new(c, LoraCfg::new(2));
    let mut r = rng(seed);
    for p in ad.pairs_mut() {
        for v in p.b.iter_mut() {
            *v = r() as f32;
        }
    }
    let path = tmp(name);
    save_adapter(path.to_str().unwrap(), &ad);
    path.to_str().unwrap().to_string()
}

/// Write a minimal F32 safetensors file (8-byte LE header length, JSON header,
/// payloads back to back) - a stand-in third-party adapter.
fn write_st(path: &std::path::Path, tensors: &[(String, Vec<usize>, Vec<f32>)]) {
    let mut header = serde_json::Map::new();
    let mut blob: Vec<u8> = Vec::new();
    for (name, shape, data) in tensors {
        let start = blob.len();
        for v in data {
            blob.extend_from_slice(&v.to_le_bytes());
        }
        header.insert(
            name.clone(),
            serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, blob.len()]}),
        );
    }
    let hdr = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut out = (hdr.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&hdr);
    out.extend_from_slice(&blob);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, out).unwrap();
}

/// A rank-1 third-party adapter over `stem`, whose `B·A` is `fill` everywhere.
fn external_adapter(name: &str, stem: &str, out: usize, inn: usize, fill: f32) -> String {
    let p = tmp(name);
    write_st(
        &p,
        &[
            (format!("diffusion_model.{stem}.lora_A.weight"), vec![1, inn], vec![fill; inn]),
            (format!("diffusion_model.{stem}.lora_B.weight"), vec![out, 1], vec![1.0; out]),
        ],
    );
    p.to_str().unwrap().to_string()
}

/// **Zero adapters must be exactly today's unadapted build.** This is the case
/// every non-LoRA generation takes, so it has to be a byte-for-byte no-op over
/// the whole tensor map - not "close", and not a rebuilt map with the same
/// values.
#[test]
fn no_adapters_leaves_every_tensor_untouched() {
    let fc = tiny_fc();
    let base = manifest_tensors(&fc, 0xD00D);
    let mut ts = base.clone();
    let reports = fold_adapters(&fc, &mut ts, &[]).expect("no adapters folds");
    assert!(reports.is_empty(), "nothing was folded, so nothing is reported");
    assert_eq!(ts, base, "a zero-adapter build must not touch the tensor map at all");
}

/// **One brain-native adapter must be exactly what `--adapter` did before this
/// existed**: `fold_adapters` with a one-element list has to agree bit for bit
/// with `LoraAdapter::fold_into_tensors_at`, the pre-existing single-adapter
/// entry point, at the default strength and at a dialled-down one.
#[test]
fn a_single_brain_adapter_is_bit_identical_to_the_single_adapter_fold() {
    let c = Cfg::tiny();
    let fc = tiny_fc();
    let path = brain_adapter(&c, 0xA11CE, "single.brain");
    let reloaded = flux2::lora::load_adapter(&path, &c).expect("reload");

    for strength in [1.0f32, 0.35, 0.0] {
        let mut want = manifest_tensors(&fc, 0xD00D);
        reloaded.fold_into_tensors_at(&mut want, strength).expect("reference fold");

        let mut got = manifest_tensors(&fc, 0xD00D);
        let reports = fold_adapters(&fc, &mut got, &[AdapterSpec { path: path.clone(), scale: strength }])
            .expect("stacked fold");
        assert_eq!(got, want, "strength {strength}: the one-adapter list must reproduce fold_into_tensors_at exactly");
        assert_eq!(reports.len(), 1);
        assert!(!reports[0].external(), "a .brain container is not the third-party family");
        assert_eq!(reports[0].rank, 2);
    }
}

/// The same guarantee for the third-party family: a single `.safetensors`
/// through `fold_adapters` must equal `fold_external_adapter` exactly.
#[test]
fn a_single_external_adapter_is_bit_identical_to_the_single_adapter_fold() {
    let fc = tiny_fc();
    let key = "double_blocks.0.img_attn.proj.weight";
    let (out, inn) = (fc.hidden, fc.hidden);
    let path = external_adapter("single.safetensors", "double_blocks.0.img_attn.proj", out, inn, 0.25);

    let mut want = manifest_tensors(&fc, 0xD00D);
    let info = fold_external_adapter(&path, &mut want, 0.7).expect("reference fold");
    assert_eq!(info.pairs, 1);

    let mut got = manifest_tensors(&fc, 0xD00D);
    let reports =
        fold_adapters(&fc, &mut got, &[AdapterSpec { path: path.clone(), scale: 0.7 }]).expect("stacked fold");
    assert_eq!(got, want);
    assert_eq!(got[key].1, want[key].1);
    assert!(reports[0].external());
    assert_eq!((reports[0].pairs, reports[0].rank), (1, 1));
}

/// **The feature.** Two third-party adapters, one targeting a tensor the other
/// also targets and one targeting a tensor it does not, each at its own
/// strength, folded in a stated order.
///
/// The shared target must end up moved by the SUM of both scaled deltas (the
/// second adapter folds onto the first's already-folded result, and additive
/// low-rank deltas compose by addition), the distinct target by only its own
/// adapter's, and every untargeted tensor not at all.
#[test]
fn two_adapters_stack_each_at_its_own_scale() {
    let fc = tiny_fc();
    let shared = "double_blocks.0.img_attn.proj.weight";
    let only_b = "double_blocks.1.img_attn.proj.weight";
    let d = fc.hidden;

    // A: delta 1.0 on `shared`. B: delta 1.0 on `shared` AND on `only_b`.
    let a = external_adapter("stack-a.safetensors", "double_blocks.0.img_attn.proj", d, d, 1.0);
    let bp = tmp("stack-b.safetensors");
    write_st(
        &bp,
        &[
            ("diffusion_model.double_blocks.0.img_attn.proj.lora_A.weight".into(), vec![1, d], vec![1.0; d]),
            ("diffusion_model.double_blocks.0.img_attn.proj.lora_B.weight".into(), vec![d, 1], vec![1.0; d]),
            ("diffusion_model.double_blocks.1.img_attn.proj.lora_A.weight".into(), vec![1, d], vec![1.0; d]),
            ("diffusion_model.double_blocks.1.img_attn.proj.lora_B.weight".into(), vec![d, 1], vec![1.0; d]),
        ],
    );
    let b = bp.to_str().unwrap().to_string();

    let base = manifest_tensors(&fc, 0xD00D);
    let mut ts = base.clone();
    let reports = fold_adapters(
        &fc,
        &mut ts,
        &[AdapterSpec { path: a.clone(), scale: 0.25 }, AdapterSpec { path: b.clone(), scale: 0.5 }],
    )
    .expect("stacked fold");
    assert_eq!(reports.len(), 2, "both adapters are reported, in order");
    assert_eq!(reports[0].path, a);
    assert_eq!(reports[1].path, b);

    // Shared target: 0.25 from A plus 0.5 from B.
    for (i, (got, was)) in ts[shared].1.iter().zip(&base[shared].1).enumerate() {
        assert!((got - (was + 0.75)).abs() < 1e-6, "{shared}[{i}]: {got} != {was} + 0.25 + 0.5");
    }
    // B-only target: 0.5, and nothing from A.
    for (i, (got, was)) in ts[only_b].1.iter().zip(&base[only_b].1).enumerate() {
        assert!((got - (was + 0.5)).abs() < 1e-6, "{only_b}[{i}]: {got} != {was} + 0.5");
    }
    // Everything else is exactly as it was.
    for (name, (_, data)) in &base {
        if name == shared || name == only_b {
            continue;
        }
        assert_eq!(&ts[name].1, data, "{name} is not adapted and must be untouched");
    }

    // Each strength scales only its own adapter: halving A's alone must move
    // the shared target by exactly A's own contribution, and leave B's target
    // where it was.
    let mut half_a = base.clone();
    fold_adapters(&fc, &mut half_a, &[AdapterSpec { path: a, scale: 0.125 }, AdapterSpec { path: b, scale: 0.5 }])
        .expect("stacked fold");
    for (i, (full, half)) in ts[shared].1.iter().zip(&half_a[shared].1).enumerate() {
        assert!((full - half - 0.125).abs() < 1e-6, "{shared}[{i}]: halving A must remove exactly 0.125");
    }
    assert_eq!(half_a[only_b].1, ts[only_b].1, "A's strength must not reach B's own target");
}

/// The two families stack together - a brain-trained identity adapter under a
/// third-party style `.safetensors`, in one run - and each still lands its own
/// delta. Composed against the two single folds run in the same order, which
/// is the definition the ordering doc states.
#[test]
fn the_two_adapter_families_stack_together() {
    let c = Cfg::tiny();
    let fc = tiny_fc();
    let native = brain_adapter(&c, 0xBEEF, "mixed.brain");
    let ext = external_adapter("mixed.safetensors", "double_blocks.0.img_attn.proj", fc.hidden, fc.hidden, 0.5);

    let mut want = manifest_tensors(&fc, 0xD00D);
    flux2::lora::load_adapter(&native, &c).expect("reload").fold_into_tensors_at(&mut want, 0.6).expect("fold");
    fold_external_adapter(&ext, &mut want, 0.9).expect("fold");

    let mut got = manifest_tensors(&fc, 0xD00D);
    let reports = fold_adapters(
        &fc,
        &mut got,
        &[AdapterSpec { path: native, scale: 0.6 }, AdapterSpec { path: ext, scale: 0.9 }],
    )
    .expect("stacked fold");
    assert_eq!(got, want, "a mixed stack is the two single folds applied in list order");
    assert_eq!(reports.iter().filter(|r| r.external()).count(), 1);
}

/// A rejected adapter anywhere in the list is a hard error naming the offending
/// tensor. It must not be reported as a partial success: a stacked run that
/// silently drops its second adapter looks exactly like one that applied it.
#[test]
fn a_bad_adapter_in_the_list_fails_loudly() {
    let fc = tiny_fc();
    let good = external_adapter("ok.safetensors", "double_blocks.0.img_attn.proj", fc.hidden, fc.hidden, 1.0);
    let bad = external_adapter("bad.safetensors", "no_such_module", 4, 4, 1.0);
    let mut ts = manifest_tensors(&fc, 0xD00D);
    let e = fold_adapters(
        &fc,
        &mut ts,
        &[AdapterSpec { path: good, scale: 1.0 }, AdapterSpec { path: bad, scale: 1.0 }],
    )
    .expect_err("the second adapter targets nothing that exists");
    assert!(e.contains("no_such_module"), "the error must name the tensor: {e}");
    assert!(e.contains("FLUX.2"), "the error must name the architecture: {e}");
}
