// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! FLUX.2 declares its parts; the engine places them. This gates the
//! declaration - that the costs are architecture-derived and that the
//! automatic text-encoder decision reproduces the layout that used to have to
//! be typed by hand as `BRAIN_FLUX2_TE_DEVICE=gpu1:i8`.
//!
//! Swedish Embedded AB implements automatic multi-device model placement for
//! its clients. If your team needs expertise in fitting a diffusion pipeline
//! across the cards a machine actually has, you can procure our services by
//! sending an email to info@swedishembedded.com.

use std::collections::HashMap;

use checkpoint::gguf::{MmapGguf, Q8_0_BLOCK_ELEMS};
use checkpoint::quantize::{convert, Policy, Tier};
use flux2::pipeline::{dit_bytes, effective_dit_precision, gguf_dit_device_bytes, plan_parts, te_bytes, vae_bytes};
use gpu_core::devices::{install_placer, Placer};
use std::sync::{Arc, Mutex, OnceLock};
use flux2::Precision;

const GIB: f64 = (1u64 << 30) as f64;
fn gib(b: u64) -> f64 {
    b as f64 / GIB
}

/// Build a small complete FLUX.2-shaped Q8_0 source. The tiny dimensions keep
/// the fixture cheap while preserving every allocation class used by the real
/// constructor: host-only modulation, fp32 boundary/down/norm buffers, and
/// packed int8 linears.
fn tiny_dit(policy: Policy) -> (flux2::Flux2Config, MmapGguf, String) {
    let cfg = flux2::Flux2Config {
        in_channels: 32,
        context_in_dim: 64,
        hidden: 32,
        n_heads: 1,
        depth_double: 1,
        depth_single: 1,
        mlp_ratio: 1.0,
        axes_dim: [8, 8, 8, 8],
        rope_theta: 2000.0,
        norm_eps: 1e-6,
        txt_len: 64,
        guidance_embed: false,
        distilled: true,
    };
    let src: HashMap<_, _> = cfg
        .tensor_manifest()
        .into_iter()
        .map(|(name, shape)| {
            let n = shape.iter().product();
            (name, (shape, vec![0.125f32; n]))
        })
        .collect();
    let path = std::env::temp_dir()
        .join(format!(
            "flux2-placement-q8-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
        .to_string_lossy()
        .into_owned();
    convert(&src, Tier::Q8_0, &policy, &[], &path, &mut |_, _| {}).unwrap();
    let g = MmapGguf::open(&path).unwrap();
    (cfg, g, path)
}

fn tiny_q8_dit() -> (flux2::Flux2Config, MmapGguf, String) {
    let (cfg, g, path) = tiny_dit(Policy::new().min_elems(1));
    let quantized = cfg
        .tensor_manifest()
        .iter()
        .filter(|(name, _)| g.dtype(name) == Some("Q8_0"))
        .count();
    assert_eq!(quantized, 19, "the fixture must preserve the writer's rank-2 Q8_0 / rank-1 F32 mix");
    (cfg, g, path)
}

/// The CLI's implicit fp32 default follows a GGUF's executable int8 format,
/// but an operator's explicit fp32 request is never silently ignored. The
/// returned precision is also what `Pipeline::build_inner` passes to planning
/// and `Flux2Model::new_from`.
#[test]
fn gguf_effective_precision_is_planned_and_built_as_int8() {
    assert_eq!(
        effective_dit_precision("model.gguf", Precision::F32, false).unwrap(),
        Precision::Int8
    );
    assert_eq!(
        effective_dit_precision("model.gguf", Precision::Int8, true).unwrap(),
        Precision::Int8
    );
    assert!(effective_dit_precision("model.gguf", Precision::F32, true).is_err());
    assert_eq!(
        effective_dit_precision("model.safetensors", Precision::F32, true).unwrap(),
        Precision::F32
    );
}

/// Q8_0 has 34 raw bytes per 32 elements. `Flux2Model::new_from` re-packs an
/// int8 linear into 32 packed bytes plus one f32 scale, or retains a deliberate
/// f32 exception as 128 bytes. This is deliberately written from the file
/// header byte lengths rather than the config element counts: the production
/// estimate must describe the source it validated. Rank-1 norms remain F32 in
/// the source and are separately counted as their f32 device buffers.
#[test]
fn q8_gguf_cost_matches_the_buffers_flux2_keeps_on_device() {
    let (cfg, g, path) = tiny_q8_dit();
    let kept_on_host = [
        "time_in.in_layer.weight",
        "time_in.out_layer.weight",
        "double_stream_modulation_img.lin.weight",
        "double_stream_modulation_txt.lin.weight",
        "single_stream_modulation.lin.weight",
        "final_layer.adaLN_modulation.1.weight",
    ];
    let f32_on_device = |name: &str| {
        name == "img_in.weight"
            || name == "txt_in.weight"
            || name == "final_layer.linear.weight"
            || name.contains("norm.query_norm.scale")
            || name.contains("norm.key_norm.scale")
            || (name.starts_with("double_blocks.") && name.ends_with("_mlp.2.weight"))
    };
    let expected_weights: u64 = cfg
        .tensor_manifest()
        .iter()
        .map(|(name, _)| {
            let (raw, ty) = g.raw_tensor_bytes(name).expect("manifest tensor");
            if kept_on_host.contains(&name.as_str()) {
                0
            } else if f32_on_device(name) {
                // Intentional F32 device buffers include rank-1 norm scales,
                // which the writer preserves as F32 in this mixed fixture.
                let elems = if ty == checkpoint::gguf::TYPE_Q8_0 {
                    assert_eq!(raw.len() % 34, 0);
                    assert_eq!(raw.len() / 34 * Q8_0_BLOCK_ELEMS, raw.len() / 34 * 32);
                    raw.len() as u64 / 34 * 32
                } else {
                    raw.len() as u64 / 4
                };
                elems * 4
            } else {
                assert_eq!(g.dtype(name), Some("Q8_0"));
                assert_eq!(raw.len() % 34, 0);
                assert_eq!(raw.len() / 34 * Q8_0_BLOCK_ELEMS, raw.len() / 34 * 32);
                assert_eq!(ty, checkpoint::gguf::TYPE_Q8_0);
                (raw.len() as u64 / 34) * 36
            }
        })
        .sum();
    let n_joint = 96;
    let got = gguf_dit_device_bytes(&g, &cfg, n_joint, 1).expect("Q8_0 header cost");
    let legacy = dit_bytes(&cfg, Precision::Int8, n_joint, 1);
    assert!(got > expected_weights, "scratch is a real device allocation");
    assert_ne!(got, legacy, "the GGUF route must not fall back to config-wide int8 pricing");
    let _ = std::fs::remove_file(path);
}

/// A non-Q8_0 normal linear cannot fall through the streamed constructor's
/// host-f32 re-quantization path. The header validator must reject it before a
/// device is chosen, while allowing rank-1 F32 norm scales and the intentional
/// F32 device exceptions covered above.
#[test]
fn gguf_non_q8_linear_is_rejected_before_placement() {
    let (cfg, g, path) = tiny_dit(Policy::new().min_elems(1).never_quantize(&["single_blocks.0.linear1.weight"]));
    assert_eq!(g.dtype("single_blocks.0.linear1.weight"), Some("F32"));
    let err = gguf_dit_device_bytes(&g, &cfg, 96, 1).unwrap_err();
    assert!(
        err.contains("single_blocks.0.linear1.weight is F32") && err.contains("will not upcast it"),
        "unexpected error: {err}"
    );
    let _ = std::fs::remove_file(path);
}

/// A GGUF DiT is costed from its header before the planner sees it, and that
/// same source resolves the model's executable precision to Int8. A fake
/// placer makes this a hardware-free contract test: legacy fp32 placement
/// would reject this layout, while the GGUF cost is accepted without falling
/// back to an ambient device.
#[test]
fn gguf_plan_uses_the_header_cost_and_the_int8_build_precision() {
    struct RecordingPlacer(std::sync::Mutex<Option<Vec<gpu_core::devices::Need>>>);
    impl Placer for RecordingPlacer {
        fn place(&self, needs: &[gpu_core::devices::Need]) -> Result<Vec<gpu_core::devices::Home>, String> {
            *self.0.lock().unwrap() = Some(needs.to_vec());
            Ok(vec![
                gpu_core::devices::Home::Gpu(0),
                gpu_core::devices::Home::Gpu(1),
                gpu_core::devices::Home::Gpu(0),
            ])
        }
    }
    static PLACER_TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let _serial = PLACER_TEST_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap();
    let (cfg, g, path) = tiny_q8_dit();
    let expected = gguf_dit_device_bytes(&g, &cfg, 96, 1).unwrap();
    let recorded = Arc::new(RecordingPlacer(Mutex::new(None)));
    install_placer(recorded.clone());
    let paths = flux2::Paths {
        dit: path.clone(),
        vae: "unused".to_string(),
        te: "unused".to_string(),
        tokenizer: "unused".to_string(),
    };
    let vae = vae::VaeConfig::flux2();
    let (homes, te) = plan_parts(&cfg, &paths, &vae, Precision::Int8, 96, 32, 1).unwrap();
    assert_eq!(homes.of("dit"), Some(gpu_core::devices::Home::Gpu(0)));
    assert_eq!(te.gpu_index, Some(1));
    let needs = recorded.0.lock().unwrap().take().unwrap();
    assert_eq!(needs[0].vram, expected);
    assert_eq!(needs[2].vram, vae_bytes(&vae, 32));
    assert_eq!(effective_dit_precision(&path, Precision::F32, false).unwrap(), Precision::Int8);
    let _ = std::fs::remove_file(path);
}

/// The measured constraint this whole mechanism exists for: on a 24 GiB card
/// an f32 truncated Qwen3-8B text encoder does NOT fit, and the int8 one
/// does. That is what makes the automatic decision pick int8 for klein-9b -
/// and it must fall out of the architecture, not out of a remembered number.
///
/// The tap layers are [9, 18, 27], so the shard keeps layers [0, 27).
#[test]
fn the_nine_b_text_encoder_needs_int8_to_fit_a_twenty_four_gib_card() {
    let te = qwen3::QwenConfig::qwen3_8b();
    let f32_bytes = te_bytes(&te, 27, 512, false);
    let i8_bytes = te_bytes(&te, 27, 512, true);
    assert!(gib(f32_bytes) > 24.0, "an f32 truncated Qwen3-8B must not be claimed to fit a 24 GiB card: {:.1} GiB", gib(f32_bytes));
    assert!(gib(i8_bytes) < 16.0, "the int8 shard must fit beside a driver context: {:.1} GiB", gib(i8_bytes));
    assert!(i8_bytes * 2 < f32_bytes, "int8 must be several times smaller: {:.1} vs {:.1} GiB", gib(i8_bytes), gib(f32_bytes));
}

/// ...and the 4B pipeline's encoder does fit in f32, so its conditioning is
/// not silently downgraded to the lossy tier by a rule written for the 9B.
#[test]
fn the_four_b_text_encoder_still_fits_a_card_in_f32() {
    let te = qwen3::QwenConfig::qwen3_4b();
    assert!(gib(te_bytes(&te, 27, 512, false)) < 22.0, "{:.1} GiB", gib(te_bytes(&te, 27, 512, false)));
}

/// A truncated shard must cost strictly less than a whole encoder - the whole
/// point of truncating it - so placement sees the difference.
///
/// The `layers = 0` probe is what makes this a gate on the WEIGHT filter and
/// not merely on the scratch term (which is trivially proportional to
/// `layers`): with no block resident, only the embedding is on the card, and
/// a cost model that ignored the truncation would still report the whole
/// stack's weights here.
#[test]
fn truncation_is_visible_to_the_cost_model() {
    let te = qwen3::QwenConfig::qwen3_8b();
    let whole = te_bytes(&te, te.n_layers as usize, 512, false);
    let cut = te_bytes(&te, 27, 512, false);
    assert!(cut < whole, "truncated {:.1} GiB must be less than whole {:.1} GiB", gib(cut), gib(whole));
    let embed_only = te_bytes(&te, 0, 0, false);
    assert!(gib(embed_only) < 4.0, "with no block resident only the embedding is on the card: {:.1} GiB", gib(embed_only));
    assert!(embed_only * 5 < whole, "the block stack must dominate: {:.1} vs {:.1} GiB", gib(embed_only), gib(whole));
}

/// The DiT cost follows the architecture: 9B is bigger than 4B, int8 is
/// smaller than f32, and a bigger joint sequence costs more scratch. A cost
/// model that ignored any of these would place a 9B run as if it were a 4B
/// one.
#[test]
fn the_dit_cost_follows_the_architecture_and_the_numeric_tier() {
    let (four, nine) = (flux2::Flux2Config::klein_4b(), flux2::Flux2Config::klein_9b());
    let n = 512 + 3072;
    assert!(dit_bytes(&nine, Precision::F32, n, 1) > dit_bytes(&four, Precision::F32, n, 1), "9B must cost more than 4B");
    assert!(dit_bytes(&nine, Precision::Int8, n, 1) < dit_bytes(&nine, Precision::F32, n, 1), "int8 must cost less than f32");
    assert!(dit_bytes(&nine, Precision::Int8, 2 * n, 1) > dit_bytes(&nine, Precision::Int8, n, 1), "a longer joint sequence costs more scratch");
    // A 9B int8 DiT plus its scratch is most of a 24 GiB card - which is why
    // the text encoder cannot join it. Bracketed, not pinned to a
    // measurement: this is a placement input, not a performance claim.
    let real = dit_bytes(&nine, Precision::Int8, n, 1);
    assert!((8.0..20.0).contains(&gib(real)), "int8 9B DiT budget out of the plausible band: {:.1} GiB", gib(real));
}

/// The VAE reservation must describe the decode that actually runs.
///
/// This is the second field failure in one assertion. Placement chose the
/// right card, every denoise step completed, and the run died in `decoding` -
/// because the plan reserved a flat 2 GiB for a stage whose real footprint at
/// a full-frame output is several times that. A decode's cost is dominated by
/// activations that scale with the image, so a constant is wrong at every size
/// except by accident.
#[test]
fn the_vae_reservation_covers_a_full_frame_decode() {
    let vc = vae::VaeConfig::flux2();
    // 768x1024 out = 48 x 64 latent tokens - the size that failed.
    let full_frame = 48 * 64;
    let got = flux2::pipeline::vae_bytes(&vc, full_frame);
    assert!(
        gib(got) > 6.0,
        "a full-frame decode needs far more than the flat 2 GiB this used to reserve: {:.2} GiB",
        gib(got)
    );
    // ...and it has to be the IMAGE that drives it, not a bigger constant.
    let quarter = flux2::pipeline::vae_bytes(&vc, full_frame / 4);
    assert!(
        got > quarter + (got - quarter) / 2,
        "the reservation must grow with the output: {:.2} GiB at a quarter frame vs {:.2} GiB full",
        gib(quarter),
        gib(got)
    );
    assert!(
        got - quarter > (1u64 << 30),
        "four times the pixels must cost GiBs more, not megabytes: {:.2} -> {:.2} GiB",
        gib(quarter),
        gib(got)
    );
}

/// Reference images enlarge the DiT's joint sequence; they do not enlarge the
/// image that gets decoded. Pricing the decode from the joint ceiling reserved
/// most of a card for an output that is never produced - and on a busy machine
/// that turns a placeable run into a refusal.
#[test]
fn references_grow_the_dit_but_not_the_decode() {
    let c = flux2::Flux2Config::klein_4b();
    let vc = vae::VaeConfig::flux2();
    let n_out = 48 * 64; // 768x1024 generated
    let n_ref = 5 * 432; // five references at --ref-size 384
    let txt = c.txt_len as u64;

    let bare = flux2::pipeline::part_needs(&c, &vc, Precision::Int8, txt + n_out, n_out, 1);
    let with_refs = flux2::pipeline::part_needs(&c, &vc, Precision::Int8, txt + n_out + n_ref, n_out, 1);

    let find = |v: &[gpu_core::devices::Need], n: &str| v.iter().find(|p| p.name == n).expect("part").vram;
    assert!(
        find(&with_refs, "dit") > find(&bare, "dit"),
        "references must grow the DiT's joint-sequence scratch"
    );
    assert_eq!(
        find(&with_refs, "vae"),
        find(&bare, "vae"),
        "references are encoded, never decoded: the VAE reservation must not move"
    );
    // The mistake is worth GiBs, which is why it is worth a gate.
    let wrong = flux2::pipeline::vae_bytes(&vc, n_out + n_ref);
    assert!(
        wrong > find(&bare, "vae") + (1u64 << 31),
        "the confusion this pins costs {:.2} GiB, not a rounding error",
        gib(wrong - find(&bare, "vae"))
    );
}

/// The VAE follows the DiT (it decodes the DiT's own latents) and the text
/// encoder is declared apart from it - the shape the placement engine relies
/// on to spread a pipeline over two cards.
#[test]
fn the_declared_shape_is_dit_te_apart_and_vae_with_the_dit() {
    let c = flux2::Flux2Config::klein_9b();
    let vc = vae::VaeConfig::flux2();
    let needs = flux2::pipeline::part_needs(&c, &vc, Precision::Int8, 512 + 3072, 3072, 1);
    let names: Vec<&str> = needs.iter().map(|n| n.name.as_str()).collect();
    assert_eq!(names, vec!["dit", "te", "vae"]);
    assert_eq!(needs[0].affinity, gpu_core::devices::Affinity::Apart);
    assert_eq!(needs[1].affinity, gpu_core::devices::Affinity::Apart);
    assert_eq!(needs[2].affinity, gpu_core::devices::Affinity::With("dit".to_string()));

    // The text encoder is declared at the DiT's own numeric tier. An int8 run
    // that reserved for an f32 encoder would plan a two-card layout it does
    // not need - or refuse a one-card one that would have worked.
    let f32_needs = flux2::pipeline::part_needs(&c, &vc, Precision::F32, 512 + 3072, 3072, 1);
    assert!(
        needs[1].vram * 2 < f32_needs[1].vram,
        "an int8 run must reserve a much smaller encoder than an f32 one: {:.2} vs {:.2} GiB",
        gib(needs[1].vram),
        gib(f32_needs[1].vram)
    );
}
