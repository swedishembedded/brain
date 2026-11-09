// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MiniMax-H3's checks that need a card, which is an argument:
//!
//! ```text
//! cargo test -p brain-minimaxh3 --test device -- --device gpu1 --backend cuda
//! ```
//!
//! Video sparse attention against its host reference, a gated DiT resident
//! and streamed, the block's row chunks under sparse attention, the DiT's and
//! the video decoder's flash attention against their materialized trio, the
//! int8 tier against fp32, and (with `--ignored`) sparse against dense
//! attention timed at ref2va's own shapes.
//!
//! Swedish Embedded AB implements sparse-attention video generation on the
//! GPUs its clients already own. If your team needs expertise in running
//! large audio-video diffusion models on constrained hardware, you can
//! procure our services by sending an email to info@swedishembedded.com.

use checkpoint::TensorSource;
use gpu_core::select::Dtype;
use gpu_core::DeviceBuffer;
use minimaxh3::block::{attn_mode, dit_attention, ActLevels, AttnMode, Ctx, ROW_CHUNK};
use minimaxh3::config::{H3TransformerConfig, TAG_TEXT};
use minimaxh3::dit_shard::StreamingDit;
use minimaxh3::model::{random_checkpoint, H3Output, H3Transformer, Int8Rounded, PackedInputs};
use minimaxh3::pipeline::{build_packed_sequence, PackedSequence, Segment, AUDIO_CHANNELS};
use minimaxh3::video_vae::{decoder_attention, VideoVaeConfig};
use minimaxh3::vsa::{attention, attention_host, DevicePlan, Spec, TileGeometry};
use model::StreamPlan;
use vae::blocks::Tensors;

/// The largest difference relative to `1 + |b|`.
fn close(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs() / (1.0 + y.abs())).fold(0.0, f32::max)
}

fn outputs_close(a: &H3Output, b: &H3Output, bound: f32, what: &str) {
    let worst = close(&[a.video.as_slice(), &a.audio].concat(), &[b.video.as_slice(), &b.audio].concat());
    assert!(worst < bound, "{what}: worst relative difference {worst}");
}

/// Cosine similarity and max absolute difference between two same-length
/// vectors - the pair of numbers this workspace's numeric parity gates are
/// always stated in.
fn cos_and_maxabs(a: &[f32], b: &[f32]) -> (f64, f64) {
    assert_eq!(a.len(), b.len());
    let (mut dot, mut na, mut nb, mut maxabs) = (0f64, 0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (*x as f64, *y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
        maxabs = maxabs.max((x - y).abs());
    }
    (dot / (na.sqrt() * nb.sqrt()), maxabs)
}

/// Deterministic pseudo-random fill - a SplitMix64 draw's top 24 bits
/// mapped into `[-1, 1)`, so both arms see byte-identical inputs without
/// pulling an RNG dependency into this module.
///
/// The RANGE is load-bearing, not cosmetic. `q·k` runs over `head_dim`
/// terms and is then scaled by only `1/sqrt(head_dim)`, so inputs an order
/// of magnitude wide push the logits far enough apart that `softmax`
/// saturates to a one-hot pick of a single `v` row - at which point BOTH
/// arms return that row verbatim, agree bit-for-bit, and prove nothing
/// about the online softmax. An earlier version of this generator divided
/// by 2048 instead of 2^23 and produced values up to 8191; it reported
/// cosine 1.0000000000 and max_abs exactly 0 at every sequence length,
/// which looked like an unusually clean pass and was in fact a vacuous
/// one. [`assert_mixing`] below is the standing guard against that.
fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 40) as f32 / 8_388_608.0) - 1.0
        })
        .collect()
}

fn std_dev(v: &[f32]) -> f64 {
    let mean = v.iter().map(|x| *x as f64).sum::<f64>() / v.len() as f64;
    (v.iter().map(|x| (*x as f64 - mean).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
}

/// Fail unless the attention actually AVERAGED over many keys.
///
/// Attention output is a convex combination of `v` rows, so a softmax that
/// spreads its mass over `n` keys shrinks the spread of the result by
/// roughly `sqrt(n)`, while one that has saturated to a single key
/// reproduces a `v` row unchanged and leaves the spread equal to `v`'s.
/// Comparing two arms on a saturated softmax compares two copies of the
/// same lookup and would pass no matter what the online softmax did with
/// its running max and rescale - the exact code this test exists to check.
/// Half of `v`'s own standard deviation is a deliberately loose threshold:
/// it is nowhere near the `1/sqrt(37)` a near-uniform mix over even the
/// shortest sequence here would give, so it fires only on genuine
/// saturation and never on an unlucky draw.
fn assert_mixing(out: &[f32], v: &[f32], seq_len: u32) {
    let (so, sv) = (std_dev(out), std_dev(v));
    assert!(
        so < 0.5 * sv,
        "seq_len {seq_len}: softmax did not mix - output std {so:.4} vs v std {sv:.4}. \
         The logits have saturated to a one-hot pick, so both attention arms are \
         returning the same v row verbatim and this comparison proves nothing."
    );
}

/// The device attention computes the host reference's: with and without a
/// gate, over dense prefix tiles, two video regions with partial edge tiles,
/// and a narrow and the full head width.
fn sparse_attention_computes_the_host_reference() {
    let cx = Ctx::new(None);
    let segments = [Segment::Dense { rows: 70 }, Segment::Video { grid: [3, 5, 6] }, Segment::Dense { rows: 9 }, Segment::Video { grid: [9, 4, 7] }];
    let rows = segments.iter().map(Segment::rows).sum::<u32>() as usize;
    for head_dim in [32usize, 128] {
        let heads = 3usize;
        let inner = heads * head_dim;
        let mut rng = data::rng::Lcg::new(31 + head_dim as u64);
        let (q, k, v) = (rng.vec_scaled(rows * inner, 1.0), rng.vec_scaled(rows * inner, 1.0), rng.vec_scaled(rows * inner, 1.0));
        let gate = rng.vec_scaled(rows * inner, 0.5);
        let slab: Vec<f32> = (0..rows).flat_map(|r| [&q, &k, &v].into_iter().flat_map(move |m| m[r * inner..][..inner].iter().copied())).collect();
        let (slab_dev, gate_dev) = (cx.upload(&slab), cx.upload(&gate));
        let sparsities = (0.6, 0.5);
        let plan = DevicePlan::new(&cx.gpu, TileGeometry::new(&segments), sparsities);
        for g in [None, Some(&gate)] {
            let want = attention_host(&plan.geometry, (&q, &k, &v), g.map(|g| g.as_slice()), (heads, head_dim), sparsities);
            let got = cx.gpu.read(&attention(&cx.gpu, &plan, &slab_dev, g.map(|_| &gate_dev), heads as u32, head_dim as u32), rows * inner);
            let worst = close(&got, &want);
            assert!(worst < 1e-4, "head_dim {head_dim}, gate {}: worst {worst}", g.is_some());
        }
    }
}

/// A tiny DiT's inputs over a text, video and audio layout with two timesteps
/// on alternating rows, so a row chunk's AdaLN indices are not one constant.
struct Tiny {
    cfg: H3TransformerConfig,
    tensors: Tensors,
    gates: Tensors,
    layout: PackedSequence,
    video: Vec<f32>,
    audio: Vec<f32>,
    text: Vec<f32>,
    timestep_indices: Vec<u32>,
}

impl Tiny {
    fn new(cfg: H3TransformerConfig) -> Tiny {
        let seed = data::rng::random_seed();
        let tensors = random_checkpoint(&cfg, seed);
        let [_, ph, pw] = cfg.patch_size;
        let layout = build_packed_sequence(&[TAG_TEXT; 3], 8, 4 * ph, 4 * pw, 3, cfg.patch_size, AUDIO_CHANNELS, &[]);
        let mut rng = data::rng::Lcg::new(seed ^ 0x5eed);
        let (inner, hidden) = (cfg.inner_dim() as usize, cfg.hidden_size as usize);
        let gates = (0..cfg.num_layers).map(|i| (format!("blocks.{i}.attn.to_gate_compress.weight"), (vec![inner, hidden], rng.vec_scaled(inner * hidden, 0.2)))).collect();
        let video = rng.vec_scaled(layout.video_indices.len() * cfg.video_patch_dim() as usize, 0.3);
        let audio = rng.vec_scaled(layout.audio_indices.len() * cfg.audio_in_channels as usize, 0.3);
        let text = rng.vec_scaled(3 * cfg.text_dim as usize, 0.3);
        let timestep_indices = (0..layout.token_tags.len() as u32).map(|r| r % 2).collect();
        Tiny { cfg, tensors, gates, layout, video, audio, text, timestep_indices }
    }

    const TIMESTEPS: [f32; 2] = [0.3, 0.7];

    fn inputs(&self, sparsity: Option<f32>, row_chunk: u32) -> PackedInputs<'_> {
        PackedInputs {
            hidden_states: &self.video,
            audio_hidden_states: &self.audio,
            encoder_hidden_states: &self.text,
            timestep: &Self::TIMESTEPS,
            timestep_indices: &self.timestep_indices,
            token_tags: &self.layout.token_tags,
            position_ids: &self.layout.position_ids,
            video_indices: &self.layout.video_indices,
            audio_indices: &self.layout.audio_indices,
            text_indices: &self.layout.text_indices,
            act_levels: ActLevels::One,
            vsa: sparsity.map(|sparsity| Spec { segments: &self.layout.segments, sparsity, reference_sparsity: sparsity }),
            row_chunk,
        }
    }

    fn resident(&self, gates: Option<&dyn TensorSource>) -> H3Transformer {
        self.resident_from(&self.tensors, Dtype::F32, gates)
    }

    /// Hold `tensors` - this model's or a view of them - at `tier`, ready to
    /// run at [`Self::TIMESTEPS`].
    fn resident_from(&self, tensors: &dyn TensorSource, tier: Dtype, gates: Option<&dyn TensorSource>) -> H3Transformer {
        H3Transformer::load_on(tensors, self.cfg, StreamPlan::single(self.cfg.num_layers as usize), None, tier, gates).expect("a resident tiny model")
    }
}

/// Sparse attention with every tile kept computes dense attention; with the
/// gates' coarse branch on, a resident and a streamed DiT agree; dropping
/// tiles and the coarse branch each change the forward.
fn a_gated_dit_runs_resident_and_streamed_alike() {
    let t = Tiny::new(H3TransformerConfig::tiny());
    let plain = t.resident(None);
    let _p = plain.prepare(&t.tensors, &Tiny::TIMESTEPS).expect("prepare");
    let dense = plain.forward(&t.inputs(None, ROW_CHUNK));
    outputs_close(&plain.forward(&t.inputs(Some(0.0), ROW_CHUNK)), &dense, 1e-3, "every tile kept, no gate");
    drop(plain);

    let gated = t.resident(Some(&t.gates));
    assert!(gated.has_gates());
    let _p = gated.prepare(&t.tensors, &Tiny::TIMESTEPS).expect("prepare");
    // Every tile kept, so the selection cannot differ between the two builds
    // and their arithmetic must agree. A dropping selection is discontinuous
    // in the scores - a last-ulp difference may flip a near tie - and is
    // checked exactly against the host reference above.
    let resident = gated.forward(&t.inputs(Some(0.0), ROW_CHUNK));
    let dit = StreamingDit::open_single(&t.cfg, None);
    let _s = dit.prepare(&t.tensors, &t.cfg, &Tiny::TIMESTEPS).expect("prepare");
    let streamed = H3Transformer::forward_streaming(&t.tensors, Some(&t.gates), &t.cfg, &dit, &t.inputs(Some(0.0), ROW_CHUNK));
    outputs_close(&resident, &streamed, 1e-3, "resident and streamed with gates");
    let sparse = gated.forward(&t.inputs(Some(0.5), ROW_CHUNK));
    assert!(sparse.video.iter().zip(&resident.video).any(|(a, b)| a != b), "dropping tiles must change the forward");
    assert!(resident.video.iter().zip(&dense.video).any(|(a, b)| a != b), "the gates' coarse branch must change the forward");
}

/// A block's row-local work over many small row chunks computes what one
/// chunk over the whole sequence does under dense and gated sparse attention
/// alike, so the gate projection's full-length buffer is written at each
/// chunk's own rows too.
fn row_chunks_compute_what_one_chunk_computes() {
    let t = Tiny::new(H3TransformerConfig::tiny());
    let seq_len = t.layout.token_tags.len() as u32;
    assert!(seq_len > 7 && !seq_len.is_multiple_of(7), "the layout spans several chunks with a remainder ({seq_len} rows)");
    let gated = t.resident(Some(&t.gates));
    let _p = gated.prepare(&t.tensors, &Tiny::TIMESTEPS).expect("prepare");
    for sparsity in [None, Some(0.0)] {
        let whole = gated.forward(&t.inputs(sparsity, seq_len));
        // 7 rows leave a short last chunk; 1 row is every row its own chunk.
        for chunk in [7u32, 1] {
            outputs_close(&gated.forward(&t.inputs(sparsity, chunk)), &whole, 1e-4, &format!("sparsity {sparsity:?} in {chunk}-row chunks"));
        }
    }
}

/// The DiT's flash attention computes what its materialized trio does - the
/// same `1/sqrt(head_dim)` scale, the same unmasked bidirectional coverage,
/// the same `[seq_len, heads*head_dim]` layout - to fp32 rounding, at the
/// real head width of 128 and at lengths coprime to every kernel tile (37 is
/// shorter than one query tile), so the partial-tile guards run.
fn the_dits_flash_attention_matches_its_materialized_trio() {
    let cx = Ctx::new(None);
    if attn_mode(&cx.gpu, 128) != AttnMode::Flash {
        brain_testutil::skip_unavailable("this card cannot run the flash-attention family: both arms would be the trio");
        return;
    }
    let (heads, head_dim) = (4u32, 128u32);
    let inner = heads * head_dim;
    for seq_len in [37u32, 100, 200] {
        let n = (seq_len * inner) as usize;
        let v_host = fill(n, 3);
        let (q, k, v) = (cx.upload(&fill(n, 1)), cx.upload(&fill(n, 2)), cx.upload(&v_host));
        let run = |mode| cx.gpu.read(&dit_attention(&cx.gpu, mode, &q, &k, &v, seq_len, heads, head_dim), n);
        let (want, got) = (run(AttnMode::Materialized), run(AttnMode::Flash));
        assert_mixing(&want, &v_host, seq_len);
        let (cos, maxabs) = cos_and_maxabs(&want, &got);
        println!("seq_len {seq_len}: cosine {cos:.10}, max_abs {maxabs:.3e}");
        assert!(cos > 0.999999, "seq_len {seq_len}: flash vs trio cosine {cos:.10} (max_abs {maxabs:.3e})");
        assert!(maxabs < 1e-4, "seq_len {seq_len}: flash vs trio max_abs {maxabs:.3e} (cosine {cos:.10})");
    }
}

/// The video VAE decoder's attention, on its own kernels: flash and the
/// materialized trio agree at the decoder's shape - its heads over a tiled
/// chunk's tokens.
fn the_decoders_flash_attention_matches_its_materialized_trio() {
    let cfg = VideoVaeConfig::real();
    let (heads, head_dim) = (cfg.decoder_num_attention_heads, cfg.decoder_attention_head_dim);
    // The decoder's own device and kernel table, not the DiT's.
    let cx = minimaxh3::video_vae::Ctx::new(None);
    if attn_mode(&cx.gpu, head_dim) != AttnMode::Flash {
        brain_testutil::skip_unavailable("this card cannot run the flash-attention family: both arms would be the trio");
        return;
    }
    let seq_len = 7 * 256 + cfg.decoder_num_register_tokens + 1;
    let n = (seq_len * heads * head_dim) as usize;
    let up = |v: &[f32]| cx.gpu.storage_init("decoder attention check", v);
    let (q, k, v) = (up(&fill(n, 1)), up(&fill(n, 2)), up(&fill(n, 3)));
    let run = |mode| cx.gpu.read(&decoder_attention(&cx.gpu, mode, &q, &k, &v, seq_len, heads, head_dim), n);
    let (want, got) = (run(AttnMode::Materialized), run(AttnMode::Flash));
    let max_abs = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let spread = want.iter().fold(0f32, |m, x| m.max(x.abs()));
    assert!(spread > 1e-3, "the trio's output is ~0 - the comparison would be vacuous");
    assert!(max_abs < 1e-4, "flash vs trio at the decoder's shape: max_abs {max_abs:.3e}");
}

/// The int8 tier a generation holds its DiT at stays close to the fp32
/// model it quantizes, and with two activation levels computes what its
/// int8 weights do. The widths are the tiny config's widened to whole
/// quantization groups.
fn an_int8_resident_model_tracks_the_fp32_one() {
    let t = Tiny::new(H3TransformerConfig { hidden_size: 64, text_dim: 32, ..H3TransformerConfig::tiny() });
    if !Ctx::new(None).ops.caps().numeric.int8_dot {
        brain_testutil::skip_unavailable("this card has no packed int8 dot product");
        return;
    }
    let run = |tensors: &dyn TensorSource, tier: Dtype, act_levels: ActLevels| {
        let m = t.resident_from(tensors, tier, None);
        let _plan = m.prepare(tensors, &Tiny::TIMESTEPS).expect("prepare");
        m.forward(&PackedInputs { act_levels, ..t.inputs(None, ROW_CHUNK) })
    };
    let cosine = |a: &[f32], b: &[f32]| {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
        let norm = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        dot / (norm(a) * norm(b))
    };
    let fp32 = run(&t.tensors, Dtype::F32, ActLevels::One);
    // One activation level against the fp32 model: 1 - cosine measured up to
    // ~1.6e-6 across seeds on this config.
    let one = run(&t.tensors, Dtype::I8, ActLevels::One);
    assert!(cosine(&one.video, &fp32.video) > 0.999_99, "One: video cosine {}", cosine(&one.video, &fp32.video));
    assert!(cosine(&one.audio, &fp32.audio) > 0.999_99, "One: audio cosine {}", cosine(&one.audio, &fp32.audio));
    // Two levels compute what the int8 weights do: against the fp32 model on
    // those same rounded weights, what is left is the activation's
    // second-level rounding, whatever the weights' own rounding costs a given
    // seed against the unrounded model: 1 - cosine measured near 1e-12.
    let rounded = Int8Rounded { inner: &t.tensors, cfg: t.cfg, blocks: 0..t.cfg.num_layers as usize, kinds: &[], group: model::int8::GROUP };
    let exact = run(&rounded, Dtype::F32, ActLevels::One);
    let two = run(&t.tensors, Dtype::I8, ActLevels::Two);
    assert!(cosine(&two.video, &exact.video) > 1.0 - 1e-9, "Two: video cosine {} against the rounded weights", cosine(&two.video, &exact.video));
    assert!(cosine(&two.audio, &exact.audio) > 1.0 - 1e-9, "Two: audio cosine {} against the rounded weights", cosine(&two.audio, &exact.audio));
}

/// Dense flash attention against sparse attention at the shapes of a ref2va
/// with one image reference at 640x352 and 768x448, 121 frames. A
/// measurement, not a check: it prints.
fn sparse_attention_time_at_a_real_shape() {
    let cx = Ctx::new(None);
    for video in [[31, 11, 20], [31, 14, 24]] {
        let segments = [Segment::Dense { rows: 4200 }, Segment::Dense { rows: 4096 }, Segment::Dense { rows: 414 }, Segment::Video { grid: video }];
        let rows = segments.iter().map(Segment::rows).sum::<u32>();
        let (heads, d) = (56u32, 128u32);
        let inner = heads * d;
        let mut rng = data::rng::Lcg::new(1);
        let slab = cx.upload(&rng.vec_scaled((rows * 3 * inner) as usize, 1.0));
        let gate = cx.upload(&rng.vec_scaled((rows * inner) as usize, 0.1));
        let time = |what: &str, run: &dyn Fn() -> DeviceBuffer| {
            cx.gpu.read(&run(), 1);
            let t = std::time::Instant::now();
            for _ in 0..3 {
                cx.gpu.read(&run(), 1);
            }
            println!("{what}: {:.0} ms per attention ({} backend, {rows} rows, video {video:?})", t.elapsed().as_secs_f64() * 1000.0 / 3.0, gpu_core::backend_name());
        };
        time("dense flash", &|| {
            let out = cx.gpu.storage((rows * inner) as u64);
            cx.gpu.submit(&[], &[model::block::flash_bidir_step(&cx.gpu, minimaxh3::block::FLASH_IDS, 1, heads, rows, d, inner, &slab, &out)]);
            out
        });
        for sparsity in [0.0, 0.75, 0.9] {
            let plan = DevicePlan::new(&cx.gpu, TileGeometry::new(&segments), (sparsity, sparsity));
            time(&format!("vsa {sparsity} with gates"), &|| attention(&cx.gpu, &plan, &slab, Some(&gate), heads, d));
        }
    }
}

gpu_core::card_tests!(
    sparse_attention_computes_the_host_reference,
    a_gated_dit_runs_resident_and_streamed_alike,
    row_chunks_compute_what_one_chunk_computes,
    the_dits_flash_attention_matches_its_materialized_trio,
    the_decoders_flash_attention_matches_its_materialized_trio,
    an_int8_resident_model_tracks_the_fp32_one;
    ignored: sparse_attention_time_at_a_real_shape,
);
