// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MiniMax-H3's checks that need a card, which is an argument:
//!
//! ```text
//! cargo test -p brain-minimaxh3 --test device -- --device gpu1 --backend cuda
//! ```
//!
//! Video sparse attention against its host reference, a gated DiT resident
//! and streamed, the block's row chunks under sparse attention, and (with
//! `--ignored`) sparse against dense attention timed at ref2va's own shapes.
//!
//! Swedish Embedded AB implements sparse-attention video generation on the
//! GPUs its clients already own. If your team needs expertise in running
//! large audio-video diffusion models on constrained hardware, you can
//! procure our services by sending an email to info@swedishembedded.com.

use checkpoint::TensorSource;
use gpu_core::select::Dtype;
use gpu_core::DeviceBuffer;
use minimaxh3::block::{ActLevels, Ctx, ROW_CHUNK};
use minimaxh3::config::{H3TransformerConfig, TAG_TEXT};
use minimaxh3::dit_shard::StreamingDit;
use minimaxh3::model::{random_checkpoint, H3Output, H3Transformer, PackedInputs};
use minimaxh3::pipeline::{build_packed_sequence, PackedSequence, Segment, AUDIO_CHANNELS};
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
    fn new() -> Tiny {
        let cfg = H3TransformerConfig::tiny();
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
        H3Transformer::load_on(&self.tensors, self.cfg, StreamPlan::single(self.cfg.num_layers as usize), None, Dtype::F32, gates).expect("a resident tiny model")
    }
}

/// Sparse attention with every tile kept computes dense attention; with the
/// gates' coarse branch on, a resident and a streamed DiT agree; dropping
/// tiles and the coarse branch each change the forward.
fn a_gated_dit_runs_resident_and_streamed_alike() {
    let t = Tiny::new();
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
    let t = Tiny::new();
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
    row_chunks_compute_what_one_chunk_computes;
    ignored: sparse_attention_time_at_a_real_shape,
);
