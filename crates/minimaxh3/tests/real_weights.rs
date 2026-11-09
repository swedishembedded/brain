// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MiniMax-H3's DiT against its real-weight reference: investigations that
//! need the real checkout in the model store and its golden
//! (`tools/goldens/minimaxh3_dit_real_dump_reference.py`), so they run only
//! when asked for, with `--ignored`. Those that run on cards take them as an
//! argument; the int8 weight-rounding sweeps run on the CPU:
//!
//! ```text
//! cargo test -p brain-minimaxh3 --test real_weights -- --ignored --device gpu0,gpu1 --backend cuda
//! cargo test -p brain-minimaxh3 --test real_weights -- --ignored int8_weight_rounding
//! ```
//!
//! Swedish Embedded AB implements numerical parity checks that hold ported
//! models to their reference implementations for its clients. If your team
//! needs expertise in validating large diffusion models layer by layer, you
//! can procure our services by sending an email to info@swedishembedded.com.

use gpu_core::select::Dtype;
use minimaxh3::block;
use minimaxh3::config::H3TransformerConfig;
use minimaxh3::model::{forward_workspace_bytes, resident_cost, H3Transformer, Int8Rounded, PackedInputs};
use model::StreamPlan;

/// The model store's MiniMax-H3 checkout root - `None` when the store holds
/// none.
fn store_root() -> Option<String> {
    brain_testutil::model_dir("MiniMaxAI/MiniMax-H3").filter(|root| std::path::Path::new(root).is_dir())
}

/// The real checkpoint and its real-weight golden
/// (`tools/goldens/minimaxh3_dit_real_dump_reference.py`), with the
/// golden's provenance proven against this config - or `None` (skipped,
/// saying why).
fn real_dit_golden() -> Option<(minimaxh3::caps::Paths, Vec<checkpoint::safetensors::StTensor>)> {
    let Some(root) = store_root() else {
        brain_testutil::skip("no MiniMax-H3 checkout in the model store to load from");
        return None;
    };
    let paths = minimaxh3::caps::Paths::resolve(&root);
    if !checkpoint::safetensors::has_model_weights(std::path::Path::new(&paths.dit)) {
        brain_testutil::skip(&format!("{} has no complete weight set yet - checkpoint not (yet) downloaded", paths.dit));
        return None;
    }
    let fixture_dir = brain_testutil::testdata_path("golden/minimaxh3/dit_real");
    let fixture_file = fixture_dir.join("minimaxh3_dit_real.safetensors");
    if !fixture_file.is_file() {
        brain_testutil::skip(&format!(
            "{} not found - run tools/goldens/minimaxh3_dit_real_dump_reference.py --checkpoint {} --out {}",
            fixture_file.display(),
            paths.dit,
            fixture_dir.display()
        ));
        return None;
    }
    let cfg = H3TransformerConfig::real();
    // Golden/checkpoint pairing, proven rather than assumed - see
    // brain_testutil::golden's own module doc.
    let src = brain_testutil::golden::Source::open(&fixture_dir, "tools/goldens/minimaxh3_dit_real_dump_reference.py")?;
    let ok = src.require(&[
        ("num_attention_heads", cfg.num_attention_heads as i64),
        ("attention_head_dim", cfg.attention_head_dim as i64),
        ("hidden_size", cfg.hidden_size as i64),
        ("num_layers", cfg.num_layers as i64),
        ("num_refiner_layers", cfg.num_refiner_layers as i64),
        ("ffn_dim", cfg.ffn_dim as i64),
        ("in_channels", cfg.in_channels as i64),
        ("audio_in_channels", cfg.audio_in_channels as i64),
        ("text_dim", cfg.text_dim as i64),
        ("freq_dim", cfg.freq_dim as i64),
        ("time_embed_hidden_dim", cfg.time_embed_hidden_dim as i64),
        ("time_embed_dim", cfg.time_embed_dim as i64),
        ("rope_freq_dim", cfg.rope_freq_dim as i64),
        ("rope_theta_x1000", (cfg.rope_theta * 1000.0) as i64),
    ]);
    if !ok {
        return None;
    }
    let fx = checkpoint::safetensors::read(fixture_file.to_str().expect("fixture path is valid UTF-8")).expect("read golden fixture");
    Some((paths, fx))
}

/// The real-weight golden's packed inputs, owned.
struct GoldenInputs {
    hidden_states: Vec<f32>,
    audio_hidden_states: Vec<f32>,
    encoder_hidden_states: Vec<f32>,
    timestep: Vec<f32>,
    timestep_indices: Vec<u32>,
    token_tags: Vec<u32>,
    position_ids: Vec<f32>,
    video_indices: Vec<u32>,
    audio_indices: Vec<u32>,
    text_indices: Vec<u32>,
}

impl GoldenInputs {
    fn read(fx: &[checkpoint::safetensors::StTensor]) -> GoldenInputs {
        let get = |name: &str| -> Vec<f32> { fx.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("golden fixture tap {name:?} missing")).data.clone() };
        let get_u32 = |name: &str| -> Vec<u32> { get(name).iter().map(|&f| f.round() as u32).collect() };
        GoldenInputs {
            hidden_states: get("input_hidden_states"),
            audio_hidden_states: get("input_audio_hidden_states"),
            encoder_hidden_states: get("input_encoder_hidden_states"),
            timestep: get("input_timestep"),
            timestep_indices: get_u32("input_timestep_indices"),
            token_tags: get_u32("input_token_tags"),
            position_ids: get("input_position_ids"),
            video_indices: get_u32("input_video_indices"),
            audio_indices: get_u32("input_audio_indices"),
            text_indices: get_u32("input_text_indices"),
        }
    }

    fn inputs(&self, act_levels: block::ActLevels) -> PackedInputs<'_> {
        PackedInputs {
            hidden_states: &self.hidden_states,
            audio_hidden_states: &self.audio_hidden_states,
            encoder_hidden_states: &self.encoder_hidden_states,
            timestep: &self.timestep,
            timestep_indices: &self.timestep_indices,
            token_tags: &self.token_tags,
            position_ids: &self.position_ids,
            video_indices: &self.video_indices,
            audio_indices: &self.audio_indices,
            text_indices: &self.text_indices,
            act_levels,
            vsa: None,
            row_chunk: block::ROW_CHUNK,
        }
    }
}

/// Whether the int8 tier's WEIGHT rounding alone moves the end result:
/// the fp32 streaming forward over the checkpoint as the int8 tier
/// stores it, against the reference, every golden tap. Runs on the CPU
/// device, so it separates weight rounding from the int8 compute arm
/// without a card.
fn int8_weight_rounding_alone_against_the_real_reference() {
    let Some((paths, fx)) = real_dit_golden() else {
        return;
    };
    let cfg = H3TransformerConfig::real();
    let want = |name: &str| fx.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("golden fixture tap {name:?} missing")).data.clone();
    let golden = GoldenInputs::read(&fx);
    let reader = minimaxh3::caps::open_dit_reader(&paths.dit).unwrap_or_else(|e| panic!("open_dit_reader: {e}"));
    let rounded = Int8Rounded { inner: &reader, cfg, blocks: 0..cfg.num_layers as usize, kinds: &[], group: model::int8::GROUP };
    let dit = minimaxh3::dit_shard::StreamingDit::open(&cfg, Some("cpu"));
    let t0 = std::time::Instant::now();
    let (out, taps) = H3Transformer::forward_streaming_with_taps(&rounded, &cfg, &dit, &golden.inputs(block::ActLevels::One));
    eprintln!("fp32 forward over int8-rounded weights in {:.0}s", t0.elapsed().as_secs_f32());
    let mut table = brain_testutil::parity::Table::new(0.0, f64::INFINITY);
    table.check("refiner_out", &taps.refiner_out, &want("tap_refiner_out"));
    table.check("block0_out", &taps.block0_out, &want("tap_block0_out"));
    table.check("block25_out (mid)", &taps.mid_block_out, &want("tap_block25_out"));
    table.check("block49_out (last)", &taps.last_block_out, &want("tap_block49_out"));
    table.check("output_video", &out.video, &want("output_video"));
    table.check("output_audio", &out.audio, &want("output_audio"));
    table.print();
    assert!(out.video.iter().chain(&out.audio).all(|x| x.is_finite()), "non-finite output");
}

/// Which DiT blocks' int8 weight rounding moves the end result: the fp32
/// streaming forward with only a range of blocks rounded, halving toward
/// the half that costs the video output more, printing every range.
fn int8_weight_rounding_bisected_over_the_blocks() {
    let Some((paths, fx)) = real_dit_golden() else {
        return;
    };
    let cfg = H3TransformerConfig::real();
    let want_video = fx.iter().find(|t| t.name == "output_video").expect("output_video").data.clone();
    let golden = GoldenInputs::read(&fx);
    let reader = minimaxh3::caps::open_dit_reader(&paths.dit).unwrap_or_else(|e| panic!("open_dit_reader: {e}"));
    let dit = minimaxh3::dit_shard::StreamingDit::open(&cfg, Some("cpu"));
    let cosine = |blocks: std::ops::Range<usize>| {
        let rounded = Int8Rounded { inner: &reader, cfg, blocks: blocks.clone(), kinds: &[], group: model::int8::GROUP };
        let out = H3Transformer::forward_streaming(&rounded, None, &cfg, &dit, &golden.inputs(block::ActLevels::One));
        let c = brain_testutil::parity::compare(&out.video, &want_video).0;
        eprintln!("blocks {blocks:?} rounded: output_video cosine {c:.6}");
        c
    };
    let mut range = 26..cfg.num_layers as usize;
    while range.len() > 1 {
        let mid = range.start + range.len() / 2;
        let (lo, hi) = (range.start..mid, mid..range.end);
        range = if cosine(lo.clone()) < cosine(hi.clone()) { lo } else { hi };
    }
    eprintln!("the most damaging single block: {}", range.start);
}

/// Which linear kind's int8 weight rounding moves the end result, and
/// whether group-16 scales hold it: the fp32 streaming forward with one
/// kind rounded across every block, then every kind at group 16.
fn int8_weight_rounding_by_linear_kind_and_group() {
    let Some((paths, fx)) = real_dit_golden() else {
        return;
    };
    let cfg = H3TransformerConfig::real();
    let want = |name: &str| fx.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("golden fixture tap {name:?} missing")).data.clone();
    let (want_video, want_audio) = (want("output_video"), want("output_audio"));
    let golden = GoldenInputs::read(&fx);
    let reader = minimaxh3::caps::open_dit_reader(&paths.dit).unwrap_or_else(|e| panic!("open_dit_reader: {e}"));
    let dit = minimaxh3::dit_shard::StreamingDit::open(&cfg, Some("cpu"));
    let cases: [(&str, &'static [&'static str], usize); 7] = [
        ("to_q", &[".to_q.weight"], 32),
        ("to_k", &[".to_k.weight"], 32),
        ("to_v", &[".to_v.weight"], 32),
        ("to_out", &[".to_out.0.weight"], 32),
        ("fc1 (value+gate)", &[".ff.net.0.proj.weight"], 32),
        ("fc2", &[".ff.net.2.weight"], 32),
        ("every kind, group 16", &[], 16),
    ];
    for (label, kinds, group) in cases {
        let rounded = Int8Rounded { inner: &reader, cfg, blocks: 0..cfg.num_layers as usize, kinds, group };
        let out = H3Transformer::forward_streaming(&rounded, None, &cfg, &dit, &golden.inputs(block::ActLevels::One));
        let (v, a) = (brain_testutil::parity::compare(&out.video, &want_video).0, brain_testutil::parity::compare(&out.audio, &want_audio).0);
        eprintln!("{label}: output_video cosine {v:.6}, output_audio cosine {a:.6}");
    }
}

/// What int8 weight rounding costs a REAL generation's forward: a prompt
/// through the real text encoder, a 256x256 five-second canvas of
/// Gaussian latents, and the dual schedule's own timesteps at its first
/// and middle steps - the fp32 forward over the checkpoint against the
/// fp32 forward over it as the int8 tier stores it. CPU only.
///
/// Measured 0.99987-0.99996 cosine across both streams and steps. The
/// real-weight golden's synthetic inputs (Gaussian x 0.3 text embeddings
/// and latents) are not the yardstick for this: there the same rounding
/// lands at 0.973 video / 0.911 audio, spread evenly over every block and
/// linear kind and barely moved by group-16 scales - the network
/// amplifying any weight perturbation off its input distribution, which
/// the tests above measure.
fn int8_weight_rounding_on_real_generation_inputs() {
    let Some((paths, _)) = real_dit_golden() else {
        return;
    };
    let cfg = H3TransformerConfig::real();
    let vae = minimaxh3::video_vae::VideoVaeConfig::real();
    let text = minimaxh3::caps::encode_presentation(&paths, "a red fox trots through fresh snow at dawn, its breath steaming", &[], Some("cpu")).expect("real text conditioning");
    let frames = minimaxh3::pipeline::align_num_frames((minimaxh3::pipeline::MIN_DURATION_S * minimaxh3::pipeline::FPS) as u32, vae.clip_length, vae.tokens_chunk_size()).expect("frames");
    let lat_f = minimaxh3::pipeline::video_latent_num_frames(frames, vae.clip_length, vae.tokens_chunk_size());
    let audio = minimaxh3::pipeline::audio_latent_num_frames(frames, minimaxh3::pipeline::FPS, minimaxh3::pipeline::AUDIO_LATENTS_PER_SECOND);
    let side = 256 / vae.spatial_compression_ratio();
    let packed = minimaxh3::pipeline::build_packed_sequence(&text.token_tags, lat_f, side, side, audio, cfg.patch_size, minimaxh3::pipeline::AUDIO_CHANNELS, &[]);
    let mut rng = data::rng::Rng::new(20261105); // seed: the int8 rounding comparison's one noise draw
    let mut noise = |n: usize| -> Vec<f32> { (0..n).map(|_| rng.next_gaussian() as f32).collect() };
    let video = noise(packed.video_indices.len() * cfg.video_patch_dim() as usize);
    let audio_noise = noise(packed.audio_indices.len() * cfg.audio_in_channels as usize);
    let mut sched = minimaxh3::schedule::DualSchedule::new();
    sched.set_timesteps(20);
    let reader = minimaxh3::caps::open_dit_reader(&paths.dit).unwrap_or_else(|e| panic!("open_dit_reader: {e}"));
    let rounded = Int8Rounded { inner: &reader, cfg, blocks: 0..cfg.num_layers as usize, kinds: &[], group: model::int8::GROUP };
    let dit = minimaxh3::dit_shard::StreamingDit::open(&cfg, Some("cpu"));
    eprintln!("{} packed rows ({} video, {} audio, {} text)", packed.position_ids.len() / 3, packed.video_indices.len(), packed.audio_indices.len(), packed.text_indices.len());
    for step in [0, sched.num_steps() / 2] {
        let (vt, at) = sched.current_timesteps(step);
        let (timestep, timestep_indices) = minimaxh3::pipeline::build_row_timesteps(&packed.video_indices, &packed.audio_indices, 0, 0, packed.text_indices.len(), vt, at, vt.max(minimaxh3::pipeline::KEYFRAME_NOISE_AUG), 1.0);
        let inp = PackedInputs {
            hidden_states: &video,
            audio_hidden_states: &audio_noise,
            encoder_hidden_states: &text.embeds,
            timestep: &timestep,
            timestep_indices: &timestep_indices,
            token_tags: &packed.token_tags,
            position_ids: &packed.position_ids,
            video_indices: &packed.video_indices,
            audio_indices: &packed.audio_indices,
            text_indices: &packed.text_indices,
            act_levels: block::ActLevels::One,
            vsa: None,
            row_chunk: block::ROW_CHUNK,
        };
        let t = std::time::Instant::now();
        let exact = H3Transformer::forward_streaming(&reader, None, &cfg, &dit, &inp);
        let int8 = H3Transformer::forward_streaming(&rounded, None, &cfg, &dit, &inp);
        let (v, a) = (brain_testutil::parity::compare(&int8.video, &exact.video), brain_testutil::parity::compare(&int8.audio, &exact.audio));
        eprintln!("step {step} (video t {vt:.3}, audio t {at:.3}): video (cosine, max_abs) {v:?}, audio {a:?}; {:.0}s", t.elapsed().as_secs_f32());
        assert!(v.0 > 0.9995 && a.0 > 0.9995, "step {step}: the int8 tier's weights move a real forward: video cosine {:.6}, audio {:.6}", v.0, a.0);
    }
}

/// What the int8 tier costs the end result, measured on the real
/// checkpoint at the golden's own inputs, all 50 blocks: every golden tap
/// of the int8 model against the reference's, then the output at one
/// activation level and at two, printed for both streams. Needs cards with the packed int8 dot product
/// (the CPU JIT has none), two of them for the ~21 GiB int8 model.
fn int8_activation_levels_against_the_real_reference_output() {
    let Some((paths, fx)) = real_dit_golden() else {
        return;
    };
    let cfg = H3TransformerConfig::real();
    let want = |name: &str| fx.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("golden fixture tap {name:?} missing")).data.clone();
    let golden = GoldenInputs::read(&fx);
    let reader = minimaxh3::caps::open_dit_reader(&paths.dit).unwrap_or_else(|e| panic!("open_dit_reader: {e}"));
    let t0 = std::time::Instant::now();
    // Split over two cards, as a generation's own workspace makes the
    // placement do: the model alone sits within a GiB of one P40's
    // allocatable memory, too close to share it with this measurement.
    let rows = (golden.position_ids.len() / 3) as u32;
    let cost = resident_cost(&cfg, Dtype::I8, false);
    let cards: Vec<(usize, u64)> = StreamPlan::cards(None, forward_workspace_bytes(&cfg, rows, block::ActLevels::Two)).into_iter().map(|(i, b)| (i, b.min(cost.total() * 3 / 5))).collect();
    let plan = StreamPlan::fit(&cost, &cards).expect("two cards hold the int8 model");
    let m = H3Transformer::load_on(&reader, cfg, plan, None, Dtype::I8, None).expect("the int8 model on two cards");
    let _plan = m.prepare(&reader, &golden.timestep).expect("prepare");
    eprintln!("int8 model loaded in {:.0}s", t0.elapsed().as_secs_f32());
    let err = |got: &[f32], want: &[f32]| (brain_testutil::parity::compare(got, want), brain_testutil::parity::rel_l2(got, want));
    // Where the int8 model leaves the reference: every tap the golden
    // carries, in forward order - the first that drops names the stage.
    let (_, taps) = m.forward_with_taps(&golden.inputs(block::ActLevels::One));
    let seq_len = golden.position_ids.len() / 3;
    let half = taps.rope_cos.len() / seq_len;
    let first_half_cols = |full: &[f32]| -> Vec<f32> { (0..seq_len).flat_map(|r| full[r * 2 * half..r * 2 * half + half].to_vec()).collect() };
    let mut table = brain_testutil::parity::Table::new(0.0, f64::INFINITY);
    table.check("refiner_out", &taps.refiner_out, &want("tap_refiner_out"));
    table.check("rope_cos", &taps.rope_cos, &first_half_cols(&want("tap_rope_cos")));
    table.check("temb", &taps.temb, &want("tap_temb"));
    table.check("block0_input", &taps.block0_input, &want("tap_block0_input"));
    table.check("block0_attn_out", &taps.block0_attn_out, &want("tap_block0_attn_out"));
    table.check("block0_out", &taps.block0_out, &want("tap_block0_out"));
    table.check("block25_out (mid)", &taps.mid_block_out, &want("tap_block25_out"));
    table.check("block49_out (last)", &taps.last_block_out, &want("tap_block49_out"));
    table.print();
    for levels in [block::ActLevels::One, block::ActLevels::Two] {
        let t = std::time::Instant::now();
        let out = m.forward(&golden.inputs(levels));
        let (v, a) = (err(&out.video, &want("output_video")), err(&out.audio, &want("output_audio")));
        eprintln!("{levels:?}: video (cosine, max_abs) {:?} rel_l2 {:.3e}; audio {:?} rel_l2 {:.3e}; forward {:.1}s", v.0, v.1, a.0, a.1, t.elapsed().as_secs_f32());
        assert!(out.video.iter().chain(&out.audio).all(|x| x.is_finite()), "{levels:?}: non-finite int8 output");
    }
}

/// The REAL checkpoint's DiT, real weights, against a live real-weight
/// `diffusers` reference run - `tools/goldens/minimaxh3_dit_real_dump_reference.py`,
/// NOT `minimaxh3_dit_dump_reference.py` (that one uses tiny dims and
/// small RANDOM weights, which validates the architecture code but
/// structurally cannot catch a bug that only shows up on real,
/// structured weights - a wrong transpose, a wrong axis order, a wrong
/// tensor-name-to-role mapping all look "fine" when every element is
/// equally arbitrary). Checked at block 0, a MIDDLE block, and the LAST
/// block (not just end-to-end) - exactly the audio VAE's own precedent
/// (`import::decode_matches_the_real_reference_numerically`) applied to
/// the DiT for the first time.
///
/// Real DiT weight loading (~66GB bf16 -> ~132GB fp32, streamed) is the
/// expensive part here (~10-20 minutes) - the forward pass itself runs
/// over a 9-row synthetic sequence, seconds not minutes.
fn dit_matches_the_real_reference_numerically_layer_by_layer() {
    let Some((paths, fx)) = real_dit_golden() else {
        return;
    };
    let cfg = H3TransformerConfig::real();
    let get = |name: &str| -> &checkpoint::safetensors::StTensor {
        fx.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("golden fixture tap {name:?} missing"))
    };
    let golden = GoldenInputs::read(&fx);
    let inp = golden.inputs(block::ActLevels::One);
    let position_ids = &golden.position_ids;

    eprintln!("running real DiT one block at a time on the ambient devices (never all 50 blocks resident - see H3Transformer::forward_streaming_with_taps's own doc) ...");
    let t0 = std::time::Instant::now();
    let reader = minimaxh3::caps::open_dit_reader(&paths.dit).unwrap_or_else(|e| panic!("open_dit_reader: {e}"));
    // Whatever placement this machine supports - one stage on a one-card
    // or CPU run, one stage per card on a multi-card one. The reference
    // tolerances below are the SAME either way: the split carries only
    // the residual, host-staged, and an f32 round trip is lossless
    // (`model::shard::set_stream_shard(false)` forces the single-stage arm for a
    // side-by-side check).
    let dit = minimaxh3::dit_shard::StreamingDit::open(&cfg, None);
    eprintln!("  block placement: {}", dit.plan().describe());
    let (out, taps) = H3Transformer::forward_streaming_with_taps(&reader, &cfg, &dit, &inp);
    eprintln!("  done in {:.1}s", t0.elapsed().as_secs_f32());

    assert_eq!(taps.mid_block_index, 25, "golden was dumped at mid_block=25; H3TransformerConfig::real()'s own 50 layers must still land on 25");

    // porting.md's own floor: "cosine >= 0.9999 for networks" at the
    // stage-parity rung. This is stage-by-stage AND real-weight, so it
    // should land far tighter than that floor (float32-noise level, the
    // audio VAE's own real-weight precedent landed at max_abs ~4e-6) -
    // the floor is a refusal threshold, not a target.
    // Same `2*half`-wide reference doubling as `tiny_config_matches_the_
    // real_reference_numerically` below - see `minimaxh3::rope`'s own doc.
    let seq_len = position_ids.len() / 3;
    let half = taps.rope_cos.len() / seq_len;
    let first_half_cols = |full: &[f32]| -> Vec<f32> { (0..seq_len).flat_map(|r| full[r * 2 * half..r * 2 * half + half].to_vec()).collect() };

    let mut r = brain_testutil::parity::Report::new(0.9999);
    r.check("refiner_out", &taps.refiner_out, &get("tap_refiner_out").data);
    r.check("rope_cos", &taps.rope_cos, &first_half_cols(&get("tap_rope_cos").data));
    r.check("rope_sin", &taps.rope_sin, &first_half_cols(&get("tap_rope_sin").data));
    r.check("temb", &taps.temb, &get("tap_temb").data);
    r.check("block0_input", &taps.block0_input, &get("tap_block0_input").data);
    r.check("block0_attn_out", &taps.block0_attn_out, &get("tap_block0_attn_out").data);
    r.check("block0_out", &taps.block0_out, &get("tap_block0_out").data);
    r.check("block25_out (mid)", &taps.mid_block_out, &get("tap_block25_out").data);
    r.check("block49_out (last)", &taps.last_block_out, &get("tap_block49_out").data);
    r.check("output_video", &out.video, &get("output_video").data);
    r.check("output_audio", &out.audio, &get("output_audio").data);
    r.finish("minimaxh3 DiT forward vs REAL-WEIGHT reference, layer by layer");
}

gpu_core::card_tests!(
    ;
    ignored: dit_matches_the_real_reference_numerically_layer_by_layer,
    int8_activation_levels_against_the_real_reference_output;
    ignored_host: int8_weight_rounding_alone_against_the_real_reference,
    int8_weight_rounding_bisected_over_the_blocks,
    int8_weight_rounding_by_linear_kind_and_group,
    int8_weight_rounding_on_real_generation_inputs,
);
