// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The batched decoder against the per-stream device path, on the checkpoint the model store
//! resolves: greedy, so both are deterministic. They run the same model in the same fp32 with
//! different reduction orders, so a long clip may part ways at a near-tie; what is asserted is
//! that they agree on the start of every clip, and that streams decoded together are each the
//! stream decoded alone. Skipped when the store holds no Qwen3-TTS checkpoint.

use capability::CancelToken;
use data::tokenizer::Tokenizer;
use qwen3tts::batched::{BatchedDecoder, Job, WeightTier};
use qwen3tts::prompt::{self, Prompt, TtsSpecials};
use qwen3tts::{GenOpts, SamplingRequest, TtsPaths};

const TEXTS: [&str; 3] = [
    "The people must be taught to value their liberty above their ease.",
    "A short one.",
    "Tomorrow the board boots on the first try, and the light blinks as it should.",
];
const HEAD: usize = 12 * 16;

fn greedy(max_frames: usize) -> GenOpts {
    GenOpts { max_frames, sampling: SamplingRequest::greedy(), ..GenOpts::default() }
}

fn prompt_for(paths: &TtsPaths, tables: &qwen3tts::npu_gen::TalkerTables, sp: &TtsSpecials, text: &str) -> Prompt {
    let tok = prompt::load_tokenizer(&paths.ckpt_dir).unwrap();
    let ids = tok.encode(&qwen3tts::pipeline::assistant_text(text));
    let (role, body) = qwen3tts::pipeline::split_input_ids(&ids).unwrap();
    prompt::build_xvector_prompt(tables, sp, &role, &body, None, sp.language_id("english").unwrap())
}

/// Decode `jobs` together, answering each one's codes in job order.
fn decode(dec: &mut BatchedDecoder, sp: &TtsSpecials, jobs: Vec<Job>) -> Vec<Vec<u32>> {
    let n = jobs.len();
    let mut queue: Vec<(usize, Job)> = jobs.into_iter().enumerate().collect();
    let mut out: Vec<Option<Vec<u32>>> = vec![None; n];
    dec.run(sp, &mut |free| queue.drain(..free.min(queue.len())).collect(), &mut |_, _| {}, &mut |i, r| out[i] = Some(r.expect("decoded")));
    out.into_iter().map(|o| o.expect("every job finished")).collect()
}

#[test]
fn batched_streams_match_the_per_stream_device_path_and_each_other() {
    let _serial = brain_testutil::env_lock();
    let Some(paths) = crate::store::store_paths() else { return };
    let sp = TtsSpecials::from_config_dir(&paths.ckpt_dir).unwrap();
    let gpu = gpu_core::Gpu::new(&[]);
    let mut dec = BatchedDecoder::load_on(&gpu, &paths, 4, WeightTier::F32).unwrap();
    let prompts: Vec<Prompt> = TEXTS.iter().map(|t| prompt_for(&paths, dec.talker_tables(), &sp, t)).collect();
    let job = |p: &Prompt| Job { prompt: p.clone(), opts: greedy(60).resolved_for(&paths.ckpt_dir), cancel: CancelToken::default() };

    // The per-stream device path the batched decoder replaces.
    let talker = qwen3tts::gen::TalkerGen::load(&paths.talker, 2048);
    let mtp = qwen3tts::mtp::MtpModel::load_inference(&paths.mtp);
    let reference = qwen3tts::pipeline::generate_codes_hooked(&talker, &mtp, &sp, &prompts[0], &greedy(60).resolved_for(&paths.ckpt_dir), &CancelToken::default(), &mut |_| {}).unwrap();

    let alone: Vec<Vec<u32>> = prompts.iter().map(|p| decode(&mut dec, &sp, vec![job(p)]).remove(0)).collect();
    let n = HEAD.min(reference.len()).min(alone[0].len());
    assert!(n >= 4 * 16, "too few frames to compare: {} and {}", reference.len() / 16, alone[0].len() / 16);
    assert_eq!(&alone[0][..n], &reference[..n], "the batched decoder starts every clip as the per-stream path does");

    let together = decode(&mut dec, &sp, prompts.iter().map(job).collect());
    for (i, (t, a)) in together.iter().zip(&alone).enumerate() {
        assert_eq!(t, a, "stream {i} decoded with the others is the stream decoded alone");
    }
}

/// The decoder at the checkpoint's own precision (bf16 weights, half-precision tensor-core products) against the
/// fp32 one: after each prompt the first codebook's distribution is the same to within a small KL with the same best
/// code, and greedy decoding from both starts with (nearly) the same frame.
#[test]
fn the_stored_precision_decoder_matches_the_fp32_one() {
    let _serial = brain_testutil::env_lock();
    let Some(paths) = crate::store::store_paths() else { return };
    let sp = TtsSpecials::from_config_dir(&paths.ckpt_dir).unwrap();
    let gpu = gpu_core::Gpu::new(&[]);
    let mut f32_dec = BatchedDecoder::load_on(&gpu, &paths, 1, WeightTier::F32).unwrap();
    let mut stored = BatchedDecoder::load_on(&gpu, &paths, 1, WeightTier::Stored).unwrap();
    // The published checkpoints are bf16: a stored tier that fell back to fp32 would pass the comparison below.
    assert_eq!(stored.weights_tiers(), (qwen3::Dtype::BF16, qwen3::Dtype::BF16), "the stored tier holds the checkpoint's bf16");
    let softmax = |l: &[f32]| -> Vec<f64> {
        let m = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let e: Vec<f64> = l.iter().map(|&x| (x as f64 - m).exp()).collect();
        let z: f64 = e.iter().sum();
        e.into_iter().map(|x| x / z).collect()
    };
    let argmax = |l: &[f32]| l.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    for text in TEXTS {
        let prompt = prompt_for(&paths, f32_dec.talker_tables(), &sp, text);
        let (a, b) = (f32_dec.prompt_logits(&prompt), stored.prompt_logits(&prompt));
        let (p, q) = (softmax(&a), softmax(&b));
        let kl: f64 = p.iter().zip(&q).filter(|(p, _)| **p > 0.0).map(|(p, q)| p * (p / q.max(1e-300)).ln()).sum();
        eprintln!("{text:?}: KL(fp32 || stored) = {kl:.2e}");
        assert_eq!(argmax(&a), argmax(&b), "{text:?}: the best first code differs");
        assert!(kl < 1e-2, "{text:?}: KL {kl}");
        let job = |p: &Prompt| Job { prompt: p.clone(), opts: greedy(2).resolved_for(&paths.ckpt_dir), cancel: CancelToken::default() };
        let (x, y) = (decode(&mut f32_dec, &sp, vec![job(&prompt)]).remove(0), decode(&mut stored, &sp, vec![job(&prompt)]).remove(0));
        let same = x[..16].iter().zip(&y[..16]).filter(|(a, b)| a == b).count();
        assert!(same >= 14, "{text:?}: the first frame shares {same} of 16 codes: {:?} vs {:?}", &x[..16], &y[..16]);
    }
}
