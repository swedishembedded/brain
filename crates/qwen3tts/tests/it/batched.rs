// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The batched decoder against the per-stream device path, on the checkpoint the model store
//! resolves: greedy, so both are deterministic. They run the same model in the same fp32 with
//! different reduction orders, so a long clip may part ways at a near-tie; what is asserted is
//! that they agree on the start of every clip, and that streams decoded together are each the
//! stream decoded alone. Skipped when the store holds no Qwen3-TTS checkpoint.

use capability::CancelToken;
use data::tokenizer::Tokenizer;
use qwen3tts::batched::{BatchedDecoder, Job};
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
    let mut dec = BatchedDecoder::load_on(&gpu, &paths, 4).unwrap();
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
