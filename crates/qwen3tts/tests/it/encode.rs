// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The `encode` action on a real recording and the store's Base checkpoint:
//! its codes are the codec's CPU-path codes of the clip at the codec's rate,
//! its x-vector is the speaker encoder's, its token ids are exactly the ones
//! generation tokenises the same text into, and a clip at another sample rate
//! is accepted. The timing test (ignored: it measures, it does not gate) is
//! the throughput number the ledger records.

use std::sync::Arc;
use std::time::Instant;

use capability::{Blob, Invocation, Media, Outcome, Provider, Registry};
use data::tokenizer::Tokenizer;
use serde_json::json;

const CLIP: &str = "tts/voice-clone-example-voice.wav";

/// The real clip and the store's checkpoint, or a skip naming what is missing.
fn fixture() -> Option<(qwen3tts::TtsPaths, Vec<u8>)> {
    let paths = crate::store::store_paths()?;
    let Ok(bytes) = std::fs::read(brain_testutil::testdata(CLIP)) else {
        brain_testutil::skip(&format!("testdata/{CLIP} absent (make testdata/fetch)"));
        return None;
    };
    Some((paths, bytes))
}

fn run(paths: &qwen3tts::TtsPaths, inv: Invocation) -> Outcome {
    let mut reg = Registry::new();
    reg.register(Arc::new(qwen3tts::caps::TtsProvider::new(Some(paths.clone()))) as Arc<dyn Provider>);
    reg.run(qwen3tts::caps::MODEL, "encode", inv, &mut |_| {}).unwrap()
}

fn words<const N: usize>(blob: &Blob) -> Vec<[u8; N]> {
    let (chunks, rest) = blob.bytes.as_chunks::<N>();
    assert!(rest.is_empty());
    chunks.to_vec()
}

#[test]
fn encode_returns_the_codec_codes_the_x_vector_and_the_generation_token_ids_of_a_real_clip() {
    let Some((paths, bytes)) = fixture() else { return };
    let text = "Det här är en mening på svenska.";
    let out = run(&paths, Invocation::new().blob("audio", Blob::new(Media::Bytes, bytes.clone())).set("text", json!(text)));

    let wav = audio::wav::parse(&bytes).unwrap();
    let codec = mimi::Codec::load_inference(&paths.codec);
    let want: Vec<u32> = codec.encode(&audio::resample::rational(&wav.samples, codec.cfg.input_sample_rate, wav.sample_rate));
    let codes: Vec<u32> = words::<2>(&out.blobs["codes"]).into_iter().map(|w| u32::from(u16::from_le_bytes(w))).collect();
    assert_eq!(codes.len(), want.len(), "the CPU-path frame count");
    if gpu_core::backend_name() == "cpu" {
        assert!(codes == want, "the codes are the codec's CPU-path encode of the clip");
    } else {
        // On a card the residual argmin's near-ties may flip (the device sums
        // reassociate); the semantic codebook and nearly every code agree.
        let agree = codes.iter().zip(&want).filter(|(a, b)| a == b).count() as f64 / want.len() as f64;
        let cb0 = codes.iter().step_by(16).zip(want.iter().step_by(16)).filter(|(a, b)| a == b).count() as f64 / (want.len() / 16) as f64;
        assert!(agree >= 0.99 && cb0 >= 0.999, "device codes agree with the CPU path's at {agree:.4} (cb0 {cb0:.4})");
    }
    assert_eq!(out.outputs["frames"], json!(want.len() / 16));
    assert_eq!(out.outputs["codebooks"], json!(16));
    assert_eq!(out.outputs["frame_rate"], json!(12.5));
    assert_eq!(out.outputs["sample_rate_in"], json!(wav.sample_rate));
    let seconds = out.outputs["seconds"].as_f64().unwrap();
    assert!((seconds - wav.samples.len() as f64 / f64::from(wav.sample_rate)).abs() < 1e-9);

    let speaker = ecapatdnn::SpeakerEncoder::load_inference(&paths.speaker);
    let xvector: Vec<f32> = words::<4>(&out.blobs["xvector"]).into_iter().map(f32::from_le_bytes).collect();
    assert!(xvector == speaker.embed_wav(&wav.samples, wav.sample_rate), "the x-vector is the speaker encoder's");
    assert_eq!(out.outputs["xvector_dim"], json!(xvector.len()));

    let producers = out.outputs["producers"].as_object().expect("the producers' digests");
    assert_eq!(producers.keys().map(String::as_str).collect::<Vec<_>>(), ["codec", "speaker_encoder", "tokenizer"]);
    for (what, digest) in producers {
        let hex = digest.as_str().and_then(|d| d.strip_prefix("sha256:")).unwrap_or_default();
        assert!(hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_hexdigit()), "{what}: {digest}");
    }
    assert_eq!(out.outputs["producers"], serde_json::to_value(qwen3tts::encode::producers(&paths).unwrap()).unwrap(), "the same files, the same digests");
    let ids: Vec<u32> = words::<4>(&out.blobs["text_ids"]).into_iter().map(u32::from_le_bytes).collect();
    let tok = qwen3tts::prompt::load_tokenizer(&paths.ckpt_dir).unwrap();
    assert_eq!(ids, tok.encode(&qwen3tts::pipeline::assistant_text(text)), "the ids generation tokenises the request into");
    let (role, body) = qwen3tts::pipeline::split_input_ids(&ids).unwrap();
    assert_eq!((role.as_slice(), body.as_slice()), (&ids[..3], &ids[3..ids.len() - 5]), "the role and text spans a shard records");

    // The same clip at 16 kHz: accepted at its own rate, about as many frames.
    let at16 = audio::wav::encode(&audio::resample::rational(&wav.samples, 16_000, wav.sample_rate), 16_000);
    let low = run(&paths, Invocation::new().blob("audio", Blob::new(Media::Bytes, at16)));
    assert_eq!(low.outputs["sample_rate_in"], json!(16_000));
    assert!(!low.blobs.contains_key("text_ids"), "no text, no ids");
    let frames = low.outputs["frames"].as_u64().unwrap() as i64;
    assert!((frames - (want.len() / 16) as i64).abs() <= 1, "{frames} frames at 16 kHz");
}

/// Audio seconds encoded per wall second on this machine through one loaded
/// encoder (the resident instance's shape): one request at a time, and
/// batches of 8 and 32 ten-second recordings encoded together.
#[test]
#[ignore = "timing measurement, not a gate"]
fn encode_throughput() {
    let Some((paths, bytes)) = fixture() else { return };
    let load = Instant::now();
    let encoder = qwen3tts::encode::Encoder::load(&paths).unwrap();
    eprintln!("encoder load: {:?} on {}", load.elapsed(), gpu_core::backend_name());
    // A corpus utterance is about ten seconds: the clip, repeated to that.
    let wav = audio::wav::parse(&bytes).unwrap();
    let ten: Vec<f32> = wav.samples.iter().copied().cycle().take(10 * wav.sample_rate as usize).collect();
    let req = qwen3tts::encode::Request::parse(&audio::wav::encode(&ten, wav.sample_rate), Some("en mening".to_string())).unwrap();
    encoder.encode(&req);
    for batch in [1usize, 8, 32] {
        let reqs: Vec<&qwen3tts::encode::Request> = std::iter::repeat_n(&req, batch).collect();
        encoder.encode_batch(&reqs);
        let t = Instant::now();
        let rounds = 64 / batch;
        for _ in 0..rounds {
            encoder.encode_batch(&reqs);
        }
        let wall = t.elapsed().as_secs_f64();
        eprintln!("batch {batch:2}: {:.1} audio s per wall s", req.seconds() * (rounds * batch) as f64 / wall);
    }
}
