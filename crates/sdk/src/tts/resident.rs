// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`ResidentTts`]: Qwen3-TTS loaded once and kept, for a caller that speaks
//! many short pieces. [`TtsPipeline`] holds only paths and reloads every
//! checkpoint on each call, which is right for one clip and costs seconds per
//! sentence for a conversation.

use std::sync::Mutex;

use capability::CancelToken;

use super::{Audio, Backend, TtsOptions, TtsPipeline, SAMPLE_RATE};
use crate::{Error, Result};

/// A synthesizer with its checkpoints resident, the Talker and the MTP on the
/// GPU when the process has one. One request is spoken at a
/// time; a second caller waits for the first.
pub struct ResidentTts {
    engine: Mutex<qwen3tts::engine::ResidentEngine>,
}

impl std::fmt::Debug for ResidentTts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResidentTts").field("arch", &"qwen3tts").finish()
    }
}

impl TtsPipeline {
    /// The resident form of this pipeline: its checkpoints loaded once and
    /// kept. Qwen3-TTS only; a CosyVoice-resolved pipeline has none and returns
    /// [`Error::MissingArgument`].
    pub fn resident(&self) -> Result<ResidentTts> {
        let Backend::Qwen3Tts(paths) = &self.backend else {
            return Err(Error::MissingArgument("cosyvoice: has no resident form".to_string()));
        };
        let engine = qwen3tts::engine::ResidentEngine::load_with(paths, qwen3tts::engine::Placement::ambient()).map_err(Error::Backend)?;
        Ok(ResidentTts { engine: Mutex::new(engine) })
    }
}

impl ResidentTts {
    /// Speaker-free text-to-speech, as [`TtsPipeline::speak_with`] but without
    /// reloading anything.
    pub fn speak_with(&self, text: &str, opts: TtsOptions) -> Result<Audio> {
        self.speak_stream(text, opts, &CancelToken::default(), &mut |_| {})
    }

    /// [`ResidentTts::speak_with`], handing each decoded chunk of audio to
    /// `on_audio` as it is produced and stopping when `cancel` fires.
    ///
    /// On a device engine the first chunk arrives once about 0.6 s of speech
    /// has been generated and the rest as the utterance grows; on a host engine
    /// chunks are decoded from the finished utterance, so speak sentence-sized
    /// pieces for a short wait. A request cancelled before it starts delivers
    /// no audio; one cancelled part-way returns [`Error::Cancelled`] after the
    /// chunks already delivered. Either way the synthesizer stays usable.
    pub fn speak_stream(&self, text: &str, opts: TtsOptions, cancel: &CancelToken, on_audio: &mut dyn FnMut(&[f32])) -> Result<Audio> {
        let gen_opts = opts.to_gen_opts();
        let mut engine = self.engine.lock().map_err(|_| Error::Backend("qwen3tts: the resident engine lock is poisoned".to_string()))?;
        let samples = engine
            .speak_streaming(text, &opts.lang, &gen_opts, cancel, &mut |pcm, _seq| on_audio(pcm))
            .map_err(|e| if e == "cancelled" { Error::Cancelled } else { Error::Backend(e) })?;
        Ok(Audio::new(samples, SAMPLE_RATE))
    }

    /// Where the time of the last completed request went.
    pub fn last_timings(&self) -> Result<qwen3tts::SpeakTimings> {
        let engine = self.engine.lock().map_err(|_| Error::Backend("qwen3tts: the resident engine lock is poisoned".to_string()))?;
        Ok(engine.last_timings())
    }
}
