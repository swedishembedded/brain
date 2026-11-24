<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# qwen3tts-sft - roadmap

Scope: Multi-speaker full fine-tuning of Qwen3-TTS (Talker + code predictor) on recorded speech, new languages and named voices, exported as an upstream-layout checkpoint.

The first target is a Swedish 1.7B checkpoint with named voices, trained from a
corpus of open Swedish speech, that loads unchanged in brain and in the
upstream `qwen-tts` package (`Qwen3TTSModel.from_pretrained`, transformers
4.57.3 `AutoModel` after its registration). The recipe that builds the corpus
and drives the run lives with the application that releases the model; this
ledger is the engine work it needs.

## Findings (2026-11-24, read from the code and the upstream package 0.1.1)

- Languages and voices are checkpoint data on both sides. brain's
  `prompt::TtsSpecials::language_id`/`speaker_id` read `talker_config.
  codec_language_id`/`spk_id` from `config.json`; upstream does the same
  (`modeling_qwen3_tts.py` builds `supported_languages`/`supported_speakers`
  from them). A new language is a new `codec_language_id` entry whose codec
  row was trained; no code names a language. Upstream also reads
  `spk_is_dialect[speaker]` for every named speaker when the language is
  `chinese` or `auto`, so an export lists every voice there (`false`).
- Free codec rows in the 1.7B Base `talker.model.codec_embedding` (norm of an
  untrained row is 0.022, of a trained language row 1.0-1.3): 2087, 2095,
  2098, 2105, 2127, 2129, 2131, 2132, 2141 in the language band; the whole
  2860-3071 band is untrained in Base (CustomVoice's preset speakers live
  there at 2861-3066). Rows 2049-2073 are almost all trained although Base
  names only ten languages.
- The upstream CustomVoice entry point defaults to `non_streaming_mode=True`:
  the whole text goes in the prefix over `codec_pad` rows, then `tts_pad +
  codec_bos`, and the trailing text is `tts_pad` only. brain builds only the
  streaming-text layout (`build_xvector_prompt`). A checkpoint served by both
  is trained on both layouts, and brain gains the non-streaming one.
- Upstream drops the speaker encoder for a non-Base checkpoint and refuses
  `generate_custom_voice` unless `tts_model_type == "custom_voice"`; a
  named-voice release is exported as `custom_voice`.
- The 1.7B Base pull is mis-attributed: its `brain.manifest.json` says
  `family: "qwen3asr"`, because `modelstore::recipe`'s `qwen3tts` row is
  pinned to the 0.6B repo only and the broader `qwen3asr` signature claims
  every other TTS repo, a trained release included.
- `sft::run_finetune` trains a causal LM over a flat token stream, not the
  Talker's prompt; the code predictor's trainable form (`MtpModel::
  new_trainable`) takes one frame per forward with host heads, and drops the
  gradients of its two Talker-supplied inputs. Neither is the objective.
- Reusable pieces: `qwen3::Qwen` takes spliced input rows
  (`enable_mm_splices`, `write_img_embeds`) and hands back their gradient
  (`read_d_img_embeds`), with an external head (`forward_hidden`/
  `backward_hidden`); `model::head_ce::HeadCe` is the device head + CE with
  backward; `model::projector::MlpProjector` (depth 1) is
  `small_to_mtp_projection` on the device; `model::hostopt::HostAdam` steps
  host-resident tables; `train::policy_rows` is the tested inference-equal row
  builder; `fit_controlled` brings checkpoints, resume and served jobs.

## Not yet done

- [x] Training shard format `brain-tts-shard-v1` (`qwen3tts::shard`): token
      ids, `[frames, 16]` codes, x-vector, voice, language, completeness, in
      one file (an artifact a job takes by digest), validated when read.
- [x] Prompt layouts as data: one builder that records, per prefix row, which
      codec rows, text tokens and x-vector it sums, for inference and for
      routing gradients; the non-streaming layout beside the streaming one
      (`prompt::PromptPlan`, `TextLayout`; the speech actions take
      `text_layout`, defaulting as the reference does; the in-context clone
      stays streaming-only).
- [ ] `prepare`: utterance manifest (wav, transcript, speaker, voice,
      language) to shards, codes from the codec encoder and x-vectors from the
      speaker encoder on the device, batched.
- [x] The speech SFT objective: the Talker full-weight on the teacher-forced
      prompt (cb0 next-frame CE, end of speech supervised on complete
      utterances only), the code predictor batched over frames (`[frames, 16]`
      positions, same-frame residual CE, weight `0.3` as upstream), gradients
      of the shared inputs routed both ways (code predictor to Talker hidden
      and cb0 row; Talker input rows to every codec table), codec tables and
      named-voice rows on the host optimizer, text embedding and projection
      frozen in the first release. Through `fit_controlled` (resume, events).
      `qwen3tts::speech_sft` (`speech_sft()`, `SpeechSftModel`,
      `SpeechObjective`); gated on tiny random checkpoints (finite differences
      of every parameter family, masking, exact resume, overfit) and on the
      real 0.6B (training rows equal generation's rows bit for bit, both
      layouts, x-vector and named-voice slot).
- [x] New language and voice rows: allocate free codec rows, initialise a
      voice row from its speaker's robust x-vector centroid and a language row
      from a configurable mix of existing rows (`speech_sft::rows`).
- [ ] Measure the objective on a card: the real-checkpoint overfit and its
      frames/s (`speech_sft::tests::the_real_checkpoint_overfits_one_utterance_and_reports_its_frame_rate`,
      written, ignored, refused on the CPU backend) and the cost of the host
      side (row materialisation and gradient routing per micro-batch).
- [ ] Batches shorter than the build: with every row spliced in, `qwen3::Qwen`
      runs its whole block (`needed_len` does not trim a spliced or
      external-head batch), so every micro-batch pays the longest record's
      rows; trim by the batch's own length or bucket records by length.
- [ ] One device per run: each decoder (`qwen3::Qwen::new`) opens its own
      device handle and the heads take a third (`new_like`); a constructor on
      a shared `Gpu` would let all three share one stream.
- [ ] The in-context clone in the non-streaming layout (refused today), and
      training the text projection (frozen in this release).
- [ ] Export to the upstream layout: one `model.safetensors` in upstream names
      and BF16, `config.json` with the trained languages and voices and
      `tts_model_type`, tokenizer and `speech_tokenizer/` carried over; gated
      by a round trip (export, reload in brain, identical tensors).
- [x] Store: a Qwen3-TTS repo is recognised by its own signature for any
      repo, not by a pin list, so the 1.7B Base and trained releases pull as
      `qwen3tts` (`modelstore::recipe::select`: at one tier a signature that
      strictly contains another's wins). A store that already pulled the 1.7B
      under the old rule keeps its `family: "qwen3asr"` manifest until it is
      pulled again.

Gates: tiny random configs for the objective's gradients (finite differences
through Talker, code predictor and the routed tables); teacher-forced rows equal
inference rows on the real checkpoint; overfit of one real utterance; held-out
recogniser CER and speaker similarity against the base checkpoint with matched
seeds.
