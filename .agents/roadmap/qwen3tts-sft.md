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
  every other TTS repo, a trained release included. Selection is fixed
  (`cfd296bfe`); `brain pull` now also re-derives a complete repo's manifest
  from its local files and rewrites a stale one (`modelstore::reattribute`),
  and `brain models list` flags a stale row on stderr.
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
  - [x] The served `encode` action (`qwen3tts::encode`, `caps::encode_spec`):
        WAV at any rate -> `codes` (u16), `xvector` (f32), `text_ids` (u32, the
        generation tokenisation), `frames`, `sample_rate_in`, `seconds`; codes
        gated against the CPU-path codec encode on a real clip
        (`tests/it/encode.rs`: equal on the CPU backend, >= 99 % and cb0
        >= 99.9 % on a card). Resident: its own instance key, one encoder
        loaded once on the placed device, every request the dispatcher hands
        over at once encoded as one batch (`Encoder::encode_batch`,
        `caps::run_encode_batch`); `brain_py` `encode_audio`.
  - [x] Encode as batched device passes: the codec (`mimi::Codec::
        encode_batch`, clips padded to a whole frame with per-clip length
        masks, the residual quantizer on the device) and the speaker encoder
        (`ecapatdnn::SpeakerEncoder::embed_batch`, utterances packed along
        time with per-utterance reflect margins and pooling) each one recorded
        pass reading back only codes and x-vectors; the encode instance
        declares device memory, so it is placed on a card when one is
        budgeted. Measured 2026-11-25 on one GH200 (gpu1, its other tenants
        idle):
        - in process, the whole encode (resample, codec, mel, speaker
          encoder) of 10 s clips (`tests/it/encode.rs` `encode_throughput`):
          248 audio s per wall s one at a time, 514 at a batch of 8, 683 at
          32; the CPU backend 3.7 at one, 7.3 at 8.
        - served over D-Bus (`brain serve --dbus --device gpu1`, `brain_py`
          `encode_audio` with a transcript, the 0.6B Base, 256 corpus clips
          of 1.5-30 s, 5.2 s mean, each sent twice): 99 audio s per wall s
          one request at a time, 222 with 8 in flight, 283 with 16, 316 with
          32. 100 h of corpus is then about 27 min at the recipe's 8 in
          flight, 19 min at 32. The encode instance held 5.4 GB of the card.
        - codec alone over the 256 clips (`mimi` `device_encode_throughput`):
          231 at batch 1, 495 at 8, 606 in one call; the residual quantizer
          went from a third of its GPU time to a small share with
          `vq_argmin_tiled`, the fp32 conv GEMMs (`matmul_f32_reg`) lead.
        - parity: device vs CPU-backend codes on 24 corpus clips (105 s)
          99.97 % of codes and 100 % of cb0 (the rest residual-argmin
          near-ties the reassociated device sums flip); the upstream golden
          (24 s, 300 frames) 100 % on the CPU path; x-vector vs the upstream
          golden rel l2 7.5e-7 on CUDA, 5.5e-7 on the CPU backend.
  - [ ] Served batches at the recipe's concurrency are smaller than its
        in-flight count, and that is the served gap (222-237 audio s per wall
        s with 8 in flight against 514 in process at a batch of 8). Measured
        by timing every `run_batch` the instance received over 512 requests
        with 8 synchronous clients: 130 batches, bimodal (39 of 1, 28 of 2,
        25 of 6, 36 of 7, 2 of 8; mean 3.9), 83 ms each on average, the
        instance busy 97 % of the wall time. The dispatcher starts a batch as
        soon as one request is queued, and a batch's clients resubmit
        together, so the next batch takes the first one or two and leaves the
        rest for the one after. Candidate fixes: a short admission window the
        model declares to the generic scheduler (a few ms against an 83 ms
        pass), or a multi-clip `encode` request.
  - [x] `encode` returns `producers` (content digests of the codec, the
        speaker encoder and the tokenizer) for a shard's `meta.producers`.
  - [ ] Shard assembly from `encode` outputs: the caller writes
        `brain-tts-shard-v1` itself today.
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
- [x] Measure the objective on a card: the real-checkpoint overfit and its
      frames/s (`speech_sft::tests::the_real_checkpoint_overfits_one_utterance_and_reports_its_frame_rate`,
      ignored, refused on the CPU backend): on one GH200, the 0.6B Base, one 60-frame utterance a step, 40
      steps, the loss falls more than threefold at 711 frames/s (45 before the device-resident build). The
      host side is index plans only (the throughput item).
- [x] Throughput on one card: the full 1.7B fine-tune (3 epochs of 100 h, about 13.5M acoustic frames)
      in under 4 hours needs at least 1000 frames/s sustained (prompt rows not counted) through the served job,
      the real objective. Bench: `crates/qwen3tts/examples/speech_sft_bench.rs` (`synth` writes a shard with a
      corpus's length distribution, 1.5-30 s, mean 6.4 s, four text tokens a second; `run` trains through
      `speech_sft`, the function the served job runs). Phases are NVTX ranges (`speech_sft.*`) for `nsys`.
      Measured on one GH200 (`BRAIN_DEVICE=gpu0 BRAIN_BACKEND=cuda`), 1.7B Base, 4000-record synthetic shard
      (80.2 frames, 25.7 text tokens a record), batch 16 utterances, fp32 kernels:

      | build | frames/s (steps 2-6) | step |
      |---|---|---|
      | fp32 kernels, three device handles (2026-11-25) | 106-121 | 11-12 s for 1400 frames |
      | tensor cores (TF32 GEMMs, fp16 flash attention), the decoders on one handle | 242-283 | 4.5-6 s |
      | rows, heads, routing, tables and their optimizer on the device; batches trimmed | 2390 (wall, steps 7-30) | 0.38-0.68 s for 1000-1660 frames |

      The device-resident build (2026-11-25): the Talker is a `qwen3::Qwen` fed input rows with its own head
      (`codec_head`) and trims each batch to its longest record; the code predictor a second one on the
      Talker's handle, its 15 heads its own head grouped by position (`HeadGroups`), trimmed to the batch's
      frames; every input row is gathered on the device (`rows_gather_sum3` over the trained tables, the new
      rows and the frozen text and x-vector rows, in generation's summation order: bit-identical to the host
      builder), every table gradient is the gather's transpose (`rows_gather_weighted_bwd` over a host-built
      CSR), the code predictor's position-0 gradient goes to the Talker's hidden state on the device
      (`Qwen::enable_hidden_grad`), the projection runs through `Ops` at the run's tier, and the tables (one
      store, the new rows beside them) step on the device optimizer. The host builds index plans and reads
      the losses, the end-of-speech picks and one gradient norm a step. Device peak 62.5 GB at batch 16
      (the builds are sized by the longest record: block 505, 6016 code-predictor frames). The largest
      single cost removed was not in the objective at all: the update gate read every gradient back to the
      host each step (knowledge 581; 1.9 s of a 2.3-2.8 s step).

      Through the served job (2026-11-25): `brain serve --device gpu0 --backend cuda --training`, `POST /v1/jobs`
      `brain/train` `speech_sft`, the 1.7B Base, a 2000-record synthetic shard (81.5 frames a record) as a blob,
      batch 16, 300 steps: 2657 frames/s by the wall clock over steps 21-300 (median step 2711), 181 s for the
      whole job including 35 s of build and first step, loss 17.8 -> 9.84 (random codes). At that rate the
      13.5M frames of three epochs of 100 h take about 1.4 h. Batch 32: 2630 frames/s, device peak 90.4 GB, so
      batch 16 is the operating point until batches are bucketed by length.

      Tensor cores against the exact fp32 kernels on the real 0.6B Base and a real 67-record shard (one new
      language and two new voices, batch 8, lr 1e-4, 60 steps, seed 20261125, `speech_sft_bench --config
      '{"arithmetic": ...}'`): step 1 loss 3.5555 vs 3.5552; ten-step window means 4.115/4.087, 2.306/2.150,
      1.211/1.215, 0.628/0.599, 0.325/0.320, 0.167/0.173 (within 7.3 %, both falling 25-fold); 3318 against
      731 frames/s. The two trajectories separate after the first steps as two runs of different rounding do,
      not by a drift of one of them.

      Open: bucketing records by length (a batch pays its longest record's rows and the build is sized by
      the longest record of the shard: block 505 for 30 s), packed rows with block-diagonal attention, bf16
      weights and activations (the tier is TF32 products of fp32 weights), and a device charge
      (`trainjob::speech::device_bytes`) that does not grow with the batch.

      `nsys` of that build (4 steps): 79 % of the kernel time in brain's fp32 CUDA-core GEMMs
      (`matmul_f32_tiled` 39 %, `matmul_f32_dw_reg` 21 %, `matmul_f32_dx_reg` 19 %), the portable fp32 flash
      attention 10 %; 1.75 s a step in host-device copies (rows up, hidden states and row gradients down);
      the build pads every micro-batch to the longest record (block 505 rows, 16 x 505 Talker rows for about
      1840 real ones, and a 6000-frame code predictor for about 1400 frames).
      With tensor cores (`nsys`, 4 steps) the host is the step: the optimizer range 0.87 s a step (the codec
      tables on the host Adam), the host row build 0.61 s, the code predictor's rows and its heads' gather
      0.3-0.45 s each, and the largest kernels the projection's portable `matmul_dw`/`matmul_dx` (0.22 and
      0.17 s each a step over every frame's 16 rows).
- [x] Batches shorter than the build: both decoders take input rows and their own heads, so each batch is
      trimmed to its longest record (Talker) and to its frames (code predictor). Records are still drawn in a
      random order, so a batch pays its longest record's rows: bucketing by length is open (the throughput
      item).
- [x] One device per run: the code predictor is built on the Talker's handle (`Qwen::new_training_on`); the
      tables and the projection take a handle of their own kernel list on the same card (`Gpu::new_like`),
      ordered against the decoders' stream by the backend's cross-handle fences.
- [ ] The in-context clone in the non-streaming layout (refused today), and
      training the text projection (frozen in this release).
- [x] Export to the upstream layout (`qwen3tts::export`, over the generic
      `checkpoint::rewrite::rewrite_safetensors`): one `model.safetensors` in
      upstream names and the base's dtype, `config.json` with the trained
      languages and voices and `tts_model_type`, tokenizer and
      `speech_tokenizer/` carried over, atomic; gated on the real 0.6B Base
      (`tests/it/export.rs`: unchanged export byte-identical, trained tensors
      reload as their BF16 values, refusals leave nothing). A `custom_voice`
      export leaves out `speaker_encoder.*` (upstream builds none for it and
      its own fine-tuning drops it; keeping it would only draw an
      unused-weights warning). Not yet checked by loading in the upstream
      package itself (no torch on the build node).
- [x] Store: a Qwen3-TTS repo is recognised by its own signature for any
      repo, not by a pin list, so the 1.7B Base and trained releases pull as
      `qwen3tts` (`modelstore::recipe::select`: at one tier a signature that
      strictly contains another's wins). A store that already pulled the 1.7B
      under the old rule keeps its `family: "qwen3asr"` manifest until it is
      pulled again.

- [x] Served speech training: `brain/train` `speech_sft` (`crates/trainjob/src/speech.rs`) stages shards by blob or
      digest, checks them and the languages/voices against the base at the door, maps its params onto
      `SpeechSftConfig` (every setting the objective takes; left out, the objective's default), runs
      `qwen3tts::speech_sft` under the `resume_key` contract, streams each step's measurements, exports over the
      base, publishes the manifest and installs; every installed release is served under its own id
      (`TtsResident::all_from_store`) with its voices as preset voices and `language` on `/v1/audio/speech`. Gated on
      a tiny trainable upstream-layout base (in-process, HTTP, D-Bus) and, over D-Bus on the real 0.6B on the CPU
      backend (`dbus::the_real_base_trains_over_the_bus_and_its_release_speaks_in_its_new_voice`, ignored: slow):
      3 steps of one 29-frame recording, loss 3.06 -> 0.91, 397 tensors trained, the release served and spoken in
      its new voice and language; about 15 s and 1.2-2.0 frames/s per step on a shared 288-core host.
- [ ] A release installed by a running server is served from the server's next start (no job registers a model
      live).
- [ ] Collecting a published artifact over the API: a remote client without the artifact directory cannot fetch a
      checkpoint's files by digest. Needs a streamed, size-bounded read (a capability blob is held in memory, a
      1.8 GB `model.safetensors` included), so it is a transport change, not an action.

Gates: tiny random configs for the objective's gradients (finite differences
through Talker, code predictor and the routed tables); teacher-forced rows equal
inference rows on the real checkpoint; overfit of one real utterance; held-out
recogniser CER and speaker similarity against the base checkpoint with matched
seeds.
