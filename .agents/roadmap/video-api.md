<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# video-api

Scope: Serving video generation and editing over the OpenRouter `/videos` contract, with encoded output (`crates/apiserve/src/video.rs`, `capability::video`).

Goal: any OpenRouter-style client submits a video job to `brain serve`, polls it,
and downloads a playable MP4; the same client edits a clip it uploads (extend,
retake, upscale) and gets a new MP4 back. The per-model video verbs of the CLI
are retired: generation is `brain caps run` locally and `/videos` remotely.

Grounded against HEAD 8b2ee7683 on 2026-11-05.

## Starting point (2026-11-05)

- Video actions: `brain/ltxv` (`t2v`, `dfr`, `t2a`, `v2v`, `extend`, `upscale`),
  `brain/minimaxh3` (`t2va`, `fl2va`, `ref2va`), `brain/wan` (`t2v`). Each returns a
  `Media::Video` blob of interleaved HWC **f32** frames, meta
  `{frames,w,h,c,fps}`, and optionally an `audio` WAV blob.
- `POST /v1/jobs` (`capabilities.rs`): in-memory jobs, 32 per caller, 1024 in
  total, 1 h retention, 1 GiB of retained results for everyone, results as
  base64 JSON. One real clip (1.3 GB for 5 s of 720p) does not fit.
- `crates/videocodec`: AV1 (rav1e) + Opus into MP4/WebM/MKV and AV1 decode
  (rav1d), in-process; `VideoWriter` writes to a path. The CLI's `save_blob`
  (`crates/cli/src/caps_cli.rs`) is the only caller that encodes an outcome.
- No param vocabulary is shared: LTX `frames` 1+8k, size multiple of 32, `fps`
  param; MiniMax `num_frames` at 24 fps fixed, 5-15 s; Wan `frames` 1+4k,
  multiple of 16. Stills for conditioning are `host_resolved` paths, refused
  over HTTP.
- `/models` lists no video modality; `/v1/jobs` and `/v1/run` do not auto-fetch.
- The mock models (`catalog::resident_mock`) have no video action.

## Design

**A declared contract, not a guess.** `capability::video::VideoContract`, set on
an `ActionSpec` by the model crate that owns it:

| Field | Meaning |
| --- | --- |
| `frames_param` | the param holding the frame count (`frames`, `num_frames`) |
| `frame_grid` | `(base, step)`: legal counts are `base + k*step` (LTX 1+8k, Wan 1+4k) |
| `fps` | `Param(name)` or `Fixed(n)` |
| `size_params` | `(width, height)` and the multiple both must be |
| `max_pixels`, `seconds` | the bounds a request is held to |
| `audio` | `None`, `Always`, or `Param(name)` |
| `seed_param` | the seed param, if any |
| `clip_input` | the `video` blob an edit action takes, if any |
| `sellable` | false for a smoke-test config (LTX `dit_config=tiny`) |

`/videos` maps `duration`, `resolution`, `aspect_ratio`, `size`,
`generate_audio` and `seed` onto it, snapping the frame count to the grid; a
value the contract cannot honour is a 400 that names it. `/videos/models` is
derived from it: `supported_durations` (whole seconds that land on the grid),
`supported_resolutions`, `supported_aspect_ratios`, `supported_sizes`,
`generate_audio`, `seed`, and the extension `actions` (each edit action with
its passthrough params).

**Routes** (`crates/apiserve/src/video.rs`, on the OpenAI and OpenRouter
surfaces, at `/videos` and `/v1/videos`):

- `POST /videos` validates, then starts a job on the existing executor path
  (`bridge::stream_progress`, hooks and admission unchanged). `Idempotency-Key`
  returns the existing job for the same caller scope and key. Answer 202
  `{id, polling_url, status: "pending"}`.
- `GET /videos/{id}`: `pending`, `in_progress` (with `progress {step,total}`),
  `completed` (with `unsigned_urls`, `usage {seconds, frames, width, height,
  audio}`), `failed` (`error`), `cancelled`.
- `GET /videos/{id}/content?index=0`: the MP4, `Content-Length`, `Range`.
- `GET /videos/models`.
- `DELETE /videos/{id}` (extension): cancel a running job, or drop a finished
  one and its file.

**Spool.** A finished outcome is encoded once into `<spool>/<id>.mp4`
(AV1 + the outcome's audio as Opus) and the frames are dropped; the job keeps
`{path, bytes, sha256, usage}`. The spool directory is owned by the server
(created private, emptied at start), bounded in bytes (`--video-spool-bytes`,
oldest finished first), and a job's file is removed with the job.

**Inputs.** The `input_video` extension
(`{"type":"video_url","video_url":{"url":"data:video/mp4;base64,..."}}`) is
decoded at the boundary into the edit action's `video` blob. `frame_images`
(`first_frame`, `last_frame`) and `input_references` (image, video, audio)
fill the host-resolved path params the action's contract names
(`VideoContract::frame_images`, `::references`): `apiserve::video_inputs`
decodes each `data:` URL (images within `MAX_IMAGE_PIXELS`, re-encoded as
PNG; WAV parsed and written again; MP4/WebM kept once it decodes), writes it
to a private directory of the job's own, sets the param, and removes the
directory when the job ends. A request with frames or references goes to the
model's first generation action that takes them (MiniMax-H3's `fl2va` or
`ref2va`, LTX-2.5's `t2v`). A remote URL is refused: the server never
fetches. A caller can still never name a path itself.

**Choosing an edit.** `provider.options.brain.action` names an action declaring
`clip_input`; its other params go in `provider.options.brain.params`, checked
by the action's own `validate_served`.

## Milestones

Done (2026-11-06), all on the mock model, no GPU:

1. `capability::video::VideoContract`, declared on `ltxv` (`t2v`, `v2v`,
   `extend`, `upscale`), `wan` (`t2v`), `minimaxh3` (`t2va`) and the mock;
   mapping unit tests (grid snapping in both directions, bounds, refusals),
   and a catalog test that every contract names params and inputs its action
   really has.
2. `brain/mock-video` (`BRAIN_MOCK`): `t2v` with an optional sound track, and
   `extend`; flat colours a test can recognise after AV1 coding.
3. `crates/apiserve/src/video.rs`: routes, spool, `Idempotency-Key`, `Range`,
   cancel/delete; `tests/it/video.rs` (7 cases: listing, generate-poll-
   download with decoded frames, ranges, idempotency, an edit fed the
   previous result, cancel, every refusal before the model runs).
4. `/models` lists a sellable video model with `video` output.
5. `tests/e2e/video_api.bats` (`make test/e2e/video-api`), in `test/e2e`.

The outcome encoder (`imaging::video::encode_clip`) is now the one way an
outcome becomes a file, for the CLI and the server.

Measured, not assumed: AV1 coding is about 185 ms a frame at 1024x512 on a
48-core host (`videocodec::EncodeSettings` default), so a 5 s 720p clip spends
tens of seconds on the CPU after the GPU finishes. The encode runs on a
blocking thread, never the async runtime; it is a candidate for an encoder
pool if real-weight runs show it dominating.

Open:

Also done: a served LTX-2.5 now comes from the model store
(`LtxvResident::all_from_store`, the `served_assemblies` lane flux2 takes),
not only from `BRAIN_LTXV_*`; one without its real DiT and text encoder runs
the random-weight smoke-test DiT and is marked not sellable, so `/videos`
never offers it. The generic CLI decodes an edit's input clip whole, at its
own rate (it used to take 1 fps, 32 frames at most, which is right only for a
multimodal prompt).

6. Retire `ltxv_cli.rs` and `wan_cli.rs` (agreed: every per-model video
   verb). **Audited 2026-11-06: not yet.** The generic path
   (`brain <model> <action>`, `caps_cli::run_do`) covers about 60% of the
   flags. Before the files can go, each of these needs a capability param or
   a generic verb, or nothing a person does today survives:
   - the generic LTX provider resolves weights from the environment only:
     catalog `spec: Some(LtxvSpec)` and an assembly-built provider, plus the
     `--vae/--dit/--text-encoder/--spatial-upsampler/--audio-vae` role
     overrides through `resolver_cli::extract_role_overrides` (Wan too);
   - v2v: a `mask` input (a mask sequence, or a host-resolved folder) and
     `strength`, with `start`/`end` optional when a mask is given;
   - several adapters, each with its own scale (one `adapter` today);
   - the IC-LoRA `reference` video input and `reference_strength`, and
     `stage_2_ic_lora`;
   - t2v `context_frames` and `scenes`; extend `seconds`; upscale
     `context_frames`;
   - the soundtrack input: `caps_cli` downmixes a WAV to 16 kHz mono, but
     `clip_from` wants the 48 kHz stereo container;
   - Wan: declare `dit_dtype` (read but undeclared, so `validate` refuses it),
     `te_device` on t2v, the 1200 s wait for one large forward, and
     `wan finetune --report` (`wan_report.rs`, its only caller) as a
     `lora_train` param or a post-step with per-step losses in progress;
   - the run record (`<out>.json` with the seed and params) the verbs write;
   - then `resolve.rs` handler rows, `seed_flag.rs`'s `SEEDED_VERBS`,
     `quickstart.sh`/`quickstart.bats`, the `wan` make target, and the docs
     and samples that call `brain ltxv`/`brain wan` (`character_replace.sh` is
     tracking, still edits and masked v2v with keyframe guides,
     `chain_images_to_video.sh` mid-frame keyframes).

## Later

- Image blob inputs replacing the `host_resolved` still params (LTX, MiniMax)
  and a Wan i2v action.
- Live previews through `Progress.chunk`, measured for overhead on a GPU.
- Real-weight end-to-end runs on hardware that holds the checkpoints.
