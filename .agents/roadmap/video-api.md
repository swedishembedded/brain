<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# video-api

Scope: Serving video generation and editing over the OpenRouter `/videos` contract, with encoded output (`crates/apiserve/src/video.rs`, `capability::video`).

Goal: any OpenRouter-style client submits a video job to `brain serve`, polls it,
and downloads a playable MP4; the same client edits a clip it uploads (extend,
retake, upscale) and gets a new MP4 back. The per-model video verbs of the CLI
are retired: generation is `brain caps run` locally and `/videos` remotely.

Grounded against HEAD 8b2ee7683 on 2026-11-05.

## What exists

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

**Inputs.** `frame_images`/`input_references` image `data:` URLs and the
`input_video` extension (`{"type":"video_url","video_url":{"url":"data:video/mp4;base64,..."}}`)
are decoded at the boundary (PNG/JPEG via `imaging`, AV1 MP4/WebM via
`videocodec`) into the action's blobs. A remote URL is refused: the server
never fetches.

**Choosing an edit.** `provider.options.brain.action` names an action declaring
`clip_input`; its other params go in `provider.options.brain.params`, checked
by the action's own `validate_served`.

## Milestones

1. `VideoContract` in `capability`, declared on `ltxv` (`t2v`, `v2v`, `extend`,
   `upscale`), `wan` (`t2v`), `minimaxh3` (`t2va`) and on the new mock video
   model; unit tests of the mapping (grid snapping, bounds, refusals).
2. `brain/mock-video` (`BRAIN_MOCK`): `t2v` drawing `synth::video_frame_hwc`
   with a sine track, and `extend` appending frames to an input clip.
3. `video.rs` routes, spool and idempotency; `tests/it/api.rs` cases for
   submit/poll/download/cancel, refusals, another caller's job, spool eviction;
   the downloaded MP4 decodes back to the mock's frames.
4. `/models` and model cards list video models with `video` output modality.
5. `tests/e2e/video_api.bats` (`BRAIN_MOCK=1`): curl drives the whole contract.
6. CLI: `ltxv_cli.rs` and `wan_cli.rs` removed; anything still needed (GGUF
   import, flops) through the generic verbs; docs and samples updated.

## Later

- Image blob inputs replacing the `host_resolved` still params (LTX, MiniMax)
  and a Wan i2v action.
- Live previews through `Progress.chunk`, measured for overhead on a GPU.
- Real-weight end-to-end runs on hardware that holds the checkpoints.
