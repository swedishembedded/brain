# DIAMOND (playable world model)

A world model lets you "play" inside a learned simulation of a game or
environment: you reset it with a few context frames, then step it forward
one discrete action at a time, and it predicts the next frame - rendered
live in a window so you can watch and steer it like a game. DIAMOND, an
Atari-100k-style EDM diffusion world model, is the one you can play end to
end today; see the [world models overview](world-models.md) for how it
compares to [GenieRedux-G](genieredux.md). Reach for it to interactively
probe what a world model has actually learned, to record and replay
episodes for offline evaluation, or to fine-tune it on your own recorded
episodes.

## Support

| Capability | Supported |
|---|---|
| Inference | [x] |
| Training from scratch | [ ] |
| LoRA fine-tune | [ ] |
| CLI (`brain diamond <action>`) | [x] |
| HTTP API | [x] |
| D-Bus | [x] |
| Batched serving | [ ] |

## Getting the weights

Model id: `brain/diamond`. A world model has no HF-style model id; the
reference checkpoint is an official Atari-100k `agent.pt`, converted once to
brain's format:

```bash
brain diamond import --source <agent.pt> --dest <F.safetensors> [--actions_count N]
```

The converted file (and any checkpoint `finetune` writes) carries a `diamond`
model card, so once it is in the model store every command finds it; or name
one outright with `--weights <F.safetensors>`.

`--model fake` (on `rollout`, `replay` and `bench`) is a deterministic,
GPU-free test model that needs no weights.

For the Intel NPU, export the UNet once and pass it as `--onnx`:

```bash
brain diamond export --dest <F.onnx>
```

## Running it

Every action's flags, defaults and bounds: `brain diamond <action> --help`.

```bash
# headless deterministic rollout: a clip, a rollout hash that is equal for equal runs
brain diamond rollout --action_seq 1,2,0 --seed $RANDOM --out clip=rollout.mp4 \
    [--model diamond|fake] [--frames N] [--actions <file>] [--fps N] [--hashes] \
    [--record DIR] [--dump_ppm DIR] [--seed_context DIR] [--denoise_steps N] [--onnx F.onnx]

# replay a recorded episode dataset, optionally verifying it against a live model
brain diamond replay --episode DIR [--verify --seed <the recording's seed> --denoise_steps N --context N --tolerance T]

# throughput, or the per-kernel time of one UNet forward
brain diamond bench [--model diamond|fake] [--frames N] [--onnx F.onnx] [--profile]

# fine-tune on a recorded episode dataset
brain diamond finetune --data <episode-dir> --save <tuned.safetensors> --seed $RANDOM \
    [--steps N] [--lr F] [--wd F] [--clip F] [--activations keep|recompute]
```

The interactive window is the `play` workflow of `samples/models/diamond`
(Space fires, A/D or the arrows move, Enter resets, `.` pauses, `e` steps,
`[` `]` change the quality, Esc quits):

```bash
make samples/models/diamond/run ARGS="play --model diamond --scale 10 [--record DIR] [--renderer auto|software|accelerated]"
```

## Behavior

- A rollout is deterministic for one model, `seed`, `denoise_steps` and action
  script; `replay --verify` regenerates a recording and compares it frame by
  frame, so the same `seed` and `denoise_steps` as at record time are what
  makes it match. `--context N` skips the first N steps and starts verifying
  from recorded frames (DIAMOND only; the fake model verifies from 0).
- A rollout returns at most 2048 frames as one clip; record longer sessions
  with the window.
- `finetune` trains an episode dataset recorded by `rollout --record`, the
  window's `--record` or `brain data gen`; it stops between steps when
  cancelled and refuses to save a run whose loss diverged (NaN/inf) - lower
  `--lr` (batch-1 fine-tuning is sensitive). `--seed` omitted draws a random
  seed, which the run reports.
- `--onnx` runs the UNet as a compiled OpenVINO graph on the Intel NPU with
  the sampler on the host; there is no `--device npu`.

## Hardware and limits

- Served over HTTP and D-Bus like every model; the actions that read or write
  a path on the machine (`import`, `export`, `finetune`, the `record`,
  `dump_ppm`, `actions`, `seed_context` and `onnx` params) run on the host
  only.
- Fine-tuning is a full fine-tune of the model; there is no LoRA or QLoRA
  path.
- One episode at a time - no batch-serving of multiple simulations.
- DIAMOND is the only architecture that is actually playable today.
