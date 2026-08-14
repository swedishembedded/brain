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
| CLI | [x] |
| HTTP API | [ ] |
| D-Bus | [ ] |
| Batched serving | [ ] |

## Getting the weights

World models have no HF-style model id - you select an architecture
directly with `--model`:

- **`fake`** (the default) - a deterministic, GPU-free test model that
  needs no weights.
- **`diamond`** - the real DIAMOND world model. Needs:
  - `--weights <F.safetensors>`, produced by:
    ```bash
    brain diamond import --arch diamond --src <agent.pt> --out <F.safetensors> [--actions-count N]
    ```
    from an official Atari-100k `agent.pt` checkpoint.
  - For the NPU path, additionally: `--onnx <F.onnx>`, produced by:
    ```bash
    brain diamond export --arch diamond --weights <F.safetensors> --onnx <F.onnx>
    ```

## Running it

```bash
# windowed play (opens a window; WASD move, Enter reset, . pause, e step, [ ] quality, Esc quit)
brain diamond play --model fake|diamond [--weights F.safetensors] [--device cpu|gpu|npu] \
    [--onnx F.onnx] [--fps N] [--scale N] [--seed N] [--adaptive] \
    [--record DIR] [--seed-context DIR] [--denoise-steps N]

# headless deterministic rollout
brain diamond play --headless --frames N [--actions FILE | --action-seq 1,2,0] \
    [--dump-ppm DIR] [--hashes]

# replay a recorded episode, optionally verifying it against a live model
brain diamond replay --episode DIR [--verify --model fake|diamond [--weights F] \
    [--device cpu|gpu|npu] [--onnx F.onnx] [--seed N] [--denoise-steps N] \
    [--context N] [--tolerance T]]

# headless throughput benchmark
brain diamond bench --model fake|diamond [--weights F] [--onnx F.onnx] [--frames N] [--seed N]

# fine-tune on a recorded episode dataset
brain diamond finetune --weights <base.safetensors> --data <episode-dir> --out <tuned.safetensors> \
    [--steps N] [--lr F] [--wd F] [--clip F] [--seed N] [--device cpu|gpu]
```

## Options

- `play`: `--device cpu|gpu|npu`, `--onnx <file>` (NPU only), `--fps <n>`,
  `--scale <n>`, `--seed <n>`, `--adaptive`, `--record <dir>` (save an
  episode as you play), `--seed-context <dir>`, `--denoise-steps <n>`.
- `play --headless`: `--frames <n>`, `--actions <file>` or
  `--action-seq <list>`, `--dump-ppm <dir>`, `--hashes`.
- `replay`: `--episode <dir>`, `--verify` plus the same model/device flags,
  `--context <n>`, `--tolerance <t>`.
- `bench`: `--frames <n>`, `--seed <n>`; add `--profile` with `--weights`
  and `--device` for a per-stage timing breakdown.
- `finetune`: `--steps` (default 300), `--lr` (default 1e-4), `--wd`
  (default 1e-2), `--clip` (default 1.0), `--seed` (default 7), `--device`.
  Trains an episode dataset produced by `brain data gen` or `brain diamond play
  --record`; refuses to save a run that diverged (NaN/inf loss).

## Hardware and limits

- CLI only (`brain diamond ...`) - no D-Bus and no HTTP route.
- Fine-tuning is a full fine-tune of the model; there is no LoRA or QLoRA
  path.
- One episode at a time - no batch-serving of multiple simulations.
- DIAMOND is the only architecture that is actually playable today.
