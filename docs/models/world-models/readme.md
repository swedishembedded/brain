# World models (`crates/wm-diamond`, `crates/wm-genie`)

Action-conditioned video world models: reset with context frames, step one
discrete action at a time, get back a frame. DIAMOND (Atari-100k EDM
diffusion UNet) is playable end-to-end on cpu/gpu/npu; GenieRedux-G (CoinRun
ST-transformer) is parity-verified through the tokenizer + MaskGIT dynamics
but not wired into the CLI yet.

## Model id and weights

- **Id:** none — no `capability::Manifest`, not registered with the residency
  scheduler (see Surfaces). Selected purely as `--model fake|diamond` on the
  CLI.
- **DIAMOND weights:** `--weights <F.safetensors>`, produced by
  `brain wm import --arch diamond --src <agent.pt> --out <F.safetensors>
  [--actions-count N]` from the official Atari-100k `agent.pt` checkpoint.
- **DIAMOND NPU path:** additionally needs `--onnx <F.onnx>` from
  `brain wm export --arch diamond --weights <F.safetensors> --onnx <F.onnx>`
  (fp32 ONNX of the UNet, run through OpenVINO).
- **`fake` model:** no weights — a deterministic, GPU-free test model
  (`--model fake`, the default).
- **GenieRedux-G:** no weights flag exists yet — `crates/wm-genie` has no
  `brain wm` verb of its own (tokenizer/MaskGIT dynamics are parity-exact but
  not exposed on the CLI). See `docs/models/world-models/status.md`.

## Surfaces

CLI only (`brain wm …`) — no `caps.rs`, not registered with the residency
scheduler, no D-Bus, no HTTP: `wm_core::WorldModel` is a stateful
`reset`/`step` trait with no request/response capability shape yet.

## Inference

### CLI

```bash
# windowed (opens an SDL window; WASD move, Enter reset, . pause, e step, [ ] quality, Esc quit)
brain wm play --model fake|diamond [--weights F.safetensors] [--device cpu|gpu|npu] \
    [--onnx F.onnx (npu only)] [--fps N] [--scale N] [--seed N] [--adaptive] \
    [--record DIR] [--seed-context DIR] [--denoise-steps N]

# headless deterministic rollout (the form CI runs)
brain wm play --headless --frames N [--actions FILE | --action-seq 1,2,0] \
    [--dump-ppm DIR] [--hashes]   # same --model/--weights/--device/--onnx flags

brain wm replay --episode DIR [--verify --model fake|diamond [--weights F] \
    [--device cpu|gpu|npu] [--onnx F.onnx] [--seed N] [--denoise-steps N] \
    [--context N] [--tolerance T]]           # re-runs a recorded episode, compares frames by MAD

brain wm bench --model fake|diamond [--weights F] [--onnx F.onnx (npu)] \
    [--frames N] [--seed N]                  # headless throughput; add --profile --weights F [--device D] for a per-kernel diamond breakdown

brain wm import --arch diamond --src <agent.pt> --out <F.safetensors> [--actions-count N]
brain wm export --arch diamond --weights <F.safetensors> --onnx <F.onnx>
```

## Training / Fine-tune

```bash
brain wm finetune --weights <base.safetensors> --data <episode-dir> --out <tuned.safetensors> \
    [--steps N] [--lr F] [--wd F] [--clip F] [--seed N] [--device cpu|gpu]
    # defaults: steps=300 lr=1e-4 wd=1e-2 clip=1.0 seed=7
```
Full fine-tune of the DIAMOND UNet trainer (all conv weights + biases
trainable, conditioning path frozen) on a `brain data gen`- or
`wm play --record`-produced episode dataset; refuses to save a diverged
(NaN/inf loss) run.

`crates/wm-genie` has no CLI verb yet — no command to give. See
`docs/models/world-models/status.md`.

## Not supported

D-Bus, HTTP, LoRA, QLoRA, batch > 1 (single-episode CLI loop only)

## See also

- Crates: `crates/wm-diamond` (DIAMOND), `crates/wm-genie` (GenieRedux-G),
  `crates/wm-core` (shared `WorldModel` trait), `crates/wm-display`
  (SDL window / pacing / sinks)
- Workstream ledger: [status.md](status.md)
- Failure playbooks: [playbooks.md](playbooks.md)
- Parity fixtures: [fixtures.md](fixtures.md)
