# World-model workstream — status

Goal: complete world-model support in brain — train / inspect / infer
action-conditioned video world models on CPU/GPU/NPU, load open pretrained
models (DIAMOND-Atari first, then GenieRedux-CoinRun, iVideoGPT, open-oasis),
play them in realtime in an SDL window with WASD, record play sessions as
training data, and fine-tune (full FT + LoRA). Reference material:
`/data/workspace/resources/world-models/` (read-only).

Mode: direct implementation on this branch (based on origin/main), gated by
the repo's standard discipline — `make build`, headless
`BRAIN_DEVICE=cpu MOE_SKIP_GPU_TESTS=1 make test` (with `DISPLAY=` unset:
a stale X11 DISPLAY breaks Vulkan enumeration), `make gradcheck`, tests
written from the specs in `docs/world-models/specs/`.

## Done
- Kernel registry mechanically derivable: `make kernels-regen`
  (`scripts/kernels-regen.sh`); registry canonicalized, 161 kernels.
- GroupNorm kernel family (`gn_stats/apply/dsum/dx/dgamma/dbeta`) fwd+bwd
  + host dispatch `wm_core::gn` — spec P1.gn.md, 9 tests incl. FD backward.
- FiLM/adaLN family (`film_chan*`, `film_row*` w/ rows_per_cond
  diffusion-forcing grouping, `gate_row*` adaLN-Zero) + `wm_core::film` —
  spec P1.film.md, 12 tests incl. FD checks.
- Glue kernels (`mul`, `scale_row`, `edm_mix`, `mse_value_w`/`mse_grad_w`,
  `pad2d`/`crop2d` adjoint pair, `nchw_nlc`/`nlc_nchw` permutes) — spec
  P1.glue.md, 15 tests incl. adjointness + FD.
- `wm-core` crate: object-safe `WorldModel` trait (CHW f32 frames, discrete
  actions, `set_nfe` quality knob) + deterministic GPU-free `FakeWorldModel`
  — spec P1.worldmodel-trait.md, 23 tests.
- DIAMOND parity fixtures: regenerable via `make wm-fixtures` (not in git);
  `crates/wm-diamond` stub crate reserves the layout.

- `wm-display` crate: SDL2 software-blit window (hand-rolled ~15-fn FFI,
  feature `sdl`, links system libSDL2; iGPU stays free for compute),
  longest-chord-wins keymap, fixed-timestep pacing with mock-clock tests and
  adaptive-quality hysteresis, headless/hash/PPM/tee sinks, `PlayIo` seam.
- `brain wm play` (windowed via `make build/wm`; `--headless` deterministic
  scripted rollouts with fnv1a golden hashes) and `brain wm bench`, running
  the FakeWorldModel end-to-end. 12 display tests, all headless.

## Next (P1 remainder → P2)
- vq_argmin/vq_argmax_dot kernels; dwconv3d PEG family; maskgit host decode;
  EDM/sampler/schedule host math (`wm_core::{edm,sampler,schedule}`).
- torch `.pt` reader (evaluate candle-core pickle / repugnant-pickle first),
  then DIAMOND UNet model + import + 3-step Euler playable path (P2) wired
  into `brain wm play --model diamond`.

Backups of the pre-restructure orchestration experiment:
`backup/wm-orchestration-v1`, `backup/wm-p1-*` branches.
