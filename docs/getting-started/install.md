# Install

## Prerequisites

The hard prerequisites are a **Rust toolchain** (install via
[rustup](https://rustup.rs)) and **CMake with a C compiler** (Debian/Ubuntu:
`sudo apt install cmake build-essential`). brain is a self-contained Cargo
workspace: no Python, no PyTorch, and one C library, libopus, which the
build compiles from its bundled source so the speech-to-text API can decode
Opus recordings (the format browsers record) and generated video carries Opus
sound. Video is written and read in process: AV1 (royalty-free) in `.mp4`,
`.mov`, `.mkv` and `.webm`, coded by the pure-Rust rav1e and rav1d.

A few things are optional, needed only for specific paths:

- **A GPU with a Vulkan/Metal/DX12 driver** — for the GPU backend. Not
  required: brain also runs entirely on CPU (see [Hardware](../guides/hardware/choosing-hardware.md)).
- **Node.js 18+ and a WebGPU-capable browser** (Chrome/Edge 113+) — only for
  the in-browser WebGPU demo (`make web/dev`).
- **ffmpeg** - only to read an input video in a codec other than AV1 (an
  H.264 phone clip, say) from the command line; brain hands such a file to the
  `ffmpeg` CLI when it is on `PATH`. A clip a client sends to `brain serve` is
  never handed to it. Nothing brain writes needs it.
- **Python + OpenVINO** — only for the Intel NPU export/run path
  (`brain npu …`) and a couple of benchmark reference scripts. `make
  environment/requirements` installs this tooling; the Rust engine itself never needs it,
  and `make build/debug`/`make test` stay green without it installed.

## Build

```bash
make build/release          # build the optimized ./target/release/brain
```

`make build/debug` also exists for an unoptimized debug build, but `make build/release`
is what you want for anything performance-sensitive — training and inference
both.

### Windows

The video stack (`brain-videocodec`, `brain-imaging`) builds and links for
`x86_64-pc-windows-msvc`, including its tests: the AV1 coders are pure Rust
and libopus builds with CMake and the MSVC (or clang-cl) compiler. From Linux,
[cargo-xwin](https://github.com/rust-cross/cargo-xwin) cross-builds it
against the Microsoft CRT and SDK it downloads:

```bash
rustup target add x86_64-pc-windows-msvc
cargo install cargo-xwin
cargo xwin build -p brain-imaging --target x86_64-pc-windows-msvc
```

The whole `brain` CLI does not build for Windows yet: `brain-dbus` (D-Bus) and
`brain-server` (Unix domain sockets) do not compile there, and the CLI's TTS
server and daemon use Unix sockets and descriptors as well. The window and
the viewport recorder link SDL2, which a Windows build has to be given.

## Test

```bash
make test             # full test suite (unit + integration)
```

`make test` needs a GPU by default. To run the entire suite on CPU only:

```bash
BRAIN_DEVICE=cpu make test
```

This runs the same tests against the CPU backend (the same WGSL kernels,
JIT-compiled), so a machine with no GPU at all can still validate the full
build.

## Correctness gate

```bash
make test/gradcheck         # finite-difference backprop correctness gate
```

This is the check that stands in for a PyTorch reference: every model's
analytic WGSL gradients are verified against finite differences of its own
forward pass. Run it after any change that touches a model's forward or
backward pass.

## Next

- [Hardware](../guides/hardware/choosing-hardware.md) - choosing and selecting a device (GPU/CPU/NPU).
- [Quickstart](quickstart.md) — train something, then run a real LLM.
