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

The world-model window (`crates/wm-display`) is drawn with **SDL2**, which
`brain` loads when a run opens a window: building needs no SDL2, and a headless
run never touches it. To open windows, install it with your package manager
(`apt install libsdl2-2.0-0` on Debian/Ubuntu, `brew install sdl2` on macOS;
`brain` asks Homebrew where it put it), or put `SDL2.dll` beside `brain.exe` on
Windows. A run that needs a window without SDL2 present says what to install.

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

The `brain` CLI builds for Windows. Natively, with the MSVC toolchain and
CMake (libopus builds from its bundled source):

```bash
cargo build --release -p brain-cli
```

From Linux, with MinGW-w64 (`apt install mingw-w64`):

```bash
rustup target add x86_64-pc-windows-gnu
cargo build --release -p brain-cli --target x86_64-pc-windows-gnu \
  --config 'target.x86_64-pc-windows-gnu.linker="x86_64-w64-mingw32-gcc"'
```

The result imports only Windows system DLLs. What Windows lacks is refused
with a message rather than left out: the D-Bus surface, the Unix-socket
servers (`sample-models-qwen3tts npu-serve`, the event socket) and detached serving
(`brain serve -d`); plain `brain serve` runs in the foreground. Opening a
window needs `SDL2.dll` beside `brain.exe`.

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
