<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 580. A served speech decode is bound by what runs between frames, not by its kernels

**Lesson:** a batched Qwen3-TTS server on a GH200 spent most of its wall time with the card idle. The kernels of a
32-stream frame took a few milliseconds; the time went to host work done on the decode thread (one host round trip per
code-predictor step, a scalar host text projection, per-request codec decodes) and to the syncs that waited for it.
Measure per frame phase with NVTX ranges before tuning a kernel.

## What happened
Served over HTTP (in-context clones of 8 s of speech, nsys with `-t cuda,nvtx`), the frame loop of 16 streams ran 16.5
ms a frame against 4.8 ms of kernels, with 7.7 host syncs a frame. A CPU-sampled profile of the decode thread put 44 %
of its samples in `TextProjection::project`, a scalar host MLP run for every prompt token while every other stream
waited. At 64 streams the codec decode of each finished clip, run on the decode thread, was 61 % of the wall time.

## What worked, measured (paired runs on one card)
- The code predictor's fifteen steps as one chained window (`Engine::forward_batched_embed_chain`): one readback a
  frame instead of fifteen.
- The text projection as two blocked host GEMMs.
- The checkpoint's stored bf16 on both decoders (KL to fp32 1e-6 to 2e-5 on codebook 0).
- Embedding-input steps replaying recorded tapes, every stream's utterance reserved at admission.
- The codec's lowered conv products on the vendor fp32 GEMM (`ConvScratch::with_gemm`): warm 100-frame decode 75-142
  ms to 58-62 ms at unchanged precision.

Solo clone RTF 1.2 to 0.15; 32 concurrent streams 52 s of speech a second.

## What did not, and why
- **A codec worker thread on its own `Gpu` handle** was slower (32 streams 52 to 30 s of speech a second): once a
  second handle exists, every submission fences against the other handles' work, so each frame's sync waited behind a
  whole clip decode.
- **TF32 for the codec** failed its fp32 gates (relative rms 5.4e-3 against a 1e-3 gate, 0.0207 against the host
  decoder's 0.02) and was not faster: the codec checkpoint is stored fp32.
- **Recording the codec decode per length bucket** (exact, the decoder is causal) gave no measured gain: a recording
  keeps about 26 MiB a frame alive, and each bucket's first clip pays the recording in front of its stream.
- **Flushing the codec's recorded steps every 48 steps** made no measurable difference.
- **Decoding the clips that finish together as one batch** (`decode_batch`, exact up to fp32 reassociation: 8.5e-5
  relative rms on real weights) made the codec slower, 3.16 s to 3.71 s of codec time for 32 clips: 32 clips went out
  in 19 batches, a batch's time grew with its clip count (500-614 ms for the largest), and each clip is padded to the
  batch's longest. Per clip the decode is bound by its own kernels (the convs' im2col and products), not by launches.

## What to do next
Make the codec's convs cheaper per clip (an implicit-GEMM conv instead of im2col plus a product); draw codebook 0 on the
device with its suppression and repetition penalty; keep the Talker hidden on the
device so a frame syncs once.

## Evidence
`.agents/roadmap/qwen3tts.md` ("Batched GPU inference and training"); the commits that landed each step name their
measurements.
