# worldmirror2 - roadmap

Port of HY-WorldMirror-2.0, a DINOv2-based multi-frame 3D scene
reconstruction model that predicts Gaussian splats, depth/normal maps, and
camera pose from one or more images, with an NPU export path. Forward parity
against the reference is verified.

## Not yet done

- [ ] NPU throughput/latency benchmarking (only numerical parity has been
      measured so far)
- [ ] wgpu memory-budget autotuning and frame pipelining
- [ ] Further CPU matmul blocking (column tiling) if forward speed needs
      another step

Training backward for cross-attention currently accumulates gradients only
within a chunk rather than across the full sequence, so training requires
the chunk size to cover the full attention span. The NPU backend executes
only in fp16, and precision drift grows with network depth - acceptable for
preview-quality geometry, less so for higher-fidelity uses.
