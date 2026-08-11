# yolo — roadmap

From-scratch anchor-free YOLO detector (CSP backbone, PAN-FPN neck, decoupled
DFL head), byte-compatible with the canonical yolov8n checkpoint format, with
the full serving contract (capability, residency, batching, D-Bus, example)
implemented.

## Not yet done

- [ ] Per-parameter layer freezing for fine-tuning — `--freeze-backbone` is
      currently accepted but is a no-op; fine-tuning always trains the whole
      network
- [ ] A representative, full-resolution calibration set for NPU INT8
      quantization (the default set is synthetic and low-resolution)
- [ ] Reference-framework parity and live NPU hardware tests running by
      default instead of opt-in only

Reference-framework parity and live NPU testing are opt-in rather than
default because the environment that gates this project doesn't have a
reference framework/GPU or NPU hardware available by default.
