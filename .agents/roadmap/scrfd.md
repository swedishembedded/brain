# scrfd - roadmap

Face recognition (`crates/scrfd` + `crates/arcface`): insightface antelopev2 -
SCRFD-10GF detection + 5-point alignment + ArcFace IResNet-100 embedding, split
into one crate and one served model per architecture, imported and
parity-gated, with ArcFace training (backbone + additive-angular-margin head)
also gradient-checked. Forward parity is verified against the reference
implementation, and the serving contract (capability, residency, D-Bus,
example) is met except for batching.

## Not yet done

- [ ] `Instance::run_batch` is the serial default - ArcFace's input batches
      trivially but the graph is pre-allocated at N=1; SCRFD's detector graph
      is also pinned to N=1 and would need looping or re-export to batch
- [ ] Gradient check only covers a tiny config (4 blocks); IResNet-100's full
      49-block network has not been finite-difference checked at full size
- [ ] Performance is unoptimized and unmeasured: the conv/PReLU blocks
      allocate more scratch than needed, per-frame device→host syncs read
      back scalar values that never change, and the forward reads back every
      debug tap instead of skipping them in production
