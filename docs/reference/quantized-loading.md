# Quantized tensors: descriptors, codecs and bounded loading

`checkpoint::packed` is the one place that knows what a quantized tensor's bytes mean and how to move
them without expanding them. Model crates supply tensor names and shapes; they never decode, convert
or read whole tensors.

## Vocabulary (`packed::encoding`)

- `QuantEncoding { element, scales }`: `value = element_value(code) * product(scales)`.
  Elements are integers, small floats (E4M3, E5M2, E2M1, E8M0, f16, bf16, f32), a fixed codebook (NF4)
  or a GGUF block format (Q8_0, Q4_K, MXFP4, ...). Scales are a hierarchy of sidecar tensors with a
  granularity (tensor, block of n, row, 2-D tile). NVFP4 is `QuantEncoding::nvfp4()` (E2M1, E4M3 per 16,
  f32 per tensor); MXFP4 is `QuantEncoding::mxfp4()`.
- `TensorDescriptor { name, shape, encoding }` gives the logical shape and the physical layout:
  `payload_bytes`, `scale_bytes`, and for any tile (a flat element range aligned to `tile_quantum`)
  `payload_range` and `scale_range`.

## Codecs (`packed::codec`)

`CodecRegistry::reference()` decodes (`decode_tile`) every encoding above on the CPU and encodes the
ones a load can target (f32, f16, bf16, block-scaled int8 and NF4, GGUF types the quantizers cover).
A new format is a `Codec` registered with `CodecRegistry::register`; no loader or model changes.

## Reading (`packed::reader`)

`PackedTensorReader::read_blocks(desc, elems, dst)` copies the payload and scale bytes of one tile and
nothing else; there is no whole-tensor method. `PackedReader` serves a memory-mapped safetensors file,
a sharded HF directory (fp8 `_scale_inv` grids and NVFP4 `_scale`/`_scale_2` siblings are discovered
and hidden behind their payload tensor) and GGUF (single or split). Torch pickles and computed
checkpoints are refused at `descriptor()` time. A `DerivedCheckpoint` that only renames (`weightio::Renamed`) declares
`passthrough`, so a renamed view of a checkpoint is read packed from its source, never as f32. Layouts that dtype alone cannot tell (brain's packed
int8/NF4 `U32` words) are declared with `PackedReader::with_descriptor`.

## Loading (`packed::load`)

```rust
let budget = MemoryBudget::new(256 << 20, 0);               // shared by every concurrent load
let loader = Loader { reader: &reader, codecs: &CodecRegistry::reference(),
                      workspace: BoundedWorkspace::new(budget.clone()),
                      config: LoadConfig::default() };
let plan = loader.plan("model.layers.0.mlp.up_proj.weight", &LoadPolicy::Preserve)?;  // inspect first
let report = loader.execute(&plan, &mut my_sink)?;           // my_sink: impl TensorSink
```

Exactly three operations exist, chosen explicitly per tensor: `Preserve` (copy packed bytes),
`Repack` (lossless layout change; only exact pairs exist, e.g. Q8_0 to int8 codes plus f32 scales) and
`Convert` (decode a tile, encode a tile; only on request). A quantized tensor is never expanded
implicitly: the only f32 data is one tile's scratch, leased from the `MemoryBudget` before it is
allocated. `LoadStep` and `LoadReport` record source and destination encoding, tile size, staging
bytes and `intermediate_full_tensors` (always 0). A tensor whose smallest legal tile exceeds the
budget fails at plan time.

The budget is process-local and shared: concurrent loads wait for each other's leases, a failed load
releases its lease and calls `TensorSink::abort`, and `Pool::Pinned` is reserved for device
transfers. File-backed pages of the mapped checkpoint are released with `madvise` after each tile and
reported as `file_resident_growth_bytes`.

## Existing by-name consumers (`packed::source`)

`PackedSource` adapts a reader to `TensorSource`, the f32 by-name interface current model builders use,
through a declarative name map (`Binding { source, add }`). Chunks are decoded tile by tile, so host
scratch is `O(chunk)`. `qwen35::fp8_source::open_dir` is that map for the Qwen3.5/3.8 HuggingFace layout
(FP8, bf16 or fp32); `qwen35::base_source::open_base` also accepts a GGUF of any decodable quantization
as a LoRA/QLoRA base. New models write a name map and call `PackedSource::new`; they never decode.
