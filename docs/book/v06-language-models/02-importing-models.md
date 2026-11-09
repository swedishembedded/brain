<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Importing Pretrained Models and Proving Parity

*How do we load someone else's trained weights into our own implementation and prove it computes the same thing?*

A model that somebody else trained reaches you as a directory of files, and nothing in them says what the numbers mean. There are names, shapes and bytes. Whether the matrix called `q_proj` is stored as outputs by inputs or the other way round, whether a fused matrix holds queries first, whether the epsilon of the norms is written down or left to a default, whether the output matrix is a copy of the embedding table: each of these is a convention of the publisher, and an implementation that guesses one wrong still loads, still runs and still produces numbers that look like a language model's. The previous chapter ended with a parity test whose only demands were that the top prediction agree at every position and that no logit be off by 1.0, and it quoted an error near $2 \times 10^{-4}$ that the test would also have accepted if it had been five thousand times larger. This chapter makes that test a proof: how a checkpoint is read, how its names become the engine's, how a configuration is read without guessing, and how two implementations are compared to a tolerance derived from the arithmetic instead of tuned until the test passes.

The tools are the ones already built. The decoder is the tiny decoder of Chapter 1 ($V = 11$, $D = 6$, two layers, four query heads over two key-value heads of width 4, $F = 10$), published here in a layout that has the awkward habits real ones have, and every claim below is measured on it. Six ideas from earlier chapters come first; the answers are at the end.

1. (Volume I, [Chapter 15](../v01-mathematics/15-floating-point.md), rounding error.) What is $u$ for 32-bit numbers, and what does the chapter bound for the error of a sum of $n$ numbers added left to right, relative to what?
2. (Volume IV, [Chapter 1](../v04-computation/01-number-formats.md), bf16 and fp16.) How many exponent and fraction bits do bf16 and fp16 have, and how do you turn a bf16 bit pattern into the 32-bit pattern of the same number?
3. (Volume I, [Chapter 5](../v01-mathematics/05-arrays.md), row-major layout.) A matrix of 2 rows and 3 columns is stored row by row in a flat buffer. Which position holds the entry in row 1, column 0, and what is the shape of its transpose?
4. (Volume I, [Chapter 3](../v01-mathematics/03-vectors.md), cosine similarity.) What happens to the cosine similarity of two vectors if one of them is multiplied by 3?
5. (Volume III, [Chapter 6](../v03-neural-networks/06-gradient-checking.md), relative error and test dimensions.) Why does a gradient check use dimensions that are all different, and what does a relative error add that an absolute one does not show?
6. (Volume VI, [Chapter 1](01-modern-decoder.md), the modern decoder recipe.) What are the shapes of $\mathbf{W}_Q$ and $\mathbf{W}_K$ of the tiny decoder, and which tensors does the head have when it is tied and when it is not?

## A file that is only a header and bytes

The way a checkpoint is laid out in a file is its **checkpoint format**: which names, shapes and number formats a file may hold, and where in the file the bytes of each tensor are. A format has to answer two questions for a reader that has never seen the file: what is in it, and where. The format brain writes, and the first it reads (it also reads quantized single files and the older archives of the PyTorch ecosystem), is **safetensors**, a file of three parts: an unsigned 64-bit little-endian number $N$ in the first eight bytes, then $N$ bytes of JSON text (the header), then the raw bytes of every tensor, back to back. The header maps each tensor name to its element type, its shape and its `data_offsets`, the first byte and one past the last byte of its data, counted from the end of the header. A brain checkpoint is itself such a file: its metadata entry holds the model's configuration as a JSON string under the key `brain.config` and a model card under `brain.card` (`crates/checkpoint/src/st.rs`), so an import ends in the format it began in.

Here is one, written byte by byte by the chapter's worked module for three small tensors: a 2 by 3 matrix in 32-bit floats, three numbers in bf16 and two in fp16.

<!-- worked: importing_models.safetensors_file -->
The file of three tensors written by `write_safetensors`: `w` is F32 with shape [2, 3] holding 1, -2, 0.5, 0.25, 3, -0.125 (row by row); `b` is BF16 with shape [3] holding 1, -2.5, 0.3; `h` is F16 with shape [2] holding 0.1, 65504. The header is the JSON of `{"__metadata__": {"format": "pt"}}` and the three entries in that order, written without spaces, with each entry's keys in the order dtype, shape, data_offsets, and padded with spaces to a multiple of 8. The file is 242 bytes. Its bytes, 16 to a row, with the offset of each row at the left:

```text
0000  c8 00 00 00 00 00 00 00 7b 22 5f 5f 6d 65 74 61
0010  64 61 74 61 5f 5f 22 3a 7b 22 66 6f 72 6d 61 74
0020  22 3a 22 70 74 22 7d 2c 22 77 22 3a 7b 22 64 74
0030  79 70 65 22 3a 22 46 33 32 22 2c 22 73 68 61 70
0040  65 22 3a 5b 32 2c 33 5d 2c 22 64 61 74 61 5f 6f
0050  66 66 73 65 74 73 22 3a 5b 30 2c 32 34 5d 7d 2c
0060  22 62 22 3a 7b 22 64 74 79 70 65 22 3a 22 42 46
0070  31 36 22 2c 22 73 68 61 70 65 22 3a 5b 33 5d 2c
0080  22 64 61 74 61 5f 6f 66 66 73 65 74 73 22 3a 5b
0090  32 34 2c 33 30 5d 7d 2c 22 68 22 3a 7b 22 64 74
00a0  79 70 65 22 3a 22 46 31 36 22 2c 22 73 68 61 70
00b0  65 22 3a 5b 32 5d 2c 22 64 61 74 61 5f 6f 66 66
00c0  73 65 74 73 22 3a 5b 33 30 2c 33 34 5d 7d 7d 20
00d0  00 00 80 3f 00 00 00 c0 00 00 00 3f 00 00 80 3e
00e0  00 00 40 40 00 00 00 be 80 3f 20 c0 9a 3e 66 2e
00f0  ff 7b
```

The header text, which is one line in the file and is broken here after each entry, with the padding spaces at its end not shown, is

```text
{"__metadata__":{"format":"pt"},
"w":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},
"b":{"dtype":"BF16","shape":[3],"data_offsets":[24,30]},
"h":{"dtype":"F16","shape":[2],"data_offsets":[30,34]}}
```

The segments of the file:

| bytes | length | what | detail |
|---|---:|---|---|
| 0 .. 8 | 8 | header length N, unsigned 64-bit little-endian | N = 200 |
| 8 .. 208 | 200 | header: JSON text, then spaces up to a multiple of 8 | 199 characters of JSON, 1 of padding |
| 208 .. 232 | 24 | data of `w` (F32 [2, 3]) | data offsets 0 .. 24 |
| 232 .. 238 | 6 | data of `b` (BF16 [3]) | data offsets 24 .. 30 |
| 238 .. 242 | 4 | data of `h` (F16 [2]) | data offsets 30 .. 34 |

Parsed back (data of `h` at its stored precision): the tensors, with the offsets that the header declares, the absolute offset of the first byte of each tensor, that offset modulo 4 (a 32-bit value read in place needs 0), and the decoded numbers.

| name | dtype | shape | offsets | absolute start | mod 4 | decoded |
|---|---|---|---:|---:|---:|---|
| `w` | F32 | [2, 3] | 0 .. 24 | 208 | 0 | 1, -2, 0.5, 0.25, 3, -0.125 |
| `b` | BF16 | [3] | 24 .. 30 | 232 | 0 | 1, -2.5, 0.300781 |
| `h` | F16 | [2] | 30 .. 34 | 238 | 2 | 0.0999756, 65504 |
<!-- /worked -->

![A strip of five boxes: an 8-byte length field with the number 200, a 200-byte JSON header, and the data of tensors w, b and h of 24, 6 and 4 bytes, with the byte offsets 0, 8, 208, 232, 238 and 242 at the boundaries, and a note below the header listing its decoded entries.](img/safetensors-layout.svg)

Everything a reader needs to know is in the first 208 bytes, and a reader can use it without trusting it, which matters because the file comes from outside. The length $N$ is the first thing read and the first thing that can lie: a corrupt or foreign file makes it arbitrary, and a reader that allocates $N$ bytes turns a bad download into an allocation the process cannot survive. In brain, commit `e26548250` (September 2026) found that every reader of the prefix trusted it, the header-only readers by allocating it and the mapped readers by computing $8 + N$ unchecked, and made one function, `validate_header_len` in `crates/checkpoint/src/safetensors.rs`, the check all of them go through: a length longer than the bytes that follow the prefix, or than a cap of 100 MB that the code's comment says is the format's reference limit, is refused with an error that names the file, the claimed length and the file size. The test `crates/checkpoint/tests/it/corrupt_header.rs` drives every reader with the length set to the largest 64-bit number, to the file size plus one and to the cap plus one. The same distrust runs down to each entry. In the book's pseudocode, the two checks a reader makes before it slices anything are these:

```text
function read_header_length(b: bytes) -> int
    require len(b) >= 8, "shorter than the length prefix"
    N = 0
    for k in 0 .. 8:
        N = N + b[k] * 256^k                  # little-endian: byte 0 is the least significant
    require N <= len(b) - 8, "header longer than the file"
    return N

function check_entry(start: int, end: int, shape: list<int>, width: int, data_len: int) -> none
    require 0 <= start and start <= end and end <= data_len, "offsets outside the data"
    count = 1
    for i in 0 .. len(shape):
        count = count * shape[i]
    require end - start == count * width, "extent is not the bytes of the shape"
```

`width` is the number of bytes of one element of the entry's type: 4 for F32, 2 for F16 and BF16. The second check is the one that catches a header that misdescribes its data, and `parse` in the same file of brain makes exactly these checks, refusing by name a tensor whose offsets are not an ordered pair inside the data or whose extent is not the bytes of its shape. It also refuses by name what it cannot decode. Brain's reader decodes 32-bit floats, fp16, bf16, 8-bit E4M3 floats and a few integer types into 32-bit floats, and answers an 8-bit E5M2 tensor with an error of its own, because the alternative is to decode its bytes as E4M3, which gives plausible numbers that are wrong. Any other type, a 64-bit float included, is refused too, so a golden stored for comparison is stored in 32 bits, and the rounding that this adds is a term of the error budget below.

The figure's last line needs one more observation. The data of the three tensors starts at byte 208, a multiple of 8, because this writer pads the header text with spaces up to a multiple of 8, which leaves it valid JSON, and the absolute start of each tensor is what matters to a reader that casts bytes in place instead of copying them. The table shows it: `w` starts at an address divisible by 4 and `h` at one that is only even, which is fine for 16-bit values and would not be for 32-bit ones. Brain's zero-copy path, the method `raw_words` of `TensorSource` in `crates/checkpoint/src/lib.rs`, hands out a tensor's bytes as 32-bit words only when the stored type is already the one the device binds and, for a memory-mapped file, the byte range is 4-byte aligned, and otherwise falls back to a decoding path. Alignment is a property of where a writer put the tensors, and a reader should check it, not assume it.

## Sixteen bits per number, and what they decode to

Most published weights are stored in bf16 or fp16, and the engine computes in 32 bits, so the import includes a conversion, and a conversion is the first place a number can silently change. The two 16-bit formats of Volume IV, Chapter 1 convert differently. A bf16 number has the exponent of a 32-bit one and 7 fraction bits, so its pattern is the top 16 bits of the 32-bit pattern, and decoding is a shift: in brain `bf16_to_f32` is `f32::from_bits((h as u32) << 16)`. An fp16 number has 5 exponent bits and 10 fraction bits, so its exponent has to be re-biased and its subnormals normalized, which is why `f16_to_f32` in the same file is a function of branches. Writing a number takes the opposite trip with a rounding in it.

<!-- worked: importing_models.dtype_decoding -->
Five numbers stored in the 16-bit formats of Volume IV, Chapter 1. Each is first rounded to 32 bits (its pattern in the second column), then to bf16 (1 sign, 8 exponent, 7 mantissa bits, ties to even) and to fp16 (1, 5, 10, ties to even); the decoded value is shown with 9 significant digits and its relative difference from the 32-bit number. A bf16 pattern is the top 16 bits of a 32-bit one after rounding; an fp16 pattern needs its exponent re-biased, so 70000 overflows to infinity and 6e-8 is a subnormal.

| x | fp32 pattern | bf16 pattern | bf16 value | bf16 relative difference | fp16 pattern | fp16 value | fp16 relative difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 0x3E99999A | 0x3E9A | 0.30078125 | 2.60e-3 | 0x34CD | 0.300048828 | 1.63e-4 |
| -2.5 | 0xC0200000 | 0xC020 | -2.5 | 0 | 0xC100 | -2.5 | 0 |
| 0.001 | 0x3A83126F | 0x3A83 | 0.000999450684 | 5.49e-4 | 0x1419 | 0.00100040436 | 4.04e-4 |
| 70000 | 0x4788B800 | 0x4789 | 70144 | 2.06e-3 | 0x7C00 | inf | inf |
| 6e-08 | 0x3380D959 | 0x3381 | 6.00703061e-08 | 1.17e-3 | 0x0001 | 5.96046448e-08 | 6.59e-3 |
<!-- /worked -->

The column of relative differences is the cost of storage. Round to nearest in a format with $p$ significand bits is off by at most $2^{-p}$ relative, which is $2^{-8}$ for bf16 (the 7 stored fraction bits and the hidden one) and $2^{-11}$ for fp16. The 70000 shows the trade the formats make: bf16 keeps the range of 32 bits and holds 70144, fp16 has three more fraction bits but overflows, and the same 6e-8 that bf16 holds to a part in a thousand is an fp16 subnormal with a single significant bit. A publisher chooses; the importer has to honour the choice by decoding what is stored, never by assuming a format from a file name.

The relative change is small per weight and is not small per model. The tiny decoder's 1,114 weights, as drawn and then rounded to bf16, give logits that differ by this much:

<!-- worked: importing_models.bf16_weights_effect -->
The weights of the tiny model as drawn (64 bits) and rounded to bf16, 1,114 numbers. Largest relative change of one weight 3.75e-3 (the bound for round to nearest is 2^-8 = 3.91e-3). Logits of the two models on the text of Chapter 1: rel_l2 2.91e-3, max_abs 9.19e-3, cosine 0.999996. The bound of the previous section, 1.38e-5, is 211 times smaller than this change, which is why a golden has to be computed from the published weights and not from the weights before they were rounded.
<!-- /worked -->

That is the first rule of a golden: it is computed from the published numbers, the ones in the file, and never from the weights as they were before the publisher rounded them. The rest of the chapter takes the bf16 weights as the model, and its goldens are the outputs of those numbers.

## When the weights do not fit in one file

A checkpoint of a few gigabytes is split into shards, because a hosting service limits the size of a file and because a writer can then write the shards one after another. The split carries a small second file, the index, which lists for each tensor name the shard that holds it and the total size. The tiny model, cut greedily at 1,024 bytes per shard (a new shard when the next tensor would take the current one past the budget, a tensor larger than the budget alone in a shard), is three files:

<!-- worked: importing_models.sharded_index -->
The 19 tensors of the published tiny model in bf16 (2,228 bytes of tensor data), in file order, cut into shards by the greedy rule (a new shard when the next tensor would take the current one past 1024 bytes). Each shard is a safetensors file of its own, in exactly the format of the single file above (the 8-byte length N, the compact JSON header with `__metadata__` `{"format":"pt"}` first and then one entry per tensor of the shard with offsets counted from the start of that shard's data, padded with spaces to a multiple of 8, then the data), so the file size is 8 + N + tensor bytes; the index lists, for every tensor name, the shard that holds it, and the total size. The tensors, in file order, each BF16:

| tensor | shape | shard |
|---|---|---:|
| `model.embed_tokens.weight` | 11 x 6 | 1 |
| `model.layers.0.input_layernorm.weight` | 6 | 1 |
| `model.layers.0.self_attn.qkv_proj.weight` | 32 x 6 | 1 |
| `model.layers.0.self_attn.q_norm.weight` | 4 | 1 |
| `model.layers.0.self_attn.k_norm.weight` | 4 | 1 |
| `model.layers.0.self_attn.o_proj.weight` | 6 x 16 | 1 |
| `model.layers.0.post_attention_layernorm.weight` | 6 | 1 |
| `model.layers.0.mlp.gate_up_proj.weight` | 20 x 6 | 1 |
| `model.layers.0.mlp.down_proj.weight` | 6 x 10 | 2 |
| `model.layers.1.input_layernorm.weight` | 6 | 2 |
| `model.layers.1.self_attn.qkv_proj.weight` | 32 x 6 | 2 |
| `model.layers.1.self_attn.q_norm.weight` | 4 | 2 |
| `model.layers.1.self_attn.k_norm.weight` | 4 | 2 |
| `model.layers.1.self_attn.o_proj.weight` | 6 x 16 | 2 |
| `model.layers.1.post_attention_layernorm.weight` | 6 | 2 |
| `model.layers.1.mlp.gate_up_proj.weight` | 20 x 6 | 2 |
| `model.layers.1.mlp.down_proj.weight` | 6 x 10 | 3 |
| `model.norm.weight` | 6 | 3 |
| `lm_head.weight` | 11 x 6 | 3 |

| shard | file bytes | tensors | tensor bytes | first .. last tensor |
|---|---:|---:|---:|---|
| `model-00001-of-00003.safetensors` | 1796 | 8 | 988 | `model.embed_tokens.weight` .. `model.layers.0.mlp.gate_up_proj.weight` |
| `model-00002-of-00003.safetensors` | 1792 | 8 | 976 | `model.layers.0.mlp.down_proj.weight` .. `model.layers.1.mlp.gate_up_proj.weight` |
| `model-00003-of-00003.safetensors` | 552 | 3 | 264 | `model.layers.1.mlp.down_proj.weight` .. `lm_head.weight` |

The index file, abbreviated to its first three names:

```text
{
 "metadata": {"total_size": 2228},
 "weight_map": {
  "model.embed_tokens.weight": "model-00001-of-00003.safetensors",
  "model.layers.0.input_layernorm.weight": "model-00001-of-00003.safetensors",
  "model.layers.0.self_attn.qkv_proj.weight": "model-00001-of-00003.safetensors",
  ...
 }
}
```

Reading every shard in sorted order returns 19 tensors, each once, with values equal to the originals: yes. The two failures a reader must refuse:

| damage | result |
|---|---|
| shard 2 deleted | index names shard model-00002-of-00003.safetensors, which is missing |
| a name pointing at the wrong shard | index lists ghost.weight, which no shard holds |
<!-- /worked -->

A reader of a sharded checkpoint has two jobs beyond reading each file: to know which files to read, and to notice when the set is not the whole. In brain, `read_model_dir` in `crates/checkpoint/src/safetensors.rs` accepts either the single `model.safetensors` or the index (`model.safetensors.index.json`, or the name the diffusion libraries use, `diffusion_pytorch_model.safetensors.index.json`), takes the distinct shard names from the index, reads them in sorted order, and returns every tensor, so the order is deterministic; the index decides which shards, and a later step decides whether the tensors are the right ones. The cheap pre-flight `has_model_weights` checks, before any tensor is read, that the index and every shard it names exist, because a download interrupted after a few shards leaves a directory that looks complete to a check that only looks for one of them. The reader of this chapter is stricter than that in one respect and the worked table shows it: it also refuses a name that the index lists and no shard holds. Brain leaves that question to the coverage check of the next section, which refuses a checkpoint that lacks a tensor the model needs by name.

## From the published names to the model's own

The tensors are in memory, decoded, under the publisher's names. The model wants its own. A **weight-name mapping** is the table and the rules that turn the names, layouts and groupings of one layout into another's: which source tensor produces each destination tensor, which rows of it, and in which orientation. For the tiny decoder, published with fused queries, keys and values in one matrix, fused gate and up in another, and every matrix stored as outputs by inputs, the mapping is:

<!-- worked: importing_models.name_mapping -->
The plan that builds the tiny decoder of Chapter 1 (V = 11, D = 6, L = 2, H = 4, 2 key-value heads of width 4, F = 10) from its published layout. The checkpoint holds 19 tensors; the plan makes 25. Layer 0 is shown (layer 1 is the same with `N` = 1); "rows a .. b" is a range of rows of the source matrix, and "transpose" turns a row-major [out, in] matrix into the [in, out] layout of the book's `x @ W`.

| checkpoint tensor | stored as | book tensor | shape | operation |
|---|---|---|---|---|
| `model.embed_tokens.weight` | 11 x 6 | W_emb | 11 x 6 | copy |
| `model.layers.N.input_layernorm.weight` | 6 | gain 1 | 6 | copy |
| `model.layers.N.self_attn.qkv_proj.weight` | 16 of 32 rows x 6 | W_Q | 6 x 16 | rows 0 .. 15, transpose |
| `model.layers.N.self_attn.qkv_proj.weight` | 8 of 32 rows x 6 | W_K | 6 x 8 | rows 16 .. 23, transpose |
| `model.layers.N.self_attn.qkv_proj.weight` | 8 of 32 rows x 6 | W_V | 6 x 8 | rows 24 .. 31, transpose |
| `model.layers.N.self_attn.q_norm.weight` | 4 | QK-norm gain, queries | 4 | copy |
| `model.layers.N.self_attn.k_norm.weight` | 4 | QK-norm gain, keys | 4 | copy |
| `model.layers.N.self_attn.o_proj.weight` | 6 x 16 | W_O | 16 x 6 | transpose |
| `model.layers.N.post_attention_layernorm.weight` | 6 | gain 2 | 6 | copy |
| `model.layers.N.mlp.gate_up_proj.weight` | 10 of 20 rows x 6 | W_gate | 6 x 10 | rows 0 .. 9, transpose |
| `model.layers.N.mlp.gate_up_proj.weight` | 10 of 20 rows x 6 | W_up | 6 x 10 | rows 10 .. 19, transpose |
| `model.layers.N.mlp.down_proj.weight` | 6 x 10 | W_down | 10 x 6 | transpose |
| `model.norm.weight` | 6 | last gain | 6 | copy |
| `lm_head.weight` | 11 x 6 | W_out | 11 x 6 | copy |

Run on the file, the plan passes its own checks (every destination once, every source used, every shape the model's) and builds parameters equal to the ones that were written: yes.
<!-- /worked -->

![Eight tensors of one layer of the published layout on the left, in boxes, and the eleven tensors of the book's decoder on the right, with an arrow for each step: the fused projection of queries, keys and values fans out into three transposed row ranges, the fused gate and up into two, and the output and down matrices are transposed.](img/name-mapping.svg)

Three kinds of step appear in the table, and each is a way the layouts of publishers differ. A rename (`input_layernorm.weight` is the gain of the first norm) is the easy kind, with an obvious check: the shapes must be those the model's configuration gives. A cut (`rows 0 .. 15` of the fused projection are the queries) needs the boundaries from the configuration, the number of query heads times the head width, and the order in which the publisher fused them: queries, keys, then values here, which is a fact about the publisher and not something the shapes reveal when the key and value widths are equal. A transpose is a statement about the orientation of the matrix, and it is the subject of the next section.

The plan is a list of steps, and what makes it safe is that it is checked before it runs: every destination is produced by exactly one step, every source tensor is used by some step or deliberately dropped, and every result has the shape the model needs. That is `apply_plan`'s own check in the worked module, and it is the shape of brain's. In `crates/qwen3/src/hf.rs` the whole mapping of a dense decoder is one table, `LEAVES`, of pairs of a published leaf name and the model's (`self_attn.q_proj.weight` is `attn.wq.weight`), read in both directions, and `HfNames::to_brain` sorts every tensor name of a checkpoint into four classes: a parameter, one the model deliberately does not store (a tied head, or the buffer of rotary frequencies that older files serialize and brain derives from the configuration), one that belongs to another component of a composite checkpoint, and one that is inside this decoder's namespace and is not something it has. The fourth is an error, by design: a bias the configuration did not declare or a normalization of queries on an architecture without one means that importing past it would lose weights. `brain_init_from_hf` in `crates/qwen3/src/import.rs` then checks coverage (every parameter in the model's list produced once, none left over), and `RemapSource` in `crates/checkpoint/src/remap.rs` expresses the same plan as data for a streaming load: a destination is the whole of a source tensor, a range of elements of one (the cut of a fused projection), the concatenation of several, or a permutation of whole rows. Its `validate` checks, from shapes alone and without reading a tensor, that every destination is produced and every size matches, and it reports every problem it finds, not the first.

Before the weights can be mapped, the architecture has to be known, and brain keeps that in one place. The registry `ARCHS` in `crates/arch/src/lib.rs` has one row per architecture with its canonical id, and for the checkpoints that carry a class name, the exact class names a published configuration declares in `architectures`. `by_hf` matches the class name exactly and never by substring, and the comment says why: a scan for `"qwen"` that matches before `"omni"` would send a checkpoint of the multimodal family to the importer of the dense one, which would then fail on tensors it does not have or, worse, not fail. Routing a checkpoint is the first mapping, from a name to a model, and it has the same rule as the others: an exact match, and an error otherwise.

## Row-major, and the matrix that is secretly transposed

A weight matrix is a rectangle of numbers laid out as a line, and two publishers can store the same rectangle in two orders. The convention of the published layout is $\mathbf{y} = \mathbf{W}\mathbf{x}$ with $\mathbf{W}$ of shape [outputs, inputs], row by row; the decoder of Chapter 1 computes $\mathbf{y} = \mathbf{x}\mathbf{W}'$ with $\mathbf{W}'$ of shape [inputs, outputs], the layout of the book. Take the smallest case:

<!-- worked: importing_models.storage_order -->
A published matrix of 2 outputs and 3 inputs, W = [[1, 2, 3], [4, 5, 6]], stored row by row as the six numbers 1, 2, 3, 4, 5, 6. The published convention is y = W x, and the input is x = (1, 2, 3). The book's layout computes y = x W' with W' of shape [3, 2].

| how W' is made from the stored buffer | W' (rows) | shape check [3, 2] | y |
|---|---|---|---|
| transpose of W | [[1, 4], [2, 5], [3, 6]] | passes | [14, 32] |
| the stored shape [2, 3] kept | [[1, 2, 3], [4, 5, 6]] | refused: x has 3 entries, W' has 2 rows | none |
| the buffer cut again into [3, 2] | [[1, 2], [3, 4], [5, 6]] | passes | [22, 28] |

The published y is (14, 32). The correct transpose agrees with it; the third row has the right shape and the right count of numbers and computes something else.
<!-- /worked -->

The third row is the one to remember. A loader that checks only that the number of elements is right (six), or that cuts the buffer into the shape the model asks for, cannot tell the transpose from the buffer read in the wrong order, and the output is a plausible vector of the right length. The row before it is the lucky case, where keeping the stored shape fails because $x$ has three entries and the matrix two rows. That luck depends on the dimensions: a matrix that is square, or two matrices whose shapes happen to coincide, has no shape to disagree.

This is the degenerate-dimension lesson of Volume III, Chapter 6 and knowledge #4 (a toy configuration in which two dimensions are equal lets a swap of two matrices pass every test), met a second time in the other direction: there it said to choose dimensions all different in a test, here it says that a published model's dimensions are often equal, and that a transposed matrix of such a model loads. In brain the convention happens to agree: the comment at the head of `crates/qwen3/src/import.rs` records that its matrix product is $\mathbf{x}\mathbf{W}^\top$ with $\mathbf{W}$ of shape [out, in] row-major, exactly the layout of a published linear layer, so the import of that crate is a rename and a conversion of the number format and no tensor is transposed. A test model of the rectangular kind verifies that claim, as the table below does for the book's decoder, and a transposed weight must be found by running the model. The check that `brain_init_from_hf` makes against the model's parameter list compares element counts, not shapes, so it cannot find one, and the kernel library documents the same hazard from the other side: commit `3656ea626` (July 2026), which added the 2-D transposed convolution, notes in its message and in the header of `crates/kernels/wgsl/convtr2d.wgsl` that its weight layout is the transpose of the plain convolution's and holds exactly the same number of elements, so that a weight in the other layout always binds, never fails a size check, and computes a different operator.

## Mistakes that load without complaint

Each of the mistakes a mapping invites can now be made on purpose and measured. In the worked module, each is a change to the plan of the previous section; each run uses 64-bit arithmetic, so that the numbers below are the effect of the mistake alone. "Square" is a second tiny decoder, with every matrix square ($D = 8$ and four heads of width 2 over four key-value heads, so $\mathbf{W}_Q$, $\mathbf{W}_K$, $\mathbf{W}_V$ and $\mathbf{W}_O$ are 8 by 8, and $F = 8$) so that the shape check has nothing to object to.

<!-- worked: importing_models.import_mistakes -->
Each mistake of the mapping, made in the import plan and run through the 64-bit forward pass on the text of Chapter 1 (ids 3, 9, 1, 7, 4, 4, 10, 2, eps 1e-6, the epsilon of the published configuration, base 100). The published numbers are the bf16 weights; the golden is the logits of those weights, and the three measures compare the 8 x 11 logits of the imported model with it (max_abs, rel_l2 and cosine are defined in the next section). "Square" is a model whose matrices are all square (V = 11, D = 8, L = 2, H = HKV = 4, head width 2, F = 8; weights from the generator `importing_models.square`, seed 0x39d64f12e64012f2, drawn in the order of Chapter 1). A plan checks the shape of each tensor it builds before it runs anything.

| mistake | model | outcome | max_abs | rel_l2 | cosine | argmax agrees |
|---|---|---|---|---:|---:|---:|
| none (the correct plan) | tiny (rectangular) | runs | 0 | 0 | 1.000000 | 8 of 8 |
| transposes left out | tiny (rectangular) | refused: W_Q (layer 0): shape [16, 6], the model needs [6, 16] |  |  |  |  |
| transposes left out | square | runs | 4.356e+0 | 1.232e+0 | 0.253373 | 2 of 8 |
| keys and values exchanged | tiny (rectangular) | runs | 2.552e+0 | 5.908e-1 | 0.824604 | 7 of 8 |
| gate and up exchanged | tiny (rectangular) | runs | 1.767e+0 | 4.146e-1 | 0.911387 | 8 of 8 |
| head dropped, tied to the embedding | tiny (rectangular) | runs | 8.186e+0 | 2.966e+0 | 0.212802 | 1 of 8 |
<!-- /worked -->

Read the rows against what a reader of the output would conclude. Left on the rectangular model, the missing transposes are refused by name at the first matrix. On the square model they are not, and the logits are the wrong function: the cosine with the right ones is 0.25 and the top prediction agrees at two of eight positions. Exchanging the keys and the values in the cut is shape-legal, because both are $2 \times 4 = 8$ rows and the model has grouped heads of the same width; seven of eight top predictions survive and the logits are off by more than half their norm. Exchanging gate and up is the most insidious, since it is always shape-legal: the network computes $\mathrm{silu}(\mathbf{x}\mathbf{W}_{\mathrm{up}}) \odot (\mathbf{x}\mathbf{W}_{\mathrm{gate}})$ instead, nothing about the shapes or the count of weights is wrong, and the top prediction is the same at all eight positions with logits that differ by 41 percent of their norm. Dropping the head because the configuration says it is tied is one line (a checkpoint can ship a head, which a reader of a tied model is allowed to ignore), and the logits are from a different function entirely. Not one of them raises an error. A whole-sentence comparison is the only evidence, and a test that stops at "the top prediction agrees" lets the gate-and-up mistake through.

Brain's history holds the same lessons at larger scale. Commit `0581928e9` (July 2026) is a missing bias: the fused query-key-value projection of a published model is a linear layer with a bias, and the checkpoint ships it, but the port's block did the matrix product and no addition. The gradient checks passed, because their synthetic weights had no such bias and a gradient check compares the model with itself (Volume III, Chapter 6). Real-weight parity did not: the mean absolute error was 1.08 on the first comparison, growing with position, and after the fix 5.3 times $10^{-6}$ with the top prediction in agreement, as the commit's message reports; the bias is applied at `crates/moondream3/src/decoder.rs` conditionally, when `attn.qkv.bias` is in the checkpoint. And commit `2c78af10f` (August 2026), a diffusion transformer, found a naming trap, in the commit's words one of four: the library that writes the checkpoint calls the cross-attention normalization `norm2` and the feed-forward normalization `norm3`, and the reference implementation calls them the other way round, so a mapping that passes the names through "imports cleanly, validates cleanly, and normalises with the wrong learned affine at both sites". The mapping in `crates/wan/src/import.rs` has the swap as one row of its table and a test that pins it. In both stories every check on the load succeeded, and what found the error was a comparison of outputs against an implementation that is not one's own.

## What the configuration leaves unsaid

The second file of a checkpoint is its configuration, a short JSON of the model's shape: width, depth, heads, the width of the feed-forward network, the vocabulary, the epsilon of the norms, the base of the rotary embedding, whether the head is tied. A reader that builds a model from it has to decide what to do with a field that is absent, and the decision is a **config default**: the value a reader substitutes for a field a configuration does not state. A default is a guess about the publisher. It is right when it equals what the publisher meant, and wrong in a way that only the model's output can show.

The tiny model's published configuration states everything, and the table removes one field at a time and lets the reader supply its own default (the key-value heads default to the number of query heads, the epsilon to 1e-5, the rotary base to 10,000, the tied head to false):

<!-- worked: importing_models.config_defaults -->
The published configuration of the tiny model, with the fields a reader needs:

```json
{
 "architectures": [
  "TinyForCausalLM"
 ],
 "vocab_size": 11,
 "hidden_size": 6,
 "num_hidden_layers": 2,
 "num_attention_heads": 4,
 "num_key_value_heads": 2,
 "head_dim": 4,
 "intermediate_size": 10,
 "rms_norm_eps": 1e-06,
 "rope_theta": 100.0,
 "tie_word_embeddings": false
}
```

One field at a time is removed and the reader's default is used instead (the reader defaults the key-value heads to the query heads, the epsilon to 1e-5, the rotary base to 10000 and the tied head to false). The weights and the text are those of the previous table; the golden is the 64-bit forward pass with the published values (eps 1e-6, base 100); rel_l2 and cosine compare the logits computed with the default against it.

| omitted field | published | default | outcome | rel_l2 of logits | cosine |
|---|---|---|---|---|---:|
| `num_key_value_heads` | 2 | 4 (= the query heads) | refused: W_V (layer 0): rows 32 .. 47 of model.layers.0.self_attn.qkv_proj.weight, which has 32 rows |  |  |
| `rms_norm_eps` | 1e-06 | 1e-05 | runs | 3.58e-6 | 1.000000 |
| `rope_theta` | 100 | 10000 | runs | 2.47e-2 | 0.999711 |
| `tie_word_embeddings` | false | false | runs (the default is the truth) | 0 | 1.000000 |
| `tie_word_embeddings`, read by a reader whose default is true | false | true | runs, `lm_head.weight` dropped | 2.97e+0 | 0.212802 |
<!-- /worked -->

Two of the defaults are stopped by the loader. Without `num_key_value_heads` the reader assumes four key-value heads, and the plan asks for 48 rows of a 32-row matrix: the fused shape contradicts the guess, so a wrong default of a field that fixes a shape fails loudly, which is the best kind of failure. Two others run, and they are the ones to fear, because they change values and shapes remain. The rotary base of 10,000 where the published one is 100 moves the logits by 2.5 percent of their norm. The epsilon of 1e-5 where the publisher used 1e-6 moves them by $3.6 \times 10^{-6}$, almost nothing, and the reason is in the formula of the RMS norm: the epsilon is added to the mean of squares, $\sqrt{\mathrm{mean}(x^2) + \epsilon_{\mathrm{norm}}}$, so it matters in proportion to how small the mean of squares is.

<!-- worked: importing_models.eps_scale_table -->
The same wrong epsilon (1e-5 for the published 1e-6) at three scales of the activations: the embedding table is multiplied by the factor in the first column, the weights are unchanged, and the golden is the 64-bit forward pass with eps 1e-6 on the same scaled table.

| factor on the embedding table | rel_l2 of logits | max_abs of logits |
|---|---:|---:|
| 1 | 3.58e-6 | 1.51e-5 |
| 0.1 | 2.43e-4 | 1.31e-3 |
| 0.01 | 5.65e-2 | 2.75e-1 |
<!-- /worked -->

The error grows roughly as the inverse square of the activation scale, from $3.6 \times 10^{-6}$ to $5.6 \times 10^{-2}$ over two decades (and more slowly once the epsilon is comparable with the mean of squares), and a model whose activations are small in some layers is a model that reveals the wrong epsilon there and nowhere else. A parity run on one prompt at one scale can miss it, and the section below returns to what that implies for tolerances.

The last two rows are one key read by two readers with opposite defaults, and each is correct for the file it expects. A published configuration that omits `tie_word_embeddings` means false, which is how the class defaults of the publisher's library behave and what brain's reader of that format does (`from_hf` in `crates/qwen3/src/hf.rs`, `tie_word_embeddings.unwrap_or(false)`). Brain's own checkpoint header is read by `QwenConfig::from_json`, which defaults the same key to true, because its own writer always writes the key. A checkpoint from one world read by the reader of the other drops its head without a word.

Brain's history has the whole arc, and Chapter 1 told its first half: commit `8dedcb408` (September 2026), knowledge #176. The writer of the configuration had not written two switches, `qk_norm` and `attention_bias`, whose reader defaults are those of one architecture, so a decoder of another family saved by brain reloaded as the first; and the loaders of a quantized single-file format read the configuration with the reader of the header, found none of its keys, and built the default tiny model. The fix is the pattern of this section: a single checked resolver, `QwenConfig::from_reader`, which reads the shape from what the format declares and refuses a configuration that lacks a shape-defining key (`SHAPE_KEYS` in `crates/qwen3/src/config.rs`) instead of defaulting it, and a test, `json_roundtrip_is_identity`, that writes every preset and reads it back and requires equality. The next day's commit `fcc0646b5` made the other half explicit for published configurations: the reader documented that an omitted field takes the value its own class defaults to, but required the width, depth, heads, feed-forward size and vocabulary, so a composite that nests a trimmed decoder configuration could not be read, and each architecture now carries the defaults of its class for those fields, in a `Defaults` table in `crates/qwen3/src/hf.rs`. The same file refuses, by name, what the decoder does not implement: biases on the output projection or the feed-forward network, an activation that is not the sigmoid-gated one, and windowed attention, because a configuration that asks for one of them and loads with the switch ignored computes a different model. Defaults and refusals are the same discipline from two sides: say what the reader assumes, and refuse what it cannot honour.

A third commit shows what makes a configuration bug live long. Commit `114ea1061` (August 2026): a LoRA fine-tune, written by `Qwen::save` with its adapter tensors, was reloaded through a configuration that did not record the adapter, so the adapters never entered the parameter list and inference ran the unmodified base weights, "silently dropped at load". The test that existed asserted only that the fine-tuned loss did not rise above the baseline, "a statistic an unchanged base model also satisfies". The regression test added with the fix saves, reloads, and requires the logits of the reloaded model to equal the trained model's and to differ from the base's. A check that a broken result also passes is not a check, and that is the subject of the rest of the chapter.

## Proving two implementations agree

The import is done when a model that has never seen the reference's code computes the reference's function. To say so, you need three things that the previous sections did not: a reference output to compare with, a way to measure a difference, and a tolerance below which the difference is rounding error and above which it is a bug.

### What a golden contains

A **reference golden** is the stored output of the reference implementation for fixed inputs, kept as a file the implementation under test is compared against. For a model the golden holds the input, the outputs and, because the end of the pipeline tells you only that something is wrong and not where, the intermediate activations: the hidden state after the embedding and after every layer. Here is the golden of a six-layer version of the tiny decoder, written by the module as one safetensors file:

<!-- worked: importing_models.golden_contents -->
The golden for a 6-layer version of the tiny decoder (V = 11, D = 6, H = 4, 2 key-value heads of width 4, F = 10, L = 6; weights from the generator `importing_models.deep`, seed 0x6b38c609eb7b8278, drawn in the order of Chapter 1 and rounded to bf16), written by `write_safetensors` as one safetensors file. It holds the text, the hidden state after the embedding and after each layer, the output of the final norm and the logits, each tensor stored in the order listed below with the data back to back. The header is the compact JSON of `write_safetensors` (`__metadata__` first, then the tensors in that order), padded with spaces to a multiple of 8, and its `__metadata__` is exactly `{"generator":"importing_models.deep","eps":"1e-06","rope_base":"100","arithmetic":"64-bit, stored as 32-bit","tolerance_basis":"rounding count x 2^-24"}`, so the header is 840 bytes. The file is 2,768 bytes (1,920 of tensor data).

| tensor | dtype | shape | bytes | what |
|---|---|---|---:|---|
| `ids` | I32 | 8 | 32 | the input text |
| `hidden_00` | F32 | 8 x 6 | 192 | the embeddings |
| `hidden_01` | F32 | 8 x 6 | 192 | output of layer 0 |
| `hidden_02` | F32 | 8 x 6 | 192 | output of layer 1 |
| `hidden_03` | F32 | 8 x 6 | 192 | output of layer 2 |
| `hidden_04` | F32 | 8 x 6 | 192 | output of layer 3 |
| `hidden_05` | F32 | 8 x 6 | 192 | output of layer 4 |
| `hidden_06` | F32 | 8 x 6 | 192 | output of layer 5 |
| `final_norm` | F32 | 8 x 6 | 192 | output of the last RMS norm, before the head |
| `logits` | F32 | 8 x 11 | 352 | the output |
<!-- /worked -->

Brain's goldens have the same structure. `tools/goldens/deepseek_dump_reference.py` runs each DeepSeek text checkpoint through the reference implementation, truncated to its first two layers (the final norm and the head still applied), in 32-bit arithmetic on the CPU, with a fixed prompt, and writes a `forward.safetensors` holding the embeddings, the input norm of layer 0, the query, key and value after the bias and after the rotation, every hidden state and the logits, with a manifest carrying the sha256 of each file and the library versions used. The manifest has a second block, written by `tools/goldens/golden_source.py`, that records which checkpoint produced the golden and the architectural identity (width, depth, head count, vocabulary) that fixes every tensor shape in it, because a golden is a claim that "this is what the reference produced" and means nothing without the weights that produced it; the reading test, `brain_testutil::golden::Source` in `crates/testutil/src/golden.rs`, demands that every field of the recorded identity equal what the checkpoint under test reports, and otherwise refuses to compare anything. The dump script's own docstring names the failures that preceded it: a golden paired with another tier's weights died deep in the importer on a shape error, or compared against the wrong reference and certified a number that meant nothing, or printed to stderr and returned, which a test runner reports as a pass.

A golden is a large binary file derived from a reference implementation and a checkpoint, and brain's policy is that none is committed: the repository's `.gitignore` excludes `/testdata/` ("populated by `make testdata/fetch`, never committed"), `*.safetensors` and the fixture directories of individual crates, and what is committed is the generator. The cost of that policy is that a test must skip, loudly, when its golden is absent (the parity test of `crates/rrdbnet/tests/it/parity.rs` skips with a message that names the file and the command that produces it), and the benefit is that the repository holds the recipe for the evidence instead of an opaque copy of it. The book's goldens follow the same rule in its own way: they are never stored, only produced by the worked module from a named generator.

The golden is stored in 32 bits (the reader decodes nothing wider), so one rounding stands between the reference's value and the stored one, and it counts in the error budget.

### Three ways to measure a difference

Let $\mathbf{a}$ be what the implementation under test produced and $\mathbf{b}$ the golden, both flattened to $n_{\mathrm{el}}$ numbers. Three measures are in common use, and each sees what the others miss:

$$
\mathrm{max\_abs}(\mathbf{a}, \mathbf{b}) = \max_i \lvert a_i - b_i \rvert, \qquad
\mathrm{rel\_l2}(\mathbf{a}, \mathbf{b}) = \frac{\lVert \mathbf{a} - \mathbf{b} \rVert}{\lVert \mathbf{b} \rVert}, \qquad
\mathrm{cos\_sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\lVert \mathbf{a} \rVert \, \lVert \mathbf{b} \rVert}.
$$

`max_abs` is in the units of the outputs and finds the one bad entry that an average dilutes. `rel_l2` is the relative error of Volume III, Chapter 6 taken over the whole tensor, a norm-wise version of the entry-wise one, and it is dimensionless, so one tolerance reads the same for logits of size 2 and hidden states of size 200. Cosine similarity is a measure of direction only. Write $\mathbf{a} = \mathbf{b} + \mathbf{e}$ with the error $\mathbf{e}$ orthogonal to $\mathbf{b}$ and let $\rho$ be `rel_l2`: then $\cos = 1/\sqrt{1 + \rho^2}$, so $1 - \cos \approx \rho^2 / 2$. Cosine is quadratic in the error, which makes it blind in two ways. An error of 1 percent leaves $1 - \cos = 5 \times 10^{-5}$, and the floor 0.9999 of brain's DeepSeek ladder admits a relative error of $\sqrt{2 \times 10^{-4}} = 1.4$ percent. And a cosine computed in 32 bits is itself uncertain by about $10^{-7}$, so it cannot resolve an error below $\sqrt{2 \times 10^{-7}} \approx 4.5 \times 10^{-4}$ however well it is gated. A scale error is the extreme: $\mathbf{a} = s\mathbf{b}$ has cosine exactly 1 for every $s > 0$. This is knowledge #2, the lesson that cosine cannot see a dropped scale factor, and the worked table puts all three measures, and the gate of Chapter 1, on eight outputs:

<!-- worked: importing_models.metric_blindness -->
Eight outputs compared with the 64-bit logits of the published tiny model (the two rows about the rotary base and the epsilon: with the logits at the published eps 1e-6, base 100) by the three measures and by three tests: cosine at least 0.9999 (the floor of the DeepSeek ladder described below), rel_l2 at most the bound 1.38e-5 of the previous table, and the gate of Chapter 1's forward-parity test (the top prediction agrees at all 8 positions and max_abs is below 1.0). 1 - cosine is printed because cosine is 1 to six places for most rows.

| output | 1 - cosine | cosine test | rel_l2 | rel_l2 test | max_abs | gate of Chapter 1 |
|---|---:|---|---:|---|---:|---|
| final gain scaled by 1.05 (a dropped or doubled scale) | below 1e-12 | pass | 5.00e-2 | fail | 1.15e-1 | pass |
| logits scaled by 0.5 | below 1e-12 | pass | 5.00e-1 | fail | 1.15e+0 | fail |
| gate and up exchanged | 8.86e-2 | fail | 4.15e-1 | fail | 1.77e+0 | fail |
| keys and values exchanged | 1.75e-1 | fail | 5.91e-1 | fail | 2.55e+0 | fail |
| head tied to the embedding | 7.87e-1 | fail | 2.97e+0 | fail | 8.19e+0 | fail |
| rotary base 10000 for the published 100 | 2.89e-4 | fail | 2.47e-2 | fail | 9.60e-2 | pass |
| epsilon 1e-5 for the published 1e-6 | 4.03e-12 | pass | 3.58e-6 | pass | 1.51e-5 | pass |
| 32-bit arithmetic, left to right | below 1e-12 | pass | 1.26e-7 | pass | 3.92e-7 | pass |
<!-- /worked -->

A final gain 5 percent too large is invisible to cosine (the cosine is 1 to a part in $10^{12}$ and the table says "below") and to the gate of Chapter 1 (0.115 is far under 1.0), and visible to `rel_l2` at exactly 0.050. The wrong rotary base passes the gate of Chapter 1 as well, with a largest error of 0.096, while its cosine of 0.9997 falls just under the floor and `rel_l2` stands at 2.5 percent. The wrong epsilon passes every test, and it should be asked why. Nothing here argues that one measure is the right one. Cosine finds direction errors, `rel_l2` finds those and scale errors, `max_abs` finds the localized ones, and a gate that asserts only one is a gate with a blind spot that the table located. Brain's records agree: `crates/rrdbnet/tests/it/parity.rs` defines the three measures and asserts both a cosine floor and a `rel_l2` ceiling at every rung, with a comment that its network has two residual scalings of 0.2 that "a cosine-only ladder would miss completely", and its test model asserts that its toy dimensions differ, citing the degenerate-dimension lesson (knowledge #4) by number. The file does not say where its ceiling of $2 \times 10^{-4}$ comes from; the derivation below is how it would be obtained. The DeepSeek ladder in `crates/qwen3/tests/it/deepseek_parity.rs` asserts the row cosine only, floor 0.9999; by the arithmetic above it would not see a doubled final gain, and it is a reason to add the `rel_l2` that knowledge #2 asks for.

### A tolerance from the error model, not from the result

A number such as 2e-4 or 0.9999999 has to come from somewhere. The usual source is the run itself: run the comparison, see what the error is, set the tolerance a little above it. That tolerance passes by construction and says nothing about the bugs it would admit. The alternative is to ask what the arithmetic itself must allow. The **parity tolerance** is the largest difference between two implementations of the same function that rounding alone can explain, derived from the rounding model of Volume I, so that anything above it is evidence of a bug.

Every operation of a 32-bit implementation is a correctly rounded result of the exact one, $\mathrm{fl}(a \circ b) = (a \circ b)(1 + \delta)$ with $\lvert \delta \rvert \le u = 2^{-24}$. A dot product of $n$ terms, in any order of summation, is computed with a relative error, measured against $\sum_i \lvert a_i b_i \rvert$, of at most $n u$ to first order: each of the $n$ products is rounded once, and each term passes through at most $n - 1$ rounded additions, the bound of Chapter 15 of Volume I, which holds for every order, from left to right to a tree. The order matters only to the actual error, which is why two implementations that add in different orders may disagree and both be right. Now chain the stages of a network. If a stage multiplies the relative error of its input by a factor at most 1 and adds at most $n_s u$ of its own, the relative error of the output is at most the sum of the $n_s$ along the chain times $u$. Write $N_{\mathrm{rnd}} = \sum_s n_s$ for the number of roundings on the longest chain of dependent operations, and set

$$
\tau_{\mathrm{par}} = N_{\mathrm{rnd}}\, u .
$$

Two assumptions sit in that sentence, and the chapter does not hide them. The factor of at most 1 is not a theorem for a network: a stage that cancels amplifies the relative error of its input by its condition number (Volume I, Chapter 15), and the bound is only as good as that is small. And only the chain counts, not the whole count of operations, since the roundings of different entries do not add. The table counts the chain for the tiny decoder, on the last of 8 positions, where the softmax and the weighted sum of the values run over 8 terms:

<!-- worked: importing_models.error_budget -->
Roundings along the longest chain of dependent operations of the tiny decoder for the last of T = 8 positions (D = 6, H = 4, head width 4, F = 10), counted by the rule that a sum of n products costs n roundings and each further operation on a result costs one. The RMS norm is the one of the book's pseudocode: `inv_std = 1 / sqrt(mean(x^2) + eps)`, `xhat = x * inv_std`, `y = gamma * xhat`, with eps = 1e-6, so the two products follow the reciprocal on one chain (D roundings for the sum of D squares, then the divide, the addition of eps, the square root, the reciprocal and the two products: D + 6):

| stage | roundings |
|---|---:|
| RMS norm 1 (sum of D squares, divide by D, add eps, sqrt, reciprocal, product with x, product with the gain) | 12 |
| projection to q, k, v (dot of length D) | 6 |
| QK-norm (sum of d_h squares and the same six operations) | 10 |
| rotation (angle, cosine or sine, product, add) | 4 |
| scores (dot of length d_h, scale) | 5 |
| softmax (subtract, exp, sum of T terms, divide) | 11 |
| weighted sum of values (T products added) | 8 |
| W_O (dot of length H d_h) | 16 |
| residual add | 1 |
| RMS norm 2 (as norm 1) | 12 |
| gate or up projection (dot of length D) | 6 |
| silu, product with up (exp, add, divide, multiply) | 4 |
| W_down (dot of length F) | 10 |
| residual add | 1 |
| one layer, total | 106 |
| final RMS norm (as norm 1) | 12 |
| output projection (dot of length D) | 6 |
| storing the golden in 32 bits | 1 |

Two layers, the head and the stored golden give N = 231 roundings, so the bound on the relative error of the logits is N u = 231 x 2^-24 = 1.38e-5, and the random-walk estimate sqrt(N) u is 9.06e-7. Measured: eight draws of the weights of the tiny model (generators `importing_models.draws.0` to `importing_models.draws.7`, same shapes, scales and draw order as Chapter 1, rounded to bf16), each run in 32-bit arithmetic with the three summation orders against the 64-bit logits stored in 32 bits. The 32-bit run rounds to the nearest 32-bit number after every operation: each product and each addition of every sum (the sum of squares of a norm, every dot product, the softmax denominator, the weighted sum of values), each division, square root and reciprocal, each exp, cos and sin (the 64-bit library value rounded once), and the constants eps, the rotary rates and 1/sqrt(d_h) before they are used; the bf16 weights are exactly representable, and the maximum of the softmax is exact. The 'left to right' order adds the terms of a sum one after another from the first; 'right to left' from the last; 'pairwise' adds the sum of the first half to the sum of the second half, recursively, the middle of n terms being n // 2. rel_l2 of the 8 x 11 logits:

| seed of the draw | left to right | right to left | pairwise |
|---|---:|---:|---:|
| 0x6886c84cce118975 | 1.61e-7 | 1.75e-7 | 1.23e-7 |
| 0xd97e1a73b4b25a0f | 2.39e-7 | 1.77e-7 | 2.31e-7 |
| 0x0f513e21f133e079 | 1.70e-7 | 1.54e-7 | 1.24e-7 |
| 0xacb2944e0b3cfc15 | 1.88e-7 | 1.37e-7 | 2.12e-7 |
| 0xe9e1be578625515c | 1.31e-7 | 1.83e-7 | 1.36e-7 |
| 0x0ee3ca3f44cc3174 | 2.66e-7 | 1.84e-7 | 3.16e-7 |
| 0xb54d5dd123aa34f5 | 2.42e-7 | 2.15e-7 | 1.40e-7 |
| 0x99bbac5c15de052b | 1.96e-7 | 1.67e-7 | 2.47e-7 |

The largest measured error is 3.16e-7, 44 times below the bound and 0.35 times the random-walk estimate.
<!-- /worked -->

The bound is $231 \times 2^{-24} = 1.38 \times 10^{-5}$. A second estimate treats the roundings as independent errors of random sign, which add in quadrature, so that $\sqrt{N_{\mathrm{rnd}}}\,u = 9.1 \times 10^{-7}$ is what a typical run should show. The measurements sit under both: 24 runs of 32-bit arithmetic against the golden, on eight independent draws of the weights and with three orders of summation each, give relative errors between $1.2 \times 10^{-7}$ and $3.2 \times 10^{-7}$, 44 times below the bound and a third of the random-walk estimate. The first set of numbers is a theorem under the assumptions, and the second set is evidence that the assumptions held on these models. That is why the bound, and not the random-walk estimate, is the tolerance: it never raises a false alarm unless the stages amplify error, and the measurement is how that is checked.

A tolerance derived this way also says what it cannot see. A difference below $\tau_{\mathrm{par}}$ is indistinguishable from rounding, so a bug whose effect is smaller than the tolerance passes. The wrong epsilon at the scale of the table above is exactly that: its relative error of $3.6 \times 10^{-6}$ is 3.8 times below $1.38 \times 10^{-5}$, so no parity test at this tolerance can see it, on this prompt, with these activations. Three responses are open, and the first two are the ones to use. Compare the configuration itself, field by field with the published values, in a test that is not about outputs, since a value read from a file is a value and can be compared exactly. Run the parity at an input where the quantity that the bug scales is made large, the way the table shrank the activations until the epsilon showed: at a factor of 0.1 on the embeddings, the same wrong epsilon moves the logits by $2.4 \times 10^{-4}$, 18 times the tolerance. And tighten the tolerance only with evidence that it is safe, by measuring the spread of correct implementations as above; do not pick the number from the failing one.

The same derivation settles how to read a mismatch between a golden and a model that has been rounded on the way. The bf16 table earlier changes the logits by $2.9 \times 10^{-3}$, 211 times the tolerance, so a comparison between a model run on the weights as trained and one run on the weights as published fails the parity test, and rightly: they are different functions. The golden and the implementation under test must start from the same bytes.

### Finding the first layer that is wrong

When a whole-model comparison fails, the failure says only that something is wrong. A golden that stores every hidden state turns the failure into a search: compare the implementation's state after each layer with the golden's, and the first one above its tolerance is where the bug is. The tolerance of a tap is the bound for the roundings so far, $\tau_k = (k\, N_{\mathrm{layer}} + 1)\,u$ after $k$ layers (the one for the golden's own storage), so it grows with depth, and a bug is a state above its own bound.

```text
function rel_l2(a: array<real>[N], b: array<real>[N]) -> real
    require N > 0 and norm(b) > 0, "a reference of norm zero"
    return norm(a - b) / norm(b)

function first_divergence(got: list<array<real>>, want: list<array<real>>,
                          bound: array<real>[S]) -> int
    # got[s] and want[s] are the hidden states of stage s, bound[s] the tolerance of stage s
    require len(got) == S and len(want) == S, "one state per stage"
    for s in 0 .. S:
        if rel_l2(got[s], want[s]) > bound[s]:
            return s
    return -1                                  # every stage within its tolerance
```

The search is run on the six-layer model, with one injected error: in layer 3 only, the import exchanges gate and up. The implementation under test is the 32-bit forward pass, so its correct version carries real rounding noise, and the comparison is against the golden of the previous table.

<!-- worked: importing_models.bisection -->
Layer-by-layer comparison for the 6-layer model of the previous table. The implementation under test is the 32-bit forward pass (summation left to right); the golden is the stored 64-bit one. The mistake is an import in which gate and up are exchanged in layer 3 only. For each hidden state, the rel_l2 of the correct import, the rel_l2 of the mistaken one, and the bound (roundings of the layers so far, plus one for the stored golden, times 2^-24). Every layer counts 106 roundings (the table above, with eps 1e-6 and the same 32-bit rules), so the bound after k layers is (k x that + 1) x 2^-24; the final norm adds its own 12 roundings (D + 6), so its bound is (6 x 106 + 12 + 1) x 2^-24.

| hidden state | rel_l2, correct import | rel_l2, mistaken import | bound | mistaken import is |
|---|---:|---:|---:|---|
| embeddings | 0 | 0 | 5.96e-8 | within |
| layer 0 | 1.15e-7 | 1.15e-7 | 6.38e-6 | within |
| layer 1 | 1.57e-7 | 1.57e-7 | 1.27e-5 | within |
| layer 2 | 1.43e-7 | 1.43e-7 | 1.90e-5 | within |
| layer 3 | 1.55e-7 | 1.42e-1 | 2.53e-5 | above |
| layer 4 | 2.29e-7 | 1.35e-1 | 3.17e-5 | above |
| layer 5 | 2.20e-7 | 1.29e-1 | 3.80e-5 | above |
| final norm | 3.23e-7 | 1.50e-1 | 3.87e-5 | above |

The first hidden state above its bound is the output of layer 3.
<!-- /worked -->

![Lines of the base-10 logarithm of the relative error at the seven hidden states after the embedding: the correct import stays between 1e-7 and 2e-7 and always under the dashed bound that grows from 6e-6 to 3.8e-5; the import with gate and up exchanged in layer 3 follows the correct one up to the output of layer 2 and jumps to about 0.14 at the output of layer 3, where it crosses the bound, and stays there.](img/error-by-layer.svg)

The correct import stays between $1.05 \times 10^{-7}$ and $2.09 \times 10^{-7}$ at every state, far under a bound that grows with depth, and the mistaken one is the same to every digit until layer 3, whose output is $1.4 \times 10^{-1}$ off, almost six orders of magnitude above the noise and 5,800 times above its bound. The search finds the first state above its bound, and it names the layer where the mistake is made, and not where the end of the network shows it. A comparison of the logits alone would have said "wrong by 15 percent" and nothing about where. In practice the search is run on the real model, with the golden's taps, and each layer's compare is one subtraction and one norm, so the scan costs next to nothing compared with producing the golden once. Brain's ladder has this shape in three rungs, the one that its test file states: that every tensor of the checkpoint maps onto a parameter and every parameter is present at its size; that the embedding rows are the checkpoint's own, compared exactly (`assert_eq!`, since a table lookup has no arithmetic to differ); and that layer 0 and then the logits agree to the cosine floor. Each rung isolates a failure that the next would only blur, and the exact comparison where nothing is computed is the same rule as the tolerance: demand equality where the arithmetic permits it, and a derived bound where it does not.

## Where brain keeps the table, the plan and the goldens

The pieces of the chapter are spread over three crates, and none of them is a hidden step. `crates/checkpoint` holds the formats (safetensors, the older archives of the PyTorch ecosystem, and the quantized single-file format) and the streaming and remapping machinery: its safetensors module is the reader of the first sections, `remap.rs` is the plan, and `weightio.rs` the by-name reader that streams a tensor on demand so that a load never holds the whole model as a second copy. `crates/arch` is the registry of what a name means, with one rule for each lookup (exact match). `crates/qwen3` is where a family of decoders is imported, and it holds the table, the configuration readers and the parity ladder; other families have import modules and parity tests of their own, as the history above showed for a diffusion transformer and a vision-language decoder. A model's import, from the first byte to the parity ladder, is a pipeline of small steps that each fail by name, and the sections above are the argument that every one of them has to.

## Check your understanding

1. A safetensors file's first eight bytes are `00 00 00 00 40 00 00 00`. Read as a little-endian number, what is $N$, and what does a reader do with a file of 1,000 bytes?
2. A loader reads a matrix of a published model whose header says shape [4096, 4096] and element type BF16, and reshapes the buffer to the model's [4096, 4096] without transposing. Why does no check of shape or size complain, and what is the cheapest test that would?
3. Someone proposes to test an importer with a toy model in which $D = H d_h = h_{\mathrm{kv}} d_h = F$. Which mistakes of this chapter can that model not detect, and what should the toy model look like?
4. A configuration omits the epsilon of its norms, and a reader fills in 1e-5 where the publisher used 1e-6. A parity run on one prompt reports `rel_l2` $= 4 \times 10^{-6}$ at tolerance $10^{-5}$. Is the import correct? What would you change in the test?
5. A parity test asserts `cos_sim` above 0.99999 at every layer. A colleague points out that the final RMS norm's gain was loaded twice as large. Does the test catch it, and why or why not?
6. The tolerance of the tiny decoder, $N_{\mathrm{rnd}} u$, was computed for 8 positions. What changes for 800, and which stage grows?
7. A golden is generated on a GPU in bf16 and the implementation under test runs in 32 bits on a CPU. Which part of the derivation of $\tau_{\mathrm{par}}$ no longer applies, and what is the right way to build the tolerance?
8. A layer-by-layer comparison reports every hidden state within its bound except the last one, the logits. Where would you look first?

## Summary

Importing a published model has four steps, and each is a place where a model that loads can still be the wrong model. The checkpoint format has to be read without trusting it: the length prefix, the offsets and the extents of safetensors are validated before anything is sized by them, the dtype of each tensor is decoded as stored and refused when it cannot be, and a sharded checkpoint is whole only when every shard its index names is there. The names are mapped by a table and a plan that is checked before it runs, for coverage and for shape, and the checks cannot see what is shape-legal: a transposed square matrix, keys exchanged with values, gate exchanged with up, or a head dropped because a default said it was tied, each of which loads and runs, which is why the importer's test model has all its dimensions different and why the model's outputs are the only evidence. The configuration is read without guessing: a default is a guess about the publisher, a missing field that fixes a shape fails loudly and one that changes a value does not, and the cure is a single resolver that refuses what it cannot interpret, round-trip tests of the writer, and comparisons of configuration values themselves. And parity is proved against a stored reference, with three measures that see different things, a tolerance derived from the rounding model (the count of roundings on the longest chain of dependent operations times $u$, here $1.38 \times 10^{-5}$ for the tiny decoder, with the measured errors 46 times below it), goldens that hold every hidden state so that a failure is a search for the first state above its bound, and an honest account of what the tolerance cannot see.

The limits are those of the model of error. The bound assumes that stages do not amplify relative error and the measurements only support it on these models; a network with an unnormalized stage of high condition number would need its own factor. The tolerance is silent about bugs smaller than rounding, and for those the test has to be about values, not outputs. And parity with a reference proves that the engine computes the reference's function, not that the function is good. The chapters that follow lean on the result: the variants of attention, the experts and the recurrent layers of Volume VI are met as checkpoints, and a variant that has not been compared with its reference through a tolerance of this kind is a variant that may be quietly wrong. The next chapter turns to the first of them, the keys and values themselves, and how they can be shared or compressed to shrink the cache.

## References

- The safetensors format specification, published with the library of that name (Hugging Face), describing the 8-byte length prefix, the JSON header with `dtype`, `shape` and `data_offsets`, and the raw data that follows.

## Exercises

1. **[Calculate]** A file holds one F32 tensor named `weight` of shape [3, 5] and nothing else, with the metadata entry `{"format":"pt"}`. Write its header (the metadata entry first, then the tensor, keys in the order dtype, shape, data_offsets, no spaces, padded with spaces to a multiple of 8) and give the value of $N$, the offset where the data starts, and the total file size.
2. **[Calculate]** The bf16 pattern of a weight is 0xBF35. Decode it to a 32-bit pattern and to a decimal number. What is the relative error bound of rounding a number to bf16, and what is the largest absolute error for a weight of size 0.02?
3. **[Calculate]** A model of 7 billion parameters is stored in bf16 and sharded at a budget of 5 GB (decimal) per shard, greedily, with all tensors much smaller than a shard. How many shards are there, and how many bytes does the index's `total_size` hold?
4. **[Calculate]** A fused matrix has shape [3 H $d_h$, $D$] with queries, keys and values in that order and equal widths ($h_{\mathrm{kv}} = H$). With $H = 4$, $d_h = 8$, $D = 32$, give the row ranges of the three parts. What are they, in the same fused matrix, if there are 2 key-value heads?
5. **[Calculate]** For a decoder of $L = 24$ layers with $D = 1{,}024$, $H d_h = 2{,}048$, $F = 3{,}072$, $d_h = 128$ and $T = 512$ positions, use the stage counts of the chapter's table (a dot product of length $n$ costs $n$; an RMS norm costs $D + 6$; the QK-norm costs $d_h + 6$; the rotation 4; the scores $d_h + 1$; the softmax $T + 3$; the weighted sum $T$; the activation of the feed-forward network 4; each residual 1) to compute $N_{\mathrm{rnd}}$ of the layer stack, the bound $N_{\mathrm{rnd}} u$, and the random-walk estimate. Which stage is the largest?
6. **[Calculate]** A relative error $\rho$ orthogonal to the reference gives $\cos = 1/\sqrt{1 + \rho^2}$. For which $\rho$ is $1 - \cos = 10^{-4}$? For which is it $10^{-7}$, the resolution of a 32-bit cosine?
7. **[Implement]** Reproduce `importing_models.safetensors_file` and `importing_models.dtype_decoding`: write the three-tensor file byte by byte (header text, padding to a multiple of 8, data in the order given), print its bytes, parse it back with the two checks of the chapter, and decode the 16-bit patterns with your own conversion, to the digits printed.
8. **[Implement]** Reproduce `importing_models.name_mapping` and `importing_models.import_mistakes`: build the published layout of the tiny decoder from the seed and draw order of Chapter 1 (rounded to bf16), write the plan, check it before it runs, and then make each mistake of the table and measure `max_abs`, `rel_l2` and cosine of the logits against the golden.
9. **[Implement]** Reproduce `importing_models.error_budget`: run the decoder in 32-bit arithmetic (round after every operation) with the three orders of summation, compare with the 64-bit logits stored in 32 bits on the eight draws named in the chapter, and compute the bound $N_{\mathrm{rnd}} u$.
10. **[Implement]** Reproduce `importing_models.golden_contents` and `importing_models.bisection`: write the golden of the six-layer model, import it with gate and up exchanged in layer 3, and find the first hidden state above its bound with the search of the chapter.
11. **[Investigate]** Add a bias to the fused query-key-value projection of the tiny decoder and publish it in the layout. Make the importer ignore it, as commit `0581928e9` describes. Which of the three measures see it at the tolerance of the chapter, and does a synthetic gradient check on weights with a zero bias see it?
12. **[Investigate]** The chapter's bound assumes stages that do not amplify relative error. Build a decoder whose first norm has a very small gain on one channel and a very large one on another, run the three orders of summation again, and measure how far the largest observed `rel_l2` is from $N_{\mathrm{rnd}} u$. What does the result say about the assumption?

## Key terms

- **checkpoint format**: the layout of a trained model in a file: which names, shapes and number formats it holds, and where the bytes of each tensor are, so that a reader that has never seen the file can find and decode them. <!-- concept: checkpoint-format -->
- **safetensors**: a checkpoint format of an unsigned 64-bit little-endian length, that many bytes of JSON mapping each tensor name to its element type, shape and byte offsets, and then the raw bytes of every tensor, back to back. <!-- concept: safetensors -->
- **weight-name mapping**: the table and rules that turn the tensor names, orientations and groupings of a published layout into those of an implementation: renames, cuts of fused matrices and transposes, checked before they run for coverage and shape. <!-- concept: weight-name-mapping -->
- **reference golden**: the stored output of a reference implementation for fixed inputs, with the intermediate activations of every layer, against which an implementation is compared, never committed and always regenerated by a recorded script. <!-- concept: reference-golden -->
- **parity tolerance**: the largest difference between two implementations of the same function that rounding alone can explain, derived from the count of roundings along the longest chain of dependent operations times the unit round-off, so that anything above it is evidence of a bug. <!-- concept: parity-tolerance -->
- **config default**: the value a reader substitutes for a field a configuration omits, which is a guess about the publisher: loud when the field fixes a shape and silent when it only changes a value. <!-- concept: config-defaults -->

## Answers

**Recall questions.**

1. $u = 2^{-24} \approx 5.96 \times 10^{-8}$. Added left to right, the error of a sum of $n$ numbers is at most about $(n-1)u \sum_i \lvert x_i \rvert$, relative to the sum of the absolute values of the terms and not to the sum.
2. bf16 has 8 exponent and 7 fraction bits and fp16 has 5 and 10. A bf16 pattern becomes the 32-bit pattern of the same number by shifting it left by 16 bits (the low 16 bits are zero); an fp16 pattern needs a new exponent bias.
3. Position $1 \times 3 + 0 = 3$ (0-indexed). Its transpose has shape 3 by 2.
4. Nothing: the cosine similarity is unchanged by a positive scale on either vector, which is why it cannot see a dropped scale factor.
5. With equal dimensions a swap or a transpose of two matrices passes, because the shapes agree; all dimensions different make the shape check catch it. A relative error says how large a difference is compared with the size of the quantity, so one tolerance reads the same for large and small values, where an absolute error does not.
6. $\mathbf{W}_Q$ is $6 \times 16$ and $\mathbf{W}_K$ is $6 \times 8$ ($H d_h = 16$, $h_{\mathrm{kv}} d_h = 8$). A tied head has no tensor of its own and uses the embedding table, an untied one has a separate $[V, D]$ matrix.

**Check your understanding.**

1. Byte 4 is 0x40 and the others are zero, and byte $k$ has weight $256^k$, so $N = 64 \times 256^4 = 2^{38}$. A file of 1,000 bytes cannot hold a header of that length, so the first check refuses it and nothing is allocated.
2. A matrix of 4096 by 4096 is square: the stored shape and the model's are the same, and the number of elements is the same, so no check sees a difference between the matrix and its transpose. The cheapest test is a run of the outputs against a golden, on a model whose matrices are not symmetric (a random matrix is not), so that its transpose is a different function.
3. A toy with all those dimensions equal cannot detect a transposed matrix, the exchange of keys and values (equal widths), or a swap of the head count with the head width. The toy should have every dimension different ($D$, $H d_h$, $h_{\mathrm{kv}} d_h$, $F$, $V$, $d_h$ and the text length), as the tiny decoder of Chapter 1 does.
4. Not necessarily. The error is 2.5 times below the tolerance, so a wrong epsilon at the scale of that prompt passes whether or not it is wrong. Add a test that compares the configuration's values with the published ones, and a run at a smaller activation scale where the epsilon is visible.
5. Likely not. A doubled gain at the end scales the output by 2 and leaves the direction unchanged, so the cosine is 1 and passes; `rel_l2` of 1.0 would fail it. The test needs a `rel_l2` (or `max_abs`) bound beside the cosine.
6. $T$ appears in the softmax ($T + 3$) and the weighted sum of values ($T$), so $N_{\mathrm{layer}}$ grows by $2 \times 792 = 1{,}584$ per layer, to $106 + 1{,}584 = 1{,}690$, and the bound grows about 16-fold. The two stages that read the whole context dominate at long contexts.
7. The derivation assumed 32-bit rounding of every operation of the implementation under test; with a bf16 golden, the reference itself carries rounding of $2^{-8}$ per operation, a factor of $2^{16}$ larger, and the tolerance must be built from the roundings of both implementations, or the golden must be produced in 32 bits.
8. The head: the final norm and the output projection were within bound at the preceding state, so the difference is in the final RMS norm's gain, in the output matrix (tied against untied, or transposed), or in the logit computation, which is where a dropped head or a wrong tie shows.

**Exercises.**

1. The header is `{"__metadata__":{"format":"pt"},"weight":{"dtype":"F32","shape":[3,5],"data_offsets":[0,60]}}`, 93 characters, padded with 3 spaces to 96, so $N = 96$, the data starts at byte $8 + 96 = 104$ and the file is $104 + 60 = 164$ bytes.
2. 0xBF35 shifted left by 16 is the 32-bit pattern 0xBF350000, the number $-(1 + 53/128) \times 2^{-1} = -0.70703125$. The relative error of rounding to bf16 is at most $2^{-8} \approx 3.9 \times 10^{-3}$, and for a weight of size 0.02 the largest absolute error is at most $0.02 \times 2^{-8} = 7.8 \times 10^{-5}$.
3. $14 \times 10^9$ bytes at $5 \times 10^9$ per shard is 3 shards (two full ones and a third with $4 \times 10^9$), and `total_size` is $14 \times 10^9$ bytes.
4. With $H = 4$, $d_h = 8$ and equal widths each part is $4 \times 8 = 32$ rows: queries rows 0 .. 31, keys 32 .. 63, values 64 .. 95. With 2 key-value heads the key and value parts are 16 rows each: queries 0 .. 31, keys 32 .. 47, values 48 .. 63.
5. Per layer: norm 1, $1{,}024 + 6 = 1{,}030$; q, k, v projection 1,024; QK-norm 134; rotation 4; scores 129; softmax 515; weighted sum 512; $\mathbf{W}_O$ 2,048; residual 1; norm 2, 1,030; gate or up 1,024; activation 4; $\mathbf{W}_{\mathrm{down}}$ 3,072; residual 1, in all 10,528 roundings. Times 24 layers is 252,672, so the bound is $252{,}672 \times 2^{-24} \approx 1.5 \times 10^{-2}$ (before the head and the golden) and the random-walk estimate is $\sqrt{252{,}672} \times 2^{-24} \approx 3.0 \times 10^{-5}$. The largest stage is $\mathbf{W}_{\mathrm{down}}$, a dot product of length 3,072.
6. $\rho = \sqrt{2 \times 10^{-4}} = 1.4 \times 10^{-2}$, 1.4 percent, and $\rho = \sqrt{2 \times 10^{-7}} = 4.5 \times 10^{-4}$.

**Implement and Investigate.** The Implement exercises must reproduce the printed tables to their digits; the Investigate exercises have no printed answer. In Exercise 11 the missing bias moves the outputs for the same reason as in the history, and a gradient check on weights with a zero bias passes because the model is compared with itself.
