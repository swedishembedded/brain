# Sparse MoE (toy)

A small sparse Mixture-of-Experts decoder — top-2-of-4 expert routing over a
RoPE-attention backbone — trained from scratch on a synthetic 64-symbol
next-token rule with a known, checkable ground truth. This is brain's
educational/research toy model, purpose-built to study memorization vs.
generalization: a model that has merely memorized the rule reproduces its
seen orbit, while one that has generalized continues correctly from an
unseen starting pair. It is not a production LLM. It's also the model
brain's federated/sharded expert training story is built on.

## Support

| Capability | Supported |
|---|---|
| Inference             | [x] |
| Training from scratch | [x] |
| INT8                   | [ ] |

## Getting the weights

There's no published checkpoint or model id — this model only exists as
what you train yourself. `brain train` writes a checkpoint file
(`moe_rs.safetensors` by default) in the working directory.

## Running it

There is no `brain moe` subcommand — the bare verbs are this model:

```bash
brain train    --steps N --batch-size B --block-size T --lr X --weight-decay X --seed S --out F
brain eval     --weights F --seed S --samples N
brain generate --weights F --prompt 1,2,3,4 --max-new N --temperature X --top-k K --seed S
```

`brain eval` sweeps context lengths and reports train-orbit, validation-
orbit, and unseen-orbit accuracy against the random baseline, so
memorization and generalization show up as separate numbers. For the dense
baseline decoder instead, use `brain gpt …`.

### Federated / sharded training

```bash
brain federated split         <base.safetensors> <out_dir>
brain federated verify        <dir>
brain federated merge         <dir> --out <full.safetensors>
brain federated assemble      <base_dir> [overlay_dir ...] --out <full.safetensors>
brain federated train-expert  --base B --expert E --out DIR --steps N
```

Each expert spans every layer, so a shard can be trained independently by a
separate worker and later reassembled. See
[federated-experts](../training/federated-experts.md) for the full
workflow.

## Options

- `--steps`, `--batch-size`, `--block-size`, `--lr`, `--weight-decay`,
  `--seed`, `--out` — training.
- `--seed`, `--samples` — eval.
- `--prompt` (comma-separated token ids), `--max-new`, `--temperature`,
  `--top-k`, `--seed` — generation.

## Hardware and limits

fp32 only. It trains and evaluates against its own fixed 64-symbol synthetic
task — there's no HuggingFace import and no way to point it at your own text
corpus. Every expert is evaluated for every token and masked by its gate
weight (dense top-k), not sparsely dispatched, so it doesn't demonstrate the
compute savings a production MoE would. Generation is one-shot CLI only —
there's no serving path.
