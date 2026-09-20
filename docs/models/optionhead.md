# Option head

A decision model that reads a situation as text, reads a list of things that
could be done right now as text, and returns a probability for each. The list
arrives at **run time** and may be different at every decision - the model has
no fixed action space and no vocabulary of actions it was built around.

It is a ranker, not a generator. It never writes an action; it scores the ones
it is handed. Everything trainable is a 445k-parameter head over a frozen
sentence encoder, which is what makes the whole model 1.7 MB.

`swedishembedded/minilm-l6-option-head` is the architecture.
`swedishembedded/minilm-l6-option-head-<task>` is a trained one.

## Support

| Capability | Supported |
|---|---|
| Inference             | [x] |
| Training from scratch | [x] |
| Fine-tuning a head    | [x] |
| INT8                  | [ ] |

## Getting the weights

Two artifacts, because a head is an adapter:

```bash
brain pull sentence-transformers/all-MiniLM-L6-v2     # the frozen encoder
```

The head comes from training one, or from a published checkpoint under
`swedishembedded/`. A head names its base in its own metadata, so a file that
turns up on its own says what it attaches to.

## How the input is laid out

Situation and options go into **one packed sequence**, tagged by segment:

```
rows 0..R      SEG_STATE   the situation, in windows of `max_span` tokens
                           with `overlap` tokens of carry between them
rows R..R+S    SEG_SLOT    one span per option, each of the form
                           "<instructions> [SEP] <option text>"
```

The instructions are prepended to **every** option, so the same situation
under different orders produces different scores. Each option span begins with
`[CLS]`, and the packer records which row that is.

## The head

```
q   = hidden[cls_rows] @ Wq  + bq      [S, 384]   one query per option
kv  = hidden[0..R]     @ Wkv + bkv     [R, 768]   K and V from the situation
ctx = crossattn(q -> kv)               [S, 384]   options attend over the state
out = LayerNorm(ctx + q)               [S, 384]
z_i = out_i . w + b                    [S]        one scalar per option
p   = softmax(z)                                  on the HOST
```

Cross-attention rather than a dot product of two embeddings is what makes this
a cross-encoder: an option's score depends on the whole situation, not on a
similarity between two independently computed vectors.

**The softmax is not on the device.** `z` is the last device-side value; the
grouping into questions, the normalisation, the loss and its gradient are host
arithmetic over a few hundred numbers. That is what keeps the option space
genuinely runtime-defined - no kernel has an opinion about how many options a
question has - at a cost that does not register beside a six-layer encoder.

**Slots are read at `[CLS]`, not mean-pooled.** A gather has an existing
adjoint; a mean over variable-length spans would need a new kernel pair.
`[CLS]` is the position BERT trains for exactly this.

## Tensors

```
head.wq.weight    [384, 384]     head.wkv.weight   [768, 384]
head.wq.bias      [384]          head.wkv.bias     [768]
head.ln.weight    [384]          head.ln.bias      [384]
head.score.weight [1, 384]       head.score.bias   [1]
```

444,673 parameters, F32, against the encoder's 22M which are frozen.

## Question types

| Type | Slots | Returns |
|---|---|---|
| `Choice` | one per option | a distribution over the options |
| `Score`  | one per level  | a position along ordered levels |
| `Noul`   | ONE            | the probability a proposition holds |

`Noul` has one slot rather than two because two slots make the state signal
common-mode and a softmax cancels it.

## What a decision costs

Measured on a Tesla P40, one call:

| state (words) | options | ms/call |
|---|---|---|
| 4   | 1  | 7.2  |
| 4   | 8  | 12.9 |
| 4   | 24 | 21.0 |
| 40  | 8  | 14.8 |
| 40  | 24 | 25.7 |
| 200 | 8  | 34.4 |

About 0.55 ms per extra option on a ~7 ms fixed cost. `cap_rows` and
`cap_slots` bound the packed rows and the option count; a request that exceeds
either is refused rather than truncated.

## Training

Two phases, and they optimise different objectives:

- **Behaviour cloning** - cross-entropy against a teacher's chosen index.
- **PPO** - a clipped surrogate on the same softmax, with GAE advantages of a
  scalar reward the environment supplies.

`ControlSpec` exposes `target_kl` (a trust region that binds per minibatch),
`anchor` (divergence from the policy the warm start produced), `gauge_episodes`
(scoring every iteration on the SAME worlds) and `average` (the mean of the
last N iterates as a second candidate).

## Limits worth knowing

**No pairwise comparison.** An option's score depends only on that option and
the situation; options are coupled solely by the softmax that normalises them.
Any preference between two options has to be inferable from the situation text
- the model cannot attend from one option to another.

**The head is meaningless alone.** 1.7 MB that does nothing without the exact
encoder it was trained against and an environment that supplies option strings
in the format it saw.

## Samples

| Sample | What it decides |
|---|---|
| `decision/intents` | does the model actually READ its options |
| `decision/triage`  | BANKING77 messages against supplied intents |
| `decision/arena`   | a corridor shooter, options change every tick |
| `decision/doom`    | DOOM, from the game's own state |
