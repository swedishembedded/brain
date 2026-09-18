<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# arena - learning to play, when the moves change every tick

A decision model learns a corridor shooter by **reinforcement**. Every tick it
is handed the situation and a list of things it could do right now - and the
list is different almost every tick.

```
 1 [@.+.D...Di..] hp 100 ammo 6  -> shoot the demon at range 4 (87%)
 4 [@D+i.D......] hp 100 ammo 3  -> shoot the imp at range 3 (75%)
 5 [@i+.D.......] hp  93 ammo 2  -> shoot the imp at range 1 (99%)
 7 [@DD.........] hp  79 ammo 0  -> grab the medkit at range 2 (51%)
 8 [.D@.........] hp  86 ammo 0  -> reload (90%)
```

Nothing is generated. The model reads the options as text and returns a
probability over them, and the whole loop runs on the machine in front of you.

```rust
ControlPipeline::from_pretrained(encoder, Arena)
    .train(spec)     // clone a scripted teacher, then PPO
    .evaluate()      // greedy episodes on seeds training never saw
    .save(policy)
    .play(3)         // watch it play, decision by decision
    .report()
    .finish()?
```

---

## Contents

1. [Why this sample exists](#1-why-this-sample-exists)
2. [Prerequisites](#2-prerequisites)
3. [Run it](#3-run-it)
4. [Reading the output](#4-reading-the-output)
5. [Using a trained policy](#5-using-a-trained-policy)
6. [How it works](#6-how-it-works)
7. [Results](#7-results)
8. [Command reference](#8-command-reference)

---

## 1. Why this sample exists

**A normal policy network cannot play this game.** Its output layer's width is
the action space, fixed when the weights are created, and every action is an
index whose meaning the network has to learn from scratch. This game does not
have a fixed action space:

* you can only shoot a monster that is **alive and in range** - so the number
  of shoot options changes as monsters die and close distance
* you can only **grab the medkit** if it is still on the floor and you are hurt
* you can only **reload** with reserve ammo left
* you can only **retreat** if you are not against the wall

So the legal set is rebuilt every tick, and each option is a sentence:
`"shoot the demon at range 3"`. The model reads what an option *means*. Add a
new monster type tomorrow and it is a new string, not a new output layer and a
retrain.

This is also the setting where the reinforcement learning is **genuine**: the
action decides which state the next decision is made from, so the policy shifts
its own data distribution. (Contrast `samples/decision/salesagent`, where
conversations are replayed and the same PPO machinery provably degenerates into
a scoring rule.)

---

## 2. Prerequisites

One thing, fetched once:

```bash
brain pull sentence-transformers/all-MiniLM-L6-v2
```

~90 MB, lands in `~/.local/share/brain/models/sentence-transformers/all-MiniLM-L6-v2`.
Override with `--encoder DIR` or `BRAIN_MINILM_DIR`.

**No dataset.** The agent generates its own data by playing. The game is plain
Rust in `src/arena.rs` with no dependencies at all.

Runs on the default device; `BRAIN_DEVICE=cpu` works and is slow.

---

## 3. Run it

```bash
make samples/decision/arena/build
make samples/decision/arena/run
```

Clones a scripted teacher, trains by PPO, evaluates on held-out seeds, saves
the policy, plays three episodes so you can watch, then waits for you to press
enter for more.

The defaults are sized to finish quickly, not to reach the numbers in
[section 7](#7-results). For those:

```bash
make samples/decision/arena/run ARGS="--iterations 100 --episodes 48 --epochs 2 --batch"
```

about 30 minutes on one Tesla P40. A quick look at the loop first:

```bash
make samples/decision/arena/run ARGS="--iterations 4 --episodes 8 --batch"
```

---

## 4. Reading the output

### 4.1 The reference band

```
arena: reference band on the evaluation seeds
         random                  11.0%  return -2.862
         teacher (always shoot)  39.0%  return -0.496   <- what cloning starts from
         best hand-written       78.0%  return +2.488   <- the ceiling worth chasing
```

Printed before anything trains, measured on **exactly the seeds `evaluate`
uses**. They are the only reason a learned win rate is readable. The ceiling
matters most: a threshold set above what any policy can reach is not a
demanding criterion, it is a broken one, and an earlier version of this sample
had exactly that.

### 4.2 Training

```
    warm start: 60 scripted episodes x 12 passes, final loss 0.0970
    iter   1  return +0.14  wins  23/48   steps   623  critic mse 3.956
    iter  71  return +2.73  wins  41/48   steps   576  critic mse 0.484
```

Per iteration: mean episode return under the **current** policy, episodes won,
decisions collected, and how well the critic predicts the return.

`return` is the honest learning curve - measured while collecting, exploration
included, so it sits below the greedy evaluation number. **`critic mse` is the
one to watch alongside it**: the advantages are only meaningful to the extent
the critic can predict the return, and a critic that is not converging means
the policy is being updated on noise.

The PPO surrogate loss printed by the step logger is **not** a progress signal
and wanders around zero. It measures how far an update wanted to move the
policy, not how good the policy is.

### 4.3 Play

The corridor: `@` is you, `i` an imp, `D` a demon, `+` a medkit. Each step
prints the option the policy chose and how much probability it put on it.
`play` is greedy - what a deployed agent would do; training samples instead, to
explore.

---

## 5. Using a trained policy

`--save` writes the **head**; the encoder is imported from the released
checkpoint and costs nothing to re-import.

```bash
make samples/decision/arena/run ARGS="--head out/arena-policy.safetensors --play 5"
```

With `--head` the chain skips training and goes straight to evaluating and
playing.

From your own code, any environment works - implement three methods:

```rust
use brain::{ControlPipeline, ControlSpec, Env};

impl Env for MyEnv {
    fn reset(&mut self, seed: u64) -> String { /* first observation */ }
    fn actions(&mut self) -> Vec<String>     { /* what is legal RIGHT NOW */ }
    fn step(&mut self, action: usize) -> (String, f32, bool) { /* obs, reward, done */ }

    // Optional but worth supplying: a scripted action to clone before PPO
    // starts. See 6.3.
    fn demo(&mut self) -> Option<usize> { None }
}

ControlPipeline::from_pretrained(encoder, MyEnv::new())
    .train(ControlSpec::default())
    .play(3)
    .finish()?;
```

`actions` is consulted every step and may return a different set each time.
Nothing is indexed by a global action id, so an environment may invent an
action mid-episode and the policy will read it.

---

## 6. How it works

### 6.1 The loop

```text
observation (text) ──┐
                     ├─► encoder ─► head ─► score per option ─► softmax ─► sample
legal actions (text)─┘                                                       │
        ▲                                                                    │
        └──────────────────── environment steps ◄──────────────────────────────┘
```

The observation is terse - `hp 78 ammo 4 reserve 6 | demon hp 5 range 1 |
medkit range 3` - because observation length is what a control loop pays per
tick.

**The encoder is frozen; only the head trains.** Two reasons, and a measured
one: a reinforcement signal is far noisier than a labelled one, and a few
hundred high-variance gradients per iteration are not enough to move 22M
pretrained parameters usefully but are plenty to destroy the language
understanding that made the option text readable. Measured on an earlier
version: fine-tuning the encoder collapsed a working policy to 13% while
freezing it held 56.5%. Freezing is also ~2x faster, because the encoder's
reverse pass is skipped entirely.

### 6.2 PPO, with the parts that make it PPO

Per iteration:

1. **Collect** full episodes under the current policy, sampling actions.
2. **Credit** each step by generalized advantage estimation against a learned
   **critic** - `A_t = sum (gamma*lambda)^k delta_{t+k}`, with a truncated
   episode bootstrapping from `V(s_last)` rather than being told its future was
   worth zero.
3. **Update** with the clipped surrogate plus an entropy bonus, in
   **minibatches of 64**, for several shuffled passes, with advantages
   normalized per minibatch and the learning rate annealed linearly to zero.

Every item in step 3 is load-bearing and each was got wrong first:

* **Minibatches.** An earlier version took one Adam step per transition. A
  single transition's advantage is an extremely noisy gradient estimate and
  Adam applied to it chases the noise; reference PPO splits a rollout into a
  handful of minibatches.
* **The critic.** An earlier version used the batch mean as a baseline. That is
  a single number for every state in the batch, so in an episode that is won at
  step 1 and lost at step 8 both are compared against the same constant and the
  advantage is dominated by *which episode this was*. An implementation without
  a value function is REINFORCE-with-baseline wearing PPO's clipped ratio.
* **Shuffling.** Consecutive steps of an episode are correlated; a sequential
  pass walks the policy along one trajectory instead of averaging.

### 6.3 Clone first, then improve

Training runs a **behaviour cloning** phase before PPO: play episodes under
`Env::demo` and fit the policy to what the teacher did. This is step one of the
recipe the method's own paper describes, and skipping it is expensive - a
policy gradient from a random start has to discover a good action by sampling
it, which over a text action space is slow enough to look like a plateau.

The teacher here is deliberately **weak**: `shoot anything in range, else
reload`, which scores 39%. Cloning it is not the result. The result is that PPO
then reaches 79%, which is what [section 7](#7-results) measures.

Cloning has to actually **converge**. At 600 demonstrations and a minibatch of
64, one pass is nine optimizer steps; the default is 12 passes. An
under-converged warm start produced 38.5% on one initialisation and 1.0% on
another, and neither number means anything.

### 6.4 Why reward-maximisation is right here

Its sibling sample, `salesagent`, must **not** maximise a hit rate - a
probability estimate graded by "was it right" converges to the *mode*, not the
rate, and comes out confidently wrong. Here the opposite holds: a control
policy should converge onto the best available **action**. Committing is the
goal, and entropy is what stops it committing before it has explored.

Both objectives live in `decide::policy` with the distinction written down -
`the_policy_optimum_is_the_conditional_rate` and
`the_control_optimum_is_the_best_action` are the two tests that pin it.

### 6.5 The game

A 12-cell corridor, 2-3 monsters, one optional medkit, 12 shots.

| | hp | damage | speed |
|---|---:|---:|---:|
| imp | 1 | 18 | 2 |
| demon | 5 | 7 | 1 |

The imp is a **glass cannon** and the demon a **slow grinder**, and that
asymmetry is the whole design. An earlier version had a tanky demon that hit
hard and a weak imp that hit softly, which sounds like it demands target
priority and does not: the right answer is always "shoot whatever dies soonest",
which is what firing at the first thing in the list already does. Seven
hand-written policies were measured on it and the one that **ignored the
observation won**. That game could not demonstrate learning by anything, so it
was replaced.

Shot accuracy falls 15% per cell of range, and ammo is scarce, so a long shot
is a real mistake. Monsters advance every tick and bite anything adjacent.
Rewards: `+0.6` an imp, `+1.2` a demon, `+2.0` for clearing the corridor,
`-1.5` for dying, small penalties for damage taken and for dithering.

---

## 7. Results

Success criteria were frozen **before** these runs, in
[`CRITERIA.md`](CRITERIA.md), including an erratum recorded rather than edited
in when one of its baselines turned out to be wrong.

100 iterations x 48 episodes x 2 PPO passes, ~60,000 environment steps,
~29 minutes on one Tesla P40, scored on 200 held-out episodes:

| | win rate | return |
|---|---:|---:|
| random | 11.0% | −2.86 |
| untrained head (median of 8 init seeds) | ~0% | - |
| behaviour cloning alone | 44.0% | −0.13 |
| best hand-written policy | 78.0% | +2.49 |
| **cloning + PPO, head seed 1** | **79.0%** | **+2.47** |
| **cloning + PPO, head seed 2** | **78.0%** | **+2.50** |

Two initialisations, one point apart, 61,418 and 58,106 environment steps. Run
it yourself with `--head-seed N`; the numbers below are from seed 1.

| # | criterion | threshold | result |
|---|---|---|---|
| A1-A2 | varying action set, options read as text | tests | **MET** |
| A3 | realtime | < 20 ms | **MET** - 8.2 ms/decision |
| A4 | environment distinguishes policies | 35-point ladder | **MET** |
| L1 | beats the teacher it cloned | > 46% | **MET** |
| L2 | above untrained and random | > 20% | **MET** |
| L3 | beats the teacher on return | > −0.50 | **MET** |
| L4 | no collapse | curve | **MET** |
| **L5** | halfway from teacher to ceiling | **> 58%** | **MET** |

**L5 is the criterion that counts.** L1 was set against a baseline that later
turned out to be mismeasured, so it no longer separates the policy gradient
from cloning; L5 is 14 points clear of converged cloning and was frozen before
any of this was known. PPO added **+35 points over cloning** and edged past the
best policy the author could hand-write.

The learning curve is monotone - rollout wins 23 → 36 → 38 → 41 of 48 - and the
critic's mean squared error falls from 3.96 to 0.28, which is what says the
advantages meant something.

**Caveats.** Two runs on one machine, not a sweep. The untrained baseline is a
*median over 8 initialisations* because one seed scored 74% where the rest
scored 0-5% - quoting a single seed for any of these numbers would be a
mistake, and it nearly was one. The reference ladder and the frozen criteria
are in [`CRITERIA.md`](CRITERIA.md) along with an erratum recorded when one of
its baselines turned out to have been measured wrong.

---

## 8. Command reference

```text
arena [--encoder DIR] [--head FILE] [--save FILE]
      [--iterations N] [--episodes N] [--epochs N] [--warmup N]
      [--play N] [--entropy F] [--head-seed N] [--train-encoder] [--batch]
```

| flag | default | meaning |
|---|---|---|
| `--encoder DIR` | the model store path | encoder checkpoint. Also `BRAIN_MINILM_DIR` |
| `--head FILE` | none | load a trained policy and **skip training** |
| `--save FILE` | `out/arena-policy.safetensors` | where to write the policy head |
| `--iterations N` | 12 | rollout-then-update cycles |
| `--episodes N` | 24 | episodes collected per iteration |
| `--epochs N` | 2 | PPO passes over each collected batch |
| `--warmup N` | 60 | episodes of scripted play to clone first; 0 skips |
| `--play N` | 3 | episodes to play out, printed |
| `--entropy F` | 0.02 | entropy bonus |
| `--head-seed N` | 1 | policy head initialisation |
| `--train-encoder` | off | fine-tune the encoder too. Measured to collapse a working policy |
| `--batch` | off | skip the interactive stage |

## Cost

One brain dependency (`brain`, feature `decision`). The game adds nothing: it
is plain Rust with no dependencies. `make check/samples` enforces the budget.

---

Swedish Embedded AB builds realtime control policies that run on the customer's
own hardware - deciding among actions a system defines at run time, in
milliseconds, with no text generator in the loop. If your team needs judgment
inside a control loop, you can procure our services by sending an email to
info@swedishembedded.com.
