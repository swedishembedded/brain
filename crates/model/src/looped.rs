// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements looped and recurrent-depth decoders for its clients. If your team needs expertise in
// models that run some of their layers more than once per token then you can procure our services by sending an
// email to info@swedishembedded.com.

//! The visit plan of a looped decoder: which physical layer runs at each logical depth.
//!
//! A looped decoder stores `prelude + core + coda` layers and runs the core `passes` times per token:
//!
//! ```text
//! prelude 0..p  |  core p..p+c  (pass 0)  |  core (pass 1)  | ..  |  coda p+c..p+c+d
//! ```
//!
//! Every run of a layer is a [`Visit`]: its logical index (the order the forward runs them, the reverse of the
//! backward's), the physical layer whose weights it reads, its segment and pass, the attention it runs (full, or a
//! causal window) and its state slot. The slot is where the visit's own per-sequence state lives in serving (its KV
//! history, convolution rows): one slot per visit by default, or, with [`VisitPlan::share_core_slots`], core visits
//! cycling over fewer slots (the recurrent-depth models' shared-history KV strategy). Training reads the same plan:
//! the residual stream has [`VisitPlan::boundaries`] boundaries (visit `i` reads boundary `i` and writes `i + 1`), a
//! layer streamed through a device window is loaded at each of its visits, and its gradient is carried across them
//! when the plan [`VisitPlan::revisits`] a layer (`model::streamed::RevisitingPass`).
//!
//! A whole-stack loop (every layer looped) is a plan with no prelude and no coda; a prelude/core/coda model is the
//! general case. What a model does between passes (re-injecting an input, a norm and a readout per pass, messages
//! from the previous pass) is the model's: the plan says where a pass starts and ends ([`VisitPlan::pass`]).

use std::ops::Range;

/// The attention a visit runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Attention {
    /// Causal attention over the whole history.
    Full,
    /// Causal attention over the last `n` positions, the current one included.
    Window(u32),
}

/// Where in the stack a visit runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Segment {
    Prelude,
    Core,
    Coda,
}

/// One run of a physical layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Visit {
    /// Logical index: the position in the forward's order.
    pub index: usize,
    /// The physical layer whose weights the visit reads.
    pub layer: usize,
    pub segment: Segment,
    /// The core pass (0-based); 0 for prelude and coda visits.
    pub pass: usize,
    pub attention: Attention,
    /// The per-sequence state slot the visit reads and writes in serving.
    pub slot: usize,
}

/// The stored layers of a looped decoder and how often its core runs: `prelude` layers, then `core` layers run
/// `passes` times, then `coda` layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LoopShape {
    pub prelude: usize,
    pub core: usize,
    pub coda: usize,
    pub passes: usize,
}

impl LoopShape {
    /// Every one of `layers` looped `passes` times (no prelude, no coda).
    pub fn whole_stack(layers: usize, passes: usize) -> LoopShape {
        LoopShape { prelude: 0, core: layers, coda: 0, passes }
    }

    /// Stored (physical) layers.
    pub fn layers(&self) -> usize {
        self.prelude + self.core + self.coda
    }

    /// Logical visits: every stored layer once, the core `passes` times.
    pub fn visits(&self) -> usize {
        self.prelude + self.core * self.passes + self.coda
    }

    /// The physical layers of `segment`.
    pub fn layers_of(&self, segment: Segment) -> Range<usize> {
        match segment {
            Segment::Prelude => 0..self.prelude,
            Segment::Core => self.prelude..self.prelude + self.core,
            Segment::Coda => self.prelude + self.core..self.layers(),
        }
    }
}

/// Every visit of a [`LoopShape`] in forward order, with its attention and state slot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VisitPlan {
    shape: LoopShape,
    visits: Vec<Visit>,
    slots: usize,
}

impl VisitPlan {
    /// The plan of `shape`, each visit running the attention `attention(layer)` declares for its physical layer, one
    /// state slot per visit. Refuses a shape with no core or no pass, and a zero window.
    pub fn new(shape: LoopShape, attention: impl Fn(usize) -> Attention) -> Result<VisitPlan, String> {
        if shape.core == 0 || shape.passes == 0 {
            return Err(format!("visit plan: a looped stack needs a core run at least once, got {} core layers x {} passes", shape.core, shape.passes));
        }
        let mut visits = Vec::with_capacity(shape.visits());
        let mut push = |layer: usize, segment: Segment, pass: usize| -> Result<(), String> {
            let attention = attention(layer);
            if attention == Attention::Window(0) {
                return Err(format!("visit plan: layer {layer} declares a window of 0 positions"));
            }
            let index = visits.len();
            visits.push(Visit { index, layer, segment, pass, attention, slot: index });
            Ok(())
        };
        for layer in shape.layers_of(Segment::Prelude) {
            push(layer, Segment::Prelude, 0)?;
        }
        for pass in 0..shape.passes {
            for layer in shape.layers_of(Segment::Core) {
                push(layer, Segment::Core, pass)?;
            }
        }
        for layer in shape.layers_of(Segment::Coda) {
            push(layer, Segment::Coda, 0)?;
        }
        let slots = visits.len();
        Ok(VisitPlan { shape, visits, slots })
    }

    /// [`VisitPlan::new`] with every layer running `attention`.
    pub fn uniform(shape: LoopShape, attention: Attention) -> Result<VisitPlan, String> {
        VisitPlan::new(shape, |_| attention)
    }

    /// Core visits cycle over `core_slots` slots instead of owning one each: the `k`-th core visit (counted over all
    /// passes) uses core slot `k % core_slots`, so a slot is reused by the visit `core_slots` core visits later.
    /// Prelude visits keep slots `0..prelude`, then the core slots, then one per coda visit. More slots than core
    /// visits is the unshared plan.
    pub fn share_core_slots(mut self, core_slots: usize) -> Result<VisitPlan, String> {
        if core_slots == 0 {
            return Err("visit plan: core visits cannot share zero slots".into());
        }
        let core_visits = self.shape.core * self.shape.passes;
        let shared = core_slots.min(core_visits);
        let mut k = 0;
        for v in &mut self.visits {
            v.slot = match v.segment {
                Segment::Prelude => v.index,
                Segment::Core => {
                    let slot = self.shape.prelude + k % shared;
                    k += 1;
                    slot
                }
                Segment::Coda => self.shape.prelude + shared + (v.layer - self.shape.layers_of(Segment::Coda).start),
            };
        }
        self.slots = self.shape.prelude + shared + self.shape.coda;
        Ok(self)
    }

    pub fn shape(&self) -> LoopShape {
        self.shape
    }

    /// Every visit in forward order.
    pub fn visits(&self) -> &[Visit] {
        &self.visits
    }

    /// Logical visits.
    pub fn len(&self) -> usize {
        self.visits.len()
    }

    /// A plan always has visits; present for the `len` convention.
    pub fn is_empty(&self) -> bool {
        self.visits.is_empty()
    }

    /// Distinct state slots.
    pub fn slots(&self) -> usize {
        self.slots
    }

    /// Residual-stream boundaries of one forward: visit `i` reads boundary `i` and writes `i + 1`.
    pub fn boundaries(&self) -> usize {
        self.visits.len() + 1
    }

    /// Whether a physical layer runs more than once (its gradient is then carried between its visits).
    pub fn revisits(&self) -> bool {
        self.shape.passes > 1
    }

    /// The prelude's visits.
    pub fn prelude(&self) -> &[Visit] {
        &self.visits[..self.shape.prelude]
    }

    /// The core visits of pass `pass`.
    pub fn pass(&self, pass: usize) -> &[Visit] {
        assert!(pass < self.shape.passes, "visit plan: pass {pass} of {}", self.shape.passes);
        let start = self.shape.prelude + pass * self.shape.core;
        &self.visits[start..start + self.shape.core]
    }

    /// The coda's visits.
    pub fn coda(&self) -> &[Visit] {
        &self.visits[self.visits.len() - self.shape.coda..]
    }

    /// Every visit of physical layer `layer`, in forward order.
    pub fn visits_of(&self, layer: usize) -> impl Iterator<Item = &Visit> {
        self.visits.iter().filter(move |v| v.layer == layer)
    }

    /// Visits running `attention`: what a serving pool charges per token of history is a function of these counts.
    pub fn count(&self, attention: Attention) -> usize {
        self.visits.iter().filter(|v| v.attention == attention).count()
    }

    /// The visit owning each slot first, in slot order (a slot's attention kind is its visits').
    pub fn slot_owners(&self) -> Vec<&Visit> {
        let mut owners: Vec<Option<&Visit>> = vec![None; self.slots];
        for v in &self.visits {
            owners[v.slot].get_or_insert(v);
        }
        owners.into_iter().map(|o| o.expect("visit plan: every slot has a visit")).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(plan: &VisitPlan) -> Vec<(usize, Segment, usize, usize)> {
        plan.visits().iter().map(|v| (v.layer, v.segment, v.pass, v.slot)).collect()
    }

    /// Spec: a prelude/core/coda shape visits the prelude once, the core once per pass and the coda once, in that
    /// order, each visit owning its own slot; a whole-stack loop is the shape with no prelude and no coda.
    #[test]
    fn visits_run_prelude_then_every_core_pass_then_coda() {
        let plan = VisitPlan::uniform(LoopShape { prelude: 1, core: 2, coda: 1, passes: 3 }, Attention::Full).unwrap();
        use Segment::*;
        assert_eq!(
            kinds(&plan),
            [(0, Prelude, 0, 0), (1, Core, 0, 1), (2, Core, 0, 2), (1, Core, 1, 3), (2, Core, 1, 4), (1, Core, 2, 5), (2, Core, 2, 6), (3, Coda, 0, 7)]
        );
        assert_eq!((plan.len(), plan.slots(), plan.boundaries()), (8, 8, 9));
        assert!(plan.visits().iter().enumerate().all(|(i, v)| v.index == i));
        assert_eq!(plan.pass(1).iter().map(|v| v.index).collect::<Vec<_>>(), [3, 4]);
        assert_eq!(plan.visits_of(2).map(|v| v.index).collect::<Vec<_>>(), [2, 4, 6]);
        assert!(plan.revisits());

        let whole = VisitPlan::uniform(LoopShape::whole_stack(3, 2), Attention::Full).unwrap();
        assert!(whole.prelude().is_empty() && whole.coda().is_empty());
        assert_eq!(whole.visits().iter().map(|v| (v.layer, v.pass)).collect::<Vec<_>>(), [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]);
        assert!(!VisitPlan::uniform(LoopShape::whole_stack(3, 1), Attention::Full).unwrap().revisits());
    }

    /// Spec: each visit runs the attention its physical layer declares, and the counts per kind are what a serving
    /// pool charges (Inkling 2 Mini: 40 windowed and 8 full visits of 24 layers, every sixth one full).
    #[test]
    fn attention_follows_the_physical_layer() {
        let shape = LoopShape { prelude: 6, core: 12, coda: 6, passes: 3 };
        let plan = VisitPlan::new(shape, |l| if (l + 1) % 6 == 0 { Attention::Full } else { Attention::Window(1024) }).unwrap();
        assert_eq!(plan.len(), 48);
        assert_eq!((plan.count(Attention::Full), plan.count(Attention::Window(1024))), (8, 40));
        assert!(plan.visits().iter().all(|v| (v.attention == Attention::Full) == ((v.layer + 1) % 6 == 0)));
        assert!(VisitPlan::new(shape, |_| Attention::Window(0)).unwrap_err().contains("window of 0"));
    }

    /// Spec: shared core slots cycle over the core visits; prelude and coda slots stay their own; more slots than
    /// core visits is the unshared plan.
    #[test]
    fn core_visits_can_share_fewer_slots() {
        let shape = LoopShape { prelude: 1, core: 2, coda: 1, passes: 3 };
        let shared = VisitPlan::uniform(shape, Attention::Full).unwrap().share_core_slots(4).unwrap();
        assert_eq!(shared.visits().iter().map(|v| v.slot).collect::<Vec<_>>(), [0, 1, 2, 3, 4, 1, 2, 5]);
        assert_eq!(shared.slots(), 6);
        assert_eq!(shared.slot_owners().iter().map(|v| v.index).collect::<Vec<_>>(), [0, 1, 2, 3, 4, 7]);
        let unshared = VisitPlan::uniform(shape, Attention::Full).unwrap();
        assert_eq!(unshared.clone().share_core_slots(100).unwrap(), unshared);
        assert!(VisitPlan::uniform(shape, Attention::Full).unwrap().share_core_slots(0).is_err());
        assert!(VisitPlan::uniform(LoopShape { prelude: 1, core: 0, coda: 1, passes: 2 }, Attention::Full).is_err());
    }
}
