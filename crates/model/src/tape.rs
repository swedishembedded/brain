// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded training of models larger than the accelerator for its clients. If
// your team needs expertise in streaming a model's layers through a fixed device window, you can procure our services
// by sending an email to info@swedishembedded.com.

//! A recorded pass in segments, each knowing the layer it runs ([`LayerTape`]).
//!
//! A trainer records its forward or backward once; a layer-streamed build ([`crate::streamed`]) must act between
//! the layers of that pass (load a layer's weights before it runs, finish its update after its backward). The tape
//! keeps the one recorded list and the boundaries, and [`LayerTape::submit_hooked`] reports every boundary as it
//! submits.

use gpu_core::{Gpu, Step};

/// A recorded pass and the segments it is submitted in.
#[derive(Default)]
pub struct LayerTape {
    steps: Vec<Step>,
    /// Each closed segment: its exclusive end in `steps` and the layer it runs (`None` for anything else: the
    /// embedding, the head, the loss).
    segs: Vec<(usize, Option<usize>)>,
}

/// A boundary of a [`LayerTape`] segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TapeEdge {
    /// Segment `index` (running `layer`, if any) is about to be submitted.
    Before { index: usize, layer: Option<usize> },
    /// Segment `index` has been submitted.
    After { index: usize, layer: Option<usize> },
}

impl LayerTape {
    /// The steps recorded so far, for appending to.
    pub fn steps_mut(&mut self) -> &mut Vec<Step> {
        &mut self.steps
    }

    /// Every recorded step in order.
    pub fn steps(&self) -> &[Step] {
        &self.steps
    }

    /// Close the steps recorded since the last close into a segment running `layer` (`None`: no layer). An empty
    /// segment is kept: its edges still report the layer.
    pub fn close(&mut self, layer: Option<usize>) {
        self.segs.push((self.steps.len(), layer));
    }

    /// Move `steps` (recorded since the last segment) into a segment running `layer`, leaving `steps` empty for the
    /// next: how a builder that records into a list of its own marks the layers.
    pub fn segment(&mut self, steps: &mut Vec<Step>, layer: Option<usize>) {
        self.steps.append(steps);
        self.close(layer);
    }

    /// The segments in order: `(index, layer, steps)`, the steps after the last close in a final unlayered segment.
    pub fn segments(&self) -> impl Iterator<Item = (usize, Option<usize>, &[Step])> {
        let tail = (self.segs.last().map_or(0, |s| s.0) < self.steps.len()).then_some((self.steps.len(), None));
        let mut start = 0;
        self.segs.iter().copied().chain(tail).enumerate().map(move |(i, (end, layer))| {
            let seg = &self.steps[start..end];
            start = end;
            (i, layer, seg)
        })
    }

    /// Submit every segment in order.
    pub fn submit(&self, gpu: &Gpu) {
        self.submit_hooked(gpu, &mut |_| {});
    }

    /// Submit every segment in order, reporting [`TapeEdge::Before`] ahead of each and [`TapeEdge::After`] once it
    /// is submitted (an empty segment included).
    pub fn submit_hooked(&self, gpu: &Gpu, edge: &mut dyn FnMut(TapeEdge)) {
        for (index, layer, steps) in self.segments() {
            edge(TapeEdge::Before { index, layer });
            if !steps.is_empty() {
                gpu.submit(&[], steps);
            }
            edge(TapeEdge::After { index, layer });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Spec: segments partition the recorded steps in order, each with its layer, the steps after the last close
    /// forming one unlayered segment, and an empty segment kept.
    #[test]
    fn segments_partition_the_steps_with_their_layers() {
        let gpu = gpu_core::testgpu::dev(&[("add2", kernels::ADD2)]);
        let (a, out) = (gpu.storage(4), gpu.storage(4));
        let step = || gpu.step(0, &[&a, &a, &out], &[4], 4);
        let mut t = LayerTape::default();
        t.steps_mut().push(step());
        t.close(None);
        t.steps_mut().extend([step(), step()]);
        t.close(Some(1));
        t.close(Some(0));
        t.steps_mut().push(step());
        let got: Vec<(usize, Option<usize>, usize)> = t.segments().map(|(i, l, s)| (i, l, s.len())).collect();
        assert_eq!(got, vec![(0, None, 1), (1, Some(1), 2), (2, Some(0), 0), (3, None, 1)]);
        let mut edges = Vec::new();
        t.submit_hooked(&gpu, &mut |e| edges.push(e));
        assert_eq!(edges.len(), 8);
        assert_eq!(edges[2], TapeEdge::Before { index: 1, layer: Some(1) });
        assert_eq!(edges[5], TapeEdge::After { index: 2, layer: Some(0) });
    }
}
